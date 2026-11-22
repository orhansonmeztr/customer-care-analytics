"""Daily call volume forecasting with an honest rolling-origin backtest.

Design choices
--------------
* Forecasts are made at the start of each month for the whole month, because staff
  schedules are fixed weeks ahead. Every lag feature is therefore at least 35 days old.
* LightGBM cannot extrapolate a trend, so it learns the *ratio* of a day to the recent level
  (log of the day over the mean of an earlier 28-day window), not the raw volume.
* Holidays come from the Turkish calendar. Religious holidays move about 11 days a year,
  which is exactly where the "same day last year" baseline fails.
* Days the data quality step marked as incomplete loads are excluded from training and
  scoring. Treating a missing load as "zero calls" would teach the model a fake drop.
* The 80th percentile forecast (quantile LightGBM) is what the staffing plan uses for risk.
  Quantile models trained in-sample under-cover out of time, so the P80 is calibrated with
  split conformal prediction on the backtest's own out-of-time errors: each fold is
  corrected only with errors from earlier folds, the future forecast with all of them.
"""

from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .calendar_tr import holiday_calendar
from .paths import Paths
from .quality import connect
from .staffing import erlang_a, estimate_patience, service_level_curve, staffing_plan

MIN_LAG = 35
HORIZON_DAYS = 28
TARGET_COVERAGE = 0.80
CONFORMAL_MIN_FOLDS = 3  # folds of past errors needed before a fold is calibrated


def load_daily(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT d.date AS ds, count(c.call_id) AS y, d.is_incomplete_load
        FROM dim_date d LEFT JOIN fact_calls c ON c.call_date = d.date
        GROUP BY ALL ORDER BY ds
    """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    df.loc[df["is_incomplete_load"], "y"] = np.nan
    return df[["ds", "y"]]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and lag features. ``df`` must cover the dates to predict (y may be NaN)."""
    d = df.copy().sort_values("ds").reset_index(drop=True)
    cal = holiday_calendar(str(d["ds"].min().date()), str(d["ds"].max().date()))
    d = d.merge(cal, left_on="ds", right_on="date", how="left").drop(columns="date")
    d["dow"] = d["ds"].dt.day_of_week
    d["month"] = d["ds"].dt.month
    doy = d["ds"].dt.day_of_year
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["is_month_end"] = d["ds"].dt.is_month_end.astype(int)
    d["is_religious"] = (d["holiday_kind"] == "religious").astype(int)
    d["is_national"] = (d["holiday_kind"] == "national").astype(int)
    for c in ("is_holiday", "is_arife", "is_post_holiday"):
        d[c] = d[c].astype(int)
    # trailing level, interpolated over excluded days, at least MIN_LAG days old
    y_filled = d["y"].interpolate(limit_direction="both")
    level = y_filled.shift(MIN_LAG).rolling(28, min_periods=20).mean()
    d["level"] = level
    for lag in (MIN_LAG, MIN_LAG + 7, MIN_LAG + 14, 364):
        d[f"rel_lag_{lag}"] = np.log(y_filled.shift(lag) / level)
    d["target"] = np.log(d["y"] / level)
    return d


FEATURES = [
    "dow",
    "month",
    "doy_sin",
    "doy_cos",
    "is_month_end",
    "is_holiday",
    "is_religious",
    "is_national",
    "is_arife",
    "is_post_holiday",
    "rel_lag_35",
    "rel_lag_42",
    "rel_lag_49",
    "rel_lag_364",
]


def _lgbm(alpha: float | None = None) -> lgb.LGBMRegressor:
    kw = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=7,
        n_jobs=2,
    )
    if alpha is not None:
        kw.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**kw)


def _fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.dropna(subset=["target", "level"])
    p50 = _lgbm().fit(tr[FEATURES], tr["target"]).predict(test[FEATURES])
    p80 = _lgbm(0.8).fit(tr[FEATURES], tr["target"]).predict(test[FEATURES])
    lv = test["level"].to_numpy()
    # two separately trained quantile models can cross; the P80 can never be below the P50
    return lv * np.exp(p50), lv * np.exp(np.maximum(p80, p50))


def _ets(history: pd.Series, steps: int) -> np.ndarray:
    y = np.log(history.interpolate(limit_direction="both").clip(lower=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
        ).fit()
    return np.exp(fit.forecast(steps).to_numpy())


def _naive(hist: pd.DataFrame, dates: pd.Series, period: int) -> np.ndarray:
    s = hist.set_index("ds")["y"].interpolate(limit_direction="both")
    out = []
    for d in dates:
        back = d - pd.Timedelta(days=period)
        while back > s.index.max():  # seasonal naive: last available same weekday
            back -= pd.Timedelta(days=period)
        out.append(s.get(back, np.nan))
    return np.array(out)


def wape(y, f) -> float:
    m = ~np.isnan(y) & ~np.isnan(f)
    return float(np.abs(y[m] - f[m]).sum() / y[m].sum())


def bias(y, f) -> float:
    m = ~np.isnan(y) & ~np.isnan(f)
    return float((f[m] - y[m]).sum() / y[m].sum())


def backtest(daily: pd.DataFrame, n_folds: int = 12) -> pd.DataFrame:
    feats = make_features(daily)
    last = daily["ds"].max()
    cutoffs = pd.date_range(
        end=last - pd.offsets.MonthBegin(1) + pd.Timedelta(days=1), periods=n_folds, freq="MS"
    )
    rows = []
    for cut in cutoffs:
        end = min(cut + pd.offsets.MonthEnd(0), last)
        test = feats[(feats["ds"] >= cut) & (feats["ds"] <= end)].copy()
        train = feats[feats["ds"] < cut]
        hist = daily[daily["ds"] < cut]
        test["naive_7"] = _naive(hist, test["ds"], 7)
        test["naive_364"] = _naive(hist, test["ds"], 364)
        test["ets"] = _ets(hist.set_index("ds")["y"], len(test))
        test["lgbm"], test["lgbm_p80"] = _fit_predict_lgbm(train, test)
        test["cutoff"] = cut
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def conformal_offset(y: np.ndarray, p80: np.ndarray, level: float = TARGET_COVERAGE) -> float:
    """Log-scale shift that makes ``p80 * exp(shift)`` cover ``level`` of the given errors.

    Split conformal: the score is log(actual / P80); its finite-sample corrected quantile is
    the shift. A positive shift widens the band, a negative one narrows it.
    """
    m = ~np.isnan(y) & ~np.isnan(p80) & (y > 0) & (p80 > 0)
    s = np.sort(np.log(y[m] / p80[m]))
    n = len(s)
    if n == 0:
        return 0.0
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(s[k - 1])


def calibrate_backtest(bt: pd.DataFrame, min_folds: int = CONFORMAL_MIN_FOLDS) -> pd.DataFrame:
    """Add ``lgbm_p80_cal``: each fold corrected with the errors of earlier folds only."""
    bt = bt.copy()
    bt["lgbm_p80_cal"] = np.nan
    cutoffs = sorted(bt["cutoff"].unique())
    for i, c in enumerate(cutoffs):
        if i < min_folds:
            continue
        past = bt[bt["cutoff"] < c]
        shift = conformal_offset(past["y"].to_numpy(float), past["lgbm_p80"].to_numpy(float))
        cur = bt["cutoff"] == c
        bt.loc[cur, "lgbm_p80_cal"] = bt.loc[cur, "lgbm_p80"] * np.exp(shift)
    return bt


def score(bt: pd.DataFrame) -> pd.DataFrame:
    y = bt["y"].to_numpy(dtype=float)
    special = (bt["is_holiday"] | bt["is_arife"] | bt["is_post_holiday"]).astype(bool).to_numpy()
    out = []
    for m in ("naive_7", "naive_364", "ets", "lgbm"):
        f = bt[m].to_numpy(dtype=float)
        out.append(
            {
                "model": m,
                "wape": wape(y, f),
                "bias": bias(y, f),
                "wape_holiday_days": wape(y[special], f[special]),
                "wape_normal_days": wape(y[~special], f[~special]),
            }
        )
    res = pd.DataFrame(out)
    m = ~np.isnan(y)
    res.attrs["p80_coverage"] = float((y[m] <= bt["lgbm_p80"].to_numpy()[m]).mean())
    if "lgbm_p80_cal" in bt and bt["lgbm_p80_cal"].notna().any():
        # compare raw and calibrated on the same folds: the ones that could be calibrated
        cal = bt["lgbm_p80_cal"].to_numpy(float)
        e = m & ~np.isnan(cal)
        raw = bt["lgbm_p80"].to_numpy(float)
        p50 = bt["lgbm"].to_numpy(float)
        res.attrs["calibrated_folds"] = int(bt.loc[e, "cutoff"].nunique())
        res.attrs["p80_coverage_raw_same_folds"] = float((y[e] <= raw[e]).mean())
        res.attrs["p80_coverage_calibrated"] = float((y[e] <= cal[e]).mean())
        res.attrs["p80_margin_raw"] = float(np.mean(raw[e] / p50[e] - 1))
        res.attrs["p80_margin_calibrated"] = float(np.mean(cal[e] / p50[e] - 1))
    return res


def intraday_profile(con, weeks: int = 8) -> pd.DataFrame:
    """Share of the day's calls and AHT per weekday and 30-minute slot, last ``weeks`` weeks."""
    return con.execute(f"""
        WITH i AS (
            SELECT v.*, dayofweek(call_date) AS dow_sun0, strftime(interval_start, '%H:%M') AS slot
            FROM v_kpi_interval v JOIN dim_date d ON d.date = v.call_date
            WHERE NOT d.is_incomplete_load AND NOT d.is_holiday
              AND call_date > (SELECT max(call_date) FROM fact_calls) - {7 * weeks}
        ),
        tot AS (SELECT call_date, sum(offered) AS day_total FROM i GROUP BY 1)
        SELECT (dow_sun0 + 6) % 7 AS dow, slot,
            avg(offered / day_total) AS share,
            sum(aht_sec * answered) / nullif(sum(answered), 0) AS aht_sec
        FROM i JOIN tot USING (call_date) GROUP BY ALL ORDER BY dow, slot
    """).df()


def future_forecast(
    daily: pd.DataFrame, days: int = HORIZON_DAYS, p80_shift: float = 0.0
) -> pd.DataFrame:
    last = daily["ds"].max()
    future = pd.DataFrame(
        {"ds": pd.date_range(last + pd.Timedelta(days=1), periods=days), "y": np.nan}
    )
    feats = make_features(pd.concat([daily, future], ignore_index=True))
    train, test = feats[feats["ds"] <= last], feats[feats["ds"] > last].copy()
    test["calls_p50"], test["calls_p80_raw"] = _fit_predict_lgbm(train, test)
    test["calls_p80"] = np.maximum(test["calls_p80_raw"] * np.exp(p80_shift), test["calls_p50"])
    return test[["ds", "dow", "holiday_name", "calls_p50", "calls_p80_raw", "calls_p80"]]


def run_forecast(
    paths: Paths, n_folds: int = 12, target_sl: float = 0.80, shrinkage: float = 0.30
) -> dict:
    con = connect(paths)
    daily = load_daily(con)
    bt = calibrate_backtest(backtest(daily, n_folds=n_folds))
    sc = score(bt)
    shift = conformal_offset(bt["y"].to_numpy(float), bt["lgbm_p80"].to_numpy(float))
    fut = future_forecast(daily, p80_shift=shift)
    prof = intraday_profile(con)
    patience = estimate_patience(con)
    con.close()

    iv = fut.merge(prof, on="dow", how="left")
    iv["interval_start"] = pd.to_datetime(iv["ds"].dt.strftime("%Y-%m-%d") + " " + iv["slot"])
    iv["calls_p50"] *= iv["share"]
    iv["calls_p80"] *= iv["share"]
    plan = staffing_plan(
        iv[["interval_start", "ds", "holiday_name", "calls_p50", "calls_p80", "aht_sec"]].dropna(
            subset=["aht_sec"]
        ),
        shrinkage=shrinkage,
        patience=patience["patience_sec"],
        target_sl=target_sl,
    )
    bt.to_parquet(paths.outputs / "forecast_backtest.parquet", index=False)
    fut.to_parquet(paths.outputs / "forecast_daily.parquet", index=False)
    plan.to_parquet(paths.outputs / "staffing_plan.parquet", index=False)
    plan.to_csv(paths.exports / "staffing_plan.csv", index=False)

    peak = plan.loc[plan["calls_p50"].idxmax()]
    curve = service_level_curve(peak["calls_p50"], peak["aht_sec"])
    curve["service_level_erlang_a"] = [
        erlang_a(peak["calls_p50"], peak["aht_sec"], patience["patience_sec"], int(n))[
            "service_level"
        ]
        for n in curve["agents"]
    ]
    curve["abandon_erlang_a"] = [
        erlang_a(peak["calls_p50"], peak["aht_sec"], patience["patience_sec"], int(n))["p_abandon"]
        for n in curve["agents"]
    ]
    curve.to_parquet(paths.outputs / "sl_curve.parquet", index=False)

    best = sc.sort_values("wape").iloc[0]
    summary = {
        "backtest_folds": n_folds,
        "scores": sc.round(4).to_dict(orient="records"),
        "best_model": best["model"],
        "p80_coverage": round(sc.attrs["p80_coverage"], 3),
        "conformal": {
            "calibrated_folds": sc.attrs.get("calibrated_folds"),
            "coverage_raw_same_folds": round(sc.attrs.get("p80_coverage_raw_same_folds", 0), 3),
            "coverage_calibrated": round(sc.attrs.get("p80_coverage_calibrated", 0), 3),
            "margin_over_p50_raw": round(sc.attrs.get("p80_margin_raw", 0), 3),
            "margin_over_p50_calibrated": round(sc.attrs.get("p80_margin_calibrated", 0), 3),
            "future_log_shift": round(shift, 4),
        },
        "patience": patience,
        "plan_days": int(fut["ds"].nunique()),
        "peak_interval": str(peak["interval_start"]),
        "peak_calls_p50": round(float(peak["calls_p50"]), 1),
        "peak_agents_p50": int(peak["agents_p50"]),
        "peak_agents_p80": int(peak["agents_p80"]),
        "agent_hours_p50": float(plan["agents_p50"].sum() / 2),
        "agent_hours_p80": float(plan["agents_p80"].sum() / 2),
        "peak_agents_p80_erlang_a": int(peak["agents_p80_erlang_a"]),
        "agent_hours_p80_erlang_a": float(plan["agents_p80_erlang_a"].sum() / 2),
        "erlang_c_plan_true_sl_mean": round(float(plan["sl_p80_erlang_c_plan_true"].mean()), 4),
        "erlang_a_plan_sl_mean": round(float(plan["sl_p80_erlang_a"].mean()), 4),
        "erlang_a_plan_abandon_mean": round(float(plan["abandon_p80_erlang_a"].mean()), 4),
    }
    _write_report(paths, summary, sc, curve)
    (paths.outputs / "forecast.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _write_report(paths: Paths, s: dict, sc: pd.DataFrame, curve: pd.DataFrame) -> None:
    names = {
        "naive_7": "Seasonal naive (same weekday last week)",
        "naive_364": "Same weekday last year (364 days)",
        "ets": "Holt-Winters ETS (log, weekly seasonality)",
        "lgbm": "LightGBM (ratio target, calendar + holiday features)",
    }
    lines = [
        "# Call volume forecast and staffing",
        "",
        f"Rolling-origin backtest: {s['backtest_folds']} monthly folds. Each forecast is made on "
        "the first day of the month for the whole month, so every input is at least 35 days old.",
        "",
        "| Model | WAPE | Bias | WAPE on holiday-related days | WAPE on other days |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sc.itertuples():
        lines.append(
            f"| {names[r.model]} | {r.wape:.1%} | {r.bias:+.1%} | "
            f"{r.wape_holiday_days:.1%} | {r.wape_normal_days:.1%} |"
        )
    lines += [
        "",
        f"The LightGBM 80th percentile forecast covered {s['p80_coverage']:.0%} of actual days "
        "(target: 80%).",
        "",
        "### Conformal calibration of the P80",
        "",
        f"Each fold's P80 is corrected with the out-of-time errors of the earlier folds only "
        f"(split conformal on log(actual / P80)); folds with fewer than {CONFORMAL_MIN_FOLDS} "
        f"earlier folds are left out. On the {s['conformal']['calibrated_folds'] or 0} folds that "
        "could be calibrated:",
        "",
        "| P80 | Coverage (target 80%) | Average margin over P50 |",
        "|---|---:|---:|",
        f"| Quantile LightGBM, raw | {s['conformal']['coverage_raw_same_folds']:.1%} | "
        f"{s['conformal']['margin_over_p50_raw']:+.1%} |",
        f"| Conformal-calibrated | {s['conformal']['coverage_calibrated']:.1%} | "
        f"{s['conformal']['margin_over_p50_calibrated']:+.1%} |",
        "",
        f"The staffing plan uses the P80 calibrated with all backtest errors (log shift "
        f"{s['conformal']['future_log_shift']:+.3f}). Conformal coverage is guaranteed only when "
        "errors are exchangeable over time; call volumes drift, so the calibrated band can land "
        "a few points off the target on later folds.",
        "",
        "Holiday-related days are public holidays, the eve of a religious holiday and the first "
        "working day after one. Unplanned events such as the brake recall are not in the "
        "features, so every model misses them; that is what intraday monitoring is for.",
        "",
        "## Staffing plan",
        "",
        f"Next {s['plan_days']} days, 30-minute intervals, target 80% of calls answered within "
        "20 seconds, maximum occupancy 90%.",
        "",
        "### Erlang C vs Erlang A",
        "",
        f"Caller patience estimated from the last {s['patience']['weeks']} weeks: "
        f"**{s['patience']['patience_sec']:.0f} s** on average ({s['patience']['abandons']:,} "
        f"abandonments after the first {s['patience']['grace_sec']:.0f} s, in "
        f"{s['patience']['calls']:,} calls). Erlang A also caps abandonment at 5%.",
        "",
        "| Model, P80 plan | Agent hours on the phone | Peak agents | Mean service level | "
        "Mean abandonment |",
        "|---|---:|---:|---:|---:|",
        f"| Erlang C | {s['agent_hours_p80']:,.0f} | {s['peak_agents_p80']} | "
        f"{s['erlang_c_plan_true_sl_mean']:.1%} (with abandonment) | - |",
        f"| Erlang A | {s['agent_hours_p80_erlang_a']:,.0f} | {s['peak_agents_p80_erlang_a']} | "
        f"{s['erlang_a_plan_sl_mean']:.1%} | {s['erlang_a_plan_abandon_mean']:.1%} |",
        "",
        "Erlang C assumes nobody hangs up, so it plans for a longer queue than will form: the "
        "Erlang C plan delivers a higher service level than the target once real abandonment "
        "is accounted for. Erlang A meets the same target with fewer agent hours.",
        "",
        f"- Peak interval: {s['peak_interval']} with {s['peak_calls_p50']:.0f} expected calls: "
        f"**{s['peak_agents_p50']} agents** on the phone at P50, **{s['peak_agents_p80']}** at P80.",
        f"- Agent hours on the phone: {s['agent_hours_p50']:,.0f} (P50) vs {s['agent_hours_p80']:,.0f}"
        f" (P80): the price of protecting the service level against an under-forecast.",
        "",
        "### Service level at the peak interval",
        "",
        "| Agents | Service level, Erlang C | Service level, Erlang A | Abandonment, Erlang A | "
        "ASA, Erlang C (s) | Offered load per agent |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in curve.head(12).itertuples():
        lines.append(
            f"| {r.agents} | {r.service_level:.1%} | {r.service_level_erlang_a:.1%} | "
            f"{r.abandon_erlang_a:.1%} | {r.asa_sec:.0f} | {r.occupancy:.0%} |"
        )
    lines += [
        "",
        "Each extra agent buys less service level than the one before. Where to stop "
        "is a cost versus customer experience decision, and the table makes it explicit.",
    ]
    (paths.reports / "forecast_staffing.md").write_text("\n".join(lines) + "\n")
