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
from .staffing import service_level_curve, staffing_plan

MIN_LAG = 35
HORIZON_DAYS = 28


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


def future_forecast(daily: pd.DataFrame, days: int = HORIZON_DAYS) -> pd.DataFrame:
    last = daily["ds"].max()
    future = pd.DataFrame(
        {"ds": pd.date_range(last + pd.Timedelta(days=1), periods=days), "y": np.nan}
    )
    feats = make_features(pd.concat([daily, future], ignore_index=True))
    train, test = feats[feats["ds"] <= last], feats[feats["ds"] > last].copy()
    test["calls_p50"], test["calls_p80"] = _fit_predict_lgbm(train, test)
    return test[["ds", "dow", "holiday_name", "calls_p50", "calls_p80"]]


def run_forecast(
    paths: Paths, n_folds: int = 12, target_sl: float = 0.80, shrinkage: float = 0.30
) -> dict:
    con = connect(paths)
    daily = load_daily(con)
    bt = backtest(daily, n_folds=n_folds)
    sc = score(bt)
    fut = future_forecast(daily)
    prof = intraday_profile(con)
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
        target_sl=target_sl,
    )
    bt.to_parquet(paths.outputs / "forecast_backtest.parquet", index=False)
    fut.to_parquet(paths.outputs / "forecast_daily.parquet", index=False)
    plan.to_parquet(paths.outputs / "staffing_plan.parquet", index=False)
    plan.to_csv(paths.exports / "staffing_plan.csv", index=False)

    peak = plan.loc[plan["calls_p50"].idxmax()]
    curve = service_level_curve(peak["calls_p50"], peak["aht_sec"])
    curve.to_parquet(paths.outputs / "sl_curve.parquet", index=False)

    best = sc.sort_values("wape").iloc[0]
    summary = {
        "backtest_folds": n_folds,
        "scores": sc.round(4).to_dict(orient="records"),
        "best_model": best["model"],
        "p80_coverage": round(sc.attrs["p80_coverage"], 3),
        "plan_days": int(fut["ds"].nunique()),
        "peak_interval": str(peak["interval_start"]),
        "peak_calls_p50": round(float(peak["calls_p50"]), 1),
        "peak_agents_p50": int(peak["agents_p50"]),
        "peak_agents_p80": int(peak["agents_p80"]),
        "agent_hours_p50": float(plan["agents_p50"].sum() / 2),
        "agent_hours_p80": float(plan["agents_p80"].sum() / 2),
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
        "Holiday-related days are public holidays, the eve of a religious holiday and the first "
        "working day after one. Unplanned events such as the brake recall are not in the "
        "features, so every model misses them; that is what intraday monitoring is for.",
        "",
        "## Staffing plan",
        "",
        f"Next {s['plan_days']} days, 30-minute intervals, target 80% of calls answered within "
        "20 seconds, maximum occupancy 90%, Erlang C.",
        "",
        f"- Peak interval: {s['peak_interval']} with {s['peak_calls_p50']:.0f} expected calls: "
        f"**{s['peak_agents_p50']} agents** on the phone at P50, **{s['peak_agents_p80']}** at P80.",
        f"- Agent hours on the phone: {s['agent_hours_p50']:,.0f} (P50) vs {s['agent_hours_p80']:,.0f}"
        f" (P80): the price of protecting the service level against an under-forecast.",
        "",
        "### Service level at the peak interval",
        "",
        "| Agents | Service level | ASA (s) | Occupancy |",
        "|---:|---:|---:|---:|",
    ]
    for r in curve.head(12).itertuples():
        lines.append(
            f"| {r.agents} | {r.service_level:.1%} | {r.asa_sec:.0f} | {r.occupancy:.0%} |"
        )
    lines += [
        "",
        "Each extra agent buys less service level than the one before. Where to stop "
        "is a cost versus customer experience decision, and the table makes it explicit.",
    ]
    (paths.reports / "forecast_staffing.md").write_text("\n".join(lines) + "\n")
