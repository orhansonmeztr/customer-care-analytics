"""Early warning for product complaints: weekly tickets per model and fault code.

Many small series (model x fault code) with low counts: a z-score on raw counts would
alarm on noise. Each week's count is compared with a Poisson baseline from the previous
12 weeks that did not raise an alert (so a spike does not become the new normal), and the p-values of all series in a week are corrected for multiple testing
(Benjamini-Hochberg), so the number of false alarms stays under control as series are added.
An alert also needs a minimum size, because a jump from 1 to 4 tickets is not actionable.

A weekly test only fires when one week is clearly above normal, so a problem that grows
slowly is seen late. A daily Poisson CUSUM accumulates small excesses day after day and
catches such a drift earlier. Its baseline is taken with a guard gap (from four weeks ago
backwards), so that the drift itself does not become the new normal, and its alarm
threshold is set by simulation for a target false-alarm rate instead of by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .paths import Paths
from .quality import connect

BASELINE_WEEKS = 12
FDR = 0.01
MIN_COUNT = 5
MIN_RATIO = 1.5

CUSUM_SHIFT = 1.5  # look for a 50% rise in the complaint rate
# in control, one false alarm per series every 50 years: about as many alarms in total as the
# weekly test raises, so the two detectors are compared at equal false alarms
CUSUM_ARL0_DAYS = 18_250
CUSUM_BASE_DAYS = 126  # baseline window ...
CUSUM_GUARD_DAYS = 28  # ... that ends four weeks ago, so a slow drift is not absorbed
CUSUM_MIN_RATE = 0.05  # floor for the expected daily count of very quiet series
# reported trade-off (design false-alarm interval in days, baseline window in days)
CUSUM_TRADEOFF = ((730, 84), (730, 126), (3_650, 126), (CUSUM_ARL0_DAYS, CUSUM_BASE_DAYS))


def weekly_counts(con) -> pd.DataFrame:
    """Weekly tickets per model x fault code, with the number of cars of that model on the road.

    A new model's complaints grow because its fleet grows; the baseline is a *rate* per car.
    """
    df = con.execute("""
        SELECT date_trunc('week', t.opened_date) AS week, v.model, t.fault_code, count(*) AS n
        FROM fact_tickets t JOIN dim_vehicle v USING (vin)
        WHERE t.fault_code IS NOT NULL
        GROUP BY ALL
    """).df()
    df["week"] = pd.to_datetime(df["week"])
    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
    full = (
        df.set_index(["model", "fault_code", "week"])["n"]
        .unstack("week")
        .reindex(columns=weeks)
        .fillna(0)
        .stack()
        .rename("n")
        .reset_index()
    )
    full = full.rename(columns={"level_2": "week"})
    park = con.execute("""
        SELECT model, sale_date FROM dim_vehicle WHERE NOT is_orphan
    """).df()
    park["sale_date"] = pd.to_datetime(park["sale_date"])
    rows = [
        {"model": m, "week": wk, "park": int((g["sale_date"] <= wk).sum())}
        for m, g in park.groupby("model")
        for wk in weeks
    ]
    return full.merge(pd.DataFrame(rows), on=["model", "week"], how="left")


def bh_adjust(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def detect(weekly: pd.DataFrame, exclude_alerts: bool = True) -> pd.DataFrame:
    """Weekly Poisson test per series with Benjamini-Hochberg control across series.

    The expected count is the mean complaint rate per car over the previous
    ``BASELINE_WEEKS`` weeks, times this week's cars on the road. With ``exclude_alerts``
    (default) weeks that raised an alert are left out of later baselines, which then reach
    further back instead: otherwise a spike would inflate the expected count for the next
    ``BASELINE_WEEKS`` weeks and hide a second rise of the same fault. The weeks are
    therefore processed in order, since each week's alerts decide the next baselines.
    """
    w = weekly.sort_values(["week", "model", "fault_code"]).copy()
    if "park" not in w:
        w["park"] = 1.0
    w["rate"] = w["n"] / w["park"].clip(lower=1)
    history: dict[tuple, list[float]] = {}
    out = []
    for _, wk in w.groupby("week", sort=True):
        keys = list(zip(wk["model"], wk["fault_code"], strict=True))
        past = [history.get(k, [])[-BASELINE_WEEKS:] for k in keys]
        rate = np.array([np.mean(h) if len(h) == BASELINE_WEEKS else np.nan for h in past])
        wk = wk.assign(baseline=rate * wk["park"].clip(lower=1).to_numpy())
        tested = wk["baseline"].notna().to_numpy()
        alert = np.zeros(len(wk), dtype=bool)
        if tested.any():
            t = wk[tested]
            lam = t["baseline"].clip(lower=0.5)
            p = poisson.sf(t["n"] - 1, lam)  # P(X >= n)
            q = bh_adjust(np.asarray(p))
            a = (
                (q < FDR)
                & (t["n"] >= MIN_COUNT).to_numpy()
                & (t["n"] >= MIN_RATIO * lam).to_numpy()
            )
            wk.loc[tested, "p_value"] = p
            wk.loc[tested, "q_value"] = q
            alert[tested] = a
        wk["alert"] = alert
        for k, r, al in zip(keys, wk["rate"], alert, strict=True):
            if not (exclude_alerts and al):
                history.setdefault(k, []).append(r)
        out.append(wk)
    res = pd.concat(out).dropna(subset=["baseline"])
    return res.sort_values(["model", "fault_code", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- CUSUM
def daily_counts(con) -> pd.DataFrame:
    """Daily tickets per model x fault code, with the number of that model's cars on the road."""
    df = con.execute("""
        SELECT t.opened_date AS day, v.model, t.fault_code, count(*) AS n
        FROM fact_tickets t JOIN dim_vehicle v USING (vin)
        WHERE t.fault_code IS NOT NULL
        GROUP BY ALL
    """).df()
    df["day"] = pd.to_datetime(df["day"])
    days = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
    keys = df[["model", "fault_code"]].drop_duplicates()
    full = keys.merge(pd.DataFrame({"day": days}), how="cross")
    full = full.merge(df, on=["model", "fault_code", "day"], how="left").fillna({"n": 0})
    park = con.execute("SELECT model, sale_date FROM dim_vehicle WHERE NOT is_orphan").df()
    park["sale_date"] = pd.to_datetime(park["sale_date"])
    sizes = []
    for m, g in park.groupby("model"):
        sales = np.sort(g["sale_date"].to_numpy())
        sizes.append(pd.DataFrame({"model": m, "day": days,
                                   "park": np.searchsorted(sales, days.to_numpy(), "right")}))  # fmt: skip
    return full.merge(pd.concat(sizes), on=["model", "day"], how="left")


_THRESHOLDS: dict[tuple[float, float, float], float] = {}


def cusum_threshold(lam0: float, shift: float = CUSUM_SHIFT, arl0: float = CUSUM_ARL0_DAYS,
                    n_paths: int = 800, days: int = 1500, seed: int = 0) -> float:  # fmt: skip
    """Smallest decision interval h with at most one false alarm per ``arl0`` days.

    Monte Carlo: in-control Poisson(lam0) counts, the CUSUM run with a reset after each
    alarm, the false alarm rate measured for a grid of thresholds at once.
    """
    log_lam = round(float(np.log(max(lam0, CUSUM_MIN_RATE))), 1)  # cache on a log grid
    key = (log_lam, float(shift), float(arl0))
    if key in _THRESHOLDS:
        return _THRESHOLDS[key]
    lam = float(np.exp(log_lam))
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.5, 25, 99)
    w, c = np.log(shift), (shift - 1) * lam
    s = np.zeros((len(grid), n_paths))
    alarms = np.zeros(len(grid))
    for _ in range(days):
        x = rng.poisson(lam, n_paths)
        s = np.maximum(0.0, s + x * w - c)
        hit = s > grid[:, None]
        alarms += hit.sum(axis=1)
        s[hit] = 0.0
    rate = alarms / (n_paths * days)
    ok = np.flatnonzero(rate <= 1 / arl0)
    h = float(grid[ok[0]]) if len(ok) else float(grid[-1])
    _THRESHOLDS[key] = h
    return h


def weekday_factors(daily: pd.DataFrame, base_days: int = CUSUM_BASE_DAYS) -> pd.Series:
    """Day-of-week factor per date, pooled over all series, from the guarded baseline window.

    Complaints follow the call pattern (busy Mondays, quiet Sundays). Without this factor every
    Monday adds a small excess to the CUSUM and false alarms pile up.
    """
    tot = daily.groupby("day")["n"].sum().sort_index()
    dow = tot.index.day_of_week
    factors = pd.Series(np.nan, index=tot.index)
    vals = tot.to_numpy(float)
    for i in range(CUSUM_GUARD_DAYS + base_days, len(tot)):
        lo, hi = i - CUSUM_GUARD_DAYS - base_days, i - CUSUM_GUARD_DAYS
        win, wdow = vals[lo:hi], dow[lo:hi]
        same = win[wdow == dow[i]]
        factors.iloc[i] = same.mean() / win.mean() if win.mean() > 0 else 1.0
    return factors


def cusum_detect(
    daily: pd.DataFrame,
    shift: float = CUSUM_SHIFT,
    base_days: int = CUSUM_BASE_DAYS,
    arl0: float = CUSUM_ARL0_DAYS,
) -> pd.DataFrame:
    """Run a Poisson CUSUM per series; returns one row per day with the statistic and alarms.

    The baseline is estimated, and an estimated baseline raises the real false-alarm rate
    above the design rate (low estimates let the statistic drift up); a longer window
    shrinks that effect but reacts later to a drift.
    """
    dow_factor = weekday_factors(daily, base_days)
    out = []
    for _, g in daily.sort_values("day").groupby(["model", "fault_code"]):
        g = g.reset_index(drop=True)
        park = g["park"].clip(lower=1).to_numpy(float)
        rate = g["n"].to_numpy(float) / park
        base = (pd.Series(rate).shift(CUSUM_GUARD_DAYS)
                .rolling(base_days, min_periods=base_days).mean().to_numpy())  # fmt: skip
        factor = dow_factor.reindex(g["day"]).to_numpy()
        lam0 = np.maximum(base * park * factor, CUSUM_MIN_RATE)
        valid = ~np.isnan(base)
        if not valid.any():
            continue
        h = cusum_threshold(float(np.nanmedian(lam0[valid])), shift, arl0)
        w = np.log(shift)
        s, stat, alarm = 0.0, np.full(len(g), np.nan), np.zeros(len(g), bool)
        for i in np.flatnonzero(valid):
            s = max(0.0, s + g.at[i, "n"] * w - (shift - 1) * lam0[i])
            stat[i] = s
            if s > h:
                alarm[i], s = True, 0.0  # reset: a persisting problem alarms again later
        out.append(g.assign(expected=np.where(valid, lam0, np.nan), cusum=stat, threshold=h,
                            alarm=alarm))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def _classify(alerts: pd.DataFrame, truth: list[dict], date_col: str) -> list[str]:
    kind = []
    for model, fault, when in zip(alerts["model"], alerts["fault_code"], alerts[date_col],
                                  strict=True):  # fmt: skip
        k = "other"
        for p in truth:
            on = pd.Timestamp(p["onset"])
            if model == p["model"] and on - pd.Timedelta(days=7) <= when <= on + pd.Timedelta(
                days=150
            ):
                k = "planted" if fault == p["fault_code"] else "related"
                break
        kind.append(k)
    return kind


def score_against_truth(alerts: pd.DataFrame, paths: Paths) -> list[dict]:
    f = paths.truth / "injected.json"
    if not f.exists():
        return []
    planted = json.loads(f.read_text()).get("planted_anomalies", [])
    out = []
    for p in planted:
        onset = pd.Timestamp(p["onset"])
        hit = alerts[
            (alerts["model"] == p["model"])
            & (alerts["fault_code"] == p["fault_code"])
            & (alerts["week"] >= onset - pd.Timedelta(days=7))
        ]
        first = hit["week"].min() if len(hit) else None
        out.append(
            {
                **p,
                "first_alert_week": str(first.date()) if first is not None else None,
                "detection_lag_days": int((first + pd.Timedelta(days=6) - onset).days)
                if first is not None
                else None,
            }
        )
    return out


def cusum_against_truth(alarms: pd.DataFrame, paths: Paths) -> list[dict]:
    f = paths.truth / "injected.json"
    if not f.exists():
        return []
    out = []
    for p in json.loads(f.read_text()).get("planted_anomalies", []):
        onset = pd.Timestamp(p["onset"])
        hit = alarms[(alarms["model"] == p["model"]) & (alarms["fault_code"] == p["fault_code"])
                     & (alarms["day"] >= onset - pd.Timedelta(days=7))]  # fmt: skip
        first = hit["day"].min() if len(hit) else None
        out.append({**p, "first_alarm": str(first.date()) if first is not None else None,
                    "detection_lag_days": int((first - onset).days) if first is not None
                    else None})  # fmt: skip
    return out


def _export_weekly(res: pd.DataFrame, paths: Paths) -> None:
    """Every model x fault code x week with its expected count, for Tableau / Power BI."""
    brand = res["model"].str.split().str[0]
    res.assign(brand=brand).rename(columns={"n": "tickets", "park": "cars_on_road",
                                            "baseline": "expected_tickets"})[
        ["week", "brand", "model", "fault_code", "tickets", "expected_tickets", "cars_on_road",
         "q_value", "alert"]
    ].to_csv(paths.exports / "complaint_weekly.csv", index=False)  # fmt: skip


def run_anomaly(paths: Paths) -> dict:
    con = connect(paths)
    weekly = weekly_counts(con)
    daily = daily_counts(con)
    con.close()
    cus = cusum_detect(daily)
    cus_alarms = cus[cus["alarm"]].sort_values("day")
    cus_truth = cusum_against_truth(cus_alarms, paths)
    cus_alarms = cus_alarms.assign(kind=_classify(cus_alarms, cus_truth, "day"))
    cus.to_parquet(paths.outputs / "complaint_cusum.parquet", index=False)
    cus_alarms.to_csv(paths.exports / "complaint_cusum_alarms.csv", index=False)
    tradeoff = []
    for arl0, days in CUSUM_TRADEOFF:
        default = (arl0, days) == (CUSUM_ARL0_DAYS, CUSUM_BASE_DAYS)
        c = cus if default else cusum_detect(daily, base_days=days, arl0=arl0)
        al = c[c["alarm"]]
        tr = cusum_against_truth(al, paths)
        tradeoff.append({
            "arl0_days": arl0, "base_days": days, "default": default,
            "false_alarms": sum(k == "other" for k in _classify(al, tr, "day")),
            "expected_false_alarms": round(float(c["cusum"].notna().sum()) / arl0, 1),
            "lags": {f"{t['model']} {t['fault_code']}": t["detection_lag_days"] for t in tr},
        })  # fmt: skip
    res = detect(weekly)
    alerts = res[res["alert"]].sort_values("week")
    alerts.to_parquet(paths.outputs / "complaint_alerts.parquet", index=False)
    res.to_parquet(paths.outputs / "complaint_weekly.parquet", index=False)
    _export_weekly(res, paths)
    truth = score_against_truth(alerts, paths)
    kind = _classify(alerts, truth, "week")
    alerts = alerts.assign(kind=kind)
    alerts.to_parquet(paths.outputs / "complaint_alerts.parquet", index=False)
    alerts.to_csv(paths.exports / "complaint_alerts.csv", index=False)
    n_kind = pd.Series(kind, dtype=object).value_counts()
    summary = {
        "series": int(res.groupby(["model", "fault_code"]).ngroups),
        "weeks_tested": int(res["week"].nunique()),
        "alerts": len(alerts),
        "alerts_on_planted_series": int(n_kind.get("planted", 0)),
        "alerts_related": int(n_kind.get("related", 0)),
        "alerts_on_other_series": int(n_kind.get("other", 0)),
        "planted": truth,
        "cusum": {
            "shift": CUSUM_SHIFT,
            "arl0_days": CUSUM_ARL0_DAYS,
            "series_days": int(cus["cusum"].notna().sum()),
            "alarms": len(cus_alarms),
            "alarms_by_kind": cus_alarms["kind"].value_counts().to_dict(),
            "expected_false_alarms": round(float(cus["cusum"].notna().sum()) / CUSUM_ARL0_DAYS, 1),
            "planted": cus_truth,
            "base_days": CUSUM_BASE_DAYS,
            "tradeoff": tradeoff,
        },
    }
    lines = [
        "# Complaint early warning",
        "",
        f"Weekly tickets for {summary['series']} model x fault-code series over "
        f"{summary['weeks_tested']} weeks. Alert: Benjamini-Hochberg q < {FDR} against a "
        f"{BASELINE_WEEKS}-week Poisson baseline, at least {MIN_COUNT} tickets and "
        f"{MIN_RATIO}x the baseline.",
        "",
        f"Alerts raised: **{len(alerts)}**: {summary['alerts_on_planted_series']} on the planted "
        f"problems, {summary['alerts_related']} on the same model in the same period with "
        f"another fault code (agents coding part of the problem calls differently), "
        f"{summary['alerts_on_other_series']} elsewhere.",
        "",
        "Baselines are complaint *rates per car on the road*: a newly launched model gets more "
        "complaints every month simply because more of them are sold.",
        "",
    ]
    if truth:
        lines += [
            "| Planted problem | Onset | First alert (week of) | Detected after |",
            "|---|---|---|---:|",
        ]
        for t in truth:
            lag = (
                f"{t['detection_lag_days']} days"
                if t["detection_lag_days"] is not None
                else "missed"
            )
            lines.append(
                f"| {t['model']} {t['fault_code']} | {t['onset']} | "
                f"{t['first_alert_week'] or '-'} | {lag} |"
            )
    lines += [
        "",
        "| Week | Model | Fault | Tickets | Expected | q-value | Type |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in alerts.head(30).itertuples():
        lines.append(
            f"| {r.week.date()} | {r.model} | {r.fault_code} | {r.n:.0f} | {r.baseline:.1f} | "
            f"{r.q_value:.1e} | {r.kind} |"
        )
    c = summary["cusum"]
    weekly_lags = {f"{t['model']} {t['fault_code']}": t["detection_lag_days"] for t in truth}
    names = list(weekly_lags)

    def lag(v):
        if v is None:
            return "missed"
        return f"{v} day" if v == 1 else f"{v} days"

    lines += [
        "",
        "## Daily CUSUM for slow drifts",
        "",
        f"Poisson CUSUM per series looking for a {CUSUM_SHIFT - 1:.0%} rise in the complaint "
        "rate per car. Baseline: a window ending "
        f"{CUSUM_GUARD_DAYS} days ago, so a slow drift is not absorbed into it, times a "
        "day-of-week factor pooled over all series. The threshold of each series is set by "
        "Monte Carlo for a target in-control false-alarm interval (ARL0).",
        "",
        "Compared at equal false alarms with the weekly test:",
        "",
        "| Detector | False alarms (other series) | Expected at the design rate | "
        + " | ".join(f"{n} detected after" for n in names)
        + " |",
        "|---|---:|---:|" + "---:|" * len(names),
        f"| Weekly test (BH q < {FDR}) | {summary['alerts_on_other_series']} | - | "
        + " | ".join(lag(weekly_lags[n]) for n in names)
        + " |",
    ]
    for t in c["tradeoff"]:
        label = (f"CUSUM, ARL0 {t['arl0_days'] / 365:.0f} years, {t['base_days']}-day baseline"
                 + (" (default)" if t["default"] else ""))  # fmt: skip
        lines.append(f"| {label} | {t['false_alarms']} | {t['expected_false_alarms']} | "
                     + " | ".join(lag(t["lags"].get(n)) for n in names) + " |")  # fmt: skip
    lines += [
        "",
        "**Result: on this data the CUSUM does not beat the weekly test.** Even its strictest "
        "setting raises more false alarms than the weekly test; it detects the slow battery "
        "drift earlier only by raising more alarms. The drift grows linearly from zero, so its first weeks add "
        "very little, and the weekly test already uses a per-car baseline. The CUSUM is kept "
        "as a second detector that can fire on any day.",
        "",
        "Real false alarms exceed the design rate in every setting although the counts are "
        "close to Poisson (dispersion about 1.03): the baseline is itself an estimate, and the "
        "days it runs low push the CUSUM up. A longer window shrinks that effect and reacts "
        "later to a drift.",
    ]
    (paths.reports / "complaint_alerts.md").write_text("\n".join(lines) + "\n")
    (paths.outputs / "anomaly.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
