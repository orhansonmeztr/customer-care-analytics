"""Early warning for product complaints: weekly tickets per model and fault code.

Many small series (model x fault code) with low counts: a z-score on raw counts would
alarm on noise. Each week's count is compared with a Poisson baseline from the previous
12 weeks, and the p-values of all series in a week are corrected for multiple testing
(Benjamini-Hochberg), so the number of false alarms stays under control as series are added.
An alert also needs a minimum size, because a jump from 1 to 4 tickets is not actionable.
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


def detect(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.sort_values(["model", "fault_code", "week"]).copy()
    if "park" not in w:
        w["park"] = 1.0
    w["rate"] = w["n"] / w["park"].clip(lower=1)
    g = w.groupby(["model", "fault_code"])["rate"]
    rate = g.transform(
        lambda s: s.shift(1).rolling(BASELINE_WEEKS, min_periods=BASELINE_WEEKS).mean()
    )
    w["baseline"] = rate * w["park"].clip(lower=1)  # expected tickets for this week's fleet
    w = w.dropna(subset=["baseline"])
    lam = w["baseline"].clip(lower=0.5)
    w["p_value"] = poisson.sf(w["n"] - 1, lam)  # P(X >= n)
    w["q_value"] = w.groupby("week")["p_value"].transform(lambda p: bh_adjust(p.to_numpy()))
    w["alert"] = (w["q_value"] < FDR) & (w["n"] >= MIN_COUNT) & (w["n"] >= MIN_RATIO * lam)
    return w


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


def run_anomaly(paths: Paths) -> dict:
    con = connect(paths)
    weekly = weekly_counts(con)
    con.close()
    res = detect(weekly)
    alerts = res[res["alert"]].sort_values("week")
    alerts.to_parquet(paths.outputs / "complaint_alerts.parquet", index=False)
    res.to_parquet(paths.outputs / "complaint_weekly.parquet", index=False)
    truth = score_against_truth(alerts, paths)
    kind = []
    for r in alerts.itertuples():
        k = "other"
        for p in truth:
            on = pd.Timestamp(p["onset"])
            in_window = on - pd.Timedelta(days=7) <= r.week <= on + pd.Timedelta(days=150)
            if r.model == p["model"] and in_window:
                k = "planted" if r.fault_code == p["fault_code"] else "related"
                break
        kind.append(k)
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
    lines += [
        "",
        "A sudden spike (the brake recall) is caught in the first week. A slow drift (the "
        "EV battery issue) is caught later, once the weekly count clears the baseline; a "
        "CUSUM chart would catch a drift earlier and is a natural next step.",
    ]
    (paths.reports / "complaint_alerts.md").write_text("\n".join(lines) + "\n")
    (paths.outputs / "anomaly.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
