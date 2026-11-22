"""Check the staffing models against a simulated, staffed contact center.

Two questions:

1. **Prediction:** given an interval's calls, handle time and the agents who were actually
   logged in, how close are the service level and abandonment that Erlang C and Erlang A
   predict to what happened?
2. **Decision:** staffed by the Erlang C plan or by the Erlang A plan, does the center reach
   80/20 with at most 5% abandonment, and for how many agent hours? Both plans see the same
   random calls (common random numbers), so the difference comes from the plan alone.

Inputs come from the warehouse: the intraday profile, daily volume by weekday, the handle
time distribution and the estimated patience. The simulator then breaks the models'
assumptions (see ``queue_sim``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .forecast import intraday_profile
from .paths import Paths
from .quality import connect
from .queue_sim import (
    CenterConfig,
    estimate_patience_from_calls,
    interval_stats,
    simulate_day,
)
from .staffing import (
    erlang_a,
    estimate_patience,
    required_agents,
    required_agents_erlang_a,
    service_level,
)

TARGET_SL, TARGET_SEC, MAX_ABANDON, MAX_OCC = 0.80, 20, 0.05, 0.90
MIN_CALLS = 10  # intervals with fewer calls say little about a rate


def _inputs(con, weeks: int = 4) -> dict:
    prof = intraday_profile(con, weeks=8)
    daily = con.execute(f"""
        SELECT (dayofweek(f.call_date) + 6) % 7 AS dow, count(*) * 1.0 / count(DISTINCT f.call_date) AS calls
        FROM fact_calls f JOIN dim_date d ON d.date = f.call_date
        WHERE NOT d.is_incomplete_load AND NOT d.is_holiday
          AND f.call_date > (SELECT max(call_date) FROM fact_calls) - {7 * weeks}
        GROUP BY 1 ORDER BY 1
    """).df()  # fmt: skip
    h = con.execute("""
        SELECT ln(handle_sec) AS lh FROM fact_calls
        WHERE is_answered AND handle_sec > 0
          AND call_date > (SELECT max(call_date) FROM fact_calls) - 56
    """).df()["lh"]
    return {"profile": prof, "daily": daily.set_index("dow")["calls"], "log_mean": float(h.mean()),
            "log_sd": float(h.std()), "patience": estimate_patience(con)["patience_sec"]}  # fmt: skip


def _plan(expected: np.ndarray, aht: np.ndarray, model: str, patience: float) -> np.ndarray:
    out = []
    for c, a in zip(expected, aht, strict=True):
        n_c = required_agents(c, a, target_sl=TARGET_SL, target_sec=TARGET_SEC,
                              max_occupancy=MAX_OCC)["agents"]  # fmt: skip
        if model == "erlang_c" or c <= 0:
            out.append(n_c)
        else:
            out.append(required_agents_erlang_a(
                c, a, patience, target_sl=TARGET_SL, target_sec=TARGET_SEC,
                max_occupancy=MAX_OCC, max_abandon=MAX_ABANDON, start=max(n_c, 1),
            )["agents"])  # fmt: skip
    return np.array(out)


def simulate(inputs: dict, days: int = 28, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Simulate ``days`` days for each staffing plan with the same random calls."""
    prof = inputs["profile"]
    cfg = CenterConfig(inputs["log_mean"], inputs["log_sd"], inputs["patience"])
    plans = {"erlang_c": [], "erlang_a": []}
    calls = {k: [] for k in plans}
    for d in range(days):
        dow = d % 7
        p = prof[prof["dow"] == dow].sort_values("slot")
        expected = inputs["daily"][dow] * p["share"].to_numpy()
        aht = p["aht_sec"].fillna(p["aht_sec"].mean()).to_numpy()
        for model in plans:
            agents = _plan(expected, aht, model, inputs["patience"])
            rng = np.random.default_rng(seed * 1000 + d)  # same calls for both plans
            day = simulate_day(rng, cfg, expected, agents, day_start=d * 86400.0)
            calls[model].append(day.assign(day=d, dow=dow))
            plans[model].append(pd.DataFrame({"day": d, "interval": np.arange(len(agents)),
                                              "planned": agents, "expected": expected,
                                              "planned_aht": aht}))  # fmt: skip
    return {m: pd.concat(calls[m], ignore_index=True) for m in calls} | {
        f"plan_{m}": pd.concat(plans[m], ignore_index=True) for m in plans
    }


def predict(obs: pd.DataFrame, patience: float) -> pd.DataFrame:
    """Erlang C and Erlang A predictions from what was observed in each interval."""
    o = obs.copy()
    sl_c, sl_a, ab_a = [], [], []
    for c, a, n in zip(o["offered"], o["aht_sec"], o["agents"], strict=True):
        traffic = c * a / 1800
        sl_c.append(service_level(traffic, int(n), a, TARGET_SEC) if n > 0 else 0.0)
        r = erlang_a(c, a, patience, int(n), target_sec=TARGET_SEC) if n > 0 else {
            "service_level": 0.0, "p_abandon": 1.0}  # fmt: skip
        sl_a.append(r["service_level"])
        ab_a.append(r["p_abandon"])
    return o.assign(sl_erlang_c=sl_c, sl_erlang_a=sl_a, abandon_erlang_a=ab_a,
                    load=o["offered"] * o["aht_sec"] / 1800 / o["agents"].clip(lower=1))  # fmt: skip


def _err(pred, obs, w):
    e = pred - obs
    return {"mae": float(np.average(np.abs(e), weights=w)), "bias": float(np.average(e, weights=w))}


def run_staffing_validation(paths: Paths, days: int = 28, seed: int = 7) -> dict:
    con = connect(paths)
    inputs = _inputs(con)
    con.close()
    sim = simulate(inputs, days=days, seed=seed)

    # 1. prediction accuracy, on the intervals of both simulated plans
    obs = pd.concat([interval_stats(sim[m]).assign(plan=m) for m in ("erlang_c", "erlang_a")])
    est_patience = estimate_patience_from_calls(sim["erlang_c"])
    pred = predict(obs.dropna(subset=["aht_sec"]), est_patience)
    pred = pred[pred["offered"] >= MIN_CALLS]
    w = pred["offered"]
    accuracy = {
        "intervals": len(pred),
        "service_level": {"erlang_c": _err(pred["sl_erlang_c"], pred["service_level"], w),
                          "erlang_a": _err(pred["sl_erlang_a"], pred["service_level"], w)},
        "abandonment": {"erlang_c": _err(0 * pred["abandon_rate"], pred["abandon_rate"], w),
                        "erlang_a": _err(pred["abandon_erlang_a"], pred["abandon_rate"], w)},
    }  # fmt: skip
    bands = pd.cut(
        pred["load"], [0, 0.8, 0.9, 1.0, 10], labels=["< 80%", "80-90%", "90-100%", "> 100%"]
    )
    by_load = pred.groupby(bands, observed=True).apply(
        lambda g: pd.Series({
            "intervals": len(g),
            "observed_sl": np.average(g["service_level"], weights=g["offered"]),
            "erlang_c_sl": np.average(g["sl_erlang_c"], weights=g["offered"]),
            "erlang_a_sl": np.average(g["sl_erlang_a"], weights=g["offered"]),
            "observed_abandon": np.average(g["abandon_rate"], weights=g["offered"]),
            "erlang_a_abandon": np.average(g["abandon_erlang_a"], weights=g["offered"]),
        }), include_groups=False,
    )  # fmt: skip

    # 2. decision: what each plan delivers
    decision = {}
    for m in ("erlang_c", "erlang_a"):
        st = interval_stats(sim[m])
        busy = st[st["offered"] >= MIN_CALLS]
        plan = sim[f"plan_{m}"]
        decision[m] = {
            "planned_agent_hours": float(plan["planned"].sum() / 2),
            "service_level": float(st["in_target"].sum() / st["offered"].sum()),
            "abandon_rate": float(st["abandoned"].sum() / st["offered"].sum()),
            "intervals_meeting_sl": float((busy["service_level"] >= TARGET_SL).mean()),
            "intervals_meeting_abandon": float((busy["abandon_rate"] <= MAX_ABANDON).mean()),
            "calls": int(st["offered"].sum()),
        }  # fmt: skip

    summary = {
        "days": days, "seed": seed,
        "inputs": {"handle_log_mean": inputs["log_mean"], "handle_log_sd": inputs["log_sd"],
                   "handle_cv": float(np.sqrt(np.exp(inputs["log_sd"] ** 2) - 1)),
                   "patience_used_by_planners": inputs["patience"],
                   "patience_true_exponential_part": inputs["patience"],
                   "patience_grace_sec": CenterConfig(0, 0, 0).patience_grace_sec,
                   "patience_estimated_from_simulation": est_patience},
        "accuracy": accuracy,
        "by_load": by_load.round(4).reset_index(names="load").to_dict(orient="records"),
        "decision": decision,
    }  # fmt: skip
    pred.to_parquet(paths.outputs / "staffing_validation_intervals.parquet", index=False)
    (paths.outputs / "staffing_validation.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    _write_report(paths, summary)
    return summary


def _write_report(paths: Paths, s: dict) -> None:
    a, d, i = s["accuracy"], s["decision"], s["inputs"]
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    pp = lambda x: f"{x * 100:+.1f} pt"  # noqa: E731
    lines = [
        "# Staffing models against a simulated, staffed center",
        "",
        f"{s['days']} simulated days, one pooled FCFS queue, calls and handle times from the data. "
        "The simulation breaks the models' assumptions on purpose: arrivals change within the "
        f"interval, handle times are lognormal (coefficient of variation {i['handle_cv']:.2f}; "
        f"exponential would be 1), nobody hangs up in the first {i['patience_grace_sec']:.0f} s, "
        "and absenteeism lowers the agents logged in below the plan.",
        "",
        f"Patience: the planners use {i['patience_used_by_planners']:.0f} s (estimated from the "
        f"warehouse); re-estimated from the simulated calls with the same estimator: "
        f"{i['patience_estimated_from_simulation']:.0f} s.",
        "",
        "## 1. Prediction: given the agents actually logged in",
        "",
        f"{a['intervals']:,} intervals with at least {MIN_CALLS} calls, weighted by calls.",
        "",
        "| Model | Service level: mean abs. error | bias | Abandonment: mean abs. error | bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, name in (("erlang_c", "Erlang C"), ("erlang_a", "Erlang A")):
        sl, ab = a["service_level"][m], a["abandonment"][m]
        lines.append(f"| {name} | {sl['mae'] * 100:.1f} pt | {pp(sl['bias'])} | "
                     f"{ab['mae'] * 100:.1f} pt | {pp(ab['bias'])} |")  # fmt: skip
    lines += ["", "By load (offered Erlangs per logged-in agent):", "",
              "| Load | Intervals | Observed SL | Erlang C | Erlang A | Observed abandonment | Erlang A |",
              "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in s["by_load"]:
        lines.append(f"| {r['load']} | {int(r['intervals'])} | {pct(r['observed_sl'])} | "
                     f"{pct(r['erlang_c_sl'])} | {pct(r['erlang_a_sl'])} | "
                     f"{pct(r['observed_abandon'])} | {pct(r['erlang_a_abandon'])} |")  # fmt: skip
    lines += [
        "",
        "## 2. Decision: what each plan delivers",
        "",
        "Same random calls for both plans. Target: 80% answered within 20 s, at most 5% abandoned.",
        "",
        "| Plan | Agent hours | Service level | Abandonment | Busy intervals meeting 80/20 | "
        "meeting the 5% abandonment cap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, name in (("erlang_c", "Erlang C"), ("erlang_a", "Erlang A")):
        x = d[m]
        lines.append(f"| {name} | {x['planned_agent_hours']:,.0f} | {pct(x['service_level'])} | "
                     f"{pct(x['abandon_rate'])} | {pct(x['intervals_meeting_sl'])} | "
                     f"{pct(x['intervals_meeting_abandon'])} |")  # fmt: skip
    (paths.reports / "staffing_validation.md").write_text("\n".join(lines) + "\n")
