"""Skills-based routing: dedicated groups, a fully pooled group, and a little cross-training.

The four queues (sales, service, complaints, general) differ in volume and handle time.
Staffing each queue on its own wastes capacity: every small group carries its own safety
margin. One pooled group is cheapest, but every agent would need to handle complaints and
sales. The usual answer is partial flexibility: keep specialists and cross-train a share of
the agents to take any queue. This module asks how large that share must be to reach the
pooled headcount without missing the target on any queue.

Plans (per 30-minute interval, all sized with Erlang A):

* dedicated: ``N_q`` agents per queue, each queue sized alone
* pooled: ``N_pool`` generalists, sized on the combined stream with the mixed handle time
* flex ``f``: the pooled headcount ``N_pool``, of which a share ``f`` is cross-trained and
  the rest are specialists, split across queues by workload (calls x handle time)

Routing: a new call goes to an idle specialist of its queue first, then to an idle
generalist. An agent who becomes free takes the longest-waiting call among its skills.
The simulation is event-driven, with the same broken assumptions as ``queue_sim``
(non-stationary arrivals, lognormal handle times, a patience grace period, absenteeism),
and all plans see the same random calls.
"""

from __future__ import annotations

import heapq
import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import intraday_profile
from .paths import Paths
from .quality import connect
from .queue_sim import INTERVAL_SEC, CenterConfig, _arrivals
from .staffing import estimate_patience, required_agents, required_agents_erlang_a

TARGET_SL, TARGET_SEC, MAX_ABANDON, MAX_OCC = 0.80, 20, 0.05, 0.90
SHORT_SEC = 5
MIN_CALLS = 10
QUEUES = ("satis", "servis", "sikayet", "genel")
FLEX_SHARES = (None, 0.1, 0.2, 0.3, 0.5, 1.0)  # None = dedicated
GENERALIST = "generalist"


@dataclass(frozen=True)
class Group:
    name: str
    skills: tuple[str, ...]


def groups_for(queues=QUEUES) -> list[Group]:
    return [Group(q, (q,)) for q in queues] + [Group(GENERALIST, tuple(queues))]


def simulate_skills_day(
    rng,
    cfg: CenterConfig,
    expected: dict[str, np.ndarray],
    handle: dict[str, tuple[float, float]],
    staffing: dict[str, np.ndarray],
    groups: list[Group],
    day_start: float = 0.0,
) -> pd.DataFrame:
    """Simulate one day with several queues and skill groups; one row per call.

    ``expected[q]`` is the expected calls per interval, ``handle[q]`` the lognormal
    (log-mean, log-sd) of handle time, ``staffing[g]`` the planned agents per interval.
    Random draws do not depend on the staffing, so different plans see the same calls.
    """
    n_int = len(next(iter(expected.values())))
    volume = float(np.exp(rng.normal(0, cfg.day_noise_sd)))
    q_of, arr, hdl, pat = [], [], [], []
    for q in expected:  # fixed order: common random numbers across plans
        t = _arrivals(rng, expected[q] * volume, day_start, cfg.stationary)
        lm, ls = handle[q]
        h = rng.exponential(np.exp(lm), len(t)) if cfg.exponential_handle else rng.lognormal(
            lm, ls, len(t))  # fmt: skip
        q_of += [q] * len(t)
        arr.append(t)
        hdl.append(h)
        pat.append(cfg.patience_grace_sec + rng.exponential(cfg.patience_sec, len(t)))
    # one center-wide absenteeism draw per interval, applied to every group
    absent = np.clip(rng.normal(cfg.absenteeism_mean, cfg.absenteeism_sd, n_int), -0.05, 0.4)
    arrived, handle_sec = np.concatenate(arr), np.concatenate(hdl)
    deadline = arrived + np.concatenate(pat)
    queue = np.array(q_of, dtype=object)
    n = len(arrived)
    wait = np.full(n, np.nan)
    answered = np.zeros(n, dtype=bool)
    served_by = np.full(n, "", dtype=object)

    agents = {g.name: np.maximum(np.round(np.asarray(staffing.get(g.name, np.zeros(n_int)))
                                          * (1 - absent)), 0).astype(int)
              for g in groups}  # fmt: skip
    idle = {g.name: 0 for g in groups}
    busy = {g.name: 0 for g in groups}
    leaving = {g.name: 0 for g in groups}  # busy agents who log off when their call ends
    waiting = {q: deque() for q in expected}
    by_skill = {q: sorted((g for g in groups if q in g.skills), key=lambda g: len(g.skills))
                for q in expected}  # specialists first  # fmt: skip
    skills_of = {g.name: g.skills for g in groups}

    events = []  # (time, kind, seq, payload); kind 0 = shift, 1 = departure, 2 = arrival
    for k in range(n_int):
        events.append((day_start + k * INTERVAL_SEC, 0, k, k))
    for i in range(n):
        events.append((arrived[i], 2, i, i))
    heapq.heapify(events)
    seq = n

    def purge(q, now):
        w = waiting[q]
        while w and deadline[w[0]] < now:
            i = w.popleft()
            wait[i] = deadline[i] - arrived[i]

    def serve(i, g, now):
        nonlocal seq
        idle[g] -= 1
        busy[g] += 1
        wait[i] = now - arrived[i]
        answered[i] = True
        served_by[i] = g
        seq += 1
        heapq.heappush(events, (now + handle_sec[i], 1, seq, g))

    def pull(g, now):
        while idle[g] > 0:
            best = None
            for q in skills_of[g]:
                if q not in waiting:
                    continue
                purge(q, now)
                if waiting[q] and (
                    best is None or arrived[waiting[q][0]] < arrived[waiting[best][0]]
                ):
                    best = q
            if best is None:
                return
            serve(waiting[best].popleft(), g, now)

    while events:
        now, kind, _, x = heapq.heappop(events)
        if kind == 0:
            for g in agents:
                target = int(agents[g][x])
                current = idle[g] + busy[g] - leaving[g]
                if target > current:
                    add = target - current
                    back = min(leaving[g], add)
                    leaving[g] -= back
                    idle[g] += add - back
                    pull(g, now)
                elif target < current:
                    drop = current - target
                    off = min(idle[g], drop)
                    idle[g] -= off
                    leaving[g] += drop - off
        elif kind == 1:
            busy[x] -= 1
            if leaving[x] > 0:
                leaving[x] -= 1
            else:
                idle[x] += 1
                pull(x, now)
        else:
            q = queue[x]
            for g in by_skill[q]:
                if idle[g.name] > 0:
                    serve(x, g.name, now)
                    break
            else:
                waiting[q].append(x)
    for q in waiting:  # nobody left to answer them
        for i in waiting[q]:
            wait[i] = deadline[i] - arrived[i]

    interval = ((arrived - day_start) // INTERVAL_SEC).astype(int)
    return pd.DataFrame({"queue": queue, "interval": interval, "arrived": arrived,
                         "wait_sec": wait, "answered": answered, "handle_sec": handle_sec,
                         "served_by": served_by})  # fmt: skip


def queue_stats(calls: pd.DataFrame, keys=("queue",)) -> pd.DataFrame:
    c = calls.assign(
        in_target=calls["answered"] & (calls["wait_sec"] <= TARGET_SEC),
        abandoned=~calls["answered"] & (calls["wait_sec"] >= SHORT_SEC),
    )
    g = c.groupby(list(keys))
    out = pd.DataFrame({"offered": g.size(), "in_target": g["in_target"].sum(),
                        "abandoned": g["abandoned"].sum()})  # fmt: skip
    out["service_level"] = out["in_target"] / out["offered"]
    out["abandon_rate"] = out["abandoned"] / out["offered"]
    return out.reset_index()


# ---------------------------------------------------------------- planning


def _inputs(con, weeks: int = 4) -> dict:
    prof = intraday_profile(con, weeks=8)
    daily = con.execute(f"""
        SELECT (dayofweek(f.call_date) + 6) % 7 AS dow, f.queue,
               count(*) * 1.0 / count(DISTINCT f.call_date) AS calls
        FROM fact_calls f JOIN dim_date d ON d.date = f.call_date
        WHERE NOT d.is_incomplete_load AND NOT d.is_holiday
          AND f.call_date > (SELECT max(call_date) FROM fact_calls) - {7 * weeks}
        GROUP BY ALL
    """).df()  # fmt: skip
    h = (
        con.execute("""
        SELECT queue, avg(ln(handle_sec)) AS lm, stddev(ln(handle_sec)) AS ls
        FROM fact_calls
        WHERE is_answered AND handle_sec > 0
          AND call_date > (SELECT max(call_date) FROM fact_calls) - 56
        GROUP BY 1
    """)
        .df()
        .set_index("queue")
    )
    return {"profile": prof, "daily": daily.pivot(index="dow", columns="queue", values="calls"),
            "handle": {q: (float(h.loc[q, "lm"]), float(h.loc[q, "ls"])) for q in QUEUES},
            "patience": estimate_patience(con)["patience_sec"]}  # fmt: skip


def _need(calls: float, aht: float, patience: float) -> int:
    if calls <= 0:
        return 0
    n_c = required_agents(calls, aht, target_sl=TARGET_SL, target_sec=TARGET_SEC,
                          max_occupancy=MAX_OCC)["agents"]  # fmt: skip
    return required_agents_erlang_a(
        calls, aht, patience, target_sl=TARGET_SL, target_sec=TARGET_SEC,
        max_occupancy=MAX_OCC, max_abandon=MAX_ABANDON, start=max(n_c, 1),
    )["agents"]  # fmt: skip


def requirements(expected: dict[str, np.ndarray], aht: dict[str, float], patience: float):
    """Per-queue requirement ``N_q`` and pooled requirement ``N_pool`` per interval."""
    per_q = {q: np.array([_need(c, aht[q], patience) for c in expected[q]]) for q in expected}
    total = sum(expected.values())
    mixed = sum(expected[q] * aht[q] for q in expected) / np.maximum(total, 1e-9)
    pooled = np.array([_need(c, a, patience) for c, a in zip(total, mixed, strict=True)])
    return per_q, pooled


def _split(total: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split an integer headcount across queues by weight (largest remainder)."""
    qs = list(weights)
    w = np.vstack([weights[q] for q in qs]).astype(float)
    w = np.divide(w, w.sum(0), out=np.zeros_like(w), where=w.sum(0) > 0)
    raw = w * total
    out = np.floor(raw).astype(int)
    for k in range(out.shape[1]):
        left = int(total[k] - out[:, k].sum())
        if left > 0:
            out[np.argsort(-(raw[:, k] - out[:, k]))[:left], k] += 1
    return {q: out[i] for i, q in enumerate(qs)}


def flex_plan(per_q: dict[str, np.ndarray], pooled: np.ndarray, f: float | None,
              workload: dict[str, np.ndarray], extra: int = 0) -> dict[str, np.ndarray]:  # fmt: skip
    """The staffing of one design for one day.

    ``f = None``: dedicated, ``N_q`` per queue. Otherwise the pooled requirement ``N_pool``,
    of which ``round((1 - f) N_pool)`` are specialists split by workload and the rest are
    generalists; ``extra`` generalists are added to every open interval.
    """
    if f is None:
        return dict(per_q) | {GENERALIST: np.zeros_like(pooled)}
    spec = _split(np.round((1 - f) * pooled).astype(int), workload)
    gen = pooled - sum(spec.values()) + extra * (pooled > 0)
    return spec | {GENERALIST: gen}


@dataclass
class _Day:
    dow: int
    expected: dict
    per_q: dict
    pooled: np.ndarray
    workload: dict


def _days(inputs: dict, days: int) -> tuple[list[_Day], dict[str, float]]:
    prof = inputs["profile"]
    aht = {q: float(np.exp(lm + ls**2 / 2)) for q, (lm, ls) in inputs["handle"].items()}
    out = []
    for d in range(days):
        dow = d % 7
        share = prof[prof["dow"] == dow].sort_values("slot")["share"].to_numpy()
        expected = {q: inputs["daily"].loc[dow, q] * share for q in QUEUES}
        per_q, pooled = requirements(expected, aht, inputs["patience"])
        out.append(_Day(dow, expected, per_q, pooled, {q: expected[q] * aht[q] for q in QUEUES}))
    return out, aht


def run_plan(inputs: dict, days: list[_Day], f: float | None, extra: int = 0, seed: int = 11):
    """Simulate one design over all days; same random calls for every design."""
    cfg = CenterConfig(0.0, 0.0, inputs["patience"])  # handle times come per queue
    calls, hours = [], {}
    for d, day in enumerate(days):
        plan = flex_plan(day.per_q, day.pooled, f, day.workload, extra)
        rng = np.random.default_rng(seed * 1000 + d)
        sim = simulate_skills_day(rng, cfg, day.expected, inputs["handle"], plan, groups_for(),
                                  day_start=d * 86400.0)  # fmt: skip
        calls.append(sim.assign(day=d))
        for g, n in plan.items():
            hours[g] = hours.get(g, 0.0) + float(n.sum()) / 2
    return pd.concat(calls, ignore_index=True), hours


def evaluate(calls: pd.DataFrame, hours: dict) -> dict:
    total = queue_stats(calls.assign(all="all"), keys=("all",)).iloc[0]
    per_q = queue_stats(calls).set_index("queue")
    iv = queue_stats(calls, keys=("queue", "day", "interval"))
    iv = iv[iv["offered"] >= MIN_CALLS]
    agent_hours = sum(hours.values())
    return {
        "agent_hours": agent_hours,
        "cross_trained_share": hours.get(GENERALIST, 0.0) / max(agent_hours, 1e-9),
        "service_level": float(total["service_level"]),
        "abandon_rate": float(total["abandon_rate"]),
        "worst_queue": str(per_q["service_level"].idxmin()),
        "worst_queue_sl": float(per_q["service_level"].min()),
        "worst_queue_abandon": float(per_q["abandon_rate"].max()),
        "busy_queue_intervals_meeting_sl": float((iv["service_level"] >= TARGET_SL).mean()),
        "per_queue": {q: {"service_level": float(per_q.loc[q, "service_level"]),
                          "abandon_rate": float(per_q.loc[q, "abandon_rate"])}
                      for q in per_q.index},
    }  # fmt: skip


def meets_target(r: dict) -> bool:
    """Every queue at 80/20 and at most 5% abandonment, over the simulated month."""
    return r["worst_queue_sl"] >= TARGET_SL and r["worst_queue_abandon"] <= MAX_ABANDON


def _label(f: float | None) -> str:
    if f is None:
        return "dedicated"
    return "pooled" if f == 1 else f"{f:.0%} cross-trained"


def run_skills_routing(paths: Paths, days: int = 28, seed: int = 11, max_extra: int = 8) -> dict:
    con = connect(paths)
    inputs = _inputs(con)
    con.close()
    plan_days, aht = _days(inputs, days)

    rows = []
    for f in FLEX_SHARES:
        # at the requirement as planned, then with the fewest extra generalists that pass
        first = evaluate(*run_plan(inputs, plan_days, f, 0, seed))
        row = {"plan": _label(f), "flex_share": f, "as_planned": first, "extra_agents": 0,
               "passing": first if meets_target(first) else None}  # fmt: skip
        if f is not None and row["passing"] is None:
            for extra in range(1, max_extra + 1):
                r = evaluate(*run_plan(inputs, plan_days, f, extra, seed))
                if meets_target(r):
                    row["extra_agents"], row["passing"] = extra, r
                    break
        rows.append(row)
    dedicated = rows[0]["as_planned"]["agent_hours"]
    for r in rows:
        for k in ("as_planned", "passing"):
            if r[k]:
                r[k]["hours_vs_dedicated"] = r[k]["agent_hours"] / dedicated - 1
    summary = {
        "days": days, "seed": seed, "patience_sec": inputs["patience"], "aht_sec": aht,
        "calls_per_day": {q: float(inputs["daily"][q].mean()) for q in QUEUES},
        "dedicated_meets_target": meets_target(rows[0]["as_planned"]),
        "plans": rows,
    }  # fmt: skip
    (paths.outputs / "skills_routing.json").write_text(json.dumps(summary, indent=2))
    _write_report(paths, summary)
    best = min((r for r in rows if r["passing"]), key=lambda r: r["passing"]["agent_hours"],
               default=None)  # fmt: skip
    return {"plans": len(rows), "dedicated_hours": dedicated,
            "cheapest_passing": best["plan"] if best else None,
            "cheapest_vs_dedicated": best["passing"]["hours_vs_dedicated"] if best else None}  # fmt: skip


def _write_report(paths: Paths, s: dict) -> None:
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Skills-based routing: how much cross-training is enough?",
        "",
        f"{s['days']} simulated days, four queues with their own volume and handle time, all "
        "designs run on the same random calls, with the same broken assumptions as the staffing "
        "validation (arrivals change within the interval, lognormal handle times, a patience "
        "grace period, absenteeism). Routing: an idle specialist of the call's queue first, "
        "then an idle generalist; a free agent takes the longest-waiting call among its skills.",
        "",
        "| Queue | Calls per day | Mean handle time |",
        "|---|---:|---:|",
    ]
    for q in QUEUES:
        lines.append(f"| {q} | {s['calls_per_day'][q]:,.0f} | {s['aht_sec'][q]:.0f} s |")
    lines += [
        "",
        "Designs, each sized with Erlang A per 30-minute interval:",
        "",
        "* **dedicated:** every queue staffed alone, specialists only",
        "* **x% cross-trained:** the pooled requirement (one Erlang A on all calls, mixed "
        "handle time), of which x% are generalists who take any queue and the rest are "
        "specialists split by workload",
        "* **pooled:** the pooled requirement, all generalists",
        "",
        "## As planned",
        "",
        "| Design | Agent hours | vs dedicated | Service level | Abandonment | Worst queue SL "
        "| Worst queue abandonment | Busy queue-intervals at 80/20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["plans"]:
        a = r["as_planned"]
        lines.append(
            f"| {r['plan']} | {a['agent_hours']:,.0f} | {a['hours_vs_dedicated']:+.1%} | "
            f"{pct(a['service_level'])} | {pct(a['abandon_rate'])} | "
            f"{pct(a['worst_queue_sl'])} ({a['worst_queue']}) | "
            f"{pct(a['worst_queue_abandon'])} | {pct(a['busy_queue_intervals_meeting_sl'])} |"
        )
    lines += [
        "",
        "## Cheapest staffing that meets the target on every queue",
        "",
        "Target: every queue at least 80% answered within 20 s and at most 5% abandoned over "
        "the month. Designs that miss it get extra generalists in every open interval until "
        "they pass.",
        "",
        "| Design | Extra generalists per interval | Agent hours | vs dedicated "
        "| Cross-trained share of hours | Worst queue SL | Worst queue abandonment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["plans"]:
        p = r["passing"]
        if p is None:
            lines.append(f"| {r['plan']} | did not pass | | | | | |")
            continue
        lines.append(
            f"| {r['plan']} | {r['extra_agents']} | {p['agent_hours']:,.0f} | "
            f"{p['hours_vs_dedicated']:+.1%} | {pct(p['cross_trained_share'])} | "
            f"{pct(p['worst_queue_sl'])} ({p['worst_queue']}) | {pct(p['worst_queue_abandon'])} |"
        )
    lines += ["", "Service level per queue, as planned:", "",
              "| Design | " + " | ".join(QUEUES) + " |", "|---|" + "---:|" * len(QUEUES)]  # fmt: skip
    for r in s["plans"]:
        lines.append(f"| {r['plan']} | " + " | ".join(
            pct(r["as_planned"]["per_queue"][q]["service_level"]) for q in QUEUES) + " |")  # fmt: skip
    (paths.reports / "skills_routing.md").write_text("\n".join(lines) + "\n")
