"""Staffing: how many agents an interval needs to reach a service level target.

Two queueing models:

* **Erlang C** (M/M/N): callers never hang up. Simple, the industry default, and it
  over-staffs, because in reality impatient callers leave the queue and shorten it.
* **Erlang A** (M/M/N+M): callers abandon after an exponential patience time. Solved here
  numerically as a birth-death chain; the service level comes from the waiting-time
  distribution of a caller who finds k people ahead (a small continuous-time Markov chain,
  solved with a matrix exponential). With infinite patience it equals Erlang C (tested), and
  it matches a discrete-event simulation (tested).

Common assumptions: Poisson arrivals within the interval, exponential handle times, one
pooled skill group. Patience is estimated from the data (``estimate_patience``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.linalg import expm


def erlang_c(traffic: float, agents: int) -> float:
    """Probability that a call has to wait, for ``traffic`` Erlangs and ``agents`` agents."""
    if traffic <= 0:
        return 0.0
    if agents <= traffic:
        return 1.0
    term, total = 1.0, 1.0  # running A^k / k! and its sum for k < N
    for k in range(1, agents):
        term *= traffic / k
        total += term
    top = term * traffic / agents * agents / (agents - traffic)  # A^N / N! * N / (N - A)
    return top / (total + top)


def service_level(traffic: float, agents: int, aht: float, target_sec: float) -> float:
    if traffic <= 0:
        return 1.0
    if agents <= traffic:
        return 0.0
    return 1 - erlang_c(traffic, agents) * math.exp(-(agents - traffic) * target_sec / aht)


def asa(traffic: float, agents: int, aht: float) -> float:
    if agents <= traffic:
        return float("inf")
    return erlang_c(traffic, agents) * aht / (agents - traffic)


def required_agents(
    calls: float,
    aht: float,
    interval_sec: int = 1800,
    target_sl: float = 0.80,
    target_sec: float = 20,
    max_occupancy: float = 0.90,
) -> dict:
    """Smallest agent count meeting the service level and occupancy limits."""
    if calls <= 0 or aht <= 0:
        return {"traffic": 0.0, "agents": 0, "service_level": 1.0, "asa_sec": 0.0, "occupancy": 0.0}
    traffic = calls * aht / interval_sec
    n = max(1, math.floor(traffic) + 1)
    while service_level(traffic, n, aht, target_sec) < target_sl or traffic / n > max_occupancy:
        n += 1
    return {
        "traffic": round(traffic, 2),
        "agents": n,
        "service_level": round(service_level(traffic, n, aht, target_sec), 4),
        "asa_sec": round(asa(traffic, n, aht), 1),
        "occupancy": round(traffic / n, 3),
    }


def service_level_curve(
    calls: float, aht: float, interval_sec: int = 1800, target_sec: float = 20
) -> pd.DataFrame:
    """Service level and ASA for a range of agent counts: the cost of each extra point."""
    traffic = calls * aht / interval_sec
    rows = []
    for n in range(math.floor(traffic) + 1, math.ceil(traffic * 1.6) + 3):
        rows.append(
            {
                "agents": n,
                "service_level": service_level(traffic, n, aht, target_sec),
                "asa_sec": asa(traffic, n, aht),
                "occupancy": traffic / n,
            }
        )
    return pd.DataFrame(rows)


def staffing_plan(
    intervals: pd.DataFrame, shrinkage: float = 0.30, patience: float | None = None, **kw
) -> pd.DataFrame:
    """Add required agents to an interval forecast.

    ``intervals`` needs ``calls_p50``, ``calls_p80`` and ``aht_sec``. Staffing on the 80th
    percentile buys protection against an under-forecast; the plan shows both. With a
    ``patience`` it also sizes the P80 plan with Erlang A and shows what the Erlang C plan
    would deliver once callers' abandonment is taken into account.
    """
    out = intervals.copy()
    for q in ("p50", "p80"):
        res = [
            required_agents(c, a, **kw)
            for c, a in zip(out[f"calls_{q}"], out["aht_sec"], strict=True)
        ]
        out[f"agents_{q}"] = [r["agents"] for r in res]
        out[f"sl_{q}"] = [r["service_level"] for r in res]
        out[f"occupancy_{q}"] = [r["occupancy"] for r in res]
    # agents on the phone / (1 - shrinkage) = agents to schedule (breaks, training, leave)
    out["scheduled_p50"] = (out["agents_p50"] / (1 - shrinkage)).apply(math.ceil)
    out["scheduled_p80"] = (out["agents_p80"] / (1 - shrinkage)).apply(math.ceil)
    if patience:
        ea_kw = {k: v for k, v in kw.items() if k in ("interval_sec", "target_sl", "target_sec",
                                                      "max_occupancy", "max_abandon")}  # fmt: skip
        ea = [
            required_agents_erlang_a(c, a, patience, start=max(int(n), 1), **ea_kw)
            for c, a, n in zip(out["calls_p80"], out["aht_sec"], out["agents_p80"], strict=True)
        ]
        out["agents_p80_erlang_a"] = [r["agents"] for r in ea]
        out["sl_p80_erlang_a"] = [r["service_level"] for r in ea]
        out["abandon_p80_erlang_a"] = [r["p_abandon"] for r in ea]
        # the Erlang C plan, evaluated with abandonment: it over-delivers
        target_sec = kw.get("target_sec", 20)
        out["sl_p80_erlang_c_plan_true"] = [
            erlang_a(c, a, patience, int(n), target_sec=target_sec)["service_level"] if n else 1.0
            for c, a, n in zip(out["calls_p80"], out["aht_sec"], out["agents_p80"], strict=True)
        ]
        out["scheduled_p80_erlang_a"] = (out["agents_p80_erlang_a"] / (1 - shrinkage)).apply(
            math.ceil
        )
    return out


# ---------------------------------------------------------------------------- Erlang A
def _erlang_a_distribution(lam: float, mu: float, gamma: float, agents: int,
                           tol: float = 1e-12) -> np.ndarray:  # fmt: skip
    """Stationary distribution of the number of callers in the system (truncated)."""
    logp = [0.0]
    n = 0
    while True:
        out_rate = min(n + 1, agents) * mu + max(n + 1 - agents, 0) * gamma
        logp.append(logp[-1] + math.log(lam) - math.log(out_rate))
        n += 1
        if n > agents and logp[-1] < max(logp) + math.log(tol):
            break
        if n > agents + 20_000:  # pragma: no cover - only with absurd inputs
            break
    lp = np.array(logp)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def erlang_a(calls: float, aht: float, patience: float, agents: int,
             interval_sec: int = 1800, target_sec: float = 20) -> dict:  # fmt: skip
    """Service level, abandonment and occupancy for ``agents`` agents under Erlang A.

    ``service_level`` is the share of all arriving callers answered within ``target_sec``.
    """
    if calls <= 0:
        return {"service_level": 1.0, "p_wait": 0.0, "p_abandon": 0.0, "occupancy": 0.0}
    lam, mu, gamma = calls / interval_sec, 1 / aht, 1 / patience
    p = _erlang_a_distribution(lam, mu, gamma, agents)
    queue = p[agents:]  # arriving callers who find k = 0, 1, ... people waiting ahead
    k = np.arange(len(queue))
    p_abandon = gamma * float((k * queue).sum()) / lam
    # tagged caller with m people ahead: m -> m-1 at N*mu + m*gamma, abandons at gamma,
    # is answered from m = 0 at rate N*mu. States 0..K, then answered, then abandoned.
    K = len(queue) - 1
    q = np.zeros((K + 3, K + 3))
    served, gone = K + 1, K + 2
    for m in range(K + 1):
        q[m, gone] = gamma
        if m == 0:
            q[m, served] = agents * mu
        else:
            q[m, m - 1] = agents * mu + m * gamma
        q[m, m] = -q[m].sum()
    answered_by_t = expm(q * target_sec)[: K + 1, served]
    sl = float(p[:agents].sum() + (queue * answered_by_t).sum())
    return {
        "service_level": sl,
        "p_wait": float(queue.sum()),
        "p_abandon": p_abandon,
        "occupancy": lam * (1 - p_abandon) / (agents * mu),
    }


def required_agents_erlang_a(calls: float, aht: float, patience: float,
                             interval_sec: int = 1800, target_sl: float = 0.80,
                             target_sec: float = 20, max_occupancy: float = 0.90,
                             max_abandon: float = 0.05,
                             start: int | None = None) -> dict:  # fmt: skip
    """Smallest agent count meeting service level, abandonment and occupancy limits (Erlang A).

    The abandonment limit matters: with impatient callers a service level target alone can
    be met by letting many of them hang up.

    ``start`` (e.g. the Erlang C answer) is an upper bound to search down from: abandonment
    only ever lowers the number of agents needed.
    """
    if calls <= 0 or aht <= 0:
        return {"agents": 0, "service_level": 1.0, "p_abandon": 0.0, "occupancy": 0.0}

    def ok(n):
        r = erlang_a(calls, aht, patience, n, interval_sec, target_sec)
        good = (r["service_level"] >= target_sl and r["occupancy"] <= max_occupancy
                and r["p_abandon"] <= max_abandon)  # fmt: skip
        return good, r

    n = start or max(1, math.ceil(calls * aht / interval_sec) + 1)
    good, res = ok(n)
    while not good:  # the start was too low: go up
        n += 1
        good, res = ok(n)
    while n > 1:  # go down while the target still holds
        good_lower, r_lower = ok(n - 1)
        if not good_lower:
            break
        n, res = n - 1, r_lower
    return {"agents": n, **{k: round(v, 4) for k, v in res.items()}}


def estimate_patience(con, weeks: int = 8, grace_sec: float = 5.0) -> dict:
    """Mean caller patience from the data, for the exponential patience of Erlang A.

    Maximum likelihood with censoring: total time callers were exposed to the risk of
    hanging up, divided by the number who did. An answered call says the caller would have
    waited at least that long; an abandoned call says exactly how long.

    Only waiting beyond ``grace_sec`` counts, and only abandonments after it: hang-ups in the
    first seconds are wrong numbers and IVR slips, not impatience (they are also left out of
    the KPIs), and counting the many short waits as exposure would inflate the estimate.
    """
    row = con.execute(f"""
        SELECT sum(greatest(wait_sec - {grace_sec}, 0)) AS exposure,
               count(*) FILTER (WHERE NOT is_answered AND wait_sec > {grace_sec}) AS abandons,
               count(*) AS calls
        FROM fact_calls f JOIN dim_date d ON d.date = f.call_date
        WHERE NOT d.is_incomplete_load AND f.time_order_ok
          AND f.call_date > (SELECT max(call_date) FROM fact_calls) - {7 * weeks}
    """).fetchall()[0]
    exposure, abandons, calls = row
    return {"patience_sec": float(exposure) / max(int(abandons), 1), "abandons": int(abandons),
            "calls": int(calls), "weeks": weeks, "grace_sec": grace_sec}  # fmt: skip
