"""Erlang C staffing: how many agents an interval needs to reach a service level target.

Assumptions (state them when presenting the numbers):
* calls arrive as a Poisson process within the interval, handle times are exponential
* callers never hang up (Erlang C ignores abandonment), so it slightly over-staffs;
  Erlang A or a simulation would be the next step
* all agents can take all queues (one pooled skill group)
"""

from __future__ import annotations

import math

import pandas as pd


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


def staffing_plan(intervals: pd.DataFrame, shrinkage: float = 0.30, **kw) -> pd.DataFrame:
    """Add required agents to an interval forecast.

    ``intervals`` needs ``calls_p50``, ``calls_p80`` and ``aht_sec``. Staffing on the 80th
    percentile buys protection against an under-forecast; the plan shows both.
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
    return out
