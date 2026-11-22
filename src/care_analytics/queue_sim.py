"""Discrete-event simulation of a staffed contact center.

The synthetic PBX does not record how many agents were logged in, so the staffing models
cannot be checked against it directly. This simulator produces "observed" intervals with a
known number of agents, and it deliberately breaks the assumptions Erlang C and Erlang A
make, so that the check is a real one:

* arrivals are not stationary within a 30-minute interval: the rate moves linearly from one
  interval's level to the next
* handle times are lognormal with the spread measured in the data (Erlang: exponential)
* callers never hang up in the first seconds, then lose patience exponentially
* fewer agents may log in than were planned (absenteeism), and agents leaving at the end of
  an interval finish the call they are on

One FCFS queue served by one pooled group of agents (skills come later).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

INTERVAL_SEC = 1800


@dataclass(frozen=True)
class CenterConfig:
    handle_log_mean: float  # lognormal handle time (seconds), from the data
    handle_log_sd: float
    patience_sec: float  # mean of the exponential part of patience
    patience_grace_sec: float = 4.0  # nobody hangs up before this
    absenteeism_mean: float = 0.03  # share of planned agents who do not log in, on average
    absenteeism_sd: float = 0.06
    day_noise_sd: float = 0.05  # day-to-day volume noise on top of Poisson
    # switches back to Erlang A's assumptions, to test the simulator itself
    exponential_handle: bool = False
    stationary: bool = False


def _arrivals(rng, rates: np.ndarray, day_start: float, stationary: bool = False) -> np.ndarray:
    """Non-stationary Poisson arrivals: piecewise-linear rate between interval midpoints."""
    n_int = len(rates)
    if stationary:  # constant rate within each interval
        counts = rng.poisson(rates)
        starts = np.repeat(np.arange(n_int) * INTERVAL_SEC, counts)
        return day_start + np.sort(starts + rng.uniform(0, INTERVAL_SEC, counts.sum()))
    peak = rates.max() / INTERVAL_SEC * 1.6 + 1e-9
    horizon = n_int * INTERVAL_SEC
    n = rng.poisson(peak * horizon)
    t = np.sort(rng.uniform(0, horizon, n))
    mids = (np.arange(n_int) + 0.5) * INTERVAL_SEC
    lam_t = np.interp(t, mids, rates / INTERVAL_SEC)
    keep = rng.uniform(0, peak, n) < lam_t  # thinning
    return day_start + t[keep]


def simulate_day(rng, cfg: CenterConfig, expected_calls: np.ndarray, agents_planned: np.ndarray,
                 day_start: float = 0.0, volume_factor: float | None = None) -> pd.DataFrame:  # fmt: skip
    """Simulate one day; returns one row per call.

    ``expected_calls`` and ``agents_planned`` are per 30-minute interval of the opening hours.
    """
    if volume_factor is None:
        volume_factor = float(np.exp(rng.normal(0, cfg.day_noise_sd)))
    arrivals = _arrivals(rng, expected_calls * volume_factor, day_start, cfg.stationary)
    absent = np.clip(
        rng.normal(cfg.absenteeism_mean, cfg.absenteeism_sd, len(agents_planned)), -0.05, 0.4
    )
    agents = np.maximum(np.round(agents_planned * (1 - absent)), 0).astype(int)
    if cfg.exponential_handle:
        handle = rng.exponential(np.exp(cfg.handle_log_mean), len(arrivals))
    else:
        handle = rng.lognormal(cfg.handle_log_mean, cfg.handle_log_sd, len(arrivals))
    patience = cfg.patience_grace_sec + rng.exponential(cfg.patience_sec, len(arrivals))

    free: list[float] = []  # min-heap of the times agents become free
    current = 0
    rows = []
    interval_of = ((arrivals - day_start) // INTERVAL_SEC).astype(int)
    next_boundary = 0
    for i, t in enumerate(arrivals):
        # apply staffing changes for every interval boundary passed
        while next_boundary <= interval_of[i]:
            target = agents[next_boundary]
            start_of = day_start + next_boundary * INTERVAL_SEC
            if target > current:
                for _ in range(target - current):
                    heapq.heappush(free, start_of)
            elif target < current:
                # agents who log off finish their call: remove the ones busy longest ahead
                free = sorted(free)[:target]
                heapq.heapify(free)
            current = target
            next_boundary += 1
        if current == 0:
            rows.append((interval_of[i], t, np.inf, False, handle[i]))
            continue
        start = max(t, free[0])
        if start - t <= patience[i]:
            heapq.heapreplace(free, start + handle[i])
            rows.append((interval_of[i], t, start - t, True, handle[i]))
        else:
            rows.append((interval_of[i], t, patience[i], False, handle[i]))
    calls = pd.DataFrame(
        rows, columns=["interval", "arrived", "wait_sec", "answered", "handle_sec"]
    )
    calls["agents"] = agents[calls["interval"].to_numpy()]
    return calls


def interval_stats(
    calls: pd.DataFrame, target_sec: float = 20, short_sec: float = 5
) -> pd.DataFrame:
    """Per-interval observations, with the same KPI definitions as the warehouse."""
    c = calls.assign(
        in_target=calls["answered"] & (calls["wait_sec"] <= target_sec),
        short=~calls["answered"] & (calls["wait_sec"] < short_sec),
        abandoned=~calls["answered"] & (calls["wait_sec"] >= short_sec),
    )
    g = c.groupby(["day", "interval"])
    out = pd.DataFrame({
        "offered": g.size(), "agents": g["agents"].first(),
        "in_target": g["in_target"].sum(), "abandoned": g["abandoned"].sum(),
        "short": g["short"].sum(),
        "aht_sec": g.apply(lambda x: x.loc[x["answered"], "handle_sec"].mean(), include_groups=False),
    }).reset_index()  # fmt: skip
    out["service_level"] = out["in_target"] / out["offered"]  # share of all arrivals (model def.)
    out["abandon_rate"] = out["abandoned"] / out["offered"]
    return out


def estimate_patience_from_calls(calls: pd.DataFrame, grace_sec: float = 5.0) -> float:
    """Same estimator as ``staffing.estimate_patience``, on simulated call records."""
    w = calls["wait_sec"].replace(np.inf, np.nan).dropna()
    exposure = np.maximum(w - grace_sec, 0).sum()
    abandons = (~calls.loc[w.index, "answered"] & (w > grace_sec)).sum()
    return float(exposure / max(abandons, 1))
