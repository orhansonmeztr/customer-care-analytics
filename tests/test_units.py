"""Unit tests that need no generated data."""

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from care_analytics.anomaly import bh_adjust, detect
from care_analytics.calendar_tr import holiday_calendar
from care_analytics.churn import assert_no_leakage, contact_policy, lift_at
from care_analytics.forecast import bias, make_features, wape
from care_analytics.staffing import erlang_c, required_agents, service_level, staffing_plan


def test_religious_holidays_move_and_have_eves():
    cal = holiday_calendar("2024-01-01", "2025-12-31").set_index("date")
    assert cal.loc["2024-04-10", "holiday_kind"] == "religious"
    assert cal.loc["2025-03-30", "holiday_kind"] == "religious"
    assert bool(cal.loc["2024-04-09", "is_arife"])
    assert bool(cal.loc["2024-06-20", "is_post_holiday"])  # Kurban 2024: 16-19 June
    assert cal.loc["2024-10-29", "holiday_kind"] == "national"


def test_erlang_c_matches_mm1():
    # with one agent Erlang C reduces to M/M/1, where P(wait) = utilisation
    assert erlang_c(0.5, 1) == pytest.approx(0.5)
    assert erlang_c(0.0, 3) == 0.0
    assert erlang_c(5, 5) == 1.0  # unstable queue


def test_service_level_increases_with_agents():
    levels = [service_level(40, n, 240, 20) for n in range(41, 60)]
    assert all(b > a for a, b in zip(levels, levels[1:], strict=False))


def test_required_agents_textbook_case():
    # 300 calls in 30 minutes, 4-minute handle time = 40 Erlangs
    r = required_agents(300, 240, target_sl=0.80, target_sec=20)
    assert r["traffic"] == 40
    assert r["agents"] == 46
    assert r["service_level"] >= 0.80
    assert required_agents(0, 240)["agents"] == 0


def test_staffing_plan_applies_shrinkage():
    plan = staffing_plan(
        pd.DataFrame({"calls_p50": [100.0], "calls_p80": [120.0], "aht_sec": [300.0]}),
        shrinkage=0.3,
    )
    assert plan["agents_p80"].iloc[0] >= plan["agents_p50"].iloc[0]
    assert plan["scheduled_p50"].iloc[0] == math.ceil(plan["agents_p50"].iloc[0] / 0.7)


def test_forecast_metrics():
    y = np.array([100.0, 200.0, np.nan])
    f = np.array([110.0, 180.0, 50.0])
    assert wape(y, f) == pytest.approx(30 / 300)
    assert bias(y, f) == pytest.approx(-10 / 300)


def test_lag_features_only_use_old_values():
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=500), "y": np.arange(500.0) + 100})
    f = make_features(df)
    row = f.iloc[-1]
    level = (np.arange(500.0) + 100)[-35 - 28 : -35].mean()
    assert row["level"] == pytest.approx(level)
    assert row["rel_lag_35"] == pytest.approx(np.log((500 - 35 - 1 + 100) / level))


def test_leakage_guard():
    ok = pd.DataFrame({"cutoff": ["2024-01-10"], "latest_input_date": ["2024-01-09"]})
    assert_no_leakage(ok)
    bad = pd.DataFrame({"cutoff": ["2024-01-10"], "latest_input_date": ["2024-01-10"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_lift_and_policy():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    assert lift_at(y, p, 0.2) == pytest.approx(5.0)
    pol = contact_policy(y, p)
    assert pol["churners_reached"].iloc[-1] == 2
    assert pol["recall"].max() == 1.0


def test_bh_matches_statsmodels():
    p = np.random.default_rng(0).uniform(size=50) ** 2
    assert np.allclose(bh_adjust(p), multipletests(p, method="fdr_bh")[1])


def test_detect_flags_a_spike_but_not_noise():
    rng = np.random.default_rng(1)
    weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    rows = []
    for series in range(20):
        n = rng.poisson(3, len(weeks))
        if series == 0:
            n[30] = 25
        rows += [
            {"model": f"M{series}", "fault_code": "X", "week": w, "n": c}
            for w, c in zip(weeks, n, strict=True)
        ]
    res = detect(pd.DataFrame(rows))
    alerts = res[res["alert"]]
    assert (alerts["model"] == "M0").any() and alerts["week"].min() == weeks[30]
    assert (alerts["model"] != "M0").sum() <= 1


# ---------------------------------------------------------------------------- Erlang A
def _simulate_mmn_m(lam, mu, gamma, n_agents, t_target, n_calls=60_000, seed=0):
    """Discrete-event simulation of M/M/N+M (FCFS, exponential patience)."""
    import heapq

    rng = np.random.default_rng(seed)
    free_at = [0.0] * n_agents  # time each agent becomes free
    heapq.heapify(free_at)
    answered_fast = abandoned = 0
    arrivals = np.cumsum(rng.exponential(1 / lam, n_calls))
    patience = rng.exponential(1 / gamma, n_calls)
    service = rng.exponential(1 / mu, n_calls)
    for i, t in enumerate(arrivals):
        # an FCFS queue with abandonment: the caller is answered when an agent frees up before
        # the caller's deadline; callers ahead who left do not hold anyone up
        start = max(t, free_at[0])
        if start - t <= patience[i]:
            heapq.heapreplace(free_at, start + service[i])
            answered_fast += start - t <= t_target
        else:
            abandoned += 1
    return answered_fast / n_calls, abandoned / n_calls


def test_erlang_a_equals_erlang_c_with_infinite_patience():
    from care_analytics.staffing import erlang_a

    for n in (42, 46, 50):
        assert erlang_a(300, 240, 1e12, n)["service_level"] == pytest.approx(
            service_level(40, n, 240, 20), abs=1e-6
        )


def test_erlang_a_matches_a_simulation():
    from care_analytics.staffing import erlang_a

    calls, aht, patience, agents = 90, 240, 60, 11  # an overloaded interval: 12 Erlangs
    sl_sim, ab_sim = _simulate_mmn_m(calls / 1800, 1 / aht, 1 / patience, agents, 20)
    r = erlang_a(calls, aht, patience, agents)
    assert r["service_level"] == pytest.approx(sl_sim, abs=0.02)
    assert r["p_abandon"] == pytest.approx(ab_sim, abs=0.01)


def test_impatient_callers_need_fewer_agents_but_respect_the_abandon_limit():
    from care_analytics.staffing import required_agents_erlang_a

    c = required_agents(152, 300)["agents"]
    a = required_agents_erlang_a(152, 300, 115, start=c)
    assert a["agents"] <= c
    assert a["service_level"] >= 0.80 and a["p_abandon"] <= 0.05
    loose = required_agents_erlang_a(152, 300, 115, start=c, max_abandon=1.0)
    assert loose["agents"] <= a["agents"]


# ---------------------------------------------------------------------------- conformal
def test_conformal_offset_reaches_the_target_coverage():
    from care_analytics.forecast import conformal_offset

    rng = np.random.default_rng(3)
    y = rng.lognormal(7, 0.2, 500)
    p80 = y.mean() * np.ones(500)  # a badly calibrated constant forecast
    shift = conformal_offset(y, p80)
    assert np.mean(y <= p80 * np.exp(shift)) == pytest.approx(0.80, abs=0.01)


def test_calibration_uses_only_earlier_folds():
    from care_analytics.forecast import calibrate_backtest

    rows = []
    for k, cut in enumerate(pd.date_range("2024-01-01", periods=5, freq="MS")):
        # later folds have huge errors: they must not influence earlier calibrations
        actual = 100.0 * (10 if k == 4 else 1)
        rows += [{"cutoff": cut, "y": actual * (1 + 0.01 * i), "lgbm_p80": 100.0, "lgbm": 90.0}
                 for i in range(20)]  # fmt: skip
    bt = calibrate_backtest(pd.DataFrame(rows), min_folds=2)
    assert bt.groupby("cutoff")["lgbm_p80_cal"].apply(lambda s: s.isna().all()).iloc[:2].all()
    assert bt.loc[bt["cutoff"] == bt["cutoff"].max(), "lgbm_p80_cal"].max() < 200


# ---------------------------------------------------------------------------- CUSUM
def test_cusum_threshold_meets_the_design_false_alarm_rate():
    from care_analytics.anomaly import cusum_threshold

    lam, arl0 = 1.0, 365
    h = cusum_threshold(lam, 1.5, arl0)
    rng = np.random.default_rng(42)  # an independent simulation
    s, alarms, days = np.zeros(2000), 0, 1000
    for _ in range(days):
        s = np.maximum(0, s + rng.poisson(lam, 2000) * np.log(1.5) - 0.5 * lam)
        hit = s > h
        alarms += hit.sum()
        s[hit] = 0
    assert alarms / (2000 * days) <= 1.25 / arl0  # at most the design rate, plus noise
    assert cusum_threshold(lam, 1.5, 10 * arl0) > h  # a rarer false alarm needs a higher bar


def _daily(n_days=500, drift_from=None, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=n_days)
    lam = np.full(n_days, 2.0)
    if drift_from is not None:
        lam[drift_from:] *= 1 + np.minimum(np.arange(n_days - drift_from) / 30, 1.0)  # to 2x
    return pd.DataFrame({"model": "M", "fault_code": "X", "day": days,
                         "n": rng.poisson(lam), "park": 1000})  # fmt: skip


def test_cusum_catches_a_drift_and_stays_quiet_without_one():
    from care_analytics.anomaly import cusum_detect

    quiet = cusum_detect(_daily(), arl0=3650)
    assert quiet["alarm"].sum() <= 1
    drift = cusum_detect(_daily(drift_from=350), arl0=3650)
    first = drift.loc[drift["alarm"], "day"].min()
    assert first >= pd.Timestamp("2024-01-01") + pd.Timedelta(days=350)
    assert (first - (pd.Timestamp("2024-01-01") + pd.Timedelta(days=350))).days <= 30


def test_weekday_factors_follow_the_weekly_pattern():
    from care_analytics.anomaly import weekday_factors

    d = _daily(n_days=400)
    d["n"] = np.where(d["day"].dt.day_of_week == 0, 4, 2)  # busy Mondays
    f = weekday_factors(d).dropna()
    mondays = f[f.index.day_of_week == 0]
    assert mondays.mean() == pytest.approx(4 / (4 + 6 * 2) * 7, rel=0.05)


# ---------------------------------------------------------------------------- simulator
def test_simulator_matches_erlang_a_when_its_assumptions_hold():
    from care_analytics.queue_sim import CenterConfig, interval_stats, simulate_day
    from care_analytics.staffing import erlang_a

    cfg = CenterConfig(handle_log_mean=np.log(240), handle_log_sd=0, patience_sec=60,
                       patience_grace_sec=0, absenteeism_mean=0, absenteeism_sd=0,
                       day_noise_sd=0, exponential_handle=True, stationary=True)  # fmt: skip
    rng = np.random.default_rng(5)
    n_int, calls, agents = 24, 100.0, 13
    days = [simulate_day(rng, cfg, np.full(n_int, calls), np.full(n_int, agents),
                         day_start=d * 86400.0).assign(day=d) for d in range(40)]  # fmt: skip
    st = interval_stats(pd.concat(days))
    st = st[st["interval"] > 0]  # the first interval starts with an empty queue
    sl = st["in_target"].sum() / st["offered"].sum()
    ab = (st["abandoned"] + st["short"]).sum() / st["offered"].sum()
    r = erlang_a(calls, 240, 60, agents)
    assert sl == pytest.approx(r["service_level"], abs=0.02)
    assert ab == pytest.approx(r["p_abandon"], abs=0.01)


def test_staffing_changes_between_intervals():
    from care_analytics.queue_sim import CenterConfig, simulate_day

    cfg = CenterConfig(np.log(200), 0.3, 120, absenteeism_mean=0, absenteeism_sd=0, day_noise_sd=0)
    calls = simulate_day(np.random.default_rng(1), cfg, np.array([60.0, 60.0, 60.0]),
                         np.array([8, 0, 8]))  # fmt: skip
    assert not calls.loc[calls["interval"] == 1, "answered"].any()  # nobody logged in
    assert calls.loc[calls["interval"] == 2, "answered"].mean() > 0.5


def _skills_cfg(**kw):
    from care_analytics.queue_sim import CenterConfig

    base = dict(handle_log_mean=0.0, handle_log_sd=0.0, patience_sec=60, patience_grace_sec=0,
                absenteeism_mean=0, absenteeism_sd=0, day_noise_sd=0)  # fmt: skip
    return CenterConfig(**(base | kw))


def test_skills_simulator_matches_erlang_a_for_one_queue():
    from care_analytics.queue_sim import interval_stats
    from care_analytics.skills_routing import Group, simulate_skills_day
    from care_analytics.staffing import erlang_a

    cfg = _skills_cfg(exponential_handle=True, stationary=True)
    rng = np.random.default_rng(5)
    n_int, calls, agents = 24, 100.0, 13
    days = [simulate_skills_day(rng, cfg, {"q": np.full(n_int, calls)}, {"q": (np.log(240), 0)},
                                {"g": np.full(n_int, agents)}, [Group("g", ("q",))],
                                day_start=d * 86400.0).assign(day=d, agents=agents)
            for d in range(40)]  # fmt: skip
    st = interval_stats(pd.concat(days))
    st = st[st["interval"] > 0]
    r = erlang_a(calls, 240, 60, agents)
    assert st["in_target"].sum() / st["offered"].sum() == pytest.approx(
        r["service_level"], abs=0.02
    )
    ab = (st["abandoned"] + st["short"]).sum() / st["offered"].sum()
    assert ab == pytest.approx(r["p_abandon"], abs=0.01)


def test_routing_prefers_specialists_and_generalists_cover_any_queue():
    from care_analytics.skills_routing import groups_for, simulate_skills_day

    cfg = _skills_cfg()
    exp = {"satis": np.array([40.0, 40.0]), "genel": np.array([40.0, 40.0])}
    handle = {"satis": (np.log(120), 0.3), "genel": (np.log(120), 0.3)}
    groups = groups_for(("satis", "genel"))
    # only sales specialists and generalists: general calls must all go to generalists
    calls = simulate_skills_day(np.random.default_rng(2), cfg, exp, handle,
                                {"satis": np.array([6, 6]), "generalist": np.array([6, 6])},
                                groups)  # fmt: skip
    ans = calls[calls["answered"]]
    assert set(ans.loc[ans["queue"] == "genel", "served_by"]) == {"generalist"}
    # with spare specialists, most sales calls are answered by them, not by generalists
    sales = ans[ans["queue"] == "satis"]
    assert (sales["served_by"] == "satis").mean() > 0.8


def test_flex_plan_keeps_headcount_and_split():
    from care_analytics.skills_routing import GENERALIST, flex_plan

    per_q = {"a": np.array([5, 0]), "b": np.array([3, 0])}
    pooled = np.array([7, 0])
    work = {"a": np.array([3.0, 0.0]), "b": np.array([1.0, 0.0])}
    ded = flex_plan(per_q, pooled, None, work)
    assert ded["a"].tolist() == [5, 0] and ded[GENERALIST].tolist() == [0, 0]
    half = flex_plan(per_q, pooled, 0.5, work, extra=2)
    assert half["a"][0] + half["b"][0] + half[GENERALIST][0] == 7 + 2
    assert half["a"][0] == 3 and half["b"][0] == 1  # 4 specialists split 3:1 by workload
    assert half[GENERALIST][1] == 0  # closed intervals get no extra agents
    pool = flex_plan(per_q, pooled, 1.0, work)
    assert pool["a"].sum() == 0 and pool[GENERALIST].tolist() == [7, 0]


def test_second_rise_after_a_spike_is_not_hidden_by_the_baseline():
    from care_analytics.anomaly import detect

    rng = np.random.default_rng(3)
    weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    rows = []
    for series in range(20):
        n = rng.poisson(3, len(weeks))
        if series == 0:
            n[20:22] = 40  # a spike ...
            n[27] = 13  # ... and, six weeks later, a second rise of the same fault
        rows += [
            {"model": f"M{series}", "fault_code": "X", "week": w, "n": c}
            for w, c in zip(weeks, n, strict=True)
        ]
    df = pd.DataFrame(rows)

    def flagged(res):
        return set(res.loc[res["alert"] & (res["model"] == "M0"), "week"])

    new, old = detect(df), detect(df, exclude_alerts=False)
    assert {weeks[20], weeks[21], weeks[27]} <= flagged(new)
    assert weeks[27] not in flagged(old)  # the spike inflated the old baseline
    # the baseline after the spike is the normal level, not the spike
    b = new.loc[(new["model"] == "M0") & (new["week"] == weeks[27]), "baseline"].iloc[0]
    assert b < 5
