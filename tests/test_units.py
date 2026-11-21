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
