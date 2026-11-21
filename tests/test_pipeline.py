"""End-to-end checks on a small generated dataset: the pipeline finds what was injected."""

import json

import pandas as pd

from care_analytics.anomaly import run_anomaly
from care_analytics.churn import build_features, run_churn, training_cohort
from care_analytics.forecast import run_forecast
from care_analytics.quality import PHONE_IN_TEXT_RE
from care_analytics.segmentation import run_segmentation


def _one(con, sql):
    return con.execute(sql).fetchall()[0][0]


def test_timezone_shift_detected_and_fixed(paths, con):
    dq = json.loads((paths.outputs / "data_quality.json").read_text())
    assert dq["timezone_shift_hours"] == {"pbx_v1": 3, "pbx_v2": 0}
    assert (
        _one(
            con,
            "SELECT count(*) FROM stg_calls WHERE hour(arrived_at) < 8 OR hour(arrived_at) >= 20",
        )
        == 0
    )


def test_incomplete_loads_detected(paths):
    dq = json.loads((paths.outputs / "data_quality.json").read_text())
    found = {d["date"]: d["status"] for d in dq["incomplete_days"]}
    truth = json.loads((paths.truth / "injected.json").read_text())["calls"]
    assert found.get(truth["missing_day"]) == "missing"
    assert found.get(truth["partial_day"]) == "partial"
    assert len(found) <= 4  # holidays must not be flagged as broken loads


def test_entity_resolution_quality(paths):
    er = json.loads((paths.outputs / "data_quality.json").read_text())["entity_resolution"]
    assert er["precision"] >= 0.98
    assert er["recall"] >= 0.90


def test_duplicates_removed_and_grain(con):
    assert _one(con, "SELECT count(*) - count(DISTINCT call_id) FROM fact_calls") == 0
    assert _one(con, "SELECT count(*) - count(DISTINCT customer_id) FROM dim_customer") == 0
    assert _one(con, "SELECT count(*) - count(DISTINCT vin) FROM dim_vehicle") == 0


def test_kpi_view_has_no_fan_out(con):
    # summing the daily KPI view must give back exactly the number of calls
    assert _one(con, "SELECT sum(offered) FROM v_kpi_daily") == _one(
        con, "SELECT count(*) FROM fact_calls"
    )
    bad = _one(
        con,
        "SELECT count(*) FROM v_kpi_daily WHERE abandon_rate NOT BETWEEN 0 AND 1 "
        "OR service_level_20s NOT BETWEEN 0 AND 1",
    )
    assert bad == 0


def test_pii_masked(con):
    leaks = _one(
        con,
        f"SELECT count(*) FROM fact_tickets WHERE regexp_matches(description_masked, "
        f"'{PHONE_IN_TEXT_RE}')",
    )
    assert leaks == 0
    assert (
        _one(con, "SELECT count(*) FROM fact_tickets WHERE description_masked LIKE '%[TELEFON]%'")
        > 0
    )


def test_legitimate_adjustments_kept(con):
    kept = _one(
        con,
        "SELECT count(*) FROM fact_service_orders WHERE amount_status = 'adjustment' "
        "AND amount_try < 0",
    )
    errors = _one(
        con,
        "SELECT count(*) FROM fact_service_orders WHERE amount_status = 'negative_error' "
        "AND amount_try IS NOT NULL",
    )
    assert kept > 0 and errors == 0


def test_exports_exclude_personal_data(paths):
    cust = pd.read_csv(paths.exports / "dim_customer.csv")
    assert not {"full_name", "phone_e164", "email"} & set(cust.columns)


def test_churn_features_respect_cutoff(con):
    cohort = training_cohort(con).head(200)
    feats = build_features(con, cohort)
    late = pd.to_datetime(feats["latest_input_date"]) >= pd.to_datetime(feats["cutoff"])
    assert not late.any()


def test_downstream_steps_run(paths):
    fc = run_forecast(paths, n_folds=2)
    assert fc["plan_days"] == 28 and fc["peak_agents_p80"] >= fc["peak_agents_p50"] > 0
    seg = run_segmentation(paths)
    assert seg["customers"] > 0
    ch = run_churn(paths)
    assert 0 < ch["churn_rate_train"] < 1
    an = run_anomaly(paths)
    assert an["series"] > 0
    for f in (
        "data_quality.md",
        "forecast_staffing.md",
        "segmentation.md",
        "churn.md",
        "complaint_alerts.md",
    ):
        assert (paths.reports / f).exists()
