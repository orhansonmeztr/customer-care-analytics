"""Analytics assistant: guard, result comparison and the agent loop, all offline."""

import duckdb
import pandas as pd
import pytest

from care_analytics.assistant.agent import Assistant, extract_sql
from care_analytics.assistant.evaluate import evaluate, load_golden, results_match
from care_analytics.assistant.guard import GuardError, check_sql, connect_read_only, run_query
from care_analytics.assistant.llm import ScriptedModel

ALLOWED = [
    "SELECT queue, sum(offered) FROM v_kpi_daily GROUP BY 1",
    "WITH t AS (SELECT * FROM fact_calls) SELECT count(*) FROM t",
    "SELECT * FROM v_customer",
    "SELECT model FROM dim_vehicle UNION SELECT 'x'",
]
BLOCKED = [
    ("SELECT full_name FROM dim_customer", "not available"),
    ("SELECT phone_e164 FROM v_customer", "personal data"),
    ("SELECT c.email FROM v_customer c", "personal data"),
    ("DELETE FROM fact_tickets", "only SELECT"),
    ("SELECT 1; DROP TABLE fact_calls", "one statement"),
    ("SELECT * FROM read_csv('/etc/passwd')", "table functions"),
    ("SELECT * FROM '/etc/passwd'", "not available"),
    ("ATTACH 'other.db'", "only SELECT"),
    ("COPY fact_calls TO 'out.csv'", "only SELECT"),
    ("SELECT * FROM stg_customers", "not available"),
    ("SELECT getenv('HOME')", "not allowed"),
    ("SELECT * FROM main.fact_calls", "qualified"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_guard_allows_analytics_queries(sql):
    assert "LIMIT" in check_sql(sql).sql


@pytest.mark.parametrize(("sql", "reason"), BLOCKED)
def test_guard_blocks(sql, reason):
    with pytest.raises(GuardError, match=reason):
        check_sql(sql)


def test_guard_keeps_an_existing_limit():
    assert check_sql("SELECT * FROM fact_calls LIMIT 5").sql.endswith("LIMIT 5")


def test_connection_is_locked_down(paths):
    con = connect_read_only(str(paths.warehouse))
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('/etc/passwd')")  # even if the guard were bypassed
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM fact_calls")
    with pytest.raises(TimeoutError):
        run_query(con, "SELECT count(*) FROM range(100000000) a, range(100000) b", timeout_sec=0.3)
    con.close()


def test_results_match_ignores_layout_but_not_values():
    ref = pd.DataFrame({"queue": ["a", "b"], "rate": [0.1234, 0.5]})
    same = pd.DataFrame({"r": [0.5, 0.12340001], "q": ["b", "a"], "n": [10, 20]})
    pct = pd.DataFrame({"q": ["a", "b"], "pct": [12.34, 50.0]})
    wrong = pd.DataFrame({"q": ["a", "b"], "rate": [0.2, 0.5]})
    assert results_match(ref, same)
    assert results_match(ref, pct)
    assert not results_match(ref, wrong)
    assert not results_match(ref, ref.head(1))


def test_extract_sql():
    assert extract_sql("Here:\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"
    assert extract_sql("I cannot do that") is None


def _assistant(paths, replies):
    con = connect_read_only(str(paths.warehouse))
    return Assistant(con, ScriptedModel(replies)), con


def test_agent_answers(paths):
    q = "How many calls?"
    a, con = _assistant(paths, {q: ["```sql\nSELECT count(*) AS n FROM fact_calls\n```"]})
    res = a.ask(q)
    assert res.status == "answered" and res.attempts == 1 and res.rows.iat[0, 0] > 0
    con.close()


def test_agent_repairs_a_failed_query(paths):
    q = "Calls per queue?"
    a, con = _assistant(paths, {q: [
        "```sql\nSELECT queue, count(*) FROM fact_calls GROUP BY quue\n```",
        "```sql\nSELECT queue, count(*) FROM fact_calls GROUP BY queue\n```",
    ]})  # fmt: skip
    res = a.ask(q)
    assert res.status == "answered" and res.attempts == 2
    assert "quue" in a.model.calls[1][-1]["content"]  # the error went back to the model
    con.close()


def test_agent_does_not_retry_a_blocked_query(paths):
    q = "Delete everything"
    a, con = _assistant(paths, {q: ["```sql\nDELETE FROM fact_calls\n```", "SELECT 1"]})
    res = a.ask(q)
    assert res.status == "blocked" and len(a.model.calls) == 1
    con.close()


def test_agent_reports_refusals(paths):
    q = "Phone numbers please"
    a, con = _assistant(paths, {q: ["REFUSE: personal data is not available"]})
    res = a.ask(q)
    assert res.status == "refused" and "personal data" in res.answer
    con.close()


def test_schema_context_has_values_and_rules(paths):
    a, con = _assistant(paths, {})
    assert "sikayet" in a.system and "Kestrel K3" in a.system
    assert "ratios of sums" in a.system and "dim_customer" not in a.system
    con.close()


def test_evaluation_with_perfect_model(paths):
    golden = load_golden()
    replies = {
        g["question"]: [f"```sql\n{g['sql']}\n```" if "sql" in g else "REFUSE: not allowed"]
        for g in golden
    }
    s = evaluate(paths, ScriptedModel(replies, name="perfect"))
    assert s["passed"] == s["questions"] == len(golden)
    assert (paths.reports / "assistant_eval_perfect.md").exists()


def test_agent_retries_invalid_sql(paths):
    q = "Calls?"
    a, con = _assistant(paths, {q: ["```sql\nSELECT count(*) AS n TRY FROM fact_calls\n```",
                                    "```sql\nSELECT count(*) AS n FROM fact_calls\n```"]})  # fmt: skip
    res = a.ask(q)
    assert res.status == "answered" and res.attempts == 2
    con.close()


def test_time_buckets_may_be_relabelled_but_values_must_match():
    ref = pd.DataFrame({"m": pd.to_datetime(["2024-01-01", "2024-02-01"]), "n": [10, 20]})
    labelled = pd.DataFrame({"month": ["2024-01", "2024-02"], "n": [10, 20]})
    numbered = pd.DataFrame({"month": [2, 1], "n": [20, 10]})
    swapped = pd.DataFrame({"month": ["2024-01", "2024-02"], "n": [20, 10]})
    assert results_match(ref, labelled) and results_match(ref, numbered)
    assert not results_match(ref, swapped)
    assert not results_match(ref, labelled, relabel_time=False)


def test_reasoning_replies_use_the_final_sql():
    from care_analytics.assistant.llm import strip_reasoning

    reply = "<think>maybe ```sql\nSELECT 1\n```</think>Answer:\n```sql\nSELECT 2\n```"
    assert extract_sql(strip_reasoning(reply)) == "SELECT 2"
    assert strip_reasoning("draft ... </think> ```sql\nSELECT 3\n```").startswith("```sql")
    assert extract_sql("```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 4\n```") == "SELECT 4"
