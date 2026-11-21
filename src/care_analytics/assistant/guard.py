"""SQL guard: the model proposes SQL, this code decides whether it runs.

The LLM is never trusted to follow its instructions. Before a query runs it must be:

* exactly one read-only statement (SELECT / WITH ... SELECT / set operations)
* over allow-listed analytics tables only; ``dim_customer`` with names, phones and e-mails is
  not on the list, the model sees the ``v_customer`` view without them
* free of personal-data columns and of table functions that read files or URLs
* bounded: a LIMIT is added when missing

Then it runs on a read-only connection with external access disabled and a time limit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "dim_date", "dim_vehicle", "dim_service_point", "v_customer",
    "fact_calls", "fact_service_orders", "fact_tickets", "fact_surveys",
    "v_kpi_daily", "v_kpi_interval",
}  # fmt: skip
BLOCKED_COLUMNS = {"full_name", "phone", "phone_e164", "phone10", "email", "email_key", "name_key"}
MAX_ROWS = 1000
TIMEOUT_SEC = 20


class GuardError(ValueError):
    """The query was refused before it ran."""


class SqlSyntaxError(GuardError):
    """The SQL could not be parsed: a mistake the model may fix, not a policy violation."""


@dataclass
class CheckedQuery:
    sql: str
    tables: list[str]


def check_sql(sql: str, max_rows: int = MAX_ROWS) -> CheckedQuery:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise GuardError("empty query")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise SqlSyntaxError(f"could not parse the SQL: {str(e).splitlines()[0]}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise GuardError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise GuardError(f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create
                      | exp.Alter | exp.Command | exp.Merge | exp.Copy | exp.Attach):  # fmt: skip
            raise GuardError(f"{node.key.upper()} is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):  # read_csv('...'), glob(...), a URL ...
            raise GuardError("table functions are not allowed")
        name = t.name.lower()
        if name in cte_names:
            continue
        if t.args.get("db") or t.args.get("catalog"):
            raise GuardError(f"qualified table names are not allowed: {t.sql()}")
        if name not in ALLOWED_TABLES:
            raise GuardError(f"table `{name}` is not available")
        tables.append(name)
    for f in tree.find_all(exp.Anonymous):
        if f.name.lower() in {"read_csv", "read_csv_auto", "read_parquet", "read_json", "glob",
                              "read_text", "read_blob", "getenv", "sniff_csv"}:  # fmt: skip
            raise GuardError(f"function `{f.name}` is not allowed")
    for c in tree.find_all(exp.Column):
        if c.name.lower() in BLOCKED_COLUMNS:
            raise GuardError(f"column `{c.name}` holds personal data and is not available")

    if not tree.args.get("limit"):
        tree = tree.limit(max_rows)
    return CheckedQuery(sql=tree.sql(dialect="duckdb"), tables=sorted(set(tables)))


def connect_read_only(warehouse_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(warehouse_path, read_only=True)
    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")  # the query itself cannot switch it back on
    except duckdb.InvalidInputException:
        # connections to the same file in one process share a database instance; it may
        # already be locked down by an earlier connection
        if con.execute("SELECT current_setting('enable_external_access')").fetchall()[0][0]:
            con.close()
            raise
    return con


def run_query(con, sql: str, timeout_sec: float = TIMEOUT_SEC) -> pd.DataFrame:
    """Run a checked query; interrupt it if it takes longer than ``timeout_sec``."""
    timer = threading.Timer(timeout_sec, con.interrupt)
    timer.start()
    try:
        return con.execute(sql).df()
    except duckdb.InterruptException as e:
        raise TimeoutError(f"query stopped after {timeout_sec:.0f} s") from e
    finally:
        timer.cancel()
