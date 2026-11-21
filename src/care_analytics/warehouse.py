"""Build the star schema and export extracts for Tableau / Power BI."""

from __future__ import annotations

from importlib import resources

import pandas as pd

from .calendar_tr import holiday_calendar
from .paths import Paths
from .quality import connect

DAY_NAMES_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

EXPORTS = {
    "dim_customer": "SELECT customer_id, city, region, nearest_service_km, created_at, "
    "n_source_records FROM dim_customer",  # no names, phones or e-mails
    "dim_vehicle": "SELECT * FROM dim_vehicle",
    "dim_service_point": "SELECT * FROM dim_service_point",
    "dim_date": "SELECT * FROM dim_date",
    "kpi_daily": "SELECT * FROM v_kpi_daily ORDER BY call_date, queue",
    "kpi_interval": "SELECT * FROM v_kpi_interval ORDER BY interval_start",
    "fact_tickets": "SELECT ticket_id, opened_date, status, channel, vin, service_code, category, "
    "fault_code, resolution_days FROM fact_tickets",
    "fact_service_orders": "SELECT * FROM fact_service_orders",
}


def _run_script(con, sql: str) -> None:
    # drop full-line comments first so a ';' inside a comment cannot split a statement
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    for stmt in (s.strip() for s in body.split(";")):
        if stmt:
            con.execute(stmt)


def build_dim_date(con, start: str, end: str) -> None:
    cal = holiday_calendar(start, end)
    cal["date"] = cal["date"].dt.date
    d = pd.to_datetime(cal["date"])
    cal["year"], cal["month"], cal["iso_week"] = d.dt.year, d.dt.month, d.dt.isocalendar().week
    cal["day_of_week"] = d.dt.day_of_week + 1  # 1 = Monday
    cal["day_name_tr"] = [DAY_NAMES_TR[i] for i in d.dt.day_of_week]
    cal["is_weekend"] = d.dt.day_of_week >= 5
    con.register("cal_df", cal)
    con.execute("""
    CREATE OR REPLACE TABLE dim_date AS
    SELECT c.*, coalesce(l.load_status <> 'ok', false) AS is_incomplete_load
    FROM cal_df c LEFT JOIN dq_daily_load l ON l.call_date = c.date
    """)


def build_warehouse(paths: Paths) -> dict:
    con = connect(paths)
    start, end = con.execute("SELECT min(call_date), max(call_date) FROM stg_calls").fetchall()[0]
    build_dim_date(con, str(start), str(end))
    _run_script(con, resources.files("care_analytics.sql").joinpath("warehouse.sql").read_text())
    counts = {}
    for name, sql in EXPORTS.items():
        df = con.execute(sql).df()
        df.to_csv(paths.exports / f"{name}.csv", index=False)
        counts[name] = len(df)
    con.close()
    return counts
