"""What the model knows about the warehouse: tables, columns, values and KPI rules.

Text-to-SQL fails mostly on meaning, not syntax: the model does not know that the queue is
called ``sikayet``, that a rate must be ``sum / sum`` and not an average of daily rates, or
that the data ends in June 2025. The context states these things explicitly.
"""

from __future__ import annotations

from .guard import ALLOWED_TABLES

# Changes to the rules or examples change the results: bump the version and re-run the
# evaluation (docs/eval keeps the result of each version).
PROMPT_VERSION = "1.3"  # 1.2: states when repeat_7d is NULL; 1.3: KPI and pattern examples

TABLE_NOTES = {
    "v_kpi_daily": "one row per day AND queue (a day has one row per queue). Use it for KPIs. Rates over a period must be "
    "re-computed from the counts (e.g. sum(abandoned) / sum(offered - short_abandons)), "
    "never averaged. is_incomplete_load marks days with a broken data load.",
    "v_kpi_interval": "one row per 30-minute interval, all queues pooled",
    "fact_calls": "one row per call. handle_sec = talk + hold + wrap of answered calls. "
    "repeat_7d: the customer called again about the same reason within 7 days. It is NULL "
    "when it cannot be known (unanswered call, anonymous caller or missing reason code); a "
    "repeat-contact rate is computed over the calls where repeat_7d is not NULL, "
    "e.g. avg(repeat_7d::INT)",
    "fact_service_orders": "one row per authorized service order. amount_try in TL; "
    "order_type 'adjustment' rows are refunds (negative amounts)",
    "fact_tickets": "one row per complaint ticket; resolution_days is NULL while open",
    "fact_surveys": "one row per post-call survey: csat 1-5, nps 0-10",
    "dim_vehicle": "one row per car (VIN), with model, segment and warranty_end",
    "dim_service_point": "one row per authorized service point",
    "v_customer": "one row per customer, without personal data",
    "dim_date": "one row per day with Turkish holiday flags",
}

CATEGORY_COLUMNS = {
    "fact_calls": ["queue", "reason_code"],
    "fact_service_orders": ["order_type"],
    "fact_tickets": ["status", "channel", "category", "fault_code"],
    "dim_vehicle": ["brand", "model", "segment"],
    "dim_service_point": ["region"],
}

RULES = """\
Rules:
- Write one DuckDB SELECT query. No other statements.
- Data covers {start} to {end}. "This year" is {year}; "last year" is {prev_year}.
- Queue names are Turkish codes: satis (sales), servis (service), sikayet (complaints), genel (general).
- KPI definitions:
  service level = answered calls within 20 s / (offered - short abandons)
  abandon rate = abandoned / (offered - short abandons)
  AHT = average handle_sec of answered calls; NPS = 100 * (share of nps >= 9 - share of nps <= 6)
  In v_kpi_daily: service level = sum(answered_in_20s) / sum(offered - short_abandons).
  Rates for a period are ratios of sums. Never average service_level_20s, abandon_rate or
  any other rate column over several days or queues.
- Date columns: fact_calls.call_date, fact_surveys.call_date, fact_tickets.opened_date,
  fact_service_orders.order_date, dim_vehicle.sale_date and warranty_end. Use year(col) = 2024
  for a year, date_trunc('month', col) for months and date_trunc('week', col) for weeks
  (weeks start on Monday). Add days with col + INTERVAL 30 DAY.
- Calls per day: count fact_calls per call_date first, then average those daily counts.
- An identified customer is customer_id IS NOT NULL; no join is needed for that.
- "For each X, the Y with the most Z": group by X and Y and keep the top row per X with
  QUALIFY row_number() OVER (PARTITION BY X ORDER BY count(*) DESC) = 1. List the GROUP BY
  columns explicitly (DuckDB does not allow QUALIFY with GROUP BY ALL).
- Use only the columns listed for each table. Use as few tables as possible: join another table
  only when the question needs one of its columns.
- Return rates as fractions between 0 and 1. Give columns short readable aliases.
- Personal data (names, phone numbers, e-mail addresses) is not available. If the question asks
  for it, or asks to change data, answer exactly: REFUSE: <one-line reason>.
- Reply with the SQL in a ```sql code block, nothing else."""

EXAMPLES = [
    ("What was the abandon rate of the sales queue in April 2025?",
     "SELECT sum(abandoned) / sum(offered - short_abandons) AS abandon_rate\n"
     "FROM v_kpi_daily\nWHERE queue = 'satis' AND call_date BETWEEN DATE '2025-04-01' AND DATE '2025-04-30'"),
    ("Kestrel K5 için açılan şikâyet sayısı fault code bazında nedir?",
     "SELECT t.fault_code, count(*) AS tickets\nFROM fact_tickets t JOIN dim_vehicle v USING (vin)\n"
     "WHERE v.model = 'Kestrel K5' AND t.fault_code IS NOT NULL\nGROUP BY 1 ORDER BY 2 DESC"),
    ("Service level of each queue by month in 2023",
     "SELECT queue, date_trunc('month', call_date) AS month,\n"
     "       sum(answered_in_20s) / sum(offered - short_abandons) AS service_level\n"
     "FROM v_kpi_daily\nWHERE year(call_date) = 2023\nGROUP BY ALL ORDER BY 1, 2"),
    ("Genel kuyruğun 2023 NPS değeri",
     "SELECT 100 * (avg((nps >= 9)::INT) - avg((nps <= 6)::INT)) AS nps\n"
     "FROM fact_surveys\nWHERE queue = 'genel' AND year(call_date) = 2023"),
    ("For each city, the reason code with the most calls in 2023",
     "SELECT c.city, f.reason_code, count(*) AS calls\n"
     "FROM fact_calls f JOIN v_customer c USING (customer_id)\n"
     "WHERE year(f.call_date) = 2023 AND f.reason_code IS NOT NULL\n"
     "GROUP BY c.city, f.reason_code\n"
     "QUALIFY row_number() OVER (PARTITION BY c.city ORDER BY count(*) DESC) = 1"),
    ("Average calls per day on holiday eves (arife) and on other days in 2023",
     "SELECT d.is_arife, avg(c.calls) AS calls_per_day\n"
     "FROM (SELECT call_date, count(*) AS calls FROM fact_calls\n"
     "      WHERE year(call_date) = 2023 GROUP BY call_date) c\n"
     "JOIN dim_date d ON d.date = c.call_date\nGROUP BY 1"),
]  # fmt: skip


def schema_context(con) -> str:
    start, end = con.execute("SELECT min(call_date), max(call_date) FROM fact_calls").fetchall()[0]
    lines = ["Tables (DuckDB):"]
    cols = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "ORDER BY table_name, ordinal_position"
    ).df()
    for t in sorted(ALLOWED_TABLES):
        c = cols[cols["table_name"] == t]
        col_list = ", ".join(f"{r.column_name} {r.data_type.lower()}" for r in c.itertuples())
        lines.append(f"- {t}: {TABLE_NOTES.get(t, '')}\n  columns: {col_list}")
        for col in CATEGORY_COLUMNS.get(t, []):
            vals = [
                r[0]
                for r in con.execute(
                    f"SELECT DISTINCT {col} FROM {t} WHERE {col} IS NOT NULL ORDER BY 1 LIMIT 30"
                ).fetchall()
            ]
            lines.append(f"  {col} values: {', '.join(map(str, vals))}")
    lines.append("Joins: fact tables join dim_vehicle on vin, v_customer on customer_id, "
                 "dim_service_point on service_code, dim_date on date = call_date.")  # fmt: skip
    rules = RULES.format(start=start, end=end, year=end.year, prev_year=end.year - 1)
    examples = "\n\n".join(f"Question: {q}\n```sql\n{s}\n```" for q, s in EXAMPLES)
    return "\n".join(lines) + "\n\n" + rules + "\n\nExamples:\n\n" + examples
