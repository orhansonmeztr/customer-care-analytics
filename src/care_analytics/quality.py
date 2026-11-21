"""Data quality: profile the raw extracts, check rules, clean into staging tables.

Principles
----------
* Never fix silently. Every cleaned column keeps a ``*_status`` next to it, so each record
  can be traced back and the defect counted in the report.
* A rule belongs to one of six quality dimensions: completeness, uniqueness, validity,
  consistency, accuracy and timeliness. Privacy (personal data in free text) is tracked too.
* Not every anomaly is an error. A negative amount on an ``adjustment`` order is a refund;
  a model year after the sale year is normal in the car business. Rules with severity
  ``info`` describe such cases so the business can confirm them.
* Some checks are data driven: the PBX timezone shift and the incomplete loading days are
  detected from the data, not hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import duckdb
import pandas as pd

from .calendar_tr import holiday_calendar
from .generate import CITIES
from .paths import Paths

RAW_TABLES = [
    "models",
    "service_points",
    "customers",
    "vehicles",
    "service_orders",
    "calls",
    "tickets",
    "surveys",
]
OPEN_FROM, OPEN_TO = 8, 20  # contact center opening hours, local time
EMAIL_RE = r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"
VIN_RE = r"^[A-HJ-NPR-Z0-9]{17}$"
PHONE_IN_TEXT_RE = r"0?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}"

# A comparison key that survives Turkish casing problems: remove the combining dot that
# Python's "İ".lower() leaves behind, fold Turkish letters to ASCII, collapse whitespace.
MACROS = r"""
CREATE OR REPLACE MACRO tr_key(x) AS
    upper(regexp_replace(trim(translate(replace(x, chr(775), ''),
          'İıŞşĞğÜüÖöÇç', 'IiSsGgUuOoCc')), '\s+', ' ', 'g'));
CREATE OR REPLACE MACRO phone_digits(x) AS regexp_replace(x, '[^0-9]', '', 'g');
CREATE OR REPLACE MACRO phone10(x) AS
    CASE
        WHEN length(phone_digits(x)) = 12 AND phone_digits(x) LIKE '905%' THEN right(phone_digits(x), 10)
        WHEN length(phone_digits(x)) = 11 AND phone_digits(x) LIKE '05%'  THEN right(phone_digits(x), 10)
        WHEN length(phone_digits(x)) = 10 AND phone_digits(x) LIKE '5%'   THEN phone_digits(x)
    END;
"""


@dataclass
class Rule:
    rule_id: str
    table: str
    dimension: str
    description: str
    sql: str  # returns one row: failed, total
    severity: str = "error"  # error | warning | info
    action: str = ""


def connect(paths: Paths, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(paths.warehouse), read_only=read_only)
    if not read_only:
        for t in RAW_TABLES:
            con.execute(
                f"CREATE OR REPLACE VIEW raw_{t} AS "
                f"SELECT * FROM read_parquet('{(paths.raw / f'{t}.parquet').as_posix()}')"
            )
        con.execute(MACROS)
    return con


# ---------------------------------------------------------------------------- detection
def detect_timezone_shift(con) -> dict[str, int]:
    """For each PBX source, find the hour shift that puts the most calls in opening hours."""
    hours = con.execute(
        "SELECT source_system, hour(arrived_at) AS h, count(*) AS n FROM raw_calls GROUP BY ALL"
    ).df()
    shifts = {}
    for src, g in hours.groupby("source_system"):

        def outside(shift, g=g):
            h = (g["h"] + shift) % 24
            return g.loc[(h < OPEN_FROM) | (h >= OPEN_TO), "n"].sum() / g["n"].sum()

        best = min(range(-6, 7), key=lambda s: (outside(s), abs(s)))
        # only shift when the raw data is clearly off and the shift clearly fixes it
        shifts[src] = best if outside(0) > 0.01 and outside(best) < outside(0) / 5 else 0
    return shifts


# ---------------------------------------------------------------------------- staging
def build_customers_pre(con, as_of: str) -> None:
    ref_city = pd.DataFrame([(c, r) for c, _, r in CITIES], columns=["city_name", "region"])
    con.register("ref_city_df", ref_city)
    con.execute(
        "CREATE OR REPLACE TABLE ref_city AS "
        "SELECT tr_key(city_name) AS city_key, city_name, region FROM ref_city_df"
    )
    con.execute(f"""
    CREATE OR REPLACE TABLE stg_customers_pre AS
    SELECT
        customer_id,
        source_system,
        full_name,
        regexp_replace(trim(full_name), '\\s+', ' ', 'g')                  AS full_name_clean,
        tr_key(full_name)                                                   AS name_key,
        phone,
        phone10(phone)                                                      AS phone10,
        CASE WHEN phone IS NULL THEN 'missing'
             WHEN phone10(phone) IS NULL THEN 'invalid' ELSE 'ok' END       AS phone_status,
        CASE WHEN phone10(phone) IS NOT NULL THEN '+90' || phone10(phone) END AS phone_e164,
        lower(trim(email))                                                  AS email_key,
        CASE WHEN email IS NULL THEN 'missing'
             WHEN NOT regexp_full_match(lower(trim(email)), '{EMAIL_RE}') THEN 'invalid'
             ELSE 'ok' END                                                  AS email_status,
        c.city                                                              AS city_raw,
        r.city_name                                                         AS city,
        r.region,
        CASE WHEN r.city_name IS NULL THEN 'unmapped'
             WHEN c.city <> r.city_name THEN 'standardized' ELSE 'ok' END   AS city_status,
        nearest_service_km,
        created_at,
        CASE WHEN created_at > DATE '{as_of}' THEN 'future'
             WHEN created_at < DATE '1990-01-01' THEN 'sentinel' ELSE 'ok' END AS created_status
    FROM raw_customers c
    LEFT JOIN ref_city r ON r.city_key = tr_key(c.city)
    """)


def build_staging(con, as_of: str, tz_shift: dict[str, int]) -> None:
    """Staging tables that need the customer master map (entity resolution) first."""
    con.execute(f"""
    CREATE OR REPLACE TABLE stg_vehicles AS
    WITH v AS (
        SELECT *, row_number() OVER (PARTITION BY vin ORDER BY sale_date) AS rn
        FROM raw_vehicles
    )
    SELECT
        v.vin,
        regexp_full_match(v.vin, '{VIN_RE}')                         AS vin_valid,
        m.master_customer_id                                         AS customer_id,
        m.master_customer_id IS NULL                                 AS is_orphan,
        v.brand, v.model, md.segment, md.is_ev, v.model_year,
        CAST(v.sale_date AS DATE)                                    AS sale_date,
        CAST(v.sale_date + to_months(v.warranty_months) AS DATE)     AS warranty_end
    FROM v
    LEFT JOIN customer_master_map m ON m.customer_id = v.customer_id
    LEFT JOIN raw_models md USING (model)
    WHERE v.rn = 1
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_service_orders AS
    WITH dedup AS (
        SELECT *, row_number() OVER (
            PARTITION BY vin, order_date, order_type, odometer_km, amount_try ORDER BY order_id
        ) AS rn
        FROM raw_service_orders
    ),
    plaus AS (
        SELECT d.*, v.sale_date,
            d.odometer_km > 1000000
              OR (v.sale_date IS NOT NULL
                  AND d.odometer_km / greatest(date_diff('day', v.sale_date, d.order_date), 1) > 300)
                AS implausible
        FROM dedup d LEFT JOIN stg_vehicles v USING (vin)
        WHERE rn = 1
    ),
    seq AS (
        -- running maximum over earlier plausible readings only, so one typo with extra zeros
        -- does not make every later reading of the car look like a rollback
        SELECT *,
            max(CASE WHEN NOT implausible THEN odometer_km END) OVER (
                PARTITION BY vin ORDER BY order_date, order_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_km
        FROM plaus
    )
    SELECT
        order_id, vin, service_code, CAST(order_date AS DATE) AS order_date, order_type,
        odometer_km AS odometer_km_raw,
        CASE
            WHEN implausible THEN 'implausible'
            WHEN odometer_km < 0.98 * prev_max_km THEN 'regression'
            ELSE 'ok'
        END AS odometer_status,
        amount_try AS amount_try_raw,
        CASE WHEN amount_try < 0 AND order_type = 'adjustment' THEN 'adjustment'
             WHEN amount_try < 0 THEN 'negative_error' ELSE 'ok' END AS amount_status,
        sale_date IS NULL AS is_orphan
    FROM seq
    """)
    con.execute("""
    ALTER TABLE stg_service_orders ADD COLUMN odometer_km INTEGER;
    ALTER TABLE stg_service_orders ADD COLUMN amount_try DOUBLE;
    UPDATE stg_service_orders SET
        odometer_km = CASE WHEN odometer_status = 'ok' THEN odometer_km_raw END,
        amount_try  = CASE WHEN amount_status <> 'negative_error' THEN amount_try_raw END;
    """)

    shift_case = " ".join(f"WHEN '{s}' THEN {h}" for s, h in tz_shift.items()) or "WHEN '' THEN 0"
    con.execute(f"""
    CREATE OR REPLACE TABLE stg_calls AS
    WITH dedup AS (
        SELECT *, row_number() OVER (PARTITION BY call_id ORDER BY call_id) AS rn FROM raw_calls
    ),
    shifted AS (
        SELECT *,
            CASE source_system {shift_case} ELSE 0 END AS tz_shift_h
        FROM dedup WHERE rn = 1
    ),
    t AS (
        SELECT
            call_id, source_system, tz_shift_h,
            arrived_at  + to_hours(tz_shift_h) AS arrived_at,
            answered_at + to_hours(tz_shift_h) AS answered_at,
            ended_at    + to_hours(tz_shift_h) AS ended_at,
            hold_sec, wrap_sec, queue, reason_code, s.customer_id AS customer_id_raw,
            m.master_customer_id AS customer_id, vin, agent_id, transferred
        FROM shifted s
        LEFT JOIN customer_master_map m ON m.customer_id = s.customer_id
    )
    SELECT
        *,
        CAST(arrived_at AS DATE)                                         AS call_date,
        time_bucket(INTERVAL 30 MINUTE, arrived_at)                       AS interval_start,
        NOT coalesce(ended_at < arrived_at OR answered_at < arrived_at
                     OR answered_at > ended_at, false)                    AS time_order_ok,
        answered_at IS NOT NULL                                           AS is_answered,
        CASE WHEN answered_at IS NOT NULL THEN date_diff('second', arrived_at, answered_at)
             ELSE greatest(date_diff('second', arrived_at, ended_at), 0) END AS wait_sec,
        CASE WHEN answered_at IS NOT NULL THEN date_diff('second', answered_at, ended_at) - hold_sec
        END                                                               AS talk_sec,
        answered_at IS NULL AND date_diff('second', arrived_at, ended_at) >= 5 AS is_abandoned,
        answered_at IS NULL AND date_diff('second', arrived_at, ended_at) < 5  AS is_short_abandon,
        answered_at IS NOT NULL AND date_diff('second', arrived_at, answered_at) <= 20
                                                                          AS answered_in_20s
    FROM t
    """)

    # Completeness of each loading day: compare with the median of the same weekday over the
    # previous 8 weeks. A missing day becomes visible only against a full calendar.
    con.execute(f"""
    CREATE OR REPLACE TABLE dq_daily_load AS
    WITH cal AS (
        SELECT CAST(d AS DATE) AS call_date
        FROM generate_series((SELECT min(call_date) FROM stg_calls), DATE '{as_of}',
                             INTERVAL 1 DAY) g(d)
    ),
    daily AS (SELECT call_date, count(*) AS n_calls FROM stg_calls GROUP BY 1),
    j AS (SELECT cal.call_date, coalesce(n_calls, 0) AS n_calls
          FROM cal LEFT JOIN daily USING (call_date)),
    e AS (
        SELECT call_date, n_calls,
            median(nullif(n_calls, 0)) OVER (
                PARTITION BY dayofweek(call_date) ORDER BY call_date
                ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS expected
        FROM j
    )
    SELECT call_date, n_calls, expected,
        CASE WHEN n_calls = 0 THEN 'missing'
             -- partial: well below the usual level, and by more than Poisson noise explains
             WHEN n_calls < 0.6 * expected AND expected - n_calls > 4 * sqrt(expected)
                 THEN 'partial' ELSE 'ok' END AS load_status
    FROM e
    """)
    # a holiday also has low volume: only flag partial days that are not public holidays
    cal = holiday_calendar("2000-01-01", as_of)
    con.register("hol_df", cal[cal["is_holiday"] | cal["is_arife"] | cal["is_post_holiday"]])
    con.execute(
        "UPDATE dq_daily_load SET load_status = 'ok' WHERE load_status = 'partial' "
        "AND call_date IN (SELECT CAST(date AS DATE) FROM hol_df)"
    )

    con.execute(f"""
    CREATE OR REPLACE TABLE stg_tickets AS
    SELECT
        ticket_id, opened_at, closed_at, status, channel,
        m.master_customer_id AS customer_id, t.vin, service_code, category, fault_code, call_id,
        regexp_replace(description, '{PHONE_IN_TEXT_RE}', '[TELEFON]', 'g') AS description_masked,
        regexp_matches(description, '{PHONE_IN_TEXT_RE}')                  AS had_pii,
        v.vin IS NULL                                                      AS is_orphan
    FROM raw_tickets t
    LEFT JOIN customer_master_map m ON m.customer_id = t.customer_id
    LEFT JOIN stg_vehicles v ON v.vin = t.vin
    """)
    con.execute("""
    CREATE OR REPLACE TABLE stg_surveys AS
    SELECT s.*, c.call_id IS NULL AS is_orphan
    FROM raw_surveys s LEFT JOIN stg_calls c USING (call_id)
    """)


# ---------------------------------------------------------------------------- entity resolution
def resolve_customers(con, name_threshold: float = 0.90) -> pd.DataFrame:
    """Link CRM records that describe the same person.

    Blocking on strong keys (phone, e-mail) keeps the comparison cheap, and a name similarity
    threshold protects against merging two different people who share a family phone.
    A false merge is worse than a missed one: it mixes two customers' personal data.
    """
    pairs = con.execute("""
        WITH k AS (
            SELECT customer_id, name_key,
                   CASE WHEN phone_status = 'ok' THEN phone10 END AS phone10,
                   CASE WHEN email_status = 'ok' THEN email_key END AS email_key
            FROM stg_customers_pre
        ),
        cand AS (
            SELECT a.customer_id AS id_a, b.customer_id AS id_b FROM k a JOIN k b
              ON a.phone10 = b.phone10 AND a.customer_id < b.customer_id
            UNION
            SELECT a.customer_id, b.customer_id FROM k a JOIN k b
              ON a.email_key = b.email_key AND a.customer_id < b.customer_id
        )
        SELECT c.id_a, c.id_b,
               jaro_winkler_similarity(a.name_key, b.name_key) AS name_sim,
               coalesce(a.phone10 = b.phone10, false)          AS phone_match,
               coalesce(a.email_key = b.email_key, false)      AS email_match
        FROM cand c JOIN k a ON a.customer_id = c.id_a JOIN k b ON b.customer_id = c.id_b
    """).df()
    pairs["is_match"] = pairs["name_sim"] >= name_threshold

    parent: dict[int, int] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for a, b in pairs.loc[pairs["is_match"], ["id_a", "id_b"]].itertuples(index=False):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    ids = con.execute("SELECT customer_id, source_system, created_at FROM stg_customers_pre").df()
    ids["cluster"] = [find(i) for i in ids["customer_id"]]
    # the master record: the CRM record first, then the oldest one
    ids["prio"] = (ids["source_system"] != "crm").astype(int)
    ids = ids.sort_values(["cluster", "prio", "created_at", "customer_id"])
    ids["master_customer_id"] = ids.groupby("cluster")["customer_id"].transform("first")
    con.register("map_df", ids[["customer_id", "master_customer_id"]])
    con.execute("CREATE OR REPLACE TABLE customer_master_map AS SELECT * FROM map_df")
    con.register("pairs_df", pairs)
    con.execute("CREATE OR REPLACE TABLE er_candidate_pairs AS SELECT * FROM pairs_df")
    return pairs


def score_entity_resolution(con, paths: Paths) -> dict:
    """Precision / recall against the duplicates the generator injected (evaluation only)."""
    truth_file = paths.truth / "duplicate_customers.parquet"
    if not truth_file.exists():
        return {}
    truth = pd.read_parquet(truth_file)
    m = (
        con.execute("SELECT * FROM customer_master_map")
        .df()
        .set_index("customer_id")["master_customer_id"]
    )
    true_pairs = {tuple(sorted(p)) for p in truth.itertuples(index=False)}
    clusters = m.groupby(m).apply(lambda s: sorted(s.index))
    found = {(c[i], c[j]) for c in clusters for i in range(len(c)) for j in range(i + 1, len(c))}
    tp = len(true_pairs & found)
    return {
        "true_duplicate_pairs": len(true_pairs),
        "linked_pairs": len(found),
        "true_positives": tp,
        "precision": round(tp / max(len(found), 1), 4),
        "recall": round(tp / max(len(true_pairs), 1), 4),
    }


# ---------------------------------------------------------------------------- rules
def rules(as_of: str) -> list[Rule]:
    R = Rule
    return [
        # customers
        R(
            "cust_name_missing",
            "customers",
            "completeness",
            "Customer name is empty",
            "SELECT count(*) FILTER (WHERE full_name IS NULL OR trim(full_name) = ''), count(*) "
            "FROM stg_customers_pre",
            "warning",
            "Kept; name filled from a linked record if any",
        ),
        R(
            "cust_phone_missing",
            "customers",
            "completeness",
            "Phone number is empty",
            "SELECT count(*) FILTER (WHERE phone_status = 'missing'), count(*) FROM stg_customers_pre",
            "warning",
            "Kept; customer cannot be reached by phone",
        ),
        R(
            "cust_phone_invalid",
            "customers",
            "validity",
            "Phone cannot be read as a Turkish mobile",
            "SELECT count(*) FILTER (WHERE phone_status = 'invalid'), count(*) FROM stg_customers_pre",
            "error",
            "phone_e164 set to NULL, status kept for feedback to the source",
        ),
        R(
            "cust_email_invalid",
            "customers",
            "validity",
            "E-mail does not match the pattern",
            "SELECT count(*) FILTER (WHERE email_status = 'invalid'), count(*) FROM stg_customers_pre",
            "error",
            "E-mail set to NULL",
        ),
        R(
            "cust_city_standardized",
            "customers",
            "consistency",
            "City written in a non-standard way (case, Turkish letters, spaces)",
            "SELECT count(*) FILTER (WHERE city_status = 'standardized'), count(*) FROM stg_customers_pre",
            "info",
            "Mapped to the reference city list with a Turkish-aware key",
        ),
        R(
            "cust_city_unmapped",
            "customers",
            "validity",
            "City not in the reference list",
            "SELECT count(*) FILTER (WHERE city_status = 'unmapped'), count(*) FROM stg_customers_pre",
            "error",
            "City set to NULL",
        ),
        R(
            "cust_created_future",
            "customers",
            "accuracy",
            f"Created after the extract date {as_of}",
            "SELECT count(*) FILTER (WHERE created_status = 'future'), count(*) FROM stg_customers_pre",
            "error",
            "created_at set to NULL",
        ),
        R(
            "cust_created_sentinel",
            "customers",
            "validity",
            "Placeholder date such as 1900-01-01",
            "SELECT count(*) FILTER (WHERE created_status = 'sentinel'), count(*) FROM stg_customers_pre",
            "error",
            "created_at set to NULL",
        ),
        R(
            "cust_duplicate_records",
            "customers",
            "uniqueness",
            "Record describes a customer that already exists (entity resolution)",
            "SELECT count(*) FILTER (WHERE customer_id <> master_customer_id), count(*) "
            "FROM customer_master_map",
            "error",
            "Linked to a master customer id",
        ),
        # vehicles
        R(
            "veh_vin_invalid",
            "vehicles",
            "validity",
            "VIN is not 17 characters of A-Z/0-9 without I, O, Q",
            "SELECT count(*) FILTER (WHERE NOT vin_valid), count(*) FROM stg_vehicles",
            "error",
            "Kept and flagged; excluded from VIN-level models",
        ),
        R(
            "veh_vin_duplicate",
            "vehicles",
            "uniqueness",
            "Same VIN appears more than once",
            "SELECT count(*) - count(DISTINCT vin), count(*) FROM raw_vehicles",
            "error",
            "First sale kept",
        ),
        R(
            "veh_orphan_customer",
            "vehicles",
            "consistency",
            "Vehicle owner not found in CRM",
            "SELECT count(*) FILTER (WHERE is_orphan), count(*) FROM stg_vehicles",
            "error",
            "Kept and flagged; excluded from customer-level analysis",
        ),
        # service orders
        R(
            "so_exact_duplicate",
            "service_orders",
            "uniqueness",
            "Same order loaded twice under a new id",
            "SELECT (SELECT count(*) FROM raw_service_orders) - count(*), "
            "(SELECT count(*) FROM raw_service_orders) FROM stg_service_orders",
            "error",
            "Lowest order id kept",
        ),
        R(
            "so_orphan_vin",
            "service_orders",
            "consistency",
            "Order for a VIN that is not in the vehicle table",
            "SELECT count(*) FILTER (WHERE is_orphan), count(*) FROM stg_service_orders",
            "warning",
            "Kept and flagged",
        ),
        R(
            "so_negative_amount",
            "service_orders",
            "validity",
            "Negative amount on a non-adjustment order",
            "SELECT count(*) FILTER (WHERE amount_status = 'negative_error'), count(*) "
            "FROM stg_service_orders",
            "error",
            "Amount set to NULL",
        ),
        R(
            "so_adjustment_credit",
            "service_orders",
            "validity",
            "Negative amount on an adjustment order (refund / goodwill credit)",
            "SELECT count(*) FILTER (WHERE amount_status = 'adjustment'), count(*) "
            "FROM stg_service_orders",
            "info",
            "Legitimate: kept as is",
        ),
        R(
            "so_odometer_regression",
            "service_orders",
            "accuracy",
            "Odometer lower than an earlier reading of the same vehicle",
            "SELECT count(*) FILTER (WHERE odometer_status = 'regression'), count(*) "
            "FROM stg_service_orders",
            "error",
            "Odometer set to NULL; worth a review (entry error or tampering)",
        ),
        R(
            "so_odometer_implausible",
            "service_orders",
            "accuracy",
            "Odometer above 1,000,000 km or more than 300 km per day since sale",
            "SELECT count(*) FILTER (WHERE odometer_status = 'implausible'), count(*) "
            "FROM stg_service_orders",
            "error",
            "Odometer set to NULL",
        ),
        # calls
        R(
            "calls_duplicate_id",
            "calls",
            "uniqueness",
            "Call loaded more than once (replayed batch)",
            "SELECT count(*) - count(DISTINCT call_id), count(*) FROM raw_calls",
            "error",
            "One row per call id kept",
        ),
        R(
            "calls_outside_hours_raw",
            "calls",
            "consistency",
            f"Arrival outside opening hours {OPEN_FROM:02d}:00-{OPEN_TO:02d}:00 before the timezone fix",
            f"SELECT count(*) FILTER (WHERE hour(arrived_at) < {OPEN_FROM} OR hour(arrived_at) >= {OPEN_TO}),"
            " count(*) FROM raw_calls",
            "error",
            "Timezone shift detected per PBX source and corrected",
        ),
        R(
            "calls_outside_hours_clean",
            "calls",
            "consistency",
            "Arrival outside opening hours after the timezone fix",
            f"SELECT count(*) FILTER (WHERE hour(arrived_at) < {OPEN_FROM} OR hour(arrived_at) >= {OPEN_TO}),"
            " count(*) FROM stg_calls",
            "warning",
            "Should be close to zero",
        ),
        R(
            "calls_time_order",
            "calls",
            "consistency",
            "End before arrival, or answer outside the call",
            "SELECT count(*) FILTER (WHERE NOT time_order_ok), count(*) FROM stg_calls",
            "error",
            "Excluded from duration KPIs",
        ),
        R(
            "calls_reason_missing",
            "calls",
            "completeness",
            "Answered call without a reason code",
            "SELECT count(*) FILTER (WHERE reason_code IS NULL), count(*) FROM stg_calls "
            "WHERE is_answered",
            "warning",
            "Reported as 'unknown'; agent coaching topic",
        ),
        R(
            "calls_unknown_customer",
            "calls",
            "consistency",
            "Customer id not found in CRM",
            "SELECT count(*) FILTER (WHERE customer_id_raw IS NOT NULL AND customer_id IS NULL), "
            "count(*) FROM stg_calls",
            "warning",
            "Treated as an anonymous caller",
        ),
        R(
            "calls_load_incomplete",
            "calls",
            "timeliness",
            "Loading day missing, or below 60% of the same weekday median by more than noise explains",
            "SELECT count(*) FILTER (WHERE load_status <> 'ok'), count(*) FROM dq_daily_load",
            "error",
            "Day excluded from KPI trends and forecast training",
        ),
        # tickets & surveys
        R(
            "tickets_pii_in_text",
            "tickets",
            "privacy",
            "Phone number typed into the free text (KVKK)",
            "SELECT count(*) FILTER (WHERE had_pii), count(*) FROM stg_tickets",
            "error",
            "Masked as [TELEFON] in description_masked",
        ),
        R(
            "tickets_orphan_vin",
            "tickets",
            "consistency",
            "Ticket for an unknown VIN",
            "SELECT count(*) FILTER (WHERE is_orphan), count(*) FROM stg_tickets",
            "warning",
            "Kept and flagged",
        ),
        R(
            "surveys_orphan_call",
            "surveys",
            "consistency",
            "Survey for a call that does not exist",
            "SELECT count(*) FILTER (WHERE is_orphan), count(*) FROM stg_surveys",
            "warning",
            "Excluded from CSAT / NPS",
        ),
        R(
            "surveys_out_of_range",
            "surveys",
            "validity",
            "CSAT outside 1-5 or NPS outside 0-10",
            "SELECT count(*) FILTER (WHERE csat NOT BETWEEN 1 AND 5 OR nps NOT BETWEEN 0 AND 10), "
            "count(*) FROM stg_surveys",
            "error",
            "Excluded",
        ),
    ]


def run_rules(con, as_of: str) -> pd.DataFrame:
    out = []
    for r in rules(as_of):
        failed, total = con.execute(r.sql).fetchall()[0]
        d = asdict(r)
        d.pop("sql")
        d.update(
            failed=int(failed),
            total=int(total),
            failed_pct=round(100 * failed / total, 3) if total else 0.0,
        )
        out.append(d)
    res = pd.DataFrame(out)
    con.register("dq_df", res)
    con.execute("CREATE OR REPLACE TABLE dq_results AS SELECT * FROM dq_df")
    return res


def profile(con) -> pd.DataFrame:
    rows = []
    for t in RAW_TABLES:
        n = con.execute(f"SELECT count(*) FROM raw_{t}").fetchall()[0][0]
        for col, null_pct in con.execute(
            f"SELECT column_name, null_percentage FROM (SUMMARIZE raw_{t})"
        ).fetchall():
            rows.append({"table": t, "rows": n, "column": col, "null_pct": float(null_pct or 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- orchestration
def run_quality(paths: Paths) -> dict:
    con = connect(paths)
    as_of = str(con.execute("SELECT max(CAST(arrived_at AS DATE)) FROM raw_calls").fetchall()[0][0])
    tz = detect_timezone_shift(con)

    build_customers_and_links(con, as_of)
    build_staging(con, as_of, tz)
    res = run_rules(con, as_of)
    er = score_entity_resolution(con, paths)
    load = con.execute(
        "SELECT call_date, n_calls, expected, load_status FROM dq_daily_load "
        "WHERE load_status <> 'ok' ORDER BY 1"
    ).df()
    summary = {
        "as_of": as_of,
        "timezone_shift_hours": tz,
        "entity_resolution": er,
        "incomplete_days": [
            {
                "date": str(pd.Timestamp(r.call_date).date()),
                "n_calls": int(r.n_calls),
                "expected": float(r.expected or 0),
                "status": r.load_status,
            }
            for r in load.itertuples()
        ],
        "rules_failed": int((res["failed"] > 0).sum()),
        "rules_total": len(res),
    }
    write_report(paths, res, summary, profile(con))
    con.close()
    return summary


def build_customers_and_links(con, as_of: str) -> None:
    """Pre-clean customers, link duplicates, then publish the cleaned customer table."""
    build_customers_pre(con, as_of)
    resolve_customers(con)
    con.execute("""
    CREATE OR REPLACE TABLE stg_customers AS
    SELECT p.*, m.master_customer_id,
        CASE WHEN created_status = 'ok' THEN created_at END AS created_at_clean
    FROM stg_customers_pre p JOIN customer_master_map m USING (customer_id)
    """)


def write_report(paths: Paths, res: pd.DataFrame, summary: dict, prof: pd.DataFrame) -> None:
    lines = ["# Data quality report", "", f"Extract date: **{summary['as_of']}**", ""]
    lines += ["## Findings that changed the data", ""]
    tz = {k: v for k, v in summary["timezone_shift_hours"].items() if v}
    if tz:
        lines.append(
            "- **Timezone:** "
            + ", ".join(
                f"`{k}` timestamps were {-v:+d} h off local time and were shifted by {v:+d} h"
                for k, v in tz.items()
            )
        )
    for d in summary["incomplete_days"]:
        lines.append(
            f"- **{d['status'].title()} load on {d['date']}:** {d['n_calls']:,} calls "
            f"vs a usual {d['expected']:,.0f} for that weekday. Excluded from trends and "
            "forecast training."
        )
    er = summary["entity_resolution"]
    if er:
        lines.append(
            f"- **Duplicate customers:** {er['linked_pairs']:,} record pairs linked. Scored "
            f"against the injected duplicates: precision {er['precision']:.1%}, "
            f"recall {er['recall']:.1%}."
        )
    lines += [
        "",
        "## Rule results",
        "",
        "| Rule | Table | Dimension | Severity | Failed | Total | % | Action |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in res.itertuples():
        lines.append(
            f"| {r.description} (`{r.rule_id}`) | {r.table} | {r.dimension} | {r.severity} "
            f"| {r.failed:,} | {r.total:,} | {r.failed_pct:.2f} | {r.action} |"
        )
    lines += [
        "",
        "## Columns with missing values (raw)",
        "",
        "| Table | Column | Null % |",
        "|---|---|---:|",
    ]
    for r in prof[prof["null_pct"] > 0].sort_values("null_pct", ascending=False).itertuples():
        lines.append(f"| {r.table} | {r.column} | {r.null_pct:.2f} |")
    lines += [
        "",
        "Missing values are not all defects: an unanswered call has no reason code "
        "or agent, and an open ticket has no close date.",
    ]
    (paths.reports / "data_quality.md").write_text("\n".join(lines) + "\n")
    (paths.outputs / "data_quality.json").write_text(
        json.dumps({**summary, "rules": res.to_dict(orient="records")}, indent=2, default=str)
    )
