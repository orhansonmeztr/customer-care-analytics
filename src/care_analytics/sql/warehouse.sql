-- Star schema built from the staging tables.
-- Grain of every table is stated in its comment. KPI definitions: docs/kpi_definitions.md

-- dim_customer: one row per real person (master customer id after entity resolution)
CREATE OR REPLACE TABLE dim_customer AS
SELECT
    master_customer_id                                              AS customer_id,
    arg_min(full_name_clean, prio)                                  AS full_name,
    arg_min(phone_e164, prio) FILTER (WHERE phone_e164 IS NOT NULL) AS phone_e164,
    arg_min(email_key, prio) FILTER (WHERE email_status = 'ok')     AS email,
    arg_min(city, prio) FILTER (WHERE city IS NOT NULL)             AS city,
    arg_min(region, prio) FILTER (WHERE region IS NOT NULL)         AS region,
    arg_min(nearest_service_km, prio)                               AS nearest_service_km,
    min(created_at_clean)                                           AS created_at,
    count(*)                                                        AS n_source_records
FROM (
    SELECT *, (CASE WHEN source_system = 'crm' THEN 0 ELSE 1 END) * 1e9 + customer_id AS prio
    FROM stg_customers
)
GROUP BY master_customer_id;

-- dim_vehicle: one row per VIN
CREATE OR REPLACE TABLE dim_vehicle AS
SELECT vin, vin_valid, customer_id, is_orphan, brand, model, segment, is_ev, model_year,
       sale_date, warranty_end
FROM stg_vehicles;

-- dim_service_point: one row per authorized service
CREATE OR REPLACE TABLE dim_service_point AS SELECT * FROM raw_service_points;

-- fact_calls: one row per call (after de-duplication and timezone correction)
CREATE OR REPLACE TABLE fact_calls AS
WITH c AS (
    SELECT *,
        lead(arrived_at) OVER (PARTITION BY customer_id, reason_code ORDER BY arrived_at) AS next_same
    FROM stg_calls
)
SELECT
    call_id, call_date, interval_start, arrived_at, queue,
    coalesce(reason_code, CASE WHEN is_answered THEN 'bilinmiyor' END) AS reason_code,
    customer_id, vin, agent_id,
    is_answered, is_abandoned, is_short_abandon, answered_in_20s,
    wait_sec,
    CASE WHEN is_answered AND time_order_ok THEN talk_sec END                       AS talk_sec,
    CASE WHEN is_answered THEN hold_sec END                                         AS hold_sec,
    CASE WHEN is_answered THEN wrap_sec END                                         AS wrap_sec,
    CASE WHEN is_answered AND time_order_ok THEN talk_sec + hold_sec + wrap_sec END AS handle_sec,
    transferred,
    -- repeat contact: the same customer calls again about the same reason within 7 days
    CASE WHEN is_answered AND customer_id IS NOT NULL AND reason_code IS NOT NULL
         THEN coalesce(next_same <= arrived_at + INTERVAL 7 DAY, false) END        AS repeat_7d,
    time_order_ok
FROM c;

-- fact_service_orders: one row per service order
CREATE OR REPLACE TABLE fact_service_orders AS
SELECT order_id, vin, service_code, order_date, order_type, odometer_km, odometer_status,
       amount_try, amount_status, is_orphan
FROM stg_service_orders;

-- fact_tickets: one row per complaint ticket, free text masked
CREATE OR REPLACE TABLE fact_tickets AS
SELECT ticket_id, CAST(opened_at AS DATE) AS opened_date, opened_at, closed_at, status, channel,
       customer_id, vin, service_code, category, fault_code, call_id, description_masked,
       date_diff('hour', opened_at, closed_at) / 24.0 AS resolution_days, is_orphan
FROM stg_tickets;

-- fact_surveys: one row per survey linked to a real call
CREATE OR REPLACE TABLE fact_surveys AS
SELECT s.survey_id, s.call_id, c.call_date, c.queue, s.responded_at, s.csat, s.nps
FROM stg_surveys s JOIN fact_calls c USING (call_id)
WHERE NOT s.is_orphan AND s.csat BETWEEN 1 AND 5 AND s.nps BETWEEN 0 AND 10;

-- v_kpi_daily: one row per day and queue
CREATE OR REPLACE VIEW v_kpi_daily AS
WITH c AS (
    SELECT call_date, queue,
        count(*)                                                AS offered,
        count(*) FILTER (WHERE is_short_abandon)                AS short_abandons,
        count(*) FILTER (WHERE is_answered)                     AS answered,
        count(*) FILTER (WHERE is_abandoned)                    AS abandoned,
        count(*) FILTER (WHERE answered_in_20s)                 AS answered_in_20s,
        avg(wait_sec) FILTER (WHERE is_answered)                AS asa_sec,
        avg(handle_sec)                                         AS aht_sec,
        avg(transferred::INT) FILTER (WHERE is_answered)        AS transfer_rate,
        1 - avg(repeat_7d::INT)                                 AS fcr_7d
    FROM fact_calls GROUP BY ALL
),
s AS (
    SELECT call_date, queue, count(*) AS surveys, avg(csat) AS csat_avg,
        avg((csat >= 4)::INT) AS csat_top2,
        100 * (avg((nps >= 9)::INT) - avg((nps <= 6)::INT)) AS nps
    FROM fact_surveys GROUP BY ALL
)
SELECT c.*,
    abandoned       / nullif(offered - short_abandons, 0) AS abandon_rate,
    answered_in_20s / nullif(offered - short_abandons, 0) AS service_level_20s,
    s.surveys, s.csat_avg, s.csat_top2, s.nps,
    d.is_incomplete_load
FROM c
LEFT JOIN s USING (call_date, queue)
LEFT JOIN dim_date d ON d.date = c.call_date;

-- v_kpi_interval: one row per 30-minute interval (all queues pooled), input to staffing
CREATE OR REPLACE VIEW v_kpi_interval AS
SELECT interval_start, CAST(interval_start AS DATE) AS call_date,
    count(*) AS offered, count(*) FILTER (WHERE is_answered) AS answered,
    avg(handle_sec) AS aht_sec,
    count(*) FILTER (WHERE answered_in_20s) / nullif(count(*) - count(*) FILTER (WHERE is_short_abandon), 0)
        AS service_level_20s
FROM fact_calls GROUP BY ALL;
