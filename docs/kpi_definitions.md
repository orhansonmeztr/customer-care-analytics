# KPI definitions

One definition per KPI, used by the SQL views, the dashboard and the exports. When two
reports show different numbers for the same KPI, the cause is almost always two definitions;
this page is the single source.

| KPI | Definition | Source |
|---|---|---|
| **Offered calls** | All calls that reached the queue, after removing replayed duplicates | `fact_calls` |
| **Short abandon** | Caller hung up within 5 seconds of arriving in the queue; excluded from service level and abandon rate denominators (usually a wrong number or an IVR misroute) | `is_short_abandon` |
| **Abandon rate** | Unanswered calls that waited 5 s or more ÷ (offered − short abandons) | `v_kpi_daily.abandon_rate` |
| **Service level (80/20)** | Calls answered within 20 s ÷ (offered − short abandons). Target 80% | `v_kpi_daily.service_level_20s` |
| **ASA** | Average wait of answered calls, in seconds | `v_kpi_daily.asa_sec` |
| **AHT** | Average handle time of answered calls: talk + hold + after-call work (wrap) | `v_kpi_daily.aht_sec` |
| **Transfer rate** | Answered calls transferred to another team ÷ answered calls | `v_kpi_daily.transfer_rate` |
| **Repeat contact (7 days)** | Answered call from an identified customer who calls again about the same reason within 7 days | `fact_calls.repeat_7d` |
| **FCR (7 days)** | 1 − repeat contact rate. A proxy: a customer who does not call back is assumed resolved | `v_kpi_daily.fcr_7d` |
| **CSAT top-2** | Surveys scoring 4 or 5 on a 1–5 scale ÷ surveys | `v_kpi_daily.csat_top2` |
| **NPS** | % promoters (9–10) − % detractors (0–6), on surveys | `v_kpi_daily.nps` |
| **Occupancy** | Traffic (Erlangs) ÷ agents on the phone; above 85–90% agents burn out | staffing plan |
| **Shrinkage** | Share of paid time agents are not available for calls (breaks, training, leave). Scheduled agents = agents on the phone ÷ (1 − shrinkage) | staffing plan |
| **Service churn** | A vehicle with no authorized service order in the 12 months after its warranty ends | `churn.py` |

## Caveats

* **Surveys are not a random sample.** Unhappy customers answer less often (missing not at
  random), so CSAT and NPS are biased upward. Compare them over time, not as absolute truth.
* **FCR is a proxy.** A customer who gave up and went to a dealer also "does not call back".
* **Incomplete loading days** (see the data quality report) are flagged in `dim_date` and
  excluded from KPI trends in the dashboard.
* **Amounts are nominal TL.** For multi-year comparisons in a high-inflation economy, deflate
  with an index first.
