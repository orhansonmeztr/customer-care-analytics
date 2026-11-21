# customer-care-analytics

[![CI](https://github.com/orhansonmeztr/customer-care-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/orhansonmeztr/customer-care-analytics/actions/workflows/ci.yml)

**Customer care analytics and capacity planning for an automotive distributor, end to end:**
messy raw extracts → data quality → star schema and KPIs → call volume forecast → Erlang C
staffing plan → customer segments → post-warranty churn with a cost-based call list →
complaint early warning → dashboard.

The company, "Meridian Auto", is fictional and so is all the data. A generator produces
2.5 years of CRM, vehicle, service, call-center, complaint and survey extracts for 150,000
customers. It injects the defects real extracts have and plants business events with known
ground truth, so every step of the pipeline can be *scored*, not just run.

## Executive summary

| Question | Answer |
|---|---|
| **Can we trust the data?** | Not as delivered. The pipeline found and fixed a 3-hour timezone error in the old PBX, one missing and one 40%-loaded day, 4,400 replayed call records, 4,400 duplicate customers (precision 100%, recall 98.3% against the injected truth) and phone numbers typed into complaint texts (masked for KVKK). |
| **How many calls next month?** | LightGBM forecasts the whole month on its first day with **9.7% WAPE**, against 15.5% for the best simple baseline. The gap is widest on holidays (22% vs 91–102%), because religious holidays move every year and "same day last year" breaks. |
| **How many agents?** | 32 agents on the phone at the busiest half hour reach 83% within 20 s; 33 give 88%. The plan covers 28 days in 30-minute intervals at P50 and P80, with 30% shrinkage. |
| **Who is about to leave the service network?** | Scored 90 days before the warranty ends. Calling the riskiest **40%** is worth **368k TL** on the test set; calling everyone is worth 25k TL, because loyal customers use the voucher too. Distance to the nearest authorized service is the strongest driver. |
| **Is a product problem emerging?** | A brake recall was flagged in its first week; a slowly growing EV battery problem after 21 days. 22 of 23 alerts were real. |

## What it shows (mapped to the role)

| Skill | Where |
|---|---|
| Python data processing and modelling | [`src/care_analytics`](src/care_analytics), pandas, scikit-learn, LightGBM, statsmodels |
| Advanced SQL | DuckDB: window functions, running max per vehicle, gap detection on a full calendar, entity resolution joins, KPI views ([`quality.py`](src/care_analytics/quality.py), [`warehouse.sql`](src/care_analytics/sql/warehouse.sql)) |
| Data preparation, cleaning and quality | 29 rules in 6 quality dimensions + privacy, `*_status` columns instead of silent fixes ([report](#data-quality)) |
| Data modelling and governance | Star schema with stated grain, one KPI definition per metric ([`docs/kpi_definitions.md`](docs/kpi_definitions.md)), personal data excluded from exports |
| Forecasting | Rolling-origin backtest against three baselines, Turkish holiday calendar, quantile forecast |
| Planning | Erlang C staffing per 30-minute interval, service level vs cost curve |
| Segmentation | RFM rules + k-means, profiled in business terms with an action per segment |
| Classification | Out-of-time validation, leakage guard, calibration (Brier), lift, cost-based threshold, explanations |
| Anomaly detection | Poisson baselines per car on the road, Benjamini-Hochberg false discovery control |
| Dashboards | Streamlit app; CSV extracts for Tableau / Power BI in `data/exports` |
| Engineering | CLI, tests on the injected truth, GitHub Actions, Docker |

## Architecture

```mermaid
flowchart LR
    G[Generator<br/>raw extracts + injected defects] --> R[(raw parquet)]
    R --> Q[Data quality<br/>profile, rules, timezone & load detection,<br/>entity resolution, PII masking]
    Q --> S[(staging tables<br/>+ status columns)]
    S --> W[(star schema<br/>dims, facts, KPI views)]
    W --> F[Forecast<br/>backtest + 28-day P50/P80]
    F --> P[Erlang C<br/>staffing plan]
    W --> SEG[Segmentation<br/>RFM + k-means]
    W --> C[Service churn<br/>call list]
    W --> A[Complaint<br/>early warning]
    W & P & SEG & C & A --> D[Dashboard<br/>+ CSV exports]
    G -. ground truth .-> T{{Scoring:<br/>tests and reports}}
```

## Run it

```bash
uv venv && uv pip install -e ".[dev,dashboard]"
care run-all                      # ~2 minutes: generate, quality, warehouse, forecast, segment, churn, anomaly
streamlit run dashboard/app.py
```

`care run-all --scale 0.1` builds a smaller dataset in seconds. Each step also runs alone
(`care quality`, `care forecast`, ...). Reports are written to `data/reports/*.md`, Tableau /
Power BI extracts to `data/exports/*.csv`.

With Docker (builds the data on first start):

```bash
docker build -t customer-care-analytics . && docker run -p 8501:8501 customer-care-analytics
```

Tests check the pipeline against what the generator injected:

```bash
pytest -q
```

## Data

| Extract | Rows (default scale) | Defects injected |
|---|---:|---|
| `customers` (CRM + dealer portal) | 154,500 | 3% duplicate people written by a second system (case, spacing, Turkish İ/ı, phone formats), invalid phones and e-mails, future and placeholder dates, 24% non-standard city spellings |
| `vehicles` | 172,500 | invalid VINs (16 characters, letter O), owners missing from CRM |
| `service_orders` | 635,000 | 1% re-loaded duplicates, odometer rollbacks and extra zeros, negative amounts (legitimate refunds *and* errors) |
| `calls` (PBX) | 1.46 M | PBX v1 wrote UTC instead of local time, one missing day, one 40% day, 0.3% replayed rows, broken time order |
| `tickets` | 139,000 | phone numbers in free text |
| `surveys` | 147,000 | unhappy customers answer less often; orphan survey links |

Planted events: a brake recall on one model (March 2024), a price-increase rush (December 2024)
and a slowly growing EV battery problem (from February 2025). Everything injected is written to
`data/raw/_truth`; the pipeline never reads it, only tests and scoring do.

## Results

### Data quality

Full report: `data/reports/data_quality.md`. Highlights:

* **Timezone:** detected from the data, not hard-coded. For each PBX source the pipeline finds
  the hour shift that puts calls inside opening hours: `pbx_v1` was 3 hours behind. Without the
  fix 3.5% of calls arrive "at night" and every intraday staffing profile is wrong.
* **Loading gaps:** each day is compared with the median of the same weekday over the previous
  8 weeks, on a complete calendar (a missing day has no rows to find otherwise). A day is
  flagged only if it is below 60% *and* further below than Poisson noise explains, and public
  holidays are excluded. These days are left out of KPI trends and forecast training: treating
  a missing load as "zero calls" teaches the model a drop that never happened.
* **Duplicate customers:** blocking on phone and e-mail, then Jaro-Winkler similarity on a
  Turkish-aware name key (Python's `"Ali".upper()` gives `ALI`, not `ALİ`; `"İ".lower()` leaves a
  combining dot). Precision 100%, recall 98.3%. A false merge mixes two people's personal data,
  so the threshold favours precision.
* **Not every anomaly is an error:** negative amounts on adjustment orders are refunds and are
  kept; negative amounts elsewhere are set to NULL. Odometer readings are checked per vehicle
  over time: one typo with extra zeros must not make every later reading look like a rollback.

### Forecast and staffing

Twelve monthly folds; each forecast is made on the first day of the month for the whole month,
so every input is at least 35 days old.

| Model | WAPE | Bias | Holiday-related days | Other days |
|---|---:|---:|---:|---:|
| Same weekday last week | 15.5% | +0.1% | 102.5% | 12.9% |
| Same weekday last year | 16.5% | −7.0% | 91.2% | 14.2% |
| Holt-Winters (log, weekly) | 16.9% | −3.7% | 94.3% | 14.6% |
| **LightGBM** (ratio-to-level target, holiday calendar) | **9.7%** | **−0.8%** | **22.3%** | **9.3%** |

LightGBM cannot extrapolate a trend, so it predicts the ratio of a day to a trailing level
instead of the raw count. The P80 forecast covers 72% of days instead of 80%: in-sample
residuals are smaller than out-of-time ones. A conformal correction from backtest errors is the
next step.

Staffing uses Erlang C with an 80/20 target and a 90% occupancy cap. At the busiest interval
(152 calls in 30 minutes):

| Agents | 30 | 31 | **32** | 33 | 34 |
|---|---:|---:|---:|---:|---:|
| Service level | 64.8% | 75.2% | **82.8%** | 88.3% | 92.2% |
| ASA (s) | 42 | 24 | **14** | 9 | 5 |

Erlang C ignores callers who hang up, so it slightly over-staffs; Erlang A or a simulation is
the refinement.

### Segmentation

127,829 customers with an authorized service visit in the last 36 months. RFM gives five
segments with an action each; the one that matters most is **"at risk (valuable)"** (12,400
customers): frequent visitors who have not come for ~10 months. k-means (k = 3 by silhouette)
separates a "high calls · high complaints" group of 33,000 customers that RFM cannot see.

### Post-warranty service churn

Churn = no authorized service visit in the 12 months after the 3-year warranty ends (38% of
cars). Trained on 10,233 earlier warranty expiries, tested out-of-time on 4,407 later ones.

| Model | ROC-AUC | PR-AUC | Lift top 10% |
|---|---:|---:|---:|
| **Logistic regression** | **0.624** | **0.516** | **1.65x** |
| LightGBM | 0.610 | 0.500 | 1.60x |

The simpler model wins and is kept: it is as accurate and easier to explain. A ROC-AUC around
0.62 is what this problem looks like: a large part of the decision (price sensitivity) is not
in any system.

The call list follows the economics, not a 0.5 threshold. A call costs 150 TL and offers a
1,000 TL voucher; it brings back 25% of churners, a retained car is worth 6,000 TL a year,
and half of the customers who would have come back anyway use the voucher too. Break-even
churn probability: 37%. Calling the riskiest 40% earns 368k TL on the test set; calling
everyone, 25k TL. Today's list: 1,555 of the 3,403 cars whose warranty ends in 60–120 days,
each with its main risk driver.

*Who* to call is in the end a causal question: the best rule ranks customers by their response
to the call, not by their risk. That is the subject of
[uplift-decision-engine](https://github.com/orhansonmeztr/uplift-decision-engine).

### Complaint early warning

Weekly tickets for 45 model × fault-code series. Each week is tested against a Poisson baseline
of the previous 12 weeks, as a **rate per car on the road** (a newly launched model gets more
complaints every month simply because more are sold), with Benjamini-Hochberg control across
series.

| Planted problem | Onset | Detected after |
|---|---|---:|
| Brake complaints, Kestrel K3 (recall) | 2024-03-01 | 2 days |
| Battery complaints, Volta E (slow drift) | 2025-02-02 | 21 days |

23 alerts, 22 on the planted problems. A CUSUM chart would catch the slow drift earlier.

## Project structure

```
src/care_analytics/
  generate.py        synthetic raw extracts, injected defects, ground truth
  calendar_tr.py     Turkish public and religious holidays
  quality.py         rules, timezone & load detection, entity resolution, staging tables
  sql/warehouse.sql  star schema and KPI views
  warehouse.py       builds the warehouse, writes Tableau / Power BI extracts
  forecast.py        features, backtest, 28-day forecast, intraday profile
  staffing.py        Erlang C
  segmentation.py    RFM and k-means
  churn.py           post-warranty churn, cost-based call list, explanations
  anomaly.py         complaint early warning
  cli.py             `care` command
dashboard/app.py     Streamlit dashboard
docs/kpi_definitions.md
tests/               unit tests + end-to-end checks against the injected truth
```

## Limitations

* Synthetic data: the relationships are the ones the generator encodes, so absolute numbers
  are illustrations. The value is in the method and in checking it against known truth.
* Amounts are nominal TL with no inflation; real multi-year analysis needs a deflator.
* Staffing assumes one pooled skill group and Erlang C.
* The churn save rate and voucher redemption are assumptions until a randomized test measures them.

## Next steps

* Conformal calibration of the forecast intervals; Erlang A with abandonment.
* CUSUM for slow complaint drifts.
* A natural-language analytics assistant over the star schema (text-to-SQL with an
  evaluation set), the pattern from [retail-rag-assistant](https://github.com/orhansonmeztr/retail-rag-assistant).
