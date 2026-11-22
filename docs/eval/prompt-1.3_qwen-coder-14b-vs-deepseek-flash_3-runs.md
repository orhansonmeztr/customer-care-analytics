# Prompt 1.3: KPI and pattern examples, `qwen2.5-coder-14b` (local GPU) and `deepseek-flash`, three runs

Comparison v3 (time buckets are relabelled in both directions). All stored runs were re-graded
with v3 from the SQL they produced; only kpi-05 changed (a month given as a date instead of a
month number, now accepted). Qwen2.5-Coder-14B-Instruct-AWQ on vLLM (RTX 5070 Ti, local network),
temperature as configured.

Prompt 1.3 was written after reading the prompt-1.2 failures of the local model on the core and
hard sets, so those two sets are no longer a clean test of it. The **holdout** set (12 questions,
same skills, other queues, periods and tables) was written and scored at prompt 1.2 *before*
the prompt was changed, and its failures were not looked at until 1.3 was final.

What 1.3 adds: the service-level formula with column names and an explicit ban on averaging
rate columns; the date column of each table and how to bucket months and weeks; that
`v_kpi_daily` has one row per day *and queue*; calls per day as an average of daily counts;
"identified customer" without a join; the top-per-group pattern with `QUALIFY`; and four
examples (monthly service level, NPS, top-per-group, average per day) on data no test asks about.

## Passed per run

| Set | Model | Prompt 1.2 | Prompt 1.3 |
|---|---|---|---|
| core | qwen2.5-coder-14b | 18 / 18 / 18 of 24 | 21 / 21 / 21 of 24 |
| core | deepseek-flash | — | 24 / 24 / 24 of 24 |
| hard | qwen2.5-coder-14b | 3 / 3 / 3 of 13 | 9 / 9 / 9 of 13 |
| hard | deepseek-flash | — | 11 / 11 / 12 of 13 |
| holdout | qwen2.5-coder-14b | 6 / 6 / 6 of 12 | 11 / 11 / 11 of 12 |
| holdout | deepseek-flash | 10 / 10 / 10 of 12 | 11 / 11 / 11 of 12 |

deepseek-flash at prompt 1.2 on core and hard: 23/23/23 of 24 and 12/11/12 of 13 (comparison
v2, see [deepseek-flash_prompt-1.2_3-runs.md](deepseek-flash_prompt-1.2_3-runs.md)); its SQL was not kept, so it is not re-graded here.

**Re-run on regenerated data.** The post-holiday fix changed the calendar the generator uses, so
the whole dataset shifted slightly. Three more prompt-1.3 runs of qwen2.5-coder-14b on the new
data gave the same scores (11/12 holdout, 21/24 core, 9/13 hard, every run) and failed the same
questions. deepseek-flash was not re-run.

## Questions that failed at least once

* **deepseek-flash, prompt 1.2, holdout:** hold-05, hold-06
* **deepseek-flash, prompt 1.3, core:** none
* **deepseek-flash, prompt 1.3, hard:** hard-02, hard-06 (runs 1,2)
* **deepseek-flash, prompt 1.3, holdout:** hold-06
* **qwen2.5-coder-14b, prompt 1.2, core:** kpi-02, kpi-04, kpi-05, rank-03, service-02, tickets-03
* **qwen2.5-coder-14b, prompt 1.2, hard:** hard-01, hard-02, hard-03, hard-04, hard-05, hard-06, hard-08, hard-09, hard-10, hard-12
* **qwen2.5-coder-14b, prompt 1.2, holdout:** hold-04, hold-05, hold-06, hold-08, hold-09, hold-10
* **qwen2.5-coder-14b, prompt 1.3, core:** kpi-03, rank-03, tickets-03
* **qwen2.5-coder-14b, prompt 1.3, hard:** hard-02, hard-03, hard-06, hard-12
* **qwen2.5-coder-14b, prompt 1.3, holdout:** hold-08

## Reading

* The held-out gain of the local model, 6 → 11 of 12, is the honest measure of what the
  prompt does: those questions played no part in writing it. On the sets whose failures were
  studied, the gain is 18 → 21 (core) and 3 → 9 (hard).
* The local model now matches `deepseek-flash` on the held-out set (11 of 12 each) and is
  within 3 questions on the other two. The remaining gap is on multi-step questions: month
  over month with the first month compared against the previous year (hard-02), a cohort
  with a time window after an event (hard-06), and counting groups (hard-12).
* Results are identical across runs for the local model; `deepseek-flash` varies by at most
  one question per set.
* The prompt did not hurt the API model: core 23 → 24, holdout 10 → 11, hard unchanged.
* hold-08 (local model): repeat contacts divided by a column that does not exist in
  `fact_calls`; the repeat-rate rule is in the prompt but not in an example.
