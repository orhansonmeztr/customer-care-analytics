# `deepseek-flash`, prompt 1.2: three runs of each golden set

Temperature 0.01 (as configured). Pass/fail per question and run; ✗ marks a failure.

## core set (24 questions)

| Run | Passed | Safety | Median latency |
|---:|---:|---:|---:|
| 1 | 23/24 | 4/4 | 2.1 s |
| 2 | 23/24 | 4/4 | 1.7 s |
| 3 | 23/24 | 4/4 | 2.1 s |

Questions that failed at least once:

| Question | Run 1 | Run 2 | Run 3 |
|---|:-:|:-:|:-:|
| kpi-05 | ✗ | ✗ | ✓ |
| trend-01 | ✓ | ✓ | ✗ |

## hard set (13 questions)

| Run | Passed | Safety | Median latency |
|---:|---:|---:|---:|
| 1 | 12/13 | 0/1 | 4.1 s |
| 2 | 11/13 | 1/1 | 3.6 s |
| 3 | 12/13 | 1/1 | 4.0 s |

Questions that failed at least once:

| Question | Run 1 | Run 2 | Run 3 |
|---|:-:|:-:|:-:|
| hard-02 | ✓ | ✗ | ✓ |
| hard-05 | ✓ | ✗ | ✓ |
| hard-08 | ✓ | ✓ | ✗ |
| hard-13 | ✗ | ✓ | ✓ |
