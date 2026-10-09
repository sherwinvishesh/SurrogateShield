# multiturn (real data, test3)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live multiturn --split test3 --out bench/results/multiturn_realdata_test3.json` at commit `3fa246c52302`

| group | arm | complete | restored | history original | gold in request | consistent | consistent, distinct | grades |
|---|---|---|---|---|---|---|---|---|
| oasst1 | ss | 20/20 | 24/24 (100.0%, 86.2%–100.0%) | 0/44 (0.0%, 0.0%–8.0%) | 0/44 (0.0%, 0.0%–8.0%) | – | – | {'correct': 13, 'partly': 5, 'incorrect': 1, 'not_needed': 1} |
| oasst1 | presidio_faker | 20/20 | 0/21 (0.0%, 0.0%–15.5%) | 0/44 (0.0%, 0.0%–8.0%) | 18/44 (40.9%, 27.7%–55.6%) | – | – | {'incorrect': 15, 'partly': 4, 'not_needed': 1} |
| oasst1 | gliner_pii | 20/20 | 0/18 (0.0%, 0.0%–17.6%) | 4/44 (9.1%, 3.6%–21.2%) | 0/44 (0.0%, 0.0%–8.0%) | – | – | {'incorrect': 12, 'not_needed': 7, 'partly': 1} |
| oasst1 | ss − presidio_faker | | | | | – | | judge +0.694 [+0.500, +0.861] |
| oasst1 | ss − gliner_pii | | | | | – | | judge +0.769 [+0.577, +0.923] |
| sharegpt | ss | 20/20 | 30/31 (96.8%, 83.8%–99.4%) | 0/55 (0.0%, 0.0%–6.5%) | 3/55 (5.5%, 1.9%–14.8%) | – | – | {'partly': 8, 'not_needed': 3, 'correct': 8, 'incorrect': 1} |
| sharegpt | presidio_faker | 20/20 | 0/30 (0.0%, 0.0%–11.3%) | 10/55 (18.2%, 10.2%–30.3%) | 30/55 (54.5%, 41.5%–67.0%) | – | – | {'partly': 8, 'incorrect': 8, 'not_needed': 4} |
| sharegpt | gliner_pii | 20/20 | 0/15 (0.0%, 0.0%–20.4%) | 11/55 (20.0%, 11.6%–32.4%) | 20/55 (36.4%, 24.9%–49.6%) | – | – | {'not_needed': 4, 'partly': 3, 'incorrect': 11, 'correct': 2} |
| sharegpt | ss − presidio_faker | | | | | – | | judge +0.433 [+0.200, +0.633] |
| sharegpt | ss − gliner_pii | | | | | – | | judge +0.464 [+0.249, +0.679] |
| wildchat | ss | 20/20 | 27/27 (100.0%, 87.5%–100.0%) | 1/51 (2.0%, 0.4%–10.3%) | 2/51 (3.9%, 1.1%–13.2%) | – | – | {'correct': 11, 'not_needed': 3, 'partly': 5, 'unavailable': 1} |
| wildchat | presidio_faker | 20/20 | 0/29 (0.0%, 0.0%–11.7%) | 6/51 (11.8%, 5.5%–23.4%) | 30/51 (58.8%, 45.2%–71.2%) | – | – | {'not_needed': 5, 'partly': 7, 'incorrect': 7, 'unavailable': 1} |
| wildchat | gliner_pii | 20/20 | 0/21 (0.0%, 0.0%–15.5%) | 12/51 (23.5%, 14.0%–36.8%) | 10/51 (19.6%, 11.0%–32.5%) | – | – | {'not_needed': 8, 'incorrect': 10, 'partly': 1, 'unavailable': 1} |
| wildchat | ss − presidio_faker | | | | | – | | judge +0.577 [+0.462, +0.731] |
| wildchat | ss − gliner_pii | | | | | – | | judge +0.750 [+0.600, +0.900] |
| all | ss | 60/60 | 81/82 (98.8%, 93.4%–99.8%) | 1/150 (0.7%, 0.1%–3.7%) | 5/150 (3.3%, 1.4%–7.6%) | – | – | {'correct': 32, 'partly': 18, 'incorrect': 2, 'not_needed': 7, 'unavailable': 1} |
| all | presidio_faker | 60/60 | 0/80 (0.0%, 0.0%–4.6%) | 16/150 (10.7%, 6.7%–16.6%) | 78/150 (52.0%, 44.1%–59.8%) | – | – | {'incorrect': 30, 'partly': 19, 'not_needed': 10, 'unavailable': 1} |
| all | gliner_pii | 60/60 | 0/54 (0.0%, 0.0%–6.6%) | 27/150 (18.0%, 12.7%–24.9%) | 30/150 (20.0%, 14.4%–27.1%) | – | – | {'incorrect': 33, 'not_needed': 19, 'partly': 5, 'correct': 2, 'unavailable': 1} |
| all | ss − presidio_faker | | | | | – | | judge +0.576 [+0.467, +0.685] |
| all | ss − gliner_pii | | | | | – | | judge +0.649 [+0.527, +0.770] |

H7''' on E7b (intervals over conversations above 0; the verdict is the pooled group's):

- oasst1: **does not hold** (judge_above_presidio_faker yes, consistency_above_presidio_faker no, judge_above_gliner_pii yes, consistency_above_gliner_pii no)
- sharegpt: **does not hold** (judge_above_presidio_faker yes, consistency_above_presidio_faker no, judge_above_gliner_pii yes, consistency_above_gliner_pii no)
- wildchat: **does not hold** (judge_above_presidio_faker yes, consistency_above_presidio_faker no, judge_above_gliner_pii yes, consistency_above_gliner_pii no)
- all: **does not hold** (judge_above_presidio_faker yes, consistency_above_presidio_faker no, judge_above_gliner_pii yes, consistency_above_gliner_pii no)
