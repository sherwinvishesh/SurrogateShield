# utility (real data, test3)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live utility --split test3 --out bench/results/utility_realdata_test3.json` at commit `cb2b83353ebd`

| group | n | BERTScore ss | ss raw | presidio_default | presidio_faker | gliner_pii | ss − presidio_default | ss − presidio_faker | ss − gliner_pii |
|---|---|---|---|---|---|---|---|---|---|
| oasst1 | 100 | 0.4514 | 0.447 | 0.36 | 0.3755 | 0.3707 | +0.091 [+0.059, +0.128] | +0.076 [+0.046, +0.108] | +0.081 [+0.039, +0.124] |
| sharegpt | 100 | 0.4191 | 0.4127 | 0.3521 | 0.332 | 0.3157 | +0.067 [+0.017, +0.118] | +0.087 [+0.037, +0.140] | +0.103 [+0.065, +0.143] |
| wildchat | 100 | 0.4551 | 0.4494 | 0.3615 | 0.3871 | 0.4218 | +0.094 [+0.049, +0.142] | +0.068 [+0.025, +0.113] | +0.033 [-0.014, +0.077] |
| all | 300 | 0.4419 | 0.4364 | 0.3579 | 0.3649 | 0.3694 | +0.084 [+0.059, +0.109] | +0.077 [+0.053, +0.103] | +0.072 [+0.046, +0.098] |

Pairwise judge (+1 when ss is preferred, −1 when the other arm is, 0 for a tie).

| group | pair | n | ss / other / tie | score | first position chosen | agrees with BERTScore |
|---|---|---|---|---|---|---|
| oasst1 | ss vs presidio_faker | 98 | 65/21/12 | +0.449 [+0.286, +0.602] | 43/86 (50.0%, 39.7%–60.3%) | 52/86 (60.5%, 49.9%–70.1%) |
| oasst1 | ss vs gliner_pii | 98 | 40/47/11 | -0.071 [-0.255, +0.102] | 33/87 (37.9%, 28.4%–48.4%) | 50/87 (57.5%, 47.0%–67.3%) |
| sharegpt | ss vs presidio_faker | 99 | 66/25/8 | +0.414 [+0.242, +0.586] | 43/91 (47.2%, 37.3%–57.4%) | 57/91 (62.6%, 52.4%–71.9%) |
| sharegpt | ss vs gliner_pii | 99 | 49/36/14 | +0.131 [-0.051, +0.313] | 45/85 (52.9%, 42.4%–63.2%) | 61/85 (71.8%, 61.4%–80.2%) |
| wildchat | ss vs presidio_faker | 99 | 64/16/19 | +0.485 [+0.333, +0.626] | 44/80 (55.0%, 44.1%–65.4%) | 51/79 (64.6%, 53.6%–74.2%) |
| wildchat | ss vs gliner_pii | 99 | 43/34/22 | +0.091 [-0.081, +0.263] | 32/77 (41.6%, 31.2%–52.7%) | 43/76 (56.6%, 45.4%–67.1%) |
| all | ss vs presidio_faker | 296 | 195/62/39 | +0.449 [+0.355, +0.540] | 130/257 (50.6%, 44.5%–56.6%) | 160/256 (62.5%, 56.4%–68.2%) |
| all | ss vs gliner_pii | 296 | 132/117/47 | +0.051 [-0.051, +0.152] | 110/249 (44.2%, 38.1%–50.4%) | 154/248 (62.1%, 55.9%–67.9%) |

H7''' on E5b (lower bounds above −0.02; the verdict is the pooled group's):

- oasst1: **does not hold** (bertscore_not_below_presidio_faker yes, judge_not_below_presidio_faker yes, bertscore_not_below_gliner_pii yes, judge_not_below_gliner_pii no; strict, deciding nothing: bertscore_above_presidio_faker yes, judge_above_presidio_faker yes, bertscore_above_gliner_pii yes, judge_above_gliner_pii no)
- sharegpt: **does not hold** (bertscore_not_below_presidio_faker yes, judge_not_below_presidio_faker yes, bertscore_not_below_gliner_pii yes, judge_not_below_gliner_pii no; strict, deciding nothing: bertscore_above_presidio_faker yes, judge_above_presidio_faker yes, bertscore_above_gliner_pii yes, judge_above_gliner_pii no)
- wildchat: **does not hold** (bertscore_not_below_presidio_faker yes, judge_not_below_presidio_faker yes, bertscore_not_below_gliner_pii yes, judge_not_below_gliner_pii no; strict, deciding nothing: bertscore_above_presidio_faker yes, judge_above_presidio_faker yes, bertscore_above_gliner_pii no, judge_above_gliner_pii no)
- all: **does not hold** (bertscore_not_below_presidio_faker yes, judge_not_below_presidio_faker yes, bertscore_not_below_gliner_pii yes, judge_not_below_gliner_pii no; strict, deciding nothing: bertscore_above_presidio_faker yes, judge_above_presidio_faker yes, bertscore_above_gliner_pii yes, judge_above_gliner_pii no)

Natural slice (nothing to protect): BERTScore against the answer to the untouched prompt.

| group | n | edited ss | edited presidio_default | edited gliner_pii | rerun | ss | presidio_default | gliner_pii | ss − rerun | presidio_default − rerun | gliner_pii − rerun | ss − presidio_default | ss − gliner_pii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oasst1 | 50 | 2/50 (4.0%, 1.1%–13.5%) | 16/50 (32.0%, 20.8%–45.8%) | 29/50 (58.0%, 44.2%–70.6%) | 0.5 | 0.4908 | 0.4045 | 0.3804 | -0.009 [-0.025, +0.000] | -0.096 [-0.161, -0.043] | -0.119 [-0.180, -0.066] | +0.086 [+0.037, +0.146] | +0.110 [+0.058, +0.171] |
| sharegpt | 50 | 12/50 (24.0%, 14.3%–37.4%) | 24/50 (48.0%, 34.8%–61.5%) | 43/50 (86.0%, 73.8%–93.0%) | 0.4958 | 0.4604 | 0.437 | 0.3129 | -0.035 [-0.066, -0.012] | -0.059 [-0.100, -0.023] | -0.183 [-0.254, -0.114] | +0.023 [-0.012, +0.066] | +0.147 [+0.079, +0.223] |
| wildchat | 50 | 9/50 (18.0%, 9.8%–30.8%) | 19/50 (38.0%, 25.9%–51.8%) | 36/50 (72.0%, 58.3%–82.5%) | 0.5288 | 0.518 | 0.4789 | 0.4024 | -0.011 [-0.026, +0.001] | -0.050 [-0.086, -0.018] | -0.126 [-0.197, -0.063] | +0.039 [+0.011, +0.069] | +0.116 [+0.051, +0.182] |
| all | 150 | 23/150 (15.3%, 10.4%–22.0%) | 59/150 (39.3%, 31.9%–47.3%) | 108/150 (72.0%, 64.3%–78.6%) | 0.5082 | 0.4897 | 0.4401 | 0.3652 | -0.018 [-0.031, -0.008] | -0.068 [-0.095, -0.043] | -0.143 [-0.181, -0.108] | +0.050 [+0.025, +0.076] | +0.124 [+0.088, +0.164] |
