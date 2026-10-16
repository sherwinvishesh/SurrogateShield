# Test-3 results

`PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict --split test3 --out bench/results/verdict_test3.json` at commit `c98579820508`; hypotheses `bench/realdata/HYPOTHESES_TEST3.md` (bd43f11d9b82), bench/realdata/test3/FREEZE.json 31d87c099d6c. Readings: `bench/realdata/verdict.py`.

| hypothesis | result | detail |
|---|---|---|
| H1''' | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H2''' | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H3''' | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H4''' | holds | shift leak SS 1/530, GLiNER-PII 44/530; Δ -8.1 [-10.6, -5.7] |
| H5''' | holds | failing types: none (tuned, beside: none) |
| H6''' | holds | p50 43.5 vs 82.1 ms; peak RSS 1348.2 vs 2318.5 MB |
| H7''' | holds | E5b BERTScore ss − Presidio+Faker +0.077 [+0.053, +0.103], ss − GLiNER-PII +0.072 [+0.046, +0.098] (judge beside +0.449 [+0.355, +0.540], +0.051 [-0.051, +0.152]); E7b judge +0.576 [+0.467, +0.685], +0.649 [+0.527, +0.770]; restored SS 81/82 vs 0/80, 0/54 |
| H8''' | holds | exact recovery SS 4/620 vs GLiNER-PII 6/556, the lowest of all arms (llm_guard 19/264, presidio_default 14/428, presidio_faker 8/428); conversation 1/177; 98 partials set aside |
| H9''' | holds | Presidio-default 1161/3677; no patterns 52/3677 (holds); no tagger 396/3677 (holds); beside: no patterns, default routing 1267/3677 |
| H10''' | holds | leak not above True, span F1 not below True |

Revised requirements:

- H7''': E5b read word by word: SS's BERTScore not below Presidio+Faker's or GLiNER-PII's (lower bound above −0.02), the judge's score beside; E7b: the judge's grade of SS above both (lower bound above 0) and the originals restored letter for letter in SS's replies more often than in theirs (Wilson 95 % intervals apart), in place of cross-turn consistency, which no sampled conversation can measure (no gold value recurs across turns).
- H8''': letter for letter: SS's exact recovery over the values it replaced not above GLiNER-PII's (pooled, point estimates; attacker_exact_test3.json); partial recoveries, parts the surrogates keep by design, counted beside.

Inputs:

- e5: `bench/results/realdata_test3.json` (59863bb21d0d, commit 8d4ea8b3a49c)
- components: `bench/results/realdata_components_test3.json` (c46dbf458458, commit 702001f5229f)
- perf: `bench/results/perf_arms_v4.json` (9d0bedd15ce8, commit b173311badad)
- utility: `bench/results/utility_realdata_test3.json` (69a71cefe58b, commit cb2b83353ebd)
- multiturn: `bench/results/multiturn_realdata_test3.json` (7bd9c04f6c28, commit 3fa246c52302)
- attacker: `bench/results/attacker_realdata_test3.json` (e9519ee3d5ae, commit d777625e9907)
- attacker_exact: `bench/results/attacker_exact_test3.json` (56708d438542, commit c98579820508)
- external: `bench/results/external_nemotron_pii.json` (02374546605a, commit 73098e3fabbe)
