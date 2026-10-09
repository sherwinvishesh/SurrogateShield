# Test-3 verdict: NO-GO

`PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict --split test3 --out bench/results/verdict_test3.json` at commit `e6d01369ab47`; pre-registration `bench/realdata/HYPOTHESES_TEST3.md` (bd43f11d9b82), bench/realdata/test3/FREEZE.json 31d87c099d6c. Readings: `bench/realdata/verdict.py`.

| hypothesis | in the GO rule | result | detail |
|---|---|---|---|
| H1''' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H2''' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H3''' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H4''' | no | holds | shift leak SS 1/530, GLiNER-PII 44/530; Δ -8.1 [-10.6, -5.7] |
| H5''' | yes | holds | failing types: none (tuned, beside: none) |
| H6''' | yes | holds | p50 43.5 vs 82.1 ms; peak RSS 1348.2 vs 2318.5 MB |
| H7''' | no | fails | E5b fails, E7b fails |
| H8''' | yes | fails | no exact (single) False, (conversation) False; recovery SS 102/620 vs GLiNER-PII 11/556 |
| H9''' | no | holds | Presidio-default 1161/3677; no patterns 52/3677 (holds); no tagger 396/3677 (holds); beside: no patterns, default routing 1267/3677 |
| H10''' | no | holds | leak not above True, span F1 not below True |

Inputs:

- e5: `bench/results/realdata_test3.json` (59863bb21d0d, commit 8d4ea8b3a49c)
- components: `bench/results/realdata_components_test3.json` (c46dbf458458, commit 702001f5229f)
- perf: `bench/results/perf_arms_v4.json` (9d0bedd15ce8, commit b173311badad)
- utility: `bench/results/utility_realdata_test3.json` (69a71cefe58b, commit cb2b83353ebd)
- multiturn: `bench/results/multiturn_realdata_test3.json` (7bd9c04f6c28, commit 3fa246c52302)
- attacker: `bench/results/attacker_realdata_test3.json` (e9519ee3d5ae, commit d777625e9907)
- external: `bench/results/external_nemotron_pii.json` (02374546605a, commit 73098e3fabbe)
