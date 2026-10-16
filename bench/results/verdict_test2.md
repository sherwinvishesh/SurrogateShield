# Test-2 verdict: incomplete

`PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict --out bench/results/verdict_test2.json` at commit `e2c8fa562643`; pre-registration `bench/realdata/HYPOTHESES_TEST2.md` (17b31a4b876e), FREEZE.json 518108b53f46. Readings: `bench/realdata/verdict.py`.

| hypothesis | in the GO rule | result | detail |
|---|---|---|---|
| H1'' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H2'' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H3'' | yes | holds | oasst1: holds; sharegpt: holds; wildchat: holds |
| H4'' | no | holds | shift leak SS 2/546, GLiNER-PII 48/546; Δ -8.4 [-10.9, -5.9] |
| H5'' | yes | fails | failing types: AGE (tuned, beside: AGE) |
| H6'' | yes | holds | p50 48.3 vs 80.3 ms; peak RSS 1352.4 vs 2374.2 MB |
| H7'' | no | not run | utility or multiturn not run |
| H8'' | yes | not run | not run |
| H9'' | no | not run | not run |
| H10'' | no | not run | not run |

Inputs:

- e5: `bench/results/realdata_test2.json` (dbf5e21692ae, commit e2c8fa562643)
- components: not run
- perf: `bench/results/perf_arms_v3.json` (72eeebfd9d40, commit e2c8fa562643)
- utility: not run
- multiturn: not run
- attacker: not run
- external: not run
