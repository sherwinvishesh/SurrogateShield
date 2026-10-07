# Latency of every arm (E7)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms_v3.json` at commit `e2c8fa562643`

200 real-data dev messages (oasst1 67, sharegpt 67, wildchat 66), offline, one arm at a time on macOS-26.3-arm64-arm-64bit-Mach-O, 10 CPUs, 16.0 GB.

| arm | venv | warm p50 ms | warm p95 ms | mean ms | max ms | model load s | cold start s | peak RSS MB | refused |
|---|---|---|---|---|---|---|---|---|---|
| `ss` | .venv | 48.3 | 209.2 | 71.7 | 391.0 | 0.0 | 7.0 | 1352.4 | 0 |
| `presidio_default` | .venv | 8.1 | 50.4 | 13.9 | 79.0 | 0.0 | 4.6 | 1205.5 | 0 |
| `presidio_faker` | .venv | 8.3 | 49.5 | 13.9 | 71.5 | 3.5 | 4.7 | 1182.4 | 0 |
| `presidio_transformers` | .venv-baselines | 83.0 | 261.2 | 120.5 | 857.1 | 0.5 | 3.6 | 1784.9 | 0 |
| `llm_guard` | .venv-baselines | 57.8 | 192.6 | 77.1 | 481.6 | 1.0 | 4.2 | 1511.1 | 0 |
| `gliner_pii` | .venv-baselines | 80.3 | 346.4 | 110.7 | 530.6 | 10.0 | 11.1 | 2374.2 | 0 |
| `gliner_pii_tuned` | .venv-baselines | 80.6 | 335.5 | 111.7 | 660.5 | 10.3 | 11.0 | 2781.1 | 0 |
