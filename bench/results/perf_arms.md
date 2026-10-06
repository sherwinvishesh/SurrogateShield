# Latency of every arm (E7)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms.json` at commit `230a470a24f8`

200 real-data dev messages (oasst1 67, sharegpt 67, wildchat 66), offline, one arm at a time on macOS-26.3-arm64-arm-64bit-Mach-O, 10 CPUs, 16.0 GB.

| arm | venv | warm p50 ms | warm p95 ms | mean ms | max ms | model load s | cold start s | peak RSS MB | refused |
|---|---|---|---|---|---|---|---|---|---|
| `ss` | .venv | 28.6 | 117.9 | 41.3 | 148.4 | 0.2 | 6.6 | 1082.7 | 0 |
| `presidio_default` | .venv | 8.5 | 50.6 | 14.2 | 73.2 | 0.0 | 4.7 | 1178.7 | 0 |
| `presidio_faker` | .venv | 8.4 | 49.9 | 14.2 | 73.1 | 3.5 | 4.8 | 1214.6 | 0 |
| `presidio_transformers` | .venv-baselines | 82.6 | 279.8 | 119.5 | 885.8 | 0.5 | 3.6 | 1576.2 | 0 |
| `llm_guard` | .venv-baselines | 56.6 | 181.4 | 75.6 | 500.3 | 1.0 | 4.2 | 1693.8 | 0 |
| `gliner_pii` | .venv-baselines | 79.5 | 329.9 | 109.0 | 518.8 | 10.0 | 11.0 | 2492.7 | 0 |
