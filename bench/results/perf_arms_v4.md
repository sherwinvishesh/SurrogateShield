# Latency of every arm (E7)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms_v4.json` at commit `b173311badad`

200 real-data dev messages (oasst1 67, sharegpt 67, wildchat 66), offline, one arm at a time on macOS-26.3-arm64-arm-64bit-Mach-O, 10 CPUs, 16.0 GB.

| arm | venv | warm p50 ms | warm p95 ms | mean ms | max ms | model load s | cold start s | peak RSS MB | refused |
|---|---|---|---|---|---|---|---|---|---|
| `ss` | .venv | 43.5 | 204.4 | 65.9 | 392.4 | 0.0 | 7.2 | 1348.2 | 0 |
| `presidio_default` | .venv | 8.2 | 49.5 | 13.8 | 71.6 | 0.0 | 4.7 | 1083.7 | 0 |
| `presidio_faker` | .venv | 8.4 | 50.3 | 14.1 | 72.7 | 3.6 | 4.8 | 1009.6 | 0 |
| `presidio_transformers` | .venv-baselines | 83.5 | 259.5 | 120.5 | 857.3 | 0.5 | 3.9 | 1610.7 | 0 |
| `llm_guard` | .venv-baselines | 58.0 | 198.9 | 78.4 | 517.3 | 1.0 | 4.2 | 1564.8 | 0 |
| `gliner_pii` | .venv-baselines | 82.1 | 340.9 | 113.8 | 540.2 | 10.2 | 11.0 | 2318.5 | 0 |
| `gliner_pii_tuned` | .venv-baselines | 79.9 | 342.1 | 109.9 | 532.3 | 10.3 | 11.3 | 2332.7 | 0 |
