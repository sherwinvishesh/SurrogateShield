# Presets and component ablations — dev

`PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.components --split dev --configs balanced no_tagger h9/no_patterns h9/no_patterns_unrouted --out bench/results/h9_ablation_dev.json`

Commit 91e1ff68d99d, library code c4e3e9d4d87d. Group `all`. Leak = protect values left in the text (injected slice); spurious = edits on natural text that touch no gold value. Δ = arm − balanced, paired cluster bootstrap 95 % CI (* excludes 0). The p50 is in-run (not quiet).

| arm | config hash | injected leak | Δ leak | shift leak | multi leak | natural spurious | Δ spurious rate | negatives untouched | in-run p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| `balanced` | a1abaa92142b0db9 | 9/627 1.4 % [0.8, 2.7] | – | 2/81 | 3/93 | 133/193 | – | 377/431 | 48.4 |
| `no_tagger` | cb25eb674717839e | 70/627 11.2 % [8.9, 13.9] | +9.7 [+7.5, +12.1] * | 14/81 | 16/93 | 101/133 | +7.0 [-1.7, +17.6] | 387/431 | 12.1 |
| `h9/no_patterns` | 18a8cd8e4a2def8c | 10/627 1.6 % [0.9, 2.9] | +0.2 [-0.6, +1.1] | 2/81 | 2/93 | 232/292 | +10.5 [+2.8, +19.7] * | 365/431 | 46.8 |
| `h9/no_patterns_unrouted` | 8ddcbca52249af08 | 208/627 33.2 % [29.6, 37.0] | +31.7 [+28.8, +34.7] * | 24/81 | 26/93 | 110/167 | -3.0 [-7.8, -0.1] * | 388/431 | 48.1 |
| `gliner_pii` | – | 59/627 9.4 % [7.4, 11.9] | +8.0 [+5.3, +10.5] * | 11/81 | 7/93 | 1069/1208 | +19.6 [+6.2, +34.9] * | 115/431 | – |
| `gliner_pii_tuned` | – | 49/627 7.8 % [6.0, 10.2] | +6.4 [+4.0, +8.7] * | 11/81 | 5/93 | 1390/1530 | +21.9 [+8.1, +37.6] * | 83/431 | – |

Leaked values by type (injected):

| arm | ADDRESS | AGE | CREDENTIAL | DATE_OF_BIRTH | EMAIL | HANDLE | ID | LOCATION | NETWORK | ORG | PERSON | PHONE | URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `balanced` | 0/30 | 1/33 | 0/32 | 0/35 | 0/55 | 0/37 | 0/53 | 0/32 | 0/37 | 3/44 | 3/152 | 1/47 | 1/40 |
| `no_tagger` | 0/30 | 1/33 | 0/32 | 0/35 | 0/55 | 0/37 | 4/53 | 5/32 | 0/37 | 14/44 | 44/152 | 1/47 | 1/40 |
| `h9/no_patterns` | 1/30 | 0/33 | 0/32 | 1/35 | 0/55 | 0/37 | 0/53 | 0/32 | 0/37 | 3/44 | 3/152 | 2/47 | 0/40 |
| `h9/no_patterns_unrouted` | 1/30 | 32/33 | 0/32 | 35/35 | 55/55 | 0/37 | 0/53 | 0/32 | 37/37 | 3/44 | 3/152 | 2/47 | 40/40 |
| `gliner_pii` | 3/30 | 0/33 | 3/32 | 3/35 | 1/55 | 4/37 | 15/53 | 0/32 | 6/37 | 7/44 | 1/152 | 0/47 | 16/40 |
| `gliner_pii_tuned` | 3/30 | 0/33 | 3/32 | 3/35 | 1/55 | 0/37 | 13/53 | 0/32 | 4/37 | 6/44 | 1/152 | 0/47 | 15/40 |
