# Off-the-shelf yardstick on dev (V3 §3.7)

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.yardstick --out bench/results/yardstick_dev.json`

git 04eb471. `ss_isotonic` is SS with `bench/arms/yardstick_isotonic.json` (sha256 c47056df8a6c) merged on the benchmark config; `gliner_pii_tuned` is `published` at 0.3.

Licences: model cc-by-nc-4.0; training data none declared. local non-commercial evaluation only (V3 §3.7); never a default, shipped, fine-tuned from, distilled or used to label data.

| arm | injected leak (95 %) | macro | message leak | natural spurious edits | negatives untouched | p50 ms |
|---|---|---|---|---|---|---|
| ss | 0.0255 (16/627; 0.016–0.041) | 0.0244 | 0.0795 (14/176; 0.048–0.129) | 385 of 490 | 0.7865 (339/431; 0.745–0.823) | 27.7 / 30.4 / 37.8 |
| ss_isotonic | 0.0207 (13/627; 0.012–0.035) | 0.017 | 0.0739 (13/176; 0.044–0.122) | 537 of 639 | 0.7425 (320/431; 0.699–0.781) | 80.6 / 80.7 / 98.8 |
| gliner_pii | 0.0941 (59/627; 0.074–0.119) | 0.109 | 0.2614 (46/176; 0.202–0.331) | 1069 of 1208 | 0.2668 (115/431; 0.227–0.310) |  |
| gliner_pii_tuned | 0.0781 (49/627; 0.060–0.102) | 0.0899 | 0.2386 (42/176; 0.182–0.307) | 1390 of 1530 | 0.1926 (83/431; 0.158–0.233) |  |

Injected values leaked by type (pooled):

| arm | ADDRESS | AGE | CREDENTIAL | DATE_OF_BIRTH | EMAIL | HANDLE | ID | LOCATION | NETWORK | ORG | PERSON | PHONE | URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ss | 0 | 1 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 5 | 3 | 1 | 1 |
| ss_isotonic | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 5 | 4 | 0 | 1 |
| gliner_pii | 3 | 0 | 3 | 3 | 1 | 4 | 15 | 0 | 6 | 7 | 1 | 0 | 16 |
| gliner_pii_tuned | 3 | 0 | 3 | 3 | 1 | 0 | 13 | 0 | 4 | 6 | 1 | 0 | 15 |
