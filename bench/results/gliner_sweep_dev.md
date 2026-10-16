# GLiNER-PII sweep on dev (V3 §3.8)

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.gliner_sweep --out bench/results/gliner_sweep_dev.json`

git 4d32bf0; model `urchade/gliner_multi_pii-v1`; one run per label set at the lowest threshold with window-span scores; each threshold keeps score > threshold (float32), then gliner_pii's overlap rule and [LABEL] edits.

Filter check (published@0.5 vs the committed arm): oasst1 215/215, sharegpt 229/229, wildchat 218/218 messages equal.

**Chosen: `published` at 0.3** (lowest pooled injected leak rate; ties: fewer natural spurious edits, then higher threshold).

| label set | threshold | injected leak (95 %) | macro | message leak | natural spurious edits | negatives untouched |
|---|---|---|---|---|---|---|
| **published** | 0.3 | 0.0781 (49/627; 0.060–0.102) | 0.0899 | 0.2386 (42/176; 0.182–0.307) | 1390 of 1530 | 0.1926 (83/431; 0.158–0.233) |
| published | 0.4 | 0.0861 (54/627; 0.067–0.111) | 0.0987 | 0.2500 (44/176; 0.192–0.319) | 1231 of 1370 | 0.2343 (101/431; 0.197–0.277) |
| published | 0.5 | 0.0941 (59/627; 0.074–0.119) | 0.109 | 0.2614 (46/176; 0.202–0.331) | 1069 of 1208 | 0.2668 (115/431; 0.227–0.310) |
| published | 0.6 | 0.1085 (68/627; 0.086–0.135) | 0.125 | 0.2784 (49/176; 0.217–0.349) | 915 of 1053 | 0.2970 (128/431; 0.256–0.342) |
| ours | 0.3 | 0.1356 (85/627; 0.111–0.165) | 0.1417 | 0.3352 (59/176; 0.270–0.408) | 1708 of 1847 | 0.1531 (66/431; 0.122–0.190) |
| ours | 0.4 | 0.1499 (94/627; 0.124–0.180) | 0.1566 | 0.3580 (63/176; 0.291–0.431) | 1462 of 1601 | 0.1995 (86/431; 0.165–0.240) |
| ours | 0.5 | 0.1627 (102/627; 0.136–0.194) | 0.1691 | 0.3920 (69/176; 0.323–0.466) | 1278 of 1414 | 0.2413 (104/431; 0.203–0.284) |
| ours | 0.6 | 0.1882 (118/627; 0.160–0.221) | 0.1953 | 0.4432 (78/176; 0.372–0.517) | 1093 of 1229 | 0.2761 (119/431; 0.236–0.320) |

Injected values leaked by type (pooled):

| label set | threshold | ADDRESS | AGE | CREDENTIAL | DATE_OF_BIRTH | EMAIL | HANDLE | ID | LOCATION | NETWORK | ORG | PERSON | PHONE | URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| published | 0.3 | 3 | 0 | 3 | 3 | 1 | 0 | 13 | 0 | 4 | 6 | 1 | 0 | 15 |
| published | 0.4 | 3 | 0 | 3 | 3 | 1 | 1 | 15 | 0 | 5 | 7 | 1 | 0 | 15 |
| published | 0.5 | 3 | 0 | 3 | 3 | 1 | 4 | 15 | 0 | 6 | 7 | 1 | 0 | 16 |
| published | 0.6 | 3 | 0 | 3 | 3 | 2 | 6 | 17 | 0 | 6 | 8 | 1 | 0 | 19 |
| ours | 0.3 | 3 | 0 | 3 | 2 | 4 | 0 | 45 | 0 | 3 | 8 | 1 | 0 | 16 |
| ours | 0.4 | 3 | 0 | 3 | 3 | 4 | 0 | 50 | 0 | 4 | 9 | 1 | 1 | 16 |
| ours | 0.5 | 3 | 0 | 3 | 3 | 6 | 0 | 53 | 0 | 4 | 10 | 1 | 2 | 17 |
| ours | 0.6 | 3 | 0 | 3 | 3 | 14 | 1 | 53 | 0 | 5 | 14 | 1 | 2 | 19 |

Per dataset (injected leak):

| label set | threshold | oasst1 | sharegpt | wildchat |
|---|---|---|---|---|
| published | 0.3 | 0.1000 (21/210; 0.066–0.148) | 0.0577 (12/208; 0.033–0.098) | 0.0766 (16/209; 0.048–0.121) |
| published | 0.4 | 0.1190 (25/210; 0.082–0.170) | 0.0577 (12/208; 0.033–0.098) | 0.0813 (17/209; 0.051–0.126) |
| published | 0.5 | 0.1286 (27/210; 0.090–0.181) | 0.0625 (13/208; 0.037–0.104) | 0.0909 (19/209; 0.059–0.138) |
| published | 0.6 | 0.1381 (29/210; 0.098–0.191) | 0.0769 (16/208; 0.048–0.121) | 0.1100 (23/209; 0.074–0.160) |
| ours | 0.3 | 0.1524 (32/210; 0.110–0.207) | 0.0962 (20/208; 0.063–0.144) | 0.1579 (33/209; 0.115–0.213) |
| ours | 0.4 | 0.1667 (35/210; 0.122–0.223) | 0.1106 (23/208; 0.075–0.161) | 0.1722 (36/209; 0.127–0.229) |
| ours | 0.5 | 0.1714 (36/210; 0.127–0.228) | 0.1346 (28/208; 0.095–0.188) | 0.1818 (38/209; 0.135–0.240) |
| ours | 0.6 | 0.2048 (43/210; 0.156–0.264) | 0.1538 (32/208; 0.111–0.209) | 0.2057 (43/209; 0.157–0.266) |
