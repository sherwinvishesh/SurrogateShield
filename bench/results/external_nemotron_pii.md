# External benchmark `nemotron_pii` (secondary check (H10'''), not used for tuning)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --external nemotron_pii --out bench/results/external_nemotron_pii.json` at commit `73098e3fabbe`

Source: nvidia/Nemotron-PII @ `b70ffaf5ff39` (CC-BY-4.0); 1000 records, strata {'intl-structured': 250, 'intl-unstructured': 250, 'us-structured': 250, 'us-unstructured': 250}. Same scorer as the real-data benchmark: leak = a protect value with a letter or digit reaching the provider; spurious = an edit touching no gold value; span F1 = value-level (precision 1 − spurious rate, recall 1 − leak rate). Δ = SS − arm, paired cluster bootstrap (2000 resamples); * = interval excludes 0.

## intl-structured (250 records, 1197 protect values)

| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |
|---|---|---|---|---|---|---|---|
| ss | 142 / 1197 | 11.9 % [10.2, 13.8] | – | 1624 | 9.4 % [8.0, 10.9] | – | 0.894 |
| presidio_default | 510 / 1197 | 42.6 % [39.8, 45.4] | -30.7 [-34.0, -27.7] * | 1530 | 13.5 % [11.9, 15.3] | -4.2 [-7.8, -0.5] * | 0.690 |
| presidio_faker | 510 / 1197 | 42.6 % [39.8, 45.4] | -30.7 [-34.0, -27.7] * | 1526 | 13.5 % [11.9, 15.3] | -4.1 [-7.7, -0.4] * | 0.690 |
| presidio_transformers | 700 / 1197 | 58.5 % [55.7, 61.2] | -46.6 [-49.9, -43.2] * | 1361 | 5.9 % [4.8, 7.3] | +3.4 [+0.6, +6.3] * | 0.576 |
| llm_guard | 696 / 1197 | 58.1 % [55.3, 60.9] | -46.3 [-49.4, -43.3] * | 1212 | 22.1 % [19.9, 24.5] | -12.8 [-17.3, -8.6] * | 0.544 |
| gliner_pii | 316 / 1197 | 26.4 % [24.0, 29.0] | -14.5 [-17.8, -11.3] * | 1590 | 26.6 % [24.5, 28.8] | -17.2 [-22.4, -12.1] * | 0.735 |
| gliner_pii_tuned | 274 / 1197 | 22.9 % [20.6, 25.4] | -11.0 [-14.0, -8.1] * | 1816 | 31.4 % [29.3, 33.6] | -22.0 [-26.6, -17.4] * | 0.726 |

## intl-unstructured (250 records, 1151 protect values)

| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |
|---|---|---|---|---|---|---|---|
| ss | 144 / 1151 | 12.5 % [10.7, 14.5] | – | 1203 | 4.4 % [3.4, 5.7] | – | 0.914 |
| presidio_default | 420 / 1151 | 36.5 % [33.8, 39.3] | -24.0 [-27.2, -21.0] * | 1243 | 6.7 % [5.4, 8.2] | -2.3 [-4.7, -0.0] * | 0.756 |
| presidio_faker | 420 / 1151 | 36.5 % [33.8, 39.3] | -24.0 [-27.2, -21.0] * | 1238 | 6.6 % [5.4, 8.2] | -2.2 [-4.6, +0.1] | 0.756 |
| presidio_transformers | 625 / 1151 | 54.3 % [51.4, 57.2] | -41.8 [-45.0, -38.8] * | 1122 | 4.3 % [3.2, 5.6] | +0.1 [-2.1, +2.2] | 0.619 |
| llm_guard | 691 / 1151 | 60.0 % [57.2, 62.8] | -47.5 [-50.7, -44.7] * | 675 | 2.4 % [1.5, 3.8] | +2.0 [-0.4, +4.3] | 0.567 |
| gliner_pii | 251 / 1151 | 21.8 % [19.5, 24.3] | -9.3 [-12.2, -6.6] * | 1227 | 20.1 % [17.9, 22.4] | -15.6 [-20.2, -11.2] * | 0.791 |
| gliner_pii_tuned | 224 / 1151 | 19.5 % [17.3, 21.9] | -7.0 [-9.7, -4.3] * | 1377 | 23.8 % [21.6, 26.1] | -19.4 [-24.1, -14.8] * | 0.783 |

## us-structured (250 records, 1151 protect values)

| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |
|---|---|---|---|---|---|---|---|
| ss | 195 / 1151 | 16.9 % [14.9, 19.2] | – | 1477 | 11.0 % [9.5, 12.7] | – | 0.859 |
| presidio_default | 527 / 1151 | 45.8 % [42.9, 48.7] | -28.8 [-32.3, -25.1] * | 1515 | 14.3 % [12.6, 16.1] | -3.3 [-7.4, +0.5] | 0.664 |
| presidio_faker | 528 / 1151 | 45.9 % [43.0, 48.8] | -28.9 [-32.4, -25.2] * | 1508 | 14.2 % [12.5, 16.0] | -3.2 [-7.4, +0.7] | 0.664 |
| presidio_transformers | 691 / 1151 | 60.0 % [57.2, 62.8] | -43.1 [-46.4, -39.8] * | 1350 | 7.3 % [6.1, 8.8] | +3.6 [+0.1, +7.3] * | 0.558 |
| llm_guard | 720 / 1151 | 62.5 % [59.7, 65.3] | -45.6 [-48.8, -42.4] * | 1050 | 30.9 % [28.2, 33.8] | -20.0 [-25.8, -14.2] * | 0.486 |
| gliner_pii | 364 / 1151 | 31.6 % [29.0, 34.4] | -14.7 [-17.9, -11.6] * | 1477 | 23.5 % [21.4, 25.7] | -12.5 [-17.3, -7.9] * | 0.722 |
| gliner_pii_tuned | 303 / 1151 | 26.3 % [23.9, 28.9] | -9.4 [-12.3, -6.6] * | 1763 | 28.8 % [26.7, 30.9] | -17.8 [-22.5, -13.1] * | 0.724 |

## us-unstructured (250 records, 1128 protect values)

| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |
|---|---|---|---|---|---|---|---|
| ss | 189 / 1128 | 16.8 % [14.7, 19.1] | – | 1168 | 6.4 % [5.1, 8.0] | – | 0.881 |
| presidio_default | 399 / 1128 | 35.4 % [32.6, 38.2] | -18.6 [-21.8, -15.5] * | 1200 | 7.7 % [6.3, 9.3] | -1.2 [-4.8, +2.4] | 0.760 |
| presidio_faker | 402 / 1128 | 35.6 % [32.9, 38.5] | -18.9 [-22.1, -15.7] * | 1200 | 7.7 % [6.3, 9.3] | -1.2 [-4.8, +2.4] | 0.758 |
| presidio_transformers | 605 / 1128 | 53.6 % [50.7, 56.5] | -36.9 [-40.4, -33.3] * | 1114 | 5.3 % [4.1, 6.8] | +1.1 [-1.9, +4.5] | 0.623 |
| llm_guard | 713 / 1128 | 63.2 % [60.4, 66.0] | -46.5 [-49.3, -43.6] * | 617 | 3.4 % [2.2, 5.1] | +3.0 [-1.8, +7.5] | 0.533 |
| gliner_pii | 263 / 1128 | 23.3 % [20.9, 25.9] | -6.6 [-9.5, -3.6] * | 1207 | 18.2 % [16.2, 20.5] | -11.8 [-16.1, -7.7] * | 0.791 |
| gliner_pii_tuned | 240 / 1128 | 21.3 % [19.0, 23.8] | -4.5 [-7.4, -1.7] * | 1341 | 22.5 % [20.4, 24.8] | -16.1 [-20.4, -11.9] * | 0.781 |

## all (1000 records, 4627 protect values)

| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |
|---|---|---|---|---|---|---|---|
| ss | 670 / 4627 | 14.5 % [13.5, 15.5] | – | 5472 | 8.1 % [7.4, 8.8] | – | 0.886 |
| presidio_default | 1856 / 4627 | 40.1 % [38.7, 41.5] | -25.6 [-27.3, -24.0] * | 5488 | 10.9 % [10.1, 11.8] | -2.8 [-4.6, -0.9] * | 0.716 |
| presidio_faker | 1860 / 4627 | 40.2 % [38.8, 41.6] | -25.7 [-27.4, -24.1] * | 5472 | 10.9 % [10.1, 11.7] | -2.8 [-4.6, -0.9] * | 0.716 |
| presidio_transformers | 2621 / 4627 | 56.6 % [55.2, 58.1] | -42.2 [-43.9, -40.4] * | 4947 | 5.8 % [5.2, 6.5] | +2.3 [+0.8, +4.0] * | 0.594 |
| llm_guard | 2820 / 4627 | 61.0 % [59.5, 62.3] | -46.5 [-48.0, -45.0] * | 3554 | 17.7 % [16.5, 19.0] | -9.7 [-12.4, -7.0] * | 0.530 |
| gliner_pii | 1194 / 4627 | 25.8 % [24.6, 27.1] | -11.3 [-12.8, -9.8] * | 5501 | 22.5 % [21.4, 23.6] | -14.4 [-17.0, -12.0] * | 0.758 |
| gliner_pii_tuned | 1041 / 4627 | 22.5 % [21.3, 23.7] | -8.0 [-9.5, -6.6] * | 6297 | 27.1 % [26.0, 28.2] | -19.0 [-21.5, -16.7] * | 0.751 |

## all — leak by type (leaked / values)

| type | ss | presidio_default | presidio_faker | presidio_transformers | llm_guard | gliner_pii | gliner_pii_tuned |
|---|---|---|---|---|---|---|---|
| ADDRESS | 15 / 167 | 141 / 167 | 141 / 167 | 129 / 167 | 167 / 167 | 7 / 167 | 5 / 167 |
| AGE | 14 / 66 | 22 / 66 | 22 / 66 | 37 / 66 | 66 / 66 | 1 / 66 | 1 / 66 |
| CREDENTIAL | 118 / 250 | 234 / 250 | 234 / 250 | 243 / 250 | 234 / 250 | 126 / 250 | 114 / 250 |
| DATE_OF_BIRTH | 0 / 174 | 0 / 174 | 0 / 174 | 0 / 174 | 174 / 174 | 0 / 174 | 0 / 174 |
| EMAIL | 5 / 444 | 5 / 444 | 5 / 444 | 5 / 444 | 4 / 444 | 18 / 444 | 18 / 444 |
| HANDLE | 21 / 102 | 84 / 102 | 84 / 102 | 94 / 102 | 86 / 102 | 1 / 102 | 1 / 102 |
| ID | 19 / 1088 | 735 / 1088 | 735 / 1088 | 765 / 1088 | 812 / 1088 | 567 / 1088 | 500 / 1088 |
| LOCATION | 106 / 495 | 115 / 495 | 115 / 495 | 411 / 495 | 495 / 495 | 138 / 495 | 98 / 495 |
| NETWORK | 1 / 129 | 15 / 129 | 15 / 129 | 51 / 129 | 40 / 129 | 41 / 129 | 37 / 129 |
| ORG | 50 / 265 | 259 / 265 | 259 / 265 | 212 / 265 | 264 / 265 | 3 / 265 | 1 / 265 |
| PERSON | 48 / 890 | 160 / 890 | 164 / 890 | 594 / 890 | 161 / 890 | 125 / 890 | 103 / 890 |
| PHONE | 0 / 245 | 22 / 245 | 22 / 245 | 15 / 245 | 5 / 245 | 10 / 245 | 6 / 245 |
| URL | 273 / 312 | 64 / 312 | 64 / 312 | 65 / 312 | 312 / 312 | 157 / 312 | 157 / 312 |

## H10''' (pooled; point estimates)

| GLiNER arm | SS leak ≤ | SS span F1 ≥ | holds |
|---|---|---|---|
| gliner_pii | yes | yes | yes |
| gliner_pii_tuned | yes | yes | yes |

## Label map (Nemotron-PII label → J2 list, type)

| label | list | type |
|---|---|---|
| bank_routing_number | optional | – |
| coordinate | optional | – |
| date | optional | – |
| date_time | optional | – |
| education_level | optional | – |
| employment_status | optional | – |
| gender | optional | – |
| language | optional | – |
| occupation | optional | – |
| postcode | optional | – |
| swift_bic | optional | – |
| time | optional | – |
| street_address | protect | ADDRESS |
| age | protect | AGE |
| api_key | protect | CREDENTIAL |
| cvv | protect | CREDENTIAL |
| http_cookie | protect | CREDENTIAL |
| password | protect | CREDENTIAL |
| pin | protect | CREDENTIAL |
| date_of_birth | protect | DATE_OF_BIRTH |
| email | protect | EMAIL |
| user_name | protect | HANDLE |
| account_number | protect | ID |
| biometric_identifier | protect | ID |
| certificate_license_number | protect | ID |
| credit_debit_card | protect | ID |
| customer_id | protect | ID |
| device_identifier | protect | ID |
| employee_id | protect | ID |
| health_plan_beneficiary_number | protect | ID |
| license_plate | protect | ID |
| medical_record_number | protect | ID |
| national_id | protect | ID |
| ssn | protect | ID |
| tax_id | protect | ID |
| unique_id | protect | ID |
| vehicle_identifier | protect | ID |
| city | protect | LOCATION |
| country | protect | LOCATION |
| county | protect | LOCATION |
| state | protect | LOCATION |
| ipv4 | protect | NETWORK |
| ipv6 | protect | NETWORK |
| mac_address | protect | NETWORK |
| company_name | protect | ORG |
| first_name | protect | PERSON |
| last_name | protect | PERSON |
| fax_number | protect | PHONE |
| phone_number | protect | PHONE |
| url | protect | URL |
| race_ethnicity | sensitive | ETHNICITY |
| blood_type | sensitive | HEALTH |
| sexuality | sensitive | ORIENTATION |
| political_view | sensitive | POLITICAL |
| religious_belief | sensitive | RELIGION |

Values dropped (not a whole-word substring of their text): 34 {'date': 3, 'education_level': 1, 'email': 2, 'employment_status': 3, 'gender': 3, 'occupation': 7, 'political_view': 5, 'race_ethnicity': 9, 'user_name': 1}; values given labels of different lists or types (first kept): 0.
