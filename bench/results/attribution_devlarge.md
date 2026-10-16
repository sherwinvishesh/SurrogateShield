# Leak attribution — `devlarge` (data split `test`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split devlarge --out bench/results/attribution_devlarge.json` at commit `3e3ae7c27ed1`. Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 2608}.

## Injected slice: cause by type

| type | values | leak | protected | refused | partial | gated | missed |
|---|---|---|---|---|---|---|---|
| ADDRESS | 135 | 0.5672 | 58 | 1 | 74 | 1 | 1 |
| AGE | 134 | 0.1194 | 118 | 0 | 0 | 0 | 16 |
| CREDENTIAL | 131 | 0.145 | 112 | 0 | 1 | 4 | 14 |
| DATE_OF_BIRTH | 127 | 0.1032 | 113 | 1 | 0 | 0 | 13 |
| EMAIL | 205 | 0.0588 | 192 | 1 | 6 | 1 | 5 |
| HANDLE | 127 | 0.063 | 119 | 0 | 5 | 0 | 3 |
| ID | 174 | 0.1494 | 148 | 0 | 0 | 0 | 26 |
| LOCATION | 142 | 0.0352 | 137 | 0 | 0 | 2 | 3 |
| NETWORK | 122 | 0.0082 | 121 | 0 | 1 | 0 | 0 |
| ORG | 179 | 0.2514 | 134 | 0 | 38 | 7 | 0 |
| PERSON | 599 | 0.0568 | 565 | 0 | 13 | 15 | 6 |
| PHONE | 166 | 0.0663 | 155 | 0 | 3 | 0 | 8 |
| URL | 127 | 0.1024 | 114 | 0 | 0 | 0 | 13 |
| **all** | 2368 | 0.118 | 2086 | 3 | 141 | 30 | 108 |

## Cause by layout

| layout | values | leak | protected | refused | partial | gated | missed |
|---|---|---|---|---|---|---|---|
| form | 551 | 0.1125 | 489 | 0 | 42 | 5 | 15 |
| inline | 538 | 0.1009 | 481 | 3 | 30 | 6 | 18 |
| intro | 627 | 0.0861 | 573 | 0 | 24 | 5 | 25 |
| json | 96 | 0.3021 | 67 | 0 | 11 | 1 | 17 |
| signature | 556 | 0.1439 | 476 | 0 | 34 | 13 | 33 |

## Cause by slice

| slice | values | leak | protected | refused | partial | gated | missed |
|---|---|---|---|---|---|---|---|
| injected_single | 1752 | 0.0869 | 1597 | 3 | 89 | 15 | 48 |
| multi | 315 | 0.1111 | 280 | 0 | 17 | 9 | 9 |
| shift | 301 | 0.3056 | 209 | 0 | 35 | 6 | 51 |

## Cause by type:format

| type:format | values | leak | protected | refused | partial | gated | missed |
|---|---|---|---|---|---|---|---|
| ADDRESS:plain | 135 | 0.5672 | 58 | 1 | 74 | 1 | 1 |
| AGE:plain | 118 | 0.0 | 118 | 0 | 0 | 0 | 0 |
| AGE:words | 16 | 1.0 | 0 | 0 | 0 | 0 | 16 |
| CREDENTIAL:hex | 30 | 0.0 | 30 | 0 | 0 | 0 | 0 |
| CREDENTIAL:password | 34 | 0.3529 | 22 | 0 | 1 | 3 | 8 |
| CREDENTIAL:secret | 37 | 0.1892 | 30 | 0 | 0 | 1 | 6 |
| CREDENTIAL:zqk | 30 | 0.0 | 30 | 0 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 5 | 1.0 | 0 | 0 | 0 | 0 | 5 |
| DATE_OF_BIRTH:ordinal-of | 8 | 1.0 | 0 | 0 | 0 | 0 | 8 |
| DATE_OF_BIRTH:plain | 114 | 0.0 | 113 | 1 | 0 | 0 | 0 |
| EMAIL:plain | 193 | 0.0 | 192 | 1 | 0 | 0 | 0 |
| EMAIL:spelled | 12 | 1.0 | 0 | 0 | 6 | 1 | 5 |
| HANDLE:plain | 127 | 0.063 | 119 | 0 | 5 | 0 | 3 |
| ID:card | 36 | 0.0 | 36 | 0 | 0 | 0 | 0 |
| ID:card-spaced | 15 | 0.0 | 15 | 0 | 0 | 0 | 0 |
| ID:iban | 18 | 0.0 | 18 | 0 | 0 | 0 | 0 |
| ID:iban-spaced | 3 | 0.0 | 3 | 0 | 0 | 0 | 0 |
| ID:license | 19 | 0.4211 | 11 | 0 | 0 | 0 | 8 |
| ID:nhs | 1 | 0.0 | 1 | 0 | 0 | 0 | 0 |
| ID:nino | 1 | 0.0 | 1 | 0 | 0 | 0 | 0 |
| ID:passport | 34 | 0.0 | 34 | 0 | 0 | 0 | 0 |
| ID:policy | 23 | 0.2174 | 18 | 0 | 0 | 0 | 5 |
| ID:ssn | 8 | 0.0 | 8 | 0 | 0 | 0 | 0 |
| ID:ssn-nodash | 2 | 0.0 | 2 | 0 | 0 | 0 | 0 |
| ID:student | 14 | 0.9286 | 1 | 0 | 0 | 0 | 13 |
| LOCATION:plain | 142 | 0.0352 | 137 | 0 | 0 | 2 | 3 |
| NETWORK:ipv4 | 78 | 0.0 | 78 | 0 | 0 | 0 | 0 |
| NETWORK:ipv6 | 8 | 0.0 | 8 | 0 | 0 | 0 | 0 |
| NETWORK:mac | 36 | 0.0278 | 35 | 0 | 1 | 0 | 0 |
| ORG:plain | 179 | 0.2514 | 134 | 0 | 38 | 7 | 0 |
| PERSON:lower | 24 | 0.1667 | 20 | 0 | 1 | 1 | 2 |
| PERSON:plain | 521 | 0.0384 | 501 | 0 | 7 | 12 | 1 |
| PERSON:surname-first | 32 | 0.1562 | 27 | 0 | 5 | 0 | 0 |
| PERSON:upper | 22 | 0.2273 | 17 | 0 | 0 | 2 | 3 |
| PHONE:dots | 15 | 0.2667 | 11 | 0 | 2 | 0 | 2 |
| PHONE:plain | 148 | 0.027 | 144 | 0 | 1 | 0 | 3 |
| PHONE:words | 3 | 1.0 | 0 | 0 | 0 | 0 | 3 |
| URL:plain | 127 | 0.1024 | 114 | 0 | 0 | 0 | 13 |

## Leak details (type → cause → detail: count)

- **ADDRESS** gated: `service_query_geo <- ner` 1
- **ADDRESS** missed: `-` 1
- **ADDRESS** partial: `house_number` 13, `house_number+street` 9, `locality+postcode` 7, `house_number+locality` 7, `house_number+locality+postcode` 7, `postcode` 6, `house_number+locality+street` 5, `house_number+postcode` 4, `unit` 3, `house_number+locality+street+unit` 3, `locality` 1, `street+unit` 1, `locality+unit` 1, `house_number+locality+postcode+street` 1, `house_number+locality+postcode+unit` 1, `house_number+unit` 1, `postcode+street` 1, `house_number+locality+postcode+street+unit` 1, `street` 1, `house_number+locality+unit` 1
- **AGE** missed: `-` 16
- **CREDENTIAL** gated: `relation_gate:junk <- ner` 3, `needs_confirmation <- slm` 1
- **CREDENTIAL** missed: `-` 14
- **CREDENTIAL** partial: `suffix` 1
- **DATE_OF_BIRTH** missed: `-` 13
- **EMAIL** gated: `implausible_org <- ner` 1
- **EMAIL** missed: `-` 5
- **EMAIL** partial: `domain+local` 5, `local` 1
- **HANDLE** missed: `-` 3
- **HANDLE** partial: `prefix` 3, `inner` 1, `suffix` 1
- **ID** missed: `-` 26
- **LOCATION** gated: `relation_gate:untied <- ner` 2
- **LOCATION** missed: `-` 3
- **NETWORK** partial: `prefix` 1
- **ORG** gated: `implausible_org <- ner` 3, `needs_confirmation <- slm` 2, `implausible_org <- slm` 1, `relation_gate:untied <- ner` 1
- **ORG** partial: `inner` 29, `suffix` 9
- **PERSON** gated: `needs_confirmation <- slm` 5, `relation_gate:untied <- ner` 4, `relation_gate:untied <- slm` 2, `implausible_org <- slm` 1, `implausible_org <- ner` 1, `relation_gate:junk <- pattern` 1, `relation_gate:public_person <- ner` 1
- **PERSON** missed: `-` 6
- **PERSON** partial: `last` 8, `middle` 2, `first+last` 1, `first+middle` 1, `first` 1
- **PHONE** missed: `-` 8
- **PHONE** partial: `suffix` 2, `prefix` 1
- **URL** missed: `in_url` 13

## Natural slice: spurious edits by source

1739 messages, 1310 untouched, 1613 spurious edits.

| source:type | edits |
|---|---|
| repeat | 563 |
| ner:PERSON | 416 |
| ner:ORG | 169 |
| ner:GPE | 101 |
| structural:PERSON | 93 |
| slm:PERSON | 73 |
| pattern:PERSON | 34 |
| pattern:age | 34 |
| slm:LOC | 27 |
| pattern:dob | 25 |
| slm:ORG | 21 |
| pattern:handle | 14 |
| ner:LOC | 13 |
| pattern:url | 10 |
| pattern:ORG | 4 |
| ner:FAC | 3 |
| structural:hostname | 3 |
| pattern:phone_intl | 3 |
| pattern:credential | 2 |
| pattern:ip_address | 1 |
| pattern:id_number | 1 |
| structural:GPE | 1 |
| pattern:zip_us | 1 |
| pattern:address | 1 |
