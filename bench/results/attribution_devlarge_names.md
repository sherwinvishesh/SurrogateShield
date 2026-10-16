# Leak attribution — `devlarge` (data split `test`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split devlarge --out bench/results/attribution_devlarge_names.json` at commit `8c6a384f958f`. Counts only.

Traced edits vs the `ss` arm's span file: {'differ': 289, 'equal': 2319}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 135 | 0.0 | 135 | 0 | 0 | 0 |
| AGE | 134 | 0.0 | 134 | 0 | 0 | 0 |
| CREDENTIAL | 131 | 0.0305 | 127 | 0 | 1 | 3 |
| DATE_OF_BIRTH | 127 | 0.0 | 127 | 0 | 0 | 0 |
| EMAIL | 205 | 0.0 | 205 | 0 | 0 | 0 |
| HANDLE | 127 | 0.063 | 119 | 5 | 1 | 2 |
| ID | 174 | 0.023 | 170 | 3 | 0 | 1 |
| LOCATION | 142 | 0.0282 | 138 | 0 | 1 | 3 |
| NETWORK | 122 | 0.0082 | 121 | 1 | 0 | 0 |
| ORG | 179 | 0.0168 | 176 | 3 | 0 | 0 |
| PERSON | 599 | 0.0401 | 575 | 10 | 9 | 5 |
| PHONE | 166 | 0.0181 | 163 | 1 | 0 | 2 |
| URL | 127 | 0.0 | 127 | 0 | 0 | 0 |
| **all** | 2368 | 0.0215 | 2317 | 23 | 12 | 16 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 551 | 0.0145 | 543 | 6 | 1 | 1 |
| inline | 538 | 0.0149 | 530 | 5 | 2 | 1 |
| intro | 627 | 0.0048 | 624 | 2 | 1 | 0 |
| json | 96 | 0.0729 | 89 | 4 | 1 | 2 |
| signature | 556 | 0.045 | 531 | 6 | 7 | 12 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 1752 | 0.0188 | 1719 | 16 | 6 | 11 |
| multi | 315 | 0.0127 | 311 | 2 | 1 | 1 |
| shift | 301 | 0.0465 | 287 | 5 | 5 | 4 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 135 | 0.0 | 135 | 0 | 0 | 0 |
| AGE:plain | 118 | 0.0 | 118 | 0 | 0 | 0 |
| AGE:words | 16 | 0.0 | 16 | 0 | 0 | 0 |
| CREDENTIAL:hex | 30 | 0.0 | 30 | 0 | 0 | 0 |
| CREDENTIAL:password | 34 | 0.1176 | 30 | 0 | 1 | 3 |
| CREDENTIAL:secret | 37 | 0.0 | 37 | 0 | 0 | 0 |
| CREDENTIAL:zqk | 30 | 0.0 | 30 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 5 | 0.0 | 5 | 0 | 0 | 0 |
| DATE_OF_BIRTH:ordinal-of | 8 | 0.0 | 8 | 0 | 0 | 0 |
| DATE_OF_BIRTH:plain | 114 | 0.0 | 114 | 0 | 0 | 0 |
| EMAIL:plain | 193 | 0.0 | 193 | 0 | 0 | 0 |
| EMAIL:spelled | 12 | 0.0 | 12 | 0 | 0 | 0 |
| HANDLE:plain | 127 | 0.063 | 119 | 5 | 1 | 2 |
| ID:card | 36 | 0.0 | 36 | 0 | 0 | 0 |
| ID:card-spaced | 15 | 0.0 | 15 | 0 | 0 | 0 |
| ID:iban | 18 | 0.0 | 18 | 0 | 0 | 0 |
| ID:iban-spaced | 3 | 0.0 | 3 | 0 | 0 | 0 |
| ID:license | 19 | 0.0 | 19 | 0 | 0 | 0 |
| ID:nhs | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:nino | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:passport | 34 | 0.0 | 34 | 0 | 0 | 0 |
| ID:policy | 23 | 0.0435 | 22 | 0 | 0 | 1 |
| ID:ssn | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:ssn-nodash | 2 | 0.0 | 2 | 0 | 0 | 0 |
| ID:student | 14 | 0.2143 | 11 | 3 | 0 | 0 |
| LOCATION:plain | 142 | 0.0282 | 138 | 0 | 1 | 3 |
| NETWORK:ipv4 | 78 | 0.0 | 78 | 0 | 0 | 0 |
| NETWORK:ipv6 | 8 | 0.0 | 8 | 0 | 0 | 0 |
| NETWORK:mac | 36 | 0.0278 | 35 | 1 | 0 | 0 |
| ORG:plain | 179 | 0.0168 | 176 | 3 | 0 | 0 |
| PERSON:lower | 24 | 0.1667 | 20 | 1 | 1 | 2 |
| PERSON:plain | 521 | 0.023 | 509 | 5 | 6 | 1 |
| PERSON:surname-first | 32 | 0.125 | 28 | 4 | 0 | 0 |
| PERSON:upper | 22 | 0.1818 | 18 | 0 | 2 | 2 |
| PHONE:dots | 15 | 0.0 | 15 | 0 | 0 | 0 |
| PHONE:plain | 148 | 0.0203 | 145 | 1 | 0 | 2 |
| PHONE:words | 3 | 0.0 | 3 | 0 | 0 | 0 |
| URL:plain | 127 | 0.0 | 127 | 0 | 0 | 0 |

## Leak details (type → cause → detail: count)

- **CREDENTIAL** gated: `needs_confirmation <- slm` 1
- **CREDENTIAL** missed: `-` 3
- **HANDLE** gated: `topical_geo <- ner` 1
- **HANDLE** missed: `-` 2
- **HANDLE** partial: `prefix` 3, `inner` 1, `suffix` 1
- **ID** missed: `-` 1
- **ID** partial: `prefix` 3
- **LOCATION** gated: `relation_gate:untied <- ner` 1
- **LOCATION** missed: `-` 3
- **NETWORK** partial: `prefix` 1
- **ORG** partial: `inner` 2, `prefix` 1
- **PERSON** gated: `needs_confirmation <- slm` 4, `implausible_org <- slm` 1, `relation_gate:untied <- ner` 1, `implausible_org <- ner` 1, `relation_gate:junk <- pattern` 1, `relation_gate:public_person <- ner` 1
- **PERSON** missed: `-` 5
- **PERSON** partial: `last` 7, `first+middle` 1, `middle` 1, `first` 1
- **PHONE** missed: `-` 2
- **PHONE** partial: `prefix` 1

## Protected ADDRESS values whose surrogate repeats a part verbatim

`none` 135

## Natural slice: spurious edits by source

1739 messages, 1306 untouched, 1635 spurious edits.

| source:type | edits |
|---|---|
| repeat | 574 |
| ner:PERSON | 411 |
| ner:ORG | 161 |
| ner:GPE | 100 |
| structural:PERSON | 100 |
| slm:PERSON | 77 |
| pattern:age | 34 |
| pattern:PERSON | 33 |
| slm:LOC | 28 |
| pattern:dob | 25 |
| slm:ORG | 22 |
| pattern:handle | 14 |
| ner:LOC | 13 |
| pattern:url | 11 |
| structural:ORG | 10 |
| pattern:ORG | 4 |
| ner:FAC | 3 |
| structural:hostname | 3 |
| pattern:phone_intl | 3 |
| pattern:credential | 2 |
| pattern:worded:age | 2 |
| pattern:ip_address | 1 |
| pattern:id_number | 1 |
| structural:GPE | 1 |
| pattern:zip_us | 1 |
| pattern:address | 1 |
