# Leak attribution — `devlarge` (data split `test`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split devlarge --out bench/results/attribution_devlarge_tagger.json` at commit `91e1ff68d99d` with 8 modified tracked file(s). Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 2608}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 135 | 0.0 | 135 | 0 | 0 | 0 |
| AGE | 134 | 0.0 | 134 | 0 | 0 | 0 |
| CREDENTIAL | 131 | 0.0 | 131 | 0 | 0 | 0 |
| DATE_OF_BIRTH | 127 | 0.0 | 127 | 0 | 0 | 0 |
| EMAIL | 205 | 0.0 | 205 | 0 | 0 | 0 |
| HANDLE | 127 | 0.0 | 127 | 0 | 0 | 0 |
| ID | 174 | 0.0057 | 173 | 1 | 0 | 0 |
| LOCATION | 142 | 0.007 | 141 | 0 | 1 | 0 |
| NETWORK | 122 | 0.0 | 122 | 0 | 0 | 0 |
| ORG | 179 | 0.0335 | 173 | 2 | 2 | 2 |
| PERSON | 599 | 0.005 | 596 | 0 | 2 | 1 |
| PHONE | 166 | 0.0 | 166 | 0 | 0 | 0 |
| URL | 127 | 0.0 | 127 | 0 | 0 | 0 |
| **all** | 2368 | 0.0046 | 2357 | 3 | 5 | 3 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 551 | 0.0036 | 549 | 1 | 1 | 0 |
| inline | 538 | 0.0093 | 533 | 0 | 4 | 1 |
| intro | 627 | 0.0032 | 625 | 1 | 0 | 1 |
| json | 96 | 0.0 | 96 | 0 | 0 | 0 |
| signature | 556 | 0.0036 | 554 | 1 | 0 | 1 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 1752 | 0.0034 | 1746 | 2 | 3 | 1 |
| multi | 315 | 0.0095 | 312 | 0 | 2 | 1 |
| shift | 301 | 0.0066 | 299 | 1 | 0 | 1 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 135 | 0.0 | 135 | 0 | 0 | 0 |
| AGE:plain | 118 | 0.0 | 118 | 0 | 0 | 0 |
| AGE:words | 16 | 0.0 | 16 | 0 | 0 | 0 |
| CREDENTIAL:hex | 30 | 0.0 | 30 | 0 | 0 | 0 |
| CREDENTIAL:password | 34 | 0.0 | 34 | 0 | 0 | 0 |
| CREDENTIAL:secret | 37 | 0.0 | 37 | 0 | 0 | 0 |
| CREDENTIAL:zqk | 30 | 0.0 | 30 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 5 | 0.0 | 5 | 0 | 0 | 0 |
| DATE_OF_BIRTH:ordinal-of | 8 | 0.0 | 8 | 0 | 0 | 0 |
| DATE_OF_BIRTH:plain | 114 | 0.0 | 114 | 0 | 0 | 0 |
| EMAIL:plain | 193 | 0.0 | 193 | 0 | 0 | 0 |
| EMAIL:spelled | 12 | 0.0 | 12 | 0 | 0 | 0 |
| HANDLE:plain | 127 | 0.0 | 127 | 0 | 0 | 0 |
| ID:card | 36 | 0.0 | 36 | 0 | 0 | 0 |
| ID:card-spaced | 15 | 0.0 | 15 | 0 | 0 | 0 |
| ID:iban | 18 | 0.0 | 18 | 0 | 0 | 0 |
| ID:iban-spaced | 3 | 0.0 | 3 | 0 | 0 | 0 |
| ID:license | 19 | 0.0 | 19 | 0 | 0 | 0 |
| ID:nhs | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:nino | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:passport | 34 | 0.0 | 34 | 0 | 0 | 0 |
| ID:policy | 23 | 0.0 | 23 | 0 | 0 | 0 |
| ID:ssn | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:ssn-nodash | 2 | 0.0 | 2 | 0 | 0 | 0 |
| ID:student | 14 | 0.0714 | 13 | 1 | 0 | 0 |
| LOCATION:plain | 142 | 0.007 | 141 | 0 | 1 | 0 |
| NETWORK:ipv4 | 78 | 0.0 | 78 | 0 | 0 | 0 |
| NETWORK:ipv6 | 8 | 0.0 | 8 | 0 | 0 | 0 |
| NETWORK:mac | 36 | 0.0 | 36 | 0 | 0 | 0 |
| ORG:plain | 179 | 0.0335 | 173 | 2 | 2 | 2 |
| PERSON:lower | 24 | 0.0 | 24 | 0 | 0 | 0 |
| PERSON:plain | 521 | 0.0058 | 518 | 0 | 2 | 1 |
| PERSON:surname-first | 32 | 0.0 | 32 | 0 | 0 | 0 |
| PERSON:upper | 22 | 0.0 | 22 | 0 | 0 | 0 |
| PHONE:dots | 15 | 0.0 | 15 | 0 | 0 | 0 |
| PHONE:plain | 148 | 0.0 | 148 | 0 | 0 | 0 |
| PHONE:words | 3 | 0.0 | 3 | 0 | 0 | 0 |
| URL:plain | 127 | 0.0 | 127 | 0 | 0 | 0 |

## Leak details (type → cause → detail: count)

- **ID** partial: `prefix` 1
- **LOCATION** gated: `service_query_geo <- ner` 1
- **ORG** gated: `implausible_org <- pii_tagger` 2
- **ORG** missed: `-` 2
- **ORG** partial: `suffix` 1, `prefix` 1
- **PERSON** gated: `relation_gate:junk <- pattern` 1, `relation_gate:public_person <- pii_tagger` 1
- **PERSON** missed: `-` 1

## Protected ADDRESS values whose surrogate repeats a part verbatim

`none` 135

## Natural slice: spurious edits by source

1739 messages, 1476 untouched, 732 spurious edits.

| source:type | edits |
|---|---|
| repeat | 317 |
| structural:PERSON | 85 |
| pii_tagger:PERSON | 70 |
| ner:GPE | 55 |
| pii_tagger:ORG | 39 |
| pattern:age | 34 |
| pattern:dob | 25 |
| pattern:PERSON | 17 |
| pattern:handle | 15 |
| pattern:ORG | 12 |
| pii_tagger:id_number | 12 |
| pattern:url | 11 |
| pii_tagger:credential | 10 |
| structural:ORG | 8 |
| pii_tagger:GPE | 4 |
| structural:hostname | 3 |
| pattern:phone_intl | 3 |
| pattern:credential | 2 |
| pattern:worded:age | 2 |
| pii_tagger:address | 2 |
| pattern:ip_address | 1 |
| pattern:id_number | 1 |
| pii_tagger:handle | 1 |
| structural:GPE | 1 |
| pattern:zip_us | 1 |
| pattern:address | 1 |
