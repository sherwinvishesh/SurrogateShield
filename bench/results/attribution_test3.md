# Leak attribution — `test3` (data split `test3`, collection `test3`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split test3 --out bench/results/attribution_test3.json` at commit `c381d5f41a61`. Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 3784}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 197 | 0.0 | 197 | 0 | 0 | 0 |
| AGE | 326 | 0.0031 | 325 | 0 | 0 | 1 |
| CREDENTIAL | 195 | 0.0 | 195 | 0 | 0 | 0 |
| DATE_OF_BIRTH | 190 | 0.0 | 190 | 0 | 0 | 0 |
| EMAIL | 300 | 0.0 | 300 | 0 | 0 | 0 |
| HANDLE | 202 | 0.0 | 202 | 0 | 0 | 0 |
| ID | 248 | 0.0121 | 245 | 2 | 0 | 1 |
| LOCATION | 217 | 0.0276 | 211 | 0 | 3 | 3 |
| NETWORK | 196 | 0.0 | 196 | 0 | 0 | 0 |
| ORG | 254 | 0.0354 | 245 | 3 | 4 | 2 |
| PERSON | 867 | 0.0115 | 857 | 1 | 3 | 6 |
| PHONE | 272 | 0.0 | 272 | 0 | 0 | 0 |
| URL | 213 | 0.0047 | 212 | 0 | 0 | 1 |
| **all** | 3677 | 0.0082 | 3647 | 6 | 10 | 14 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 968 | 0.0052 | 963 | 4 | 1 | 0 |
| inline | 816 | 0.0135 | 805 | 2 | 4 | 5 |
| intro | 840 | 0.0036 | 837 | 0 | 3 | 0 |
| json | 187 | 0.0 | 187 | 0 | 0 | 0 |
| signature | 866 | 0.0127 | 855 | 0 | 2 | 9 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 2652 | 0.0079 | 2631 | 4 | 5 | 12 |
| multi | 495 | 0.0162 | 487 | 2 | 4 | 2 |
| shift | 530 | 0.0019 | 529 | 0 | 1 | 0 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 197 | 0.0 | 197 | 0 | 0 | 0 |
| AGE:plain | 280 | 0.0036 | 279 | 0 | 0 | 1 |
| AGE:words | 46 | 0.0 | 46 | 0 | 0 | 0 |
| CREDENTIAL:hex | 53 | 0.0 | 53 | 0 | 0 | 0 |
| CREDENTIAL:password | 62 | 0.0 | 62 | 0 | 0 | 0 |
| CREDENTIAL:secret | 35 | 0.0 | 35 | 0 | 0 | 0 |
| CREDENTIAL:zqk | 45 | 0.0 | 45 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 23 | 0.0 | 23 | 0 | 0 | 0 |
| DATE_OF_BIRTH:ordinal-of | 14 | 0.0 | 14 | 0 | 0 | 0 |
| DATE_OF_BIRTH:plain | 153 | 0.0 | 153 | 0 | 0 | 0 |
| EMAIL:plain | 275 | 0.0 | 275 | 0 | 0 | 0 |
| EMAIL:spelled | 25 | 0.0 | 25 | 0 | 0 | 0 |
| HANDLE:plain | 202 | 0.0 | 202 | 0 | 0 | 0 |
| ID:card | 52 | 0.0 | 52 | 0 | 0 | 0 |
| ID:card-spaced | 23 | 0.0 | 23 | 0 | 0 | 0 |
| ID:iban | 18 | 0.0 | 18 | 0 | 0 | 0 |
| ID:iban-spaced | 11 | 0.0 | 11 | 0 | 0 | 0 |
| ID:license | 24 | 0.0 | 24 | 0 | 0 | 0 |
| ID:nhs | 3 | 0.0 | 3 | 0 | 0 | 0 |
| ID:nino | 4 | 0.0 | 4 | 0 | 0 | 0 |
| ID:passport | 49 | 0.0204 | 48 | 0 | 0 | 1 |
| ID:policy | 24 | 0.0 | 24 | 0 | 0 | 0 |
| ID:ssn | 13 | 0.0 | 13 | 0 | 0 | 0 |
| ID:ssn-nodash | 6 | 0.0 | 6 | 0 | 0 | 0 |
| ID:student | 21 | 0.0952 | 19 | 2 | 0 | 0 |
| LOCATION:plain | 217 | 0.0276 | 211 | 0 | 3 | 3 |
| NETWORK:ipv4 | 141 | 0.0 | 141 | 0 | 0 | 0 |
| NETWORK:ipv6 | 16 | 0.0 | 16 | 0 | 0 | 0 |
| NETWORK:mac | 39 | 0.0 | 39 | 0 | 0 | 0 |
| ORG:plain | 254 | 0.0354 | 245 | 3 | 4 | 2 |
| PERSON:lower | 35 | 0.0 | 35 | 0 | 0 | 0 |
| PERSON:plain | 757 | 0.0132 | 747 | 1 | 3 | 6 |
| PERSON:surname-first | 35 | 0.0 | 35 | 0 | 0 | 0 |
| PERSON:upper | 40 | 0.0 | 40 | 0 | 0 | 0 |
| PHONE:dots | 27 | 0.0 | 27 | 0 | 0 | 0 |
| PHONE:plain | 238 | 0.0 | 238 | 0 | 0 | 0 |
| PHONE:words | 7 | 0.0 | 7 | 0 | 0 | 0 |
| URL:plain | 213 | 0.0047 | 212 | 0 | 0 | 1 |

## Leak details (type → cause → detail: count)

- **AGE** missed: `-` 1
- **ID** missed: `-` 1
- **ID** partial: `prefix` 2
- **LOCATION** gated: `relation_gate:public_org <- ner,pii_tagger` 3
- **LOCATION** missed: `-` 3
- **ORG** gated: `implausible_org <- pii_tagger` 4
- **ORG** missed: `-` 2
- **ORG** partial: `prefix` 3
- **PERSON** gated: `relation_gate:public_person <- pii_tagger` 3
- **PERSON** missed: `-` 6
- **PERSON** partial: `last` 1
- **URL** missed: `in_url` 1

## Protected ADDRESS values whose surrogate repeats a part verbatim

`none` 197

## Natural slice: spurious edits by source

2476 messages, 2160 untouched, 1053 spurious edits.

| source:type | edits |
|---|---|
| repeat | 496 |
| structural:PERSON | 119 |
| pii_tagger:PERSON | 86 |
| ner:GPE | 85 |
| pii_tagger:ORG | 42 |
| pattern:age | 39 |
| pii_tagger:GPE | 30 |
| pattern:handle | 28 |
| pattern:PERSON | 21 |
| pii_tagger:id_number | 17 |
| pattern:ORG | 16 |
| pattern:credential | 15 |
| pattern:dob | 12 |
| pattern:ip_address | 9 |
| structural:ORG | 7 |
| pattern:url | 7 |
| structural:hostname | 5 |
| pii_tagger:handle | 3 |
| pii_tagger:address | 3 |
| pii_tagger:credential | 2 |
| pattern:phone_us | 2 |
| pattern:worded:age | 2 |
| pattern:email | 2 |
| pattern:gender_indicator | 2 |
| pattern:id_number | 1 |
| pattern:address | 1 |
| structural:GPE | 1 |
