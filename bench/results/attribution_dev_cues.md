# Leak attribution — `dev` (data split `dev`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split dev --out bench/results/attribution_dev_cues.json` at commit `7c1666859fe8`. Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 597, 'differ': 65}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 30 | 0.0 | 30 | 0 | 0 | 0 |
| AGE | 33 | 0.0303 | 32 | 0 | 0 | 1 |
| CREDENTIAL | 32 | 0.0 | 32 | 0 | 0 | 0 |
| DATE_OF_BIRTH | 35 | 0.0 | 35 | 0 | 0 | 0 |
| EMAIL | 55 | 0.0 | 55 | 0 | 0 | 0 |
| HANDLE | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ID | 53 | 0.0755 | 49 | 1 | 0 | 3 |
| LOCATION | 32 | 0.0312 | 31 | 0 | 1 | 0 |
| NETWORK | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ORG | 44 | 0.1136 | 39 | 3 | 1 | 1 |
| PERSON | 152 | 0.0461 | 145 | 6 | 1 | 0 |
| PHONE | 47 | 0.0213 | 46 | 0 | 0 | 1 |
| URL | 40 | 0.025 | 39 | 0 | 0 | 1 |
| **all** | 627 | 0.0319 | 607 | 10 | 3 | 7 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 129 | 0.0155 | 127 | 2 | 0 | 0 |
| inline | 133 | 0.0451 | 127 | 3 | 2 | 1 |
| intro | 172 | 0.0174 | 169 | 2 | 0 | 1 |
| json | 21 | 0.1429 | 18 | 2 | 1 | 0 |
| signature | 172 | 0.0349 | 166 | 1 | 0 | 5 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 453 | 0.0177 | 445 | 4 | 1 | 3 |
| multi | 93 | 0.0645 | 87 | 2 | 0 | 4 |
| shift | 81 | 0.0741 | 75 | 4 | 2 | 0 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 30 | 0.0 | 30 | 0 | 0 | 0 |
| AGE:plain | 31 | 0.0323 | 30 | 0 | 0 | 1 |
| AGE:words | 2 | 0.0 | 2 | 0 | 0 | 0 |
| CREDENTIAL:hex | 11 | 0.0 | 11 | 0 | 0 | 0 |
| CREDENTIAL:password | 5 | 0.0 | 5 | 0 | 0 | 0 |
| CREDENTIAL:secret | 3 | 0.0 | 3 | 0 | 0 | 0 |
| CREDENTIAL:zqk | 13 | 0.0 | 13 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 1 | 0.0 | 1 | 0 | 0 | 0 |
| DATE_OF_BIRTH:ordinal-of | 4 | 0.0 | 4 | 0 | 0 | 0 |
| DATE_OF_BIRTH:plain | 30 | 0.0 | 30 | 0 | 0 | 0 |
| EMAIL:plain | 55 | 0.0 | 55 | 0 | 0 | 0 |
| HANDLE:plain | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ID:card | 10 | 0.0 | 10 | 0 | 0 | 0 |
| ID:card-spaced | 3 | 0.0 | 3 | 0 | 0 | 0 |
| ID:iban | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:iban-spaced | 4 | 0.0 | 4 | 0 | 0 | 0 |
| ID:license | 6 | 0.1667 | 5 | 0 | 0 | 1 |
| ID:nino | 1 | 1.0 | 0 | 0 | 0 | 1 |
| ID:passport | 8 | 0.125 | 7 | 0 | 0 | 1 |
| ID:policy | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:ssn | 2 | 0.0 | 2 | 0 | 0 | 0 |
| ID:ssn-nodash | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:student | 2 | 0.5 | 1 | 1 | 0 | 0 |
| LOCATION:plain | 32 | 0.0312 | 31 | 0 | 1 | 0 |
| NETWORK:ipv4 | 30 | 0.0 | 30 | 0 | 0 | 0 |
| NETWORK:ipv6 | 2 | 0.0 | 2 | 0 | 0 | 0 |
| NETWORK:mac | 5 | 0.0 | 5 | 0 | 0 | 0 |
| ORG:plain | 44 | 0.1136 | 39 | 3 | 1 | 1 |
| PERSON:lower | 8 | 0.125 | 7 | 1 | 0 | 0 |
| PERSON:plain | 130 | 0.0154 | 128 | 2 | 0 | 0 |
| PERSON:surname-first | 7 | 0.2857 | 5 | 2 | 0 | 0 |
| PERSON:upper | 7 | 0.2857 | 5 | 1 | 1 | 0 |
| PHONE:dots | 4 | 0.0 | 4 | 0 | 0 | 0 |
| PHONE:plain | 43 | 0.0233 | 42 | 0 | 0 | 1 |
| URL:plain | 40 | 0.025 | 39 | 0 | 0 | 1 |

## Leak details (type → cause → detail: count)

- **AGE** missed: `-` 1
- **ID** missed: `-` 3
- **ID** partial: `prefix` 1
- **LOCATION** gated: `needs_confirmation <- slm` 1
- **ORG** gated: `implausible_org <- ner` 1
- **ORG** missed: `-` 1
- **ORG** partial: `suffix` 2, `prefix` 1
- **PERSON** gated: `needs_confirmation <- slm` 1
- **PERSON** partial: `last` 2, `first` 2, `middle` 2
- **PHONE** missed: `-` 1
- **URL** missed: `in_url` 1

## Protected ADDRESS values whose surrogate repeats a part verbatim

`none` 30

## Natural slice: spurious edits by source

442 messages, 339 untouched, 385 spurious edits.

| source:type | edits |
|---|---|
| repeat | 123 |
| ner:PERSON | 95 |
| ner:ORG | 43 |
| ner:GPE | 28 |
| structural:PERSON | 27 |
| slm:PERSON | 17 |
| pattern:age | 13 |
| pattern:ORG | 7 |
| slm:LOC | 7 |
| pattern:phone_uk | 5 |
| pattern:PERSON | 4 |
| ner:FAC | 3 |
| pattern:handle | 3 |
| slm:ORG | 3 |
| structural:ORG | 2 |
| structural:address | 1 |
| pattern:url | 1 |
| pattern:dob | 1 |
| ner:LOC | 1 |
| structural:GPE | 1 |
