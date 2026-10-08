# Leak attribution — `dev` (data split `dev`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split dev --out bench/results/attribution_dev_tagger.json` at commit `91e1ff68d99d` with 6 modified tracked file(s). Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 662}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 30 | 0.0 | 30 | 0 | 0 | 0 |
| AGE | 33 | 0.0303 | 32 | 0 | 1 | 0 |
| CREDENTIAL | 32 | 0.0 | 32 | 0 | 0 | 0 |
| DATE_OF_BIRTH | 35 | 0.0 | 35 | 0 | 0 | 0 |
| EMAIL | 55 | 0.0 | 55 | 0 | 0 | 0 |
| HANDLE | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ID | 53 | 0.0 | 53 | 0 | 0 | 0 |
| LOCATION | 32 | 0.0 | 32 | 0 | 0 | 0 |
| NETWORK | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ORG | 44 | 0.0682 | 41 | 1 | 1 | 1 |
| PERSON | 152 | 0.0197 | 149 | 1 | 1 | 1 |
| PHONE | 47 | 0.0213 | 46 | 0 | 0 | 1 |
| URL | 40 | 0.025 | 39 | 0 | 0 | 1 |
| **all** | 627 | 0.0144 | 618 | 2 | 3 | 4 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 129 | 0.0078 | 128 | 1 | 0 | 0 |
| inline | 133 | 0.0226 | 130 | 1 | 1 | 1 |
| intro | 172 | 0.0116 | 170 | 0 | 2 | 0 |
| json | 21 | 0.0 | 21 | 0 | 0 | 0 |
| signature | 172 | 0.0174 | 169 | 0 | 0 | 3 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 453 | 0.0088 | 449 | 1 | 1 | 2 |
| multi | 93 | 0.0323 | 90 | 0 | 2 | 1 |
| shift | 81 | 0.0247 | 79 | 1 | 0 | 1 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 30 | 0.0 | 30 | 0 | 0 | 0 |
| AGE:plain | 31 | 0.0323 | 30 | 0 | 1 | 0 |
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
| ID:license | 6 | 0.0 | 6 | 0 | 0 | 0 |
| ID:nino | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:passport | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:policy | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:ssn | 2 | 0.0 | 2 | 0 | 0 | 0 |
| ID:ssn-nodash | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:student | 2 | 0.0 | 2 | 0 | 0 | 0 |
| LOCATION:plain | 32 | 0.0 | 32 | 0 | 0 | 0 |
| NETWORK:ipv4 | 30 | 0.0 | 30 | 0 | 0 | 0 |
| NETWORK:ipv6 | 2 | 0.0 | 2 | 0 | 0 | 0 |
| NETWORK:mac | 5 | 0.0 | 5 | 0 | 0 | 0 |
| ORG:plain | 44 | 0.0682 | 41 | 1 | 1 | 1 |
| PERSON:lower | 8 | 0.125 | 7 | 0 | 0 | 1 |
| PERSON:plain | 130 | 0.0077 | 129 | 0 | 1 | 0 |
| PERSON:surname-first | 7 | 0.1429 | 6 | 1 | 0 | 0 |
| PERSON:upper | 7 | 0.0 | 7 | 0 | 0 | 0 |
| PHONE:dots | 4 | 0.0 | 4 | 0 | 0 | 0 |
| PHONE:plain | 43 | 0.0233 | 42 | 0 | 0 | 1 |
| URL:plain | 40 | 0.025 | 39 | 0 | 0 | 1 |

## Leak details (type → cause → detail: count)

- **AGE** gated: `relation_gate:junk <- pii_tagger` 1
- **ORG** gated: `implausible_org <- pii_tagger` 1
- **ORG** missed: `-` 1
- **ORG** partial: `prefix` 1
- **PERSON** gated: `relation_gate:junk <- pii_tagger` 1
- **PERSON** missed: `-` 1
- **PERSON** partial: `last` 1
- **PHONE** missed: `-` 1
- **URL** missed: `in_url` 1

## Protected ADDRESS values whose surrogate repeats a part verbatim

`none` 30

## Natural slice: spurious edits by source

442 messages, 381 untouched, 133 spurious edits.

| source:type | edits |
|---|---|
| repeat | 46 |
| structural:PERSON | 21 |
| pattern:age | 13 |
| pii_tagger:PERSON | 13 |
| ner:GPE | 11 |
| pattern:ORG | 7 |
| pattern:phone_uk | 5 |
| pii_tagger:ORG | 3 |
| structural:ORG | 2 |
| pattern:handle | 2 |
| pii_tagger:handle | 2 |
| pii_tagger:GPE | 2 |
| pii_tagger:credential | 1 |
| pattern:PERSON | 1 |
| structural:address | 1 |
| pattern:url | 1 |
| pattern:dob | 1 |
| structural:GPE | 1 |
