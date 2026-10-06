# Leak attribution — `dev` (data split `dev`, collection `test1`)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split dev --out bench/results/attribution_dev.json` at commit `3e3ae7c27ed1`. Counts only.

Traced edits vs the `ss` arm's span file: {'equal': 662}.

## Injected slice: cause by type

| type | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS | 30 | 0.4667 | 16 | 13 | 1 | 0 |
| AGE | 33 | 0.0909 | 30 | 0 | 0 | 3 |
| CREDENTIAL | 32 | 0.0938 | 29 | 0 | 1 | 2 |
| DATE_OF_BIRTH | 35 | 0.1429 | 30 | 0 | 0 | 5 |
| EMAIL | 55 | 0.0 | 55 | 0 | 0 | 0 |
| HANDLE | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ID | 53 | 0.1321 | 46 | 0 | 0 | 7 |
| LOCATION | 32 | 0.0312 | 31 | 0 | 1 | 0 |
| NETWORK | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ORG | 44 | 0.2955 | 31 | 11 | 1 | 1 |
| PERSON | 152 | 0.0395 | 146 | 6 | 0 | 0 |
| PHONE | 47 | 0.0213 | 46 | 0 | 0 | 1 |
| URL | 40 | 0.2 | 32 | 0 | 0 | 8 |
| **all** | 627 | 0.0973 | 566 | 30 | 4 | 27 |

## Cause by layout

| layout | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| form | 129 | 0.0853 | 118 | 7 | 0 | 4 |
| inline | 133 | 0.1053 | 119 | 8 | 1 | 5 |
| intro | 172 | 0.0756 | 159 | 9 | 0 | 4 |
| json | 21 | 0.2857 | 15 | 3 | 2 | 1 |
| signature | 172 | 0.0988 | 155 | 3 | 1 | 13 |

## Cause by slice

| slice | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| injected_single | 453 | 0.0706 | 421 | 18 | 2 | 12 |
| multi | 93 | 0.0968 | 84 | 4 | 0 | 5 |
| shift | 81 | 0.2469 | 61 | 8 | 2 | 10 |

## Cause by type:format

| type:format | values | leak | protected | partial | gated | missed |
|---|---|---|---|---|---|---|
| ADDRESS:plain | 30 | 0.4667 | 16 | 13 | 1 | 0 |
| AGE:plain | 31 | 0.0323 | 30 | 0 | 0 | 1 |
| AGE:words | 2 | 1.0 | 0 | 0 | 0 | 2 |
| CREDENTIAL:hex | 11 | 0.0909 | 10 | 0 | 0 | 1 |
| CREDENTIAL:password | 5 | 0.2 | 4 | 0 | 1 | 0 |
| CREDENTIAL:secret | 3 | 0.3333 | 2 | 0 | 0 | 1 |
| CREDENTIAL:zqk | 13 | 0.0 | 13 | 0 | 0 | 0 |
| DATE_OF_BIRTH:month-ordinal | 1 | 1.0 | 0 | 0 | 0 | 1 |
| DATE_OF_BIRTH:ordinal-of | 4 | 1.0 | 0 | 0 | 0 | 4 |
| DATE_OF_BIRTH:plain | 30 | 0.0 | 30 | 0 | 0 | 0 |
| EMAIL:plain | 55 | 0.0 | 55 | 0 | 0 | 0 |
| HANDLE:plain | 37 | 0.0 | 37 | 0 | 0 | 0 |
| ID:card | 10 | 0.0 | 10 | 0 | 0 | 0 |
| ID:card-spaced | 3 | 0.0 | 3 | 0 | 0 | 0 |
| ID:iban | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:iban-spaced | 4 | 0.0 | 4 | 0 | 0 | 0 |
| ID:license | 6 | 0.5 | 3 | 0 | 0 | 3 |
| ID:nino | 1 | 1.0 | 0 | 0 | 0 | 1 |
| ID:passport | 8 | 0.125 | 7 | 0 | 0 | 1 |
| ID:policy | 8 | 0.0 | 8 | 0 | 0 | 0 |
| ID:ssn | 2 | 0.0 | 2 | 0 | 0 | 0 |
| ID:ssn-nodash | 1 | 0.0 | 1 | 0 | 0 | 0 |
| ID:student | 2 | 1.0 | 0 | 0 | 0 | 2 |
| LOCATION:plain | 32 | 0.0312 | 31 | 0 | 1 | 0 |
| NETWORK:ipv4 | 30 | 0.0 | 30 | 0 | 0 | 0 |
| NETWORK:ipv6 | 2 | 0.0 | 2 | 0 | 0 | 0 |
| NETWORK:mac | 5 | 0.0 | 5 | 0 | 0 | 0 |
| ORG:plain | 44 | 0.2955 | 31 | 11 | 1 | 1 |
| PERSON:lower | 8 | 0.125 | 7 | 1 | 0 | 0 |
| PERSON:plain | 130 | 0.0154 | 128 | 2 | 0 | 0 |
| PERSON:surname-first | 7 | 0.2857 | 5 | 2 | 0 | 0 |
| PERSON:upper | 7 | 0.1429 | 6 | 1 | 0 | 0 |
| PHONE:dots | 4 | 0.0 | 4 | 0 | 0 | 0 |
| PHONE:plain | 43 | 0.0233 | 42 | 0 | 0 | 1 |
| URL:plain | 40 | 0.2 | 32 | 0 | 0 | 8 |

## Leak details (type → cause → detail: count)

- **ADDRESS** gated: `needs_confirmation <- slm` 1
- **ADDRESS** partial: `house_number+locality+postcode` 3, `house_number+street` 2, `house_number` 2, `house_number+locality` 2, `house_number+postcode` 2, `locality+postcode+unit` 1, `postcode` 1
- **AGE** missed: `-` 3
- **CREDENTIAL** gated: `needs_confirmation <- slm` 1
- **CREDENTIAL** missed: `-` 2
- **DATE_OF_BIRTH** missed: `-` 5
- **ID** missed: `-` 7
- **LOCATION** gated: `needs_confirmation <- slm` 1
- **ORG** gated: `implausible_org <- ner` 1
- **ORG** missed: `-` 1
- **ORG** partial: `inner` 6, `suffix` 4, `prefix` 1
- **PERSON** partial: `last` 2, `first` 2, `middle` 2
- **PHONE** missed: `-` 1
- **URL** missed: `in_url` 8

## Natural slice: spurious edits by source

442 messages, 339 untouched, 384 spurious edits.

| source:type | edits |
|---|---|
| repeat | 123 |
| ner:PERSON | 95 |
| ner:ORG | 44 |
| ner:GPE | 28 |
| structural:PERSON | 26 |
| slm:PERSON | 18 |
| pattern:age | 13 |
| pattern:ORG | 7 |
| slm:LOC | 7 |
| pattern:phone_uk | 5 |
| ner:FAC | 3 |
| pattern:handle | 3 |
| slm:ORG | 3 |
| pattern:PERSON | 3 |
| structural:ORG | 1 |
| structural:address | 1 |
| pattern:url | 1 |
| pattern:dob | 1 |
| ner:LOC | 1 |
| structural:GPE | 1 |
