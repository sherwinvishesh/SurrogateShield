# Real-world corpus — annotation guide

The corpus behind gate J2 (`bench/realworld.py`). Each file is JSON Lines, one
message per line, written the way people actually type into a chat assistant.
Every person, address, number and account in it is invented.

* `dev.jsonl` — used while developing detection. Look at it freely.
* `test.jsonl` — written by a separate author who never saw the detection
  code. **Do not tune on it.** It is scored once per release with
  `python bench/realworld.py --split test`.

## Record format

```json
{"id": "rw-test-0001",
 "category": "lowercase_chat",
 "lang": "en",
 "service_query": false,
 "text": "hi im sarah mitchell from austin, call me on 512 784 0923",
 "protect":   [{"value": "sarah mitchell", "type": "PERSON"},
               {"value": "austin", "type": "LOCATION"},
               {"value": "512 784 0923", "type": "PHONE"}],
 "sensitive": [],
 "optional":  [],
 "keep":      []}
```

Every `value` must be an exact, case-sensitive substring of `text`. If the
same value occurs more than once, list it once; every occurrence counts.

## The four lists

**protect** — a reasonable user expects this never to reach the provider.
Leaking any of these is a failure.

| type | what |
|---|---|
| PERSON | names of private people: full, first-only, surname-only, nicknames, any casing, inside signatures / JSON / markdown |
| EMAIL | e-mail addresses |
| PHONE | phone and fax numbers, any country, any format, with or without extension |
| ADDRESS | street line ("1126 E Apache Blvd"), unit ("Apt 4B"), PO box |
| LOCATION | city, town, neighbourhood, or postal/ZIP code **where a private person lives, works, studies or is located**. A ZIP/postal code is always LOCATION |
| ORG | employer, school, clinic, gym, church, etc. **tied to a private person** ("I work at Mercy General", "my son goes to Kyrene Middle School") |
| DATE_OF_BIRTH | a birth date or birth year of a person, any format |
| AGE | a stated age of a person ("34 years old", "turned 7", "aged 81") |
| ID | government / financial / account identifiers: SSN, passport, driver licence, tax IDs (EIN, ITIN, NI number), bank account, routing, IBAN, card, medical record, member/policy/customer number, VIN, licence plate |
| NETWORK | IP address, MAC address, device serial |
| URL | URLs or domains that identify a person (profile URLs, personal sites, shared-document links with a person's name) |
| HANDLE | usernames, social handles, gamer tags, Discord tags |
| CREDENTIAL | passwords, PINs, one-time / backup / 2FA codes, API keys, tokens |

**sensitive** — special-category attributes of a private person (GDPR Art. 9):
health conditions and medications tied to the person (`HEALTH`), religion
(`RELIGION`), ethnicity / nationality (`ETHNICITY`), sexual orientation or
gender identity (`ORIENTATION`), political or union affiliation
(`POLITICAL`). Scored and reported separately from `protect`.

**optional** — values where protecting or keeping are both defensible: a US
state or country of residence, a travel destination, a large employer named in
passing, a relative's role ("my mom"). Neither a leak nor a spurious
replacement, whatever the system does.

**keep** — text a careful system must **not** replace: invoice / order / PO /
ticket numbers, quantities, prices, versions, ship dates and other non-birth
dates, times, product names, public figures, public companies and places in a
topical or general-knowledge question ("history of Rome", "is Tesla stock a
buy", "what did Taylor Swift say"), capitalised common words, acronyms (DM,
HR, EIN, MAC, IBAN, PDF), programming identifiers. Replacing any part of a
`keep` value is a spurious replacement.

Anything not listed in any of the four lists that the system replaces also
counts as a spurious replacement — so list every value a system might
reasonably touch.

## service_query

`true` only when the user asks for something **near a location they give**
("is there a pharmacy near 1126 E Apache Blvd, Tempe?"). The documented policy
for these keeps the city and shifts the house number, so address / LOCATION
values in such a message are reported as *policy* rather than *leaked*. All
other values (names, phones, …) in the message are scored normally.

## Mix (per file)

* At least 20 % of messages have an empty `protect` list (negatives), many of
  them deliberately tricky (number-heavy, acronym-heavy, public figures).
* About 5 % non-English (`lang` set to `es`, `de`, `fr`, `pt`, `hi`, `zh`, …).
* Cover every `protect` type at least 5 times, and every style: lowercase,
  ALL CAPS, typos, no punctuation, e-mail signatures, forwarded e-mail headers,
  JSON / CSV / markdown / code snippets, pasted forms, multi-line messages,
  bullet lists, names with apostrophes / hyphens / particles / diacritics /
  initials, non-Western names, possessives and plurals, the same person in
  different casings, emoji, URLs, very short and fairly long messages
  (up to ~150 words).
* Realistic intents: health questions, legal/tenancy questions, HR,
  job applications, travel, banking, IT support, school, relationships,
  customer service, coding help with pasted logs/configs.
