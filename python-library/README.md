# SurrogateShield

SurrogateShield is a Python library that acts as a privacy-preserving proxy between your application and any large language model. Before your text reaches the LLM it intercepts every piece of personally identifiable information, replaces each one with a realistic-looking fake value called a surrogate, and after the model responds it swaps the surrogates back to the real values; so the output your users see contains their own data, but the model never processed it.

The library is self-contained, requires no external API, and works with any LLM provider: Anthropic Claude, OpenAI, Google Gemini, or a locally-hosted model.

Paper available on arXiv: [https://arxiv.org/abs/2606.29567](https://arxiv.org/abs/2606.29567)


## Why SurrogateShield exists

When users interact with LLM-powered applications they routinely type their real names, phone numbers, email addresses, home addresses, dates of birth, and other sensitive fields without thinking about where that data goes. Most hosted LLMs log requests, use them for training, or process them on infrastructure you do not control.

SurrogateShield solves this at the application layer. Rather than asking users to sanitize their own inputs, the library does it automatically and transparently. The user types real data, the model sees fake data, and the final answer is presented with the real data restored. From the user's perspective nothing changes; from a privacy perspective the model never had access to the sensitive fields.

This approach is grounded in k-anonymity research (Sweeney 2000) and is designed to be useful in practice: it handles the common combinations of fields that can re-identify individuals even without any single obviously-sensitive field present, such as ZIP code + date of birth + gender.


## How it works

SurrogateShield runs a three-stage detection cascade on every piece of text before it is sent to the LLM.

**Stage 1; PatternScan** uses regular expressions to detect structurally identifiable PII: email addresses, phone numbers, SSNs, credit card numbers, street addresses, IP addresses, API keys, dates, postal codes, and cryptocurrency wallet addresses. Pattern matching is done first so that these spans are masked before the NER models see the text.

**Stage 2; EntityTrace** loads a spaCy NER model (en_core_web_lg by default) and extracts named entities: PERSON, GPE (geopolitical entity), LOC, ORG, and FAC. It skips any span already found by PatternScan. Entities above a high confidence threshold are confirmed immediately; entities in a middle band are passed to Stage 3 for a second opinion.

**Stage 3; ContextGuard** runs a HuggingFace transformer model (dslim/distilbert-NER, ~250 MB, downloaded once and cached) over the text that PatternScan and EntityTrace have not yet claimed. It also makes the final call on borderline entities from EntityTrace.

After the three detection stages, four post-processing passes refine the entity list:

- **Pass A** applies a structural regex to find company names followed by organizational suffixes (corporation, LLC, holdings, etc.) that the NER models might miss.
- **Pass B** reclassifies any ORG entity whose text is a prefix of a detected email username, since spaCy sometimes labels standalone first names as ORG when the email address has already been masked.
- **Pass C** deduplicates PERSON entities by word-component containment: if both "Mitchell" and "Sarah Mitchell" are detected, the shorter one is removed and the reconstruction pass handles any standalone occurrence using the full-name surrogate.
- **Pass D** is the topical geo-entity filter. GPE and LOC entities that appear only inside question sub-clauses ("what restaurants are near London?") are dropped because they are the topic of the query, not personal information. A geo entity that appears in any non-query clause is kept.

The library also scores every detected entity set for quasi-identifier combination risk using the Sweeney k-anonymity model. Combinations like ZIP code + date of birth, name + SSN, or name + employer + city are flagged internally even when each individual field seems innocuous.

Once detection is complete, MimicGen creates a realistic surrogate for each detected value using the Faker library; a fake email for a real email, a Luhn-valid credit card number for a real one, a properly formatted SSN, and so on. The surrogates are type-consistent and unique within a session.

The original → surrogate mapping is inverted (surrogate → original) and stored in an in-memory ShadowMap. After the LLM responds, ResolvePass runs three passes to restore the original values: exact string replacement, component-word matching for multi-word surrogates the model may have split, and rapidfuzz fuzzy matching for cases where the model slightly reformatted a surrogate.


## Installation

Install the package from PyPI:

```bash
pip install surrogateshield
```

PyPI has 1.0.0. This README describes the unreleased 2.1.0 on the `v2`
branch; until it is released, install it from a checkout:

```bash
pip install ./python-library
```

Then install the spaCy language model once (~600 MB):

```bash
python -m spacy download en_core_web_lg
```

Detection fails closed: if a model is missing or fails, `mask()` and `scan()` raise `surrogateshield.DetectorUnavailable` (with the install command in the message) instead of sending text that was only partly checked.

If you want the Rich terminal output (colour tables showing detected PII and surrogates), install the optional display dependency:

```bash
pip install "surrogateshield[display]"
```

The HuggingFace ContextGuard model (dslim/distilbert-NER, ~250 MB) is downloaded on the first call to `mask()` (network needed once) and cached by the transformers library in your local HuggingFace cache directory. Offline without that cache, `mask()` raises `DetectorUnavailable`; turn ContextGuard off with `shield.config(context_guard_enabled=False)` if you do not want it.


## Dependencies

The core package installs the following automatically:

| Package | Purpose |
|---|---|
| `faker` | Generates realistic fake values for each PII type |
| `cryptography` | AES-256-GCM encryption for the persistent shadow map |
| `rapidfuzz` | Fuzzy string matching in the reconstruction pass |
| `spacy` | Named-entity recognition (Stage 2) |
| `en-core-web-lg` | spaCy English NER model: downloaded automatically on first use (~750 MB, cached) |
| `transformers` | HuggingFace NER pipeline (Stage 3 ContextGuard) |
| `torch` | Required backend for the transformers pipeline |

Rich is optional (`pip install "surrogateshield[display]"`) and only affects terminal output formatting.


## Quick start: Claude (Anthropic)

```python
import anthropic
import surrogateshield as shield

client = anthropic.Anthropic()

user_message = (
    "Hi, I'm Sarah Mitchell. My email is sarah.mitchell@gmail.com, "
    "my SSN is 123-45-6789, and I was born on 04/12/1990."
)

# shield.mask() runs the full detection cascade and replaces every detected PII
# field with a realistic fake. The returned string is safe to send to any LLM.
# With detailed_view=True (off by default) it also prints a colour table
# showing what was detected and what surrogate replaced it.
sanitized = shield.mask(user_message)
# sanitized might look like:
# "Hi, I'm Rachel Torres. My email is torresrachel@yahoo.com,
#  my SSN is 876-32-1045, and I was born on 09/27/1983."

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": sanitized}],
)

# shield.unmask() accepts any Anthropic response object directly.
# It extracts the text, looks up every surrogate in the session shadow map,
# and returns the response with original values restored.
restored = shield.unmask(response)
# restored contains "Sarah Mitchell", "sarah.mitchell@gmail.com", etc.
print(restored)

# shield.flush() ends the session: discards the shadow map; the next call
# starts a new session. Call it when a conversation or request lifecycle ends.
shield.flush()
```


## Sessions: one per conversation

Every mapping lives in a `Session`. The module-level `mask()` / `unmask()` / `scan()` / `flush()` use the *current* session: one bound with `use_session()`, otherwise one created on first use by the calling thread or asyncio task. A session is never shared between threads or tasks implicitly, so two users of a web server cannot restore each other's values.

In a server, make the session explicit, one per end-user conversation:

```python
import surrogateshield as shield

session = shield.Session()                 # or Session("conv-42", storage_dir=...) to resume
sanitized = session.mask(user_text)
restored = session.unmask(llm_response)
session.close()                            # discard the mappings

# or bind it for code that calls the module-level functions:
with shield.use_session(session):
    shield.unmask(shield.mask(user_text))
```

A `Session` is thread-safe (one lock per session). `Session.mask_result(text)` returns the sanitized text together with every detection and the replacements made; `Session.forget(value)` erases one value's mappings.


## Multi-turn conversation: OpenAI

Multi-turn conversations require care: the conversation history sent to the model must use surrogates throughout, but the history shown to the user should contain real values. SurrogateShield keeps the session shadow map alive across turns so every surrogate from every previous turn can still be resolved.

```python
from openai import OpenAI
import surrogateshield as shield

client = OpenAI()

# Two separate history lists: one with surrogates for the API, one with real
# values for display. shield.mask() and shield.unmask() handle the translation.
api_history = []
display_history = []

def chat(user_input: str) -> str:
    # Mask PII before it enters the API history
    sanitized = shield.mask(user_input)
    api_history.append({"role": "user", "content": sanitized})
    display_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_history,
    )

    # The raw reply from the model contains surrogates
    raw_reply = response.choices[0].message.content

    # unmask() restores the original PII values
    restored_reply = shield.unmask(response)

    # Store the surrogate version in the API history so future turns
    # are consistent — the model never sees real PII in any turn
    api_history.append({"role": "assistant", "content": raw_reply})
    display_history.append({"role": "assistant", "content": restored_reply})

    return restored_reply

# Turn 1
reply1 = chat("My name is John Doe and I live at 42 Baker Street, London. I'm 34 years old.")
print(reply1)
# The reply will refer to John Doe and 42 Baker Street
# even though the model received a fake name and address

# Turn 2 — the shadow map still holds all surrogates from turn 1
reply2 = chat("What was the address I mentioned earlier?")
print(reply2)
# "42 Baker Street, London" is restored correctly

# End the session
shield.flush()
```


## Google Gemini

```python
from google import genai          # pip install google-genai
import surrogateshield as shield

client = genai.Client()            # reads GEMINI_API_KEY

user_message = "My credit card number is 4532015112830366 and my IP is 192.168.1.100."

sanitized = shield.mask(user_message)
# Credit card (Luhn-validated) and IP address are replaced with fakes

response = client.models.generate_content(model="gemini-2.5-flash", contents=sanitized)

# shield.unmask() accepts Gemini response objects directly via response.text
restored = shield.unmask(response)
print(restored)

shield.flush()
```


## Local / Ollama

```python
import requests as http
import surrogateshield as shield

def ask_ollama(prompt: str) -> str:
    resp = http.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )
    return resp.json()["response"]

user_message = "My phone is +44 7911 123456 and my postcode is SW1A 1AA."

sanitized = shield.mask(user_message)
raw_reply = ask_ollama(sanitized)

# unmask() also accepts plain strings
restored = shield.unmask(raw_reply)
print(restored)

shield.flush()
```


## scan(): detect PII without changing anything

`scan()` runs the full detection cascade, with the same settings as `mask()`, and returns a list of `Detection` objects (`text`, `type`, `start`, `end`, `score`, `source`, `masked`), one per occurrence, in text order. `masked` is `False` for types in `pii_off`. It does not generate surrogates, does not update the shadow map, and does not modify the text. `scan(text, as_dict=True)` returns the old `{value: type}` dict.

```python
import surrogateshield as shield

text = (
    "Contact Alice Nguyen at alice.nguyen@company.org, "
    "call her on +1-415-555-0198, "
    "or write to 99 Market Street, San Francisco, CA 94105."
)

for d in shield.scan(text):
    print(f"{d.type:20s} {d.start:>4}-{d.end:<4} {d.text}")
```

`pii_finder` is an alias for `scan()` provided for readability in data-pipeline contexts:

```python
found = shield.pii_finder(text)
```


## pii_off: detect but do not replace specific types

Sometimes you want SurrogateShield to detect every PII type for awareness but only replace a subset. `pii_off` accepts a list of type names or short aliases. Detected entities whose type matches an entry in `pii_off` are not substituted in the output; they are reported with `masked=False` by `scan()` and in `shield.mask_result(text).unmasked`.

```python
import surrogateshield as shield

# Scenario: a location-based app where city names are needed
# for functionality but personal names must still be protected.
shield.config(pii_off=["location", "org"])

text = "Emma Johnson works at Deloitte in New York and her email is emma@deloitte.com."
sanitized = shield.mask(text)
# "Emma Johnson" → replaced with a fake name
# "emma@deloitte.com" → replaced with a fake email
# "Deloitte" → kept (org is in pii_off)
# "New York" → kept (location is in pii_off)
print(sanitized)

shield.flush()
```

Available aliases and what they expand to:

| Alias | Expands to |
|---|---|
| `phone` | `phone_us`, `phone_uk`, `phone_intl` |
| `postal_code` | `zip_us`, `postcode_uk` |
| `zip` | `zip_us` |
| `postcode` | `postcode_uk` |
| `name` or `names` | `PERSON` |
| `location` | `GPE`, `LOC` |
| `org` | `ORG` |
| `facility` | `FAC` |
| `crypto` | `crypto` |
| `bank` | `us_bank_number` |
| `license` | `us_driver_license` |

You can also pass raw type strings directly, e.g. `pii_off=["email", "dob", "ssn"]`.


## Address handling (v2)

Addresses are parsed by a canonical structured parser that captures the FULL
address — house number, directionals, street, apartment/suite/unit, city,
state, and ZIP — as one unit. Numbered streets ("123 5th Ave"), PO boxes,
ZIP+4, multi-line mailing blocks, comma-free formats, and highway forms
("1500 Highway 50") are all recognized, and the components are never
surrogated independently (the v1 "garbled address" failure mode).

How the surrogate is built is controlled by `address_mode`:

| Mode | Behaviour |
|------|-----------|
| `"shift"` *(default)* | Only the house number changes, by up to ±`address_shift_range` (default ±1 — roughly one building). Street, city, state, ZIP, and the original formatting are preserved **byte-for-byte**: `789 Crescent Row, Tempe, AZ 85281` → `790 Crescent Row, Tempe, AZ 85281`. Maximum answer utility. |
| `"replace"` | A structure-preserving fake: every component is replaced by a fake of the same shape in the same slot (`789 Crescent Row, Tempe, AZ 85281` → `412 Nguyen Row, South Debra, PR 90830`). Strongest privacy — no real component reaches the LLM. |
| `"auto"` | The v1 smart behaviour: `shift` for service queries ("find a coffee shop near …"), `replace` for everything else. Sensitive topics (medical, legal, shelter-related) always force `replace`. |

```python
import surrogateshield as shield

# default: shift mode
sanitized = shield.mask("Find parking near 42 Baker Street, London.")
# → "Find parking near 43 Baker Street, London." (or 41)

# strongest privacy: full structure-preserving replacement
shield.config(address_mode="replace")

# the v1 context-aware behaviour
shield.config(address_mode="auto")

# widen the shift window to ±4
shield.config(address_mode="shift", address_shift_range=4)
```

A shifted address never collides with another real address in the
conversation ("789 Crescent Row" and "790 Crescent Row" both present are
shifted apart automatically), and the same original always maps to the same
surrogate within a session.

No address is sent to a geocoder. `verify_addresses` (a Nominatim existence
check whose result never changed the output) is deprecated and has no effect.


## Service query detection

Service queries ("find a coffee shop near …", directions, weather) suppress
replacement of standalone city/state names so the LLM can give useful local
answers, and drive the `shift`/`replace` decision when `address_mode="auto"`.

```python
# Service query detection is on by default (service=True)
# To disable it and always treat messages uniformly:
shield.config(service=False)
```

Sensitive topic override: even when a message matches the service-query pattern, if it contains keywords related to medical, legal, or social-service topics (HIV, abortion, shelter, domestic violence, rehab, etc.), full anonymization is always applied regardless.


## Persistent shadow map

By default (`pii_mem="temp"`) the surrogate mappings are stored in memory and are lost when the Python process exits. For applications where sessions survive across process restarts; a web server, a long-running pipeline, or a multi-worker deployment; you can point `pii_mem` at a directory on disk and SurrogateShield will persist the shadow map with AES-256-GCM encryption.

```python
import os
import surrogateshield as shield

# The directory must already exist
os.makedirs("/var/app/shadowmaps", exist_ok=True)

shield.config(pii_mem="/var/app/shadowmaps")

# Now every call to mask() atomically writes an encrypted .shadowmap file
# (owner-only, 0o600). shield.flush() deletes it and resets the session.
sanitized = shield.mask("My name is Clara Oswald and my phone is 555-123-4567.")
response_text = "Thanks Clara, I've noted your phone."
restored = shield.unmask(response_text)
print(restored)

shield.flush()
```

The encryption scheme: a 32-byte device secret is generated once at `~/.surrogateshield/device.key` (`$SURROGATESHIELD_HOME` overrides the directory; file `0o600`, directory `0o700`). No key is stored next to the data. The AES-256-GCM key for a session is derived with HKDF-SHA256 (device secret as input, session ID as salt, `shadowmap-v1` as info), and the session ID is bound into the ciphertext as associated data. The file is `"SSv1"`, a fresh 12-byte nonce, then the ciphertext. A file that cannot be decrypted raises `CorruptStoreError` (it is moved aside, not deleted); a device secret that cannot be stored raises `ShadowMapStorageError`. Files written by 0.x (with a `<session>.key` beside them) are still read and are upgraded on the next save.


## Detailed output

SurrogateShield prints nothing by default. For debugging, `detailed_view=True` prints a table after each `scan()`, `mask()` and `unmask()` call. The tables show the ORIGINAL values, so never enable it where stdout is logged:

```python
import surrogateshield as shield

shield.config(detailed_view=True)
```


## Quasi-identifier detection

SurrogateShield identifies quasi-identifier combinations based on Sweeney's k-anonymity research. Even when no field is individually sensitive, certain combinations of fields can uniquely re-identify a person. The following combinations are scored:

| Combination | Risk | Basis |
|---|---|---|
| ZIP code + DOB + Gender | High | 87% of the US population is uniquely identified by all three (Sweeney 2000) |
| Postcode + DOB | High | UK ICO guidance: sufficient for re-identification |
| Name + SSN | High | Enables direct identity theft |
| Name + DOB | High | Standard identity verification combination worldwide |
| Phone + Name | High | Directly identifies an individual |
| IP Address + Name | High | Enables device-level identification |
| Name + Employer + City | Medium | Uniquely identifies in most cities |
| Email + Location | Medium | Narrows to a specific named individual |
| Phone + Location | Medium | Narrows to a local individual |
| DOB + Location + Employer | Medium | Highly specific triple |

When a quasi-identifier combination is detected in your text, SurrogateShield logs it internally and all constituent fields are included in the entity set that gets replaced, not just the obviously-sensitive ones.


## All detectable PII types

| Type string | Detection method | Notes |
|---|---|---|
| `email` | Regex | Standard email format |
| `ssn` | Regex + checksum | Formatted (123-45-6789) and bare 9-digit; disambiguated from ABA routing numbers |
| `phone_us` | Regex | US format with optional country code |
| `phone_uk` | Regex | UK format with +44 or leading 0 |
| `phone_intl` | Regex | All other international formats |
| `address` | Regex | Street number + name + type suffix; detected before NER so addresses are always protected |
| `credit_card` | Regex + Luhn | 16-digit numbers; invalid Luhn checksums are rejected |
| `dob` | Regex | ISO dates, slash/dash formats, written month names |
| `ip_address` | Regex | IPv4 only |
| `api_key` | Regex | OpenAI sk-, Anthropic ant-api-, AWS AKIA, GitHub ghp_/gho_, Google AIzaSy, Bearer tokens |
| `gender_indicator` | Regex | "gender: female", "I am a man", he/him, she/her, they/them |
| `postcode_uk` | Regex | Full UK postcode format |
| `zip_us` | Regex | 5-digit and ZIP+4 |
| `crypto` | Regex | Bitcoin P2PKH (1...), P2SH (3...), Bech32 (bc1...), Ethereum (0x...) |
| `us_bank_number` | Regex + ABA checksum | 9-digit ABA routing numbers; validated by the standard checksum |
| `us_driver_license` | Regex (context-gated) | Fires only when preceded by "driver's license", "DL", etc. |
| `PERSON` | spaCy NER + HuggingFace NER | Personal names; two-model consensus for accuracy |
| `ORG` | spaCy NER + structural regex | Organisation names; structural suffix detection catches names the NER models miss |
| `GPE` | spaCy NER | Geopolitical entities (towns, regions); major cities, countries, and US states are on a whitelist and are never replaced |
| `LOC` | spaCy NER | Other location references |
| `FAC` | spaCy NER | Facilities: buildings, airports, stadiums |

The geographic whitelist covers all 50 US states, major countries, and cities with population above ~500 000. These are never replaced because they are not personally identifying on their own and replacing them would destroy answer quality.


## Surrogate generation

Each PII type has a dedicated generator inside MimicGen that produces a realistic fake:

- Email addresses are generated by Faker and look like real email addresses.
- SSNs follow the 3-2-4 formatted pattern.
- Phone numbers are formatted correctly for their region.
- Credit card numbers pass the Luhn checksum.
- ABA routing numbers pass the ABA checksum.
- Dates of birth are drawn from the range of 18–80 year olds.
- Street addresses are real Faker addresses reformatted to a single line.
- Cryptocurrency addresses follow the correct character-set and length rules for each format.
- Driver's license numbers use the California format (letter + 7 digits) as the most common template.
- Names, company names, and city names come from Faker's locale-aware generators.

All surrogates are unique within a session. If the same real value appears multiple times in a conversation, it always maps to the same surrogate.


## Reconstruction passes

After the LLM responds, `unmask()` runs three passes to restore original values:

**Pass 1; Exact replacement** replaces every surrogate in the shadow map that appears verbatim in the response. This handles the majority of cases.

**Pass 2; Component matching** handles multi-word surrogates that the LLM used only partially. For example if the surrogate was "Rachel Torres" but the model wrote only "Rachel", the first-name component is matched and replaced with the original first name. This pass only runs on surrogates that Pass 1 did not find, to prevent partial matches from corrupting unrelated text.

**Pass 3; Fuzzy matching** uses rapidfuzz `partial_ratio` to find surrogates that the model slightly reformatted (changed capitalisation, added punctuation, etc.). The threshold defaults to 85 out of 100.


## config(): all parameters

```python
# config() changes only the arguments you pass; the others keep their values.
shield.config(
    detailed_view=False,
    # When True, prints Rich-formatted tables to stdout showing what was
    # detected (original values!), what surrogates were assigned, and how
    # many values were restored. Off by default.

    pii_mem="temp",
    # Controls where the session shadow map is stored.
    # "temp" (default): held in memory only, lost when the process exits.
    # Any directory path: encrypted to disk using AES-256-GCM. The directory
    # must exist. Raises ValueError if the path does not exist or is not a
    # directory.

    pii_off=None,
    # List of PII type names or aliases whose detected values should NOT be
    # replaced. They are still detected and shown in scan results. Accepts
    # short aliases ("phone", "location", "name") or direct type strings
    # ("email", "ssn", "dob"). See the alias table above for the full list.

    service=True,
    # When True, messages that match service-query patterns (restaurant
    # searches, directions, weather queries, etc.) trigger minimal address
    # fuzzing instead of full surrogate replacement. The house number is
    # shifted ±1 and the rest of the address is preserved, allowing the LLM
    # to give useful location-based answers. Sensitive topics (medical,
    # legal, shelter-related) always override this and force full replacement.

    spacy_model="en_core_web_lg",
    # The spaCy model EntityTrace loads. In the default detection config
    # ("balanced") spaCy reads places only; the PII tagger reads names and
    # organisations. Install it once with python -m spacy download en_core_web_lg.

    context_guard_enabled=False,
    # ContextGuard, a second NER pass with dslim/distilbert-NER (~250 MB),
    # is off by default: the PII tagger reads names. True turns it back on.

    entity_trace_high_threshold=0.90,
    entity_trace_low_threshold=0.70,
    entity_trace_fallback_threshold=0.75,
    # spaCy gives no per-entity score: EntityTrace scores PERSON 0.88,
    # GPE / ORG 0.85, LOC 0.74, FAC 0.70, so these act as type gates.
    # At or above high: kept. Between low and high: sent to ContextGuard
    # when it is on, else kept at or above fallback. Below low: dropped.

    context_guard_threshold=0.70,
    # The ContextGuard score at or above which its entity is kept.

    fuzzy_threshold=85,
    # The rapidfuzz score (0–100) used in the fuzzy reconstruction pass of
    # unmask(). Lowering this value recovers more surrogates that the model
    # reformatted, at the cost of a higher chance of incorrect replacements.
    # 85 is a conservative default.

    address_mode="auto",
    # How detected addresses are surrogated:
    #   "auto"    — shift for service queries, replace for everything else
    #               (default).
    #   "shift"   — house number shifted by up to ±address_shift_range;
    #               street, city, state, ZIP, and formatting preserved
    #               byte-for-byte.
    #   "replace" — structure-preserving fake address (every component
    #               faked, same shape, one unit).
    # Raises ValueError for any other value.

    address_shift_range=1,
    # Maximum house-number delta for shift mode. Must be an integer >= 1.
    # ±1 keeps the geographic error to roughly one building.

    context_guard_model="dslim/distilbert-NER",
    # The HuggingFace model used by ContextGuard. Any token-classification
    # model id works, e.g. "dslim/bert-base-NER" for higher recall at the
    # cost of speed.

    context_guard_device=-1,
    # Device for ContextGuard inference: -1 = CPU (default), 0+ = GPU id.

    detection=None,
    # The detection config: a preset name ("balanced", "fast", "strict",
    # "classic"), a DetectionConfig, or a dict of one. None: the config the
    # environment names (SURROGATESHIELD_PRESET or
    # SURROGATESHIELD_DETECTION_CONFIG), else "balanced". The stages, their
    # models and per-type thresholds, and which stage may report each type
    # are set there (ages: the patterns only, by default); see
    # CONFIGURATION.md. The flat settings above apply on top of it where
    # they differ from their defaults.
)
```

Every parameter is validated: an unknown `address_mode`, a threshold outside
its range, or an unrecognized `pii_off` entry raises `ValueError` with an
actionable message instead of failing silently later.


## Full API reference

**`shield.config(**kwargs) -> Config`**
Changes the settings new sessions start from, and the current session's settings. Only the arguments you pass change; the rest keep their current values. Raises `ValueError` on any invalid setting, and `RuntimeError` if `pii_mem` changes while the current session holds mappings.

**`shield.Session(session_id=None, *, config=None, storage_dir=None)`**
The masking state of one conversation, thread-safe. Methods: `mask(text) -> str`, `mask_result(text) -> MaskResult`, `scan(text) -> list[Detection]`, `unmask(response) -> str`, `forget(value) -> int`, `close()`; property `mappings` (a copy of surrogate → original). Usable as a context manager (closes on exit). With `storage_dir`, the same `session_id` reopens the same encrypted map.

**`shield.use_session(session)` / `shield.current_session()`**
Bind a session for a `with` block (inherited by asyncio tasks and `asyncio.to_thread` started inside it) / return the session the module-level functions use.

**`shield.scan(text: str, *, as_dict=False) -> list[Detection]`**
Runs the detection cascade with the same settings as `mask()` and returns one `Detection(text, type, start, end, score, source, masked)` per occurrence. Does not modify the text, generate surrogates, or update the shadow map. Detections of `pii_off` types have `masked=False`. `as_dict=True` returns `{value: type}`.

**`shield.pii_finder`**
An alias for `shield.scan`. Provided for readability in data-pipeline contexts.

**`shield.mask(text: str) -> str`** / **`shield.mask_result(text: str) -> MaskResult`**
Runs detection, generates surrogates, applies substitutions, and updates the session shadow map. Types in `pii_off` are detected but not replaced; `mask_result()` reports them in `.unmasked`, together with `.detections` and `.replacements`. Raises `DetectorUnavailable` if a model is missing (nothing is masked) and `TypeError` if `text` is not a str.

**`shield.unmask(response) -> str`**
Accepts a plain string or an Anthropic / OpenAI / Gemini response (SDK object or dict). Anthropic responses: every text block is concatenated; an OpenAI tool-call reply with no content gives `""`. Raises `TypeError` for `None` or an object with no text. Returns the text with original values restored.

**`shield.forget(value) -> int`**
Erases every mapping for one original value from the current session.

**`shield.flush()`**
Ends the current session: clears its shadow map (and deletes its file in persistent mode). The next call starts a new session. Call this at the end of every conversation or request lifecycle.

**Exceptions:** `DetectorUnavailable` (a detection model is missing or failed), `StorageError` (the persistent store cannot be read or written), `TypeError`, `ValueError`.


## Limitations

SurrogateShield reduces the personal data that reaches a provider. It does
not remove all of it. The measured results, including the failures, are in
[`bench/results/README.md`](https://github.com/sherwinvishesh/SurrogateShield/blob/v2/bench/results/README.md), each with the command
that produced it.

- **Detection misses PII in text it was not tuned on.** On real-world
  messages the rules never saw, some names, places, IDs, handles, ages and
  organisations go through unmasked. Some ordinary text gets replaced. Gate
  J2 (≤ 2 % leaked, ≤ 3 % spurious) fails on unseen text.
- **English first.** Detection is built and measured mainly on English. Other
  languages are covered only for the forms listed in
  [`CHANGELOG.md`](https://github.com/sherwinvishesh/SurrogateShield/blob/v2/CHANGELOG.md), such as some Hindi, Chinese and Japanese
  kin frames, and street forms outside the US and UK.
- **Special-category data is not replaced.** Health conditions, medications,
  religion, ethnicity, sexual orientation and political affiliation reach the
  provider as written. In "my mother has diabetes", the diagnosis is sent.
- **Names judged public are kept on purpose.** A name the relation gate takes
  for a public figure, brand or author is sent verbatim. It can also keep a
  private person who shares that name, when nothing in the text ties them to
  the user.
- **Service queries keep the street and city.** With `address_mode="auto"`,
  "nearest pharmacy to …" shifts the house number and sends the rest.
- **Surrogates hide values, not context.** The provider still sees what the
  message says about the person. Enough context can identify someone without
  any of the replaced values. Resistance to an attacker has not been
  re-measured since the attacker protocol was corrected.
- **Restoration can miss a rewritten surrogate.** If the model rewrites a
  surrogate past the fuzzy threshold, it stays in the answer.
- **Detection needs its local models.** Without spaCy's `en_core_web_lg`, or
  without the cached ContextGuard model when offline, `mask()` raises
  `DetectorUnavailable` and sends nothing. The first mask after start-up
  takes seconds.

## Troubleshooting

**ContextGuard model download on first run**

The first call to `mask()` with `context_guard_enabled=True` will download dslim/distilbert-NER (~250 MB) from HuggingFace Hub. This is normal. The model is cached in `~/.cache/huggingface/` and is not downloaded again on subsequent runs.

**Slow first call**

Both spaCy and the HuggingFace model are loaded lazily on the first call. Subsequent calls are fast. If you want to pre-warm the models at application startup:

```python
import surrogateshield as shield

# Pre-warm by scanning a short text — loads the models now
shield.Session().scan("warm up")
```

**Disabling ContextGuard for faster inference**

```python
shield.config(context_guard_enabled=False)
```

This skips the HuggingFace model entirely. spaCy alone handles NER with slightly lower recall.

**Silent operation in production**

```python
shield.config(detailed_view=False)
```

**Surrogate not restored in response**

If the LLM heavily reformatted a surrogate (for example changed the casing of an email domain, split a name with a comma, or abbreviated a company name), neither the exact nor the component pass will find it. The fuzzy pass will attempt a match. You can lower `fuzzy_threshold` to increase recall:

```python
shield.config(fuzzy_threshold=75)
```

Values below 70 are not recommended as they increase the risk of incorrect replacements.

---

Made with ❤️ by Sherwin Vishesh Jathanna
