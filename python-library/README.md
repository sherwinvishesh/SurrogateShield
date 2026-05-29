# SurrogateShield

SurrogateShield is a Python library that acts as a privacy-preserving proxy between your application and any large language model. Before your text reaches the LLM it intercepts every piece of personally identifiable information, replaces each one with a realistic-looking fake value called a surrogate, and after the model responds it swaps the surrogates back to the real values — so the output your users see contains their own data, but the model never processed it.

The library is self-contained, requires no external API, and works with any LLM provider: Anthropic Claude, OpenAI, Google Gemini, or a locally-hosted model.


## Why SurrogateShield exists

When users interact with LLM-powered applications they routinely type their real names, phone numbers, email addresses, home addresses, dates of birth, and other sensitive fields without thinking about where that data goes. Most hosted LLMs log requests, use them for training, or process them on infrastructure you do not control.

SurrogateShield solves this at the application layer. Rather than asking users to sanitize their own inputs, the library does it automatically and transparently. The user types real data, the model sees fake data, and the final answer is presented with the real data restored. From the user's perspective nothing changes; from a privacy perspective the model never had access to the sensitive fields.

This approach is grounded in k-anonymity research (Sweeney 2000) and is designed to be useful in practice: it handles the common combinations of fields that can re-identify individuals even without any single obviously-sensitive field present, such as ZIP code + date of birth + gender.


## How it works

SurrogateShield runs a three-stage detection cascade on every piece of text before it is sent to the LLM.

**Stage 1 — PatternScan** uses regular expressions to detect structurally identifiable PII: email addresses, phone numbers, SSNs, credit card numbers, street addresses, IP addresses, API keys, dates, postal codes, and cryptocurrency wallet addresses. Pattern matching is done first so that these spans are masked before the NER models see the text.

**Stage 2 — EntityTrace** loads a spaCy NER model (en_core_web_lg by default) and extracts named entities: PERSON, GPE (geopolitical entity), LOC, ORG, and FAC. It skips any span already found by PatternScan. Entities above a high confidence threshold are confirmed immediately; entities in a middle band are passed to Stage 3 for a second opinion.

**Stage 3 — ContextGuard** runs a HuggingFace transformer model (dslim/distilbert-NER, ~250 MB, downloaded once and cached) over the text that PatternScan and EntityTrace have not yet claimed. It also makes the final call on borderline entities from EntityTrace.

After the three detection stages, four post-processing passes refine the entity list:

- **Pass A** applies a structural regex to find company names followed by organizational suffixes (corporation, LLC, holdings, etc.) that the NER models might miss.
- **Pass B** reclassifies any ORG entity whose text is a prefix of a detected email username, since spaCy sometimes labels standalone first names as ORG when the email address has already been masked.
- **Pass C** deduplicates PERSON entities by word-component containment: if both "Mitchell" and "Sarah Mitchell" are detected, the shorter one is removed and the reconstruction pass handles any standalone occurrence using the full-name surrogate.
- **Pass D** is the topical geo-entity filter. GPE and LOC entities that appear only inside question sub-clauses ("what restaurants are near London?") are dropped because they are the topic of the query, not personal information. A geo entity that appears in any non-query clause is kept.

The library also scores every detected entity set for quasi-identifier combination risk using the Sweeney k-anonymity model. Combinations like ZIP code + date of birth, name + SSN, or name + employer + city are flagged internally even when each individual field seems innocuous.

Once detection is complete, MimicGen creates a realistic surrogate for each detected value using the Faker library — a fake email for a real email, a Luhn-valid credit card number for a real one, a properly formatted SSN, and so on. The surrogates are type-consistent and unique within a session.

The original → surrogate mapping is inverted (surrogate → original) and stored in an in-memory ShadowMap. After the LLM responds, ResolvePass runs three passes to restore the original values: exact string replacement, component-word matching for multi-word surrogates the model may have split, and rapidfuzz fuzzy matching for cases where the model slightly reformatted a surrogate.


## Installation

Install the package from PyPI:

```bash
pip install surrogateshield
```

Then download the spaCy language model. This is a one-time step and the model is cached locally afterwards:

```bash
python -m spacy download en_core_web_lg
```

If you want the Rich terminal output (colour tables showing detected PII and surrogates), install the optional display dependency:

```bash
pip install "surrogateshield[display]"
```

The HuggingFace ContextGuard model (dslim/distilbert-NER, ~250 MB) is downloaded automatically on the first call to `mask()` and cached by the transformers library in your local HuggingFace cache directory. No manual step is required.


## Dependencies

The core package installs the following automatically:

| Package | Purpose |
|---|---|
| `faker` | Generates realistic fake values for each PII type |
| `cryptography` | AES-256-GCM encryption for the persistent shadow map |
| `rapidfuzz` | Fuzzy string matching in the reconstruction pass |
| `requests` | Address verification via OpenStreetMap Nominatim |
| `spacy` | Named-entity recognition (Stage 2) |
| `transformers` | HuggingFace NER pipeline (Stage 3 ContextGuard) |
| `torch` | Required backend for the transformers pipeline |

Rich is optional (`pip install "surrogateshield[display]"`) and only affects terminal output formatting.


## Quick start — Claude (Anthropic)

```python
import anthropic
import SurrogateShield as ss

client = anthropic.Anthropic()

user_message = (
    "Hi, I'm Sarah Mitchell. My email is sarah.mitchell@gmail.com, "
    "my SSN is 123-45-6789, and I was born on 04/12/1990."
)

# ss.mask() runs the full detection cascade and replaces every detected PII
# field with a realistic fake. The returned string is safe to send to any LLM.
# With detailed_view=True (default) it also prints a colour table showing
# what was detected and what surrogate replaced it.
sanitized = ss.mask(user_message)
# sanitized might look like:
# "Hi, I'm Rachel Torres. My email is torresrachel@yahoo.com,
#  my SSN is 876-32-1045, and I was born on 09/27/1983."

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": sanitized}],
)

# ss.unmask() accepts any Anthropic response object directly.
# It extracts the text, looks up every surrogate in the session shadow map,
# and returns the response with original values restored.
restored = ss.unmask(response)
# restored contains "Sarah Mitchell", "sarah.mitchell@gmail.com", etc.
print(restored)

# ss.flush() clears the session: discards the shadow map and generates a
# new session ID. Call it when a conversation or request lifecycle ends.
ss.flush()
```


## Multi-turn conversation — OpenAI

Multi-turn conversations require care: the conversation history sent to the model must use surrogates throughout, but the history shown to the user should contain real values. SurrogateShield keeps the session shadow map alive across turns so every surrogate from every previous turn can still be resolved.

```python
from openai import OpenAI
import SurrogateShield as ss

client = OpenAI()

# Two separate history lists: one with surrogates for the API, one with real
# values for display. ss.mask() and ss.unmask() handle the translation.
api_history = []
display_history = []

def chat(user_input: str) -> str:
    # Mask PII before it enters the API history
    sanitized = ss.mask(user_input)
    api_history.append({"role": "user", "content": sanitized})
    display_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_history,
    )

    # The raw reply from the model contains surrogates
    raw_reply = response.choices[0].message.content

    # unmask() restores the original PII values
    restored_reply = ss.unmask(response)

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
ss.flush()
```


## Google Gemini

```python
import google.generativeai as genai
import SurrogateShield as ss

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

user_message = "My credit card number is 4532015112830366 and my IP is 192.168.1.100."

sanitized = ss.mask(user_message)
# Credit card (Luhn-validated) and IP address are replaced with fakes

response = model.generate_content(sanitized)

# ss.unmask() accepts Gemini response objects directly via response.text
restored = ss.unmask(response)
print(restored)

ss.flush()
```


## Local / Ollama

```python
import requests as http
import SurrogateShield as ss

def ask_ollama(prompt: str) -> str:
    resp = http.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )
    return resp.json()["response"]

user_message = "My phone is +44 7911 123456 and my postcode is SW1A 1AA."

sanitized = ss.mask(user_message)
raw_reply = ask_ollama(sanitized)

# unmask() also accepts plain strings
restored = ss.unmask(raw_reply)
print(restored)

ss.flush()
```


## scan() — detect PII without changing anything

`scan()` runs the full detection cascade and returns a dict mapping each detected value to its PII type. It does not generate surrogates, does not update the shadow map, and does not modify the text. Use it when you want to inspect what SurrogateShield would find before committing to masking.

```python
import SurrogateShield as ss

text = (
    "Contact Alice Nguyen at alice.nguyen@company.org, "
    "call her on +1-415-555-0198, "
    "or write to 99 Market Street, San Francisco, CA 94105."
)

found = ss.scan(text)
# {
#   "alice.nguyen@company.org": "email",
#   "Alice Nguyen": "PERSON",
#   "+1-415-555-0198": "phone_us",
#   "99 Market Street": "address",
# }

for value, pii_type in found.items():
    print(f"{pii_type:20s}  {value}")
```

`pii_finder` is an alias for `scan()` provided for readability in data-pipeline contexts:

```python
found = ss.pii_finder(text)
```


## pii_off — detect but do not replace specific types

Sometimes you want SurrogateShield to detect every PII type for awareness but only replace a subset. `pii_off` accepts a list of type names or short aliases. Detected entities whose type matches an entry in `pii_off` are identified in the scan results but are not substituted in the output.

```python
import SurrogateShield as ss

# Scenario: a location-based app where city names are needed
# for functionality but personal names must still be protected.
ss.config(pii_off=["location", "org"])

text = "Emma Johnson works at Deloitte in New York and her email is emma@deloitte.com."
sanitized = ss.mask(text)
# "Emma Johnson" → replaced with a fake name
# "emma@deloitte.com" → replaced with a fake email
# "Deloitte" → kept (org is in pii_off)
# "New York" → kept (location is in pii_off)
print(sanitized)

ss.flush()
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


## Service query detection

When a user asks a location-based question such as "find a coffee shop near 99 Market Street", replacing the address with a completely different fake address would make the LLM's answer useless. SurrogateShield detects these service queries and applies minimal address fuzzing instead: the house number is shifted by exactly ±1 (maximum real-world displacement ~20 metres) while the street name, city, and state are preserved verbatim.

```python
import SurrogateShield as ss

# Service query detection is on by default (service=True)
text = "Find a parking space near 42 Baker Street, London."
sanitized = ss.mask(text)
# Address becomes "43 Baker Street, London" or "41 Baker Street, London"
# The model can still answer usefully about that neighbourhood
print(sanitized)

# To disable service query detection and always apply full surrogates:
ss.config(service=False)
```

Sensitive topic override: even when a message matches the service-query pattern, if it contains keywords related to medical, legal, or social-service topics (HIV, abortion, shelter, domestic violence, rehab, etc.), full anonymization is always applied regardless.


## Persistent shadow map

By default (`pii_mem="temp"`) the surrogate mappings are stored in memory and are lost when the Python process exits. For applications where sessions survive across process restarts — a web server, a long-running pipeline, or a multi-worker deployment — you can point `pii_mem` at a directory on disk and SurrogateShield will persist the shadow map with AES-256-GCM encryption.

```python
import os
import SurrogateShield as ss

# The directory must already exist
os.makedirs("/var/app/shadowmaps", exist_ok=True)

ss.config(pii_mem="/var/app/shadowmaps")

# Now every call to mask() writes an encrypted .shadowmap file
# and a per-session .key file (owner-read-only, 0o600 permissions).
# ss.flush() deletes both files and resets the session.
sanitized = ss.mask("My name is Clara Oswald and my phone is 555-123-4567.")
response_text = "Thanks Clara, I've noted your phone."
restored = ss.unmask(response_text)
print(restored)

ss.flush()
```

The encryption scheme: a 32-byte random session key is generated per session and stored at `storage_dir/session_id.key` with owner-only permissions. An AES-256-GCM key is derived from the session key using HKDF-SHA256 with the session ID as salt. The shadow map file stores a fresh 12-byte nonce followed by the ciphertext. The nonce is regenerated on every save.


## Turning off detailed output

By default SurrogateShield prints a table to stdout after each `scan()`, `mask()`, and `unmask()` call. In production or when integrating into an API backend you will want to disable this:

```python
import SurrogateShield as ss

ss.config(detailed_view=False)

# All operations now run silently
sanitized = ss.mask("Contact Bob at bob@example.com.")
restored = ss.unmask("Thanks for reaching out, Bob.")
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

**Pass 1 — Exact replacement** replaces every surrogate in the shadow map that appears verbatim in the response. This handles the majority of cases.

**Pass 2 — Component matching** handles multi-word surrogates that the LLM used only partially. For example if the surrogate was "Rachel Torres" but the model wrote only "Rachel", the first-name component is matched and replaced with the original first name. This pass only runs on surrogates that Pass 1 did not find, to prevent partial matches from corrupting unrelated text.

**Pass 3 — Fuzzy matching** uses rapidfuzz `partial_ratio` to find surrogates that the model slightly reformatted (changed capitalisation, added punctuation, etc.). The threshold defaults to 85 out of 100.


## config() — all parameters

```python
ss.config(
    detailed_view=True,
    # When True, prints Rich-formatted tables to stdout showing what was
    # detected, what surrogates were assigned, and how many values were
    # restored. Set to False for silent / production operation.

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
    # The spaCy model used by EntityTrace for named-entity recognition.
    # Must be downloaded before first use:
    #     python -m spacy download en_core_web_lg
    # You can substitute a smaller model such as en_core_web_sm for faster
    # inference at the cost of NER accuracy.

    context_guard_enabled=True,
    # When True, a second NER pass using dslim/distilbert-NER (~250 MB) is
    # run over text not already claimed by PatternScan or EntityTrace. It
    # also makes the final call on borderline EntityTrace entities.
    # Set to False to use spaCy only; this is faster but has lower recall
    # for edge-case names and organisations.

    entity_trace_high_threshold=0.85,
    # spaCy entities with a confidence score at or above this value are
    # confirmed immediately without passing to ContextGuard.

    entity_trace_low_threshold=0.60,
    # spaCy entities with a score at or above this value but below the high
    # threshold are treated as borderline and sent to ContextGuard for
    # verification. Entities below this value are discarded.

    context_guard_threshold=0.70,
    # The HuggingFace NER confidence score at or above which a borderline
    # entity or a new ContextGuard-detected entity is promoted to confirmed.

    entity_trace_fallback_threshold=0.65,
    # Used only when context_guard_enabled=False. Borderline EntityTrace
    # entities with a score at or above this value are promoted to confirmed
    # directly, since there is no ContextGuard to consult.

    fuzzy_threshold=85,
    # The rapidfuzz partial_ratio score (0–100) used in the third
    # reconstruction pass of unmask(). Lowering this value recovers more
    # surrogates that the model reformatted, at the cost of a higher chance
    # of incorrect replacements. 85 is a conservative default.
)
```


## Full API reference

**`ss.config(**kwargs)`**
Updates the global configuration object. All keyword arguments are optional; unspecified parameters retain their current values. Raises `ValueError` if `pii_mem` is not `"temp"` and the specified path does not exist or is not a directory.

**`ss.scan(text: str) -> dict`**
Runs the full detection cascade on `text` and returns `{detected_value: pii_type}`. Does not modify the text, does not generate surrogates, and does not update the shadow map. Always returns all detected PII regardless of `pii_off` settings. If `detailed_view=True`, prints a scan results table to stdout.

**`ss.pii_finder`**
An alias for `ss.scan`. Provided for readability in data-pipeline contexts.

**`ss.mask(text: str) -> str`**
Runs detection, generates surrogates, applies substitutions, and updates the session shadow map. Respects `pii_off` settings — types in that list are detected but not replaced. Returns the sanitized string safe to send to an LLM. If `detailed_view=True`, prints a masking results table.

**`ss.unmask(response) -> str`**
Accepts any LLM SDK response object or a plain string. Extracts the text content, looks up every surrogate in the session shadow map, and returns the response with original values restored. Tries Anthropic, OpenAI, and Gemini response formats automatically before falling back to `str(response)`. If `detailed_view=True`, prints a one-line restore confirmation.

**`ss.flush()`**
Resets the session: clears the shadow map (and deletes disk files if in persistent mode), discards the MimicGen instance, and generates a new session ID. Call this at the end of every conversation or request lifecycle to prevent surrogate mappings from one session bleeding into the next. If `detailed_view=True`, prints a confirmation line.


## Troubleshooting

**spaCy model not found**

```
OSError: [E050] Can't find model 'en_core_web_lg'.
```

Run `python -m spacy download en_core_web_lg` in the same Python environment where surrogateshield is installed.

**ContextGuard model download on first run**

The first call to `mask()` with `context_guard_enabled=True` will download dslim/distilbert-NER (~250 MB) from HuggingFace Hub. This is normal. The model is cached in `~/.cache/huggingface/` and is not downloaded again on subsequent runs.

**Slow first call**

Both spaCy and the HuggingFace model are loaded lazily on the first call. Subsequent calls are fast. If you want to pre-warm the models at application startup:

```python
import SurrogateShield as ss

# Pre-warm by scanning an empty string — loads the models now
ss.scan("")
```

**Disabling ContextGuard for faster inference**

```python
ss.config(context_guard_enabled=False)
```

This skips the HuggingFace model entirely. spaCy alone handles NER with slightly lower recall.

**Silent operation in production**

```python
ss.config(detailed_view=False)
```

**Surrogate not restored in response**

If the LLM heavily reformatted a surrogate (for example changed the casing of an email domain, split a name with a comma, or abbreviated a company name), neither the exact nor the component pass will find it. The fuzzy pass will attempt a match. You can lower `fuzzy_threshold` to increase recall:

```python
ss.config(fuzzy_threshold=75)
```

Values below 70 are not recommended as they increase the risk of incorrect replacements.
