# SurrogateShield

Privacy-preserving PII proxy — intercepts text before it reaches any LLM, replaces all personally identifiable information with realistic fake surrogates, and restores the real values in the LLM response. The LLM never sees real PII.

---

## Installation

```bash
pip install surrogateshield
```

After install, download the spaCy language model (required for named-entity recognition):

```bash
python -m spacy download en_core_web_lg
```

---

## Quick start — Claude (Anthropic)

```python
import anthropic
import SurrogateShield as ss

client = anthropic.Anthropic()

user_message = "Hi, I'm Sarah Mitchell. My email is sarah@example.com and my SSN is 123-45-6789."

# Replace real PII with surrogates before sending to Claude
# Output: "Hi, I'm [fake name]. My email is [fake email] and my SSN is [fake SSN]."
sanitized = ss.mask(user_message)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": sanitized}],
)

# Restore the original PII values in Claude's response
# Claude's reply mentions the surrogate name/email; unmask() swaps them back
restored = ss.unmask(response)
print(restored)

# Clear session state when the conversation ends
ss.flush()
```

---

## Multi-turn example — ChatGPT (OpenAI)

```python
from openai import OpenAI
import SurrogateShield as ss

client = OpenAI()
history = []

def chat(user_input: str) -> str:
    # Mask PII before adding to the conversation history
    sanitized = ss.mask(user_input)
    history.append({"role": "user", "content": sanitized})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )

    # Restore PII in the model's reply
    restored = ss.unmask(response)

    # Store the surrogate version in history (never the real PII)
    raw_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": raw_reply})

    return restored

# Turn 1
reply1 = chat("My name is John Doe and I live at 42 Baker Street, London.")
print(reply1)  # John Doe and the real address appear in the output

# Turn 2 — the session remembers all surrogates from turn 1
reply2 = chat("Can you repeat back my name and address?")
print(reply2)  # Real values restored from the shadow map

ss.flush()
```

---

## scan() — inspect without masking

```python
import SurrogateShield as ss

text = "Contact Alice at alice@corp.com or call +1-555-123-4567."

found = ss.scan(text)
# found == {"alice@corp.com": "email", "+1-555-123-4567": "phone_us", "Alice": "PERSON"}

# pii_finder is an alias for scan()
found = ss.pii_finder(text)
```

---

## pii_off — suppress specific PII types

```python
import SurrogateShield as ss

# Don't replace phone numbers or locations — only mask names and emails
ss.config(pii_off=["phone", "location"])

text = "Call Emma at 555-867-5309 from Chicago."
sanitized = ss.mask(text)
# Phone number and Chicago are kept; "Emma" is replaced with a fake name.
print(sanitized)

ss.flush()
```

---

## Detectable PII types

| Type string          | Description                                      |
|----------------------|--------------------------------------------------|
| `email`              | Email addresses                                  |
| `ssn`                | US Social Security Numbers (formatted or bare 9-digit) |
| `phone_us`           | US phone numbers                                 |
| `phone_uk`           | UK phone numbers                                 |
| `phone_intl`         | International phone numbers (non-US, non-UK)     |
| `address`            | Street addresses (structural regex)              |
| `credit_card`        | Credit card numbers (Luhn-validated)             |
| `dob`                | Dates of birth / dates                           |
| `ip_address`         | IPv4 addresses                                   |
| `api_key`            | API keys and secrets (OpenAI, AWS, GitHub, etc.) |
| `gender_indicator`   | Explicit gender declarations or pronouns         |
| `postcode_uk`        | UK postcodes                                     |
| `zip_us`             | US ZIP codes                                     |
| `crypto`             | Bitcoin (P2PKH, P2SH, Bech32) and Ethereum addresses |
| `us_bank_number`     | US ABA routing numbers (checksum-validated)      |
| `us_driver_license`  | US driver's license numbers (context-gated)      |
| `PERSON`             | Personal names (spaCy NER + ContextGuard)        |
| `ORG`                | Organisation names                               |
| `GPE`                | Geopolitical entities (cities, regions not on whitelist) |
| `LOC`                | Other locations                                  |
| `FAC`                | Facilities (buildings, airports, etc.)           |

---

## config() parameters

```python
ss.config(
    detailed_view=True,
    # Print detection tables to stdout after each mask()/scan()/unmask() call.
    # Set False for production / silent operation.

    pii_mem="temp",
    # "temp" — keep surrogate mappings in memory only (default).
    # "/path/to/dir" — persist mappings to disk with AES-256-GCM encryption.
    # The directory must exist. Useful for multi-process or long-running apps.

    pii_off=None,
    # List of PII type names or aliases to detect but NOT replace.
    # Aliases: "phone" → phone_us/phone_uk/phone_intl
    #          "postal_code" → zip_us/postcode_uk
    #          "name" / "names" → PERSON
    #          "location" → GPE/LOC
    #          "org" → ORG
    #          "facility" → FAC
    #          "crypto" → crypto
    #          "bank" → us_bank_number
    #          "license" → us_driver_license
    #          "zip" → zip_us
    #          "postcode" → postcode_uk
    # Direct type strings (e.g. "email", "ssn") also accepted.

    service=True,
    # Enable service-query detection.  When True, messages matching
    # location-lookup patterns ("restaurants near 42 Baker Street") receive
    # minimal address fuzzing (house number ±1) instead of full replacement,
    # so the LLM can still answer the query usefully.

    spacy_model="en_core_web_lg",
    # spaCy model for named-entity recognition.
    # Must be downloaded separately: python -m spacy download en_core_web_lg

    context_guard_enabled=True,
    # Enable the HuggingFace NER second-pass (dslim/distilbert-NER).
    # Downloads ~250 MB on first use; cached afterwards.
    # Set False to use spaCy only (faster, slightly lower recall).

    entity_trace_high_threshold=0.85,
    # spaCy confidence score at or above which an entity is auto-confirmed.

    entity_trace_low_threshold=0.60,
    # spaCy confidence score at or above which an entity is passed to
    # ContextGuard for a second opinion.

    context_guard_threshold=0.70,
    # ContextGuard confidence score at or above which a borderline entity
    # is promoted to confirmed.

    entity_trace_fallback_threshold=0.65,
    # When context_guard_enabled=False, borderline entities above this score
    # are promoted to confirmed directly.

    fuzzy_threshold=85,
    # rapidfuzz partial_ratio threshold (0–100) for the fuzzy reconstruction
    # pass in unmask().  Lower values recover more surrogates at the cost of
    # potential false replacements.
)
```
