# SurrogateShield

> A privacy-preserving CLI proxy for LLMs — PII never leaves your device.

SurrogateShield intercepts your messages before they reach any LLM API, detects all personally identifiable information (PII), replaces it with realistic fake surrogates, sends the sanitised message, and restores your real values in the response. All cryptographic operations run locally. Nothing sensitive is ever transmitted.


## How It Works

```
User message
    │
    ▼
[ServiceQueryDetector]
    ├─ service query + street address → fuzz house number ±1, preserve city/state
    ├─ service query, no street addr  → send unchanged (location not PII here)
    └─ not a service query            → full detection cascade
    │
    ▼
SentinelLayer (PatternScan → EntityTrace → ContextGuard)
    │
    ▼
MimicGen → generate type-consistent surrogate values
    │
    ▼
Apply substitutions → sanitised message
    │
    ▼
ShadowMap.update({surrogate: original}) + save (AES-256-GCM encrypted)
    │
    ▼
[Optional] RAG query → prepend anonymised context
    │
    ▼
LLM API  ← receives surrogates only, never real values
    │
    ▼
ResolvePass → restore original values in response
    │
    ▼
Display to user
```



## Features

- **Three-stage PII detection cascade** — regex patterns → spaCy NER → distilbert-NER
- **Realistic surrogate generation** — fake names look like names, fake SSNs pass format checks
- **AES-256-GCM encrypted ShadowMap** — surrogate-to-original mappings never stored in plaintext
- **Multi-provider support** — Claude, Gemini, ChatGPT, or fully offline via Ollama
- **Service-query intelligence** — location queries (restaurants near X) get minimal address fuzzing instead of full replacement, preserving answer quality
- **Quasi-identifier risk detection** — warns when combinations like ZIP+DOB+gender risk re-identification (Sweeney k-anonymity)
- **Privacy-aware RAG** — documents are anonymised before indexing; surrogates are used in all vector store operations
- **PII Finder mode** — test detection on any text with zero API calls
- **Presidio comparison** — side-by-side Microsoft Presidio results in PII Finder show the difference between placeholder `[ENTITY_TYPE]` redaction and SurrogateShield's surrogate approach
- **Batch evaluation** — precision, recall, F1, and per-entity-type breakdown against ground-truth answer keys
- **API Transparency panel** — see exactly what was sent, what was received, and the final restored output



## Architecture

### 1. Detection — SentinelLayer

Three detectors run in sequence. Each masks spans it claims so downstream detectors never double-process the same text.

#### PatternScan (`detection/pattern_scan.py`)

Regex-based structural detection. Runs first so structured PII is masked before any NER model sees it.

| Pattern | Examples |
|---|---|
| Street address |  `99 Cathedral Close` |
| SSN | `544-87-2944` |
| Email | `user@example.com` |
| Phone US | `+1-480-555-1234` |
| Phone UK | `+44 7911 123456` |
| Phone (international) | `+49 8234 927461` |
| Credit card (Luhn-validated) | `4111 1111 1111 1111` |
| Date of birth | `01/15/1990` |
| IPv4 | `192.168.1.100` |
| API key / secret | `sk-ant-...`, `Bearer ...` |
| US ZIP code | `85281` |
| UK postcode | `SW1A 1AA` |

#### EntityTrace (`detection/entity_trace.py`)

spaCy `en_core_web_lg` NER. Extracts `PERSON`, `GPE`, `LOC`, `ORG`, and `FAC` entities. Returns two tiers:

- **Confirmed** — score ≥ 0.85 (promoted immediately)
- **Borderline** — score 0.60–0.85 (passed to ContextGuard for verification)

Includes ORG→GPE reclassification when location prepositions appear before an organisation name (e.g. "lives in Google").

#### ContextGuard (`detection/context_guard.py`)

`dslim/distilbert-NER` (~250 MB, downloaded once from HuggingFace Hub on first run, no server required). Verifies borderline entities from EntityTrace and independently detects anything missed. Applies word-piece artefact cleaning and a blocklist of short / title tokens that commonly cause false positives.

#### Post-processing passes (`detection/logic.py`)

Four additional passes run on the combined entity set:

| Pass | What it does |
|---|---|
| A — Structural ORG | Regex for `[the/a/an] <name> [corporation|company|corp|inc|ltd|llc…]`; no name lists |
| B — Email-username reclassification | Corrects ORG→PERSON when the entity text is a prefix of a detected email username |
| C — PERSON component dedup | Removes standalone surnames that are sub-components of already-detected full names |
| D — Topical geo-entity filter | Drops a GPE/LOC only if it appears exclusively in knowledge-query sub-clauses |

#### ServiceQueryDetector (`detection/service_query.py`)

Identifies messages like "restaurants near 1126 E Apache Blvd, Tempe, AZ" and applies a lighter touch:

- Street addresses get the **house number shifted by ±2–8** (max geographic error ~100 m), street name and city preserved
- City/state names are **not replaced** — the LLM needs them to give useful answers
- A sensitive-topic override (medical, legal, shelter, immigration keywords) forces full anonymisation regardless of query structure
- Address existence is verified via OpenStreetMap Nominatim (optional, skippable in offline environments)

#### Quasi-Identifier Scorer (`detection/quasi_identifier.py`)

Based on Sweeney's k-anonymity research. Detects risky entity-type combinations and issues warnings:

| Combination | Risk |
|---|---|
| ZIP + DOB + Gender | High — 87% of US population uniquely identifiable (Sweeney 2000) |
| Postcode + DOB | High |
| Name + Employer + Location | Medium |
| IP + Location | Medium |



### 2. Generation — MimicGen (`generation/logic.py`)

Generates type-consistent surrogates using [Faker](https://faker.readthedocs.io/). Guarantees no collisions within a session via a `used_surrogates` set. Every surrogate is unique and realistic for its type:

| Entity type | Generated surrogate looks like |
|---|---|
| `PERSON` | `Sarah Mitchell` |
| `email` | `jdoe@example.net` |
| `ssn` | `XXX-XX-XXXX` (valid format) |
| `phone_us` | `+1-###-###-####` |
| `phone_uk` | `+44 7### ######` |
| `phone_intl` | `+49 8234 927461` |
| `address` | `789 Crescent Row, Springfield, IL` |
| `credit_card` | Valid Luhn-format number |
| `dob` | `MM/DD/YYYY` (age 18–80) |
| `ip_address` | `10.x.x.x` |
| `zip_us` / `postcode_uk` | Correct format |
| `api_key` | `sk-` + 32 random chars |
| `GPE` / `LOC` / `ORG` / `FAC` | Faker city/company names |
| `gender_indicator` | Grammatically valid gender expression |



### 3. Storage — ShadowMap (`storage/logic.py`)

An encrypted, per-conversation mapping of `surrogate → original`.

| Property | Detail |
|---|---|
| Encryption | AES-256-GCM with a fresh 12-byte nonce per save |
| Key derivation | HKDF-SHA256 with device secret as IKM and conversation ID as salt |
| Device secret | Generated once at `~/.surrogateshield/device.key` with `0o600` permissions |
| File location | `conversations/<conv_id>.shadowmap` (binary, not human-readable) |
| Graceful degradation | Missing or corrupt file → empty mapping, no crash |



### 4. Reconstruction — ResolvePass (`reconstruction/logic.py`)

Three-pass restoration of original values in LLM responses:

1. **Exact replacement** — handles the vast majority of cases
2. **Component matching** — for multi-word surrogates (e.g. `Ashley` from surrogate `Ashley Wise`), scoped to unresolved surrogates only to prevent corruption of adjacent text
3. **Fuzzy matching** — [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) `partial_ratio` with configurable threshold (default 85)

Every failure is categorised as `exact_miss`, `fuzzy_hit`, or `fuzzy_miss` for research analysis.



### 5. RAG Integration (`chatbot/rag.py`)

Local Retrieval-Augmented Generation backed by [ChromaDB](https://www.trychroma.com/) and [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`).

- No server required — ChromaDB runs in-process with persistent storage in `./chroma_db`
- Documents are **anonymised through the full SentinelLayer pipeline before indexing** — real PII never enters the vector store
- Queries are anonymised before retrieval
- Retrieved context is prepended to the sanitised message before the LLM call
- Surrogate mappings from indexed documents are stored in a shared `rag_global` ShadowMap so they can be restored in responses



## Supported LLM Providers

| Provider | Model | Env var required |
|---|---|---|
| Claude (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-1.5-flash` | `GEMINI_API_KEY` |
| ChatGPT | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Local (Ollama) | `llama3.2` (configurable) | None — runs fully offline |

Switch providers from the **Settings** menu inside the dashboard (press `S`).



## PII Types Detected

| Category | Types |
|---|---|
| Structural (regex) | SSN, email, phone (US/UK/international), credit card, street address, DOB, IPv4, API keys/secrets, US ZIP, UK postcode |
| Named entities (NER) | PERSON, GPE (geo-political entity), LOC, ORG, FAC (facility) |
| Inferred | Gender indicator, implicit location |
| Combination risk | Quasi-identifier sets per Sweeney k-anonymity |



## Quick Start

> **You must activate a virtual environment before installing or running.**
> Installing into the system or base conda Python is the most common cause of
> "package not found" errors at runtime.

```bash
# Clone
git clone <repo-url>
cd SurrogateShield

# Create and activate a virtual environment  ← required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy model
# Required by EntityTrace (stage 2 NER detection).
# Also required by the Presidio comparison panel if you enable it.
python -m spacy download en_core_web_lg

# ContextGuard (stage 3 NER) downloads its model automatically from
# HuggingFace Hub on first use — no manual command needed.
# Model: dslim/distilbert-NER (~250 MB, cached after the first run).

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Launch
./run.sh
```

The first run downloads the distilbert-NER model (~250 MB) from HuggingFace Hub and caches it locally. Subsequent runs are instant.

> **Troubleshooting — Presidio shows "not installed":** this almost always means
> the packages were installed into a different Python than the one running the app.
> Fix: activate your venv first, then re-run `pip install -r requirements.txt`
> and `python -m spacy download en_core_web_lg` from inside the venv.

### Enabling the Presidio comparison panel

The Presidio comparison panel is **off by default**. It adds a side-by-side
Microsoft Presidio result below every PII Finder detection, which is useful for
research comparisons but adds latency and screen noise during normal use.

To enable it:

1. Make sure the packages and model are installed **inside your venv**:
   ```bash
   pip install presidio-analyzer presidio-anonymizer
   python -m spacy download en_core_web_lg   # also used by EntityTrace
   ```
2. Open the app and press **S → C** (Settings → Presidio Comparison) to toggle it on.

To disable it again, press **S → C** from the dashboard.



## Installation

### Requirements

- Python 3.9+
- pip

### Dependencies

```
anthropic>=0.25.0           # Claude API client
python-dotenv>=1.0.0        # .env file loading
spacy>=3.7.0                # EntityTrace NER + Presidio NLP backend
faker>=24.0.0               # Surrogate generation
cryptography>=42.0.0        # AES-256-GCM, HKDF
rapidfuzz>=3.6.0            # Fuzzy reconstruction matching
typer>=0.12.0               # CLI framework
rich>=13.7.0                # Terminal UI
chromadb>=0.4.0             # RAG vector store
sentence-transformers>=2.7.0  # RAG embeddings
transformers>=4.40.0        # ContextGuard (distilbert-NER)
torch>=2.0.0                # ContextGuard inference
requests>=2.31.0            # Address verification (Nominatim)
ollama>=0.1.8               # Local LLM (optional)
presidio-analyzer>=2.2.0    # Presidio comparison panel in PII Finder
presidio-anonymizer>=2.2.0  # Presidio anonymization (companion to analyzer)
```

> **spaCy model:** `en_core_web_lg` is required by EntityTrace (always) and by
> the optional Presidio comparison panel. One download covers both:
> `python -m spacy download en_core_web_lg`.
>
> **ContextGuard model:** `dslim/distilbert-NER` (~250 MB) is downloaded
> automatically from HuggingFace Hub on the first run — no manual command needed.

### Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...       # Required for Claude
GEMINI_API_KEY=...                 # Required for Gemini
OPENAI_API_KEY=sk-...              # Required for ChatGPT
```



## Configuration

### Runtime settings (Dashboard → S → Settings)

These are changed interactively from inside the app and persist across sessions in `~/.surrogateshield/settings.json`.

| Key | Default | What it controls |
|---|---|---|
| `llm_provider` | `claude` | Active LLM backend — Claude / Gemini / ChatGPT / Local |
| `detailed_view` | `true` | Show pipeline stage logs, per-entity PII table, and the API transparency panel in each chat turn |
| `presidio_comparison` | `false` | Show the Presidio side-by-side panel below each PII Finder result. **Off by default** — requires `presidio-analyzer`, `presidio-anonymizer`, and `python -m spacy download en_core_web_lg` to be installed first (see *Enabling the Presidio comparison panel* above) |

### Advanced constants (`config.py`)

Hard-coded thresholds and flags. Edit the file directly to change them; no restart required for PII Finder (restart required for chat sessions).

| Setting | Default | Description |
|---|---|---|
| `ENTITY_TRACE_HIGH_THRESHOLD` | `0.85` | spaCy score above which an entity is immediately confirmed |
| `ENTITY_TRACE_LOW_THRESHOLD` | `0.60` | spaCy score above which an entity is forwarded to ContextGuard |
| `CONTEXT_GUARD_CONFIDENCE_THRESHOLD` | `0.70` | distilbert score required to confirm a borderline entity |
| `FUZZY_MATCH_THRESHOLD` | `85` | rapidfuzz `partial_ratio` threshold for ResolvePass reconstruction |
| `SERVICE_QUERY_DETECTION_ENABLED` | `True` | Enable the lightweight address-fuzzing path for location queries |
| `SERVICE_QUERY_VERIFY_ADDRESSES` | `True` | Verify fuzzed addresses via OpenStreetMap Nominatim (disable for offline use) |
| `SHOW_API_TRANSPARENCY` | `True` | Show the sent / received / restored transparency panel after each chat turn |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model identifier |
| `SPACY_MODEL` | `en_core_web_lg` | spaCy model used by EntityTrace and Presidio |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for RAG embeddings |



## CLI Reference

```bash
# Interactive dashboard (recommended)
python main.py
./run.sh

# Start a new conversation directly
python main.py chat

# Continue a saved conversation
python main.py chat --load <conversation-id>

# New conversation with RAG enabled
python main.py chat --rag

# List all saved conversations
python main.py list

# PII detection sandbox (no API calls, no credits)
python main.py pii-finder

# Index a document into the RAG store
python main.py add-doc path/to/document.txt
```

### Dashboard Keyboard Shortcuts

| Key | Action |
|---|---|
| `N` | New conversation |
| `R` | New conversation with RAG mode |
| `P` | PII Finder — test detection without any API call |
| `1–9` | Open saved conversation by number |
| `D1–D9` | Delete conversation by number |
| `J` | JSON Test — batch-process a question file |
| `E` | Evaluation — score pipeline quality against ground-truth |
| `S` | Settings (provider, view mode) |
| `Q` | Quit |



## Evaluation & Batch Testing

### JSON Batch Test

Place a question file in `experiment/<name>.json`:

```json
[
  { "input": "My name is Revanth and my SSN is 544-87-2944. What are Wyoming's tax benefits?" },
  { "input": "My email is revanth@gmail.com and phone is 480-555-1234. Draft a resignation letter." }
]
```

Press `J` in the dashboard, enter the filename, select which fields to capture, and run. Output is saved to `experiment/<name>_answers.json` with progress flushed every 25 questions — safe to interrupt and resume.

Captured fields include: detected PII at each stage, surrogate map, sanitised input, LLM response, and per-stage timings in milliseconds.

### Evaluation (Precision / Recall / F1)

Pair your question file with an answer-key file at `experiment/<name>_key.json`:

```json
[
  {
    "Question": "My name is Revanth and my SSN is 544-87-2944...",
    "Answer-Key": {
      "name": "Revanth",
      "ssn": "544-87-2944"
    }
  }
]
```

Press `E` in the dashboard to score a completed answers file against its key. The evaluator reports:

- Overall precision, recall, F1, and accuracy
- Per-entity-type breakdown (PERSON, email, SSN, phone, address, etc.)
- Answer rate (non-empty LLM responses)
- Average stage timings
- ResolvePass surrogate leak rate
- Sanitisation quality (PII leak rate to LLM)



## Project Structure

```
SurrogateShield/
├── main.py                  # CLI entry point and interactive dashboard
├── pipeline.py              # End-to-end message pipeline orchestration
├── config.py                # All constants and thresholds (single source of truth)
├── util.py                  # Shared dataclasses (DetectedEntity, Conversation), logging helpers
├── settings_manager.py      # Persistent user settings (~/.surrogateshield/settings.json)
├── evaluator.py             # Precision/recall/F1 evaluation logic
├── json_tester.py           # Batch JSON question processing
├── run.sh                   # Launcher script (venv activation, .env loading)
├── requirements.txt
│
├── detection/               # SentinelLayer — three-stage PII detection cascade
│   ├── logic.py             # Cascade orchestration + post-processing passes A–D
│   ├── pattern_scan.py      # PatternScan — regex-based structured PII detection
│   ├── entity_trace.py      # EntityTrace — spaCy NER (en_core_web_lg)
│   ├── context_guard.py     # ContextGuard — distilbert-NER (dslim/distilbert-NER)
│   ├── service_query.py     # ServiceQueryDetector — address fuzzing for location queries
│   ├── quasi_identifier.py  # Quasi-identifier combination risk scorer (k-anonymity)
│   └── geo_data.py          # Geographic pass-through whitelist (US states, countries)
│
├── generation/
│   └── logic.py             # MimicGen — type-consistent surrogate generation (Faker)
│
├── storage/
│   └── logic.py             # ShadowMap — AES-256-GCM encrypted surrogate mapping store
│
├── reconstruction/
│   └── logic.py             # ResolvePass — three-pass surrogate→original restoration
│
├── presidio/                # Presidio integration layer (optional comparison feature)
│   ├── __init__.py
│   ├── engine.py            # Lazy singleton AnalyzerEngine wrapper
│   ├── detect.py            # detect(text) → list[PresidioEntity]
│   └── redact.py            # redact(text, entities) → [TYPE]-placeholder string
│
├── chatbot/
│   ├── chat.py              # Multi-provider LLM conversation handler (Claude/Gemini/OpenAI/Ollama)
│   └── rag.py               # RAGStore — ChromaDB + sentence-transformers vector store
│
├── experiment/              # Batch test input/output files
│   ├── example.json         # Sample question file
│   ├── example_key.json     # Sample ground-truth answer key
│   └── example_answers.json # Sample output from JSON Test
│
├── tests/                   # Unit and integration tests
│   ├── test1.py
│   ├── test2.py
│   └── test3.py
│
└── conversations/           # Runtime: encrypted .shadowmap + conversation .json files
```



## Security Design

| Component | Mechanism |
|---|---|
| Device secret | 32-byte random key at `~/.surrogateshield/device.key`, `0o600` permissions |
| Per-conversation key | HKDF-SHA256 with device secret as IKM and conversation ID as salt |
| ShadowMap encryption | AES-256-GCM with fresh 12-byte nonce per write |
| API transmission | Only surrogates sent — real values never leave the device |
| Conversation history | Stored locally in `conversations/`; conversation JSON holds surrogate text, not originals |
| `.gitignore` | `*.shadowmap`, `conversations/*.json`, `device.key`, `.env` all excluded |

The ShadowMap file format is: `nonce (12 bytes) || AES-GCM ciphertext`. Without the device key, the mapping is unreadable even if the file is obtained.



## Privacy Guarantees

- **No PII crosses the API boundary.** Every entity confirmed by SentinelLayer is replaced before the HTTP request is made.
- **Service queries get proportional protection.** A restaurant search near your home address gets the house number shifted; the city name is preserved so the answer is useful. Sensitive topic overrides (medical, legal, shelter) force full anonymisation regardless.
- **Geographic generality is preserved.** US states, countries, and major cities are never replaced — they provide no meaningful re-identification risk and destroying them would break answer quality.
- **Quasi-identifier risks are surfaced.** If your message contains combinations like ZIP+DOB+gender that are statistically re-identifying even without traditional PII, you are warned before the message is sent.
- **RAG documents are anonymised at index time.** Real PII never enters the vector store. Retrieval and context injection all operate on surrogates.
