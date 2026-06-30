# SurrogateShield

```
      ◆  ────────────────────────────────────  ◆       
                                                       
      S   U   R   R   O   G   A   T   E                
                                                       
      ███████╗██╗  ██╗██╗███████╗██╗     ██████╗       
      ██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗      
      ███████╗███████║██║█████╗  ██║     ██║  ██║      
      ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║      
      ███████║██║  ██║██║███████╗███████╗██████╔╝      
      ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝       
                                                       
      ◆  ────────────────────────────────────  ◆       
                                                       
      Privacy-preserving proxy for LLMs                
      PII never leaves your device                     

```

<p align="center">
  <a href="https://pypi.org/project/surrogateshield/"><img src="https://img.shields.io/pypi/v/surrogateshield?label=PyPI&color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/surrogateshield/"><img src="https://img.shields.io/pypi/pyversions/surrogateshield?label=Python" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <img src="https://img.shields.io/badge/spaCy-3.7%2B-09A3D5?logo=spacy&logoColor=white" alt="spaCy">
  <img src="https://img.shields.io/badge/encryption-AES--256--GCM-blueviolet" alt="AES-256-GCM">
  <a href="https://arxiv.org/abs/2606.29567"><img src="https://img.shields.io/badge/arXiv-2606.29567-b31b1b.svg" alt="arXiv"></a>
  <br/>
  <a href="https://sherwinvishesh.github.io/SurrogateShield/"><img src="https://img.shields.io/badge/website-SurrogateShield-1f6feb?logo=githubpages&logoColor=white" alt="Website"></a>
  <a href="https://sherwinvishesh.github.io/SurrogateShield/docs.html"><img src="https://img.shields.io/badge/docs-read%20now-0d9488?logo=readthedocs&logoColor=white" alt="Documentation"></a>
</p>


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

- **Three-stage PII detection cascade**: regex patterns → spaCy NER → distilbert-NER
- **Realistic surrogate generation**: fake names look like names, fake SSNs pass format checks, fake Bitcoin addresses match Base58 format
- **AES-256-GCM encrypted ShadowMap**: surrogate-to-original mappings never stored in plaintext
- **Multi-provider support**: Claude, Gemini, ChatGPT, or fully offline via Ollama
- **Service-query intelligence**: location queries (restaurants near X) get minimal address fuzzing instead of full replacement, preserving answer quality
- **Quasi-identifier risk detection**: warns when combinations like ZIP+DOB+gender risk re-identification (Sweeney k-anonymity)
- **Privacy-aware RAG**: documents are anonymised before indexing; surrogates are used in all vector store operations
- **PII Finder mode**: test detection on any text with zero API calls
- **Presidio comparison**: side-by-side Microsoft Presidio results in PII Finder and Evaluation, including per-entity-type F1/precision/recall
- **Batch evaluation**: precision, recall, F1, per-entity-type breakdown, ResolvePass leak rate, sanitisation quality, BERTScore utility preservation, Presidio side-by-side comparison, and **ablation study** against ground-truth answer keys
- **API Transparency panel**: see exactly what was sent, what was received, and the final restored output
- **Attacker Experiment**: simulates an informed adversary who intercepts sanitised API traffic and attempts to recover original PII from both SurrogateShield and Presidio output; proves surrogate-based anonymisation achieves equivalent inference resistance to placeholder redaction
- **Standalone Python library**: the full detection and masking pipeline is also packaged as `surrogateshield` (`pip install surrogateshield`), a self-contained API with no dashboard dependency, for embedding directly into any Python application or service



## Architecture

### 1. Detection: SentinelLayer

Three detectors run in sequence. Each masks spans it claims so downstream detectors never double-process the same text.

#### PatternScan (`detection/pattern_scan.py`)

Regex-based structural detection. Runs first so structured PII is masked before any NER model sees it. Validators (Luhn, ABA checksum) reject false positives before any entity is emitted.

| Pattern | Examples | Validator |
|---|---|---|
| Street address | `99 Cathedral Close`, `456 Innovation Plaza` | None |
| SSN | `544-87-2944` | None |
| Email | `user@example.com` | None |
| Phone US | `+1-480-555-1234` | None |
| Phone UK | `+44 7911 123456` | None |
| Phone (international) | `+49 8234 927461` | None |
| Credit card | `4111 1111 1111 1111` | Luhn algorithm |
| Date of birth | `01/15/1990`, `March 14 1990` | None |
| IPv4 | `192.168.1.100` | None |
| API key / secret | `sk-ant-...`, `Bearer ...`, `ghp_...`, `AKIA...`, `AIzaSy...` | None |
| Gender indicator | `gender: female`, `she/her`, `I am a man` | None |
| UK postcode | `SW1A 1AA` | None |
| US ZIP code | `85281`, `85281-1234` | None |
| **Crypto wallet** *(new)* | Bitcoin P2PKH/P2SH/Bech32, Ethereum `0x...` | None |
| **ABA routing number** *(new)* | `021000021`, `122105155` | ABA 9-digit checksum |
| **US driver's license** *(new)* | `B7654321`, `F123456789012` | Context-gated (keyword required) |

Pattern order matters; patterns claim character spans; later patterns cannot overlap earlier ones. In particular: `crypto` and `us_bank_number` run before `zip_us` so that 9-digit routing numbers and long hex strings are claimed before the ZIP pattern can fragment them.

**ABA checksum** (`_aba_routing_valid`): `(3·d₀ + 7·d₁ + d₂ + 3·d₃ + 7·d₄ + d₅ + 3·d₆ + 7·d₇ + d₈) mod 10 = 0`. Eliminates false positives; random 9-digit numbers almost always fail.

**Driver's license** is context-gated: the regex requires a keyword (`driver's license`, `license number`, `DL`, `D.L.`) within the same phrase. Only the license value itself (captured in group 1) is marked as an entity; the keyword prefix is left in the sanitised text.

#### EntityTrace (`detection/entity_trace.py`)

spaCy `en_core_web_lg` NER. Extracts `PERSON`, `GPE`, `LOC`, `ORG`, and `FAC` entities. Returns two tiers:

- **Confirmed**: score ≥ 0.85 (promoted immediately)
- **Borderline**: score 0.60–0.85 (passed to ContextGuard for verification)

Includes ORG→GPE reclassification when location prepositions appear before an organisation name (e.g. "lives in Google").

#### ContextGuard (`detection/context_guard.py`)

`dslim/distilbert-NER` (~250 MB, downloaded once from HuggingFace Hub on first run, no server required). Verifies borderline entities from EntityTrace and independently detects anything missed. Applies word-piece artefact cleaning and a blocklist of short / title tokens that commonly cause false positives.

#### Post-processing passes (`detection/logic.py`)

Four additional passes run on the combined entity set:

| Pass | What it does |
|---|---|
| A: Structural ORG | Regex for `[the/a/an] <name> [corporation|company|corp|inc|ltd|llc…]`; no name lists |
| B: Email-username reclassification | Corrects ORG→PERSON when the entity text is a prefix of a detected email username |
| C: PERSON component dedup | Removes standalone surnames that are sub-components of already-detected full names |
| D: Topical geo-entity filter | Drops a GPE/LOC only if it appears exclusively in knowledge-query sub-clauses |

#### ServiceQueryDetector (`detection/service_query.py`)

Identifies messages like "restaurants near 1126 E Apache Blvd, Tempe, AZ" and applies a lighter touch:

- Street addresses get the **house number shifted by ±2–8** (max geographic error ~100 m), street name and city preserved
- City/state names are **not replaced**: the LLM needs them to give useful answers
- A sensitive-topic override (medical, legal, shelter, immigration keywords) forces full anonymisation regardless of query structure
- Address existence is verified via OpenStreetMap Nominatim (optional, skippable in offline environments)

#### Quasi-Identifier Scorer (`detection/quasi_identifier.py`)

Based on Sweeney's k-anonymity research. Detects risky entity-type combinations and issues warnings:

| Combination | Risk |
|---|---|
| ZIP + DOB + Gender | High: 87% of US population uniquely identifiable (Sweeney 2000) |
| Postcode + DOB | High |
| Name + Employer + Location | Medium |
| IP + Location | Medium |



### 2. Generation: MimicGen (`generation/logic.py`)

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
| **`crypto`** *(new)* | Bitcoin P2PKH format: `1` + Base58 chars, 26–35 chars |
| **`us_bank_number`** *(new)* | Valid 9-digit ABA routing number (passes checksum) |
| **`us_driver_license`** *(new)* | CA-format: letter + 7 digits (e.g. `B4923817`) |



### 3. Storage: ShadowMap (`storage/logic.py`)

An encrypted, per-conversation mapping of `surrogate → original`.

| Property | Detail |
|---|---|
| Encryption | AES-256-GCM with a fresh 12-byte nonce per save |
| Key derivation | HKDF-SHA256 with device secret as IKM and conversation ID as salt |
| Device secret | Generated once at `~/.surrogateshield/device.key` with `0o600` permissions |
| File location | `conversations/<conv_id>.shadowmap` (binary, not human-readable) |
| Graceful degradation | Missing or corrupt file → empty mapping, no crash |



### 4. Reconstruction: ResolvePass (`reconstruction/logic.py`)

Three-pass restoration of original values in LLM responses:

1. **Exact replacement**: handles the vast majority of cases
2. **Component matching**: for multi-word surrogates (e.g. `Ashley` from surrogate `Ashley Wise`), scoped to unresolved surrogates only to prevent corruption of adjacent text
3. **Fuzzy matching**: [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) `partial_ratio` with configurable threshold (default 85)

Every failure is categorised as `exact_miss`, `fuzzy_hit`, or `fuzzy_miss` for research analysis.



### 5. RAG Integration (`chatbot/rag.py`)

Local Retrieval-Augmented Generation backed by [ChromaDB](https://www.trychroma.com/) and [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`).

- No server required; ChromaDB runs in-process with persistent storage in `./chroma_db`
- Documents are **anonymised through the full SentinelLayer pipeline before indexing**: real PII never enters the vector store
- Queries are anonymised before retrieval
- Retrieved context is prepended to the sanitised message before the LLM call
- Surrogate mappings from indexed documents are stored in a shared `rag_global` ShadowMap so they can be restored in responses


### 6. Python Library (`python-library/surrogateshield`)

All five components above; PatternScan, EntityTrace, ContextGuard, MimicGen, ShadowMap, and ResolvePass; are re-packaged as a self-contained pip-installable library. The library carries its own copies of every module; it shares no code with the main application at runtime. The public surface is five functions (`config`, `scan`, `mask`, `unmask`, `flush`) and a single import line. See the [Python Library](#python-library) section below for usage.



## Supported LLM Providers

| Provider | Model | Env var required |
|---|---|---|
| Claude (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-1.5-flash` | `GEMINI_API_KEY` |
| ChatGPT | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Local (Ollama) | `llama3.2` (configurable) | None: runs fully offline |

Switch providers from the **Settings** menu inside the dashboard (press `S`).



## PII Types Detected

| Category | Types |
|---|---|
| Structural (regex) | SSN, email, phone (US/UK/international), credit card, street address, DOB, IPv4, API keys/secrets, gender indicator, US ZIP, UK postcode, **crypto wallet** (Bitcoin/Ethereum), **ABA routing number**, **US driver's license** |
| Named entities (NER) | PERSON, GPE (geo-political entity), LOC, ORG, FAC (facility) |
| Inferred | Implicit location |
| Combination risk | Quasi-identifier sets per Sweeney k-anonymity |

### New in recent release

| Type | Description | Detection mechanism |
|---|---|---|
| `crypto` | Bitcoin P2PKH (`1…`), P2SH (`3…`), Bech32 (`bc1…`), Ethereum (`0x` + 40 hex) | Regex: highly distinctive character sets, no validator needed |
| `us_bank_number` | US ABA routing numbers (9 digits) | Regex + ABA checksum: `(3·d₀ + 7·d₁ + d₂ + …) mod 10 = 0` |
| `us_driver_license` | State DL numbers: letter + 7 digits (CA), letter + 12 digits (FL), etc. | Context-gated regex: fires only when preceded by a license keyword |



## Quick Start

> **You must activate a virtual environment before installing or running.**
> Installing into the system or base conda Python is the most common cause of
> "package not found" errors at runtime.

```bash
# Clone
git clone https://github.com/sherwinvishesh/SurrogateShield.git
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

> **Troubleshooting; Presidio shows "not installed":** this almost always means
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
bert-score>=0.3.13          # Utility preservation scoring (BERTScore comparison)
ollama>=0.1.8               # Local LLM (optional)
presidio-analyzer>=2.2.0    # Presidio comparison panel in PII Finder
presidio-anonymizer>=2.2.0  # Presidio anonymization (companion to analyzer)
```

> **spaCy model:** `en_core_web_lg` is required by EntityTrace (always) and by
> the optional Presidio comparison panel. One download covers both:
> `python -m spacy download en_core_web_lg`.
>
> **ContextGuard model:** `dslim/distilbert-NER` (~250 MB) is downloaded
> automatically from HuggingFace Hub on the first run; no manual command needed.
>
> **BERTScore:** `bert-score` and its `roberta-large` model (~1.4 GB) are only
> needed if you enable BERTScore fields in JSON Test. The model is downloaded
> automatically on first use.

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
| `llm_provider` | `claude` | Active LLM backend: Claude / Gemini / ChatGPT / Local |
| `detailed_view` | `false` | Show pipeline stage logs, per-entity PII table, and the API transparency panel in each chat turn |
| `presidio_comparison` | `false` | Show the Presidio side-by-side panel below each PII Finder result. **Off by default**: requires `presidio-analyzer`, `presidio-anonymizer`, and `python -m spacy download en_core_web_lg` to be installed first (see *Enabling the Presidio comparison panel* above) |

### Advanced constants (`config.py`)

Hard-coded thresholds and flags. Edit the file directly to change them; no restart required for PII Finder (restart required for chat sessions).

| Setting | Default | Description |
|---|---|---|
| `ENTITY_TRACE_HIGH_THRESHOLD` | `0.85` | spaCy score above which an entity is immediately confirmed |
| `ENTITY_TRACE_LOW_THRESHOLD` | `0.60` | spaCy score above which an entity is forwarded to ContextGuard |
| `CONTEXT_GUARD_CONFIDENCE_THRESHOLD` | `0.70` | distilbert score required to confirm a borderline entity |
| `ENTITY_TRACE_FALLBACK_THRESHOLD` | `0.65` | Score used when ContextGuard is disabled to promote borderline entities |
| `FUZZY_MATCH_THRESHOLD` | `85` | rapidfuzz `partial_ratio` threshold for ResolvePass reconstruction |
| `SERVICE_QUERY_DETECTION_ENABLED` | `True` | Enable the lightweight address-fuzzing path for location queries |
| `SERVICE_QUERY_VERIFY_ADDRESSES` | `True` | Verify fuzzed addresses via OpenStreetMap Nominatim (disable for offline use) |
| `SHOW_API_TRANSPARENCY` | `True` | Show the sent / received / restored transparency panel after each chat turn |
| `RAG_TOP_K` | `3` | Number of document chunks retrieved per RAG query |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk when splitting indexed documents |
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
| `P` | PII Finder: test detection without any API call |
| `1–9` | Open saved conversation by number |
| `D1–D9` | Delete conversation by number |
| `J` | JSON Test: batch-process a question file |
| `E` | Evaluation: score pipeline quality against ground-truth |
| `A` | Attacker Experiment: simulate adversarial PII recovery |
| `S` | Settings (provider, view mode) |
| `H` | Help |
| `Q` | Quit |



## Evaluation & Batch Testing

### JSON Batch Test

Place a question file in `experiment/<name>.json`:

```json
[
  { "input": "My name is Sherwin and my SSN is 544-87-2944. What are Wyoming's tax benefits?" },
  { "input": "My email is sjathann@asu.edu and phone is 480-555-1234. Draft a resignation letter." }
]
```

Press `J` in the dashboard, enter the filename, select which fields to capture, and run. Output is saved to `experiment/<name>_answers.json` with progress flushed every 25 questions; safe to interrupt and resume.

#### Available output fields

| Field | What it captures |
|---|---|
| `question` | The original question text |
| `pattern_scan_pii` | Entities detected by PatternScan (regex stage) |
| `entity_trace_pii` | Entities detected by EntityTrace (spaCy NER) |
| `context_guard_pii` | Entities detected by ContextGuard (distilbert-NER) |
| `confirmed_pii` | Final combined confirmed entity list |
| `pii_detail` | Per-entity type, score, and source |
| `quasi_id_risks` | Quasi-identifier combination risks detected |
| `surrogate_map` | Mapping of original → surrogate for every replaced entity |
| `sanitized_input` | The exact text sent to the LLM |
| `llm_response` | Raw LLM response (surrogates, before restoration) |
| `stage_timings_ms` | PatternScan / EntityTrace / ContextGuard / surrogate gen / LLM latency in ms |
| `recognized_not_replaced` | Entities detected as PII but intentionally skipped (e.g. topical geo in service queries) |
| `presidio_sanitized_input` | Presidio `[TYPE]`-placeholder redaction (baseline for BERTScore comparison) |
| `presidio_found_piis` | Presidio raw detected entities: type, value, score |
| `bertscore_ss` | BERTScore of original vs SurrogateShield sanitised input |
| `bertscore_presidio` | BERTScore of original vs Presidio sanitised input |

By default, `presidio_sanitized_input`, `presidio_found_piis`, `bertscore_ss`, and `bertscore_presidio` are **off**: enable them in the field selection screen to generate data for the Presidio comparison and BERTScore tables in Evaluation.

### Evaluation (Precision / Recall / F1)

Pair your question file with an answer-key file at `experiment/<name>_key.json`:

```json
[
  {
    "Question": "My name is Sherwin and my SSN is 544-87-2944...",
    "Answer-Key": {
      "name": "Sherwin",
      "ssn": "544-87-2944"
    }
  }
]
```

Press `E` in the dashboard to score a completed answers file against its key.

#### Supported answer-key labels

The evaluator maps flexible label names to internal types:

| Label(s) in key file | Internal type |
|---|---|
| `name`, `person`, `PERSON` | `PERSON` |
| `email` | `email` |
| `phone`, `phone_us`, `phone_uk`, `phone_intl` | `phone` |
| `ssn` | `ssn` |
| `address` | `address` |
| `dob`, `date_of_birth` | `dob` |
| `org`, `ORG`, `organization` | `ORG` |
| `gpe`, `GPE`, `location` | `GPE` |
| `credit_card` | `credit_card` |
| `api_key` | `api_key` |
| `ip`, `ip_address` | `ip_address` |
| `zip`, `zip_us`, `postcode`, `postcode_uk` | `postal_code` |
| `gender` | `gender_indicator` |
| `fac`, `FAC` | `FAC` |
| **`crypto`, `bitcoin`, `ethereum`, `wallet`** *(new)* | `crypto` |
| **`bank_number`, `bank_account`, `us_bank_number`, `routing_number`, `routing`** *(new)* | `us_bank_number` |
| **`driver_license`, `us_driver_license`, `drivers_license`, `dl`, `license`** *(new)* | `us_driver_license` |

#### Evaluation metrics reported

| Metric | Description |
|---|---|
| Questions / Answered / Empty | Total count, non-empty responses, failures |
| Answer rate | Fraction of questions with non-empty LLM responses |
| Surrogate counts | Total found vs key total; averages per question |
| Precision / Recall / F1 / Accuracy | Overall surrogate detection quality |
| Error (miss rate) | Fraction of key PII values not detected |
| Stage timings | Average ms per pipeline stage |
| ResolvePass leak rate | Fraction of responses where a surrogate was not restored |
| Sanitisation quality | Fraction of questions where real PII reached the LLM |
| Per-entity-type breakdown | F1 / precision / recall per PII type |
| Presidio comparison (Table 1) | SS vs Presidio side-by-side per comparable type + overall |
| BERTScore comparison (Table 2) | Semantic utility preservation: SS vs Presidio vs no-anonymisation baseline |
| Ablation study (Table 4) | Per-stage F1 contribution across four pipeline configurations |

#### Presidio comparison table

The per-type comparison covers all types both systems can detect. Types SS detects that Presidio cannot are shown in a separate SS-Only table.

**Comparable types** (both SS and Presidio):

| Type | SS source | Presidio source |
|---|---|---|
| PERSON | EntityTrace / ContextGuard | Presidio NER |
| email | PatternScan | Presidio regex |
| phone | PatternScan | Presidio regex |
| ssn | PatternScan | Presidio regex |
| credit_card | PatternScan (Luhn) | Presidio regex (Luhn) |
| ip_address | PatternScan | Presidio regex |
| dob *(approximate)* | PatternScan | Presidio DATE_TIME |
| GPE *(approximate)* | EntityTrace | Presidio LOCATION |
| **crypto** *(new)* | PatternScan | Presidio CRYPTO |
| **us_bank_number** *(new)* | PatternScan (ABA checksum) | Presidio US_BANK_NUMBER |
| **us_driver_license** *(new)* | PatternScan (context-gated) | Presidio US_DRIVER_LICENSE |

**SS-Only types** (Presidio cannot detect these):

| Type | Notes |
|---|---|
| `api_key` | SK/Bearer/GHP/AKIA/AIzaSy prefixes |
| `address` | Structural street-address regex |
| `postal_code` | US ZIP + UK postcode |
| `gender_indicator` | Explicit gender declarations |
| `ORG` | Organisation names (NER) |
| `FAC` | Facility names (NER) |

#### BERTScore utility preservation (Table 2)

BERTScore (`roberta-large`) measures how well the semantic meaning of the original message is preserved after anonymisation. Higher F1 = better utility.

| Approach | Expected BERTScore F1 |
|---|---|
| No anonymisation (baseline) | 100% |
| SurrogateShield (realistic surrogates) | ~92–97%: type-consistent replacements preserve sentence structure |
| Presidio (placeholder redaction) | ~80–88%: `[ENTITY_TYPE]` tokens break semantic continuity |

Enable the `BERTScore SS` and `BERTScore Presidio` fields in JSON Test to generate data for this table. The `roberta-large` model (~1.4 GB) is downloaded automatically on first use and can take 15–30 minutes to score on CPU.

#### Ablation study (Table 4)

The ablation study quantifies how much each detection stage contributes to overall F1. It is computed **post-hoc from the existing answers file**: no pipeline re-run needed. Each answer already records which stage detected each entity via the `source` field in `pii_detail`.

Four pipeline configurations are simulated by combining stage contributions:

| Configuration | Detected set |
|---|---|
| PatternScan only | `pattern_scan_pii` |
| PatternScan + EntityTrace | `pattern_scan_pii` ∪ `entity_trace_pii` |
| PatternScan + ContextGuard | `pattern_scan_pii` ∪ `context_guard_pii` |
| Full cascade (all three) | `confirmed_pii` |

For each configuration, precision / recall / F1 are computed overall **and per entity type**. The evaluation screen shows:

- **Entity Detection Attribution panel**: how many entities each stage contributed and the percentage of cases where EntityTrace or ContextGuard were strictly necessary (i.e., PatternScan alone would have missed at least one key entity)
- **Overall Performance by Configuration table**: precision, recall, F1, TP/FP/FN for all four configs with delta-F1 annotations showing each stage's incremental gain
- **F1 Per Entity Type by Configuration table**: per-type breakdown with a *Key stage* column indicating which configuration first achieves ≥80% F1 for that type (PatternScan for structured types like `ssn` and `email`; EntityTrace for `PERSON`, `GPE`, `ORG`)
- **Ablation Summary panel**: absolute F1 improvements from adding each stage

To generate ablation data, enable the following fields in JSON Test before running:

| Field | Required for |
|---|---|
| `pattern_scan_pii` | Stage 1 contribution |
| `entity_trace_pii` | Stage 2 contribution |
| `context_guard_pii` | Stage 3 contribution |
| `confirmed_pii` | Full cascade (baseline) |

All four are **on by default** in JSON Test. If the answers file pre-dates the ablation feature, re-run JSON Test with those fields enabled.



## Attacker Experiment

The Attacker Experiment (`attacker.py`) simulates an **informed adversary** who intercepts the sanitised text SurrogateShield sends to the LLM API and actively tries to recover the original PII values. It is the adversarial counterpart to the Evaluation suite and answers a critical research question: does replacing PII with realistic-looking surrogates (rather than `[PLACEHOLDER]` tokens) make it easier for an attacker to infer the originals?

### What it tests

Two variants are run on each question from an existing answers file:

| Variant | What the attacker sees | Goal |
|---|---|---|
| **SurrogateShield** | Sanitised text with realistic fake values (fake names, SSNs, emails, …) | Try to recover the originals from the surrogates |
| **Presidio** | Redacted text with `[ENTITY_TYPE]` placeholder tokens | Try to recover the originals from the placeholders |

The attacker is given a carefully constructed adversarial prompt that discloses the PII types that were replaced and instructs the model to use every available inference technique; linguistic analysis, contextual reasoning, demographic inference, cross-field correlation, format patterns, and more.

### Expected result

**0% recovery for both systems.** Surrogates have no cryptographic or statistical relationship to the original values. This proves SurrogateShield achieves *equivalent inference resistance* to blunt placeholder redaction, while preserving significantly higher semantic utility as measured by BERTScore.

### Running the experiment

Press `A` in the dashboard. The four-screen flow will:

1. Explain the experiment
2. Prompt for an existing answers file from `experiment/` (must contain `surrogate_map`, `sanitized_input`, and optionally `presidio_sanitized_input` / `presidio_found_piis` fields)
3. Show question counts and estimated API calls, then run
4. Display a results summary on completion

Output files are written to `experiment/`:

| File | Contents |
|---|---|
| `<stem>_Attacker_Experiment.json` | Per-question recovery attempt details for both variants |
| `<stem>_Attacker_Experiment_Analysis.json` | Aggregated recovery rates, per-type breakdown, overall assessment |

The experiment supports **resume**: if interrupted, re-running with the same answers file picks up where it left off.

### Analysis metrics

| Metric | Description |
|---|---|
| Questions available | Questions with SS / Presidio data to attack |
| Total targeted | Total PII values the attacker attempted to recover |
| Total recovered | Values where the attacker's guess matched the original (exact, lowercased) |
| Recovery rate | `recovered / targeted`: lower is better for privacy |
| Recovery rate (excl. address) | Rate excluding address-type entities (service queries fuzz addresses rather than fully replace them, so proximity recovery is theoretically possible; tracked separately) |
| By-type breakdown | Per-PII-type targeted/recovered counts and rate |

### Address handling

Street addresses in service queries receive **house-number fuzzing** (`±2–8`) rather than full replacement. Exact address recovery is still impossible, but proximity-based recovery is theoretically possible. Address results are tracked separately via `address_recovered_count` / `non_address_recovered_count` and excluded from the primary recovery rate to keep the comparison fair.

### Generating compatible answers files

Enable these fields in JSON Test before running to ensure the experiment has full data:

| Field | Required for |
|---|---|
| `surrogate_map` | SS attacker: original PII values and their surrogates |
| `sanitized_input` | SS attacker: the text to attack |
| `pii_detail` | SS attacker: type metadata for the prompt |
| `presidio_sanitized_input` | Presidio attacker: the placeholder-redacted text |
| `presidio_found_piis` | Presidio attacker: type metadata for the prompt |

### Test coverage (`tests/test6.py`)

`test6.py` covers the full `attacker.py` module with mocked API calls; no real API key or network required:

| Test area | What is verified |
|---|---|
| `_build_types_list()` | Readable bullet formatting, label mapping, unknown-type fallback |
| `_types_from_pii_detail()` | Type normalization from SS `pii_detail` dicts |
| `_types_from_presidio_found()` | Presidio entity type mapping to internal types |
| `score_recovery()` | Exact-match scoring, address/non-address separation, null-guess handling |
| `run_attacker_call()` | API response parsing, JSON fence stripping, error fallback |
| `compute_analysis()` | Rate aggregation, per-type counts, address exclusion logic |
| `run_experiment()` | End-to-end flow with mocked API client and temp files |



## Running the Tests

```bash
# Activate your venv first
source .venv/bin/activate

# Unit + integration tests (no API key required)
python tests/test1.py
python tests/test2.py
python tests/test3.py
python tests/test6.py

# Python library tests (run from repo root, no venv needed if surrogateshield is installed)
python tests/test7.py
```

`test1.py` covers PatternScan, EntityTrace, SentinelLayer cascade, MimicGen, ShadowMap, ResolvePass, and a full no-API pipeline round-trip. All tests run without an API key.

`test6.py` covers the Attacker Experiment module (`attacker.py`); type formatting, recovery scoring, API response parsing, analysis aggregation, and end-to-end experiment flow. All API calls are mocked; no real API key or network access needed.

`test7.py` covers the standalone `surrogateshield` pip package; library entities, ShadowMap memory and persistent modes, response parser, state singletons, pipeline pii_off filtering, threshold wiring, ResolvePass, and all five public API functions (`config`, `scan`, `mask`, `unmask`, `flush`). 134 checks, no API key required.



## Python Library

SurrogateShield is also available as a standalone pip package; no dashboard, no CLI, just the core detection and masking pipeline as a clean Python API. The package is self-contained: it does not import from the main application and carries its own copies of the detection, generation, storage, and reconstruction modules.

```bash
pip install surrogateshield
```

Both the spaCy model (`en_core_web_lg`) and the ContextGuard transformer model (`dslim/distilbert-NER`) download automatically on first use; no separate download step needed. Each model is cached locally after the first run.


### Quick start

```python
import surrogateshield as shield

# Silence the default Rich tables (recommended for production)
shield.config(detailed_view=False)

user_text = (
    "Hi, I'm Sarah Mitchell. My email is sarah.mitchell@gmail.com, "
    "my SSN is 123-45-6789, and I was born on 04/12/1990."
)

sanitized = shield.mask(user_text)
# "Hi, I'm Rachel Torres. My email is torresrachel@yahoo.com,
#  my SSN is 876-32-1045, and I was born on 09/27/1983."

response = any_llm.chat(sanitized)   # send surrogates, not real PII
restored = shield.unmask(response)   # real values restored from the session map
shield.flush()                       # clear session before the next conversation
```


### Provider support

`unmask()` accepts native SDK response objects directly; no manual text extraction needed:

```python
# Anthropic
import anthropic
import surrogateshield as shield

client = anthropic.Anthropic()
sanitized = shield.mask(user_text)
response = client.messages.create(model="claude-opus-4-8", max_tokens=1024,
                                   messages=[{"role": "user", "content": sanitized}])
print(shield.unmask(response))   # passes Anthropic Message object directly
shield.flush()

# OpenAI
from openai import OpenAI
client = OpenAI()
sanitized = shield.mask(user_text)
response = client.chat.completions.create(model="gpt-4o",
                                          messages=[{"role": "user", "content": sanitized}])
print(shield.unmask(response))   # passes ChatCompletion object directly
shield.flush()

# Gemini
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-flash")
sanitized = shield.mask(user_text)
response = model.generate_content(sanitized)
print(shield.unmask(response))   # passes GenerateContentResponse directly
shield.flush()

# Any local model / plain string
raw_reply = ask_ollama(sanitized)
print(shield.unmask(raw_reply))  # plain strings work too
shield.flush()
```


### Public API

| Function | Description |
|---|---|
| `shield.config(**kwargs)` | Set thresholds, storage mode, pii_off list, and display options |
| `shield.mask(text)` | Detect all PII, replace with surrogates, store the mapping |
| `shield.unmask(response)` | Restore original PII in an LLM response (any provider object or plain string) |
| `shield.scan(text)` | Detect PII and return `{value: type}` dict: no substitution, no shadow map update |
| `shield.pii_finder` | Alias for `shield.scan` |
| `shield.flush()` | Clear the session shadow map and generate a new session ID |


### scan(): inspect without masking

```python
found = shield.scan(
    "Contact Alice Nguyen at alice@corp.com or call +1-415-555-0198."
)
# {"Alice Nguyen": "PERSON", "alice@corp.com": "email", "+1-415-555-0198": "phone_us"}

for value, pii_type in found.items():
    print(f"{pii_type:15s}  {value}")
```

`scan()` always returns every detected entity regardless of `pii_off` settings. It is safe to call in a read-only context; it never modifies the session.


### pii_off: suppress specific types

```python
shield.config(pii_off=["location", "org"])

sanitized = shield.mask(
    "Emma Johnson works at Deloitte in New York, email emma@deloitte.com."
)
# "Emma Johnson"      → replaced (PERSON not in pii_off)
# "emma@deloitte.com" → replaced (email not in pii_off)
# "Deloitte"          → kept  (org in pii_off)
# "New York"          → kept  (location in pii_off)
```

Available aliases: `phone`, `name`, `location`, `org`, `email`, `ssn`, `dob`, `address`, `zip`, `postcode`, `postal_code`, `credit_card`, `ip_address`, `api_key`, `crypto`, `bank`, `license`, `gender_indicator`. Raw type strings (`"PERSON"`, `"GPE"`, etc.) also accepted.


### Persistent shadow map

By default the session mapping is in-memory only. For web servers or multi-process deployments, point `pii_mem` at a directory and the map is encrypted to disk with AES-256-GCM:

```python
import os, surrogateshield as shield

os.makedirs("/var/app/shadowmaps", exist_ok=True)
shield.config(pii_mem="/var/app/shadowmaps")

sanitized = shield.mask("My name is Clara Oswald, phone 555-123-4567.")
restored  = shield.unmask(llm.chat(sanitized))
shield.flush()   # deletes the .shadowmap and .key files for this session
```


### Key config parameters

| Parameter | Default | Effect |
|---|---|---|
| `detailed_view` | `True` | Print Rich tables after each call; set `False` in production |
| `pii_mem` | `"temp"` | `"temp"` for in-memory; a directory path for AES-256-GCM disk persistence |
| `pii_off` | `[]` | PII types to detect but not replace |
| `service` | `True` | Minimal address fuzzing for location queries instead of full replacement |
| `context_guard_enabled` | `True` | Run the HuggingFace second-pass NER; set `False` for faster, spaCy-only detection |
| `fuzzy_threshold` | `85` | rapidfuzz score (0–100) used in the third reconstruction pass of `unmask()` |
| `spacy_model` | `"en_core_web_lg"` | spaCy model for EntityTrace; swap for `en_core_web_sm` for faster but lower-accuracy NER |


### What the library detects

The same 20-type detection cascade as the full application: email, SSN, phone (US / UK / international), credit card (Luhn-validated), street address, date of birth, IPv4, API keys, gender indicator, US ZIP, UK postcode, Bitcoin/Ethereum wallet addresses, ABA routing numbers, US driver's licenses, plus named entities PERSON, ORG, GPE, LOC, and FAC via the spaCy + distilbert-NER two-model pipeline.


See [`python-library/README.md`](python-library/README.md) for the complete API reference, all config parameters, surrogate generation details, and troubleshooting.



## Project Structure

```
SurrogateShield/
├── main.py                  # CLI entry point and interactive dashboard
├── pipeline.py              # End-to-end message pipeline orchestration
├── config.py                # All constants and thresholds (single source of truth)
├── util.py                  # Shared dataclasses (DetectedEntity, Conversation), logging helpers
├── settings_manager.py      # Persistent user settings (~/.surrogateshield/settings.json)
├── evaluator.py             # Precision/recall/F1 evaluation logic + Presidio/BERTScore/ablation study
├── json_tester.py           # Batch JSON question processing
├── attacker.py              # Adversarial PII recovery experiment (Attacker Experiment)
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
│   ├── example_answers.json # Sample output from JSON Test
│   ├── *_Attacker_Experiment.json          # Per-question attacker results (auto-generated)
│   └── *_Attacker_Experiment_Analysis.json # Aggregated recovery-rate analysis (auto-generated)
│
├── tests/                   # Test suite (no API key required)
│   ├── test1.py             # PatternScan, EntityTrace, cascade, MimicGen, ShadowMap, ResolvePass, round-trip
│   ├── test2.py             # Additional detection and generation tests
│   ├── test3.py             # Additional pipeline and storage tests
│   ├── test6.py             # Attacker Experiment — type formatting, scoring, analysis, end-to-end (mocked API)
│   └── test7.py             # Python library — entities, ShadowMap, ResolvePass, pipeline, full public API (134 checks)
│
├── python-library/          # Standalone pip package (surrogateshield)
│   ├── pyproject.toml       # Package metadata and dependencies
│   ├── README.md            # Full library API reference
│   └── surrogateshield/     # Package source — self-contained, zero imports from the main app
│       ├── __init__.py      # Public API: config, scan, mask, unmask, flush
│       ├── _state.py        # Module-level cfg and session singletons
│       ├── _display.py      # Optional Rich terminal output (graceful fallback if Rich absent)
│       ├── _response_parser.py  # extract_text() — Anthropic / OpenAI / Gemini / plain string
│       └── core/
│           ├── entities.py              # DetectedEntity, mask_spans, remove_span_overlap
│           ├── detection/
│           │   ├── pipeline.py          # run_cascade(), deduplicate(), pii_off alias table
│           │   ├── pattern_scan.py      # 17-pattern regex scanner with Luhn / ABA validators
│           │   ├── entity_trace.py      # spaCy NER (model-name-keyed cache)
│           │   ├── context_guard.py     # distilbert-NER second pass (optional)
│           │   ├── service_query.py     # Service-query detection + address fuzzing
│           │   ├── quasi_identifier.py  # Quasi-identifier combination risk scorer
│           │   └── geo_data.py          # Geographic pass-through whitelist
│           ├── generation/
│           │   └── mimic.py             # MimicGen — Faker-based surrogate generation
│           ├── storage/
│           │   └── shadow_map.py        # ShadowMap — memory mode + AES-256-GCM disk mode
│           └── reconstruction/
│               └── resolve.py           # ResolvePass — exact / component / fuzzy restoration
│
└── conversations/           # Runtime — auto-created on first use
    ├── <conv_id>.json        # Conversation history (surrogate text only, not originals)
    └── <conv_id>.shadowmap   # AES-256-GCM encrypted surrogate→original mapping
```



## Security Design

| Component | Mechanism |
|---|---|
| Device secret | 32-byte random key at `~/.surrogateshield/device.key`, `0o600` permissions |
| Per-conversation key | HKDF-SHA256 with device secret as IKM and conversation ID as salt |
| ShadowMap encryption | AES-256-GCM with fresh 12-byte nonce per write |
| ShadowMap format | `nonce (12 bytes) ‖ AES-GCM ciphertext`: unreadable without device key |
| API transmission | Only surrogates sent: real values never leave the device |
| Conversation history | Stored locally in `conversations/`; JSON holds surrogate text, not originals |
| `.gitignore` | `*.shadowmap`, `conversations/*.json`, `device.key`, `.env` excluded |



## Privacy Guarantees

- **No PII crosses the API boundary.** Every entity confirmed by SentinelLayer is replaced before the HTTP request is made.
- **Service queries get proportional protection.** A restaurant search near your home address gets the house number shifted; the city name is preserved so the answer is useful. Sensitive topic overrides (medical, legal, shelter) force full anonymisation regardless.
- **Geographic generality is preserved.** US states, countries, and major cities are never replaced; they provide no meaningful re-identification risk and destroying them would break answer quality.
- **Quasi-identifier risks are surfaced.** If your message contains combinations like ZIP+DOB+gender that are statistically re-identifying even without traditional PII, you are warned before the message is sent.
- **RAG documents are anonymised at index time.** Real PII never enters the vector store. Retrieval and context injection all operate on surrogates.
- **Financial and identity credentials are protected.** Bitcoin/Ethereum wallet addresses, ABA routing numbers, and driver's license numbers are detected and replaced alongside traditional PII.

---

Made with ❤️ by Sherwin Vishesh Jathanna
