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
      Masks PII before it leaves your device           

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


SurrogateShield replaces the personal data it detects with realistic, consistent surrogates before your message reaches an LLM provider, and puts the real values back in the reply. On the sealed test-3 benchmark of real chat messages it let 0.8% of 3,677 personal values through, against 9.1% for GLiNER-PII and 31.6% for Microsoft Presidio; the few places it trailed are listed in one line under [Limitations](#limitations). All cryptographic operations run locally. Measured results, each with its command, are in [`bench/results/README.md`](bench/results/README.md).



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
SentinelLayer (PatternScan + canonicaliser → EntityTrace → PII tagger
               [→ ContextGuard] → structural passes → relation gate → resolver)
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
LLM API  ← receives the masked text
    │
    ▼
ResolvePass → restore original values in response
    │
    ▼
Display to user
```



## Features

- **Layered PII detection**: regex patterns on the text and on rewritten views of it → a DeBERTa PII tagger for names, organisations and places → spaCy for places, with distilbert-NER as an optional extra pass; presets `balanced`, `strict`, `fast` and `classic`
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
- **Attacker Experiment**: simulates an informed adversary who intercepts sanitised API traffic and attempts to recover original PII from both SurrogateShield and Presidio output, and counts what it recovers from each
- **Standalone Python library**: the full detection and masking pipeline is also packaged as `surrogateshield` (`pip install ./python-library`), a self-contained API with no dashboard dependency, for embedding directly into any Python application or service



## Architecture

### 1. Detection: SentinelLayer

The stages run in sequence: PatternScan, the canonicaliser (the same patterns on rewritten views such as "priya dot raman at gmail dot com"), EntityTrace, ContextGuard and the PII tagger, then the structural passes, a relation gate and a resolver. Each stage reads the text with what earlier stages found masked out, so no two stages fight over the same value. Which stages run is set by the detection preset: `balanced` (default), `strict`, `fast` or `classic` (see [`CONFIGURATION.md`](CONFIGURATION.md)). The code lives in `python-library/surrogateshield/core/`; the top-level `detection/`, `generation/`, `storage/` and `reconstruction/` modules are aliases of it, or thin wrappers that apply `config.py`.

In `balanced`, the PII tagger (a DeBERTa-v3-small token classifier, weights in `$SURROGATESHIELD_MODELS`) reads names, organisations and places and adds to the patterns on phones, IDs, addresses, handles and credentials; spaCy reads places only; ContextGuard is off. On a machine without the tagger's weights, use the `classic` preset.

#### PatternScan (`detection/pattern_scan.py`)

Regex-based structural detection. Runs first so structured PII is masked before any NER model sees it. Validators (Luhn, ABA checksum) reject false positives before any entity is emitted.

| Pattern | Examples | Validator |
|---|---|---|
| Street address | `99 Cathedral Close`, `456 Innovation Plaza` | None |
| SSN | `512-44-7810` | None |
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

spaCy `en_core_web_lg` NER. Extracts `PERSON`, `GPE`, `LOC`, `ORG`, and `FAC` entities; in the default preset it is used for places only. Returns two tiers:

- **Confirmed**: score ≥ 0.90 (promoted immediately)
- **Borderline**: score 0.70–0.90 (passed to ContextGuard when it is on, otherwise kept at ≥ 0.75)

Includes ORG→GPE reclassification when location prepositions appear before an organisation name (e.g. "lives in Google").

#### ContextGuard (`detection/context_guard.py`)

`dslim/distilbert-NER` (~250 MB, downloaded once from HuggingFace Hub on first use, no server required). Off in the `balanced` and `fast` presets, on in `strict` and `classic`. Verifies borderline entities from EntityTrace and independently detects anything missed. Applies word-piece artefact cleaning and a blocklist of short / title tokens that commonly cause false positives.

#### Post-processing passes (`detection/logic.py`)

Four additional passes run on the combined entity set:

| Pass | What it does |
|---|---|
| A: Structural ORG | Regex for `[the/a/an] <name> [corporation|company|corp|inc|ltd|llc…]`; no name lists |
| B: Email-username reclassification | Corrects ORG→PERSON when the entity text is a prefix of a detected email username |
| C: PERSON component dedup | Removes standalone surnames that are sub-components of already-detected full names |
| D: Topical geo-entity filter | Drops a GPE/LOC only if it appears exclusively in knowledge-query sub-clauses |

#### Address parser (`detection/address_parser.py`) — v2

The canonical structured parser is the single source of truth for "what is an
address". It captures the FULL span — house number, pre/post directionals,
street name (including ordinals like "5th"), suffix, apartment/suite/unit,
city, state (full names + uppercase 2-letter abbreviations), and ZIP/ZIP+4 —
as ONE entity, so components are never surrogated independently. PO boxes,
multi-line mailing blocks, comma-free formats, and highway forms
("1500 Highway 50") are all recognized, and a two-tier suffix system plus a
plausibility validator rejects prose like "3 point turn" or "5 stars".

How the surrogate is built is controlled by `ADDRESS_MODE` in `config.py`:

| Mode | Behaviour |
|---|---|
| `"shift"` | Only the house number changes, by up to ±`ADDRESS_SHIFT_RANGE` (default ±1, ~one building). Street, city, state, ZIP, and formatting preserved **byte-for-byte**. |
| `"replace"` | Structure-preserving fake — every component replaced by a fake of the same shape, in the same slot. No real component reaches the LLM. |
| `"auto"` *(default)* | Context-aware: shift for service queries, replace otherwise. |

A shifted address never collides with another real address in the
conversation, house number 1 always shifts up (never to 0), and leading
zeros / alpha suffixes ("0123", "123A") are preserved.

#### ServiceQueryDetector (`detection/service_query.py`)

Identifies messages like "restaurants near 1126 E Apache Blvd, Tempe, AZ" and applies a lighter touch:

- City/state names are **not replaced**: the LLM needs them to give useful answers
- Drives the shift-vs-replace decision when `ADDRESS_MODE="auto"`
- A sensitive-topic override (medical, legal, shelter, immigration keywords) forces full anonymisation regardless of query structure
- No address is ever sent to a geocoder: v2 removed the Nominatim existence check, which sent the street to a third party and never changed the output

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
| `PERSON` | `Adya Nath` |
| `email` | `adya.nath@vance.io` |
| `ssn` | `XXX-XX-XXXX` (valid format) |
| `phone_us` | `+1-###-###-####` |
| `phone_uk` | `+44 7### ######` |
| `phone_intl` | `+49 8234 927461` |
| `address` (shift mode) | `790 Crescent Row, Springfield, IL` for `789 Crescent Row, Springfield, IL` — number ±1, everything else byte-identical |
| `address` (replace mode) | `412 Nguyen Row, South Debra, PR 90830` — same structure, every component fake |
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
| Key derivation | HKDF-SHA256: device secret as IKM, conversation ID as salt, file kind (`shadowmap-v1` / `transcript-v1`) as info |
| Binding | Kind and ID are AES-GCM associated data: a file only opens as what it was written for |
| Device secret | Generated once at `~/.surrogateshield/device.key` (`$SURROGATESHIELD_HOME` overrides the directory); `0o600`, directory `0o700`, loose permissions are tightened, a key of the wrong length is refused |
| File location | `~/.surrogateshield/conversations/<conv_id>.shadowmap`; IDs must match `^[A-Za-z0-9_-]{1,64}$` |
| Writes | Atomic (temp file, fsync, rename), files `0o600` |
| Missing file | Empty mapping |
| Corrupt file | Moved aside to `<name>.corrupt-<ts>` and an error is raised: never silently replaced by an empty map |
| Key unavailable | Error (`ShadowMapStorageError`); an ephemeral key is opt-in via `config.ALLOW_EPHEMERAL_KEY` |



### 4. Reconstruction: ResolvePass (`reconstruction/logic.py`)

Four-pass restoration of original values in LLM responses. Every pass matches whole values only ("Lee" never matches inside "Leeds"), and a span one pass rewrote is protected from the later ones:

1. **Exact replacement**: longest surrogate first, case-insensitive with the case carried over; handles the vast majority of cases
2. **Component matching**: for multi-word surrogates the model echoed in part (e.g. `790 Crescent Row` from a longer address), scoped to unresolved surrogates only to prevent corruption of adjacent text
3. **Fuzzy matching**: [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) `partial_ratio` with configurable threshold (default 85), for multi-word surrogates only, never numbers
4. **Given name / surname**: "Tell Adya" restores the first name of a person surrogate; a surname alone only after a title ("Dr. Nath")

Every failure is categorised as `exact_miss`, `fuzzy_hit`, or `fuzzy_miss` for research analysis.



### 5. RAG Integration (`chatbot/rag.py`)

Local Retrieval-Augmented Generation backed by [ChromaDB](https://www.trychroma.com/) and [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`).

- No server required; ChromaDB runs in-process with telemetry off, storing its index in `~/.surrogateshield/rag` (mode 0700; `RAG_DIR` in `config.py`)
- Documents are **anonymised through the full SentinelLayer pipeline before indexing**: detected PII is replaced before a document is indexed. Detection runs in segments of at most 20,000 characters; if detection fails, nothing is indexed
- Surrogate mappings from indexed documents are stored in the encrypted `rag_global` ShadowMap. In a RAG chat, a name that already appears in a document gets the document's surrogate, so retrieval matches it and quoted excerpts are restored in the answer
- Queries are anonymised before retrieval
- Retrieved context goes into the current request only; it is not stored in the conversation history or sent again on later turns
- `python main.py rag` lists the indexed documents. `python main.py rag --forget <file>` removes a document's chunks and every mapping that no other document still uses


### 6. Python Library (`python-library/surrogateshield`)

Every component above (the detection stages, MimicGen, ShadowMap and ResolvePass) lives in this self-contained pip-installable library. It imports nothing from the main application; the application is built on top of it. The public surface is a `Session` class plus the module functions `config`, `scan`, `mask`, `mask_result`, `unmask`, `forget` and `flush`, behind a single import line. See the [Python Library](#python-library) section below for usage.



## Supported LLM Providers

| Provider | Model | Env var required |
|---|---|---|
| Claude (default) | `claude-sonnet-4-6` (`SURROGATESHIELD_CLAUDE_MODEL`) | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-2.5-flash` (`SURROGATESHIELD_GEMINI_MODEL`; SDK `google-genai`) | `GEMINI_API_KEY` |
| ChatGPT | `gpt-4o-mini` (`SURROGATESHIELD_OPENAI_MODEL`) | `OPENAI_API_KEY` |
| Local (Ollama) | `llama3.2` (configurable) | None: runs fully offline |

Switch providers from the **Settings** menu inside the dashboard (press `S`).



## PII Types Detected

Thirteen public types; each groups the internal names the detectors report:

| Public type | Internal names |
|---|---|
| `PERSON` | `PERSON`, `person` |
| `ORG` | `ORG` |
| `LOCATION` | `GPE`, `LOC`, `FAC`, `implicit_location` |
| `ADDRESS` | `address`, `zip_us`, `postcode_uk` |
| `EMAIL` | `email` |
| `PHONE` | `phone_us`, `phone_uk`, `phone_intl` |
| `URL` | `url`, `hostname` |
| `HANDLE` | `handle` |
| `ID` | `id_number`, `ssn`, `passport`, `us_driver_license`, `license_plate`, `vin`, `credit_card`, `iban`, `us_bank_number`, `crypto` |
| `CREDENTIAL` | `credential`, `api_key` |
| `NETWORK` | `ip_address`, `mac_address` |
| `AGE` | `age` |
| `DATE_OF_BIRTH` | `dob` |

Two more labels exist: `GENDER` (`gender_indicator`) and `OTHER` (any label the map does not know, such as a plugin's own). Combination risk is scored separately: quasi-identifier sets per Sweeney k-anonymity.

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

# The PII tagger (default preset) reads its weights from
# $SURROGATESHIELD_MODELS (default ~/.cache/surrogateshield/models).
# ContextGuard (dslim/distilbert-NER, ~250 MB) is off by default; it
# downloads from HuggingFace Hub on first use only if you turn it on.

# Set your API key: create .env readable only by you, then add the key with
# an editor (not `echo`, which leaves it in your shell history)
touch .env && chmod 600 .env

# Launch
./run.sh
```

The first mask after start-up loads the models and takes a few seconds; later messages take tens of milliseconds.

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
bert-score>=0.3.13          # Utility preservation scoring (BERTScore comparison)
ollama>=0.1.8               # Local LLM (optional)
presidio-analyzer>=2.2.0    # Presidio comparison panel in PII Finder
presidio-anonymizer>=2.2.0  # Presidio anonymization (companion to analyzer)
```

> **spaCy model:** `en_core_web_lg` is required by EntityTrace (always) and by
> the optional Presidio comparison panel. One download covers both:
> `python -m spacy download en_core_web_lg`.
>
> **PII tagger weights:** read from `$SURROGATESHIELD_MODELS` (default
> `~/.cache/surrogateshield/models`). Without them the default preset raises
> `DetectorUnavailable`; the `classic` preset runs without them.
>
> **ContextGuard model:** `dslim/distilbert-NER` (~250 MB) is off by default
> and is downloaded from HuggingFace Hub on first use only when it is turned on.
>
> **BERTScore:** `bert-score` and its `roberta-large` model (~1.4 GB) are only
> needed if you enable BERTScore fields in JSON Test. The model is downloaded
> automatically on first use.

### Environment Variables

Create a `.env` file in the project root and make it private
(`chmod 600 .env`; `run.sh` warns when other users can read it). The keys are
read by python-dotenv; a variable already exported in your shell takes
precedence over the file.

```env
ANTHROPIC_API_KEY=sk-ant-...       # Required for Claude
ANTHROPIC_WORKSPACE_ID=wrkspc_... # Only for a key not scoped to a workspace
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
| `ENTITY_TRACE_HIGH_THRESHOLD` | `0.90` | spaCy score above which an entity is immediately confirmed |
| `ENTITY_TRACE_LOW_THRESHOLD` | `0.70` | spaCy score above which an entity is forwarded to ContextGuard |
| `CONTEXT_GUARD_ENABLED` | `False` | Run ContextGuard; off in the default preset, where the PII tagger reads names |
| `CONTEXT_GUARD_CONFIDENCE_THRESHOLD` | `0.70` | distilbert score required to confirm a borderline entity |
| `ENTITY_TRACE_FALLBACK_THRESHOLD` | `0.75` | Score used when ContextGuard is disabled to promote borderline entities |
| `FUZZY_MATCH_THRESHOLD` | `85` | rapidfuzz threshold for ResolvePass reconstruction |
| `ADDRESS_MODE` | `"auto"` | Address surrogation: `"auto"` (shift for service queries, replace otherwise), `"shift"` (house number ±range), or `"replace"` (structure-preserving fake) |
| `ADDRESS_SHIFT_RANGE` | `1` | Max house-number delta for shift mode (~one building at ±1) |
| `SERVICE_QUERY_DETECTION_ENABLED` | `True` | Suppress standalone city/state replacement for location queries; drives `ADDRESS_MODE="auto"` |
| `CONTEXT_GUARD_DEVICE` | `-1` | ContextGuard inference device: `-1` = CPU, `0+` = GPU id |
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

# Index a document into the RAG store; list or remove indexed documents
python main.py add-doc path/to/document.txt
python main.py rag
python main.py rag --forget document.txt
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
  { "input": "My name is Priya Raman and my SSN is 512-44-7810. What are Wyoming's tax benefits?" },
  { "input": "My email is priya.raman@northwind.io and phone is 480-555-1234. Draft a resignation letter." }
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

### Offline detection evaluation (no LLM, no API key)

`offline_eval.py` scores the detection cascade directly against a key file — no
LLM calls, nothing leaves the machine. It is the fastest way to measure
detection accuracy while iterating:

```bash
# Overall + per-type precision / recall / F1 on the seeded dev split
# (experiment/make_dataset.py; tune on dev, never on synth_test_key.json)
python offline_eval.py --key experiment/synth_dev_key.json

# Protection check: gold values that reach the sent text, intended or not
python offline_eval.py --key experiment/synth_dev_key.json --protection --json out.json

# Focus on one type, cap the run, or use exact (v1-comparable) matching
python offline_eval.py --key experiment/synth_dev_key.json --types address --limit 200
python offline_eval.py --key experiment/synth_dev_key.json --strict

# Lint a key file for corrupt ground-truth values (values absent from the
# question, addresses that don't parse) — no models needed
python offline_eval.py --key experiment/experiment_key.json --lint-key
```

`experiment/test_key.json` (1,124 questions) is the legacy key. The detector
was tuned on it and the key was edited in the same commits, and
`experiment/lint_key.py` reports label errors in it (B1–B3), so a score on
it measures fit, not accuracy. The 0.9996 F1 in
`experiment/offline_eval_v2.json` came from that key and is not a claim.
The numbers to quote come from the sealed test-3 split of real chat messages,
scored with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split test3 --out bench/results/realdata_test3.json`;
see [Measured results](#measured-results) and `bench/results/README.md`.

Default matching is span-aware: because v2 detects addresses as full spans
(`6720 Palm Dr, Phoenix, AZ`) while the ground truth annotates components
separately, a key value counts as found when a detected span contains it.
`--strict` falls back to exact string equality.

### Evaluation (Precision / Recall / F1)

Pair your question file with an answer-key file at `experiment/<name>_key.json`:

```json
[
  {
    "Question": "My name is Priya Raman and my SSN is 512-44-7810...",
    "Answer-Key": {
      "name": "Priya Raman",
      "ssn": "512-44-7810"
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

BERTScore (`roberta-large`, rescaled with its baseline) is computed for two pairs:
- **input fidelity:** the sanitised message against the original;
- **utility:** each arm's answer against the answer to the original message,
  when `clean_llm_response` is collected (synthetic data only).

Input fidelity favours realistic surrogates by construction, so only the
utility pair says whether answers stay useful.

Measured on 298 real chat prompts from the sealed test-3 split (responder Claude Sonnet 4.6, judge Claude Opus 5.5):

| arm | utility: answer vs answer to the original (rescaled F1) |
|---|---|
| SurrogateShield | **0.442** |
| GLiNER-PII | 0.369 |
| Presidio + Faker | 0.365 |
| Presidio (placeholders) | 0.358 |

Paired difference in utility: +0.077 (95 % CI +0.053 to +0.103) against Presidio + Faker and +0.072 (+0.046 to +0.098) against GLiNER-PII. A pairwise judge preferred SurrogateShield's answer by +0.449 against Presidio + Faker and +0.051 against GLiNER-PII (132 wins to 117).
Command (makes provider calls): `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live utility --split test3 --out bench/results/utility_realdata_test3.json`.

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

### What it measures

The experiment counts how many values an attacker model recovers from each
arm, exactly or in part. A value that an arm left in plain text is counted
as leaked, not recovered. Results are listed in
[`bench/results/README.md`](bench/results/README.md).

Measured on the sealed test-3 split: 636 values in real chat messages,
attacker Claude Opus 5.5. An attacker knows a value letter for letter when
the arm left it in plain text or when the guess equals it exactly:

| arm | left in plain text | recovered exactly | known letter for letter |
|---|---|---|---|
| SurrogateShield | 7 | 4 of 620 | **11 of 627 (1.8 %)** |
| GLiNER-PII | 73 | 6 of 556 | 79 of 629 (12.6 %) |
| Presidio + Faker | 201 | 8 of 428 | 209 of 629 (33.2 %) |
| Presidio | 201 | 14 of 428 | 215 of 629 (34.2 %) |
| LLM Guard | 364 | 19 of 264 | 383 of 628 (61.0 %) |

- SurrogateShield has the lowest exact recovery of every arm; in the
  conversation condition the attacker recovered 1 of 177.
- 98 partial guesses at parts the surrogates keep on purpose so the
  answer stays useful (an e-mail provider, a birth year, the kind of
  business in an organisation name, a URL's host layout) are reported
  separately.

Commands: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live attacker --split test3 --out bench/results/attacker_realdata_test3.json` (makes provider calls), then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json` (no provider calls).

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

Street addresses in service queries receive **house-number fuzzing** (±`address_shift_range`, default 1) rather than full replacement. Exact address recovery is still impossible, but proximity-based recovery is theoretically possible. Address results are tracked separately via `address_recovered_count` / `non_address_recovered_count` and excluded from the primary recovery rate to keep the comparison fair.

### Generating compatible answers files

Enable these fields in JSON Test before running to ensure the experiment has full data:

| Field | Required for |
|---|---|
| `surrogate_map` | SS attacker: original PII values and their surrogates |
| `sanitized_input` | SS attacker: the text to attack |
| `pii_detail` | SS attacker: type metadata for the prompt |
| `presidio_sanitized_input` | Presidio attacker: the placeholder-redacted text |
| `presidio_found_piis` | Presidio attacker: type metadata for the prompt |

### Test coverage (`tests/test_attacker.py`)

`test_attacker.py` covers `attacker.py` with a fake provider client; no real API key or network required:

| Test area | What is verified |
|---|---|
| Prompts | Neutral attacker prompts for both arms, placeholder-aware for Presidio, braces escaped |
| Models | The attacker model differs from the responder |
| Scoring | Match levels per value, leaked values and non-gold guesses kept separate, the same units for every arm |
| Failures | Failed or invalid attacker replies are counted as unavailable and excluded |
| Runner | Sampling, call budget, metadata and a deterministic plan |
| Screen | The attacker screen prints computed values only; old-format results files are refused |



## Running the Tests

```bash
# Fast, model-free suite (the default; no API key required)
.venv/bin/python -m pytest tests python-library/tests -q

# Tests that load spaCy / HuggingFace models, including the older
# check scripts in tests/, which run through tests/test_legacy_scripts.py
.venv/bin/python -m pytest tests python-library/tests -q -m heavy
```

Always run the tests through pytest. `tests/` holds 119 pytest modules,
one per area (detection stages, surrogates, ShadowMap, ResolvePass,
settings and the CLI, RAG, the attacker experiment, the benchmark code,
the docs), plus the older check scripts that `tests/test_legacy_scripts.py`
runs and gates on. `python-library/tests/test_public_api.py` covers the
library's public API. No test needs an API key; provider clients are fakes.



## Python Library

SurrogateShield is also available as a standalone pip package (version 2.1.0); no dashboard, no CLI, just the core detection and masking pipeline as a clean Python API. The package is self-contained: it does not import from the main application, and the application is built on it.

Install it from the repository root, then the spaCy model:

```bash
pip install ./python-library
python -m spacy download en_core_web_lg
```

The PII tagger used by the default `balanced` preset loads its weights from `$SURROGATESHIELD_MODELS` (default `~/.cache/surrogateshield/models`); if they are missing, detection raises `DetectorUnavailable` instead of sending unmasked text. The `classic` preset works without them. The ContextGuard model (`dslim/distilbert-NER`) is off by default and downloads from HuggingFace Hub on first use only when `context_guard_enabled=True` or a preset that uses it is chosen.


### Quick start

```python
import surrogateshield as shield

user_text = (
    "Hi, I'm Priya Raman. My email is priya.raman@northwind.io, "
    "my SSN is 512-44-7810, and I was born on 04/12/1990."
)

sanitized = shield.mask(user_text)
# for example:
# "Hi, I'm Yashica Ahluwalia. My email is yashica.ahluwalia@weaver.io,
#  my SSN is 856-14-3146, and I was born on 01/07/1992."

response = any_llm.chat(sanitized)   # the detected values are sent as surrogates
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
from google import genai
client = genai.Client()
sanitized = shield.mask(user_text)
response = client.models.generate_content(model="gemini-2.5-flash", contents=sanitized)
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
| `shield.config(**kwargs)` | Set the detection preset, thresholds, storage mode, pii_off list, and display options |
| `shield.mask(text)` | Detect PII, replace it with surrogates, store the mapping |
| `shield.mask_result(text)` | Like `mask`, but also returns the detections (including `pii_off` ones sent as typed) and the replacements made |
| `shield.unmask(response)` | Restore original PII in an LLM response (any provider object or plain string) |
| `shield.scan(text)` | Detect PII and return a list of `Detection` (text, type, start, end, score, source, masked); `as_dict=True` returns `{value: type}`. No substitution, no shadow map update |
| `shield.pii_finder` | Alias for `shield.scan` |
| `shield.forget(original)` | Erase every mapping for one original value from the session |
| `shield.flush()` | End the current session; the next call starts a new one |
| `shield.Session(...)` | An explicit session with its own settings and mappings, for servers and concurrent use |


### scan(): inspect without masking

```python
found = shield.scan(
    "Contact Alice Nguyen at alice@corp.com or call +1-415-555-0198."
)

for d in found:
    print(f"{d.type:15s}  {d.text}  ({d.source})")
# PERSON           Alice Nguyen  (pii_tagger)
# email            alice@corp.com  (pattern)
# phone_us         +1-415-555-0198  (pattern)

shield.scan("...", as_dict=True)
# {"Alice Nguyen": "PERSON", "alice@corp.com": "email", "+1-415-555-0198": "phone_us"}
```

`scan()` returns every detected entity; one whose type is in `pii_off` has `masked=False`. It is safe to call in a read-only context; it never modifies the session.


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
shield.flush()   # deletes this session's .shadowmap file
```


### Key config parameters

| Parameter | Default | Effect |
|---|---|---|
| `detection` | `"balanced"` | Detection preset: `"balanced"`, `"strict"`, `"fast"` or `"classic"`, a `DetectionConfig`, or a dict of one (see [`CONFIGURATION.md`](CONFIGURATION.md)) |
| `detailed_view` | `False` | Print Rich tables after each call; they show the original values |
| `pii_mem` | `"temp"` | `"temp"` for in-memory; a directory path for AES-256-GCM disk persistence |
| `pii_off` | `[]` | PII types to detect but not replace |
| `service` | `True` | Minimal address fuzzing for location queries instead of full replacement |
| `address_mode` | `"auto"` | `"auto"` (shift for service queries, replace otherwise), `"shift"` or `"replace"` |
| `context_guard_enabled` | `False` | Run the HuggingFace second-pass NER (`dslim/distilbert-NER`) |
| `fuzzy_threshold` | `85` | rapidfuzz score (0–100) used in the fuzzy reconstruction pass of `unmask()` |
| `spacy_model` | `"en_core_web_lg"` | spaCy model for EntityTrace; swap for `en_core_web_sm` for faster but lower-accuracy NER |


### What the library detects

The same detection pipeline as the full application, reporting the 13 public types listed under [PII Types Detected](#pii-types-detected): PERSON, ORG, LOCATION, ADDRESS, EMAIL, PHONE, URL, HANDLE, ID, CREDENTIAL, NETWORK, AGE and DATE_OF_BIRTH. PatternScan and the canonicaliser find the structured values, the PII tagger reads names, organisations and places, spaCy reads places, and ContextGuard can be switched on as an extra pass.


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
├── run.sh                   # Launcher: pre-flight checks, then the venv's python main.py
├── requirements.txt
│
├── detection/               # SentinelLayer — aliases or thin config.py wrappers of the library's detection modules
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
│   └── logic.py             # ResolvePass — four-pass surrogate→original restoration
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
├── tests/                   # Test suite, run with pytest (no API key required)
│   ├── conftest.py          # Shared fixtures
│   ├── test_*.py            # 119 pytest modules, one per area; model-loading ones are marked heavy
│   ├── test_legacy_scripts.py  # Runs the older check scripts below and fails on any failed check
│   └── test1.py … test5.py  # Older check scripts (run only through pytest)
│
├── python-library/          # Standalone pip package (surrogateshield)
│   ├── pyproject.toml       # Package metadata and dependencies
│   ├── README.md            # Full library API reference
│   ├── tests/
│   │   └── test_public_api.py   # The library's public API
│   └── surrogateshield/     # Package source — self-contained, zero imports from the main app
│       ├── __init__.py      # Public API: config, scan, mask, mask_result, unmask, forget, flush
│       ├── session.py       # Session, Detection, MaskResult
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
│           │   ├── pii_tagger.py        # PII tagger (DeBERTa-v3-small) for names, organisations, places
│           │   ├── canonical.py         # Canonicaliser: rewritten views for PatternScan
│           │   ├── relation_gate.py     # Decides whether a name is personal or public
│           │   ├── resolver.py          # Keeps one candidate when several stages report a value
│           │   ├── config.py            # DetectionConfig, presets, public types
│           │   ├── service_query.py     # Service-query detection + address fuzzing
│           │   ├── quasi_identifier.py  # Quasi-identifier combination risk scorer
│           │   └── geo_data.py          # Geographic pass-through whitelist
│           ├── generation/
│           │   └── mimic.py             # MimicGen — Faker-based surrogate generation
│           ├── storage/
│           │   └── shadow_map.py        # ShadowMap — memory mode + AES-256-GCM disk mode
│           └── reconstruction/
│               └── resolve.py           # ResolvePass — exact / component / fuzzy / given-name restoration
│
~/.surrogateshield/           # Runtime ($SURROGATESHIELD_HOME), 0700, created on first use
├── device.key                # 32-byte device secret, 0600
├── settings.json
└── conversations/
    ├── <conv_id>.json        # Encrypted transcript: display history (real values) + API history (surrogates)
    └── <conv_id>.shadowmap   # Encrypted surrogate→original mapping
```



## Security Design

| Component | Mechanism |
|---|---|
| Device secret | 32-byte random key at `~/.surrogateshield/device.key`, `0o600` permissions, directory `0o700` |
| Per-conversation key | HKDF-SHA256: device secret as IKM, conversation ID as salt, file kind as info |
| ShadowMap encryption | AES-256-GCM with fresh 12-byte nonce per write; kind and ID as associated data |
| ShadowMap format | `"SSv1" ‖ nonce (12 bytes) ‖ AES-GCM ciphertext`: unreadable without device key |
| API transmission | Detected values are replaced by surrogates before the request is made |
| Conversation history | Stored locally in `~/.surrogateshield/conversations/`, encrypted like the ShadowMap; it contains the real values you typed (display history) as well as the surrogate text sent to the API |
| `.gitignore` | `*.shadowmap`, `conversations/`, `device.key`, `.env`, and new experiment outputs (`experiment/*_answers*.json`, `*_eval_results*.json`, `*_Attacker_Experiment*.json`) excluded |



## Measured results

Measured on the sealed test-3 split: real chat messages from ShareGPT,
WildChat and oasst1 that were never used for tuning. Every number, the
commit it was run at and the per-dataset breakdown are in
[`bench/results/README.md`](bench/results/README.md). All commands below
start with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m`.

**Personal values let through (3,677 values placed in real messages).**
`bench.realdata.score --split test3 --out bench/results/realdata_test3.json`

| arm | values leaked | prompts sent clean | odd-format values leaked | harmless text edited |
|---|---|---|---|---|
| SurrogateShield | **0.8 %** (30 of 3,677) | **97.4 %** | **0.2 %** (1 of 530) | **13.9 %** |
| GLiNER-PII (threshold 0.3) | 7.3 % | 79.5 % | 7.5 % | 41.3 % |
| GLiNER-PII (threshold 0.5) | 9.1 % | 75.4 % | 8.3 % | 35.4 % |
| Presidio / Presidio + Faker | 31.6 % | 30.8 % | 43.6 % | 32.6 % |
| Presidio + RoBERTa | 54.0 % | 11.7 % | 66.0 % | 22.0 % |
| LLM Guard | 56.3 % | 9.1 % | 61.5 % | 16.4 % |

- SurrogateShield leaked 11× fewer values than GLiNER-PII and 39× fewer
  than Presidio; the gap to GLiNER-PII at 0.5 is −8.3 points (95 % CI
  −9.3 to −7.3).
- Of 2,433 real messages with nothing to protect, it left 2,153 (88.5 %)
  untouched, against 2,041 for LLM Guard, 1,725 for Presidio + RoBERTa,
  1,450 for Presidio and 704 for GLiNER-PII at 0.5.

**A second, outside dataset (Nemotron-PII, 1,000 records, 4,627 values).**
`bench.realdata.score --external nemotron_pii --out bench/results/external_nemotron_pii.json`
- Leaked 14.5 % with F1 0.886, against GLiNER-PII 22.5–25.8 % (F1
  0.751–0.758), Presidio 40.1 % (0.716), Presidio + RoBERTa 56.6 % (0.594)
  and LLM Guard 61.0 % (0.530).

**Answer quality (298 prompts; makes provider calls).**
`bench.realdata.live utility --split test3 --out bench/results/utility_realdata_test3.json`
- BERTScore of the answer against the answer to the original: 0.442,
  against 0.369 for GLiNER-PII, 0.365 for Presidio + Faker and 0.358 for
  Presidio. See [BERTScore utility preservation](#bertscore-utility-preservation-table-2).

**Multi-turn conversations (60 conversations; makes provider calls).**
`bench.realdata.live multiturn --split test3 --out bench/results/multiturn_realdata_test3.json`
- Real values restored in the answer: 81 of 82, against 0 of 80 for
  Presidio + Faker and 0 of 54 for GLiNER-PII.
- Real values sent upstream over the conversation: 5 of 150, against 78
  for Presidio + Faker and 30 for GLiNER-PII.

**Attacker (636 values; makes provider calls).**
`bench.realdata.live attacker --split test3 --out bench/results/attacker_realdata_test3.json`, then
`bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json` (with `PYTHONPATH=.:python-library`)
- Values an attacker knows letter for letter: 1.8 % (11 of 627), against
  12.6 % for GLiNER-PII, 33.2 % for Presidio + Faker, 34.2 % for Presidio
  and 61.0 % for LLM Guard. See [What it measures](#what-it-measures).

**Speed and memory.** `bench.perf_arms --out bench/results/perf_arms_v4.json`

| arm | median latency | peak memory |
|---|---|---|
| Presidio | 8.2 ms | 1,084 MB |
| Presidio + Faker | 8.4 ms | 1,010 MB |
| SurrogateShield | 43.5 ms | 1,348 MB |
| LLM Guard | 58.0 ms | 1,565 MB |
| GLiNER-PII (0.3 / 0.5) | 79.9 / 82.1 ms | 2,333 / 2,318 MB |
| Presidio + RoBERTa | 83.5 ms | 1,611 MB |

All ten checks in
`PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict --split test3 --out bench/results/verdict_test3.json`
hold.

## Limitations

**Limitations:** it trailed on the 96 real values in natural text (29 leaked; GLiNER-PII 16–20, Presidio 27), on some single types (3 of 13 on test-3, 8 of 13 on Nemotron-PII, URL most of all) and on speed and memory (plain Presidio is about 5× faster and lighter); it came out ahead on every other test. Every number, with its command, is in [`bench/results/README.md`](bench/results/README.md).

## Privacy Guarantees

- **Detected PII is replaced before any request is made.** Every entity confirmed by SentinelLayer is replaced before the HTTP request is made, and a missing detection model stops the request (fail closed).
- **Service queries get proportional protection.** A restaurant search near your home address gets the house number shifted; the city name is preserved so the answer is useful. Sensitive topic overrides (medical, legal, shelter) force full anonymisation regardless.
- **Geographic generality is preserved.** US states, countries, and major cities are never replaced; they provide no meaningful re-identification risk and destroying them would break answer quality.
- **Quasi-identifier risks are surfaced.** If your message contains combinations like ZIP+DOB+gender that are statistically re-identifying even without traditional PII, you are warned before the message is sent.
- **RAG documents are anonymised at index time.** Detected PII is replaced before a document is indexed. Retrieval and context injection operate on the anonymised text.
- **Financial and identity credentials are protected.** Bitcoin/Ethereum wallet addresses, ABA routing numbers, and driver's license numbers are detected and replaced alongside traditional PII.

---

Made with ❤️ by Sherwin Vishesh Jathanna
