# SurrogateShield Documentation (v2.1.0)

> This is the plain-Markdown copy of [docs.html](docs.html), for AI agents and other tools that read text. It has the same content as the web page. Links of the form `(#section)` point to the `<a id="section">` anchors below.

## Contents

- **Getting Started:** [Overview](#overview) · [Installation](#installation) · [Environment Variables](#env-vars) · [Quick Start](#quick-start) · [First Run](#first-run)
- **Results:** [Overview](#results) · [Method](#benchmark-method) · [Leak Rate by Type](#results-types) · [Harmless Text](#results-natural) · [Nemotron-PII](#results-nemotron) · [Answer Quality](#results-utility) · [Multi-turn](#results-multiturn) · [Attacker Test](#results-attacker) · [Speed and Memory](#results-speed) · [Presets and Parts](#results-presets) · [Tools Compared](#competitors) · [Reproduce](#reproduce) · [Limitations](#limitations)
- **How It Works:** [Pipeline Overview](#pipeline) · [Service Queries](#service-query) · [SentinelLayer](#sentinellayer) · [PatternScan](#patternscan) · [Canonicaliser](#canonicaliser) · [EntityTrace](#entitytrace) · [ContextGuard](#contextguard) · [PIITagger](#piitagger) · [Structural Passes](#postprocessing) · [Relation Gate](#relation-gate) · [Resolver](#resolver) · [Fail-Closed Behaviour](#fail-closed) · [MimicGen](#mimicgen) · [ShadowMap](#shadowmap) · [ResolvePass](#resolvepass)
- **Dashboard & CLI:** [Dashboard Commands](#cli) · [Keyboard Shortcuts](#keyboard-shortcuts) · [Chat Mode](#chat-mode) · [PII Finder](#pii-finder) · [RAG Mode](#rag-mode) · [Settings Screen](#settings-screen) · [LLM Providers](#providers)
- **Evaluation Tools:** [JSON Batch Test](#json-test) · [Evaluation](#evaluation-mode) · [Offline Evaluation](#offline-eval) · [Attacker Experiment](#attacker-experiment) · [Benchmark Harness](#bench-harness)
- **PII Types:** [PII Types](#pii-types) · [Examples](#pii-examples) · [Quasi-identifier Warnings](#quasi-identifier)
- **Configuration:** [Dashboard Config](#config-constants) · [Presets](#presets) · [DetectionConfig](#detection-config) · [type_actions](#type-actions)
- **Python Library:** [Library Installation](#library-install) · [Library Quick Start](#library-quickstart) · [Sessions](#library-sessions) · [API Reference](#library-api) · [scan() and Detection](#library-scan) · [mask_result() and MaskResult](#library-mask-result) · [Sending Types As Is](#library-pii-off) · [Addresses and Service Queries](#library-address) · [Persistent Sessions and Storage](#library-persistent) · [Async](#library-async) · [Using It With Providers](#library-providers) · [Multi-Turn Conversations](#library-multiturn) · [Configuration](#library-config) · [Detector Plugins](#library-plugins) · [Command Line](#library-cli) · [Errors](#library-errors) · [FastAPI Example](#library-fastapi)
- **Reference:** [Security Design](#security-design) · [What Is Sent](#privacy-guarantees) · [Project Structure](#project-structure) · [Running Tests](#running-tests) · [Troubleshooting](#troubleshooting) · [FAQ](#faq)

<a id="overview"></a>

SurrogateShield is a privacy proxy for large language models. Before a message goes to a model, it finds the personal values in it (names, e-mails, phone numbers, addresses, IDs, credentials and more) and replaces each one with a realistic surrogate of the same kind. The map from surrogate to original stays on your machine. When the answer comes back, your real values are put back in. On real chat messages it let **0.8%** of personal values through, against **9.1%** for GLiNER-PII and **31.6%** for Microsoft Presidio.

### What it does to a message

**You type:**

> Hi, I'm Priya Raman. Email me at priya.raman@northwind.io or call +1 415 555 0142. My order NW-48213-K hasn't arrived at 1847 Alder Street, Portland. Can you draft a complaint to Northwind Outfitters?

**The provider receives:**

> Hi, I'm Adya Nath. Email me at adya.nath@vance.io or call +1 213 581 8333. My order VW-02898-Z hasn't arrived at 6237 James Street, Olympia. Can you draft a complaint to Northwind Outfitters?

The model writes its complaint letter for "Adya Nath". `unmask()` turns every surrogate in the letter back into your own value, so the letter you read is signed "Priya Raman". The company you are complaining to is kept, because it is only mentioned and not tied to you. Surrogates differ from run to run unless you pass a seed.

### Why surrogates and not [REDACTED]

A tag such as `[PERSON]` tells the model a value was there but not what kind. The model then writes tags back, loses track of who is who, or asks for the missing data. A realistic surrogate keeps the sentence natural. The model reasons as it would with the real values, and its answer can be mapped back. On the benchmark this is visible: SurrogateShield's answers scored 0.442 BERTScore against the answer to the original prompt, against 0.365 for Presidio + Faker and 0.369 for GLiNER-PII. Over 60 multi-turn conversations, it restored 81 of 82 values. The placeholder tools restored none.

### Two ways to use it

| Form | What you get | Install |
|---|---|---|
| Python library + CLI | The engine as a Python API: `Session`, `mask`, `unmask`, `scan`, `mask_result`, `forget`, `flush`, `config`, async twins, presets, plugins. Also a `surrogateshield` command with `scan`, `mask`, `unmask`, `bench` and `doctor`. | `pip install ./python-library` |
| Terminal dashboard | A full terminal app built on the library: chat with Claude, Gemini, ChatGPT or a local Ollama model, PII Finder, RAG over your own files, batch tests, evaluation against a key file, and the Attacker Experiment. | `pip install -r requirements.txt` |

The dashboard imports the library. It does not have its own copy of the detection, generation or restore code, so both forms behave the same.

### At a glance

| What | SurrogateShield | Best alternative tested |
|---|---|---|
| Personal values leaked (real chat text, 3,677 values) | **0.8%** | GLiNER-PII 7.3% (tuned) / 9.1% (default); Presidio 31.6% |
| Values in unusual formats leaked (530) | **0.2%** | GLiNER-PII 7.5%; Presidio 43.6% |
| Harmless messages left untouched (2,433) | **88.5%** | LLM Guard 83.9%; Presidio 59.6%; GLiNER-PII 28.9% |
| Leaked on Nemotron-PII (4,627 values) | **14.5%**, span F1 0.886 | GLiNER-PII 22.5%, F1 0.751 |
| Answer quality (BERTScore, 298 prompts) | **0.442** | GLiNER-PII 0.369; Presidio + Faker 0.365 |
| Multi-turn values restored | **81 / 82** | 0 for both Presidio + Faker and GLiNER-PII |
| Values an attacker model knows exactly | **1.8%** | GLiNER-PII 12.6%; Presidio 34.2% |
| Median time per message (CPU) | 43.5 ms | Presidio 8.2 ms; GLiNER-PII 82.1 ms |

The full tables and the method are in [Results](#results); the one place-by-place list of where it trailed is in [Limitations](#limitations).

### Where to start

| You want to… | Read |
|---|---|
| Chat with an LLM from the terminal | [Installation](#installation) → [First Run](#first-run) → [Chat Mode](#chat-mode) |
| Protect prompts in your own Python code | [Library Installation](#library-install) → [Quick Start](#library-quickstart) → [API Reference](#library-api) |
| Mask text from a shell script | [Library CLI](#library-cli) |
| Change what gets detected or replaced | [Presets](#presets) → [DetectionConfig](#detection-config) → [type_actions](#type-actions) |
| Add your own detector | [Plugins](#library-plugins) |
| Know how it works inside | [Pipeline Overview](#pipeline) and the stages after it |
| Check the numbers yourself | [Results](#results) → [Reproduce](#reproduce) |
| See where it trailed | [Limitations](#limitations) |
| See exactly what is sent to the provider | [What Is Sent](#privacy-guarantees) |

---

<a id="installation"></a>

## Installation

### Requirements

- Python 3.9 or newer (the test matrix covers 3.9 and 3.13)
- pip and a virtual environment
- A few GB of disk for torch, transformers and the spaCy model `en_core_web_lg`
- About 1.4 GB of RAM while the default preset runs (measured RSS 1,348 MB)
- No GPU. Everything runs on the CPU; a GPU index can be set per stage with `device`
- An API key for the provider you chat with (dashboard only). The library never calls an LLM provider itself

> **Install into a virtual environment.** Installing into the system or base conda Python is the most common cause of "package not found" and "model not found" errors. `run.sh` picks `.venv/bin/python` first, then `venv/bin/python`, then `python3`, and warns when no virtual environment is found.

### Install the dashboard (and the library with it)

### 1. Clone the repository

```bash
git clone https://github.com/sherwinvishesh/SurrogateShield.git
cd SurrogateShield
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your prompt.

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

This installs the library in editable mode (`-e ./python-library`), the spaCy model `en_core_web_lg` 3.8.0 from its wheel URL, the Anthropic client, Presidio, BERTScore and the terminal UI. The first install takes a few minutes.

### 4. Install the PII tagger's weights

```bash
PYTHONPATH=.:python-library python -m bench.tagger.install bench/tagger/build/models/dv3s-40k
```

The default preset uses a DeBERTa-v3-small token classifier, `pii-tagger-dv3s-40k`. Its weights are not downloaded from anywhere: they are a local folder. The command copies the folder to `$SURROGATESHIELD_MODELS/pii-tagger-dv3s-40k` (default `~/.cache/surrogateshield/models`), checks the weights against the SHA-256 pin `f9e5821151ac…` and prints the revision. If you do not have the weights, use the `classic` preset (step 6).

### 5. Add your API key

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
chmod 600 .env
```

Only the provider you use needs a key. See [Environment Variables](#env-vars) for every variable. `run.sh` warns when `.env` is readable by other users.

### 6. Check the install

```bash
surrogateshield doctor --smoke
```

`doctor` prints the active preset and its hash, each stage with its model and thresholds, and one `[ ok]` / `[warn]` / `[FAIL]` row per check: Python version, required packages, the spaCy model and licence, the ContextGuard cache, the tagger folder and pin, and the permissions of `~/.surrogateshield` (0700) and `device.key` (0600). `--smoke` masks a sample e-mail and SSN. Without the tagger folder the tagger row is `[FAIL]`. Then either install the weights or run with `SURROGATESHIELD_PRESET=classic`, which needs only spaCy and distilbert-NER.

### 7. Optional extras

```bash
pip install chromadb sentence-transformers   # RAG mode
pip install google-genai                      # Gemini
pip install openai                            # ChatGPT
pip install ollama                            # local models through Ollama
```

### Install only the library

```bash
pip install ./python-library
python -m spacy download en_core_web_lg
PYTHONPATH=.:python-library python -m bench.tagger.install bench/tagger/build/models/dv3s-40k
surrogateshield doctor
```

The library is installed from the repository folder. Its `pyproject.toml` pulls in faker, cryptography, rapidfuzz, spaCy, transformers and torch. The extra `pip install "./python-library[display]"` adds `rich`. See [Library Installation](#library-install).

### ContextGuard model

ContextGuard (`dslim/distilbert-NER`, revision `dfa2838a…`, about 250 MB) is **off** in the default preset. It is used by `strict` and `classic`, or when you set `context_guard_enabled=True`. It is loaded from the Hugging Face cache at its pinned revision. If it is not cached, it is downloaded the first time a stage needs it, unless `HF_HUB_OFFLINE=1` is set. In that case detection fails closed with `DetectorUnavailable`. To fetch it ahead of time, run one scan with it switched on while online, for example `surrogateshield scan --preset strict "warm up"`.

### Full dependency list

| Package | Version | Needed by | Purpose |
|---|---|---|---|
| surrogateshield | 2.1.0 (editable) | both | The library: detection, surrogates, storage, restore |
| faker | ≥24 | library | Locale-aware fallbacks for surrogate values |
| cryptography | ≥42 | library | AES-256-GCM and HKDF-SHA256 for the ShadowMap and transcripts |
| rapidfuzz | ≥3.6 | library | Fuzzy matching in ResolvePass |
| spacy | ≥3.7 (app pins <3.9) | library | EntityTrace NER |
| en_core_web_lg | 3.8.0 | library | The spaCy model (MIT) |
| transformers | ≥4.40 | library | PII tagger and ContextGuard |
| torch | ≥2.0 | library | Inference for both transformer stages |
| rich | ≥13.7 | app, library extra `display` | Terminal UI and tables |
| typer | ≥0.12 | app | The `main.py` command line |
| anthropic | ≥0.25.0 | app | Claude client |
| python-dotenv | ≥1.0.0 | app | Loads `.env` |
| pyarrow | ≥15 | app / bench | Reading benchmark data files |
| presidio-analyzer, presidio-anonymizer | ≥2.2 | app | The Presidio comparison panel and fields |
| bert-score | ≥0.3.13 | app | BERTScore in the JSON test and the evaluator |
| numpy, scipy | ≥1.26, ≥1.13 | app | Statistics in the evaluator |
| chromadb, sentence-transformers | ≥0.4, ≥2.7 | optional | RAG mode (vector store, `all-MiniLM-L6-v2` embeddings) |
| google-genai, openai, ollama | any recent; ollama ≥0.1.8 | optional | The Gemini, ChatGPT and local providers |
| pytest, pytest-cov | ≥8, ≥5 | dev | The test suite |

<a id="env-vars"></a>

## Environment Variables

The dashboard loads `.env` from the project folder with python-dotenv at start-up. The library reads only the variables marked "library". It never reads `.env` itself.

### Provider keys and models (dashboard)

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Needed for Claude chat and the Attacker Experiment. If it is missing, the chat stops with "ANTHROPIC_API_KEY is not set. Add it to your .env file." |
| `ANTHROPIC_WORKSPACE_ID` | — | Optional. Sent as the `anthropic-workspace-id` header |
| `GEMINI_API_KEY` | — | Needed for Gemini |
| `OPENAI_API_KEY` | — | Needed for ChatGPT |
| `SURROGATESHIELD_CLAUDE_MODEL` | `claude-sonnet-4-6` | The Claude model used for chat |
| `SURROGATESHIELD_GEMINI_MODEL` | `gemini-2.5-flash` | The Gemini model |
| `SURROGATESHIELD_OPENAI_MODEL` | `gpt-4o-mini` | The OpenAI model |
| `LOCAL_LLM_MODEL` | `llama3.2` | The Ollama model |
| `LOCAL_LLM_HOST` | `http://localhost:11434` | The Ollama server |
| `SURROGATESHIELD_ATTACKER_MODEL` | none | The model that plays the attacker. It must be set, and it must differ from the chat model, so a model never attacks its own answers |
| `SURROGATESHIELD_LOG_LEVEL` | `WARNING` | Log level of the dashboard. Detailed View raises it to INFO |

### Detection and storage (library and dashboard)

| Variable | Default | Effect |
|---|---|---|
| `SURROGATESHIELD_PRESET` | unset (`balanced`) | `fast`, `balanced`, `strict` or `classic`. It applies when no detection config is set in code. When it is set, the dashboard uses it instead of the detection values in `config.py` |
| `SURROGATESHIELD_DETECTION_CONFIG` | unset | Path to a JSON detection config, partial or whole. It is merged onto the preset. See [DetectionConfig](#detection-config) |
| `SURROGATESHIELD_HOME` | `~/.surrogateshield` | Holds `device.key`, `sessions/` (CLI maps), `conversations/`, `settings.json` and `rag/`. It is created with mode 0700 |
| `SURROGATESHIELD_MODELS` | `~/.cache/surrogateshield/models` | Where the PII tagger's folder is looked up |
| `HF_HUB_OFFLINE` | unset | When it is `1`, ContextGuard never downloads. An uncached model then fails closed |
| `TRANSFORMERS_OFFLINE` | unset | Set by `offline_eval.py` and the benchmark arms, together with `HF_HUB_OFFLINE` |
| `HF_HOME` / `HF_HUB_CACHE` | Hugging Face defaults | Where cached transformer models live |
| `PYTHONHASHSEED` | — | The live benchmark scripts set it to `0` and re-exec themselves so runs repeat |

### Example .env

```bash
ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...
# OPENAI_API_KEY=sk-...
# SURROGATESHIELD_CLAUDE_MODEL=claude-sonnet-4-6
# SURROGATESHIELD_ATTACKER_MODEL=claude-opus-5-5
# SURROGATESHIELD_PRESET=strict
# SURROGATESHIELD_LOG_LEVEL=INFO
```

> **Keep `.env` private.** Run `chmod 600 .env`. It is git-ignored; never commit it.

<a id="quick-start"></a>

## Quick Start

### Dashboard in four commands

```bash
source .venv/bin/activate
./run.sh             # opens the dashboard
./run.sh chat        # straight into a new chat
./run.sh pii-finder  # see what would be masked, with no API call
```

Type a message with personal details, for example *"Hi, I'm Priya Raman, call me on +1 415 555 0142."* With Detailed View on (the default), you see the detection table, what was sent, what came back, and the restored answer.

### Library in ten lines

```python
import anthropic
import surrogateshield as ss

client = anthropic.Anthropic()
prompt = "I'm Priya Raman (priya.raman@northwind.io). Draft a complaint about order NW-48213-K."

masked = ss.mask(prompt)                       # surrogates in, real values stay here
reply = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": masked}])
print(ss.unmask(reply))                        # the reply with your real values back
ss.flush()                                     # forget this conversation's map
```

### CLI in two commands

```bash
$ echo "Call Priya Raman on +1 415 555 0142" | surrogateshield mask
Call Adya Nath on +1 213 581 8333
session: 3f2c9a...                    # printed on stderr; surrogates differ per run

$ echo "I'll call Adya Nath at +1 213 581 8333." | surrogateshield unmask -s 3f2c9a... --close
I'll call Priya Raman at +1 415 555 0142.
```

<a id="first-run"></a>

## First Run

```bash
./run.sh
```

`run.sh` does four things before it starts the app:

- changes to its own folder, so it works from anywhere;
- picks the interpreter: `.venv/bin/python`, then `venv/bin/python`, then `python3` (with a warning);
- warns if `.env` is readable by other users and suggests `chmod 600 .env`. It never reads the file;
- checks that `en_core_web_lg` is installed. If not, it prints the install steps and exits with code 1.

Then it runs `exec python main.py "$@"`, so `./run.sh chat --rag` is the same as `python main.py chat --rag`.

### What happens the first time

- `~/.surrogateshield/` is created with mode 0700, and a 32-byte `device.key` is written into it with mode 0600. This key encrypts every map and transcript. Losing it makes them unreadable.
- `settings.json` is created with the defaults: provider Claude, Detailed View on, Presidio Comparison off.
- Each model is loaded the first time a stage needs it and then cached for the process, so the first message is slower than the rest. After that, masking takes tens of milliseconds per message (median 43.5 ms on the benchmark machine).

### The dashboard

The dashboard shows the banner, your saved conversations (if any) and a pipeline overview. Press `N` to start a conversation and type a message with personal details. You will see:

- **Detection table**: each value with its type, score, the stage that found it and its surrogate;
- **Quasi-identifier warning**: shown when values such as a ZIP code, birth date and gender appear together;
- **API Transparency panel**: the text sent, the text received and the final text (each shortened to 300 characters);
- **The answer**, with your real values restored, in a panel titled with the provider's name.

Borderline values ask *"Replace this? [y/N]"* before anything is sent. Type `exit`, `quit` or `back` to return to the dashboard.

---

<a id="results"></a>

## Results — How SurrogateShield Compares

Every tool was run on the same real user messages from three public chat datasets (ShareGPT, WildChat and oasst1), and the same scorer judged all of them. The test split is sealed: the scorer refuses to run on it unless its frozen hash is present, and all tuning was done on separate development splits. On that benchmark SurrogateShield let **30 of 3,677 personal values (0.8%)** reach the provider. The next best tool, GLiNER-PII, let through 9.1% at its default threshold. Microsoft Presidio let through 31.6%.

| Tool | Values leaked | Prompts sent fully clean | Odd-format leak | Harmless text edited |
|---|---|---|---|---|
| SurrogateShield (balanced) | **0.8%** (30 / 3,677) | **97.4%** (1,031 / 1,059) | **0.2%** (1 / 530) | 13.9% |
| GLiNER-PII @0.3 (tuned) | 7.3% | 79.5% | 7.5% | 41.3% |
| GLiNER-PII @0.5 (default) | 9.1% | 75.4% | 8.3% | 35.4% |
| Presidio | 31.6% | 30.8% | 43.6% | 32.6% |
| Presidio + Faker | 31.6% | 30.8% | 43.6% | 32.6% |
| Presidio + RoBERTa (obi/deid_roberta_i2b2) | 54.0% | 11.7% | 66.0% | 22.0% |
| LLM Guard | 56.3% | 9.1% | 61.5% | 16.4% |

That is **11× fewer leaked values than GLiNER-PII** and **39× fewer than Presidio**. The gap to GLiNER-PII at its default is −8.3 percentage points, with a 95% interval of [−9.3, −7.3].

- **Values leaked**: placed values with any letter or digit left unedited, or copied into the surrogate that covers it. A half-masked value counts as leaked. Lower is better.
- **Prompts sent fully clean**: prompts and conversations in which no placed value leaked, out of 1,059 (1,308 messages). Higher is better.
- **Odd-format leak**: the 530 values in 144 prompts written in unusual formats, such as phone numbers in words, e-mails written "name at host dot com" or ordinal dates. Lower is better.
- **Harmless text edited**: the share of a tool's edits that overlap no value that had to or could be hidden. SurrogateShield made 612 such edits out of 4,405. Lower is better.

> **Recompute it.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split test3 --out bench/results/realdata_test3.json`. Every command, commit and file hash is listed in `bench/results/README.md`.

<a id="benchmark-method"></a>

## How the Benchmark Works

The benchmark is called *test-3*. It was built so that no tool, including SurrogateShield, could be tuned to it.

### 1. Real messages, sealed

User prompts from ShareGPT, WildChat and oasst1 were filtered (15–400 words, at most 3 user turns, duplicates removed) and drawn into a held-out split. Its hash is frozen in `bench/realdata/test3/FREEZE.json`, and `bench.realdata.score` refuses to score a sealed split without it. Development and tuning used the separate `dev` and `devlarge` splits.

### 2. Injected slice — 3,677 known values

Fake identities, with values of all 13 types, were generated locally from a seeded generator and placed into real prompts that had no personal data, by exact span edits. Claude Sonnet 4.6 chose where each value goes but never invented a value. So the scorer knows exactly what must be hidden. The word lists the values are built from (first names, surnames, streets, cities, companies, mail providers, password words) are split in half by a keyed hash. The PII tagger was trained on the other half, so no name, street or company in test-3 was seen in training. Digits are random draws, and structural words such as street types and months are shared.

### 3. Odd-format subset — 530 values

40 prompts per dataset (144 in all) use formats a regex may not expect: lower-case, ALL-CAPS or surname-first names, SSNs without dashes, phones with dots or in words, cards and IBANs spaced in fours, dates and ages written out, an e-mail written "name at domain dot com". A third of them put the details in a JSON object. This tests whether a tool reads what a value is or only its usual shape.

### 4. Natural slice — 2,433 untouched messages

Real prompts exactly as their authors wrote them, with nothing added. Claude Sonnet 4.6 labelled them with a written guide that has four classes: *protect*, *sensitive*, *optional* and *keep*. The slice counts how much harmless text each tool edits (every needless edit can hurt the answer). It also holds 96 real personal values in the protect class, which test what real users actually type.

### 5. One run, every tool, same scorer

Seven tool configurations ran offline on the same text: SurrogateShield (balanced), GLiNER-PII at 0.5 and 0.3, Presidio, Presidio + Faker, Presidio + RoBERTa and LLM Guard. A value counts as leaked if any letter or digit of it is left unedited, or is copied into its surrogate. Rates carry Wilson 95% intervals, and differences between tools use a paired bootstrap with 2,000 resamples. Every arm writes the hash of its config, and a run that mixes two hashes fails.

The benchmark config replaces every address (`address_mode="replace"`). The service-query address shift of the default `auto` mode is therefore not part of the leak figures.

<a id="results-types"></a>

## Leak Rate by Type

Percentage of placed values of each type that reached the provider. SurrogateShield is lowest or joint-lowest on 10 of the 13 types and leaks nothing on 7 of them. It is behind GLiNER-PII on PERSON and LOCATION, and behind Presidio on URL.

| Type (values) | SS | GLiNER 0.5 | GLiNER 0.3 | Presidio | + RoBERTa | LLM Guard |
|---|---|---|---|---|---|---|
| ADDRESS (197) | **0** | 8.6 | 7.1 | 97.0 | 93.9 | 100 |
| AGE (326) | **0.3** | 0.9 | 0.6 | 40.8 | 69.3 | 100 |
| CREDENTIAL (195) | **0** | 7.2 | 5.1 | 95.4 | 96.9 | 76.9 |
| DATE_OF_BIRTH (190) | **0** | 6.3 | 5.8 | 7.9 | 38.4 | 100 |
| EMAIL (300) | **0** | 3.0 | 2.3 | 8.3 | 8.3 | 8.3 |
| HANDLE (202) | **0** | 6.9 | 3.5 | 93.1 | 86.6 | 82.7 |
| ID (248) | **1.2** | 33.1 | 25.4 | 22.2 | 22.2 | 52.0 |
| LOCATION (217) | 2.8 | 6.5 | **1.4** | 15.7 | 65.9 | 93.1 |
| NETWORK (196) | **0** | 18.4 | 16.3 | **0** | 19.9 | 19.9 |
| ORG (254) | **3.5** | 8.7 | 5.5 | 85.4 | 73.6 | 91.7 |
| PERSON (867) | 1.2 | 0.3 | **0.1** | 10.6 | 76.5 | 21.3 |
| PHONE (272) | **0** | 3.7 | 3.3 | 9.2 | 9.2 | 5.9 |
| URL (213) | 0.5 | 46.9 | 45.1 | **0** | **0** | 99.5 |

<a id="results-natural"></a>

## Harmless Text Left Alone

A tool that edits everything leaks nothing, but the answer suffers. On the 2,433 natural messages, SurrogateShield left more messages completely untouched than any other tool and made the fewest edits overall.

| Tool | Messages untouched (of 2,433) | Edits to harmless text / all edits |
|---|---|---|
| SurrogateShield | **2,153 (88.5%)** | **1,053 / 1,389** |
| LLM Guard | 2,041 (83.9%) | 1,391 / 1,710 |
| Presidio + RoBERTa | 1,725 (70.9%) | 2,259 / 2,749 |
| Presidio | 1,450 (59.6%) | 3,438 / 4,044 |
| Presidio + Faker | 1,450 (59.6%) | 3,425 / 4,030 |
| GLiNER-PII @0.5 | 704 (28.9%) | 6,012 / 6,697 |
| GLiNER-PII @0.3 | 486 (20.0%) | 7,789 / 8,496 |

LLM Guard and Presidio + RoBERTa leave many messages alone because they detect little; they also leaked more than half of the placed values.

<a id="results-nemotron"></a>

## Outside Data: Nemotron-PII

A second check on data nobody here wrote: 1,000 records from the test file of NVIDIA's Nemotron-PII (4,627 values, 250 records per stratum), with its labels mapped to our 13 types. SurrogateShield again leaked the least and had the best span F1.

| Tool | Values leaked | Span F1 |
|---|---|---|
| SurrogateShield | **14.5%** | **0.886** |
| GLiNER-PII @0.3 | 22.5% | 0.751 |
| GLiNER-PII @0.5 | 25.8% | 0.758 |
| Presidio | 40.1% | 0.716 |
| Presidio + Faker | 40.2% | 0.716 |
| Presidio + RoBERTa | 56.6% | 0.594 |
| LLM Guard | 61.0% | 0.530 |

SurrogateShield's leak rate per stratum was 11.9%, 12.5%, 16.9% and 16.8%. Leaks are higher here than on test-3 because Nemotron-PII labels many kinds of values that SurrogateShield does not treat as personal in a chat prompt. Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --external nemotron_pii --out bench/results/external_nemotron_pii.json`.

<a id="results-utility"></a>

## Answer Quality

Hiding values only helps if the answer is still good. For 298 prompts, each tool's masked prompt was sent to `claude-sonnet-4-6`, and the answer was compared with the answer to the original prompt.

| Tool | BERTScore vs the original answer |
|---|---|
| SurrogateShield | **0.442** |
| GLiNER-PII | 0.369 |
| Presidio + Faker | 0.365 |
| Presidio | 0.358 |

- SurrogateShield over Presidio + Faker: **+0.077** [+0.053, +0.103]. Over GLiNER-PII: **+0.072** [+0.046, +0.098].
- A blind judge (`claude-opus-5-5`) preferred SurrogateShield's answer over Presidio + Faker's 195 times, the other 62 times, with 39 ties: **+0.449** [+0.355, +0.540]. Against GLiNER-PII it preferred SurrogateShield's answer 132 times, GLiNER-PII's 117 times, with 47 ties: +0.051 [−0.051, +0.152]. The answer-quality check reads BERTScore, word by word, and reports the judge beside it.
- On 150 natural prompts, SurrogateShield edited 23, Presidio 59 and GLiNER-PII 108. BERTScore: a rerun of the original prompt 0.508, SurrogateShield 0.490, Presidio 0.440, GLiNER-PII 0.365. SurrogateShield is close to the noise floor of simply asking twice.

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live utility --split test3 --out bench/results/utility_realdata_test3.json` (makes provider calls).

<a id="results-multiturn"></a>

## Multi-turn Conversations

60 conversations (20 per dataset) were run turn by turn. A value introduced in turn 1 has to be masked every time it comes back, and restored every time the model repeats it.

| Tool | Values restored in the answer | Judge: correct / partly correct | Real value sent upstream |
|---|---|---|---|
| SurrogateShield | **81 / 82** | **32 / 18** | **5 / 150** |
| Presidio + Faker | 0 / 80 | 0 / 19 | 78 / 150 |
| GLiNER-PII | 0 / 54 | 2 / 5 | 30 / 150 |

Placeholder tools cannot restore: the model writes `[PERSON]` back and the user has to fill it in. Judge preference for SurrogateShield: **+0.576** [+0.467, +0.685] against Presidio + Faker and **+0.649** [+0.527, +0.770] against GLiNER-PII. Command: `... -m bench.realdata.live multiturn --split test3 --out bench/results/multiturn_realdata_test3.json`.

<a id="results-attacker"></a>

## Attacker Test

Could a strong model work out the real values from what each tool sent? `claude-opus-5-5` was given each tool's masked text and asked to name the original of every value, on 636 values (60 prompts per dataset).

| Tool | Values left in plain text | Recovered exactly from the masked text | Attacker knows the value, letter for letter |
|---|---|---|---|
| SurrogateShield | **7** | **4 / 620** | **11 / 627 (1.8%)** |
| GLiNER-PII | 73 | 6 / 556 | 79 / 629 (12.6%) |
| Presidio + Faker | 201 | 8 / 428 | 209 / 629 (33.2%) |
| Presidio | 201 | 14 / 428 | 215 / 629 (34.2%) |
| LLM Guard | 364 | 19 / 264 | 383 / 628 (61.0%) |

- The attacker knew **7× fewer values** with SurrogateShield than with GLiNER-PII and **19× fewer** than with Presidio.
- SurrogateShield's exact recovery is the lowest of all five tools, on each dataset and pooled. All 4 of its exact recoveries were URLs whose surrogate changed one part of the host and kept the path.
- In the conversation condition, the attacker recovered 1 of 177 values exactly and knew 3 of 179.
- The attacker also guessed 98 parts of values that a surrogate keeps on purpose (38 e-mail domains, 19 organisation words such as "Pharmacy", 18 URL hosts, 14 address parts, 8 birth years, 1 area code). They are reported apart, not counted as recovered.

A value counts as recovered only when the guess equals it letter for letter. The scored run, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live attacker --split test3 --out bench/results/attacker_realdata_test3.json`, makes provider calls. The exact-recovery reading is recomputed from its cached outcomes with no provider call: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json`.

<a id="results-speed"></a>

## Speed and Memory

Median time to mask one message and peak memory, on CPU, same machine, same messages.

| Tool | p50 per message | Peak RSS |
|---|---|---|
| Presidio | 8.2 ms | 1,084 MB |
| Presidio + Faker | 8.4 ms | 1,010 MB |
| SurrogateShield | 43.5 ms | 1,348 MB |
| LLM Guard | 58.0 ms | 1,565 MB |
| GLiNER-PII @0.3 | 79.9 ms | 2,333 MB |
| GLiNER-PII @0.5 | 82.1 ms | 2,318 MB |
| Presidio + RoBERTa | 83.5 ms | 1,611 MB |

SurrogateShield is about 2× faster than GLiNER-PII and uses about 1 GB less memory. Presidio is about 5× faster, and leaks 39× more. Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms_v4.json`.

<a id="results-presets"></a>

## Presets and What Each Part Is Worth

The same benchmark, run with each preset. "Untouched" is the number of the 2,433 natural messages left unedited.

| Preset | Values leaked | Natural messages untouched | p50 per message |
|---|---|---|---|
| strict | 0.2% | 1,264 | 67.1 ms |
| balanced (default) | 0.8% | 2,153 | 47.1 ms |
| fast | 0.8% | 2,147 | 38.2 ms |
| classic (no tagger weights) | 2.5% | 1,885 | 29.2 ms |

And with one part of the balanced pipeline removed at a time (injected leak):

| Removed | Values leaked |
|---|---|
| nothing (balanced) | 0.8% |
| pattern rules | 1.4% |
| canonicaliser | 3.9% |
| PIITagger | 10.8% |
| every model | 15.7% |
| every regex broken | 26.1% |

Every part pulls its weight. The tagger matters most, and the regex rules and the model cover for each other. Commands: `bench.realdata.operating_curve --split test3`, `bench.realdata.components --split test3` and `bench.realdata.regex_ablation --split test3`.

<a id="competitors"></a>

## The Tools Compared

Each competitor ran as its authors ship it, with the settings in its documentation or model card, unless a row says otherwise. The model-based competitors are pinned in `bench/arms/requirements-baselines.txt`, and every arm ran offline.

| Arm | What runs | Replacement | Settings |
|---|---|---|---|
| SurrogateShield | This library, `benchmark()` config (balanced with every address replaced, hash `a1abaa92142b0db9`) | Realistic surrogates, restored afterwards | Defaults |
| GLiNER-PII @0.5 | `urchade/gliner_multi_pii-v1` (Apache-2.0), gliner 0.2.29, zero-shot NER with the model card's labels | `[LABEL]` | Threshold 0.5 (the model card's); 200-word windows, 30-word overlap |
| GLiNER-PII @0.3 | The same model at its best development setting | `[LABEL]` | Threshold 0.3, chosen by a sweep of 0.3–0.6 × two label sets on the dev split before test-2 was opened, so it is the tool's tuned best |
| Presidio | Presidio as shipped, default recognisers, spaCy `en_core_web_lg` | `[ENTITY_TYPE]` | Score threshold 0.4; overlaps go to the longest span, then the highest score |
| Presidio + Faker | The same detection | A Faker value of the same type, the same value always getting the same fake within a message | Presidio's own "synthesize" sample calls OpenAI, so Faker stands in for it locally |
| Presidio + RoBERTa | Presidio's documented transformers engine with `obi/deid_roberta_i2b2` as the NER model | `[ENTITY_TYPE]` | Threshold 0.4 |
| LLM Guard | llm-guard 0.3.16, the `Anonymize` input scanner with a fresh Vault per message, `Isotonic/deberta-v3-base_finetuned_ai4privacy_v2` plus its regex recognisers | Faker values (`use_faker=True`) | Threshold 0.5. A crash in 0.3.16's Faker map for some types (e.g. CRYPTO) is wrapped so the scanner can run |

The live tests (answer quality, multi-turn, attacker) used `claude-sonnet-4-6` as the model that answers, and `claude-opus-5-5` as the judge and the attacker, so the attacker is a different, stronger model than the responder.

<a id="reproduce"></a>

## Reproduce the Numbers

Every figure on this page comes from a JSON file in `bench/results/`. `bench/results/README.md` lists, for each file, the command, the commit and the config hash. The offline commands make no provider calls. The live ones do and need `ANTHROPIC_API_KEY`.

| Result | File | Command (after `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m`) |
|---|---|---|
| Leak, clean, spurious, per type, natural | `realdata_test3.json` | `bench.realdata.score --split test3 --out bench/results/realdata_test3.json` |
| Nemotron-PII | `external_nemotron_pii.json` | `bench.realdata.score --external nemotron_pii --out bench/results/external_nemotron_pii.json` |
| Speed and memory | `perf_arms_v4.json` | `bench.perf_arms --out bench/results/perf_arms_v4.json` |
| Answer quality (live) | `utility_realdata_test3.json` | `bench.realdata.live utility --split test3 --out bench/results/utility_realdata_test3.json` |
| Multi-turn (live) | `multiturn_realdata_test3.json` | `bench.realdata.live multiturn --split test3 --out bench/results/multiturn_realdata_test3.json` |
| Attacker (live) | `attacker_realdata_test3.json` | `bench.realdata.live attacker --split test3 --out bench/results/attacker_realdata_test3.json` |
| Attacker, exact recovery (no provider call) | `attacker_exact_test3.json` | `bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json` (needs `PYTHONPATH=.:python-library`) |
| Component ablations | `realdata_components_test3.json` | `bench.realdata.components --split test3 --out bench/results/realdata_components_test3.json` (needs `PYTHONPATH`) |
| Presets curve | `operating_curve_test3.json` | `bench.realdata.operating_curve --split test3 --out bench/results/operating_curve_test3.json` (needs `PYTHONPATH`) |
| Regex ablation | `realdata_regex_ablation_test3.json` | `bench.realdata.regex_ablation --split test3 --out bench/results/realdata_regex_ablation_test3.json` |
| All ten checks (every one holds) | `verdict_test3.json` | `bench.realdata.verdict --split test3 --out bench/results/verdict_test3.json` (needs `PYTHONPATH`) |

### Re-running the tools

The scorer reads saved spans. To produce them again, build the inputs and run every arm. SurrogateShield and the two Presidio arms run in `.venv`; GLiNER, Presidio + RoBERTa and LLM Guard run in `.venv-baselines` (Python 3.12, `bench/arms/requirements-baselines.txt`).

```bash
python -m bench.arms.inputs --collection test3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.arms.run --natural --collection test3
```

> **Building a new split costs money.** `bench.realdata.pull` downloads the datasets. `bench.realdata.label` and `bench.realdata.inject` call Claude through the Batch API; both have `--estimate` and `--pilot` modes, and they save results only with `--write`.

<a id="limitations"></a>

## Limitations

**Limitations:** it trailed on the 96 real values in natural text (29 leaked; GLiNER-PII 16–20, Presidio 27), on some single types (3 of 13 on test-3, 8 of 13 on Nemotron-PII, URL most of all) and on speed and memory (plain Presidio is about 5× faster and lighter); it came out ahead on every other test.

---

<a id="pipeline"></a>

## Pipeline Overview

Every message goes through the same path, in the library and in the dashboard. Detection is a cascade called **SentinelLayer** (the function `run_cascade()` in `core/detection/pipeline.py`). Each stage reads the text with what the earlier stages found already masked out, so a later stage never fights over a value an earlier stage owns. All paths below are under `python-library/surrogateshield/core/`.

- **ServiceQuery** (detection/service_query.py): Classifies the message as `none`, `service` ("pharmacies near 1847 Alder Street") or `service_coarse` (a service query on a sensitive topic). It decides the address mode, and in a service query it stops the model stages from reporting places, so the city stays real and the answer stays useful.
- **PatternScan** (detection/pattern_scan.py · rules): Regular expressions with checksums (Luhn, ABA, IBAN mod-97, VIN) and context gates for e-mails, phones, IDs, cards, IBANs, IPs, MACs, API keys, credentials, dates of birth, ages, ZIP and UK postcodes, URLs, handles, crypto addresses and street addresses. Every hit scores 1.0.
- **Canonicaliser** (detection/canonical.py · rules): Runs the same patterns on rewritten views of the text: number words as digits, "at"/"dot" e-mails, worded dates, look-alike characters folded, spaced digit groups joined. Catches "four one five, five five five…" and "priya dot raman at gmail dot com".
- **EntityTrace** (detection/entity_trace.py · spaCy en_core_web_lg 3.8.0): spaCy named-entity recognition for PERSON, ORG, GPE, LOC and FAC. In the default preset it is used for places only. It runs on the text with every pattern hit masked out.
- **ContextGuard** (detection/context_guard.py · dslim/distilbert-NER): Confirms borderline spaCy entities whose spaCy score reaches `accept` (0.70), then runs its own transformer NER and adds what it finds: confirmed at or above `accept`, otherwise flagged for confirmation. **Off** in `balanced` and `fast`; on in `strict` and `classic`. When it is off, borderline entities at or above the fallback threshold (0.75) are kept.
- **PIITagger + plugins** (detection/pii_tagger.py · DeBERTa-v3-small): A token classifier trained on generated text. It reads names, organisations, places and address parts, and backs up the patterns for phones, IDs, handles and credentials at high thresholds. Your plugins run here too, with the same routing and thresholds.
- **Structural passes** (detection/pipeline.py, structural.py, org_assembly.py): Layout rules: names after "my name is", titles, greetings and form labels; organisation names read whole; addresses read part by part; CSV columns; chat speakers; "PAYMENT TO" lines; display names in e-mail headers. They also drop implausible organisations and merge split names.
- **Relation gate** (detection/relation_gate.py): Keeps a name as typed when nothing ties it to a person: a city that is only the topic ("How far is Tempe from Phoenix?"), a public figure, a well-known company, an acronym. A place or company tied to someone ("we live near Reno", "I work at Halvorsen Dental") is masked.
- **Resolver** (detection/resolver.py): When several stages report the same value, keeps one: best source rank, then the type-conflict table, then the highest score. Overlapping spans are cut so each character belongs to one value.
- **MimicGen** (generation/mimic.py, identity.py): Makes a realistic surrogate of the same kind and shape for each value: a linked fake person whose name, e-mail and handle agree; a phone with the same country code and layout; a card that passes Luhn; a date of birth in the same format. The same original always gets the same surrogate within a session.
- **ShadowMap** (storage/shadow_map.py · AES-256-GCM): Holds the surrogate → original map for the session. In memory by default; with `pii_mem` or `--store` it is written to disk encrypted with a key derived from your device secret, file mode 0600.
- **LLM provider** (your code / chatbot/providers.py): Receives the masked text and answers it as usual.
- **ResolvePass** (reconstruction/resolve.py · rapidfuzz): Puts your values back into the answer in four passes: exact match, multi-word fragments, fuzzy match for respelled names, and name parts ("Ms. Nath" → "Ms. Raman").

### The exact order inside run_cascade()

| # | Step | What it does |
|---|---|---|
| 1 | PatternScan | All pattern families. URLs are read first; non-personal URLs become opaque so nothing inside them is touched later (except secrets in query parameters and e-mails) |
| 2 | Canonicaliser | Pattern hits on the rewritten views are added. The remaining text has every pattern hit and every opaque URL masked |
| 3 | EntityTrace | spaCy on the remaining text |
| 4 | ContextGuard or fallback | Confirms borderline entities, or, when off, keeps those at or above the fallback threshold |
| 5 | PIITagger and plugins | Each gets the text and the canonical view. Candidates are routed by `type_sources` and per-type thresholds. Tagger ages must have an age's shape. Places are dropped in a service query. Any entity inside a canonical-view hit is removed |
| 6 | Anchor and snap | Spans are kept on one line and snapped to word boundaries; quotes and leading prepositions are stripped, and one-word PERSONs and ORGs are extended |
| 7 | Passes A, E, F, G, G2, O, H, B, C | The structural passes (table below) |
| 8 | Opaque filter | Drops entities that overlap an opaque URL |
| 9 | Pass D | GPE and LOC only, skipped in a service query: drops places that only appear in a question's sub-clause, and lower-case places mid-sentence unless the writer writes in lower case; a place anchored to the writer is kept |
| 10 | Sentence frame | Model PERSONs are trimmed to the name itself (`trim_person`); one left empty is dropped (reason `sentence_frame`) |
| 11 | Pass R: relation gate | Model-stage entities and every PERSON go through the gate, unless their type is bypassed. Private persons stay; places and organisations not tied to a person are dropped (reason `not_tied_to_person`) |
| 12 | Pass S: structural rules | When the structural stage is on: adds and removes spans from layout (`structural.py`); its spans are routed and opaque-filtered like the rest |
| 13 | Gender pass | A gender word that matches a named person is left as written (reason `gender_follows_name`); other gender words are masked |
| 14 | Overlap cut | Partial overlaps go to the better source rank, then the longer span, then the earlier one |
| 15 | Quasi-identifier scoring | Logs risky combinations (see [Quasi-identifiers](#quasi-identifier)) |
| 16 | `pii_off` and `keep` | Marks those types as not masked |
| 17 | Return | `(confirmed, needs_confirmation)` |

Every candidate, from any stage, passes the same router: the stage must be allowed to report that type (`type_sources`), and its score must reach the stage's threshold for that type. A stage that runs past its `max_latency_ms` budget is logged as `<stage>_over_budget_ms`, and its candidates are still kept.

<a id="service-query"></a>

## Service Queries & Address Modes

Some messages are only useful with a real location: "Is there a late-night pharmacy near 1847 Alder Street, Portland?" Replacing Portland with Olympia makes the answer useless. SurrogateShield recognises these **service queries** and handles the address differently.

### How a service query is recognised

- A sentence with a request intent ("find", "recommend", "looking for", "is there", "open now", a ranking word such as "best" or "nearest" right before a venue word, or a question opener such as "what", "which", "where") **and** a place in the same sentence ("near …", "close to …", "within 5 miles of …", "near me", or "in Tempe" next to a venue word); or
- a short fragment (15 words or fewer, not in the first person) that names a venue and a place ("dentist near 316 Citrus Blvd").
- If the message also mentions a sensitive topic, it becomes `service_coarse`. The topic words are HIV, AIDS, STD/STI, abortion, rehab(il…), addiction, mental health, psychiatr(y/ist…), therapy, therapist, counsel(ling/lor…), domestic violence, shelter, homeless, immigration, undocumented, substance abuse, overdose, suicide, self-harm, eating disorder and detox, matched anywhere inside a word.

### Address modes

| Mode | Outside a service query | In a service query | In a sensitive service query |
|---|---|---|---|
| `auto` (default) | replace | shift | coarse |
| `shift` | shift | shift | shift |
| `replace` | replace | replace | replace |

**shift — "pharmacies near":**

> Late-night pharmacies near 1847 Alder Street, Portland?  
→ Late-night pharmacies near 1848 Alder Street, Portland?

**coarse — sensitive topic:**

> Rehab clinic near 316 Citrus Blvd, Orlando  
→ Rehab clinic near my area, Orlando

- **shift** moves the house number by ±1 up to ±`address_shift_range` (default 1), in random order. If every number in that range collides, it widens up to range + 8. It never goes below 1, and it keeps leading zeros and letter suffixes ("12B"). If no shift works, it falls back to replace. The street and city stay real, so the answer is about the right place.
- **coarse** turns the street line into "my area" and removes the ZIP code. The city stays, so the answer is still local without pointing at your door.
- **replace** redraws every part in the same shape: a real town, another two-letter state if one was there, a ZIP+4 if one was there.
- `service=False` (library) or `SERVICE_QUERY_DETECTION_ENABLED = False` (dashboard) turns recognition off, so `auto` always replaces.
- In a service query, places from the model stages are not reported, so "Portland" stays. Places in other messages are masked when tied to you.

> **Always replace?** Set `address_mode="replace"` (library) or `ADDRESS_MODE = "replace"` (dashboard) and every address gets a new street, number and city. The benchmark ran this way.

<a id="sentinellayer"></a>

## SentinelLayer — Detection

SentinelLayer is the name of the whole detection cascade. It is configured by one frozen `DetectionConfig` (see [DetectionConfig](#detection-config)): which stages run, each stage's model, revision, device, thresholds and options, which stage may report which type, what happens to each type, the gate and the conflict rules. A config has a stable SHA-256 hash, so a result can name the exact config behind it.

### The six built-in stages

| Stage | Kind | Model (revision, licence) | Default thresholds (balanced) |
|---|---|---|---|
| `pattern_scan` (regex) | Rules with checksums and context | — | — (every hit is 1.0) |
| `canonicaliser` (regex) | Rules on rewritten views | — | — |
| `entity_trace` (NER) | spaCy NER | `en_core_web_lg` 3.8.0, MIT | high 0.90, low 0.70, fallback 0.75 |
| `context_guard` (NER) | Transformer NER | `dslim/distilbert-NER` @ `dfa2838a…`, Apache-2.0 | accept 0.70 (off) |
| `pii_tagger` (NER) | Token classifier | `pii-tagger-dv3s-40k`, DeBERTa-v3-small, MIT, pinned by SHA-256 | PERSON / ORG / LOCATION 0.7, ADDRESS 0.9, PHONE / ID / HANDLE 0.95, CREDENTIAL 0.97, gate_above 0.97 |
| `structural` (rules) | Layout passes | — | — |

### Which stage may report which type (balanced)

| Public type | Allowed sources |
|---|---|
| EMAIL, URL, NETWORK, DATE_OF_BIRTH | pattern_scan, canonicaliser, structural |
| PHONE, ID, ADDRESS, HANDLE, CREDENTIAL | pattern_scan, canonicaliser, structural, pii_tagger (at its high thresholds) |
| AGE | pattern_scan, canonicaliser, structural (the tagger's ages are not routed in balanced) |
| PERSON, ORG | pattern_scan, canonicaliser, structural, pii_tagger, context_guard |
| LOCATION | the same, plus entity_trace (spaCy) |
| OTHER | `"*"` — every stage |

A model's guess at a phone number or an ID is ignored unless the type routes it, so a model cannot turn an order number into an "ID". On the development data, spaCy's names and organisations added far more harmless edits than leaks caught, so `balanced` uses spaCy for places only. `strict` and `classic` let every stage report PERSON, ORG and LOCATION.

<a id="patternscan"></a>

## PatternScan

`core/detection/pattern_scan.py`. The first stage, and the most trusted (rank 1 in the resolver). Order of work:

### 1. URLs

A personal URL (a profile path, a personal subdomain, a personal domain) becomes a `url` value. A non-personal URL (documentation, a news article) becomes *opaque*: nothing inside it is masked later, except secrets in its query string (`?token=…`) and e-mails.

### 2. Street addresses

An address parser and an address assembler both read the text. The longer span wins, and the parser wins a tie. Street numbers, street names and types, unit markers, cities, states and ZIPs are recognised in US, UK and European layouts.

### 3. The pattern list, in order

e-mail → API key → credential → handle → crypto → MAC → IPv6 / IPv4 → VIN → IBAN → ID number → phone (international, US, UK, national) → credit card → US driver's licence → passport → licence plate → card fallback → SSN → US bank number → date of birth → age → gender words → UK postcode → nearby ID number → labelled ZIP → bare ZIP.

### 4. Tails and tables

Phone extensions ("x204") and age units are attached; a list that continues a matched passport, licence, ID or credential is followed ("Passport: A09382716 / A11746620"), and so is a list of ages ("aged 7 and 10"); an "Age" column in a table is read as ages.

### Families, validators and gates

| Internal type | Examples | Check |
|---|---|---|
| `email` (regex) | priya.raman@northwind.io | Shape; spelled forms via the canonicaliser |
| `api_key` (regex) | Vendor prefixes (`sk-…`, `ghp_…`, `AKIA…`, `xox…`), JWTs | Known prefix and length |
| `credential` (regex) | "password: Tr0ub4dor&3", seed phrases, TOTP secrets | Needs a keyword (multilingual) |
| `handle` (regex) | @priya_r, priya#4821 | Shape and context |
| `crypto` (regex) | Bitcoin P2PKH / P2SH / Bech32, Ethereum 0x… | Chain prefix and alphabet |
| `mac_address`, `ip_address` (regex) | 3c:22:fb:9a:10:4e, 10.0.4.17, 2001:db8::7 | IPv4 rejects 127., 0. and 255. addresses |
| `vin` (regex) | 1HGCM82633A004352 | VIN check digit |
| `iban` (regex) | GB33 BUKB 2020 1555 5555 55 | mod-97 |
| `id_number` (regex) | "member ID 88213407", "NHS no. 943 476 5919" | Needs an ID keyword; 8+ digits after an ID label on the same line |
| `phone_us`, `phone_uk`, `phone_intl` (regex) | (415) 555-0142, +44 20 7946 0958, +49 30 901820 | Shape and length; extensions attached |
| `credit_card` (regex) | 4111 1111 1111 1111 | 13–19 digits and Luhn; a Luhn-invalid number is still taken when card words are near |
| `us_driver_license` (regex) | "licence B7654321" | Needs a keyword |
| `passport`, `license_plate` (regex) | "passport 533380006", "plate 7ABC123" | Keyword; a plate needs both letters and digits, 8 or fewer |
| `ssn` (regex) | 531-72-4408 | The dashed or spaced form always; a bare 9-digit number only with SSN context and only if it is not a valid ABA routing number |
| `us_bank_number` (regex) | 021000021 | ABA checksum plus bank words within 60 characters |
| `dob` (regex) | "born 14 March 1988", "DOB: 03/14/1988" | A birth cue within 40 characters, or a year at least 5 years ago; non-birth date labels such as "due date" reject it |
| `age` (regex) | "I'm 34", "aged 63", "34yo", "29F", "Marta Kowalczyk, 41" | 1–120 with an age cue, and no unit after it ("45 minutes" is not an age) |
| `gender_indicator` (regex) | "gender: female", "sex: M", "I'm a woman", "identifies as …", he/him, she/her, they/them | A word that matches a named person is left as written (see the gender pass) |
| `zip_us`, `postcode_uk` (regex) | 97205, 97205-1234, SW1A 1AA | A ZIP needs a label, geo words, a state abbreviation, or the +4 form |
| `address` (regex) | 1847 Alder Street, Portland, OR 97205 | Parser and assembler |
| `url` (regex) | linkedin.com/in/priya-raman, priyaraman.dev | Personal URLs only. Device names that carry an owner's name ("Ritikas-iPhone", "DESKTOP-ritika") are read as `hostname` by Pass S |

Order, invoice and ticket numbers are not read as an SSN, ZIP, phone or card, and an "order ID" or "tracking ID" is not an ID. A serial number is treated as an ID. A code such as `NW-48213-K` with no ID label near it matches no pattern; reading it is left to the tagger.

<a id="canonicaliser"></a>

## Canonicaliser (v2)

`core/detection/canonical.py`. People write values in odd ways. The canonicaliser builds rewritten *views* of the text, runs the patterns on each view, and maps every hit back to the exact characters in the original. Its `render_like` step writes the surrogate back in the same style, so a phone written in words gets a surrogate written in words.

| View | Rewrites | Example caught |
|---|---|---|
| `worded` | Runs of 7 or more digit words into digits; cardinal numbers | "four one five five five five zero one four two" |
| `spelled` | "at" / "dot" e-mail forms. The TLD must be known. A bare " at " needs a local part with several dotted parts or a digit, "_" or "+"; or a webmail domain with a local part that is not an ordinary word; or a mail cue in the 60 characters before | "priya dot raman at gmail dot com" |
| `dates` | Ordinal and worded dates to ISO | "the fourteenth of March nineteen eighty-eight" |
| `folded` | Removes zero-width characters and soft hyphens, maps Cyrillic and Greek look-alikes, and applies NFKC to a character when the result is printable ASCII | "рriya@northwind.io" with a Cyrillic р |
| `joined` | Digit groups split by single spaces or hyphens: the first group 2–5 digits, 9–19 digits in all | "4111 1111 1111 1111" |
| `casefold` | Lower-cases (not in the default views) | — |

Pick views with `with_stage("canonicaliser", options={"views": ["worded", "spelled"]})`. Plugins receive the folded-then-worded view as their `view` argument. Without the canonicaliser, the injected leak rises from 0.8% to 3.9%.

<a id="entitytrace"></a>

## EntityTrace

`core/detection/entity_trace.py` runs spaCy `en_core_web_lg` on the text with every pattern hit masked. It reads PERSON, GPE, LOC, ORG and FAC. The model is loaded once per process and is never downloaded by the library: install it with `python -m spacy download en_core_web_lg`.

spaCy gives no per-entity confidence, so EntityTrace assigns a fixed score per label. The thresholds therefore act as type gates:

| Label | Score | At balanced (high 0.90 / low 0.70 / fallback 0.75) |
|---|---|---|
| PERSON | 0.88 | Borderline → kept at fallback (but not routed for PERSON in balanced) |
| GPE | 0.85 | Borderline → kept at fallback → LOCATION |
| ORG | 0.85 | Borderline → kept at fallback (not routed for ORG in balanced) |
| LOC | 0.74 | Borderline, below fallback → dropped |
| FAC | 0.70 | Borderline, below fallback → dropped |

- An ORG with a location cue in the previous 50 characters (in, near, live / lives / lived, grew, born, raised, moved, relocate(d), residing, reside, hometown, birthplace, based) is re-read as a place (GPE, 0.85).
- Spans that cross a line break are skipped. Edge punctuation and possessives ("Raman's") are trimmed.
- A blocklist drops common false positives.
- Score ≥ high → confirmed; score ≥ low → borderline, sent to ContextGuard when it is on, otherwise kept if ≥ fallback.
- In `strict` the thresholds are 0.70 / 0.40 / 0.45 and every label passes; in `classic` they are 0.85 / 0.60 / 0.65.

<a id="contextguard"></a>

## ContextGuard

`core/detection/context_guard.py` runs `dslim/distilbert-NER` at the pinned revision `dfa2838a127384aabb82ed7719e16dab84c42a2a` through the Hugging Face NER pipeline (`aggregation_strategy="simple"`). It reads PER/PERSON, ORG, LOC and GPE; the `labels` option can map others.

- A borderline EntityTrace entity is confirmed when its spaCy score is at or above `accept` (0.70; 0.50 in `strict`).
- The model then runs its own NER and adds what it finds (source `slm`): confirmed at or above `accept`, otherwise flagged for confirmation. Its places are dropped in a service query.
- Names shorter than 3 characters are dropped; fragments are grown to whole words (up to 12 characters each side); a blocklist applies.
- It loads from the local Hugging Face cache. If the model is not cached and `HF_HUB_OFFLINE` is unset, it is downloaded once; otherwise detection fails closed.
- Off in `balanced` and `fast`, because the PII tagger reads names better. Turn it on with `ss.config(context_guard_enabled=True)`, `.with_stage("context_guard", enabled=True)`, or the `strict` / `classic` presets. `--no-context-guard` turns it off on the CLI.
- `device`: -1 for CPU (default) or a GPU index.

<a id="piitagger"></a>

## PIITagger (v2)

`core/detection/pii_tagger.py`. A token classifier fine-tuned from `microsoft/deberta-v3-small` (MIT) on generated text only (`bench/tagger/`). No real user message was used to train it.

| Property | Value |
|---|---|
| Name | `pii-tagger-dv3s-40k` |
| Revision | `sha256:f9e5821151ac…` (the SHA-256 of the weights file); other weights under the same name fail closed |
| Location | `$SURROGATESHIELD_MODELS/pii-tagger-dv3s-40k`, default `~/.cache/surrogateshield/models`; `model` may also be a path |
| Weights file | `model.safetensors` or `pytorch_model.bin` |
| Tags (BIO) | PERSON, ORG, STREET, CITY, REGION, POSTCODE, LOCATION, AGE, DATE_OF_BIRTH, EMAIL, PHONE, HANDLE, URL, NETWORK, ID, CREDENTIAL |
| Address parts | STREET, CITY, REGION and POSTCODE next to each other merge into one ADDRESS (`merge_address`). A lone CITY becomes LOCATION; a lone REGION is dropped |
| Windowing | 256-token windows with stride 64. Each token takes its label from the window in which it sits most centrally |
| Score | The mean B/I probability over the span |
| Options | `device`, `batch` (8), `merge_address` (true), `gate_above` |
| Install | `PYTHONPATH=.:python-library python -m bench.tagger.install bench/tagger/build/models/dv3s-40k [--name N]` |

`gate_above` (0.97 in balanced) is the score at which a tagger entity is *vouched* and passes the relation gate without a tie. The tagger is resolved through the plugin registry, so `register_detector("pii_tagger", …)` may replace it. It is the single most valuable stage: without it, the injected leak rises from 0.8% to 10.8%.

<a id="postprocessing"></a>

## Structural Passes

Rules that read layout and phrasing rather than single words. They run after the model stages and can add, extend, merge or remove spans. Turning the `structural` stage off (the `no_structural` ablation) skips them, except Pass D.

| Pass | What it does | Example |
|---|---|---|
| A | Organisation names from "the X Corporation / Company / Inc / Ltd / LLC / Group" and Title-Case runs ending in Consulting, Group, Holdings, Agency… | "at Halvorsen Dental Group" |
| E | Names from frames: "my name is", "I am" / "this is" (2+ words), "his/her name is", titles in several languages (Herr, Frau, Monsieur, professora…), "Dear …" lines, "Name <email>", "… here", form labels (Name:, Patient Name:, Customer:), log-in names in logs | "Dear Ms. Raman," |
| F | Drops implausible organisations: digits, model codes, lower-case words, acronym + number, and (ContextGuard ORGs only) a single word at sentence start | "RTX 4090" is not an ORG |
| G, G2 | Merges adjacent names; grows a name over particles ("van der", "de la") and "Surname, Given" | "Raman, Priya" |
| O | Assembles a whole organisation name from labels, work phrases, sign-offs and ISO 20275 legal forms | "Northwind Outfitters GmbH" |
| H | A capitalised word right before a card number becomes an ORG (the issuer) | "Chase 4111…" |
| B | An ORG that is the start of an e-mail user name becomes a PERSON | "Raman" in raman@… |
| C | Removes a PERSON whose words are all inside a longer PERSON | "Priya" inside "Priya Raman" |
| D | GPE and LOC only, skipped in a service query. Drops places that appear only in a question's sub-clause, and lower-case places mid-sentence unless the writer writes in lower case; a place anchored to the writer is kept | "which is bigger, Tempe or Mesa?" |
| S | The structural rule set: trims intros, extends particles, contact frames, streets, places, address parts, CSV columns, chat speakers ("Priya: …"), e-mail local-part names, device names carrying an owner's name (score 0.9), "PAYMENT TO" lines, international names, kin names ("my sister Anjali"), verb frames, display names, institutions, signature organisations, organisation frames, components | a `name,email,phone` CSV |

<a id="relation-gate"></a>

## Relation Gate (v2)

`core/detection/relation_gate.py`. Masking every name hurts the answer: "Compare Portland and Seattle for a move" needs both cities. The gate keeps a name or place as typed unless something ties it to a person. It applies to ORG, GPE, LOC, FAC and PERSON from the model stages, and to PERSON from any stage. Pattern hits of other types never go through it.

### 1. Drop the obvious

Junk spans, public figures (from the bundled `public_people.txt`) and common words read as names are dropped.

### 2. Look for ties

A tie is a device name, issued-ID context within 60 characters, a first-person cue before or after ("I work at", "we live near", "my doctor"), a field label, a link to a person, a CSV label, an organisation slot, or a signature or e-mail header. An issued ID (card, bank number, IBAN, ID, passport or licence) within 60 characters in the same sentence also ties. Ties spread through people and through places and organisations that are already tied.

### 3. Decide

Private persons are always masked; ties decide places and organisations. Multi-word organisations with a legal form are masked unless public. Public organisations and common nouns are dropped. Tied or *vouched* entities (a stage's score at or above its `gate_above`) are masked. If the message holds any anchor (an e-mail, phone, SSN, date of birth, address, card, licence, passport, ID, bank number, IBAN, handle, URL, age, credential or a person's name), untied places and organisations that are neither public nor common nouns are masked too. Everything else is sent as typed, with the reason `not_tied_to_person`.

**Kept — the topic:**

> How far is Tempe from Phoenix? Is Microsoft hiring there?

**Masked — tied to you:**

> We live near Reno and I work at Halvorsen Dental.

- `gate=False` turns the gate off (the `no_gate` ablation).
- `gate_bypass=["PERSON"]` masks every name whatever the gate says; `["*"]` bypasses it for every type, as `strict` does.

<a id="resolver"></a>

## Resolver

`core/detection/resolver.py`. Several stages can read the same value: the tagger and spaCy both read "Portland", or PatternScan and a plugin both read an ID. A value gets one surrogate, so the resolver keeps one candidate per value:

### 1. Source rank

`source_priority`, best first: `pattern_scan` → `canonicaliser` → `structural` → `pii_tagger` → `context_guard` → `entity_trace`. Plugins not listed share the last rank. A pattern hit found on a view ranks as canonicaliser; a pattern PERSON or ORG ranks as structural.

### 2. Type table

If the best-ranked candidates disagree on the type, `type_conflicts` decides. It is keyed by the sorted pair (`"ORG|PERSON"`) and gives the winning type or `"score"`. Every default pair (LOCATION|ORG, LOCATION|PERSON, ORG|PERSON) is `"score"`. A cycle keeps all.

### 3. Score

The highest score wins; on a tie, the first reported.

Spans of different values that overlap are cut: the better source rank, then the longer span, then the earlier one. When spans nest, the longer value is replaced first, and the shorter value still gets its surrogate wherever it appears on its own. `source_priority=()` with every pair set to `"score"` gives a pure score rule.

<a id="fail-closed"></a>

## Fail-Closed Behaviour

If a stage that is switched on cannot run, nothing is masked and nothing should be sent: `scan()`, `mask()` and `mask_result()` raise `DetectorUnavailable` (a `RuntimeError`). The dashboard ends the chat; the CLI exits with code 3. A detector that silently returned nothing would send your data unmasked, so this is deliberate.

| Cause | Fix |
|---|---|
| spaCy is not installed, or `en_core_web_lg` is missing | `pip install spacy`; `python -m spacy download en_core_web_lg` |
| spaCy failed on this input | Report the input; switch the stage off with `with_stage("entity_trace", enabled=False)` to continue |
| transformers or torch missing; ContextGuard cannot load or is not cached while offline | `pip install transformers torch`; run once online, or turn ContextGuard off |
| The tagger folder, weights file or SHA-256 pin is missing or wrong | Run `bench.tagger.install`, or use the `classic` preset |
| A config lists a plugin that is not registered | Call `register_detector()` before the first scan, or install the package with the entry point |

<a id="mimicgen"></a>

## MimicGen — Surrogate Generation

`core/generation/mimic.py` and `identity.py`. A good surrogate is the same kind of thing, in the same shape, and agrees with the other surrogates in the conversation. Faker is only the last resort.

### People, e-mails and handles

- Each given name and each surname gets one surrogate token for the whole conversation, so "Priya", "Ms. Raman", "PRIYA RAMAN" and `priya.raman@…` all point to the same fake person.
- Gender is taken from a title first, then from given-name lists, then from pronouns in the message; otherwise neutral.
- Locale and script are kept: an Indian name gets an Indian surrogate, a Chinese name keeps family-name-first order.
- Titles, suffixes (Jr., III), particles (van, de), initials, hyphens and capitalisation are kept. Names that are common words and tokens shorter than 3 letters are never picked.
- E-mail: the local part is rebuilt from the linked surrogate person in the same style (`p.raman` → `a.nath`), and a `+tag` is kept. Role addresses (info@, admin@, support@) keep their local part. Free-mail domains (gmail.com, outlook.com…) are kept; other domains get one fake label per conversation with the same suffix (`.co.uk` stays).
- Handles keep `@` and a `#1234` discriminator.

### Numbers and IDs

| Type | Surrogate rule |
|---|---|
| Phone | Same country code, trunk prefix, first national digit, separators and extension. US/Canada numbers get an area code that is not N11, N9X or toll-free (a toll-free area code is kept), and an exchange that is not N11 or 555 |
| SSN | Area 001–899 except 666, non-zero group and serial, same layout |
| Credit card | Same first two digits and length, valid Luhn |
| IBAN | Same country, valid mod-97, same grouping |
| ABA routing number | Stays checksum-valid; other bank numbers keep their shape |
| VIN, passport, ID number, licence plate, driver's licence, credential, MAC, UK postcode | Same character class at each position (digit for digit, letter for letter, hex stays hex) |
| API key | Keeps the vendor prefix (`sk-`, `ghp_`…) |
| Crypto address | Keeps the chain prefix and alphabet |
| IP address | A private IPv4 stays private and a public one public; CIDR and port kept; IPv6 keeps its shape |
| URL | On a profile host (LinkedIn, GitHub…) or a person path, the host is kept and the personal segment becomes a slug linked to the surrogate person; a personal-host suffix gets a new subdomain; otherwise the domain label changes. The scheme and path shape are kept |

### Dates, ages, places, organisations

- **Date of birth** moves 30–730 days, never into the future, in exactly the same format (order, separators, padding, ordinal, month-name style, weekday). A year alone moves 1–2 years; a year and month moves 1–24 months.
- **Age** moves by 1–3 years, never below 1. An age in words is re-spelled in words.
- **Places** become a real place of the same kind in the same country; a country becomes another country.
- **Organisations** keep their legal and industry words ("Dental", "LLC", "Pharmacy") and swap the name words; "University of Oregon" becomes another real place's university.
- **Addresses** follow the [address mode](#service-query).
- **Gender words**: one that matches a named person is left as written; any other is swapped for another of the same form (she/her ↔ he/him).
- **Redact** action: `[PHONE_1]`, `[PERSON_2]`… unique in the session and restored like surrogates.

### Consistency and collisions

- An original that already has a surrogate in the session keeps it. Matching is exact, then case-insensitive with the case adapted ("PRIYA RAMAN" → "ADYA NATH").
- A surrogate never equals any original in the message or the session, never contains an original as a whole value (addresses excepted), and avoids text already in the message.
- Up to 20 retries per value. If no fallback exists after that, a `RuntimeError` is raised rather than sending a doubtful surrogate.
- With `seed=`, every draw comes from one seeded RNG, so runs repeat (dates of birth depend on today's date).

<a id="shadowmap"></a>

## ShadowMap — Encrypted Storage

`core/storage/shadow_map.py`. The map from surrogate to original for one session. By default it lives only in memory and is gone when the session closes. With `pii_mem="<dir>"`, `Session(storage_dir=…)`, the CLI's default store, or the dashboard, it is also written to disk: the library writes after every change, and the dashboard saves at its own checkpoints.

| Property | Value |
|---|---|
| File | `<storage_dir>/<session_id>.shadowmap`, mode 0600; directory forced to 0700 |
| Format | `b"SSv1"` ‖ 12-byte random nonce ‖ AES-256-GCM ciphertext of the JSON `{surrogate: original}` |
| Key | HKDF-SHA256, 32 bytes. Input key = the device secret; salt = the session id; info = `shadowmap-v1` (`transcript-v1` for dashboard transcripts) |
| Associated data | `kind\|id`, so a file renamed to another session or kind fails to decrypt |
| Device secret | `$SURROGATESHIELD_HOME/device.key` (default `~/.surrogateshield/device.key`): 32 random bytes, mode 0600, in a 0700 folder. Created atomically; looser permissions are fixed with a warning; a wrong length is an error |
| Session id | 1–64 characters of `A–Z a–z 0–9 _ -`, otherwise `ValueError` |
| Writes | Atomic: temp file, chmod 0600, write, fsync, rename. A failed write leaves the old file |
| `forget()` | Saved to disk immediately |
| `close()` / `flush()` | Clears memory and **deletes** the file |
| Corrupt file | A bad tag, truncation or bad JSON moves the file to `<name>.corrupt-<unix time>` and raises `CorruptStoreError` |
| Older files | Files from before `SSv1` (no magic, no associated data, or a separate `<id>.key`) are still read and rewritten as v1 on the next save |

> **The device secret is the only key.** Anyone who can read `~/.surrogateshield/device.key` and a `.shadowmap` file can read the map. Losing `device.key` makes every stored map and transcript unreadable. Back it up the way you back up an SSH key, or not at all.

Storage errors: `StorageError` (a `RuntimeError`, exported) with the subclasses `ShadowMapStorageError` (the device secret cannot be created or read) and `CorruptStoreError`, both in `surrogateshield.core.storage.shadow_map`. The dashboard can be told to use a process-only key instead of failing (`ALLOW_EPHEMERAL_KEY = True` in `config.py`); data written that way cannot be read after the process exits.

<a id="resolvepass"></a>

## ResolvePass — Restoring the Answer

`core/reconstruction/resolve.py`. The model's answer contains surrogates, sometimes changed: re-cased, shortened to a first name, or slightly respelled. ResolvePass maps them back in four passes. Every rewritten span is protected from later passes, and nothing is replaced inside a longer word or number.

| Pass | Matches | Example (surrogate → original) |
|---|---|---|
| 1. Exact | The whole surrogate, longest first, case-insensitive with the case carried over. A lower-case match counts only for multi-word surrogates or ones with a digit or `@` | "ADYA NATH" → "PRIYA RAMAN" |
| 2. Component | Runs of 2+ words from a multi-word surrogate, longest first, aligned to the original word by word (a run that splits a changed block of words is not used) | "6237 James" → "412 Alder" (surrogate "6237 James Street, Olympia", original "412 Alder Street, Portland") |
| 3. Fuzzy | Letters-only surrogates of 2+ words and 6+ characters, with one word of 4+ letters present verbatim. rapidfuzz `partial_ratio` ≥ `fuzzy_threshold` (85), same word count, then `ratio` ≥ 85; capitalised words must stay capitalised | "Adya Naath" → "Priya Raman" |
| 4. Name parts | The given name alone (not when the next word is capitalised) or the surname after a title. Only for a person-shaped surrogate (2+ capitalised name words, no organisation words) with as many name words as the original, when the part is in no other mapping and is not preceded mid-sentence by a capitalised word. Case-sensitive | "Adya" → "Priya", "Ms. Nath" → "Ms. Raman" |

- Low-entropy surrogates (2 characters or fewer, ages, gender words) are restored only if they were sent in the last turn, so "34" in an unrelated sentence is not touched.
- Words you typed yourself this turn are not rewritten by Passes 2–4 (and by Pass 1 for one-word, letters-only surrogates from earlier turns), so a surrogate name that is also a word in your message is left alone. `Session` passes your message automatically (`sent=`).
- Names that are common words and parts shorter than 3 characters are never restored by Pass 4.
- `unmask()` accepts a string, an Anthropic `Message`, an OpenAI `ChatCompletion`, a Gemini response (`.text`), or the same as dicts.

Measured on the benchmark: across 60 multi-turn conversations, the originals were restored in 81 of the 82 replies that needed it.

---

<a id="cli"></a>

## Dashboard Commands

The terminal app is `main.py` (Typer and Rich). `./run.sh` starts it inside the virtual environment. With no subcommand it opens the interactive dashboard.

| Command | What it does |
|---|---|
| `python main.py` | Opens the dashboard: banner, a 7-step pipeline summary, saved conversations, and the action menu |
| `python main.py chat` | Starts a new conversation |
| `python main.py chat --rag` | Starts a new conversation that retrieves from your indexed documents |
| `python main.py chat --load ID` | Resumes a saved conversation. A conversation started with RAG resumes with RAG |
| `python main.py chat --delete ID [-y]` | Deletes the transcript and its surrogate map without decrypting either, so an unreadable conversation can still be removed. Asks first unless `-y`/`--yes`. Exit 1 if you cancel or the id does not exist; exit 2 for an invalid id |
| `python main.py pii-finder` | Opens the [PII Finder](#pii-finder) (no API call) |
| `python main.py list` | Prints the saved conversations: #, ID, Created, Turns, Mode |
| `python main.py add-doc FILE` | Masks a UTF-8 text file and indexes it for RAG. Prints "N chunks indexed (total: X)" |
| `python main.py rag` | Lists indexed documents as "source · doc_id · N chunks" |
| `python main.py rag --forget DOC [-y]` | Removes a document by file name or id, then erases every document mapping no remaining chunk uses. Prints "removed N chunks, erased M mappings" |

```bash
./run.sh                                # the dashboard, inside .venv
python main.py chat --rag               # chat over your documents
python main.py chat --load 1f0c9e6a…    # resume
python main.py add-doc notes/lease.txt  # index a document
python main.py rag --forget lease.txt -y
```

<a id="keyboard-shortcuts"></a>

## Keyboard Shortcuts

On the dashboard, type a key and press Enter. Keys are not case-sensitive.

| Key | Action |
|---|---|
| `N` | New conversation with PII protection |
| `R` | New conversation with RAG |
| `P` | PII Finder: test detection on any text, no API call |
| `1` – `9` | Open the saved conversation with that number |
| `D1` – `D9` | Delete that conversation (asks y/N) |
| `J` | JSON Test: run a file of questions through the whole pipeline |
| `E` | Evaluation: score a batch run against a key file |
| `A` | Attacker Experiment: a second model tries to recover the originals |
| `S` | Settings: provider, Detailed View, Presidio Comparison |
| `H` | Help screen |
| `Q` | Quit (Ctrl-C works too) |

Inside a chat or the PII Finder, type `exit`, `quit` or `back` to return. In sub-screens, `B` goes back.

<a id="chat-mode"></a>

## Chat Mode

Each message you type goes through one turn of the pipeline (`Pipeline.process_turn` in `pipeline.py`):

### 1. Service query check

Decides the address mode for this message ([Service Queries](#service-query)).

### 2. Detection

SentinelLayer runs on your message. When you quote back a surrogate from an earlier turn, the pattern scan skips it, so an e-mail, phone number or ID is not masked twice. A quoted surrogate name can still be found by the tagger and masked again.

### 3. Confirmation

When ContextGuard is on, entities it is unsure about are listed and you are asked `Replace this? [y/N]` for each. With the default preset ContextGuard is off and you are not asked.

### 4. Surrogates

A value seen earlier in the conversation, or in your documents in RAG mode, keeps its surrogate. A new surrogate never equals a real value from any turn. The map is saved encrypted after the turn.

### 5. Send

The masked message goes to the active provider with the system prompt and the masked history. In RAG mode, the top 3 chunks for your masked question are added to this request only.

### 6. Restore and display

ResolvePass puts your values back. The restored answer is shown and saved; the masked answer stays in the history sent next time.

### Typing messages

- One line is one message.
- End a line with `\` to continue on the next.
- Start a line with `"""` to paste several lines; the block ends at a line ending with `"""`.
- `exit`, `quit` or `back` returns to the dashboard. There are no slash commands.
- Ctrl-C while a turn runs cancels it ("Turn cancelled"). Nothing is written to the history until a reply arrives, so nothing needs undoing.

### What the provider is told

The system prompt tells the model that the personal details are anonymised aliases, to treat them as real information, and not to comment on them or issue security warnings about them. Without it, some models refuse or warn when a name and an e-mail do not match their expectations.

### Errors during a chat

| Error | What happens |
|---|---|
| Missing key, or 401 / 403 from the provider | "Configuration error", the chat ends. Fix the key in `.env` or Settings |
| `DetectorUnavailable` | The model and the fix are printed, and the chat ends. Nothing was sent |
| Timeout, 408, 409, 429, 5xx, connection error | Retried up to 3 attempts in all, with backoff from 5 s (capped at 60 s), honouring `retry-after`. If all fail: "not sent — try again", and the chat continues |
| Empty reply | Reported as an error, not retried |
| Corrupt transcript or map, unusable device key | "Cannot open conversation" / "Cannot open the surrogate map"; a corrupt file is moved aside |

### Where conversations are stored

- `~/.surrogateshield/conversations/<id>.json`: the transcript, encrypted (`SSv1`, `transcript-v1`), mode 0600. It holds what you typed and what you saw, plus the masked history that is sent. Plain-text transcripts from older versions are re-encrypted when opened.
- `~/.surrogateshield/conversations/<id>.shadowmap`: the conversation's surrogate map.
- Deleting a conversation removes both files.

<a id="detailed-view"></a>

### Detailed View

Turn it on in Settings (`S` → `D`). Each turn then shows:

- INFO logs from every stage of the cascade;
- a table of every detection with its type, score, source stage and surrogate;
- a quasi-identifier warning, or "No quasi-identifier combination risk detected";
- the **API Transparency** panel: "Sent to <provider>", "Received from <provider>" and "Final output (real values restored)", each trimmed to 300 characters. Set `SHOW_API_TRANSPARENCY = False` in `config.py` to hide this panel.

**Sent to Claude:**

> I'm Adya Nath, my landlord at 6237 James Street won't return my deposit. Draft a letter.

**Final output:**

> Dear Landlord, I, Priya Raman, former tenant at 1847 Alder Street, request the return of my deposit…

<a id="pii-finder"></a>

## PII Finder

Press `P` or run `python main.py pii-finder`. Type any message and see exactly what would be sent. It runs the same code as a real turn and makes no API call ("No API calls · No credits spent").

- **"SentinelLayer — Detected PII"** table: Original, Type, Score, Source, Surrogate. Rows marked "needs confirmation" would be asked about in chat; rows marked "skipped — topical query" are places kept as the topic.
- **"Would send to <provider>"**: the masked text, or "No PII detected" when nothing was found.
- **Service query panel**: when the message is a service query, the address mapping and the mode used ("House number ±1, city/state unchanged", "Street line coarsened to 'my area'", or "Structure-preserving fake address").
- With Detailed View on, the quasi-identifier line.
- With Presidio Comparison on, a **Presidio panel**: Presidio's detections (value, type, score) and its `[TYPE]`-redacted text, for side-by-side comparison.
- `reset` clears the surrogate memory, so the next message gets fresh surrogates. Within one Finder session, the same value keeps the same surrogate.

> **Use it before you trust it.** Paste the kind of message you plan to send. If something is missed, see [Troubleshooting](#troubleshooting); if something harmless is replaced, see [type_actions](#type-actions) and [pii_off](#library-pii-off).

<a id="rag-mode"></a>

## RAG Mode

Chat over your own documents. A document is masked before it is indexed, so the vector store holds surrogates, and answers that quote it are restored like any other.

### 1. Index: `python main.py add-doc FILE`

The file is read as UTF-8 and masked in segments of up to 20,000 characters (cut at a paragraph, line or space). If detection fails in any segment, nothing is stored. Addresses are replaced, never shifted. All document pairs go into one shared map, `rag_global`: a name already seen in another document keeps its surrogate, and no surrogate equals one used by any conversation.

### 2. Chunk and embed

512-character chunks with an overlap of max(50, 512 ÷ 5) = 102 characters, embedded with `all-MiniLM-L6-v2` into a local ChromaDB collection (`surrogateshield_rag`, cosine distance) in `~/.surrogateshield/rag`. Each chunk carries `doc_id` (12 hex characters), `source` (the file name) and `chunk_index`. Chroma telemetry is off.

### 3. Ask: `R` or `chat --rag`

Your message is masked first, and the masked text is the retrieval query. A name you type that is in a document gets the document's surrogate, so retrieval matches. The top 3 chunks are added before your question as "[Document excerpt 1] … --- Question: …", in this request only.

### 4. Restore

The conversation's map and the document map are both used; if they disagree, the conversation's pair wins.

### 5. Forget: `rag --forget DOC`

Deletes the document's chunks, then erases every `rag_global` pair whose surrogate no remaining chunk contains.

> **Indexing runs detection locally; embedding is local too.** No provider is called at index time. The excerpts are sent with your question and hold the surrogates the document was indexed with. An old `./chroma_db` folder in the project is not read; the app warns if it finds one.

<a id="settings-screen"></a>

## Settings Screen

Press `S` on the dashboard.

| Key | Setting | Default |
|---|---|---|
| `L` | LLM Provider: `1` Claude, `2` Gemini, `3` ChatGPT, `4` Local LLM (Ollama). Each provider screen shows setup steps, `T` to test the connection, `A` to make it active, `B` to go back | Claude |
| `D` | Detailed View: logs, detection table, quasi-identifier line, API Transparency panel | On |
| `C` | Presidio Comparison: the Presidio panel in the PII Finder | Off |
| `B` | Back | — |

The test (`T`) sends "Hi" with the system prompt "Reply with one word." through the same adapter, model and retry policy as the chat, so "connection successful" means the configured model answered. It re-reads `.env` first; a variable already exported in your shell wins.

Settings are stored in `settings.json` (written atomically, mode 0600). An invalid value falls back to its default; a corrupt file is renamed to `settings.json.corrupt-<time>` and the defaults are used.

<a id="providers"></a>

## LLM Providers

| Provider | Key | Model setting (default) | Extra install |
|---|---|---|---|
| Claude (Anthropic) | `ANTHROPIC_API_KEY`; optional `ANTHROPIC_WORKSPACE_ID` | `SURROGATESHIELD_CLAUDE_MODEL` (`claude-sonnet-4-6`) | — (in requirements) |
| Gemini (Google) | `GEMINI_API_KEY` | `SURROGATESHIELD_GEMINI_MODEL` (`gemini-2.5-flash`) | `pip install google-genai` |
| ChatGPT (OpenAI) | `OPENAI_API_KEY` | `SURROGATESHIELD_OPENAI_MODEL` (`gpt-4o-mini`) | `pip install openai` |
| Local LLM (Ollama) | none | `LOCAL_LLM_MODEL` (`llama3.2`), `LOCAL_LLM_HOST` (`http://localhost:11434`) | Ollama: `ollama pull llama3.2`, `ollama serve` |

All four go through one adapter layer (`chatbot/providers.py`) with one retry policy: 60 s timeout, 3 attempts, exponential backoff from 5 s capped at 60 s, `retry-after` honoured, SDK retries off, `max_tokens` 4096. Authentication failures (401 / 403) are not retried.

> **Ollama keeps the whole conversation on your machine.** With a local model, masking is not needed for privacy from a provider, but the PII Finder and the evaluation tools work the same way.

---

<a id="json-test"></a>

## JSON Batch Test

Press `J`. Runs a file of questions through the whole pipeline (detect, mask, send, restore) and writes everything to one JSON file. This is the input for [Evaluation](#evaluation-mode) and the [Attacker Experiment](#attacker-experiment).

### 1. Put the questions in `experiment/<name>.json`

```json
[
  {"input": "My name is Priya Raman and I need to change my flight."},
  {"input": "Write a cover letter for a nurse job in Leeds."}
]
```

### 2. Choose the fields

Type a number to toggle a field, Enter to run, `B` to go back. If `<name>_answers.json` exists, the run resumes after the last answered question.

### 3. Run

Each question is seeded (base seed 20260101 plus its index), so surrogates repeat across runs. Results are saved every 25 questions and on Ctrl-C; run the same file again to resume. A failed question is written as an error row. Output: `experiment/<name>_answers.json` and `<name>_answers.meta.json` (the run's settings).

### Output fields

| Field | Content | Default |
|---|---|---|
| `question` | The question as asked | on |
| `pattern_scan_pii`, `entity_trace_pii`, `context_guard_pii` | What each stage found | on |
| `confirmed_pii` | The final combined list | on |
| `pii_detail` | Type, score and source of each detection | on |
| `pii_spans` | Offsets, type and whether replaced — what the evaluator scores | on |
| `quasi_id_risks` | Quasi-identifier combinations found | on |
| `surrogate_map` | Original → surrogate | on |
| `sanitized_input` | The exact text sent | on |
| `llm_response`, `final_output` | The raw answer, and the answer with your values restored | on |
| `stage_timings_ms` | Time per stage, surrogate generation and the LLM call | on |
| `recognized_not_replaced` | Values recognised but deliberately left (service-query places, topical places) | on |
| `presidio_sanitized_input`, `presidio_found_piis` | Presidio's `[TYPE]` redaction and its raw detections | off |
| `presidio_llm_response` | The answer to Presidio's text (an extra API call per question) | off |
| `clean_llm_response` | The answer to the original, unmasked question — **for synthetic data only** (an extra API call that sends the original) | off |
| `bertscore_ss`, `bertscore_presidio` | Rescaled BERTScore (`roberta-large`, batch 16) of input fidelity and of the answer against the clean answer. Needs `pip install bert-score`; can take 15–30 minutes on CPU | off |

Before each send, the runner checks that the text about to be sent is exactly the recorded `sanitized_input`; a mismatch stops the row (`SendMismatch`). `address_mode` is always written.

<a id="evaluation-mode"></a>

## Evaluation

Press `E`. Scores a batch run against a ground-truth key. You give three files in `experiment/`, all with the same number of entries: the questions, the answers file from the JSON Test, and the key.

### Key format

```json
[
  {
    "Question": "Hi, I'm Priya Raman, call me on +1 415 555 0142.",
    "Answer-Key": {"name": "Priya Raman", "phone": "+1 415 555 0142"},
    "Spans": [{"start": 8, "end": 19, "value": "Priya Raman", "type": "name"}]
  },
  {"Question": "What is the capital of Peru?", "Answer-Key": {}}
]
```

- A label's value may be a string or a list. Labels are mapped to types (`name` / `person` → PERSON, `phone_us` → phone, `zip` → postal code, `location` → GPE, and so on); an unknown label becomes `other`.
- `Spans` is optional; with it, matching uses exact offsets. A legacy comma-separated string key is still read.
- An empty `Answer-Key` is a negative question: any replacement in it is a false positive.

### Report sections (each can be toggled)

| Section | What it measures |
|---|---|
| Answers | Answered, empty and error rows (error rows are excluded) |
| Surrogate counts | Values replaced against gold values |
| Detection | Protection (replaced spans) and recognition (replaced or deliberately left), micro and macro, per type, precision on negative questions |
| Sanitization | Gold values found in the text sent, split into deliberate (policy) and unintended |
| Restoration | In `final_output`: surrogates left in, over-restored text, collateral edits |
| Presidio comparison | Both tools on the same rows and a shared type set, with bootstrap confidence intervals |
| BERTScore | Input fidelity and answer utility, paired statistics |
| Ablation | Points to `python offline_eval.py --key <key file> --ablation` (needs a cascade re-run) |
| Stage timings | Mean, p50 and p95 per stage |

A gold value left in the text on purpose is counted as deliberate when it falls under one of these reasons: `service_query_address_shift`, `service_query_address_coarse`, `service_query_location_suppressed`, `topical_geo_filtered`, `not_tied_to_person`, `gender_follows_name`, `phone_country_code_kept`. Everything else is unintended. Press `S` on the results screen to save `experiment/<answers>_eval_results.json`.

<a id="offline-eval"></a>

## Offline Evaluation

`offline_eval.py` runs the batch runner's exact send path (cascade, seeded surrogates, span substitution) over a key file and scores it, with **no LLM call** and no API key. Use it to test detection changes quickly.

```bash
python offline_eval.py --key experiment/test_key.json
python offline_eval.py --key experiment/test_key.json --protection --json out.json
python offline_eval.py --key experiment/test_key.json --types address --limit 200
python offline_eval.py --key experiment/test_key.json --ablation --limit 200
python offline_eval.py --key experiment/test_key.json --lint-key
```

| Flag | Meaning |
|---|---|
| `--key FILE` | The key file (default `experiment/test_key.json`). Exit 2 if it does not exist |
| `--limit N` | Only the first N questions |
| `--types a,b` | Filter the per-type report |
| `--json FILE` | Write everything, including each miss |
| `--seed N` | Base surrogate seed |
| `--protection` | Also checks the sent text: every gold value found verbatim is a leak, split into deliberate and unintended; shifted addresses must appear exactly as their surrogate. **Exit 1** if there is any unintended leak, so it works as a CI gate |
| `--ablation` | Re-runs with PatternScan only, PatternScan + EntityTrace, no post-passes, and the full cascade, and reports each with a signed delta and a bootstrap CI |
| `--lint-key` | Only checks the key file (no models) |

<a id="attacker-experiment"></a>

## Attacker Experiment

Press `A`, or run `attacker.py`. Can a second model work out your real values from what was sent? For each sampled question, an attacker model reads the text each tool sent (SurrogateShield's `sanitized_input` and Presidio's `presidio_sanitized_input`) and guesses the original of every personal value it can see.

```bash
python attacker.py --answers test_answers.json --key test_key.json --sample 50 --seed 0 --model <attacker model> --dry-run
```

| Rule | Detail |
|---|---|
| Neutral prompt | The attacker is told a filter may have replaced some values, changed parts of others, or left them unchanged |
| Attacker ≠ responder | `SURROGATESHIELD_ATTACKER_MODEL` (or `--model`) must be set and must differ from the chat model. Anthropic models only |
| Same denominators | Both arms are scored against the same gold values from the key |
| Outcomes | `leaked_verbatim` (still visible, not an inference), `exact`, `partial` (same e-mail domain, area code, a shared name token, shared street or city words, same birth year), `not_recovered`; values redacted that are not in the key are `non_gold` |
| Failures | An API error or unparseable reply marks the arm unavailable; it is excluded and counted, never scored as zero |
| Reproducible | `--sample N --seed S` picks the questions; indices, model, prompt version and token usage are stored. Wilson confidence intervals |
| Cost control | `--dry-run` prints the number of calls and makes none; `--max-calls` caps them |
| Output | `experiment/<answers stem>_Attacker_Experiment.json` with `meta`, `rows` and `analysis`; saved every 5 questions and resumable |

The benchmark version of this experiment, against every competitor, is in [Results → Attacker](#results-attacker).

<a id="bench-harness"></a>

## Benchmark Harness

Everything behind the [Results](#results) is in `bench/`. Every result file is listed with its exact command in `bench/results/README.md`; the main commands are in [Reproduce](#reproduce).

| Folder | Contents |
|---|---|
| `bench/realdata/` | The real-chat benchmark: building the splits (injected, format-shift, multi-turn, natural), scoring, the operating curve, and the live runs (answer quality, multi-turn, attacker) |
| `bench/arms/` | One adapter per competitor (GLiNER-PII, Presidio, Presidio + Faker, Presidio + RoBERTa, LLM Guard) with a common interface |
| `bench/tagger/` | Generating training text, training the PII tagger, and installing it |
| `bench/realworld/` | Annotation guide and the keep / mask rules |
| `bench/results/` | Every result JSON, with a README naming the command for each |

- The competitors run in a separate environment, `.venv-baselines` (Python 3.12), so their dependencies do not change SurrogateShield's.
- Offline runs set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; live runs set `PYTHONHASHSEED=0`.
- The `surrogateshield bench` command is different: it times `mask()` on built-in messages on your machine ([Library CLI](#library-cli)).

---

<a id="pii-types"></a>

## PII Types

Every detection has one of 13 public types, plus `GENDER` and `OTHER`. Inside, the stages use finer internal names; `scan()` reports the internal name in `Detection.type`, and configuration accepts either.

| Public type | Covers | Internal names | Surrogate |
|---|---|---|---|
| `PERSON` (NER) | People's names, including lower-case, surname-first and spelled-out forms | `PERSON`, `person` | A name of the same apparent origin and gender, same token count and title |
| `ORG` (NER) | Organisations tied to the writer: employer, school, clinic, bank | `ORG` | An organisation of the same kind: legal suffix and industry words kept ("Dental", "LLP"), name words replaced |
| `LOCATION` (NER) | Places tied to the writer | `GPE`, `LOC`, `FAC`, `implicit_location` | A real place of the same kind: a town of the same country, a state for a state; a facility keeps its kind word |
| `ADDRESS` (Regex) | Street addresses, US ZIP codes, UK postcodes | `address`, `zip_us`, `postcode_uk` | Same structure, new values; or the same street, shifted ([Service Queries](#service-query)) |
| `EMAIL` (Regex) | E-mail addresses, including "name at host dot com" | `email` | Local part rebuilt from the linked surrogate person in the same style; a free-mail domain (`gmail.com`) kept, any other domain replaced by one made-up domain per conversation with the same suffix |
| `PHONE` (Regex) | US, UK and international numbers, in digits or words | `phone_us`, `phone_uk`, `phone_intl` | Same country code, trunk prefix, first digit and grouping, new digits; a US number gets a valid area code and exchange |
| `URL` (Regex) | Personal links, and device names that carry the owner's name (`Galaxy-S23-Alex`) | `url`, `hostname` | Scheme and host style kept; the identifying part (a profile's path segment, a personal sub-domain or the domain label) becomes a user name of the surrogate person, the rest of the path kept. A device name keeps its device words |
| `HANDLE` (Regex) | Usernames and social handles | `handle` | Built from the surrogate name, same style |
| `ID` (Regex) | SSN, passport, driver's licence, licence plate, VIN, payment card, IBAN, bank and account numbers, crypto wallets, order and customer numbers | `id_number`, `ssn`, `passport`, `us_driver_license`, `license_plate`, `vin`, `credit_card`, `iban`, `us_bank_number`, `crypto` | Same format and separators; a valid SSN, a card with its first two digits and a valid Luhn digit, an IBAN of the same country with valid check digits, an ABA-valid routing number; other IDs keep their shape |
| `CREDENTIAL` (Regex) | Passwords, PINs, API keys, tokens | `credential`, `api_key` | Same character classes and length; key prefixes kept |
| `NETWORK` (Regex) | IPv4, IPv6 and MAC addresses | `ip_address`, `mac_address` | A valid address of the same family; a private IPv4 address keeps its private prefix, a public one stays public |
| `AGE` (Regex) | Ages with a cue ("34 years old", "aged 34", "thirty-four") | `age` | An age one to three years off, written the same way |
| `DATE_OF_BIRTH` (Regex) | Birth dates in any common format | `dob` | A date 30 days to two years off, never in the future, in the same format |
| `GENDER` (Regex) | Stated gender and pronouns (`gender: F`, "I'm a woman", `she/her`) | `gender_indicator` | Another gender term in the same form; sent as written when it matches the gender of a name in the same message (that name's surrogate keeps the gender) |
| `OTHER` | Any other label, such as a plugin's own | — | A random string of the same shape (letters for letters, digits for digits) |

### Where a detection came from

`Detection.source` names the stage that found the value:

| `source` | Stage |
|---|---|
| `pattern` | PatternScan, the canonicaliser (a hit on a normalised view) or a structural pass that reads names off the message's layout |
| `pii_tagger` | The PII tagger |
| `ner` | spaCy (EntityTrace) |
| `slm` | ContextGuard (distilbert-NER) |
| `structural` | The structural passes |
| a plugin's name | That plugin |

<a id="pii-examples"></a>

## Examples

Real output of `Session(seed=7).mask_result(text)` with the default preset. With a seed, the same input always gives these surrogates.

### Identity

**Input:**

> I'm Priya Raman, 34 years old, born 12/03/1991. My SSN is 512-44-7810.

**Sent:**

> I'm Forum Rao, 31 years old, born 07/03/1993. My SSN is 097-47-9549.

PERSON from the tagger; age, date of birth and SSN from PatternScan. The age and the date of birth are drawn separately: the age moves by one to three years, the date by 30 days to two years.

### Contact

**Input:**

> Ping me at priya.raman@northwind.io or +44 7911 123456. My handle is @priya_r.

**Sent:**

> Ping me at carol.craig@miller.io or +44 7815 908301. My handle is @carol_c.

The e-mail and the handle follow one surrogate identity, and the UK number keeps its country code and grouping.

### Financial and credentials

**Input:**

> Card 4532 0151 1283 0366, IBAN GB29 NWBK 6016 1331 9268 19. My password is Tulip!2024.

**Sent:**

> Card 4526 0181 5908 3012, IBAN GB72 PPCH 1860 9139 0996 03. My password is Btejp!2819.

### Network and web

**Input:**

> My server is at 203.0.113.42, MAC 00:1B:44:11:3A:B7, and my site is https://priyaraman.dev/cv.

**Sent:**

> My server is at 39.202.24.19, MAC 81:5E:08:30:1D:D1, and my site is https://deannariley.dev/cv.

### Work and home

**Input:**

> I work at Halvorsen Dental in Reno and live at 1847 Alder Street, Portland, OR 97205.

**Sent:**

> I work at Moore Dental in Worcester and live at 5908 Meyer Street, Waco, AR 16613.

The employer keeps its kind word ("Dental"). The full address, with city, state and ZIP, is one detection and gets one structured surrogate.

### A service query, and a topical question

**Service query:**

> Find a plumber near 1847 Alder Street, Portland. → Find a plumber near 1846 Alder Street, Portland.

**Topical question:**

> How far is Tempe from Phoenix? → sent unchanged: the places are the topic, not tied to anyone.

<a id="quasi-identifier"></a>

## Quasi-identifier Warnings

Some combinations identify a person even when each value alone does not. After detection, the types found in a message are checked against these combinations, and with Detailed View on a warning is shown in the chat and in the PII Finder. Every value in the combination is still replaced as usual; the warning is information only. The check reads internal types: Phone is a US number (`phone_us`), Location a `GPE`, ZIP a US ZIP code (`zip_us`), Employer an `ORG`.

| Combination | Fires on | Risk |
|---|---|---|
| ZIP + DOB + Gender | any 2 of the 3 | high |
| Postcode + DOB | both | high |
| Name + SSN | both | high |
| Name + DOB | both | high |
| Phone + Name | both | high |
| IP Address + Name | both | high |
| Name + Employer + City | all 3 | medium |
| Email + Location | both | medium |
| Phone + Location | both | medium |
| DOB + Location + Employer | all 3 | medium |

```text
⚠  Quasi-identifier risk: Name + DOB combination detected
    (Name + DOB is the standard combination used in identity verification worldwide)
```

---

<a id="config-constants"></a>

## Dashboard Configuration (`config.py`)

The dashboard reads its constants from `config.py` at the project root. The Settings screen covers the provider and the two display switches; everything else is set here. The library has its own settings ([`ss.config()`](#library-config)).

| Constant | Default | What it does |
|---|---|---|
| `ENTITY_TRACE_HIGH_THRESHOLD` / `ENTITY_TRACE_LOW_THRESHOLD` | 0.90 / 0.70 | spaCy score gates |
| `ENTITY_TRACE_FALLBACK_THRESHOLD` | 0.75 | spaCy promotion score when ContextGuard is off |
| `CONTEXT_GUARD_ENABLED` | `False` | Turns on the distilbert-NER stage |
| `CONTEXT_GUARD_MODEL` / `CONTEXT_GUARD_CONFIDENCE_THRESHOLD` / `CONTEXT_GUARD_DEVICE` | `dslim/distilbert-NER` / 0.70 / -1 | ContextGuard model, accept score, device (-1 = CPU) |
| `SPACY_MODEL` | `en_core_web_lg` | spaCy model |
| `FUZZY_MATCH_THRESHOLD` | 85 | Restore score for a reworded surrogate (0–100) |
| `ADDRESS_MODE` | `"auto"` | `auto`, `shift` or `replace` |
| `ADDRESS_SHIFT_RANGE` | 1 | How far a shift moves the house number |
| `SERVICE_QUERY_DETECTION_ENABLED` | `True` | Service-query detection (the `auto` rule) |
| `SHOW_API_TRANSPARENCY` | `True` | The Sent / Received / Final panel in Detailed View |
| `CLAUDE_MODEL`, `GEMINI_MODEL`, `OPENAI_MODEL` | `claude-sonnet-4-6`, `gemini-2.5-flash`, `gpt-4o-mini` | Chat models (each overridable by its environment variable) |
| `ATTACKER_MODEL` | none | From `SURROGATESHIELD_ATTACKER_MODEL`; required by the Attacker Experiment |
| `LOCAL_LLM_MODEL` / `LOCAL_LLM_HOST` | `llama3.2` / `http://localhost:11434` | Ollama model and host (overridable by the `LOCAL_LLM_MODEL` / `LOCAL_LLM_HOST` environment variables) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | RAG embeddings |
| `RAG_TOP_K` / `RAG_CHUNK_SIZE` | 3 / 512 | Chunks retrieved per question; chunk length in characters |
| `RAG_COLLECTION_NAME` / `RAG_DIR` | `surrogateshield_rag` / `None` | ChromaDB collection; `None` = `~/.surrogateshield/rag` |
| `SHADOWMAP_DIR` / `DEVICE_KEY_PATH` | `None` | `None` = under `~/.surrogateshield` |
| `ALLOW_EPHEMERAL_KEY` | `False` | When the device key cannot be written, use a temporary in-memory key instead of stopping (maps then cannot be reopened) |
| `VERSION` | 2.1.0 | Shown on the dashboard |

The detection preset for the dashboard comes from `SURROGATESHIELD_PRESET` / `SURROGATESHIELD_DETECTION_CONFIG`, as for the library. When either is set, that config replaces the detection constants above and `ADDRESS_MODE`, `ADDRESS_SHIFT_RANGE` and `SERVICE_QUERY_DETECTION_ENABLED`.

<a id="presets"></a>

## Presets

Detection is described by a `DetectionConfig`. Four presets are built in; each has a stable hash, so a result can always be traced to the exact config that produced it.

| Preset | What runs | Leaked on test-3 | Natural messages untouched (of 2,433) | Hash |
|---|---|---|---|---|
| `balanced` (default) | PatternScan, canonicaliser, PII tagger (names, organisations, places from 0.7), spaCy for places only, structural passes. ContextGuard off | 0.8% (30) | 2,153 | `9470f9336d853f67` |
| `strict` | Every stage on, spaCy and ContextGuard back on names with lower thresholds, the tagger from 0.3, every type past the relation gate | 0.2% (8) | 1,264 | `d3afc88e9e1c94c5` |
| `fast` | No spaCy; the tagger reads places too, from 0.4 | 0.8% (28) | 2,147 | `2c41977891a20c88` |
| `classic` | No tagger; spaCy and ContextGuard read names, organisations and places. For a machine without the tagger's weights | 2.5% (91) | 1,885 | `6e5419989f2b9a70` |

```bash
# library
ss.config(detection="strict")

# environment: library, CLI and dashboard
export SURROGATESHIELD_PRESET=fast
export SURROGATESHIELD_DETECTION_CONFIG=/path/det.json   # JSON merged onto the preset

# CLI
surrogateshield scan --preset strict "..."               # fast, balanced or strict; also on mask, unmask, bench, doctor

# see the config in effect
surrogateshield doctor --show-config
```

The test-3 columns were measured with every address replaced, as `benchmark()` does, so those runs carry their own hashes (the `balanced` run's is `a1abaa92142b0db9`); the Hash column is the preset as shipped.

Five ablation configs are also built in, each with one part of `balanced` switched off, for experiments: `no_tagger` (`798d34bb643dc2dc`), `no_canonicaliser` (`8eea8f77acd7b97a`), `no_gate` (`8a36546801cf1dc4`), `no_models` (`67fa4d27a25d2aef`), `no_structural` (`4ac83b16fbb5307d`). `preset()` and `SURROGATESHIELD_PRESET` accept these names too. The benchmark itself used `benchmark()`: `balanced` with every address replaced.

> **Which preset?** Keep `balanced`. Use `strict` when replacing too much is fine and missing anything is not; `fast` to skip loading spaCy; `classic` only without the tagger's weights. The CLI's `--preset` takes `fast`, `balanced` and `strict`; choose `classic` through `SURROGATESHIELD_PRESET` or the library.

<a id="detection-config"></a>

## DetectionConfig

A frozen dataclass: every change returns a new config. It is validated when built (an unknown type, stage, action or threshold raises `ValueError`), converts to and from JSON, and `config_hash()` is the SHA-256 of its canonical JSON.

| Field | Default (`balanced`) | Meaning |
|---|---|---|
| `preset` | `"balanced"` | The preset it started from |
| `detectors` | the six built-in stages | A tuple of `Stage`s. Built-in stages keep the order `pattern_scan`, `canonicaliser`, `entity_trace`, `context_guard`, `pii_tagger`, `structural`; plugins run after the model stages, in list order |
| `type_sources` | see [SentinelLayer](#sentinellayer) | Which stages may report each public type; `"*"` = any stage |
| `type_actions` | `{"ADDRESS": "auto"}` | What happens to each type ([type_actions](#type-actions)) |
| `gate` | `True` | The [relation gate](#relation-gate) on or off |
| `gate_bypass` | `()` | Types the gate never drops; `"*"` = all |
| `source_priority` | `pattern_scan`, `canonicaliser`, `structural`, `pii_tagger`, `context_guard`, `entity_trace` | Which stage's reading wins when two disagree ([Resolver](#resolver)) |
| `type_conflicts` | `LOCATION\|ORG`, `LOCATION\|PERSON`, `ORG\|PERSON` → `"score"` | For two types read at the same rank: the type that wins, or `"score"` |
| `address_shift_range` | 1 | House-number shift |
| `service_queries` | `True` | Service-query detection |

### Stage fields

| Field | Meaning |
|---|---|
| `name`, `enabled` | The stage and whether it runs |
| `model`, `revision` | The model and its pinned revision (a hub commit, a spaCy version, or `sha256:` of the tagger's weights) |
| `device` | -1 for the CPU, 0 or more for a GPU |
| `thresholds` | The stage's own keys (`entity_trace`: `high`, `low`, `fallback`; `context_guard`: `accept`) and, per public type, the lowest score kept. All in 0–1 |
| `window`, `stride` | Positive ints for windowed models |
| `max_latency_ms` | A stage that runs longer is logged and reported as `<stage>_over_budget_ms`; its findings are kept |
| `options` | Stage options, e.g. `{"views": [...]}` for the canonicaliser, `{"gate_above": 0.97}` for the tagger, `{"labels": {...}}` for ContextGuard |

### Building a config

```python
from surrogateshield import preset, DetectionConfig
import surrogateshield as ss

cfg = (preset("balanced")
       .with_stage("context_guard", enabled=True, thresholds={"accept": 0.6}, device=0)
       .with_stage("pii_tagger", thresholds={"PERSON": 0.6})       # thresholds and options merge
       .with_actions(PHONE="redact", GENDER="keep")
       .replace(gate_bypass=("PERSON",)))

ss.config(detection=cfg)
print(cfg.config_hash()[:16])
open("det.json", "w").write(cfg.to_json())
same = DetectionConfig.from_json(open("det.json").read())
```

| Method | Returns |
|---|---|
| `with_stage(name, **changes)` | A copy with that stage changed (`thresholds` and `options` merge); a new name is appended |
| `with_actions(**actions)` | A copy with those `type_actions` |
| `with_sources(**sources)` | A copy with those `type_sources` |
| `with_plugin(name, types=("*",), **stage)` | Adds a plugin stage and lists it as a source of `types` |
| `merged(overrides)` | A partial JSON config applied on top |
| `replace(**fields)` | Any other field changed |
| `to_dict()`, `to_json()`, `from_dict()`, `from_json()` | JSON form |
| `config_hash()` | SHA-256 of the canonical JSON |
| `validate()` | Runs the checks again (they run on every build); raises `ValueError` |
| `stage(name)`, `enabled(name)`, `plugin_stages()` | A stage, whether it runs, the enabled stages run through the detector registry (plugins and the PII tagger) |
| `allows(stage, type)`, `min_score(stage, type)` | Whether a stage may report a type; the lowest score kept for it |
| `action_for(type)`, `address_mode` (property) | The action for a type; the address mode (`auto`, `shift` or `replace`; `replace` when ADDRESS is redacted or kept) |
| `redacted_types()`, `redacts(type)`, `keeps(type)` | The types set to `redact`; whether a type is redacted or kept |
| `bypasses_gate(type)`, `vouches(stage, score)` | Whether the gate leaves a type alone (also when `gate` is off); whether a score reaches the stage's `gate_above` option, which lets a candidate past the gate |
| `rank(stage)`, `conflict_winner(a, b)` | A stage's place in `source_priority`; the winner for two types (`"score"` = by score) |

Module functions: `preset(name="balanced")`, `benchmark()`, `from_partial(overrides, base=None)` (a partial dict or JSON merged onto a preset), `from_env()` (the config the environment variables select), `env_selected()` (whether either is set), `from_settings(base=None, **settings)` (the older keyword settings applied to a config).

### Partial JSON

A JSON config can be partial. It merges onto the preset it names: `detectors` merge by stage name; `type_sources`, `type_actions` and `type_conflicts` merge by key; anything else replaces.

```json
{
  "preset": "balanced",
  "detectors": [{"name": "context_guard", "enabled": true, "thresholds": {"accept": 0.6}, "device": 0}],
  "type_actions": {"PHONE": "redact", "GENDER": "keep"},
  "gate_bypass": ["EMAIL"]
}
```

Use it with `ss.config(detection={...})`, `SURROGATESHIELD_DETECTION_CONFIG=det.json`, or `--detection-config det.json`.

<a id="type-actions"></a>

## type_actions

What happens to each type once detected. A key is a public type (`"PHONE"`) or an internal one (`"phone_uk"`); the internal key wins. `"*"` sets every type not listed.

| Action | Effect | Restored in the answer |
|---|---|---|
| `replace` (default) | A realistic surrogate of the same kind | Yes |
| `auto` | ADDRESS only (its default): `shift` in a service query, `replace` otherwise | Yes |
| `shift` | ADDRESS only: the same street and city, the house number moved by ±`address_shift_range` | Yes |
| `redact` | A placeholder such as `[PHONE_1]`, unique within the session | Yes |
| `keep` | Detected and reported (`masked=False` in `scan`), sent as written | — |

```python
cfg = preset("balanced").with_actions(PHONE="redact", GENDER="keep")
cfg = preset("balanced").with_actions(ADDRESS="replace")      # never shift
cfg = preset("balanced").with_actions(**{"*": "redact"})      # placeholders for everything
```

`pii_off` ([see below](#library-pii-off)) is the simpler switch for `keep`: a list of aliases (`phone`, `name`, `location`, …) or type names written the way `scan()` spells them (`email`, `phone_uk`). Other spellings switch nothing off.

---

<a id="library-install"></a>

## Library Installation

The Python library is the `surrogateshield` package in the repository's `python-library/` folder: version `2.1.0`, Apache-2.0, Python 3.9 or later. It runs entirely in your process. Detection needs model files on the machine: the spaCy model and the PII tagger's weights for the default preset, and distilbert-NER only when ContextGuard is switched on.

### 1. Install the package

From the repository root:

```bash
pip install ./python-library
```

This installs faker ≥24.0.0, cryptography ≥42.0.0, rapidfuzz ≥3.6.0, spaCy ≥3.7.0, transformers ≥4.40.0 and torch ≥2.0.0, and puts the `surrogateshield` command on your `PATH` (`python -m surrogateshield` works too). The `display` extra adds rich ≥13.7.0, which `detailed_view` uses to draw its tables:

```bash
pip install "./python-library[display]"
```

### 2. Download the spaCy model

```bash
python -m spacy download en_core_web_lg
```

EntityTrace reads places with it. The presets pin version 3.8.0; `surrogateshield doctor` prints the version you have.

### 3. Install the PII tagger's weights

```bash
PYTHONPATH=.:python-library python -m bench.tagger.install bench/tagger/build/models/dv3s-40k
```

Run it from the repository root. It copies the trained tagger to `$SURROGATESHIELD_MODELS/pii-tagger-dv3s-40k` (default `~/.cache/surrogateshield/models`). The default preset pins the weights by SHA-256 (`sha256:f9e5821151ac…`) and checks them every time it loads them: a folder with other weights raises `DetectorUnavailable` and nothing is masked.

### 4. Check the install

```bash
surrogateshield doctor --smoke
```

Every row should read `ok`. Real output (paths shortened):

```text
detection config: preset balanced, hash 9470f9336d853f67 (default)
  stage pattern_scan   on
  stage canonicaliser  on
  stage entity_trace   on, en_core_web_lg@3.8.0, fallback=0.75 high=0.9 low=0.7
  stage context_guard  off, dslim/distilbert-NER@dfa2838a1273, accept=0.7
  stage pii_tagger     on, pii-tagger-dv3s-40k@sha256:f9e58, ADDRESS=0.9 CREDENTIAL=0.97 HANDLE=0.95 ID=0.95 LOCATION=0.7 ORG=0.7 PERSON=0.7 PHONE=0.95
  stage structural     on
  actions        ADDRESS auto (others replace)
  relation gate  on
[  ok] python 3.13.2
[  ok] import faker
[  ok] import cryptography
[  ok] import rapidfuzz
[  ok] import spacy
[  ok] import transformers
[  ok] import torch
[  ok] spaCy model en_core_web_lg — version 3.8.0, licence MIT
[  ok] ContextGuard model dslim/distilbert-NER@dfa2838a1273 — licence Apache-2.0
[  ok] PIITagger model pii-tagger-dv3s-40k — ~/.cache/surrogateshield/models/pii-tagger-dv3s-40k, weights sha256:f9e5821151ac…, licence MIT
[  ok] storage dir ~/.surrogateshield — mode 0700
[  ok] device key — mode 0600
[  ok] smoke: mask hides an email and an SSN
```

### Models

| Model | Stage | When it is needed | Where it comes from | Licence |
|---|---|---|---|---|
| `en_core_web_lg` 3.8.0 | EntityTrace (spaCy) | `balanced`, `strict`, `classic` | `python -m spacy download en_core_web_lg` | MIT |
| `pii-tagger-dv3s-40k` | PII tagger | `balanced`, `fast`, `strict` | A local folder under `$SURROGATESHIELD_MODELS`, pinned by SHA-256 | MIT |
| `dslim/distilbert-NER` at revision `dfa2838a1273` | ContextGuard | `strict`, `classic`, or `context_guard_enabled=True` | Hugging Face, downloaded into the Hugging Face cache by the first scan that runs ContextGuard | Apache-2.0 |

> **No tagger weights on this machine?** Set `SURROGATESHIELD_PRESET=classic`. The classic preset reads names with spaCy and ContextGuard and runs without the tagger. The `--preset` flag of the command line accepts only `fast`, `balanced` and `strict`, so use the environment variable there too.

> **Offline.** Once the models are on disk, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and the Hugging Face libraries load only local files. To put ContextGuard's model into the cache ahead of time, run one scan with it switched on while online, for example `surrogateshield scan --preset strict "warm up"`. `doctor --cold-start` loads only models that are already present: it is skipped when a required row fails, and a missing ContextGuard model is a required row whenever ContextGuard is on.

<a id="library-quickstart"></a>

## Library Quick Start

Create one `Session` per conversation. Mask what you send, call your model with the masked text, and unmask the reply. The library makes no provider calls; you keep your own client.

- **mask_result(text)** (Session): Detects the personal values, picks a surrogate for each, stores `surrogate → original` in the session's map and returns the text to send.
- **your provider call** (Anthropic, OpenAI, Gemini, Ollama, …): The model reads and answers the masked text.
- **unmask(reply)** (Session): Puts the original values back into the reply. Takes a string or the provider's response object.
- **close()** (Session): Discards the map (and its file, for a persistent session). Leaving a `with` block does it for you.

```python
import surrogateshield as shield

text = ("Hi, I'm Priya Raman and I work at Northwind Traders in Leeds. "
        "My email is priya.raman@northwind-traders.co.uk and my phone is +44 7700 900123. "
        "I'm 34 years old.")

with shield.Session(seed=7) as session:
    result = session.mask_result(text)
    print(result.text)                          # send this

    reply = "Thanks Forum Rao, noted."           # what the model answered
    print(session.unmask(reply))                # Thanks Priya Raman, noted.
```

**Your text:**

> Hi, I'm Priya Raman and I work at Northwind Traders in Leeds. My email is priya.raman@northwind-traders.co.uk and my phone is +44 7700 900123. I'm 34 years old.

**result.text:**

> Hi, I'm Forum Rao and I work at Hart Foster in Cambridge. My email is forum.rao@craig-miller.co.uk and my phone is +44 7830 166131. I'm 37 years old.

`result.detections` for that message:

| text | type | start–end | score | source |
|---|---|---|---|---|
| Priya Raman | `PERSON` | 8–19 | 0.9 | `structural` |
| Northwind Traders | `ORG` | 34–51 | 0.9 | `structural` |
| Leeds | `GPE` | 55–60 | 0.85 | `ner` |
| priya.raman@northwind-traders.co.uk | `email` | 74–109 | 1.0 | `pattern` |
| +44 7700 900123 | `phone_uk` | 126–141 | 1.0 | `pattern` |
| 34 years old | `age` | 147–159 | 1.0 | `pattern` |

The surrogates keep the shape of each value: the email's local part is rebuilt from the surrogate name, the UK mobile keeps `+44 7` and its grouping, the age moves by a few years and keeps "years old", Leeds becomes another English town. `seed=7` makes the run repeatable: the same seed and the same messages give the same surrogates. Use a seed in tests and audits. Leave it out otherwise, and every session draws a fresh one from the operating system.

### Without a Session object

The module-level functions use the current context's session, created on first use:

```python
import surrogateshield as shield

sent = shield.mask("My name is Priya Raman and I live in Leeds.")
reply = ask_my_model(sent)                       # your provider call
print(shield.unmask(reply))
shield.flush()                                   # end the session and drop its map
```

That is convenient in a script or a notebook. In a server, create one `Session` per conversation ([Sessions](#library-sessions)).

<a id="library-sessions"></a>

## Sessions

A `Session` is the masking state of one conversation: its surrogate map (the ShadowMap, `surrogate → original`), its surrogate generator and the record of the last message it masked. Within a session, the same original always gets the same surrogate, so the model sees one consistent person, company or address across turns. Two sessions never see each other's values.

Session(session_id: Optional[str] = None, *, config: Optional[Config] = None, storage_dir: Optional[str] = None, seed: Optional[int] = None)

| Argument | Default | Meaning |
|---|---|---|
| `session_id` | a random `uuid4().hex` | The conversation's id. It must match `^[A-Za-z0-9_-]{1,64}$`, otherwise `ValueError: invalid id 'bad id!': must match ^[A-Za-z0-9_-]{1,64}$`. With persistent storage, the same id reopens the same map. |
| `config` | a copy of the module settings | A `Config` for this session only. The session copies it, so changing your object later has no effect on it. |
| `storage_dir` | `None` | A directory for an encrypted map ([Persistent Sessions](#library-persistent)). `None` keeps the map in memory, unless `config.pii_mem` names a directory, which is then used. `~` is expanded. |
| `seed` | `None` | Seed for the surrogate generator. `None` draws one from the operating system. |

### What a session offers

| Member | Returns | What it does |
|---|---|---|
| `scan(text)` | `List[Detection]` | Detects without masking ([scan()](#library-scan)) |
| `mask(text)` | `str` | The masked text |
| `mask_result(text)` | `MaskResult` | The masked text, every detection and the replacements ([MaskResult](#library-mask-result)) |
| `unmask(response)` | `str` | The reply with the originals restored |
| `ascan`, `amask`, `amask_result`, `aunmask` | the same, awaitable | Run in a worker thread ([Async](#library-async)) |
| `forget(original)` | `int` | Erases every mapping to that original value and returns how many were removed; a persistent map is rewritten at once |
| `close()` | `None` | Discards every mapping and deletes the persistent file. Calling it twice is harmless |
| `mappings` | `Dict[str, str]` | A copy of the map, `surrogate → original` |
| `detection_config` | `DetectionConfig` | The detection configuration this session runs ([Configuration](#library-config)) |
| `id` | `str` | The session id |
| `config` | `Config` | This session's settings |

`repr(session)` shows the id and the number of mappings, never the values: `<surrogateshield.Session id=conv-42 mappings=3>`.

### Lifecycle

- `with Session() as s:` and `async with Session() as s:` call `close()` when the block ends, whether it ends normally or with an exception.
- After `close()`, every method raises `RuntimeError: session <id> is closed`.
- `forget("Priya Raman")` returns `1` after the quick-start message: one surrogate pointed at that name. Use it to honour a request to erase one value and keep the rest of the conversation.
- A session can be shared between threads: every method takes the session's lock. Detection itself runs outside the lock, so two threads masking in the same session do not wait for each other's models.

### Restoring short surrogates

Some surrogates are too short or too common to look up anywhere in a reply: a bare age ("72"), a gender term, a value of one or two characters. `unmask()` restores those only if the most recent `mask()` of the same session sent them, so a "72" in an answer about a later message is left alone. A first name or surname that the user typed unmasked in that message ("Peter the Great") is not taken for part of a surrogate. Unmask each reply before you mask the next message, and keep one `Session` object per conversation in a process.

### The current session

The module-level functions (`shield.mask()`, `shield.unmask()`, `shield.scan()`, `shield.forget()`, `shield.flush()`) act on `current_session()`:

- the session bound with `with shield.use_session(session):`, if there is one. Asyncio tasks and `asyncio.to_thread()` calls started inside the block see it too;
- otherwise a session the calling thread or asyncio task created on first use. That session belongs to the thread and task that made it. Another thread or task, even one started with a copy of the context, gets its own. A warm-up `shield.mask()` at server start therefore never turns into one map shared by every request.

```python
import surrogateshield as shield

session = shield.Session("support-1187")
with shield.use_session(session):
    sent = shield.mask("Hi, I'm Priya Raman.")     # uses `session`
    print(shield.current_session() is session)    # True
```

`shield.flush()` closes the current session; the next call in that context starts a new one. `current_session(create=False)` returns `None` instead of creating one. `loop.run_in_executor()` does not copy the context, so pass the `Session` to the worker function yourself.

<a id="library-api"></a>

## API Reference

Everything below is importable from `surrogateshield` (`import surrogateshield as shield`). The package's `__all__`: `config`, `scan`, `pii_finder`, `mask`, `mask_result`, `unmask`, `forget`, `flush`, `Session`, `Config`, `Detection`, `MaskResult`, `current_session`, `use_session`, `DetectorUnavailable`, `StorageError`, `__version__`, `DetectionConfig`, `preset`, `register_detector`, `Candidate`.

### Module functions

config(*, detailed_view, pii_mem, pii_off, service, spacy_model, context_guard_enabled, entity_trace_high_threshold, entity_trace_low_threshold, context_guard_threshold, entity_trace_fallback_threshold, fuzzy_threshold, address_mode, address_shift_range, verify_addresses, context_guard_model, context_guard_device, detection) → Config

Changes the settings new sessions start from, and the current context's session at once. Every argument is keyword-only and optional; only the ones you pass change. `shield.config()` with no arguments returns the current settings. Raises `ValueError` for an invalid value and `RuntimeError` when `pii_mem` changes while the current session holds mappings. Every key: [Configuration](#library-config).

scan(text: str, *, as_dict: bool = False) → List[Detection] | Dict[str, str]

Detects personal values in `text` with the current session's settings and changes nothing. `as_dict=True` returns `{text: type}` instead. `pii_finder` is the same function. Raises `DetectorUnavailable` and `TypeError`.

mask(text: str) → str

Returns `text` with every detected value replaced by its surrogate, and records the mapping in the current session. Raises `DetectorUnavailable` (nothing is masked; do not send the text) and `TypeError` if `text` is not a `str`.

mask_result(text: str) → MaskResult

Like `mask()`, and also returns every detection (those sent as is included) and the replacements made.

unmask(response) → str

Restores the originals in a reply. `response` is a `str`, or an Anthropic, OpenAI or Gemini response as an SDK object or a dict ([Providers](#library-providers)). Raises `TypeError` for `None` or an object with no text it can find.

forget(original: str) → int

Erases every mapping to `original` from the current session and returns the number removed.

flush() → None

Ends the current session: its mappings and its persistent file are discarded. The next call in this context starts a new session.

current_session(*, create: bool = True) → Optional[Session]

The session the module functions use in this context: the one bound with `use_session()`, else the one this thread or task created. With `create=False`, returns `None` rather than creating one.

use_session(session: Session) → Iterator[Session]

A context manager that binds `session` as the current session for the `with` block, and for asyncio tasks and `asyncio.to_thread()` calls started inside it. Raises `TypeError: use_session() needs a surrogateshield.Session` for anything else.

preset(name: str = "balanced") → DetectionConfig

A preset detection configuration: `fast`, `balanced`, `strict` or `classic`, plus the ablations `no_tagger`, `no_canonicaliser`, `no_gate`, `no_models` and `no_structural`. Any other name raises `ValueError: unknown preset 'nope'; presets: (...), ablations: (...)`. See [Presets](#presets).

register_detector(name: str, factory, *, replace: bool = False) → None

Makes `factory` the detector of the plugin stage `name` ([Plugins](#library-plugins)).

### Session

Session(session_id: Optional[str] = None, *, config: Optional[Config] = None, storage_dir: Optional[str] = None, seed: Optional[int] = None)

One conversation's masking state ([Sessions](#library-sessions)). Raises `ValueError` for an invalid id, `ShadowMapStorageError` when the device key cannot be read or written, and `CorruptStoreError` when an existing map cannot be decrypted (both are `StorageError`s).

Session.scan(text: str) → List[Detection]

Session.mask(text: str) → str

Session.mask_result(text: str) → MaskResult

Session.unmask(response) → str

Session.forget(original: str) → int

Session.close() → None

The module functions above, on this session. `Session.scan()` has no `as_dict`.

async Session.ascan(text: str) → List[Detection]

async Session.amask(text: str) → str

async Session.amask_result(text: str) → MaskResult

async Session.aunmask(response) → str

The same methods run in a worker thread with `asyncio.to_thread()`: the same results, the same exceptions.

Session.mappings → Dict[str, str] · Session.detection_config → DetectionConfig · Session.id → str · Session.config → Config

A copy of the map (`surrogate → original`), the effective detection configuration, the id and the settings.

### Data classes

Detection(text: str, type: str, start: int, end: int, score: float, source: str, masked: bool = True)

One detected value; frozen. Fields: [scan() and Detection](#library-scan).

MaskResult(text: str, detections: Tuple[Detection, ...], replacements: Dict[str, str], service_query: bool)

What one `mask_result()` did; frozen. Property `unmasked`: the detections sent as is. Fields: [MaskResult](#library-mask-result).

Config(detailed_view=False, pii_mem="temp", pii_off=[], service=True, spacy_model="en_core_web_lg", context_guard_enabled=False, entity_trace_high_threshold=0.9, entity_trace_low_threshold=0.7, context_guard_threshold=0.7, entity_trace_fallback_threshold=0.75, fuzzy_threshold=85, address_mode="auto", address_shift_range=1, context_guard_model="dslim/distilbert-NER", context_guard_device=-1, detection=None)

A session's settings, a plain dataclass. `config()` validates what you pass; a `Config` you build yourself is not validated until a session uses it.

Candidate(start: int, end: int, type: str, score: float = 1.0)

One span a detector plugin reports; frozen. Offsets index the message.

DetectionConfig

The frozen detection configuration: stages, models, thresholds, type routing, per-type actions, the relation gate. Its methods: `with_stage`, `with_actions`, `with_sources`, `with_plugin`, `merged`, `replace`, `to_dict`, `to_json(indent=2)`, `from_dict`, `from_json`, `config_hash`, `stage`, `enabled`, `plugin_stages`, `allows`, `action_for`, `redacts`, `keeps`. See [DetectionConfig](#detection-config).

### Exceptions and version

DetectorUnavailable(RuntimeError) · StorageError(RuntimeError) · __version__ = "2.1.0"

See [Errors](#library-errors).

<a id="library-scan"></a>

## scan() and Detection

`scan()` runs the same detection as `mask()`, with the same settings (presets, service queries, `pii_off`), and changes nothing: no surrogate is drawn and the map is untouched. Use it to see what a message contains, to log types without values, or to decide whether a message may be sent at all.

```python
import surrogateshield as shield

text = ("Hi, I'm Priya Raman and I work at Northwind Traders in Leeds. "
        "My email is priya.raman@northwind-traders.co.uk and my phone is +44 7700 900123. "
        "I'm 34 years old.")

for d in shield.scan(text):
    print(d.type, d.source, d.start, d.end, d.text)

shield.scan(text, as_dict=True)
# {'Priya Raman': 'PERSON', 'Northwind Traders': 'ORG', 'Leeds': 'GPE',
#  'priya.raman@northwind-traders.co.uk': 'email', '+44 7700 900123': 'phone_uk',
#  '34 years old': 'age'}
```

The detections come in order of position, one per span. With `as_dict=True` the same text found twice appears once.

### Detection fields

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | The value as written in the message |
| `type` | `str` | The internal type name: `email`, `phone_uk`, `PERSON`, `GPE`, `id_number`, … (the public type each belongs to: [PII Types](#pii-types)) |
| `start`, `end` | `int` | Character offsets in the scanned text |
| `score` | `float` | The finding stage's confidence; pattern matches report `1.0` |
| `source` | `str` | The stage that found it (below) |
| `masked` | `bool` | `False` when the value is sent as is: its type is in `pii_off`, or its type's action is `keep` |

### Sources

| `source` | Stage |
|---|---|
| `pattern` (Regex) | PatternScan, and hits the canonicaliser found on a rewritten view of the message (look-alike characters folded, spelled-out digits written as digits) |
| `pii_tagger` | The PII tagger |
| `ner` (NER) | EntityTrace (spaCy) |
| `slm` (NER) | ContextGuard (distilbert-NER) |
| `structural` | The structural passes, which read names and organisations from the message's wording and layout |
| a plugin's name | That plugin, e.g. `employee_ids` ([Plugins](#library-plugins)) |

When two stages find the same span, the resolver keeps one detection and its `source` is the stage that ranks higher in `source_priority`.

### Seeing it in the terminal

`shield.config(detailed_view=True)` prints a table after every scan and mask: "SurrogateShield — Scan Results" (Detected Value, Type, Span, Score, Source, Masked) and "SurrogateShield — Masked" (Original, Type, Score, Source, Surrogate). After `unmask()` it prints `[SurrogateShield] Restored N surrogate(s)`. The tables use rich when the `display` extra is installed, and plain text otherwise.

> **detailed_view prints the original values** to standard output. It is off by default. Switch it on only where that output stays on your screen and out of logs.

<a id="library-mask-result"></a>

## mask_result() and MaskResult

`mask()` returns only the text. `mask_result()` returns a `MaskResult` that also says what was found, what was replaced with what, and what was sent as is.

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | The masked text, ready to send |
| `detections` | `Tuple[Detection, ...]` | Every detection in the message, masked or not |
| `replacements` | `Dict[str, str]` | `original → surrogate` for every value replaced in this message, including values that reuse a surrogate from an earlier turn. A redacted value maps to its placeholder, such as `[EMAIL_1]` |
| `service_query` | `bool` | `True` when the message was read as a service query ([Addresses](#library-address)) |
| `unmasked` (property) | `Tuple[Detection, ...]` | The detections with `masked=False`: sent exactly as written |

The quick-start message with phones switched off, `Session(seed=7, config=Config(pii_off=["phone"]))`:

```text
>>> r.text
"Hi, I'm Forum Rao and I work at Hart Foster in Cambridge. My email is forum.rao@craig-miller.co.uk and my phone is +44 7700 900123. I'm 31 years old."
>>> r.unmasked
(Detection(text='+44 7700 900123', type='phone_uk', start=126, end=141, score=1.0, source='pattern', masked=False),)
>>> r.replacements
{'Priya Raman': 'Forum Rao', 'priya.raman@northwind-traders.co.uk': 'forum.rao@craig-miller.co.uk',
 'Northwind Traders': 'Hart Foster', 'Leeds': 'Cambridge', '34 years old': '31 years old'}
```

The phone went out as written. The age differs from the quick start's 37 because, with the phone skipped, the seeded generator drew one value fewer before it.

### Redact and keep

Per-type actions ([type_actions](#type-actions)) also show in the result. With `EMAIL` redacted and `PHONE` kept:

```python
import surrogateshield as shield

det = shield.preset("balanced").with_actions(EMAIL="redact", PHONE="keep")
s = shield.Session(seed=7, config=shield.Config(detection=det))
r = s.mask_result("Write to priya.raman@northwind-traders.co.uk or ring +44 7700 900123, Priya Raman here.")

r.text           # 'Write to [EMAIL_1] or ring +44 7700 900123, Forum Rao here.'
r.replacements   # {'Priya Raman': 'Forum Rao', 'priya.raman@northwind-traders.co.uk': '[EMAIL_1]'}
[d.type for d in r.unmasked]                   # ['phone_uk']
s.unmask("I emailed [EMAIL_1] for you, Lipika.")
# 'I emailed priya.raman@northwind-traders.co.uk for you, Lipika.'
```

A placeholder the model repeats is restored like any surrogate. A kept value is detected and reported with `masked=False`, exactly like a `pii_off` type.

### When detection fails

If a stage that is switched on cannot run (a model is missing, the tagger's weights do not match their pin, a plugin is not registered, or a model fails on this input), `mask()` and `mask_result()` raise `DetectorUnavailable` and return nothing. The text is never passed through unmasked. Catch it and do not send the message ([Errors](#library-errors)).

<a id="library-pii-off"></a>

## Sending Types As Is (pii_off)

`pii_off` lists types that are detected but sent unchanged. They still appear in `scan()` and in `MaskResult.detections`, with `masked=False`, and in `MaskResult.unmasked`. No surrogate is drawn for them and nothing is added to the map. A typical use: a support bot that must pass the customer's order number through to a lookup tool.

```python
import surrogateshield as shield

shield.config(pii_off=["phone", "org"])                         # module settings
s = shield.Session(config=shield.Config(pii_off=["email"]))     # one session
```

From the command line, put the text before the flag (or end the list with `--`), because `--pii-off` takes any number of values:

```bash
surrogateshield scan "My phone is +44 7700 900123." --pii-off phone
   12-27    phone_uk           pattern  '+44 7700 900123'  (pii_off: sent as is)
```

### Aliases

| Alias | Switches off |
|---|---|
| `phone` | `phone_us`, `phone_uk`, `phone_intl` |
| `postal_code` | `zip_us`, `postcode_uk` |
| `zip` | `zip_us` |
| `postcode` | `postcode_uk` |
| `name`, `names` | `PERSON` |
| `location` | `GPE`, `LOC` |
| `org` | `ORG` |
| `facility` | `FAC` |
| `crypto` | `crypto` |
| `bank` | `us_bank_number` |
| `license` | `us_driver_license` |
| `username` | `handle` |
| `password` | `credential` |

Aliases are matched in any letter case.

### Type names

Besides the aliases, `config()` accepts these entries: `email`, `ssn`, `phone_us`, `phone_uk`, `phone_intl`, `address`, `person`, `credit_card`, `dob`, `ip_address`, `zip_us`, `postcode_uk`, `api_key`, `crypto`, `us_bank_number`, `us_driver_license`, `gpe`, `loc`, `org`, `fac`, `gender_indicator`, `implicit_location`, `iban`, `vin`, `mac_address`, `passport`, `id_number`, `license_plate`, `url`, `handle`, `credential`, `age`. Anything else raises:

```text
ValueError: Unknown pii_off entry 'phones'. Valid entries: address, age, api_key, bank, ...
```

The check ignores letter case, but an entry that is not an alias is compared with `Detection.type` exactly as written. Write it the way `scan()` spells the type:

| Entry | Effect |
|---|---|
| `"email"`, `"address"`, `"age"`, `"phone_uk"` | Switches that type off |
| `"EMAIL"`, `"ADDRESS"`, `"AGE"` | Passes the check and switches nothing off (the types are lower-case) |
| `"PERSON"`, `"GPE"`, `"ORG"` | Switches that type off |
| `"person"`, `"gpe"` | Passes the check and switches nothing off for names the tagger or spaCy report (they report `PERSON` and `GPE`); use `"name"` and `"location"` |
| `"LOCATION"`, `"Phone"` | Works: an alias in another case |
| `"ID"` | `ValueError`: public type names are not entries |

> **Keep a whole public type.** To send every `ID` or every `EMAIL` unchanged, whatever its internal name, use the `keep` action: `shield.config(detection=shield.preset().with_actions(ID="keep"))`. Kept values are reported with `masked=False` just like `pii_off` ones ([type_actions](#type-actions)).

The command line passes `--pii-off` values straight to the session without the check, so a misspelt entry there (`--pii-off phones`) switches nothing off and raises no error.

<a id="library-address"></a>

## Addresses and Service Queries

Three settings decide what happens to a street address: `address_mode`, `address_shift_range` and `service`. A service query is a message that asks for something near a place ("Find a pharmacy near …"). In a service query, a replaced address would give a useless answer about another town, so `auto` keeps the street and moves only the house number. How messages are classified: [Service Queries](#service-query).

| Setting | Values | Default | Effect |
|---|---|---|---|
| `address_mode` | `"auto"`, `"shift"`, `"replace"` | `"auto"` | `auto`: shift in a service query, "my area" in a service query on a sensitive topic, replace everywhere else. `shift`: every address keeps its street, town, state and postcode, and its house number moves. `replace`: every address becomes a made-up address of the same shape |
| `address_shift_range` | integer ≥ 1 | `1` | The largest change to the house number when shifting |
| `service` | `True`, `False` | `True` | Detect service queries. `False`: no message is a service query, so `auto` always replaces and `service_query` is always `False` |

Real output, a new `Session(seed=3)` for each cell:

| Message | `service_query` | `auto` | `shift` | `replace` |
|---|---|---|---|---|
| Find a pharmacy near 221 Hollis Avenue, Springfield, IL 62704. | `True` | 220 Hollis Avenue, Springfield, IL 62704 | 220 Hollis Avenue, Springfield, IL 62704 | 825 Diaz Avenue, Lancaster, ND 91907 |
| Is there a therapist near 221 Hollis Avenue, Springfield, IL 62704? | `True` | my area, Springfield, IL | 220 Hollis Avenue, Springfield, IL 62704 | 825 Diaz Avenue, Lancaster, ND 91907 |
| My home address is 221 Hollis Avenue, Springfield, IL 62704. | `False` | 825 Diaz Avenue, Lancaster, ND 91907 | 220 Hollis Avenue, Springfield, IL 62704 | 825 Diaz Avenue, Lancaster, ND 91907 |

**Sensitive service query:**

> Is there a therapist near 221 Hollis Avenue, Springfield, IL 62704?

**auto → coarse:**

> Is there a therapist near my area, Springfield, IL?

The street line becomes "my area" and the postcode goes, so the answer is still about Springfield. `unmask("Here are three therapists near my area, Springfield, IL.")` gives back "Here are three therapists near 221 Hollis Avenue, Springfield, IL 62704."

- In a service query the model stages do not report places, so a city named in it stays as written and the answer stays local.
- The first surrogate an address gets is reused for the rest of the session. If the same address comes back later in a message of another kind, it keeps that first surrogate, not the one the new message's mode would pick.
- In a [DetectionConfig](#detection-config), the same settings are `type_actions["ADDRESS"]`, `address_shift_range` and `service_queries`.
- `config(verify_addresses=...)` has no effect and emits a `DeprecationWarning`.

```python
import surrogateshield as shield

shield.config(address_mode="replace")              # always redraw addresses
shield.config(address_mode="shift", address_shift_range=4)
s = shield.Session(config=shield.Config(service=False))
```

<a id="library-persistent"></a>

## Persistent Sessions and Storage

By default a session's map lives in memory and ends with the process. Give it a `storage_dir` and it is written to an encrypted file, so a conversation can continue after a restart or in another worker process. Reopen it with the same id and the same directory.

```python
import surrogateshield as shield

STORE = "/var/lib/myapp/shadowmaps"

a = shield.Session("conv-42", storage_dir=STORE, seed=11)
sent = a.mask("Priya Raman's account number is 12345678 at Northwind Traders.")
# "Vrinda Shah's account number is 89328792 at Garcia Nguyen."

# later, in another process
b = shield.Session("conv-42", storage_dir=STORE)
b.unmask(sent)
# "Priya Raman's account number is 12345678 at Northwind Traders."
b.close()                                          # deletes conv-42.shadowmap
```

A reopened session keeps issuing the same surrogates for the same values, and new values never receive a surrogate already in use. Instead of passing `storage_dir` to each session, `shield.config(pii_mem="/var/lib/myapp/shadowmaps")` makes every new session persistent. The directory must already exist (the check does not expand `~`), and `pii_mem` cannot change while the current session holds mappings: call `shield.flush()` first or create a new `Session`.

### Files

| Path | Mode | Contents |
|---|---|---|
| `$SURROGATESHIELD_HOME` (default `~/.surrogateshield`) | 0700 | The library's home folder |
| `<home>/device.key` | 0600 | 32 random bytes, created once, the first time a persistent session opens. Two processes starting together agree on one key. Looser permissions are tightened, with a warning in the log |
| `<storage_dir>/` | 0700 | Created if missing |
| `<storage_dir>/<id>.shadowmap` | 0600 | The sealed map `{surrogate: original}` |
| `<home>/sessions/` | 0700 | Where the command line keeps maps unless `--store` says otherwise |

### Encryption

```python
file = b"SSv1" || nonce (12 bytes) || AES-256-GCM(key, JSON map, aad)
key  = HKDF-SHA256(ikm = device.key, salt = session id, info = b"shadowmap-v1")
aad  = b"shadowmap-v1" || b"|" || session id
```

- Each session has its own key, and the id is bound into the ciphertext: a file opens only under the id it was written for.
- The map is written after every change: each `mask()` that adds a mapping, and each `forget()` that removes one. Every write goes to a temporary file that is flushed to disk and renamed over the old one, so a crash leaves the previous map or the new one, never half of each. A write that fails raises `OSError` from `mask()`, because a map that cannot be stored cannot restore a later reply.
- A file that cannot be decrypted is moved aside to `<id>.shadowmap.corrupt-<timestamp>` and `CorruptStoreError` is raised. It never turns into an empty map.
- If `device.key` cannot be read or created, or is not 32 bytes long, `ShadowMapStorageError` is raised. No key is derived from a damaged file.
- Maps written by library versions before the `SSv1` format are still read, and are rewritten in the current format at the next save.
- `close()`, `shield.flush()` and `surrogateshield unmask --close` delete the file. `forget(original)` removes one value and rewrites the file at once.

> **The maps belong to one device key.** To move stored conversations to another machine, move `device.key` with them. A new `device.key` cannot open the old maps: opening one raises `CorruptStoreError` and moves it aside.

<a id="library-async"></a>

## Async

Detection is model inference and keeps a CPU busy. `ascan()`, `amask()`, `amask_result()` and `aunmask()` run the method of the same name in a worker thread through `asyncio.to_thread()`, so the event loop keeps serving other requests in the meantime. The results and exceptions are those of the synchronous methods.

```python
import asyncio
import surrogateshield as shield

async def main():
    async with shield.Session(seed=2) as s:
        sent = await s.amask("Call me on +44 7700 900123.")
        print(sent)                               # Call me on +44 7011 524493.
        print(await s.aunmask(sent))              # Call me on +44 7700 900123.
        found = await s.ascan("priya.raman@northwind-traders.co.uk")

asyncio.run(main())
```

- `async with Session()` closes the session when the block ends.
- The async methods exist on `Session` only. The module functions are synchronous; in async code, call a session's `a*` methods.
- A session bound with `use_session()` stays current in child tasks and in `asyncio.to_thread()` calls started inside the block, because both copy the context.
- A session the module functions create on their own belongs to the task that created it: other tasks get their own, so concurrent requests never share one map by accident.
- `loop.run_in_executor()` does not copy the context. Pass the `Session` to the function you run there.
- Two coroutines may use one session at once (its lock keeps the map consistent), but a conversation's turns should still go one at a time. Then each reply is unmasked against the message it answers ([Restoring short surrogates](#library-sessions)).

<a id="library-providers"></a>

## Using It With Providers

The library does not call any LLM provider; you make the call with your own client. `unmask()` takes either the reply text or the response object, and finds the text in these shapes:

| You pass | `unmask()` reads |
|---|---|
| a `str` | the string |
| an Anthropic `Message`, as an object or a dict | every `text` block of `content`, joined; `tool_use` and `thinking` blocks are skipped |
| an OpenAI chat completion, as an object or a dict | `choices[0].message.content`; `None` (a reply with only tool calls) gives `""`. A legacy completion's `choices[0].text` also works |
| a Gemini response, or any object with a string `.text` | `.text` |
| `None` | `TypeError: unmask() got None — pass the LLM response text or object` |
| anything else | `TypeError: unmask() cannot find text in a int; pass a str or an Anthropic / OpenAI / Gemini response` |

For other shapes, pass the text: the OpenAI Responses API's `response.output_text`, Ollama's `response.message.content`.

### Anthropic

```python
import anthropic
import surrogateshield as shield

client = anthropic.Anthropic()

with shield.Session() as s:
    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": s.mask(user_text)}],
    )
    print(s.unmask(message))
```

### Anthropic, streaming

```python
with shield.Session() as s:
    with client.messages.stream(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": s.mask(user_text)}],
    ) as stream:
        answer = "".join(stream.text_stream)
    print(s.unmask(answer))
```

Unmask the whole answer once the stream ends: a surrogate can be split across two chunks.

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

with shield.Session() as s:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": s.mask(user_text)}],
    )
    print(s.unmask(completion))

    response = client.responses.create(model="gpt-4o-mini", input=s.mask(user_text))
    print(s.unmask(response.output_text))
```

### Gemini

```python
from google import genai

client = genai.Client()

with shield.Session() as s:
    response = client.models.generate_content(model="gemini-2.5-flash", contents=s.mask(user_text))
    print(s.unmask(response))
```

### Ollama

```python
import ollama

with shield.Session() as s:
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": s.mask(user_text)}])
    print(s.unmask(response.message.content))
```

With a local model, masking still keeps the values out of the prompts, logs and caches the model server keeps.

<a id="library-multiturn"></a>

## Multi-Turn Conversations

A chat sends the whole history with every request. Keep that history in its masked form, the way the model saw it, and mask only the new message each turn.

### 1. One session per conversation

Create it when the conversation starts and keep it until it ends. For conversations that outlive a process, give it the conversation's id and a `storage_dir`.

### 2. Mask the new user message

Append the masked text to the history as the user turn.

### 3. Store the model's reply as it came

Append the reply, surrogates and all, as the assistant turn. That keeps the history consistent with what the model wrote.

### 4. Unmask a copy for the user

Show the user `session.unmask(reply)`. Do it before masking the next message.

```python
import anthropic
import surrogateshield as shield

client = anthropic.Anthropic()
session = shield.Session("support-1187", storage_dir="~/.myapp/shadowmaps")
history = []                                     # what the model has seen

def turn(user_text: str) -> str:
    history.append({"role": "user", "content": session.mask(user_text)})
    message = client.messages.create(model="claude-opus-5-5", max_tokens=1024, messages=history)
    answer = "".join(b.text for b in message.content if b.type == "text")
    history.append({"role": "assistant", "content": answer})
    return session.unmask(answer)                # for the user's screen only
```

Real output of two turns in one `Session(seed=21)`:

| Turn | User typed | Model received |
|---|---|---|
| 1 | Hi, I'm Priya Raman and I manage the Leeds branch of Northwind Traders. | Hi, I'm Udyati Mander and I manage the Shrewsbury branch of Dickerson Ewing. |
| 2 | Can you draft a note from Priya Raman to the Leeds team about Friday's stock count? | Can you draft a note from Udyati Mander to the Shrewsbury team about Friday's stock count? |

Priya Raman stays Udyati Mander and Leeds stays Shrewsbury for the whole conversation, and the same holds after the session is reopened from its file. The [FastAPI example](#library-fastapi) applies the same pattern to a server.

<a id="library-config"></a>

## Configuration

Settings live in a `Config`. There are two ways to set them:

- `shield.config(**changes)` changes the module settings that new sessions copy, and applies the change to the current context's session at once. Other `Session` objects that already exist keep their own copy. Only the arguments you pass change; `shield.config()` with no arguments returns the settings.
- `Session(config=shield.Config(...))` gives one session its own settings.

| Key | Type | Default | What it does | In a DetectionConfig |
|---|---|---|---|---|
| `detailed_view` | `bool` | `False` | Prints scan, mask and unmask tables with the original values to stdout ([scan()](#library-scan)) | — |
| `pii_mem` | `str` | `"temp"` | `"temp"` keeps maps in memory; an existing directory makes new sessions persistent there ([Persistent Sessions](#library-persistent)) | — |
| `pii_off` | `list[str]` | `[]` | Types detected and sent as is ([pii_off](#library-pii-off)) | — |
| `service` | `bool` | `True` | Detects service queries ([Addresses](#library-address)) | `service_queries` |
| `spacy_model` | `str` | `"en_core_web_lg"` | EntityTrace's spaCy model. Another name drops the pinned version | `entity_trace` model |
| `context_guard_enabled` | `bool` | `False` | Runs ContextGuard (distilbert-NER) as a second reader of names | `context_guard` enabled |
| `entity_trace_high_threshold` | `float` in [0, 1] | `0.9` | A spaCy entity at or above this score is confirmed | `entity_trace` `high` |
| `entity_trace_low_threshold` | `float` in [0, 1] | `0.7` | At or above this score it is borderline | `entity_trace` `low` |
| `entity_trace_fallback_threshold` | `float` in [0, 1] | `0.75` | With ContextGuard off, a borderline entity at or above this score is confirmed | `entity_trace` `fallback` |
| `context_guard_threshold` | `float` in [0, 1] | `0.7` | With ContextGuard on, its score at or above which a borderline entity is confirmed | `context_guard` `accept` |
| `fuzzy_threshold` | number in [0, 100] | `85` | The rapidfuzz partial-ratio score at which `unmask()` accepts a surrogate the model wrote slightly differently | — |
| `address_mode` | `"auto"` \| `"shift"` \| `"replace"` | `"auto"` | What happens to addresses | `type_actions["ADDRESS"]` |
| `address_shift_range` | `int` ≥ 1 | `1` | Largest house-number change when shifting | `address_shift_range` |
| `verify_addresses` | any | — | Deprecated and ignored; emits a `DeprecationWarning` | — |
| `context_guard_model` | `str` | `"dslim/distilbert-NER"` | ContextGuard's Hugging Face model. Another id drops the pinned revision | `context_guard` model |
| `context_guard_device` | `int` | `-1` | `-1` = CPU, `0` and up = that GPU | `context_guard` device |
| `detection` | preset name, `DetectionConfig`, `dict` or `None` | `None` | The detection configuration to start from (below) | the whole config |

### How the detection configuration is chosen

Every time a session detects, it computes its `detection_config` like this:

### 1. The detection setting

`detection` as given: a preset name (`"fast"`, `"balanced"`, `"strict"`, `"classic"`), a `DetectionConfig`, or a dict, whole or partial, merged onto the preset it names (else `balanced`).

### 2. Else the environment

When `detection` is `None`: the preset in `SURROGATESHIELD_PRESET`, with the JSON file in `SURROGATESHIELD_DETECTION_CONFIG` merged on top. Either variable may be set alone.

### 3. Else balanced

`preset("balanced")`, hash `9470f9336d853f67`.

### 4. Flat settings on top

Every flat key in the table that differs from its default is applied over the result. Keys left at their defaults change nothing, so `SURROGATESHIELD_PRESET=strict` keeps ContextGuard on even though `context_guard_enabled` defaults to `False`.

```python
import surrogateshield as shield

shield.config(address_mode="replace")
det = shield.current_session().detection_config
det.preset           # 'balanced'
det.type_actions     # {'ADDRESS': 'replace'}

shield.config(detection="fast")                                   # a preset
shield.config(detection={"type_actions": {"PERSON": "redact"}})   # a partial dict
shield.mask("Priya Raman wrote this.")                            # '[PERSON_1] wrote this.'
shield.config(detection=shield.preset("strict").with_actions(EMAIL="keep"))
shield.config(detection=None)                                     # back to the environment / balanced
```

### Validation

`config()` checks every value before changing anything:

| Call | Error |
|---|---|
| `config(address_mode="nearby")` | `ValueError: address_mode must be one of ('shift', 'replace', 'auto'), got 'nearby'` |
| `config(address_shift_range=True)` | `ValueError: address_shift_range must be an integer >= 1, got True` |
| `config(fuzzy_threshold=101)` | `ValueError: fuzzy_threshold must be a number in [0, 100], got 101` |
| `config(entity_trace_low_threshold=1.5)` | `ValueError: entity_trace_low_threshold must be a number in [0.0, 1.0], got 1.5` |
| `config(context_guard_device="cuda")` | `ValueError: context_guard_device must be an integer (-1 = CPU, >=0 = GPU id), got 'cuda'` |
| `config(pii_off=["phones"])` | `ValueError: Unknown pii_off entry 'phones'. Valid entries: …` |
| `config(pii_mem="/nonexistent")` | `ValueError: pii_mem path does not exist or is not a directory: '/nonexistent'` |
| `config(detection=3)` | `ValueError: detection must be a preset name, a DetectionConfig, a dict or None, got int` |
| `config(detection="nope")` | `ValueError: unknown preset 'nope'; presets: (...), ablations: (...)` |
| `config(pii_mem=...)` while the current session holds mappings | `RuntimeError: pii_mem cannot change while the current session holds mappings; call flush() first or create a new Session` |

`spacy_model` and `context_guard_model` must be non-empty strings. A `DetectionConfig` validates itself when built: an unknown type, stage, action or threshold raises `ValueError`.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SURROGATESHIELD_PRESET` | unset (`balanced`) | The preset used when `detection` is `None` |
| `SURROGATESHIELD_DETECTION_CONFIG` | unset | A JSON file, partial or whole, merged onto that preset |
| `SURROGATESHIELD_HOME` | `~/.surrogateshield` | Folder for `device.key` and the command line's `sessions/` |
| `SURROGATESHIELD_MODELS` | `~/.cache/surrogateshield/models` | Folder that holds `pii-tagger-dv3s-40k` |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | unset | Read by the Hugging Face libraries: `1` loads models from local files only |

Presets, stages, thresholds and type routing in full: [Presets](#presets) and [DetectionConfig](#detection-config).

<a id="library-plugins"></a>

## Detector Plugins

A plugin adds a detector of your own, for values only your organisation knows the shape of: badge numbers, internal project code names, ticket ids. Its detections go through everything a model's do: type routing (`type_sources`), the stage's per-type thresholds, the rule that nothing inside a URL is taken, the [relation gate](#relation-gate) (unless the type bypasses it) and the resolver. Its values get surrogates and are restored like any others.

### The interface

A detector is any object with a `detect(text, view)` method that returns a list of `Candidate`s:

- `text`: the message as typed.
- `view`: the same message with look-alike characters folded and spelled-out digits written as digits, or `None` when nothing needed rewriting. `view.text` is the rewritten text, and `view.original(start, end)` maps a span of it back to offsets in `text`.
- `Candidate(start, end, type, score=1.0)`: offsets in `text`; `type` a public type (`"ID"`) or an internal one (`"ssn"`); `score` compared with the stage's threshold for that type. A candidate with offsets outside the text is dropped with a warning in the log, and one that covers only whitespace is dropped.

You register a factory, not a detector. The factory is called with the stage's settings (a `Stage`: `name`, `enabled`, `model`, `revision`, `device`, `thresholds`, `window`, `stride`, `max_latency_ms`, `options`) and returns the detector. One detector is built per distinct stage setting and reused.

### Example: badge numbers

```python
import re
import surrogateshield as shield
from surrogateshield import Candidate

class EmployeeIds:
    PATTERN = re.compile(r"\bEMP-\d{6}\b")

    def __init__(self, stage):
        self.score = stage.options.get("score", 0.99)

    def detect(self, text, view):
        found = [Candidate(m.start(), m.end(), "ID", self.score) for m in self.PATTERN.finditer(text)]
        if view is not None:                     # also catch full-width or spelled-out digits
            for m in self.PATTERN.finditer(view.text):
                start, end = view.original(m.start(), m.end())
                found.append(Candidate(start, end, "ID", self.score))
        return found

shield.register_detector("employee_ids", EmployeeIds)
det = shield.preset("balanced").with_plugin("employee_ids", types=["ID"], options={"score": 0.98})
s = shield.Session(seed=4, config=shield.Config(detection=det))

s.mask_result("Badge EMP-004417 was lost.")
# text: 'Badge DZN-721106 was lost.'
# detections: [('EMP-004417', 'id_number', 'employee_ids')]
s.mask_result("Badge EMP-００４４１７ was lost.")
# text: 'Badge TKB-388542 was lost.'    (found through the view)
```

A plugin's `ID` is reported as `id_number`, the internal type its surrogates are drawn as. The surrogate keeps the value's shape: three capitals, a hyphen, six digits.

### Example: code names

```python
class Codenames:
    def __init__(self, stage):
        self.words = set(stage.options.get("words", ()))

    def detect(self, text, view):
        return [Candidate(m.start(), m.end(), "OTHER")
                for m in re.finditer(r"\b\w+\b", text) if m.group() in self.words]

shield.register_detector("codenames", Codenames)
det = shield.preset("balanced").with_plugin("codenames", types=["OTHER"], options={"words": ["Bluefin"]})
s = shield.Session(seed=4, config=shield.Config(detection=det))
sent = s.mask("The Bluefin launch moved to May.")   # 'The Dznrecc launch moved to May.'
s.unmask(sent)                                     # 'The Bluefin launch moved to May.'
```

### Registering

register_detector(name: str, factory, *, replace: bool = False) → None

Makes `factory` the detector of stage `name`.

- Registering the same factory again does nothing. A different factory under a taken name raises `ValueError: a detector is already registered as 'employee_ids' (replace=True to swap it)`.
- The built-in stage names (`pattern_scan`, `canonicaliser`, `entity_trace`, `context_guard`, `structural`) raise `ValueError: 'entity_trace' is a built-in stage`. `pii_tagger` is the exception: a factory registered under that name replaces the built-in tagger.
- A factory that is not callable raises `TypeError`, and a detector without `detect()` raises `TypeError` the first time it is built.
- `surrogateshield.core.detection.plugins.unregister_detector(name)` removes a factory and its built detectors; `plugins.registered()` lists the registered names.

### Switching it on

DetectionConfig.with_plugin(name: str, types=("*",), **stage) → DetectionConfig

Adds the stage `name`, switched on, with any `Stage` field given as a keyword (`options`, `thresholds`, `max_latency_ms`, …), and lets it report `types`, a list of public types. `"*"` means every type. Plugin stages run after the model stages, in the order they were added.

Pass the result as the `detection` setting, or save `det.to_json()` to a file and point `SURROGATESHIELD_DETECTION_CONFIG` or `--detection-config` at it. The config hash changes when a plugin is added: the badge example's is `def63261eb9ad4ed`.

### Packaging a plugin

A package can make its detector available without a `register_detector()` call by declaring an entry point in the `surrogateshield.detectors` group. It is loaded the first time a config lists the stage:

```python
# pyproject.toml of your package
[project.entry-points."surrogateshield.detectors"]
employee_ids = "acme_detectors.badges:EmployeeIds"
```

A registered factory takes precedence over the built-in tagger and over an entry point. A config that lists a stage nobody provides fails closed at the first detection:

```text
DetectorUnavailable: no detector is registered as 'my_missing_stage' (register_detector or an entry point in 'surrogateshield.detectors'); switch the stage off to run without it
```

### Notes

- When a plugin and a built-in stage find the same span, the built-in stage's label is kept: stages missing from `source_priority` rank after all the listed ones. In "Priya Raman's badge is EMP-004417; she sits in the Leeds office." the tagger also found the badge, so the detection reads `source='pii_tagger'`. The value is masked either way.
- A stage that takes longer than its `max_latency_ms` is logged as over budget and its candidates are kept.
- `surrogateshield doctor` adds a required row `detector <name>` for each plugin stage in the config, and fails if the detector cannot be built.

<a id="library-cli"></a>

## Command Line

The package installs the `surrogateshield` command (also `python -m surrogateshield`): "Mask PII before text reaches an LLM."

```bash
surrogateshield {scan,mask,unmask,bench,doctor} ...
```

| Command | What it does |
|---|---|
| `scan` | List the PII found in TEXT |
| `mask` | Print TEXT with PII replaced by surrogates |
| `unmask` | Restore the originals in an LLM answer |
| `bench` | Mask latency on built-in sample messages |
| `doctor` | Check models, dependencies and key storage |

### Flags every command takes

| Flag | Effect |
|---|---|
| `text` | The text (`scan`, `mask`, `unmask`). Omitted or `-`: read from standard input |
| `--no-context-guard` | Skip the distilbert-NER stage |
| `--pii-off [TYPE ...]` | Types to send as is, e.g. `phone email`. Takes several values, so put the text before it or end the list with `--` |
| `--preset {fast,balanced,strict}` | Detection preset. Default: `balanced`, or `$SURROGATESHIELD_PRESET`, which also accepts `classic` |
| `--detection-config FILE` | A DetectionConfig as JSON, partial or whole, merged onto the preset. If both name a preset and they differ: `surrogateshield: --preset fast and the file's preset 'strict' disagree`, exit 1 |

### scan

Flags: `--json`, `--seed SEED`. One line per detection: start, end, type, source and the value, with `(pii_off: sent as is)` after values sent unchanged.

```bash
$ surrogateshield scan "My phone is +44 7700 900123." --pii-off phone
   12-27    phone_uk           pattern  '+44 7700 900123'  (pii_off: sent as is)

$ surrogateshield scan "I'm Priya Raman, call me on +44 7700 900123." --pii-off phone --json
[{"text": "Priya Raman", "type": "PERSON", "start": 4, "end": 15, "score": 1.0, "source": "pii_tagger", "masked": true}, {"text": "+44 7700 900123", "type": "phone_uk", "start": 28, "end": 43, "score": 1.0, "source": "pattern", "masked": false}]
```

Scores in the JSON are rounded to three decimals.

### mask

| Flag | Effect |
|---|---|
| `-s`, `--session SESSION` | Reuse a stored session (default: a new one) |
| `--store STORE` | Directory for encrypted maps (default `~/.surrogateshield/sessions`, i.e. `$SURROGATESHIELD_HOME/sessions`) |
| `--no-store` | Keep the map in memory only; nothing can be unmasked later |
| `--seed SEED` | Seed the surrogates (reproducible output) |
| `--json` | Print `session`, `text`, `service_query`, `replacements` and `detections` as JSON |

Without `--session`, a new session is stored under a random id, and the id goes to standard error as `session: <id>`, so the masked text alone goes to standard output.

```bash
$ surrogateshield mask -s conv-7 --seed 4 "I'm Priya Raman, call me on +44 7700 900123."
I'm Dalaja Rastogi, call me on +44 7721 106840.

$ surrogateshield mask --seed 4 --json "I'm Priya Raman."
session: 0007282ae20a4b31b1fb1e8543d71ada
{"session": "0007282ae20a4b31b1fb1e8543d71ada", "text": "I'm Dalaja Rastogi.", "service_query": false, "replacements": {"Priya Raman": "Dalaja Rastogi"}, "detections": [{"text": "Priya Raman", "type": "PERSON", "start": 4, "end": 15, "score": 1.0, "source": "pii_tagger", "masked": true}]}
```

### unmask

Flags: `-s`/`--session SESSION` (required), `--store STORE`, and `--close` to erase the stored map afterwards.

```bash
$ surrogateshield unmask -s conv-7 "Thanks Dalaja Rastogi, I'll ring +44 7721 106840."
Thanks Priya Raman, I'll ring +44 7700 900123.

$ surrogateshield unmask -s conv-7 --close "Bye Dalaja."
Bye Priya.

$ surrogateshield unmask -s conv-7 "Hello again."
surrogateshield: no stored session 'conv-7' under ~/.surrogateshield/sessions      (exit 1)
```

A shell pipeline around any model command:

```bash
surrogateshield mask -s conv-7 < question.txt | my-llm-cli | surrogateshield unmask -s conv-7
```

### bench

Flags: `--rounds ROUNDS` (default 3) and `--json`. It masks 8 built-in sample messages once to warm up, then times the given number of rounds and prints the median and 95th percentile per configuration; the JSON adds `max_ms`. Without `--preset` or `--detection-config` it measures the default configuration with and without ContextGuard (`--no-context-guard` skips the second). With one of them, it measures only that configuration.

```bash
$ surrogateshield bench --rounds 1 --no-context-guard
without ContextGuard   p50    39.1 ms   p95    43.6 ms   (8 masks)
```

### doctor

Prints the effective detection configuration and where it came from (`--preset / --detection-config`, `the environment`, `settings` or `default`), then one row per check: the Python version, each dependency, each model an enabled stage loads (installed or cached at its pinned revision, with version and licence), each plugin stage, the storage folder and the device key. Rows for what the configuration needs are required: `FAIL` if missing. The others read `warn`. The output of a healthy install is under [Library Installation](#library-install).

| Flag | Effect |
|---|---|
| `--smoke` | Also mask a sample message with an email and an SSN, and check both are gone |
| `--cold-start` | Also load each model and time it |
| `--show-config` | Print the whole detection config as JSON |

`--smoke` and `--cold-start` run only when every required row passed. An invalid detection config prints `[FAIL] detection config — …` and exits 1.

```bash
$ SURROGATESHIELD_PRESET=classic surrogateshield doctor
detection config: preset classic, hash 6e5419989f2b9a70 (the environment)
...
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | A required `doctor` check failed; no stored session under that id; a storage error; an invalid setting, session id or detection config file |
| `2` | Usage error, e.g. `--preset classic`: `invalid choice: 'classic' (choose from fast, balanced, strict)` |
| `3` | A detector was unavailable: `surrogateshield: detector unavailable, nothing was masked: …` |

<a id="library-errors"></a>

## Errors

| Exception | Import from | Raised when | What to do |
|---|---|---|---|
| `DetectorUnavailable` (a `RuntimeError`) | `surrogateshield` | A stage that is switched on cannot run: a model or package is missing, the tagger's weights do not match their pin, a plugin stage has no detector, or a model failed on this input. Raised by `scan`, `mask` and `mask_result`; the text is not masked | Do not send the message. Install what the message names, or switch the stage off in the configuration |
| `StorageError` (a `RuntimeError`) | `surrogateshield` | Base class of the two below | Catch it to handle both |
| `ShadowMapStorageError` | `surrogateshield.core.storage.shadow_map` | The device key cannot be read or written, or has the wrong length | Fix the permissions or the disk under `$SURROGATESHIELD_HOME` |
| `CorruptStoreError` | `surrogateshield.core.storage.shadow_map` | A stored map cannot be decrypted; it has been moved to `<id>.shadowmap.corrupt-<timestamp>` | Start the conversation with a new session; the old replies cannot be restored with this key |
| `OSError` | built in | Writing a persistent map failed during `mask()` | Do not send the message; a reply to it could not be restored |
| `ValueError` | built in | An invalid setting, session id, preset name, detection config or plugin registration | The message names the value and what is allowed |
| `TypeError` | built in | Text that is not a `str` (`text must be a str, got int`); `unmask(None)` or a response without text; `use_session()` without a `Session`; a plugin factory or detector of the wrong kind | Pass the right type |
| `RuntimeError` | built in | `session <id> is closed`; `pii_mem` changed while the current session holds mappings | Create a new session; call `flush()` before changing `pii_mem` |
| `DeprecationWarning` | built in | `config(verify_addresses=...)` | Remove the argument |

The `DetectorUnavailable` message says what to do. Without the tagger's weights:

```text
DetectorUnavailable: PIITagger: no model folder 'pii-tagger-dv3s-40k' under ~/.cache/surrogateshield/models ($SURROGATESHIELD_MODELS). Copy the trained tagger there (python -m bench.tagger.install), point $SURROGATESHIELD_MODELS at the folder that holds it, or use the 'classic' detection preset (spaCy and ContextGuard read names; no tagger).
```

```python
import surrogateshield as shield

def send_safely(session, text):
    try:
        sent = session.mask(text)
    except shield.DetectorUnavailable as exc:
        raise RuntimeError(f"not sent: PII detection unavailable ({exc})") from exc
    except (shield.StorageError, OSError) as exc:
        raise RuntimeError(f"not sent: the surrogate map could not be stored ({exc})") from exc
    return ask_my_model(sent)
```

<a id="library-fastapi"></a>

## FastAPI Example

A chat endpoint that masks each user message, sends the conversation to Claude and returns the unmasked reply. Each conversation has one `Session`, stored encrypted, so a restart does not lose its map. It uses the async methods, so detection runs off the event loop.

```python
import asyncio
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import surrogateshield as shield

STORE = os.environ.get("SHADOWMAP_DIR", os.path.expanduser("~/.myapp/shadowmaps"))
MODEL = "claude-opus-5-5"

app = FastAPI()
client = anthropic.AsyncAnthropic()            # reads ANTHROPIC_API_KEY

@dataclass
class Conversation:
    session: shield.Session
    history: List[dict] = field(default_factory=list)    # what the model has seen
    turn: asyncio.Lock = field(default_factory=asyncio.Lock)

_conversations: Dict[str, Conversation] = {}
_registry = threading.Lock()

def open_conversation(cid: str) -> Conversation:
    with _registry:
        conv = _conversations.get(cid)
        if conv is None:
            # the same id reopens the encrypted map written before a restart
            conv = Conversation(shield.Session(cid, storage_dir=STORE))
            _conversations[cid] = conv
        return conv

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str
    masked: List[str]           # the types replaced in this message

async def ask_model(history: List[dict]):
    return await client.messages.create(model=MODEL, max_tokens=1024, messages=history)

@app.post("/conversations/{cid}/messages", response_model=ChatOut)
async def chat(cid: str, body: ChatIn) -> ChatOut:
    try:
        conv = open_conversation(cid)
    except ValueError as exc:                 # id must match ^[A-Za-z0-9_-]{1,64}$
        raise HTTPException(422, str(exc))
    except shield.StorageError as exc:        # device key unreadable, or the map is corrupt
        raise HTTPException(500, str(exc))

    async with conv.turn:                     # one turn at a time per conversation
        try:
            result = await conv.session.amask_result(body.message)
        except shield.DetectorUnavailable as exc:
            # detection failed closed: nothing was masked, so nothing is sent
            raise HTTPException(503, f"PII detection unavailable: {exc}")

        conv.history.append({"role": "user", "content": result.text})
        response = await ask_model(conv.history)
        answer = "".join(b.text for b in response.content if b.type == "text")
        conv.history.append({"role": "assistant", "content": answer})   # kept as sent

        reply = await conv.session.aunmask(response)

    return ChatOut(reply=reply, masked=sorted({d.type for d in result.detections if d.masked}))

@app.delete("/conversations/{cid}", status_code=204)
def end_conversation(cid: str) -> None:
    with _registry:
        conv = _conversations.pop(cid, None)
    try:
        session = conv.session if conv else shield.Session(cid, storage_dir=STORE)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    session.close()                           # erases the map and its file
```

```bash
pip install fastapi uvicorn anthropic
uvicorn app:app
```

### How it fits together

- **One session per conversation id.** The registry keeps one `Session` object per conversation in the process, so the low-entropy restore sees the message each reply answers. The id doubles as the map's file name, which is why it must match `^[A-Za-z0-9_-]{1,64}$`; anything else becomes a 422.
- **One turn at a time.** The per-conversation `asyncio.Lock` keeps the history in order and pairs each reply with its message. Different conversations run in parallel.
- **Fail closed.** If detection cannot run, the endpoint answers 503 and the provider is never called.
- **Masked history.** The history holds the masked user turns and the model's replies as written. The map survives a restart. The history here lives in memory; store it yourself (it holds surrogates only) if conversations must outlive the process.
- **Erasure.** `DELETE /conversations/{id}` closes the session, which deletes its map file, including a map left by an earlier process.
- **Several worker processes.** Each process opens the same file for a conversation, but each holds its own copy of the map in memory. Route a conversation's requests to one worker (sticky sessions), or run a single worker per conversation.

### Testing it without calling the provider

Replace `ask_model` with a stub that echoes what it received:

```python
from anthropic.types import Message, TextBlock, Usage
from fastapi.testclient import TestClient
import app as appmod

async def fake_ask(history):
    return Message(id="msg_1", type="message", role="assistant", model=appmod.MODEL,
                   content=[TextBlock(type="text", text="Noted: " + history[-1]["content"])],
                   stop_reason="end_turn", stop_sequence=None,
                   usage=Usage(input_tokens=1, output_tokens=1))

appmod.ask_model = fake_ask
client = TestClient(appmod.app)
r = client.post("/conversations/conv-1/messages",
                json={"message": "Hi, I'm Priya Raman, my email is priya.raman@northwind-traders.co.uk."})
print(r.json())
```

Real run:

| Step | The stub received | The client got |
|---|---|---|
| POST, turn 1 | Hi, I'm Vansha Kar, my email is vansha.kar@garcia-avery.co.uk. | `{"reply": "Noted: Hi, I'm Priya Raman, my email is priya.raman@northwind-traders.co.uk.", "masked": ["PERSON", "email"]}` |
| POST, turn 2 | Please write to Vansha Kar again. | `{"reply": "Noted: Please write to Priya Raman again.", "masked": ["PERSON"]}` |
| registry cleared (a restart), POST | Vansha Kar here again. | `{"reply": "Noted: Priya Raman here again.", "masked": ["PERSON"]}` |
| POST to `/conversations/bad id!/messages` | — | 422 `invalid id 'bad id!': must match ^[A-Za-z0-9_-]{1,64}$` |
| DELETE `/conversations/conv-1` | — | 204, and `conv-1.shadowmap` is gone |

---

<a id="security-design"></a>

## Security Design

SurrogateShield keeps two kinds of sensitive files on your disk: the **surrogate maps** that turn stand-ins back into your real values, and the **chat transcripts**. Both are sealed with AES-256-GCM under keys derived from one device secret. The app and the library share one implementation of this, `python-library/surrogateshield/core/storage/shadow_map.py`; the app's `storage/logic.py` only binds it to `config.py`.

### Key hierarchy

```python
<home>/device.key                      32 random bytes, mode 0600, created once
      │
      └── HKDF-SHA256(ikm  = device secret,
                      salt = store id,          e.g. the conversation id
                      info = kind)              b"shadowmap-v1" or b"transcript-v1"
            │
            ├── key for <id>.shadowmap         {surrogate: original}
            └── key for <id>.json              the chat transcript
```

`<home>` is `$SURROGATESHIELD_HOME`, or `~/.surrogateshield` when the variable is unset. Every file gets its own 32-byte key, so two conversations never share one, and a map and a transcript with the same id do not either.

### Sealed file format (v1)

```python
b"SSv1" || nonce (12 bytes) || AES-256-GCM(key, plaintext, aad)

aad = kind || b"|" || id
```

The nonce is 12 fresh random bytes on every write. The associated data binds the file to the kind and the id it was written for: a map renamed to another conversation's id, or a transcript presented as a map, fails authentication instead of opening.

### Mechanisms

| Component | Mechanism |
|---|---|
| Home folder | `$SURROGATESHIELD_HOME` or `~/.surrogateshield`. Created with mode `0700`; a folder found with a looser mode is tightened to `0700`. Every private sub-folder (`conversations/`, `rag/`, `sessions/`) goes through the same check. |
| Device secret | `device.key`: 32 bytes from `os.urandom`, mode `0600`. It is written to a temporary file, synced, then linked into place with `os.link`, which fails if the file already exists, so two processes starting together agree on one secret. Override the path with `config.DEVICE_KEY_PATH`. |
| Loose key permissions | Tightened to `0600` with the warning `[ShadowMap] <path> had loose permissions; fixed to 0600`. |
| Damaged key | A key file that is not exactly 32 bytes raises `ShadowMapStorageError`: `<path> is corrupt (<n> bytes, expected 32); refusing to derive keys from it`. Keys are not derived from a truncated secret. |
| Unreadable or unwritable key | Raises `ShadowMapStorageError`: `cannot read or persist the device secret at <path>: <error>`. Only if `config.ALLOW_EPHEMERAL_KEY` is `True` (default `False`) does the process use a random secret for its own lifetime, logging `EPHEMERAL device secret in use (...); nothing stored by this process will be readable after it exits`. |
| Key derivation | HKDF-SHA256, 32-byte output, salt = the store id, info = the kind (`shadowmap-v1` or `transcript-v1`). |
| Encryption | AES-256-GCM (`cryptography`'s `AESGCM`), fresh 12-byte nonce per write, associated data `kind\|id`. |
| Store ids | Must match `^[A-Za-z0-9_-]{1,64}$`, so an id cannot contain a path separator or `..`. |
| Atomic writes | Every private file is written to a temporary file in the same folder, set to `0600` before any byte is written, flushed and `fsync`ed, then moved over the old file with `os.replace`. A crash leaves the old file or the new one, never half of one. |
| A map that does not decrypt | Moved aside to `<name>.corrupt-<unix time>` (never deleted) and `CorruptStoreError` is raised: `shadow map <name> could not be decrypted (<reason>); moved to <new name>`. It never turns silently into an empty map, which would leave later replies unrestored. |
| A transcript that does not decrypt | `ValueError`: `conversation <id> cannot be decrypted with this device key (<reason>)`. The app prints `Cannot open conversation:` and stops. |
| Older files | Pre-v1 maps (`nonce \|\| ciphertext`, HKDF info `shadowmap`, no associated data) and pre-v1 transcripts are still read and are rewritten in the v1 format on the next save; an old per-conversation `.key` file is deleted at that save. A plaintext transcript from an earlier version (first byte `{`) is read with a warning and encrypted on the next save. A transcript without `api_messages` starts a fresh provider context instead of re-sending the display history, which holds real values. |
| Settings | `<home>/settings.json`, mode `0600`, written atomically. A file that is not valid JSON is moved to `settings.json.corrupt-<unix time>` and the defaults are used. It holds no secrets: the provider name, Detailed View and Presidio comparison. |
| Deleting | Deleting a conversation from the list removes its transcript and its map without decrypting them, so it also works on files this key cannot open. The library's `flush()` and `Session.close()` delete the session's persistent map. Files are removed with a normal unlink. |
| Detection fails closed | If a detection stage that is switched on cannot run (missing package, missing model, failed load), `DetectorUnavailable` is raised. The text is not masked and is not sent; the app shows a red **Detector unavailable** panel ending in *Nothing was masked and nothing was sent.* See [Fail-Closed Detection](#fail-closed). |
| Send check (JSON Test) | Before each request the batch runner rebuilds the expected text from the original question and the planned edits. If the text about to be sent differs, or a surrogate is not at its planned offset, it raises `SendMismatch` and stops the run. |
| Provider clients | Built with a 60-second timeout and the SDK's own retries off. `chatbot/providers.py` makes at most 3 attempts, only for 408, 409, 429, 5xx and connection or timeout errors, and honours `Retry-After` up to 60 seconds. A rejected key is not retried. A failed call records nothing in either history. |
| API keys | Read from the environment, or from `.env` in the project folder (loaded with python-dotenv). `run.sh` warns `.env is readable by other users (mode <mode>). Run: chmod 600 .env`. |
| RAG index telemetry | ChromaDB is opened with `anonymized_telemetry=False`. |
| Repository | `.gitignore` excludes `.env` and `.env.*`, `*.key`, `*.pem`, `*.p12`, `conversations/`, `.surrogateshield/`, `chroma/`, `chroma_db/`, `*.chroma`, the batch outputs `experiment/*_answers*.json`, `experiment/*_eval_results*.json` and `experiment/*_Attacker_Experiment*.json`, and the benchmark's raw downloads and rebuilt text (`bench/realdata/raw/`, `bench/realdata/build/`, `experiment/realdata/`). `tests/test_gitignore.py` checks these entries. |

> **Keep `device.key` with your data.** A sealed file opens only with the device secret it was written under. Copy `device.key` along with the rest of `<home>` when you move to another machine or restore a backup; without it the maps and transcripts cannot be opened. Anyone who can read both `device.key` and a sealed file can open that file, so the folder's `0700` and the key's `0600` modes matter.

<a id="privacy-guarantees"></a>

## What Is Sent

This section lists, for every mode, the text that leaves your machine and the text that stays on it. "The provider" is the one chosen in Settings: Claude (`ANTHROPIC_API_KEY`), Gemini (`GEMINI_API_KEY`), ChatGPT (`OPENAI_API_KEY`), or a local Ollama server at `LOCAL_LLM_HOST` (default `http://localhost:11434`). Detected values are replaced before the request is built. Values no stage detected, and the values listed under [Values sent as typed](#privacy-guarantees) below, go out as you wrote them.

### Per mode

| Mode | Sent to the provider | Stays on your machine |
|---|---|---|
| **Chat** (`N`, `python main.py chat`) | One request per turn: the system prompt (below), the conversation's `api_messages` (every earlier sanitised message and every earlier reply as the model wrote it, with surrogates), and the new sanitised message. `max_tokens` 4096 for Claude, Gemini and ChatGPT. | What you typed, the detection results, the surrogate map (sealed `<id>.shadowmap`), the restored reply, and the transcript (sealed `<id>.json`, written after each answered turn). |
| **Chat with RAG** (`R`, `chat --rag`) | As Chat, plus the 3 index chunks closest to your sanitised message, placed before it in this request only (`Use the following retrieved context`, `[Document excerpt 1]`, ...). The chunks hold the surrogates the document was indexed with. They are not stored in `api_messages`, so later turns do not re-send them. | The index query itself: retrieval runs in the local ChromaDB index, embedded with `all-MiniLM-L6-v2`. |
| **Detailed View** (setting, on by default) | Nothing extra. | Prints on your screen the detection table with each original and its surrogate, the quasi-identifier line, and the API Transparency panel ("Sent to", "Received from", "Final output", each cut to 300 characters). It also raises the app's own log level to INFO. |
| **PII Finder** (`P`, `pii-finder`) | Nothing. | Runs detection on the text you paste. With Presidio comparison on, Presidio also runs locally beside it. |
| **Indexing a document** (`add-doc <file>`) | Nothing. | Detection runs locally in segments of at most 20,000 characters (`DOC_SEGMENT_CHARS`); if any segment fails, nothing is stored. Addresses use `replace` when the address mode is `auto`. The pairs go to the sealed `rag_global.shadowmap`; the sanitised chunks (512 characters each), their embeddings and the file name go to the index. |
| **Listing or forgetting documents** (`rag`, `rag --forget DOC`) | Nothing. | `--forget` removes the document's chunks and erases every pair in `rag_global` that no remaining chunk contains. |
| **JSON Test** (batch questions) | Per question, one request with the system prompt and that question's `sanitized_input` alone (no history). Two optional fields add one request each: `presidio_llm_response` sends the Presidio-redacted text (needs Presidio comparison on), and `clean_llm_response` sends the **original** question. Both are off by default; the field list labels the second "synthetic data only, extra API call". | `experiment/<name>_answers.json` and `<name>_answers.meta.json`, mode `0600`, written atomically. They hold the original questions, the maps and the responses in clear text, which is why they are git-ignored. BERTScore (`roberta-large`) runs locally. |
| **Evaluation** | Nothing. It makes no LLM call. | Reads an answers file and a key file from `experiment/`; `S` saves `experiment/<answers>_eval_results.json`. |
| **`offline_eval.py`** | Nothing; no API key needed. | Runs the batch runner's send path (detect, surrogates, substitute) and scores it. |
| **Attacker Experiment** | Anthropic only, to `ATTACKER_MODEL`, which must differ from `CLAUDE_MODEL`. For each question it sends the text each tool sent (`sanitized_input`, and `presidio_sanitized_input` when present) with the attacker prompt. `--dry-run` prints the number of calls and sends nothing. | `experiment/<stem>_Attacker_Experiment.json`, mode `0600`, saved every 5 questions. |
| **Library** `mask` / `unmask` / `scan` | Nothing. The library makes no provider call: `mask` returns text, and your code decides where to send it. | The map, in memory by default (`pii_mem="temp"`). With `pii_mem` set to a folder it is sealed to `<folder>/<session id>.shadowmap` after every change. `detailed_view=True` prints original values to stdout; it is off by default. |
| **Library CLI** (`surrogateshield mask`, `unmask`, `scan`, `bench`, `doctor`) | Nothing. | `mask` prints the masked text on stdout and the session id on stderr, and seals the map to `<home>/sessions/` (`--store DIR` elsewhere, `--no-store` memory only). `unmask -s ID --close` erases the stored map afterwards. |
| **Model downloads** | Not your text. Model files are fetched from Hugging Face when a stage needs one that is not cached: ContextGuard (`dslim/distilbert-NER`, only when switched on), `all-MiniLM-L6-v2` (RAG) and `roberta-large` (BERTScore). With `HF_HUB_OFFLINE=1` nothing is fetched; a stage whose model is missing then fails closed. A cached model is loaded without a network request. | The spaCy model and the PII tagger's weights are local files you install once. |

### The chat request

```python
system   = SYSTEM_PROMPT
messages = api_messages                                # earlier turns, surrogates only
         + [{"role": "user",
             "content": rag_context + sanitised_message}]   # rag_context: --rag only, this turn only
```

The system prompt, word for word: *"You are a helpful assistant. The names, email addresses, phone numbers, and other personal details in this conversation are anonymised aliases — placeholders for the user's real information. Treat every alias as you would treat real information: answer helpfully and naturally. Do not comment on the aliases, do not flag them as unusual, and do not issue security warnings about them. The user's privacy is already protected by the system they are using."*

A conversation keeps two histories. `messages` is what you see: your text as typed and the reply with your values restored. `api_messages` is what is re-sent: the sanitised text and the reply as the model wrote it. Only `api_messages` ever goes to the provider.

### Values sent as typed

These values are found and left in place on purpose, because replacing them would change the question without hiding anyone. Each has a named reason in the code; the evaluator counts a value as deliberate only under one of them.

| What goes out unchanged | When | Reason name |
|---|---|---|
| A city or state on its own | Service query ("coffee near Tempe") | `service_query_location_suppressed` |
| Street, city, state and ZIP of an address; the house number moves by up to ±1 (`ADDRESS_SHIFT_RANGE`) | Service query, address mode `shift` (what `auto` picks) | `service_query_address_shift` |
| City and state; the street line becomes "my area" and the ZIP code is removed | Service query on a sensitive topic (HIV, rehab, shelter, immigration and the other topic words), mode `coarse` | `service_query_address_coarse` |
| A place that is only the topic of a question ("Japan's GDP") | The message holds no direct identifier of a person | `topical_geo_filtered` |
| An organisation, place or name that nothing in the message ties to a person (acronyms, code, public figures, companies) | Relation gate; public figures come from `public_people.txt` | `not_tied_to_person` |
| A gender word of the same gender as a named person in the message | The person's surrogate keeps that gender | `gender_follows_name` |
| A phone number's country calling code | Kept inside the phone surrogate | `phone_country_code_kept` |

Three more cases depend on you:

- In chat, a borderline entity asks `Replace this? [y/N]`. The default, `N`, sends it as typed.
- A type listed in `pii_off`, or given the action `keep` in [type_actions](#type-actions), is detected and sent as typed.
- A type given the action `redact` is sent as a numbered placeholder of its type (`[PERSON_1]`) instead of a surrogate.

Some surrogates also keep part of the original by design, so that the answer still fits: a free-mail e-mail domain, a town replaced by another real town of the same country, a date of birth moved by at most two years, an organisation's legal and industry words ("Pharmacy"), a phone's country code and trunk digit, and a URL's host and path layout. The [attacker test](#results-attacker) reports guesses of these parts separately.

### Where things are stored

| Path | What it holds | Protection |
|---|---|---|
| `~/.surrogateshield/` (`$SURROGATESHIELD_HOME`) | Everything below except `.env` and `experiment/` | Folder `0700` |
| `device.key` | The 32-byte device secret | `0600`, not encrypted (it is the key) |
| `conversations/<id>.shadowmap` | The conversation's map, surrogate → original | AES-256-GCM, kind `shadowmap-v1`, `0600` |
| `conversations/<id>.json` | The transcript: `messages` (your text and the restored replies, so real values), `api_messages` (sanitised), `id`, `created`, `rag_mode` | AES-256-GCM, kind `transcript-v1`, `0600` |
| `conversations/rag_global.shadowmap` | One map for every indexed document | AES-256-GCM, `0600` |
| `rag/` | ChromaDB collection `surrogateshield_rag` (cosine): sanitised chunks, their embeddings, and metadata `doc_id`, `source` (the file name) and `chunk_index` | Folder `0700`; the chunks are not encrypted and hold surrogates |
| `settings.json` | Provider, Detailed View, Presidio comparison | `0600`, plain JSON |
| `sessions/<id>.shadowmap` | Maps stored by the `surrogateshield` CLI | AES-256-GCM, `0600` |
| Your `pii_mem` folder | Library session maps, when `pii_mem` is a folder | AES-256-GCM, `0600` |
| `.env` (project folder) | API keys | Plain text; set it to `0600` (`run.sh` warns otherwise); git-ignored |
| `experiment/*_answers.json`, `*_Attacker_Experiment.json` | Batch questions, maps and responses in clear text | `0600`; git-ignored |
| `experiment/*_eval_results.json` | Evaluation metrics | Written with your default file mode; git-ignored |
| `~/.cache/surrogateshield/models/` (`$SURROGATESHIELD_MODELS`) | The PII tagger's weights | Model files; the SHA-256 of the weights is checked against the pin at load |

The folder paths can be changed in `config.py`: `SHADOWMAP_DIR` for `conversations/`, `RAG_DIR` for `rag/` and `DEVICE_KEY_PATH` for the key. A `./chroma_db` folder from before v2 is no longer read; the app warns `[RAG] ./chroma_db (pre-v2 location) is no longer used; the index now lives in <dir>. Re-run add-doc for those documents.`

<a id="project-structure"></a>

## Project Structure

The tracked files of the repository, grouped by folder. The package version is `2.1.0` (`python-library/pyproject.toml` and `surrogateshield.__version__`; a test keeps the two equal).

```text
SurrogateShield/
├── main.py                  # Terminal app (Typer): dashboard, chat, list, pii-finder, add-doc, rag
├── pipeline.py              # One chat turn: detect, confirm borderlines, surrogates, send, restore; RAG indexing
├── config.py                # App constants: model ids, thresholds, paths, address mode, display switches
├── settings_manager.py      # <home>/settings.json: provider, Detailed View, Presidio comparison
├── util.py                  # Shared helpers: DetectedEntity, Conversation, splice, the borderline prompt
├── help_screen.py           # The in-app help screen
├── json_tester.py           # JSON Test: batch questions through the send path, answers file
├── evaluator.py             # Evaluation of an answers file against a key file (no LLM calls)
├── eval_metrics.py          # The single definition of every detection and protection metric
├── eval_report.py           # Renders an evaluation result with Rich
├── offline_eval.py          # Detection and protection evaluation with no LLM calls
├── attacker.py              # Attacker Experiment (Anthropic only)
├── run.sh                   # Launcher: picks .venv, checks .env's mode and the spaCy model
├── requirements.txt         # App, benchmark and test dependencies; installs ./python-library in editable mode
├── pytest.ini               # Default run skips tests marked heavy
├── conftest.py              # Gives every test a private SURROGATESHIELD_HOME
├── README.md  CONFIGURATION.md  CHANGELOG.md  CONTRIBUTING.md  SECURITY.md  CODE_OF_CONDUCT.md  LICENSE
│
├── python-library/          # The surrogateshield package
│   ├── pyproject.toml       # Package metadata, version 2.1.0, the surrogateshield console script
│   ├── README.md
│   ├── surrogateshield/
│   │   ├── __init__.py      # Public API: config, scan, mask, unmask, flush, Session, presets, plugins
│   │   ├── session.py       # Session: one conversation, one ShadowMap
│   │   ├── cli.py           # surrogateshield scan | mask | unmask | bench | doctor
│   │   ├── _state.py  _display.py  _response_parser.py  __main__.py  py.typed
│   │   └── core/
│   │       ├── detection/   # pipeline.py (SentinelLayer), pattern_scan.py, canonical.py, pii_tagger.py,
│   │       │                # entity_trace.py, context_guard.py, relation_gate.py, resolver.py,
│   │       │                # structural.py, address_parser.py, address_assembly.py, org_assembly.py,
│   │       │                # service_query.py, quasi_identifier.py, plugins.py, config.py (presets),
│   │       │                # geo_data.py, public_names.py, public_people.txt
│   │       ├── generation/  # mimic.py (MimicGen), identity.py, places.py
│   │       ├── reconstruction/resolve.py      # ResolvePass
│   │       ├── storage/shadow_map.py          # ShadowMap, device secret, sealed files
│   │       └── consistency.py  entities.py  errors.py
│   ├── examples/fastapi_app.py
│   ├── tools/build_public_people.py           # Rebuilds public_people.txt
│   └── tests/               # conftest.py, test_public_api.py
│
├── detection/               # App bindings of the package's detection modules to config.py
├── generation/logic.py      # App binding of MimicGen
├── storage/logic.py         # App ShadowMap: conversations_dir(), explicit saves
├── reconstruction/logic.py  # App binding of ResolvePass
├── chatbot/
│   ├── chat.py              # Conversation with two histories, sealed transcripts
│   ├── providers.py         # Claude, Gemini, ChatGPT, Ollama: clients, retries, errors
│   └── rag.py               # RAGStore: ChromaDB index, all-MiniLM-L6-v2 embeddings
├── presidio/                # Presidio comparison: engine.py, detect.py, redact.py
├── examples/
│   └── employee_id_detector.py   # A detector plugin
├── experiment/              # Sample question and key files, the generated synth_* set,
│                            # make_dataset.py, lint_key.py, README.md
├── bench/
│   ├── realdata/            # Real-data benchmark: pull, label, inject, build, score, live,
│   │                        # attacker_exact, external, components, operating_curve, verdict;
│   │                        # per-source and per-split folders, configs/, external/
│   ├── arms/                # One adapter per compared tool (ss, presidio_*, gliner_pii*, llm_guard),
│   │                        # run.py, requirements-baselines.txt
│   ├── tagger/              # PIITagger: data, pools, splits, train, evaluate, select, accept, install
│   ├── realworld/           # Robustness messages (dev*.jsonl, test.jsonl) and GUIDE.md
│   ├── results/             # Every committed result (JSON and Markdown); README.md gives each command
│   └── compare.py  echo.py  perf.py  perf_arms.py  realworld.py  phase8.py
├── tests/                   # 119 test_*.py modules, six numbered legacy check scripts,
│                            # conftest.py, data/address_corpus.json
├── website/                 # index.html, docs.html
└── .github/                 # Issue and pull request templates; workflows tests.yml,
                             # deploy.yml (Pages), deploy-lib.yml, publish-pypi.yml
```

### Where to look

| To change or read | Open |
|---|---|
| What is detected, and in which order | `python-library/surrogateshield/core/detection/pipeline.py`; the presets in `core/detection/config.py` |
| The regex rules | `core/detection/pattern_scan.py` |
| How a surrogate is built | `core/generation/mimic.py`, `identity.py`, `places.py` |
| How replies are restored | `core/reconstruction/resolve.py` |
| Encryption and file handling | `core/storage/shadow_map.py` |
| The app's thresholds, models and paths | `config.py` |
| Provider calls | `chatbot/providers.py` |
| The terminal screens and commands | `main.py` |
| A benchmark number | `bench/results/` and `bench/results/README.md` |

The folders `detection/`, `generation/`, `storage/` and `reconstruction/` at the top level hold no second copy of the engine: each module reads its values from `config.py` and passes everything else through to the package, so the app and the library run the same code.

<a id="running-tests"></a>

## Running Tests

The tests need no API key and make no provider call. Run them through pytest from the repository root.

### 1. Install the dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` includes `pytest>=8` and `pytest-cov>=5`, and installs `./python-library` in editable mode.

### 2. Run the fast suite

```bash
.venv/bin/python -m pytest tests python-library/tests -q
```

`pytest.ini` adds `-m "not heavy"`, so this run skips the tests that load spaCy or Hugging Face models.

### 3. Run the heavy tests

```bash
.venv/bin/python -m pytest tests python-library/tests -m heavy -q
```

These load the real models: the spaCy model `en_core_web_lg`, the PII tagger's weights and the Hugging Face models the presets use. Install them first (see [Installation](#installation)).

### Other useful runs

```bash
# one file, or one test by name
.venv/bin/python -m pytest tests/test_shadow_map.py -q
.venv/bin/python -m pytest tests -k service_query -q

# the CI command: offline, with the coverage gate on the core package
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests python-library/tests -q -p no:warnings \
  --cov=python-library/surrogateshield/core --cov-fail-under=85
```

`.github/workflows/tests.yml` runs that last command on Python 3.9 and 3.13, with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `ANONYMIZED_TELEMETRY=False`.

### Isolation

- The root `conftest.py` points `SURROGATESHIELD_HOME` at a new temporary folder before any test runs, and removes it at exit. No test reads or writes your real `~/.surrogateshield`: not the device key, not the conversations, not the settings. Subprocesses started by tests inherit it.
- After every test it restores the library's default configuration and ends the current session.
- `tests/conftest.py` puts the repository root and `python-library/` on `sys.path`, and provides two fixtures: `stub_tagger` (the `pii_tagger` stage with no model, so model-free tests can run the default preset) and `corpus` (the frozen address corpus in `tests/data/address_corpus.json`).

### How the test files are organised

| Files | What they cover |
|---|---|
| `test_I*`, `test_J*`, `test_F*`, `test_E2_*`, `test_C3_*`, `test_B_*`, `test_N7_*` | Named after the audit item or release gate they lock down. Examples: `test_I6_service_query` (service queries), `test_F2_I20_I21_providers` (provider errors and retries), `test_F3_settings`, `test_J13_cli` (the `surrogateshield` CLI), `test_J15_perf`, `test_J7_async` and `test_J7_fastapi_example`, `test_N7_no_network` (masking makes no network request once models are cached). |
| `test_J16_docs` | The docs: both READMEs and the website carry the one-line Limitations, every result file under `bench/results/` is listed in its README with its command, the published numbers match the result files, and no claim the results do not support appears in the READMEs, the website pages, `main.py` or `help_screen.py`. |
| `test_V3_*`, `test_V4_*` | The current detector: the PII tagger and its selection, pools, the frozen config, external data, foreign-language fragments, age patterns and age surrogates. |
| `test_realdata_*`, `test_realworld_*`, `test_arms`, `test_perf_arms`, `test_gliner_sweep`, `test_regex_ablation` | The benchmark code: pulling, labelling, injecting, scoring, the tool adapters and the live-run helpers, all without provider calls. |
| `test_address_*`, `test_pattern_scan`, `test_canonical`, `test_cue_rules`, `test_org_assembly`, `test_person_completion`, `test_name_tokens_unicode`, `test_resolver`, `test_detection_config`, `test_cascade_trace`, `test_service_query` | Detection. |
| `test_generation`, `test_consistency`, `test_substitution_plan`, `test_protection_policy` | Surrogates and substitution. |
| `test_shadow_map`, `test_reconstruction`, `test_roundtrip`, `test_rag_roundtrip`, `test_chat_history`, `test_fail_closed` | Storage, restoring, the two chat histories and fail-closed detection. |
| `test_settings`, `test_config_validation`, `test_gitignore`, `test_run_sh`, `test_runner`, `test_evaluator`, `test_eval_metrics`, `test_attacker`, `test_presidio_baseline`, `test_single_implementation` | App plumbing, the batch tools, and a check that the app and the library share one engine. |
| `python-library/tests/test_public_api.py` | The package's public API: the version matches `pyproject.toml`, `__all__`, `config()`, sessions; the tests that run the full cascade are marked heavy. |

### Heavy tests

Marked `@pytest.mark.heavy`: `test_ablation`, `test_bench_compare`, `test_detection_heavy`, `test_I3_I5_consistency`, `test_library_concurrency`, `test_legacy_scripts`, `test_N7_no_network`, `test_V3_pii_tagger`, `test_V3_spacy_vectors`, and the cascade tests in `python-library/tests/test_public_api.py`.

> **Run the numbered legacy check scripts through pytest only.** `tests/` still holds six older numbered scripts that print `Results: X/Y passed`. pytest does not collect them (`collect_ignore` in `tests/conftest.py`). `tests/test_legacy_scripts.py` (heavy) runs each one in a subprocess, under the test home folder, and fails unless every check passed. Do not start them with `python` directly.

<a id="troubleshooting"></a>

## Troubleshooting

The messages below are the ones the code prints, with the values in angle brackets filled in at run time. Start with `surrogateshield doctor`: it checks the packages, the spaCy model, the ContextGuard cache, the tagger folder and its pin, and the storage modes, and prints `[ ok]`, `[warn]` or `[FAIL]` for each. Add `--smoke` to mask a sample message, `--cold-start` to load and time the models, and `--show-config` to print the effective detection config.

### Detection models

| Message | Fix |
|---|---|
| `EntityTrace needs the spaCy model 'en_core_web_lg', which is not installed. Run: python -m spacy download en_core_web_lg` | Run `python -m spacy download en_core_web_lg` with the same Python that runs SurrogateShield (with `.venv` active, or `.venv/bin/python -m spacy download en_core_web_lg`). |
| `Error: spaCy model 'en_core_web_lg' is not installed.` (from `run.sh`) | Run the command it prints next: `Install it with: <python> -m spacy download en_core_web_lg`. |
| `EntityTrace needs spaCy, which is not installed. Run: pip install spacy` | `pip install ./python-library` (or `pip install -r requirements.txt` for the app) in the active environment. |
| `PIITagger: no model folder 'pii-tagger-dv3s-40k' under <models dir> ($SURROGATESHIELD_MODELS). Copy the trained tagger there (python -m bench.tagger.install), point $SURROGATESHIELD_MODELS at the folder that holds it, or use the 'classic' detection preset (spaCy and ContextGuard read names; no tagger).` | Install a trained tagger: `PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.install bench/tagger/build/models/dv3s-40k`. `bench/tagger/build/` is not in the repository; it is where `bench.tagger.train` writes. Without the weights, set `SURROGATESHIELD_PRESET=classic`. The CLI's `--preset` accepts only `fast`, `balanced` and `strict`, so use the variable. |
| `PIITagger: '<path>' has no weights file (model.safetensors, pytorch_model.bin)` | The folder exists but is incomplete. Run `bench.tagger.install` again. |
| `PIITagger: the weights in '<path>' are not the pinned ones`, followed by the first 12 hex digits of the found and the pinned SHA-256 | The folder holds other weights than the preset's pin (`sha256:f9e5821151ac` and so on, for the default). Install the matching weights, or name your model and its revision in a detection config. |
| `PIITagger needs transformers and torch, which are not installed. Run: pip install transformers torch, or use the 'classic' detection preset.` | `pip install transformers torch`. |
| `ContextGuard needs transformers and torch, which are not installed. Run: pip install transformers torch — or switch ContextGuard off.` | `pip install transformers torch`, or turn ContextGuard off (it is off in `balanced` and `fast`). |
| `ContextGuard could not load 'dslim/distilbert-NER' (<error>). Download it once with network access (HF_HUB_OFFLINE unset), or switch ContextGuard off.` | Unset `HF_HUB_OFFLINE` once and run one scan with ContextGuard on, for example `surrogateshield scan --preset strict "warm up"`, which downloads and caches it. After that it loads from the cache without a network request. |
| `no detector is registered as '<name>' (register_detector or an entry point in '<group>'); switch the stage off to run without it` | A detection config names a plugin stage that is not registered. Call `register_detector` before masking, install the package that provides the entry point, or switch the stage off. See `examples/employee_id_detector.py`. |
| Panel **Detector unavailable**, ending in *Nothing was masked and nothing was sent.* | One of the messages above, raised during a chat turn, an `add-doc` or a batch run. Fix the cause it names and send again. |
| `surrogateshield: detector unavailable, nothing was masked: <reason>` | The same, from the library CLI. |

### Providers and API keys

| Message | Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set. Add it to your .env file.` (or `GEMINI_API_KEY`, `OPENAI_API_KEY`) | Add the line to `.env` in the project folder, or export the variable. The app shows it as `Configuration error:` with *Press S from the dashboard to configure your LLM provider.* |
| `claude rejected the key — check ANTHROPIC_API_KEY` (also for `gemini`, `chatgpt`) | The key is wrong or revoked. Replace it in `.env`. An Anthropic key that is not scoped to a workspace needs the workspace ID on every request; set it in `ANTHROPIC_WORKSPACE_ID`. Test with `T` in the provider settings. |
| `google-genai package not installed. Run: pip install google-genai` (also `openai`, `ollama`) | Run the command it gives. `requirements.txt` already includes `anthropic` and `ollama`; `google-genai` and `openai` are installed separately. |
| `<provider> call failed after 3 attempts: <error>` | Three attempts hit a timeout, a connection error, 408, 409, 429 or 5xx. Wait and send again; nothing was recorded for the failed turn. |
| `<provider> call failed: <error>` | An error that retrying does not fix, such as a bad model id. Check `SURROGATESHIELD_CLAUDE_MODEL`, `SURROGATESHIELD_GEMINI_MODEL` or `SURROGATESHIELD_OPENAI_MODEL` (defaults `claude-sonnet-4-6`, `gemini-2.5-flash`, `gpt-4o-mini`). |
| `local call failed after 3 attempts: <connection error>` | No Ollama server answers at `LOCAL_LLM_HOST` (default `http://localhost:11434`). Start it with `ollama serve` and fetch the model with `ollama pull llama3.2` (or the one in `LOCAL_LLM_MODEL`). |
| `Ollama returned no text`, `Claude returned no text (stop_reason=<reason>)`, `OpenAI returned no choices`, `Gemini blocked the prompt` | The provider answered with no text. These are not retried. Rephrase, or check the provider's safety settings. |
| `Unknown LLM provider: '<name>'` | `llm_provider` in `settings.json` must be `claude`, `gemini`, `chatgpt` or `local`; pick it again with `S`. |
| `No attacker model set: pass --model or set SURROGATESHIELD_ATTACKER_MODEL (it must differ from the responder model)` | Set `SURROGATESHIELD_ATTACKER_MODEL` to a Claude model other than `CLAUDE_MODEL`. |

### Device key and stored files

| Message | Fix |
|---|---|
| `cannot read or persist the device secret at <path>: <error>` | The home folder is not writable. Fix its owner and mode (`chmod 700 ~/.surrogateshield`), or set `SURROGATESHIELD_HOME` to a writable folder. |
| `<path> is corrupt (<n> bytes, expected 32); refusing to derive keys from it` | Restore `device.key` from a backup. Deleting it creates a new key on the next start, and every file sealed under the old key can then no longer be opened. |
| `[ShadowMap] <path> had loose permissions; fixed to 0600` | Informational: the mode was already corrected. |
| `Cannot open the surrogate map: shadow map <id>.shadowmap could not be decrypted (<reason>); moved to <id>.shadowmap.corrupt-<time>` | The map was written under another device key, or was changed on disk. The file is kept beside it. Restore the matching `device.key` and rename the file back to open it again. |
| `Cannot open conversation: conversation <id> cannot be decrypted with this device key (<reason>)` | Same cause, for the transcript. The conversation can still be deleted from the list without decrypting it. |
| `Old-format conversation '<id>' has no api_messages. Starting fresh API context to prevent PII leakage. Display history is preserved.` | Informational: the provider loses the earlier context of that conversation, and the display history stays. |
| `surrogateshield: no stored session '<id>' under <dir>` | Pass the same `--store` to `unmask` that `mask` used, and the session id `mask` printed on stderr. |
| `pii_mem path does not exist or is not a directory: '<path>'` | Create the folder before calling `config(pii_mem=...)`. |
| `pii_mem cannot change while the current session holds mappings; call flush() first or create a new Session` | Call `flush()` first, or start a new `Session`. |
| `Warning: .env is readable by other users (mode <mode>). Run: chmod 600 .env` | `chmod 600 .env`. |

### RAG

| Message | Fix |
|---|---|
| `chromadb is not installed. Run: pip install chromadb` | Run it, or `pip install -r requirements.txt`. |
| `sentence-transformers is not installed. Run: pip install sentence-transformers` | Run it. |
| `Failed to load embedding model 'all-MiniLM-L6-v2': <error>` | The model is not cached and could not be fetched. Run once with network access and `HF_HUB_OFFLINE` unset. |
| `Failed to open the RAG index at <dir>: <error>` | Check the folder's owner and mode. The index is rebuilt from your documents with `add-doc` if you move the folder aside. |
| `RAG unavailable: <reason>` | Shown in chat when the index cannot start; the chat goes on without retrieval. Fix the reason above. |
| `[RAG] ./chroma_db (pre-v2 location) is no longer used; the index now lives in <dir>. Re-run add-doc for those documents.` | Index the documents again with `python main.py add-doc <file>`, then remove `./chroma_db`. |
| `No indexed document named '<name>'.` | `python main.py rag` lists the file names and ids that `--forget` accepts. |

### Presidio, BERTScore and restoring

| Message | Fix |
|---|---|
| `presidio-analyzer not importable: <error> — run: pip install presidio-analyzer` | Install it into the same environment that runs the app. Installing into another Python is the usual cause. |
| `spaCy model not found: <error> — run: python -m spacy download en_core_web_lg` | Presidio uses the same spaCy model; run the command. |
| `bert-score not installed — BERTScore skipped.` | `pip install bert-score`. The first run downloads `roberta-large`. |
| `[ResolvePass] rapidfuzz not installed — skipping fuzzy pass` | `pip install rapidfuzz`; the exact restore passes still run without it. |

### A value was not replaced

- Paste the message into the [PII Finder](#pii-finder) (`P`) to see what each stage found.
- Check whether it is one of the [values sent as typed](#privacy-guarantees): a city in a service query, a public figure, an organisation nothing ties to a person.
- In chat, a borderline value asks `Replace this? [y/N]`; answer `y`.
- Check that its type is not in `pii_off` and not set to `keep` in [type_actions](#type-actions).
- Try `SURROGATESHIELD_PRESET=strict`. On the benchmark it let 0.2% of values through against 0.8% for `balanced`, and left 1,264 of the 2,433 natural messages untouched against 2,153 ([Presets](#results-presets)).
- For an in-house identifier format, register a detector plugin; `examples/employee_id_detector.py` shows one.

<a id="faq"></a>

## FAQ

### Does the provider ever see my real values?

Detected values are replaced before the request is built, apart from the [values sent as typed](#privacy-guarantees) on purpose. Values no stage detects go out as written. On the sealed test split of the real-data benchmark, 30 of 3,677 personal values (0.8%) reached the provider, against 9.1% for GLiNER-PII and 31.6% for Presidio ([Results](#results)).

### How do I install it?

For the library: `pip install ./python-library` from the repository folder, then `python -m spacy download en_core_web_lg`, then the PII tagger's weights (or the `classic` preset). For the terminal app: `pip install -r requirements.txt`. `surrogateshield doctor` checks the result. See [Installation](#installation).

### Does the library call an LLM?

No. `mask` returns the masked text and `unmask` restores an answer you pass in; your code makes the provider call. The `surrogateshield` CLI makes none either.

### Can I use it with no API key, or offline?

Yes, for everything except a cloud chat: the PII Finder, `add-doc`, Evaluation, `offline_eval.py`, the library, the CLI and the tests need no key. For chat, choose the local provider in Settings and run Ollama (`ollama pull llama3.2`, `ollama serve`). Masking makes no network request once its models are installed or cached; set `HF_HUB_OFFLINE=1` to be sure, and a stage whose model is missing then stops with `DetectorUnavailable`.

### Where are my conversations, and can I read them without the app?

In `~/.surrogateshield/conversations/`: `<id>.json` (the transcript) and `<id>.shadowmap` (the map), both AES-256-GCM sealed under a key derived from `device.key`. Without that key file they cannot be opened. See [Security Design](#security-design).

### What if I lose `device.key` or move to a new machine?

Copy `device.key` with the rest of `~/.surrogateshield`; the keys depend only on it and on each file's id, so the files open on the new machine. If the key is lost, a new one is created on the next start and the old files cannot be opened. A map that fails to decrypt is moved to `<name>.corrupt-<time>`, never deleted.

### Why did the app stop with "Detector unavailable"?

A detection stage that is switched on could not run. Rather than send text it could not check, SurrogateShield stops, masks nothing and sends nothing. The panel names the cause; [Troubleshooting](#troubleshooting) lists every message and its fix.

### Do I need the PII tagger's weights?

The default preset, `balanced`, runs the tagger `pii-tagger-dv3s-40k` and stops with `DetectorUnavailable` if its folder is missing. Without the weights, set `SURROGATESHIELD_PRESET=classic`, where spaCy and ContextGuard read names. On the benchmark `classic` let 2.5% of values through, against 0.8% for `balanced` ([Presets](#results-presets)).

### Is the RAG index encrypted?

The chunks in `~/.surrogateshield/rag/` are not encrypted; they are the document's sanitised text, with surrogates, in a `0700` folder. The surrogate-to-original pairs live in the sealed `rag_global.shadowmap`. The file name of each document is stored in the index as its `source`.

### How are my values put back into the answer?

ResolvePass reads the conversation's map on your machine and replaces each surrogate in the reply with its original: exact matches first, then a fuzzy pass (`rapidfuzz`, threshold 85) for surrogates the model reworded. In the multi-turn test it restored 81 of 82 values in the answers, and 5 of 150 real values were sent upstream ([Multi-turn](#results-multiturn)).

### Could a strong model work out the originals from the surrogates?

In the attacker test, `claude-opus-5-5` recovered 4 of the 620 values SurrogateShield replaced letter for letter, the lowest exact-recovery rate of the five tools tested; all 4 were URLs. Counting the values it left in, the attacker knew 11 of 627 (1.8%) ([Attacker](#results-attacker)).

### Does masking make the answers worse?

Answers to SurrogateShield's masked prompts scored a BERTScore of 0.442 against the answer to the original prompt, against 0.365 for Presidio + Faker and 0.369 for GLiNER-PII ([Answer Quality](#results-utility)).

### How fast is it?

Median 43.5 ms to mask one message on CPU, with 1,348 MB peak memory ([Speed](#results-speed)).

### Where does it trail?

**Limitations:** it trailed on the 96 real values in natural text (29 leaked; GLiNER-PII 16–20, Presidio 27), on some single types (3 of 13 on test-3, 8 of 13 on Nemotron-PII, URL most of all) and on speed and memory (plain Presidio is about 5× faster and lighter); it came out ahead on every other test. See [Limitations](#limitations).

### How do I delete everything it stored?

Delete a conversation from the conversation list (it removes the transcript and the map without decrypting them), forget a document with `python main.py rag --forget DOC`, and end a library session with `flush()`, which deletes its stored map. Removing `~/.surrogateshield` removes all of it, including `device.key`. Batch outputs live in the project's `experiment/` folder.
