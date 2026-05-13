# SurrogateShield

> A privacy-preserving CLI proxy for Claude — PII never leaves your device.

SurrogateShield intercepts your messages before they reach the Claude API, replaces all detected personally identifiable information (PII) with realistic fake surrogates, sends the sanitised message, and restores your real values in Claude's response. Everything runs locally. Nothing sensitive is ever transmitted.

---

## Architecture

```
User message
    │
    ▼
SentinelLayer  ──────────────────────────────────────────────────────
    ├── PatternScan     regex, score=1.0, always confirmed
    ├── EntityTrace     spaCy NER, score≥0.85 confirmed, 0.60–0.85 borderline
    └── ContextGuard    Ollama phi3:mini, implicit PII, quasi-identifiers
    │
    ▼ confirmed_entities + needs_confirmation
MimicGen → realistic surrogates (Faker, collision-resistant)
    │
    ▼
ShadowMap (AES-256-GCM encrypted, conversation-tied)
    │
    ▼
Claude API (claude-sonnet-4-6)
    │
    ▼
ResolvePass → exact + fuzzy (rapidfuzz) swap-back
    │
    ▼
User sees real values in response
```

---

## Project Structure

```
SurrogateShield/
├── main.py                  # CLI entry point (Typer)
├── pipeline.py              # Full message flow orchestration
├── config.py                # All constants, thresholds, model names
├── util.py                  # Shared helpers, dataclasses, Rich display
│
├── detection/
│   ├── __init__.py
│   ├── logic.py             # SentinelLayer cascade
│   ├── pattern_scan.py      # Regex-based PII detection
│   ├── entity_trace.py      # spaCy NER detection
│   └── context_guard.py     # Ollama SLM detection
│
├── generation/
│   ├── __init__.py
│   └── logic.py             # MimicGen surrogate generation
│
├── storage/
│   ├── __init__.py
│   └── logic.py             # ShadowMap encrypted cache
│
├── reconstruction/
│   ├── __init__.py
│   └── logic.py             # ResolvePass swap-back
│
├── chatbot/
│   ├── __init__.py
│   ├── chat.py              # Claude API handler
│   └── rag.py               # RAG — ChromaDB + sentence-transformers
│
└── conversations/           # Auto-created — .json + .shadowmap files
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy model

```bash
python -m spacy download en_core_web_lg
```

### 3. Install and run Ollama (for ContextGuard)

Ollama provides the local SLM layer. If Ollama is not running, SurrogateShield degrades gracefully — PatternScan and EntityTrace still run.

```bash
# Install Ollama: https://ollama.ai
ollama pull phi3:mini
```

### 4. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

On Windows:
```cmd
set ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

### Start a new conversation

```bash
python main.py chat
```

### Continue an existing conversation

```bash
python main.py chat --load <conversation-id>
```

### Delete a conversation (and its ShadowMap)

```bash
python main.py chat --delete <conversation-id>
```

### Start a conversation with RAG mode

```bash
python main.py chat --rag
```

### List all saved conversations

```bash
python main.py list
```

### Add a document to the RAG vector store

```bash
python main.py add-doc /path/to/document.txt
```

---

## Example session

```
SurrogateShield v1.0
Privacy-preserving Claude proxy · PII stays on your device

New conversation started.
Conversation ID: 3a7f2e1c-...

You: My name is Ahmed Al-Rashidi and my email is ahmed@gmail.com

╭─ SentinelLayer — PII Detected ─────────────────────────────────╮
│ Original          │ Type   │ Score │ Source  │ Surrogate        │
│ Ahmed Al-Rashidi  │ PERSON │ 0.85  │ ner     │ Marcus Ellison   │
│ ahmed@gmail.com   │ email  │ 1.00  │ pattern │ d.lee@yahoo.com  │
╰─────────────────────────────────────────────────────────────────╯

╭─ Claude ────────────────────────────────────────────────────────╮
│ Nice to meet you, Ahmed! How can I help you today?              │
╰─────────────────────────────────────────────────────────────────╯
```

---

## Detection pipeline

| Stage | Method | Confidence | Action |
|-------|--------|------------|--------|
| PatternScan | Regex (SSN, email, phone, card, DOB, IP, API keys, postcodes) | 1.0 always | Auto-replace |
| EntityTrace | spaCy `en_core_web_lg` (PERSON, GPE, LOC, ORG, FAC) | spaCy score | ≥0.85: replace; 0.60–0.85: ask ContextGuard |
| ContextGuard | Ollama `phi3:mini` (implicit PII, quasi-identifiers) | SLM JSON | ≥0.70: replace; <0.70: ask user |

---

## Security properties

| Attacker | Access | Recovery |
|----------|--------|----------|
| A1: Network observer | Messages in transit | Surrogates only |
| A2: LLM provider | API requests | Sanitised text only |
| A3: Stolen files | Encrypted ShadowMap | Nothing without the key |
| A4: Full device compromise | Out of scope | — |

The ShadowMap key is derived per-conversation via HKDF-SHA256 from `conversation_id + device_secret`. The device secret is generated once and stored in `~/.surrogateshield/device.key`. The key is never stored alongside the ShadowMap.

---

## Configuration

All thresholds and model names are in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `ENTITY_TRACE_HIGH_THRESHOLD` | 0.85 | spaCy score → auto-replace |
| `ENTITY_TRACE_LOW_THRESHOLD` | 0.60 | spaCy score → send to ContextGuard |
| `CONTEXT_GUARD_CONFIDENCE_THRESHOLD` | 0.70 | SLM confidence → auto-replace |
| `FUZZY_MATCH_THRESHOLD` | 85 | rapidfuzz partial_ratio threshold |
| `RAG_TOP_K` | 3 | Chunks retrieved per query |
| `CLAUDE_MODEL` | claude-sonnet-4-6 | Claude API model |
| `CONTEXT_GUARD_MODEL` | phi3:mini | Ollama SLM model |

---

## Graceful degradation

- **Ollama not running**: ContextGuard is skipped. PatternScan and EntityTrace still run. No crash.
- **spaCy model missing**: EntityTrace is skipped with a clear error message. PatternScan still runs.
- **ShadowMap file missing**: Starts with empty mappings. No crash.
- **Network error**: Claude API raises an exception with a clear message.

---

## Research context

SurrogateShield is a research prototype targeting submission to PETS 2027. See the project design document (`SurrogateShield_v3_OnePaper.docx`) for the full system architecture, evaluation plan, threat model, and academic framing.
