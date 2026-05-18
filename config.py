"""
config.py — SurrogateShield Global Configuration

All constants, model names, thresholds, and paths used across the project.
Centralised here so every module reads from one source of truth.
"""

# ─────────────────────────────────────────────
# Detection thresholds
# ─────────────────────────────────────────────

ENTITY_TRACE_HIGH_THRESHOLD: float = 0.85   # spaCy score → confirmed
ENTITY_TRACE_LOW_THRESHOLD: float = 0.60    # spaCy score → borderline (sent to ContextGuard)

CONTEXT_GUARD_CONFIDENCE_THRESHOLD: float = 0.70  # ContextGuard confidence → confirmed

# ─────────────────────────────────────────────
# Model names
# ─────────────────────────────────────────────

# ContextGuard now uses a local HuggingFace model — no Ollama server needed.
# First run will download ~250 MB from HuggingFace Hub (cached afterwards).
CONTEXT_GUARD_MODEL: str = "dslim/distilbert-NER"
CONTEXT_GUARD_ENABLED: bool = True            # always on — no Ollama required
CONTEXT_GUARD_FALLBACK_TO_OLLAMA: bool = False  # set True to use phi3:mini instead

CLAUDE_MODEL: str = "claude-sonnet-4-6"          # Claude API model
GEMINI_MODEL: str = "gemini-1.5-flash"            # Gemini API model
OPENAI_MODEL: str = "gpt-4o-mini"                 # OpenAI API model
LOCAL_LLM_MODEL: str = "llama3.2"                 # Default Ollama model
LOCAL_LLM_HOST: str = "http://localhost:11434"     # Default Ollama server host
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"        # sentence-transformers for RAG
SPACY_MODEL: str = "en_core_web_lg"              # spaCy NER model

# ─────────────────────────────────────────────
# RAG settings
# ─────────────────────────────────────────────

RAG_TOP_K: int = 3                               # Number of chunks to retrieve
RAG_COLLECTION_NAME: str = "surrogateshield_rag"
RAG_CHUNK_SIZE: int = 512                        # Characters per chunk when splitting docs

# ─────────────────────────────────────────────
# Storage paths
# ─────────────────────────────────────────────

SHADOWMAP_DIR: str = "conversations"             # Relative to project root
DEVICE_KEY_PATH: str = "~/.surrogateshield/device.key"

# ─────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────

FUZZY_MATCH_THRESHOLD: int = 85                  # rapidfuzz partial_ratio threshold (0–100)

# ─────────────────────────────────────────────
# Crypto
# ─────────────────────────────────────────────

HKDF_INFO: bytes = b"shadowmap"                  # HKDF derivation info label
AES_NONCE_SIZE: int = 12                         # GCM nonce length in bytes

# ─────────────────────────────────────────────
# Detection (fallback)
# ─────────────────────────────────────────────

# When ContextGuard is disabled, borderline NER entities above this score
# are promoted to confirmed rather than silently dropped.
# Catches LOC (0.74) and FAC (0.70) entities in the default configuration.
ENTITY_TRACE_FALLBACK_THRESHOLD: float = 0.65

# ─────────────────────────────────────────────
# Service query detection
# ─────────────────────────────────────────────

# When True, messages matching service/knowledge query patterns receive minimal
# address fuzzing (house-number shift ±2–8) instead of full surrogate replacement.
SERVICE_QUERY_DETECTION_ENABLED: bool = True

# If True, each fuzzed address is verified via OpenStreetMap Nominatim (~1-2s per
# address). Set False to skip network call (useful in tests or offline environments).
SERVICE_QUERY_VERIFY_ADDRESSES: bool = True

# ─────────────────────────────────────────────
# Logging / display
# ─────────────────────────────────────────────

LOG_LEVEL: str = "INFO"
SHOW_DETECTION_TABLE: bool = True                # Print detection results in chat mode

# Show a transparency panel after each turn: what was sent to Anthropic,
# what raw response came back, and what the final restored output is.
SHOW_API_TRANSPARENCY: bool = True