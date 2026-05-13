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

CONTEXT_GUARD_CONFIDENCE_THRESHOLD: float = 0.70  # ContextGuard JSON confidence → confirmed

# ─────────────────────────────────────────────
# Model names
# ─────────────────────────────────────────────

CONTEXT_GUARD_MODEL: str = "phi3:mini"           # Ollama local SLM
CONTEXT_GUARD_ENABLED: bool = False              # Set True after: ollama pull phi3:mini
CLAUDE_MODEL: str = "claude-sonnet-4-6"          # Claude API model
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
# Logging / display
# ─────────────────────────────────────────────

LOG_LEVEL: str = "INFO"
SHOW_DETECTION_TABLE: bool = True                # Print detection results in chat mode