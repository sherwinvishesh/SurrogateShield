# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
config.py — SurrogateShield Global Configuration

All constants, model names, thresholds, and paths used across the project.
Centralised here so every module reads from one source of truth.
"""

import os
from typing import Optional

# ─────────────────────────────────────────────
# Detection thresholds
# ─────────────────────────────────────────────

# spaCy's NER gives no per-entity confidence, so EntityTrace assigns one per
# label: PERSON 0.88, GPE/ORG 0.85, LOC 0.74, FAC 0.70. These thresholds are
# therefore TYPE GATES, not confidence cut-offs: with the defaults PERSON/GPE/
# ORG are confirmed and LOC/FAC are borderline (tests/test_F3_settings.py).
ENTITY_TRACE_HIGH_THRESHOLD: float = 0.85   # type score ≥ this → confirmed
ENTITY_TRACE_LOW_THRESHOLD: float = 0.60    # type score ≥ this → borderline (sent to ContextGuard)

CONTEXT_GUARD_CONFIDENCE_THRESHOLD: float = 0.70  # borderline type score ≥ this → confirmed

# ─────────────────────────────────────────────
# Model names
# ─────────────────────────────────────────────

# ContextGuard now uses a local HuggingFace model — no Ollama server needed.
# First run will download ~250 MB from HuggingFace Hub (cached afterwards).
CONTEXT_GUARD_MODEL: str = "dslim/distilbert-NER"
CONTEXT_GUARD_DEVICE: int = -1                # -1 = CPU, >= 0 = GPU device id
CONTEXT_GUARD_ENABLED: bool = True            # always on — no Ollama required

# Model ids live here only (audit F2); each can be overridden from the
# environment without editing code.
CLAUDE_MODEL: str = os.getenv("SURROGATESHIELD_CLAUDE_MODEL") or "claude-sonnet-4-6"
# Attacker experiment (attacker.py). Must differ from the model that answered
# the questions (audit I30); there is no default — set it explicitly or via
# SURROGATESHIELD_ATTACKER_MODEL.
ATTACKER_MODEL: Optional[str] = os.getenv("SURROGATESHIELD_ATTACKER_MODEL") or None
GEMINI_MODEL: str = os.getenv("SURROGATESHIELD_GEMINI_MODEL") or "gemini-2.5-flash"
OPENAI_MODEL: str = os.getenv("SURROGATESHIELD_OPENAI_MODEL") or "gpt-4o-mini"
LOCAL_LLM_MODEL: str = "llama3.2"                 # Default Ollama model (LOCAL_LLM_MODEL env)
LOCAL_LLM_HOST: str = "http://localhost:11434"     # Default Ollama server host
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"        # sentence-transformers for RAG
SPACY_MODEL: str = "en_core_web_lg"              # spaCy NER model

# ─────────────────────────────────────────────
# RAG settings
# ─────────────────────────────────────────────

RAG_TOP_K: int = 3                               # Number of chunks to retrieve
RAG_COLLECTION_NAME: str = "surrogateshield_rag"
RAG_CHUNK_SIZE: int = 512                        # Characters per chunk when splitting docs
RAG_DIR: Optional[str] = None                    # chroma index; None → ~/.surrogateshield/rag

# ─────────────────────────────────────────────
# Storage paths
# ─────────────────────────────────────────────

# None → under $SURROGATESHIELD_HOME (default ~/.surrogateshield), never the
# current directory (audit I11): <home>/conversations and <home>/device.key.
SHADOWMAP_DIR: Optional[str] = None
DEVICE_KEY_PATH: Optional[str] = None
# If the device secret cannot be persisted, raise (default) instead of using a
# process-lifetime secret whose data is unreadable after exit (audit F1).
ALLOW_EPHEMERAL_KEY: bool = False

# ─────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────

FUZZY_MATCH_THRESHOLD: int = 85                  # rapidfuzz partial_ratio threshold (0–100)

# ─────────────────────────────────────────────
# Detection (fallback)
# ─────────────────────────────────────────────

# When ContextGuard is disabled, borderline NER entities above this score
# are promoted to confirmed rather than silently dropped.
# Catches LOC (0.74) and FAC (0.70) entities in the default configuration.
ENTITY_TRACE_FALLBACK_THRESHOLD: float = 0.65

# ─────────────────────────────────────────────
# Address handling (v2)
# ─────────────────────────────────────────────

# How detected addresses are surrogated:
#   "auto"    — shift for non-sensitive service queries ("coffee near 12 Elm
#               St, Tempe, AZ"), replace for everything else; sensitive
#               topics always force replace (default — the only mode whose
#               verbatim street/city/ZIP is covered by a documented policy).
#   "shift"   — house number shifted by up to ±ADDRESS_SHIFT_RANGE for EVERY
#               address; street, city, state, ZIP, and formatting are sent
#               unchanged (a billing address in a letter goes out ±1).
#   "replace" — structure-preserving fake address (every component faked,
#               same shape — strongest privacy).
ADDRESS_MODE: str = "auto"

# Maximum house-number delta for shift mode (>= 1). ±1 keeps the geographic
# error to roughly one building.
ADDRESS_SHIFT_RANGE: int = 1

# ─────────────────────────────────────────────
# Service query detection
# ─────────────────────────────────────────────

# When True, service/knowledge queries suppress standalone city/state
# replacement (preserves answer utility) and drive ADDRESS_MODE="auto".
SERVICE_QUERY_DETECTION_ENABLED: bool = True


# ─────────────────────────────────────────────
# Logging / display
# ─────────────────────────────────────────────

# Log level: SURROGATESHIELD_LOG_LEVEL (environment), read in main.py.

# Show a transparency panel after each turn: what was sent to Anthropic,
# what raw response came back, and what the final restored output is.
SHOW_API_TRANSPARENCY: bool = True

# ─────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────

VERSION: str = "2.1.0"


# ─────────────────────────────────────────────
# Config validation (runs on import — fails fast on bad edits)
# ─────────────────────────────────────────────

def validate_config() -> None:
    """Raise ValueError with an actionable message on any invalid setting."""
    if ADDRESS_MODE not in ("shift", "replace", "auto"):
        raise ValueError(
            f"ADDRESS_MODE must be 'shift', 'replace', or 'auto', got {ADDRESS_MODE!r}"
        )
    if not isinstance(ADDRESS_SHIFT_RANGE, int) or ADDRESS_SHIFT_RANGE < 1:
        raise ValueError(
            f"ADDRESS_SHIFT_RANGE must be an integer >= 1, got {ADDRESS_SHIFT_RANGE!r}"
        )
    if not (0 <= FUZZY_MATCH_THRESHOLD <= 100):
        raise ValueError(
            f"FUZZY_MATCH_THRESHOLD must be in [0, 100], got {FUZZY_MATCH_THRESHOLD!r}"
        )
    for name, value in (
        ("ENTITY_TRACE_HIGH_THRESHOLD", ENTITY_TRACE_HIGH_THRESHOLD),
        ("ENTITY_TRACE_LOW_THRESHOLD", ENTITY_TRACE_LOW_THRESHOLD),
        ("CONTEXT_GUARD_CONFIDENCE_THRESHOLD", CONTEXT_GUARD_CONFIDENCE_THRESHOLD),
        ("ENTITY_TRACE_FALLBACK_THRESHOLD", ENTITY_TRACE_FALLBACK_THRESHOLD),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    if not isinstance(CONTEXT_GUARD_DEVICE, int):
        raise ValueError(
            f"CONTEXT_GUARD_DEVICE must be an integer (-1 = CPU), got {CONTEXT_GUARD_DEVICE!r}"
        )


validate_config()