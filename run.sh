#!/usr/bin/env bash
# Paper available on arXiv: https://arxiv.org/abs/2606.29567
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — SurrogateShield launcher
#
# Usage:
#   ./run.sh              Start the interactive dashboard
#   ./run.sh chat         Start a new conversation directly
#   ./run.sh list         List saved conversations
#
# First time: chmod +x run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Locate project root (the directory this script lives in) ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m' # No Colour

# ── 1. Pick the interpreter (venv first; stock macOS has no `python`) ─────
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="$(command -v python3 || true)"
    if [ -z "$PY" ]; then
        echo -e "${RED}Error: python3 not found.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}No .venv found. Running with $PY.${NC}"
    echo -e "${DIM}To create one: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt${NC}"
fi

# ── 2. API keys ───────────────────────────────────────────────────────────
# main.py loads .env itself (python-dotenv) and the selected provider reports
# its own missing key, so Gemini / ChatGPT / Ollama users are not blocked by
# an Anthropic check here (audit I24). The file is never parsed or printed
# by this script; only its permissions are checked.
if [ -f ".env" ]; then
    ENV_MODE="$(stat -f '%Lp' .env 2>/dev/null || stat -c '%a' .env 2>/dev/null || echo 600)"
    if [ "${ENV_MODE#?}" != "00" ]; then
        echo -e "${YELLOW}Warning: .env is readable by other users (mode $ENV_MODE). Run: chmod 600 .env${NC}"
    fi
fi

# ── 3. Check the spaCy model (detection fails closed without it, audit I17) ─
SPACY_MODEL="$("$PY" -c 'import config; print(config.SPACY_MODEL)')"
if ! "$PY" -c "import spacy.util, sys; sys.exit(0 if spacy.util.is_package('$SPACY_MODEL') else 1)" 2>/dev/null; then
    echo -e "${RED}Error: spaCy model '$SPACY_MODEL' is not installed.${NC}"
    echo -e "Install it with: ${DIM}$PY -m spacy download $SPACY_MODEL${NC}"
    exit 1
fi

# ── 4. Launch ─────────────────────────────────────────────────────────────
exec "$PY" main.py "$@"
