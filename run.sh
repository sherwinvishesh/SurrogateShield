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
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m' # No Colour

# ── 1. Find and activate virtual environment ──────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${YELLOW}No .venv found. Running with system Python.${NC}"
    echo -e "${DIM}To create one: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt${NC}"
fi

# ── 2. Load .env if it exists ─────────────────────────────────────────────
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

# ── 3. Check ANTHROPIC_API_KEY ────────────────────────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY is not set.${NC}"
    echo ""
    echo "Fix it one of these ways:"
    echo ""
    echo -e "  ${BLUE}Option A${NC} — add to your shell profile (permanent):"
    echo -e "  ${DIM}echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc && source ~/.zshrc${NC}"
    echo ""
    echo -e "  ${BLUE}Option B${NC} — create a .env file in this folder (per-project):"
    echo -e "  ${DIM}echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env${NC}"
    echo ""
    echo -e "  ${BLUE}Option C${NC} — set it just for this run:"
    echo -e "  ${DIM}ANTHROPIC_API_KEY=sk-ant-... ./run.sh${NC}"
    echo ""
    exit 1
fi

# ── 4. Launch ─────────────────────────────────────────────────────────────
python main.py "$@"