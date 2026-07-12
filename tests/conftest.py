"""Shared pytest fixtures for the SurrogateShield test suite."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python-library"))

# Legacy check()-style scripts — executed via tests/test_legacy_scripts.py
# subprocess wrapper, never imported by pytest collection (they run at import).
collect_ignore = [
    "test1.py", "test2.py", "test3.py", "test4.py",
    "test5.py", "test6.py", "test7.py",
]

_CORPUS_PATH = Path(__file__).parent / "data" / "address_corpus.json"


@pytest.fixture(scope="session")
def corpus():
    """The frozen address corpus: [{text, expected, note}, …]."""
    with open(_CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)
