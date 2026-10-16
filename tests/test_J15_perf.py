"""Gate J15 — bench/perf.py measures what the gate names: distinct invented
messages of about 200 tokens, the same set for a given seed. Model-free (the
blank spaCy tokenizer); the timings themselves are in bench/results/j15_perf.json.
"""

import importlib.util
from pathlib import Path

import spacy

_spec = importlib.util.spec_from_file_location(
    "perf", Path(__file__).resolve().parent.parent / "bench" / "perf.py")
perf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf)


def test_J15_messages_are_200_tokens_distinct_and_seeded():
    tok = spacy.blank("en").tokenizer
    texts = perf.messages(12, seed=4)
    assert len(set(texts)) == 12
    assert all(190 <= len(tok(t)) <= 210 for t in texts)
    assert texts == perf.messages(12, seed=4) != perf.messages(12, seed=5)


def test_J15_every_message_carries_contact_details():
    for t in perf.messages(6):
        assert "@example.com" in t and "(480) 555-" in t and "my name is" in t


def test_J15_gates():
    assert perf.GATE_MS == {False: 50.0, True: 150.0}
