"""Corpus-wide address detection: exact-span match rate must stay >= 98%.

The corpus (tests/data/address_corpus.json, 300+ entries) encodes the intended
behavior of the canonical parser across formats: cross-product street forms,
directionals, units, PO boxes, ZIP+4, full state names, no-comma, multi-line,
lowercase, highways, ordinals, ambiguous suffixes, and the experiment
ground-truth values. Any regression below the threshold fails loudly with the
full mismatch list.
"""

from surrogateshield.core.detection import address_parser as ap

REQUIRED_RATE = 0.98  # target is 1.0; corpus was frozen at 100%


def test_corpus_exact_span_rate(corpus):
    mismatches = []
    for entry in corpus:
        found = ap.find_addresses(entry["text"])
        got = found[0].full_text if found else None
        if got != entry["expected"]:
            mismatches.append(
                f"({entry['note']}) {entry['text']!r}\n"
                f"    expected {entry['expected']!r}\n"
                f"    got      {got!r}"
            )

    rate = 1 - len(mismatches) / len(corpus)
    detail = "\n".join(mismatches)
    assert rate >= REQUIRED_RATE, (
        f"corpus exact-span rate {rate:.4f} < {REQUIRED_RATE} "
        f"({len(mismatches)}/{len(corpus)} mismatches):\n{detail}"
    )


def test_corpus_size(corpus):
    assert len(corpus) >= 300, "address corpus shrank below 300 entries"


def test_corpus_spans_are_consistent(corpus):
    """start/end offsets must always slice exactly to full_text."""
    for entry in corpus:
        for parsed in ap.find_addresses(entry["text"]):
            assert entry["text"][parsed.start:parsed.end] == parsed.full_text
