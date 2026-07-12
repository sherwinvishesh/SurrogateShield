"""Root tree and python-library must behave identically (anti-drift gate)."""

import random

from surrogateshield.core.detection import address_parser as lib_ap
from surrogateshield.core.generation.mimic import shift_house_number as lib_shift
from detection import address_parser as root_ap
from generation.logic import shift_house_number as root_shift


def _fingerprint(parsed):
    return (
        parsed.full_text, parsed.start, parsed.end,
        parsed.house_number, parsed.house_number_span,
        parsed.pre_directional, parsed.street_name, parsed.suffix,
        parsed.post_directional, parsed.unit, parsed.city,
        parsed.state, parsed.zip_code, parsed.is_po_box,
    )


def test_parsers_identical_over_corpus(corpus):
    for entry in corpus:
        lib_result = [_fingerprint(p) for p in lib_ap.find_addresses(entry["text"])]
        root_result = [_fingerprint(p) for p in root_ap.find_addresses(entry["text"])]
        assert lib_result == root_result, f"parser drift on {entry['text']!r}"


def test_shift_identical_with_same_seed(corpus):
    for entry in corpus[:50]:
        lib_parsed = lib_ap.find_addresses(entry["text"])
        root_parsed = root_ap.find_addresses(entry["text"])
        for lp, rp in zip(lib_parsed, root_parsed):
            ls = lib_shift(lp, rng=random.Random(7))
            rs = root_shift(rp, rng=random.Random(7))
            assert ls == rs, f"shift drift on {entry['text']!r}"


def test_source_files_differ_only_in_imports():
    """The parser twins must stay byte-identical below the import header."""
    lib_src = open("python-library/surrogateshield/core/detection/address_parser.py").read()
    root_src = open("detection/address_parser.py").read()

    def _body(src):
        # everything after the logger line is the shared implementation
        marker = "\n\n\n# ─"
        return src[src.index(marker):]

    assert _body(lib_src) == _body(root_src)
