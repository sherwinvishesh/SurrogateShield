"""Pattern-level mask → echo → unmask round trips (no NER models needed).

Uses PatternScan + MimicGen + apply_entity_surrogates + ResolvePass directly,
covering the full v2 address path in all three modes.
"""

import pytest

from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.entities import apply_entity_surrogates
from surrogateshield.core.generation.mimic import MimicGen
from surrogateshield.core.reconstruction.resolve import ResolvePass
from surrogateshield.core.storage.shadow_map import ShadowMap


def _mask(text, mode, gen=None, shadow=None):
    """Minimal mask() replica for pattern-detectable entities."""
    gen = gen or MimicGen(seed=99)
    entities = pattern_scan.scan(text)
    surrogate_map = {}
    new = []
    for e in entities:
        key = e.text.strip()
        existing = shadow.lookup_original(key) if shadow else None
        if existing is not None:
            surrogate_map[key] = existing
        else:
            new.append(e)
    if new:
        forbidden = set(shadow.originals()) if shadow else set()
        surrogate_map.update(
            gen.generate_all(new, address_mode=mode, forbidden=forbidden)
        )
    if shadow is not None:
        shadow.update({v: k for k, v in surrogate_map.items()})
    return apply_entity_surrogates(text, entities, surrogate_map), surrogate_map


@pytest.mark.parametrize("mode", ["shift", "replace"])
@pytest.mark.parametrize("text", [
    "Ship it to 789 Crescent Row Apt 4B, Tempe, AZ 85281-1234 and email jane@example.com.",
    "My SSN is 123-45-6789 and I live at 500 Main St, Mesa, AZ 85204.",
    "Send mail to PO Box 1234, Flagstaff, AZ 86001 before 5 pm.",
    "Card 4111 1111 1111 1111 billed to 12 Oak Rd, Suite 200, Mesa, AZ.",
])
def test_mask_echo_unmask_roundtrip(text, mode):
    sanitized, surrogate_map = _mask(text, mode)

    # nothing sensitive survives in the sanitized text
    for original in surrogate_map:
        assert original not in sanitized

    # the LLM echoes the sanitized text verbatim → unmask restores the original
    restored = ResolvePass().resolve(
        sanitized, {v: k for k, v in surrogate_map.items()}, fuzzy_threshold=85
    )
    assert restored == text


def test_shift_mode_keeps_street_and_city_visible():
    sanitized, _ = _mask("I live at 789 Crescent Row, Tempe, AZ 85281.", "shift")
    assert "Crescent Row" in sanitized  # utility preserved
    assert "Tempe" in sanitized
    assert "789" not in sanitized       # number shifted


def test_replace_mode_hides_every_component():
    sanitized, _ = _mask("I live at 789 Crescent Row, Tempe, AZ 85281.", "replace")
    assert "789" not in sanitized
    assert "Tempe" not in sanitized
    assert "85281" not in sanitized


def test_repeated_address_gets_one_surrogate():
    text = ("Bill to 789 Crescent Row, Tempe, AZ 85281. "
            "Also ship to 789 Crescent Row, Tempe, AZ 85281.")
    sanitized, surrogate_map = _mask(text, "shift")
    assert len(surrogate_map) == 1
    surrogate = next(iter(surrogate_map.values()))
    assert sanitized.count(surrogate) == 2


def test_multiturn_reuse_via_shadow_map():
    shadow = ShadowMap("test-session")
    gen = MimicGen(seed=77)

    s1, m1 = _mask("I live at 789 Crescent Row, Tempe, AZ 85281.", "shift", gen, shadow)
    s2, m2 = _mask("Confirm: 789 Crescent Row, Tempe, AZ 85281 is correct.", "shift", gen, shadow)

    key = "789 Crescent Row, Tempe, AZ 85281"
    assert m1[key] == m2[key], "same original must reuse the same surrogate across turns"

    # a NEW address in turn 3 must not collide with turn-1 original values
    s3, m3 = _mask("Also note 790 Crescent Row, Tempe, AZ 85281.", "shift", gen, shadow)
    assert m3["790 Crescent Row, Tempe, AZ 85281"] != key


def test_two_addresses_same_message_roundtrip():
    text = "From 12 Oak Rd, Mesa, AZ to 789 Crescent Row, Tempe, AZ."
    sanitized, surrogate_map = _mask(text, "shift")
    assert len(surrogate_map) == 2
    restored = ResolvePass().resolve(
        sanitized, {v: k for k, v in surrogate_map.items()}, fuzzy_threshold=85
    )
    assert restored == text
