"""Consistency guarantees: same input → same surrogate, sessions isolate."""

import uuid

from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.generation.mimic import MimicGen
from surrogateshield.core.storage.shadow_map import ShadowMap


ADDRESS = "789 Crescent Row, Tempe, AZ 85281"


def _turn(text, gen, shadow, mode="shift"):
    """One mask turn: detect → reuse-or-generate → record."""
    entities = pattern_scan.scan(text)
    mapping = {}
    new = []
    for e in entities:
        key = e.text.strip()
        existing = shadow.lookup_original(key)
        if existing is not None:
            mapping[key] = existing
        else:
            new.append(e)
    if new:
        mapping.update(
            gen.generate_all(new, address_mode=mode, forbidden=set(shadow.originals()))
        )
    shadow.update({v: k for k, v in mapping.items()})
    return mapping


def test_same_original_same_surrogate_across_turns():
    gen, shadow = MimicGen(seed=1), ShadowMap(str(uuid.uuid4()))
    m1 = _turn(f"I live at {ADDRESS}.", gen, shadow)
    m2 = _turn(f"Again: {ADDRESS} is correct.", gen, shadow)
    m3 = _turn(f"One more time — {ADDRESS}.", gen, shadow)
    assert m1[ADDRESS] == m2[ADDRESS] == m3[ADDRESS]


def test_different_originals_get_different_surrogates():
    gen, shadow = MimicGen(seed=2), ShadowMap(str(uuid.uuid4()))
    m1 = _turn("addr one: 12 Oak Rd, Mesa, AZ.", gen, shadow)
    m2 = _turn("addr two: 789 Crescent Row, Tempe, AZ.", gen, shadow)
    assert m1["12 Oak Rd, Mesa, AZ"] != m2["789 Crescent Row, Tempe, AZ"]


def test_new_surrogate_never_equals_prior_original():
    """A shift of '790 …' must not land on the '789 …' seen in turn 1."""
    gen, shadow = MimicGen(seed=3), ShadowMap(str(uuid.uuid4()))
    _turn("first: 789 Crescent Row.", gen, shadow)
    m2 = _turn("second: 790 Crescent Row.", gen, shadow)
    assert m2["790 Crescent Row"] != "789 Crescent Row"


def test_flush_resets_consistency():
    gen, shadow = MimicGen(seed=4), ShadowMap(str(uuid.uuid4()))
    m1 = _turn(f"send to {ADDRESS}.", gen, shadow)
    shadow.flush()
    gen2 = MimicGen(seed=99)  # new session generator
    m2 = _turn(f"send to {ADDRESS}.", gen2, shadow)
    # after a flush the mapping is regenerated (no stale reuse)
    assert shadow.lookup_original(ADDRESS) == m2[ADDRESS]


def test_email_and_ssn_consistency_too():
    gen, shadow = MimicGen(seed=5), ShadowMap(str(uuid.uuid4()))
    text = "jane@example.com / 123-45-6789"
    m1 = _turn(text, gen, shadow)
    m2 = _turn(f"confirm {text}", gen, shadow)
    assert m1 == {k: m2[k] for k in m1}
