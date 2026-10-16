"""Span-based substitution (util.plan_substitutions / splice) — audit E1.

Both trees must behave identically until Phase 6 removes the duplicate.
"""

import pytest

import util as root_util
from surrogateshield.core import entities as lib_entities


def _ent(mod, text, start, typ="PERSON"):
    return mod.DetectedEntity(text=text, start=start, end=start + len(text), type=typ)


@pytest.mark.parametrize("mod", [root_util, lib_entities], ids=["root", "library"])
def test_E1_surrogate_never_rewritten(mod):
    text = "Ann met Bob."
    ents = [_ent(mod, "Ann", 0), _ent(mod, "Bob", 8)]
    mapping = {"Ann": "Bob Lee", "Bob": "Carl"}
    # Sequential str.replace would turn "Bob Lee" into "Carl Lee".
    assert mod.apply_entity_surrogates(text, ents, mapping) == "Bob Lee met Carl."


@pytest.mark.parametrize("mod", [root_util, lib_entities], ids=["root", "library"])
def test_E1_edits_in_original_coordinates(mod):
    text = "Ann met Bob. Ann left."
    ents = [_ent(mod, "Ann", 0), _ent(mod, "Bob", 8)]
    edits = mod.plan_substitutions(text, ents, {"Ann": "Zoe", "Bob": "Max"})
    assert sorted(edits) == [(0, 3, "Ann", "Zoe"), (8, 11, "Bob", "Max"), (13, 16, "Ann", "Zoe")]
    for start, end, original, _ in edits:
        assert text[start:end] == original
    assert mod.splice(text, edits) == "Zoe met Max. Zoe left."


@pytest.mark.parametrize("mod", [root_util, lib_entities], ids=["root", "library"])
def test_E1_whole_token_repeats_only(mod):
    text = "Ann filed the Annual report; Ann signed."
    ents = [_ent(mod, "Ann", 0)]
    out = mod.apply_entity_surrogates(text, ents, {"Ann": "Zoe"})
    assert out == "Zoe filed the Annual report; Zoe signed."


@pytest.mark.parametrize("mod", [root_util, lib_entities], ids=["root", "library"])
def test_E1_overlapping_entities_keep_longest(mod):
    text = "Mail Ann Lee now"
    ents = [_ent(mod, "Ann", 5), _ent(mod, "Ann Lee", 5)]
    edits = mod.plan_substitutions(text, ents, {"Ann": "Zoe", "Ann Lee": "Max Roe"})
    assert edits == [(5, 12, "Ann Lee", "Max Roe")]
