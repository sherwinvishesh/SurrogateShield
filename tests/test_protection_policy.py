"""What may reach the provider verbatim — gate J1, audit I22, plus the
generator and geo-filter fixes the J1 run exposed.

Model-free: Prepared objects and entity lists are built by hand.
"""

import random

import pytest

import json_tester as jt
import util
from detection import service_query as root_sq
from detection.logic import _filter_topical_geo_entities, _has_personal_anchor
from generation.logic import MimicGen, _swap_gender
from offline_eval import POLICY_REASONS, check_sent_text
from surrogateshield.core.detection import pipeline as lib_pipeline
from surrogateshield.core.detection import service_query as lib_sq


def _prep(question, ents, mapping, mode, skipped=(), skip_reason="topical_geo_filtered"):
    edits = util.plan_substitutions(question, ents, mapping)
    return jt.Prepared(
        question=question, is_service_query=mode == "shift", address_mode=mode,
        confirmed=list(ents), skipped=list(skipped), skip_reason=skip_reason,
        surrogate_map=mapping, edits=edits, sanitized=util.splice(question, edits),
    )


def _e(text, question, typ):
    start = question.index(text)
    return util.DetectedEntity(text=text, start=start, end=start + len(text), type=typ)


ADDR_Q = "Coffee near 12 Elm St, Tempe, AZ please"
ADDR = "12 Elm St, Tempe, AZ"


# ── I22: address policy ──────────────────────────────────────────────────────

@pytest.mark.parametrize("sq", [root_sq, lib_sq], ids=["root", "library"])
def test_I22_auto_shifts_only_service_queries(sq):
    assert sq.resolve_address_mode("auto", True) == "shift"
    assert sq.resolve_address_mode("auto", False) == "replace"
    assert sq.resolve_address_mode("replace", True) == "replace"
    # a sensitive service query is never shifted (±1 keeps the real street);
    # I6 coarsens it to "my area" + city/state instead
    q = "Find a free legal aid clinic for undocumented immigrants near 12 Elm St, Tempe, AZ"
    assert sq.resolve(q, "auto") == (True, "coarse")
    assert sq.resolve_address_mode("auto", True, sensitive=True) == "coarse"


def test_I22_default_address_mode_is_auto():
    import config
    from surrogateshield._state import cfg
    assert config.ADDRESS_MODE == "auto"
    assert cfg.address_mode == "auto"


# ── J1: leak classification ──────────────────────────────────────────────────

def test_J1_shift_policy_components_are_deliberate():
    prep = _prep(ADDR_Q, [_e(ADDR, ADDR_Q, "address")], {ADDR: "13 Elm St, Tempe, AZ"}, "shift")
    chk = check_sent_text(ADDR_Q, {"address": ADDR, "gpe": ["Tempe", "AZ"]}, prep)
    assert chk["shift_mismatch"] == []
    assert {l["value"] for l in chk["leaks"]} == {"Tempe", "AZ"}
    assert all(l["deliberate"] and l["reasons"] == ["service_query_address_shift"]
               for l in chk["leaks"])


def test_J1_same_components_outside_shift_mode_are_unintended():
    # replace mode that (wrongly) kept the city: no policy covers it
    prep = _prep(ADDR_Q, [_e(ADDR, ADDR_Q, "address")], {ADDR: "98 Oak Ave, Tempe, AZ"}, "replace")
    chk = check_sent_text(ADDR_Q, {"gpe": "Tempe"}, prep)
    assert [l["deliberate"] for l in chk["leaks"]] == [False]


def test_J1_shift_mismatch_detected():
    prep = _prep(ADDR_Q, [_e(ADDR, ADDR_Q, "address")], {ADDR: "13 Elm St, Tempe, AZ"}, "shift")
    prep.sanitized = ADDR_Q                     # the shifted address never made it out
    assert check_sent_text(ADDR_Q, {}, prep)["shift_mismatch"] == [ADDR]


def test_J1_unreplaced_without_policy_reason_is_unintended():
    q = "Ann says gender: male here"
    g = _e("gender: male", q, "gender_indicator")
    # surrogate equal to the original → edit is a no-op, no reason recorded
    prep = _prep(q, [g], {"gender: male": "gender: male"}, "replace")
    leaks = check_sent_text(q, {"gender": "gender: male"}, prep)["leaks"]
    assert leaks and not leaks[0]["deliberate"]
    assert leaks[0]["reasons"] == ["no_reason_recorded"]
    assert "no_reason_recorded" not in POLICY_REASONS


# ── Generator: a surrogate never equals an original ──────────────────────────

def test_J4_gender_surrogate_keeps_form_and_differs():
    rng = random.Random(0)
    for text in ("gender: male", "she/her", "identifies as non-binary", "Female", "sex: female"):
        for _ in range(20):
            out = _swap_gender(text, rng)
            assert out.lower() != text.lower()
    assert _swap_gender("gender: male", rng).startswith("gender: ")
    assert _swap_gender("she/her", rng) in ("he/him", "they/them")
    assert _swap_gender("identifies as non-binary", rng).startswith("identifies as ")


@pytest.mark.parametrize("seed", range(30))
def test_J4_generate_all_never_returns_an_original(seed):
    q = "gender: male, she/her, Ann"
    ents = [_e("gender: male", q, "gender_indicator"), _e("she/her", q, "gender_indicator"),
            _e("Ann", q, "PERSON")]
    mapping = MimicGen(seed=seed).generate_all(ents, address_mode="replace")
    originals = {e.text.lower() for e in ents}
    assert not any(v.lower() in originals for v in mapping.values())


# ── Topical geo: a person's place is not a topic ─────────────────────────────

@pytest.mark.parametrize("filt", [_filter_topical_geo_entities, lib_pipeline._filter_topical_geo_entities],
                         ids=["root", "library"])
def test_J1_topical_geo_kept_when_person_anchored(filt):
    q = "Help me find the LinkedIn profile for Carlos Mendez in Naperville."
    geo = _e("Naperville", q, "GPE")
    person = _e("Carlos Mendez", q, "PERSON")
    assert _has_personal_anchor([person, geo])
    kept, skipped = filt([person, geo], q, True)
    assert geo in kept and skipped == []
    # without a person, the same query clause makes the place a topic
    q2 = "What is the population of Naperville?"
    geo2 = _e("Naperville", q2, "GPE")
    kept2, skipped2 = filt([geo2], q2, False)
    assert skipped2 == [geo2]
