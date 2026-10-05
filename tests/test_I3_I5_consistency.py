"""Audit I3 / I5 — one surrogate per original across a conversation, and a
long conversation's map never corrupts later answers (gate J4).

Before: a case variant of an earlier original got a new surrogate (I3), and
age surrogates such as "70"→"72" from early turns rewrote every later
"72" — 24 of 250 echo turns were corrupted (bench/echo.py, I5).
"""

import re

import pytest

from surrogateshield.core.consistency import (
    assign_surrogates, glued, is_low_entropy, occurs, quoted_back, strip_clitic,
)
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen
from surrogateshield.core.storage.shadow_map import ShadowMap


def _ents(text, values, type_="PERSON"):
    return [DetectedEntity(m.group(), m.start(), m.end(), type_, 0.9, "t")
            for v in values for m in re.finditer(re.escape(v), text)]


def _shadow(pairs=None):
    sh = ShadowMap(f"t-{id(pairs)}-{len(pairs or {})}", storage_dir=None)
    sh.update(pairs or {})
    return sh


# ── helpers ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["72", "8", "70 years old", "45-year-old", "female",
                                   "Gender: non-binary", "she/her", "M"])
def test_I5_low_entropy_values(value):
    assert is_low_entropy(value)


@pytest.mark.parametrize("value", ["Sarah Mitchell", "85004", "602-555-0143", "Leeds"])
def test_I5_not_low_entropy(value):
    assert not is_low_entropy(value)


def test_I5_quoted_back_excludes_low_entropy():
    assert quoted_back({"72", "female", "Ann Lee", "85004"}) == {"Ann Lee", "85004"}


def test_I4_strip_clitic():
    assert strip_clitic("Mia Lopez's") == "Mia Lopez"
    assert strip_clitic("Jennings'") == "Jennings"
    assert strip_clitic("Mia Lopez") == "Mia Lopez"


def test_E2_glued_cjk_is_not_a_word_neighbour():
    text = "收件人李娜邮箱"
    i = text.index("李娜")
    assert not glued(text, i, i + 2)
    assert occurs("Ask Sarah.", "sarah") and not occurs("Sarahs", "Sarah")


# ── assignment ───────────────────────────────────────────────────────────────

def test_I3_case_variant_of_earlier_original_reuses_surrogate():
    sh = _shadow({"Douglas Riley": "Sarah Mitchell"})
    text = "later SARAH MITCHELL signed and sarah mitchell left"
    m = assign_surrogates(_ents(text, ["SARAH MITCHELL", "sarah mitchell"]), text, MimicGen(seed=1),
                          [sh])
    assert m == {"SARAH MITCHELL": "DOUGLAS RILEY", "sarah mitchell": "douglas riley"}


def test_I3_case_variants_in_one_message_share_one_surrogate():
    text = "Sarah Mitchell asked; SARAH MITCHELL signed; sarah mitchell left."
    m = assign_surrogates(_ents(text, ["Sarah Mitchell", "SARAH MITCHELL", "sarah mitchell"]),
                          text, MimicGen(seed=2), [_shadow()])
    base = m["Sarah Mitchell"]
    assert m["SARAH MITCHELL"] == base.upper() and m["sarah mitchell"] == base.lower()
    assert len(set(m.values())) == 3                      # each maps back to one original


def test_I3_case_form_already_taken_gets_its_own_surrogate():
    # "Sarah mitchell" has no case form distinct from "Sarah Mitchell"'s surrogate
    text = "Sarah Mitchell and Sarah mitchell"
    m = assign_surrogates(_ents(text, ["Sarah Mitchell", "Sarah mitchell"]), text, MimicGen(seed=3),
                          [_shadow()])
    assert m["Sarah Mitchell"] != m["Sarah mitchell"]


def test_I5_low_entropy_surrogate_not_reused_when_in_message():
    sh = _shadow({"72": "70"})
    text = "I was 70 then; now my father is 72."
    ents = [DetectedEntity("70", 6, 8, "age", 0.9, "t")]
    m = assign_surrogates(ents, text, MimicGen(seed=4), [sh])
    assert m["70"] != "72" and not occurs(text, m["70"])


def test_I5_generator_avoids_values_already_in_message():
    # age candidates are 70±1..3; 69, 71 and 72 are taken by the message
    text = "Ages 71, 72 and 69; mine is 70."
    i = text.rindex("70")
    ents = [DetectedEntity("70", i, i + 2, "age", 0.9, "t")]
    for seed in range(30):
        assert MimicGen(seed=seed).generate_all(ents, text=text)["70"] in {"67", "68", "73"}


def test_I5_generator_falls_back_when_every_candidate_is_taken(caplog):
    text = "Ages 67, 68, 69, 71, 72, 73; mine is 70."
    i = text.rindex("70")
    m = MimicGen(seed=1).generate_all([DetectedEntity("70", i, i + 2, "age", 0.9, "t")], text=text)
    assert m["70"] != "70" and "already used in this message" in caplog.text


def test_E2_masking_and_restoring_share_one_boundary_rule():
    from surrogateshield.core.entities import apply_entity_surrogates

    text = "aged 8, ran 8.5 km; 1,800 fans; Tom Okafor-Smith"
    ents = [DetectedEntity("8", 5, 6, "age", 0.9, "t"),
            DetectedEntity("Tom Okafor", text.index("Tom"), text.index("Tom") + 10, "PERSON", 0.9, "t")]
    sent = apply_entity_surrogates(text, ents, {"8": "5", "Tom Okafor": "Aaron Patterson"})
    assert sent == "aged 5, ran 8.5 km; 1,800 fans; Aaron Patterson-Smith"


# ── J4: multi-turn echo, model-free ─────────────────────────────────────────

NAMES = ["Sarah Mitchell", "SARAH MITCHELL", "sarah mitchell", "Lee", "Tom Okafor"]
_DETECT = [
    (re.compile("|".join(map(re.escape, NAMES))), "PERSON"),
    (re.compile(r"[\w.+-]+@[\w-]+\.\w+"), "email"),
    (re.compile(r"\b\d{3}-\d{3}-\d{4}\b"), "phone_us"),
    (re.compile(r"(?<=aged )\d{1,3}\b"), "age"),
    (re.compile(r"\b(?:female|male)\b"), "gender_indicator"),
    (re.compile(r"\b8\d{4}\b"), "zip_us"),
]

MESSAGES = [
    "Sarah Mitchell asked for the file; later SARAH MITCHELL signed it and sarah mitchell left.",
    "Lee flew to Leeds. Lee is aged 70 and female.",
    "Email sarah@corp.com or call 602-555-0143 about zip 85004.",
    "My grandfather is 72 and ran 8.5 km; 1,800 people watched. Tom Okafor is aged 72.",
    "first,last,phone,zip\nTom Okafor,602-555-0143,85004\nLee,480-555-0199,85251",
    "Tom Okafor's report: Tom Okafor-Smith is a different person.",
    "The male nurse said 70 is fine. Sarah Mitchell agreed.",
    "Quote: sarah@corp.com, 85004, Lee, Tom Okafor.",
]


def fake_cascade(text, skip_values=None, skip_location_entities=False, **_):
    skip = skip_values or set()
    found, taken = [], []
    for pat, type_ in _DETECT:
        for m in pat.finditer(text):
            if m.group() in skip or glued(text, m.start(), m.end()):
                continue
            if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append(DetectedEntity(m.group(), m.start(), m.end(), type_, 0.95, "fake"))
    return found, []


def test_J4_echo_round_trip_model_free():
    from bench.echo import run

    result = run(MESSAGES, 40, cascade=fake_cascade)
    assert result["_corrupted"] == []
    assert result["originals_with_several_surrogates"] == 0
    assert result["surrogate_equals_original"] == []
    assert result["J4"] == "PASS"


def test_I3_same_person_same_surrogate_across_turns():
    from bench.echo import run

    result = run(MESSAGES, 24, cascade=fake_cascade)
    assert result["originals_with_several_surrogates"] == 0


def test_I5_session_unmask_restores_low_entropy_only_from_last_mask(monkeypatch):
    import surrogateshield as ss
    from surrogateshield.core.detection import pipeline

    monkeypatch.setattr(pipeline, "run_cascade", fake_cascade)
    s = ss.Session()
    sent = s.mask("Lee is aged 70.")
    age = re.search(r"aged (\d+)", sent).group(1)
    assert s.unmask(sent) == "Lee is aged 70."
    s.mask("Tom Okafor called.")
    # the earlier age surrogate is an ordinary number in a later answer
    assert s.unmask(f"Lee turns {age} soon.") == f"Lee turns {age} soon."


@pytest.mark.heavy
def test_J4_echo_250_turns_real_models():
    from bench.echo import DEFAULT_SOURCES, load_messages, run

    result = run(load_messages(DEFAULT_SOURCES), 250)
    assert result["_corrupted"] == [], result["_corrupted"][:3]
    assert result["J4"] == "PASS"


def test_J2_surrogate_never_contains_an_original_of_the_message():
    text = "From: Daniel Kowalczyk. Can you thank Daniel? Ask Sarah too."
    ents = [DetectedEntity(v, text.index(v), text.index(v) + len(v), "PERSON", 0.9, "t")
            for v in ("Daniel Kowalczyk", "Daniel", "Sarah")]
    for seed in range(200):
        m = MimicGen(seed=seed).generate_all(ents, text=text)
        for surrogate in m.values():
            assert not occurs(surrogate, "Daniel") and not occurs(surrogate, "Sarah"), (seed, m)


def test_J2_surrogate_never_contains_an_earlier_original():
    ents = [DetectedEntity("Mia Lopez", 0, 9, "PERSON", 0.9, "t")]
    for seed in range(200):
        s = MimicGen(seed=seed).generate_all(ents, forbidden={"Daniel", "Sarah"})["Mia Lopez"]
        assert not occurs(s, "Daniel") and not occurs(s, "Sarah"), (seed, s)


def test_J2_structured_original_never_embedded_in_surrogate():
    text = "[Sarah](https://acme.com/team/sarah)"
    url = "https://acme.com/team/sarah"
    ents = [DetectedEntity("Sarah", 1, 6, "PERSON", 0.9, "t"),
            DetectedEntity(url, text.index(url), text.index(url) + len(url), "url", 0.9, "t")]
    for seed in range(200):
        m = MimicGen(seed=seed).generate_all(ents, text=text)
        assert url not in m[url].lower(), (seed, m)


def test_J2_surrogate_never_contains_its_own_original():
    ents = [DetectedEntity("Will", 0, 4, "PERSON", 0.9, "t")]
    for seed in range(300):
        assert "will" not in MimicGen(seed=seed).generate_all(ents)["Will"].lower()


def test_J4_surrogate_never_equals_an_earlier_original(monkeypatch):
    # forbidden used to bind addresses only: a dob surrogate could equal an
    # earlier turn's real date of birth
    draws = iter(["11/02/1987", "03/14/1990"])
    monkeypatch.setattr(MimicGen, "generate", lambda self, ent, **kw: next(draws))
    ent = DetectedEntity("05/16/1987", 0, 10, "dob", 0.9, "t")
    m = MimicGen(seed=5).generate_all([ent], forbidden={"11/02/1987"})
    assert m == {"05/16/1987": "03/14/1990"}
