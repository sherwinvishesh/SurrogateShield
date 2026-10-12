"""The ensemble fallback (entity_trace ``options.ensemble``, P2-C1): the
tagger misses a natural name mostly by never proposing it, a plain
well-formed name in running text. spaCy reads those, and its PERSON stands
in on a message where no stage named anyone, when the name is tied to a
person the message talks about (``pipeline._tied_fallback``). Model-free:
spaCy is stood in for by a stub; texts and names are invented."""

import pytest

from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pipeline, plugins, structural
from surrogateshield.core.entities import DetectedEntity

NAMES = ("Ravi Thandeka", "Okonkwo Adebayo", "Mira Qeli", "Sol Ibarra")
KW = dict(use_context_guard=False, use_tagger=False)

TIED = [
    "Please tell Ravi Thandeka that the keys are with the neighbour.",
    "Ravi Thandeka said he would pick up the kids at six.",
    "Ravi Thandeka's mom is coming on Sunday, can you draft a note?",
    "I saw Dr Okonkwo Adebayo yesterday about the test results.",
    "This is from Mira Qeli, she asked me to forward it.",
    "My sister Mira Qeli is turning thirty next week.",
]
UNTIED = [
    "Hi Ravi Thandeka, thanks for the quick reply.",                 # the addressee
    "Dear Mira Qeli, your order has shipped.",                       # the addressee
    "Ravi Thandeka",                                                 # a bare name
    "Explain the theory of Ravi Thandeka in simple terms.",         # no personal cue
    "The Okonkwo Adebayo method of proof is taught in year two.",   # no personal cue
    "tell ravi thandeka that the keys are with the neighbour.",     # not capitalised
]
FICTION = ("???: \"Who goes there?\"\n"
           "Mara: \"A friend. Tell Mira Qeli I am late.\"\n"
           "???: \"Friends knock.\"\n"
           "Mara: \"Then I'll knock.\"\n")


def _person(text, name, score=0.88):
    s = text.index(name)
    return DetectedEntity(name, s, s + len(name), "PERSON", score, "ner")


def _in(text):
    return [n for n in NAMES if n in text]


def _stub_trace(names):
    def trace(remaining_text, existing_entities, spacy_model, high_threshold, low_threshold):
        return [_person(remaining_text, n) for n in names], []
    return trace


# ── the tie ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", TIED)
def test_a_tied_name_stands_in(text):
    cands = [_person(text, n) for n in _in(text)]
    assert pipeline._tied_fallback(text, cands) == cands


@pytest.mark.parametrize("text", UNTIED)
def test_an_untied_name_does_not(text):
    cands = [_person(text, n.lower() if n.lower() in text else n) for n in NAMES
             if n in text or n.lower() in text]
    assert cands and pipeline._tied_fallback(text, cands) == []


@pytest.mark.parametrize("text, name, tied", [
    ("Explain Thandeka's theorem in simple terms.", "Thandeka", False),      # an eponym: a bare possessive alone
    ("What do we do for Thandeka's day this year?", "Thandeka", False),      # a holiday
    ("Thandeka's mom is coming on Sunday, can you draft a note?", "Thandeka", True),   # a personal possession
    ("Please tell Thandeka's team we are running late.", "Thandeka", True),  # a cue before it
    ("Ravi Thandeka's theorem is taught in year two.", "Ravi Thandeka", True),   # a full name under a possessive
    ("Can you analyse Thandeka LLC's latest filing for me?", "Thandeka LLC", False),  # a company
    ("Can you analyse Thandeka Partners's filing for me?", "Thandeka Partners", False),
])
def test_a_bare_possessive_ties_only_a_full_name(text, name, tied):
    cands = [_person(text, name)]
    assert (pipeline._tied_fallback(text, cands) == cands) is tied


def test_not_in_a_fiction_frame():
    assert structural._fiction_labels(FICTION)
    cands = [_person(FICTION, "Mira Qeli")]
    assert pipeline._tied_fallback(FICTION, cands) == []
    plain = "Tell Mira Qeli I am late."
    assert pipeline._tied_fallback(plain, [_person(plain, "Mira Qeli")])


def test_at_most_three_names_in_the_message():
    # P3: three since phase 3 (two before)
    three = "Tell Ravi Thandeka, tell Mira Qeli and tell Sol Ibarra the news."
    four = "Tell Ravi Thandeka, tell Mira Qeli, tell Sol Ibarra and tell Okonkwo Adebayo the news."
    assert pipeline._FALLBACK_MAX_PERSONS == 3
    assert len(pipeline._tied_fallback(three, [_person(three, n) for n in _in(three)])) == 3
    assert pipeline._tied_fallback(four, [_person(four, n) for n in NAMES]) == []


def test_other_types_pass_as_given():
    text = "Lagos is far from here."
    place = DetectedEntity("Lagos", 0, 5, "GPE", 0.85, "ner")
    assert pipeline._tied_fallback(text, [place]) == [place]


# ── the config ───────────────────────────────────────────────────────────────

def test_the_default_reads_persons_back_at_spacys_score():
    assert C.ENTITY_TRACE_ENSEMBLE == {"PERSON": 0.88}
    assert C.preset("balanced").stage("entity_trace").options["ensemble"] == C.ENTITY_TRACE_ENSEMBLE
    assert not C.preset("balanced").allows("entity_trace", "PERSON")       # still not a PERSON source
    off = C.preset("balanced").with_stage("entity_trace", options={"ensemble": {}})
    assert off.stage("entity_trace").options["ensemble"] == {} and off.config_hash() != C.preset("balanced").config_hash()
    assert C.DetectionConfig.from_dict(off.to_dict()) == off


# ── the cascade ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", TIED)
def test_a_tied_spacy_name_is_confirmed_when_no_stage_named_anyone(monkeypatch, text):
    names = _in(text)
    monkeypatch.setattr(pipeline.entity_trace, "trace", _stub_trace(names))
    conf, _ = pipeline.run_cascade(text, **KW)
    assert [(e.text, e.type, e.source) for e in conf if e.type == "PERSON"] == [(n, "PERSON", "ner") for n in names]


@pytest.mark.parametrize("text", UNTIED[:5])
def test_an_untied_spacy_name_is_not(monkeypatch, text):
    names = _in(text)
    monkeypatch.setattr(pipeline.entity_trace, "trace", _stub_trace(names))
    conf, pending = pipeline.run_cascade(text, **KW)
    assert not [e for e in conf + pending if e.source == "ner"]


def test_off_by_option_or_without_spacy(monkeypatch):
    text = TIED[0]
    monkeypatch.setattr(pipeline.entity_trace, "trace", _stub_trace(_in(text)))
    off = C.preset("balanced").with_stage("entity_trace", options={"ensemble": {}})
    conf, _ = pipeline.run_cascade(text, config=off, **KW)
    assert not [e for e in conf if e.source == "ner"]
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, **KW)
    assert not [e for e in conf if e.source == "ner"]


def test_a_stage_that_named_someone_keeps_spacy_out(monkeypatch):
    text = "Please tell Ravi Thandeka that Mira Qeli has the keys."

    class Names:
        def detect(self, text, view):
            s = text.index("Mira Qeli")
            return [plugins.Candidate(s, s + len("Mira Qeli"), "PERSON", 0.95)]
    monkeypatch.setattr(pipeline.entity_trace, "trace", _stub_trace(["Ravi Thandeka"]))
    plugins.register_detector("names", lambda stage: Names(), replace=True)
    try:
        cfg = C.preset("balanced").with_plugin("names", types=("PERSON",))
        conf, _ = pipeline.run_cascade(text, config=cfg, **KW)
        assert {e.text for e in conf if e.type == "PERSON"} == {"Mira Qeli"}
        assert not [e for e in conf if e.source == "ner"]
    finally:
        plugins.unregister_detector("names")


def test_the_trace_marks_the_stage(monkeypatch):
    text = TIED[1]
    monkeypatch.setattr(pipeline.entity_trace, "trace", _stub_trace(_in(text)))
    trace = []
    pipeline.run_cascade(text, trace=trace, **KW)
    stages = [t["stage"] for t in trace]
    i = stages.index("ensemble")
    assert stages[i - 1] == "context_guard"
    before = [r for r in trace[i - 1]["entities"] if r[2] == "PERSON"]
    after = [r for r in trace[i]["entities"] if r[2] == "PERSON"]
    assert not before and len(after) == 1
