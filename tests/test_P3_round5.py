"""Phase 3, round 5: a place too large to identify anyone, a country, a US
state or a larger region (``pipeline._misreads``, ``nation``), is read as
no identifier; a town or a city still is one. Model-free; texts and names
are invented."""

import pytest

from surrogateshield.core.detection import geo_data, pipeline
from surrogateshield.core.entities import DetectedEntity

KW = dict(use_context_guard=False, use_tagger=False, use_entity_trace=False)


def _ent(text, value, typ, source):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def _rule(text, value, typ="GPE", source="pii_tagger"):
    found = pipeline._misreads(text, [_ent(text, value, typ, source)])
    return found[0][1] if found else None


@pytest.mark.parametrize("text, value, typ, source", [
    ("I'll be based in Canada for the next two years.", "Canada", "GPE", "pii_tagger"),
    ("What are the visa rules for the United States?", "the United States", "GPE", "ner"),
    ("Is it legal to keep chickens in Ohio?", "Ohio", "GPE", "ner"),
    ("The economy of Russia slowed down last year.", "Russia", "GPE", "ner"),
    ("Moving from the UK next month, any tips?", "UK", "GPE", "pii_tagger"),
    ("We toured Southeast Asia on a budget.", "Southeast Asia", "LOC", "ner"),
    ("Canada's winters are long.", "Canada's", "GPE", "pii_tagger"),
    ("Country: Peru\nNotes: none", "Peru", "GPE", "pii_tagger"),                    # a form: still a country
    ("I grew up in Georgia and moved north.", "Georgia", "GPE", "ner"),              # a given name, after "in"
    ("We are leaving India in May.", "India", "GPE", "ner"),
])
def test_a_country_state_or_region_is_no_identifier(text, value, typ, source):
    assert _rule(text, value, typ, source) == "nation"


@pytest.mark.parametrize("text, value, typ, source", [
    ("I live in Faro with my two cats.", "Faro", "GPE", "ner"),                       # a town
    ("I live in Lagos, near the lagoon.", "Lagos", "GPE", "pii_tagger"),              # a major city
    ("City: Phoenix\nZip: 85001", "Phoenix", "GPE", "pii_tagger"),
    ("I grew up in New York and moved west.", "New York", "GPE", "ner"),             # a state, and a city
    ("We moved to Singapore last year.", "Singapore", "GPE", "pii_tagger"),         # a city-state
    ("Georgia said she would call back.", "Georgia", "GPE", "ner"),                   # a given name, no place cue
    ("Ask Jordan about the keys.", "Jordan", "GPE", "ner"),
    ("I met India at the gym.", "India", "GPE", "ner"),
    ("Canada Goose sent a refund.", "Canada Goose", "ORG", "pii_tagger"),            # a company
    ("Ohio", "Ohio", "PERSON", "pii_tagger"),                                        # not read as a place
])
def test_not_a_nation(text, value, typ, source):
    assert _rule(text, value, typ, source) != "nation"


def test_the_lists():
    assert {"canada", "russia", "south korea", "united kingdom", "vatican city"} <= geo_data.COUNTRIES
    assert "us" not in geo_data.COUNTRIES and "us" in pipeline._NATIONS          # "US" as a place only
    assert not pipeline._NATIONS & geo_data.MAJOR_CITIES                              # a city stays one
    assert {"ohio", "southeast asia"} <= pipeline._NATIONS
    assert all(n in pipeline._NATIONS for n in pipeline._NATION_GIVEN_NAMES)


def test_the_cascade_keeps_the_country_and_says_why(monkeypatch):
    text = "I moved to Canada from Faro last spring."

    def trace(remaining_text, existing_entities, spacy_model, high_threshold, low_threshold):
        return [_ent(remaining_text, "Canada", "GPE", "ner"), _ent(remaining_text, "Faro", "GPE", "ner")], []
    monkeypatch.setattr(pipeline.entity_trace, "trace", trace)
    conf, _ = pipeline.run_cascade(text, use_context_guard=False, use_tagger=False)
    assert "Canada" not in {e.text for e in conf}
    assert "nation" in conf._skip_reasons.values()
