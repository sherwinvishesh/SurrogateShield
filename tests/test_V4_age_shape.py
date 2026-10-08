"""V4 §3.3 — the shape guard for a model's ages: a tagger AGE candidate counts
only as a number from 1 to 120, in digits or words, with at most an age cue,
and no unit or count after it (the patterns' ``_NOT_AGE_AFTER``). Model-free;
every name and number is invented."""

import pytest

from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pipeline, plugins
from surrogateshield.core.detection.pattern_scan import age_shape
from surrogateshield.core.detection.plugins import Candidate


def _shape(text, span):
    i = text.index(span)
    return age_shape(text, i, i + len(span))


@pytest.mark.parametrize("text,span", [
    ("Thanks again.\nMarta Kowalczyk, 41", "41"),
    ("my nephew is 7", "7"),
    ("I'm 34 years old", "34 years old"),
    ("I'm 34 years old", "34"),
    ("she turned thirty-four", "thirty-four"),
    ("he is forty two", "forty two"),
    ("aged 63", "aged 63"),
    ("me (29F) and him", "29F"),
    ("34yo, retired", "34yo"),
    ("Age: 120", "120"),
])
def test_an_age_in_an_ages_shape_is_kept(text, span):
    assert _shape(text, span)


@pytest.mark.parametrize("text,span", [
    ("it took 12 hours", "12"),            # a unit after it
    ("about 30% of them", "30"),
    ("we ran 5 miles", "5"),
    ("12 of the 40", "12"),                # a count
    ("version 3.5", "3"),                  # a decimal
    ("5'10 tall", "5"),                    # a height
    ("the year 2023", "2023"),             # out of range
    ("aged 121", "aged 121"),
    ("score 0", "0"),
    ("a hundred", "hundred"),              # not a number we read as an age
    ("Room 4", "Room 4"),                  # a word that is no cue
    ("Chapter Two", "Chapter Two"),
])
def test_a_number_that_is_not_an_age_is_dropped(text, span):
    assert not _shape(text, span)


def test_the_cascade_drops_a_tagger_age_without_an_age_shape():
    text = "Ines Duarte, 52\nThe run took 45 minutes."
    spans = [(text.index("52"), text.index("52") + 2), (text.index("45"), text.index("45") + 2)]

    class Ages:
        def detect(self, t, view=None):
            return [Candidate(s, e, "AGE", 0.99) for s, e in spans] if t == text else []
    plugins.register_detector("pii_tagger", lambda stage: Ages(), replace=True)
    try:
        cfg = C.from_partial({"type_sources": {"AGE": ["pattern_scan", "canonicaliser", "structural", "pii_tagger"]}},
                             C.preset("no_models").with_plugin("pii_tagger", types=("AGE",)))
        conf, _ = pipeline.run_cascade(text, config=cfg)
        tagged = {e.text for e in conf if e.source == "pii_tagger"}
        assert tagged == {"52"}
    finally:
        plugins.unregister_detector("pii_tagger")
