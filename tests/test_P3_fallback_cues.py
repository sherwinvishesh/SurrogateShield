"""Campaign phase 3: the ensemble fallback (``pipeline._tied_fallback``)
reads ", who" and "and his/her/their" after a name as a personal
continuation, and a dash that opens a line as a sign-off only when the
name ends the line. Model-free; texts and names are invented."""

import pytest

from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity


def _person(text, name, score=0.88):
    s = text.index(name)
    return DetectedEntity(name, s, s + len(name), "PERSON", score, "ner")


@pytest.mark.parametrize("text, name", [
    ("The new hire is Tomasz Wierzba, who starts on Monday in the back office.", "Tomasz Wierzba"),
    ("We stayed with Ilse Varga and her two dogs over the weekend.", "Ilse Varga"),
    ("Last year Ilse Varga and their team moved the office.", "Ilse Varga"),
    ("Thanks for the help.\n— Ilse Varga", "Ilse Varga"),
    ("Thanks for the help.\n- Tomasz", "Tomasz"),
])
def test_a_personal_continuation_or_a_sign_off_ties_the_name(text, name):
    cands = [_person(text, name)]
    assert pipeline._tied_fallback(text, cands) == cands


@pytest.mark.parametrize("text, name", [
    ("Steps:\n- Tomasz Wierzba reviews the logs\n- the team ships", "Tomasz Wierzba"),   # a bullet
    ("Checklist:\n- LOGS checked before the release", "LOGS"),
    ("The Varga Prize, which is given every year, went to a chemist.", "Varga Prize"),
])
def test_a_bullet_or_a_bare_clause_does_not(text, name):
    assert pipeline._tied_fallback(text, [_person(text, name)]) == []
