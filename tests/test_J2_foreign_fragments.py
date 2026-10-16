"""Gate J2 — an English NER model reading German, French, Spanish, Italian,
Portuguese or Dutch marks clause fragments as names ("Kannst du diese",
"Mi aiuti", "Der Vertrag"). In a message whose function words are mostly
of one of those languages (spaCy's stop-word lists), a model span made of
function words is dropped. A capitalised word that is no function word
keeps the span (fail closed). Model-free: entities are built by hand, as
NER would emit them.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def junk(text, span, type_="PERSON", source="ner"):
    i = text.index(span)
    return rg.is_junk(DetectedEntity(span, i, i + len(span), type_, 0.9, source), text)


DE = ("Hallo Herr Kowalski, die Heizung in der Wohnung ist seit Montag kaputt. "
      "Kannst du mir helfen? Ich bin Mitglied bei Kraftwerk Fitness. "
      "Der Vertrag läuft bis März. Rufen Sie mich an.")
IT = "Ciao, Mi aiuti a scrivere una mail al mio medico? Ho bisogno di un certificato. Sono Sara."
PT = "Preciso de reclamar porque me cobraram duas vezes. Ele nasceu em 1990, sou a Ana Souza."
NL = "Hallo, ik ben Ben en kun je een bezwaarbrief aan de gemeente schrijven voor mij?"
FR = "Bonjour, je dois écrire une lettre pour l'absence de mon fils. Mon fils est malade."
HI = "bhai mera naam Rohit Deshmukh hai, Nagpur se. card block karna hai aur jaldi"


@pytest.mark.parametrize("text,lang", [(DE, "de"), (IT, "it"), (PT, "pt"), (NL, "nl"),
                                       (FR, "fr"), (HI, None),
                                       ("Can you help me write to my landlord?", None)])
def test_J2_message_language(text, lang):
    assert rg.message_language(text) == lang


@pytest.mark.parametrize("text,span", [
    (DE, "Heizung"), (DE, "mir helfen"), (DE, "Der Vertrag"), (DE, "Rufen Sie"),
    (IT, "Mi aiuti"), (IT, "al mio medico"), (IT, "Ho bisogno di"),
    (PT, "Preciso de reclamar"), (PT, "Ele nasceu"),
    (NL, "een bezwaarbrief aan de gemeente"),
    (FR, "pour l'absence de mon"), (FR, "Mon"),
])
def test_J2_foreign_fragment_dropped(text, span):
    assert junk(text, span)


@pytest.mark.parametrize("text,span", [
    (DE, "Kowalski"),
    (DE, "Ich bin Mitglied bei Kraftwerk"),      # a name after "bei": keep the span
    (PT, "Ana Souza"),
    (IT, "Sara"),                                # a stop word that is a given name
    (NL, "Ben"),
    (HI, "Nagpur se"),                           # romanised Hindi is not Italian
])
def test_J2_foreign_name_kept(text, span):
    assert not junk(text, span)


def test_J2_foreign_rule_skips_pattern_spans():
    assert not junk(DE, "Kraftwerk Fitness", "ORG", "pattern")


def test_J2_english_message_untouched():
    t = "Mi casa es su casa, said Ben to my landlord Hans Müller in Leeds yesterday."
    assert rg.message_language(t) is None
    assert not junk(t, "Hans Müller")
