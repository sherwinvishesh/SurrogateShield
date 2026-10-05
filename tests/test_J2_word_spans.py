"""Gate J2 — spans that are words of the message, not names: a title alone
("Mr" of "Mr. Okonkwo-Hale"), a sentence-initial imperative before an object
pronoun ("Ping me"), a column name in a table header, a contract's defined
term ("Tenant"), and a possessive phrase ("mera account"). Each was a
spurious edit. Model-free except the last test, which runs the structural
detector only.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.detection import structural
from surrogateshield.core.entities import DetectedEntity


def junk(text, span, source="ner"):
    i = text.index(span)
    return rg.is_junk(DetectedEntity(span, i, i + len(span), "PERSON", 0.9, source), text)


TABLE = "turn this into a table\nName,Age,Allergy,Parent phone\nKai Nakamura,8,peanuts,0412 555 019"
LEASE = 'between Tom Hale ("Landlord"), and Rafael Dias ("Tenant"). The Tenant shall not keep pets.'
HI = "bhai mera naam Rohit Deshmukh hai, HDFC ka number chahiye, mera account number hai"


@pytest.mark.parametrize("text,span", [
    ("Need to email Mr. Okonkwo-Hale about it.", "Mr"),
    ("Should I ask Dr. Siddiqui about it?", "Dr"),
    ("Use the v3 template. Ping me on +358 40 718 2265 if blocked.", "Ping"),
    ("Het adres klopt niet. Bel me even op 06-41982277. Groet, Femke en de rest van het team", "Bel"),
    (TABLE, "Allergy"),
    (LEASE, "Tenant"),
    (HI, "mera"),
    (HI, "mera account"),
])
def test_J2_word_span_dropped(text, span):
    assert junk(text, span)


@pytest.mark.parametrize("text,span", [
    (TABLE, "Kai Nakamura"),
    ("Maria,Sales,Berlin\nJon,Ops,Leeds", "Maria"),          # no header row
    (LEASE, "Rafael Dias"),
    ("Ask Ben to call me tomorrow.", "Ben"),
    ("Hola, necesito ayuda con una carta. Anna me dijo que no es posible y que el contrato es de mi casero.",
     "Anna"),                                                # Romance clitic: Anna is the subject
    ("hi my sarah is sick", "my sarah"),                     # all lower case: kept (fail closed)
])
def test_J2_word_span_name_kept(text, span):
    assert not junk(text, span)


def test_J2_verb_frame_skips_titles():
    text = "Need to email Mr. Okonkwo-Hale and ask Dr. Siddiqui. Ping Kev too."
    added, _ = structural._verb_frames(text, [])
    assert [e.text for e in added] == ["Kev"]
