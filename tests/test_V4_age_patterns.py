"""V4 §3.1 B1 and B2: ages next to a name.

B1: a capitalised two- or three-part name, a comma and two digits closing
the line ("Ana Ruiz, 34⏎", a sign-off), beside the earlier mid-sentence
("Ana Ruiz, 34, MSc") and bracket ("Ana Ruiz (34)") forms. It was motivated
by test-2's misses (four values of that shape) and is disclosed as such.
B2: the name-bearing age rules read Latin letters beyond ASCII ("Özge",
"Ștefan", "Łukasz", "İlkay"), capitals and small letters kept apart.
Every name and value is invented. Model-free."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


def ages(text):
    ents = ps.scan(text)
    for e in ents:
        assert text[e.start:e.end] == e.text
    return [e.text for e in ents if e.type == "age"]


# ── B1: the number closes the line ───────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("Thanks for reading,\nAna Ruiz, 34", ["34"]),
    ("Best,\nTomás Ferreira-Lima, 52.\nLisbon", ["52"]),
    ("— Kwame Asante Boateng, 27\n", ["27"]),
    ("Signed by Ana Ruiz, 34)", ["34"]),
    ("Ana Ruiz, 34   \nsecond line", ["34"]),
    ("Ana Ruiz, 34\nPeople Ops lead, 12 years at the firm", ["34"]),   # a unit word on the next line
    ("Our intern Folake Mensah (45) starts November 19.", ["45"]),            # the bracket form stays
    ("About me: Ana-Maria Popescu, 29, MSc from Politehnica", ["29"]),         # and the mid-sentence one
])
def test_B1_a_named_age_at_a_line_end(text, expected):
    assert ages(text) == expected


@pytest.mark.parametrize("text", [
    "Upgrade to Windows Server, 12",
    "Meet me in Room 4, 12",
    "Read Chapter Two (12)",
    "I still use an iPhone 15, 12",
    "Final score: Arsenal 2, Chelsea 1",
    "Manchester United, 3",                 # a single digit at a line end: a score, a rank
    "Harry Potter, 7",
    "Total Price, 45.99",
    "Meeting Room B, 10:30",
    "Monday Standup, 09:15\nWeekly Review, 14:00",
    "Step One, 12\nStep Two, 14",
    "Item Count, 12",
    "Doors Open, 19 pm",
    "Ana Ruiz, 34 points",
    "New York, 10",
    "Ana Ruiz, 34 and counting more text",  # not the line's end: the mid-line form needs its comma
    "Ana Ruiz,\n34",                         # the number on its own line
])
def test_B1_near_misses(text):
    assert ages(text) == []


# ── B2: Latin letters beyond ASCII in every name-bearing age rule ───────────

@pytest.mark.parametrize("text, expected", [
    ("Thanks,\nÖzge Yıldız, 34", ["34"]),                       # B1 + B2
    ("— Ștefan Popescu, 41.", ["41"]),
    ("Łukasz Nowak, 29\nWarsaw", ["29"]),
    ("Jürgen Müller, 38, joins us on Monday", ["38"]),           # _NAMED_AGE, mid-sentence
    ("Our intern Zoë Lefèvre (45) starts Monday", ["45"]),       # _NAMED_AGE, bracket
    ("my grandma Ülkü is 89", ["89"]),                          # kin + name
    ("Şebnem is 13 and loves chess", ["13"]),                   # name + is + number
    ("(gülşen, 63)", ["63"]),                                   # quoted or bracketed name, number
    ("My daughter Ayşe Çelik (14)", ["14"]),                    # kin + name + (number)
    ("İlkay's 80th birthday is next week", ["80th"]),           # possessive + ordinal
    ("Đorđe Šimić, 33", ["33"]),
])
def test_B2_latin_names(text, expected):
    assert ages(text) == expected


def test_B2_keeps_capitals_apart():
    # a small letter cannot start the name in the case-sensitive rules
    assert ages("özge yıldız, 34") == []
    assert ages("şebnem is 13 and loves chess") == []
    # a capital beyond ASCII (Turkish dotted İ) is a capital
    assert ages("İrem Aydın, 22") == ["22"]
    assert "İ" in ps._UP and "ı" in ps._LO and "ß" in ps._LO
    assert "×" not in ps._UP + ps._LO and "÷" not in ps._UP + ps._LO


@pytest.mark.parametrize("text", [
    "Chapter Über (12)",
    "Room Ölçü, 12, then",
])
def test_B2_near_misses_keep_their_guards(text):
    assert ages(text) == []
