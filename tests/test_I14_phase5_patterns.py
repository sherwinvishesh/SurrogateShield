"""Audit I1 / I8 / I14, gate J2 — PII families missed on the J2 dev2 split:
IDs labelled in other languages or a few words before the number, IBAN / VIN
with a failing checksum, config-file secrets, JSON and env user names,
ages written next to a name, share links and personal hosting domains.
Every positive case comes from the dev2 split or the synth dev split, and
every negative is a keep value or a near miss. Model-free.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    ents = ps.scan(text)
    for e in ents:
        assert text[e.start:e.end] == e.text
    return [(e.type, e.text) for e in ents]


def values(text):
    return [v for _t, v in found(text)]


# ── labelled IDs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("Il mio codice fiscale è FRRGCM85M14F257X, è corretto?", "FRRGCM85M14F257X"),
    ("mon numéro de sécurité sociale commence par 2 ? C'est 2 84 07 75 112 058 43, je veux",
     "2 84 07 75 112 058 43"),
    ("sort code 20-45-61, account 30418877", "20-45-61"),
    ("Superior Court case number 19-1-04472-3 was dismissed", "19-1-04472-3"),
    ("Applicant: Mr. Chen Jianguo; Alien Registration No. A-219-448-031", "A-219-448-031"),
    ("ORCID 0000-0002-8814-3307", "0000-0002-8814-3307"),
    ("my IRD number is 112-847-305", "112-847-305"),
    ("my employee ID (QM-20417) was deactivated", "QM-20417"),
    ("身份证号是110105199203157728", "110105199203157728"),
    ("My Kenyan ID number is 31594026.", "31594026"),
    ("steam id 76561198044728913", "76561198044728913"),
    ("mon matricule 111 248 906", "111 248 906"),
    ("my Global Entry number is 98417730", "98417730"),
    ("my FMLA case number is 26-FM-3381.", "26-FM-3381"),
])
def test_I14_labelled_ids_other_languages(text, value):
    assert value in values(text)


@pytest.mark.parametrize("text", [
    "Order #A88-310271 arrived damaged",
    "Can you tell me if the IBAN looks legit?",
    "account number 1234. The build is 2024 1101 5532",
])
def test_I8_id_near_misses(text):
    vals = values(text)
    assert "A88-310271" not in vals and "2024 1101 5532" not in vals
    assert not any(t in ("iban", "vin") for t, _ in found(text))


def test_I14_nearby_id_needs_label_on_same_line():
    assert values("my passport photo\n2 84 07 75 112 058 43") == []
    assert values("Vodafone customer number is 300417762") == ["300417762"]


def test_I14_plus_country_code_is_not_a_us_phone():
    assert ("phone_uk", "+44 116 296 4471") in found("ORCID 0000-0002-8814-3307\n+44 116 296 4471")


@pytest.mark.parametrize("text, kind, value", [
    ("His IBAN is DE27 1007 0024 0458 3311 00, name Friedrich Haas.", "iban",
     "DE27 1007 0024 0458 3311 00"),
    ("should go to IBAN DK31 5021 0007 3344 89. Best, Rasmus", "iban", "DK31 5021 0007 3344 89"),
    ("is VIN 2T1BURHE6KC218845 valid?", "vin", "2T1BURHE6KC218845"),
])
def test_I14_labelled_iban_vin_without_checksum(text, kind, value):
    assert (kind, value) in found(text)


# ── credentials ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("OTP 559 104, what now?", "559 104"),
    ("the reset code they sent was 482913 and now it says expired", "482913"),
    ("DB_HOST=10.0.3.17\nDB_USER=pkaur_admin\nDB_PASS=Wn7#vQ2r!Lp9", "Wn7#vQ2r!Lp9"),
    ("  data:\n    stripe_key: rk_live_51Hx8pQ2mZtY0vD3\nHow do I rotate this",
     "rk_live_51Hx8pQ2mZtY0vD3"),
    ("TELL KENJI THE ROUTER LOGIN IS ADMIN / HARBOR!LIGHT29 AND THE GATEWAY", "HARBOR!LIGHT29"),
    ("the keys are under the matt, the alram code is 4091. -tash", "4091"),
])
def test_I14_credentials_phase5(text, value):
    assert ("credential", value) in found(text)


@pytest.mark.parametrize("text", [
    'class M: id = Column(primary_key=True); user_id = ForeignKey(foreign_key="users.id")',
    "reset my password for Gmail please, the door code is broken",
    "the login page for admin/settings is slow",
])
def test_I8_credential_near_misses(text):
    assert not any(t == "credential" for t, _ in found(text))


def test_I8_router_login_user_is_kept():
    assert "ADMIN" not in values("THE ROUTER LOGIN IS ADMIN / HARBOR!LIGHT29")


# ── handles ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ('{"user":"rafa_ortiz_92","ip":"203.0.113.47"}', "rafa_ortiz_92"),
    ("DB_USER=pkaur_admin\nDB_PASS=x", "pkaur_admin"),
    ("my steam name is xX_Valkyr_Xx and i got banned", "xX_Valkyr_Xx"),
    ("Reddit user u/quietlark_88 doxxed me and posted on r/legaladvice", "u/quietlark_88"),
])
def test_I14_handles_phase5(text, value):
    assert ("handle", value) in found(text)


@pytest.mark.parametrize("text", ["DB_USER=root", "my steam account got hacked, what do I do",
                                  "posted on r/legaladvice"])
def test_I8_handle_near_misses(text):
    assert not any(t == "handle" for t, _ in found(text))


# ── phones and plates ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("Santiago Barrientos,sbarrientos@correo.cl,(+56) 9 7741 2208,Valparaíso", "(+56) 9 7741 2208"),
    ("they ported my number 06 1184 2297 to another provider", "06 1184 2297"),
])
def test_I14_phones_phase5(text, value):
    assert value in values(text)


def test_I8_my_number_of_is_not_a_phone():
    assert values("my number of attempts is 12345678") == []


def test_I14_lowercase_plate():
    assert ("license_plate", "yk19 rzt") in found("in leeds, my plate is yk19 rzt. can i appeal")
    assert found("my plate is empty and the tag is v2 release") == []


# ── ages ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("About me: Ana-Maria Popescu, 29, MSc from Politehnica Bucharest", ["29"]),
    ("Our intern Folake Mensah (45) starts November 19.", ["45"]),
    ("Our intern Hyun-woo D'Souza (61) starts September 17.", ["61"]),
    ("- I'm 63, ex-smoker\n- dad had a heart attack at 55", ["63", "55"]),
    ("with our two kids, aged 7 and 10. What border", ["aged 7", "10"]),
    ("my kids are aged 3, 6 and 9 years", ["aged 3", "6", "9"]),
    ("my wife and i are both 35 and thinking", ["35"]),
    ("he retired at the age of 60 and moved", ["60"]),
])
def test_I14_ages_phase5(text, expected):
    assert [v for t, v in found(text) if t == "age"] == expected


@pytest.mark.parametrize("text", [
    "See Chapter Two (12) and Room Four, 12, for details.",
    "New York, 10, Paris, 3",
    "Windows Server (19) is broken; Python Version (3)",
    "mom had lunch at 12:30",
])
def test_I8_age_near_misses(text):
    assert not any(t == "age" for t, _ in found(text))


def test_I8_age_list_stops_at_a_unit():
    assert [v for t, v in found("kids aged 7 and 10 minutes later") if t == "age"] == ["aged 7"]


# ── URLs ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("authDomain: 'tilly-wedding.firebaseapp.com' }", "tilly-wedding.firebaseapp.com"),
    ("shared doc: https://docs.google.com/document/d/1xQ7vB_Lm9pE3-kYtR2/edit?usp=sharing - it's",
     "https://docs.google.com/document/d/1xQ7vB_Lm9pE3-kYtR2/edit?usp=sharing"),
    ("Personal site: https://quentin-bakker.dev — contact me", "https://quentin-bakker.dev"),
    ("Profile: https://hiroshi-reyes.dev Twitter: @hiroshi_94", "https://hiroshi-reyes.dev"),
])
def test_I14_personal_urls_phase5(text, value):
    assert ("url", value) in found(text)


@pytest.mark.parametrize("text", [
    "See the docs at https://docs.google.com/spreadsheets/u/0/ and https://www.dropbox.com/features",
    "Our company profile is at https://acme.com and the docs at https://docs.python.org/3/",
])
def test_I1_public_urls_stay_opaque(text):
    assert found(text) == []


# ── surrogates for the new values ────────────────────────────────────────────

@pytest.mark.parametrize("kind, value", [
    ("iban", "DE27 1007 0024 0458 3311 00"), ("vin", "2T1BURHE6KC218845"),
    ("id_number", "2 84 07 75 112 058 43"), ("credential", "559 104"),
    ("handle", "u/quietlark_88"), ("phone_intl", "(+56) 9 7741 2208"),
    ("license_plate", "yk19 rzt"),
    ("url", "https://docs.google.com/document/d/1xQ7vB_Lm9pE3-kYtR2/edit?usp=sharing"),
])
def test_I14_phase5_values_get_a_different_surrogate(kind, value):
    out = MimicGen(seed=1).generate(DetectedEntity(value, 0, len(value), kind, 1.0, "pattern"))
    assert out and out != value
