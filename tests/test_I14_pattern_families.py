"""Audit I1 / I8 / I14 — PII families the cascade missed on real chat text:
ages, birth years, handles, personal URLs, credentials, labelled IDs and
national phone formats. Every case comes from the J2 dev split or one of its
"keep" lists. Model-free: pattern_scan and MimicGen only.
"""

import re

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    return [(e.type, e.text) for e in ps.scan(text)]


def assert_offsets(text):
    for e in ps.scan(text):
        assert text[e.start:e.end] == e.text


# ── detection ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("I'm 34 years old, born in 1990, and my daughter just turned 7.",
     [("age", "34 years old"), ("dob", "1990"), ("age", "turned 7")]),
    ("my mom is 72 and my grandma Evelyn is 89",
     [("age", "72"), ("age", "89")]),
    ("My daughter Ava Lindqvist (14) has a rash", [("age", "14")]),
])
def test_I14_age_and_birth_year(text, expected):
    assert found(text) == expected
    assert_offsets(text)


@pytest.mark.parametrize("text", [
    "I'm 5 minutes away", "I'm 100% sure", "after 6 years", "Years exp: 7",
    "Invoice 48213 for 250 units", "sshd[2231]: session opened",
])
def test_I14_age_negatives(text):
    assert [t for t, _ in found(text) if t == "age"] == []


@pytest.mark.parametrize("text, value", [
    ("DM me @sarahm_92 on X", "@sarahm_92"),
    ("I'm sarah.mitchell#4421 on Discord", "sarah.mitchell#4421"),
    ("my xbox gamertag is xX_ShadowFang_Xx", "xX_ShadowFang_Xx"),
    ("my psn is ryu.kobayashi07", "ryu.kobayashi07"),
    ("Your username is gabi.ferreira84.", "gabi.ferreira84"),
    ("sshd[2231]: Failed password for jmorales from 10.1.2.3", "jmorales"),
])
def test_I14_handles(text, value):
    assert ("handle", value) in found(text)


@pytest.mark.parametrize("text", [
    "@app.route('/x')\n@dataclass\nclass A: pass",
    "@media screen and (max-width: 600px)",
    "@here can someone review?",
    "my username is incorrect",
])
def test_I14_handle_negatives(text):
    assert [v for t, v in found(text) if t == "handle"] == []


@pytest.mark.parametrize("text, value", [
    ("profile: https://www.linkedin.com/in/sarah-mitchell-1a2b3c",
     "https://www.linkedin.com/in/sarah-mitchell-1a2b3c"),
    ("my site sarahmitchell.dev", "sarahmitchell.dev"),
    ("[Sarah](https://acme.com/team/sarah)", "https://acme.com/team/sarah"),
    ("Here's my portfolio https://www.kateryna-bondar.com",
     "https://www.kateryna-bondar.com"),
])
def test_I14_personal_urls(text, value):
    assert ("url", value) in found(text)


def test_I1_generic_urls_are_opaque():
    text = ("proxy_pass http://127.0.0.1:8080; docs at "
            "https://api.weatherly.io/v2/forecast?zip=85281")
    assert found(text) == []
    assert ps.opaque_spans(text) == [(m.start(), m.end()) for m in
                                     re.finditer(r"https?://\S+?(?=;|$)", text)]


def test_I1_secret_inside_url_is_a_credential():
    text = "reset: https://accounts.medvault.com/reset?t=8f3a91c2e7"
    assert found(text) == [("credential", "8f3a91c2e7")]


@pytest.mark.parametrize("text, value", [
    ("Login: username sarahm92, password Hunter2!2024, ok", "Hunter2!2024"),
    ("2FA backup code 8841-2201.", "8841-2201"),
    ("Coinbase verification code is 482913", "482913"),
    ("your temporary password is Tq7#mPz2!vL. Thanks", "Tq7#mPz2!vL"),
])
def test_I14_credentials(text, value):
    assert ("credential", value) in found(text)


@pytest.mark.parametrize("text", [
    "token = os.environ['API_TOKEN']",
    "password: ${DB_PASSWORD}",
    "reset my password for Gmail",
    "How do I delete this API key token: sk-AbCdEf1234567890XyZ987654321vUtS?",
])
def test_I14_credential_negatives(text):
    assert [v for t, v in found(text) if t == "credential"] == []


@pytest.mark.parametrize("text, value", [
    ("Our EIN is 12-3456789", "12-3456789"),
    ("customer no. 55-019283", "55-019283"),
    ("Mon numéro client est 7741 2290 18.", "7741 2290 18"),
    ("Account 8774 10 223 1186654 for rent", "8774 10 223 1186654"),
    ("card ending 4417", "4417"),
    ("student ID is 2204417.", "2204417"),
    ('"policy_no": "HX-4471-90321"', "HX-4471-90321"),
    ("serial VNB3K12345, MAC a4:5d:36:9e:01:7c", "VNB3K12345"),
])
def test_I14_labelled_ids(text, value):
    assert ("id_number", value) in found(text)
    assert_offsets(text)


def test_I14_passport_list_continuation():
    assert found("Passport: A09382716 / A11746620") == [
        ("passport", "A09382716"), ("passport", "A11746620")]


@pytest.mark.parametrize("text, value", [
    ("teléfono 612 345 678", "612 345 678"),
    ("मेरा फ़ोन नंबर 98765 43210", "98765 43210"),
    ("电话：139 1234 5678", "139 1234 5678"),
    ("电话是13812345678", "13812345678"),
])
def test_I14_national_phones(text, value):
    assert ("phone_intl", value) in found(text)


def test_I14_markdown_wrapped_phone():
    assert ("phone_us", "312-849-2031") in found("*Phone:* _312-849-2031_")


@pytest.mark.parametrize("text", [
    "lease ends 05/31/2027", "logged 2026-10-03T14:21:55Z", "due 10/02/26",
    "listen 0.0.0.0:443", "ISBN 978-0-306-40615-7",
])
def test_I8_keep_values_untouched(text):
    assert found(text) == []


def test_I14_dob_short_year_and_cue():
    assert ("dob", "14 March '90") in found("my wife was born 14 March '90")
    assert ("dob", "03/14/90") in found("DOB 03/14/90")


# ── surrogates keep the format ───────────────────────────────────────────────

def _gen(typ, value, seed=7):
    return MimicGen(seed=seed).generate(DetectedEntity(value, 0, len(value), typ, 1.0, "pattern"))


@pytest.mark.parametrize("seed", range(20))
def test_I14_surrogate_formats(seed):
    url = _gen("url", "https://www.linkedin.com/in/sarah-mitchell-1a2b3c", seed)
    assert url.startswith("https://www.linkedin.com/in/") and "sarah" not in url
    assert _gen("url", "sarahmitchell.dev", seed).endswith(".dev")
    team = _gen("url", "https://acme.com/team/sarah", seed)
    assert team.startswith("https://acme.com/team/") and team != "https://acme.com/team/sarah"

    h = _gen("handle", "sarah.mitchell#4421", seed)
    assert re.fullmatch(r"[a-z]+\.[a-z\-]+#\d{4}", h) and h != "sarah.mitchell#4421"
    assert _gen("handle", "@sarahm_92", seed).startswith("@")

    age = _gen("age", "34 years old", seed)
    assert re.fullmatch(r"\d+ years old", age) and age != "34 years old"
    assert 1 <= int(_gen("age", "1", seed)) <= 4

    d = _gen("dob", "14 March '90", seed)
    # D2 (Phase 4): the date moves by at most two years, so the month may stay
    assert re.fullmatch(r"\d{1,2} [A-Z][a-z]+ '\d{2}", d) and d != "14 March '90"
    assert abs(int(d[-2:]) - 90) <= 2
    assert re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", _gen("dob", "14.03.1990", seed))
    assert re.fullmatch(r"(19|20)\d{2}", _gen("dob", "1990", seed))
    assert re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}(st|nd|rd|th), \d{4}",
                        _gen("dob", "Sep 3rd, 1988", seed))

    c = _gen("credential", "Hunter2!2024", seed)
    assert len(c) == 12 and c[7] == "!" and c != "Hunter2!2024"


def test_I14_new_types_accepted_by_pii_off():
    import surrogateshield as ss
    from surrogateshield.core.detection.pipeline import resolve_pii_off
    ss.config(pii_off=["url", "handle", "credential", "age", "password"])
    assert {"url", "handle", "credential", "age"} <= resolve_pii_off(ss.config().pii_off)


def test_I14_label_phone_does_not_claim_an_ssn():
    # "Contact" is a phone label; the SSN 25 characters later is not a phone
    ents = ps.scan("Contact alice@corp.com, SSN 123-45-6789.")
    assert {(e.text, e.type) for e in ents} >= {("alice@corp.com", "email"), ("123-45-6789", "ssn")}
    assert [e.type for e in ps.scan("teléfono 612 345 678")] == ["phone_intl"]
