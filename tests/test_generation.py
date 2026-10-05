"""Surrogate generation: replace-mode structure preservation, uniqueness,
type-consistent formats."""

import re

import pytest

from surrogateshield.core.detection import address_parser as ap
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen, _aba_check


def _entity(text, etype="address"):
    parsed = ap.parse(text) if etype == "address" else None
    return DetectedEntity(text=text, start=0, end=len(text), type=etype, parsed=parsed)


# ── Replace mode: structure-preserving, one unit ──────────────────────────────

def test_replace_preserves_structure_full_address():
    gen = MimicGen(seed=7)
    original = "789 Crescent Row, Tempe, AZ 85281"
    fake = gen._gen_address(_entity(original), mode="replace")
    assert fake != original
    # same shape: 3-digit number, one street word, same suffix, city, state, zip
    assert re.fullmatch(r"\d{3} [A-Z][A-Za-z'\-]+ Row, [A-Za-z .'\-]+, [A-Z]{2} \d{5}", fake), fake
    # no component of the original survives
    assert "789" not in fake and "Crescent" not in fake
    assert "Tempe" not in fake and "85281" not in fake


def test_replace_keeps_unit_designator_fakes_number():
    gen = MimicGen(seed=9)
    fake = gen._gen_address(_entity("500 Main St Apt 4B, Tempe, AZ 85281"), mode="replace")
    assert "Apt" in fake
    assert "Main" not in fake
    assert ", " in fake  # separators survive


def test_replace_zip4_shape_kept():
    gen = MimicGen(seed=11)
    fake = gen._gen_address(_entity("55 Birch Ln, Salem, OR 97301-1234"), mode="replace")
    assert re.search(r"\d{5}-\d{4}$", fake), fake
    assert "97301-1234" not in fake


def test_replace_full_state_name_stays_full():
    gen = MimicGen(seed=13)
    fake = gen._gen_address(_entity("88 Pine St, Portland, Oregon 97205"), mode="replace")
    assert not re.search(r", [A-Z]{2} \d{5}$", fake), fake  # not abbreviated


def test_replace_never_concatenates_extra_components():
    """v1 replaced a street line with a FULL Faker address (own apt/city/zip)."""
    gen = MimicGen(seed=15)
    fake = gen._gen_address(_entity("789 Crescent Row"), mode="replace")
    # same shape: number + street name + suffix, nothing more
    assert re.fullmatch(r"\d{3} [A-Z][A-Za-z'\-]+ Row", fake), fake


# ── Uniqueness ────────────────────────────────────────────────────────────────

def test_surrogates_unique_within_session():
    gen = MimicGen(seed=21)
    seen = set()
    for i in range(50):
        s = gen.generate(_entity(f"user{i}@example.com", "email"))
        assert s not in seen
        seen.add(s)


def test_generate_all_dedupes_repeated_entity():
    gen = MimicGen(seed=23)
    e1 = _entity("789 Crescent Row, Tempe, AZ 85281")
    e2 = _entity("789 Crescent Row, Tempe, AZ 85281")
    mapping = gen.generate_all([e1, e2], address_mode="shift")
    assert len(mapping) == 1


# ── Type-consistent formats ───────────────────────────────────────────────────

def test_ssn_format():
    s = MimicGen().generate(_entity("123-45-6789", "ssn"))
    assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", s)


def test_phone_us_format():
    # D3 (Phase 4): the surrogate keeps the original's layout, no "+1-" added
    s = MimicGen().generate(_entity("480-555-1234", "phone_us"))
    assert re.fullmatch(r"[2-9]\d{2}-[2-9]\d{2}-\d{4}", s) and s != "480-555-1234"


def test_generated_routing_number_passes_aba_checksum():
    for _ in range(5):
        s = MimicGen().generate(_entity("021000021", "us_bank_number"))
        assert _aba_check(s), s


def test_crypto_surrogate_is_base58():
    s = MimicGen().generate(_entity("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "crypto"))
    assert s[0] == "1"
    assert not set(s) & set("0OIl")  # base58 excludes these


def test_gender_surrogate_stays_readable():
    s = MimicGen().generate(_entity("she/her", "gender_indicator"))
    assert s in {
        "male", "female", "non-binary",
        "he/him", "she/her", "they/them",
        "gender: male", "gender: female", "sex: male", "sex: female",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Gate J5 — linked, gender-consistent, shape- and meaning-preserving
# surrogates, deterministic under a session seed (audit D1–D4, I7).
# ═════════════════════════════════════════════════════════════════════════════

import datetime

from surrogateshield.core.consistency import COMMON_WORD_NAMES, assign_surrogates
from surrogateshield.core.detection.geo_data import US_STATE_ABBREVS, US_STATES
from surrogateshield.core.detection.pipeline import gender_follows_name, term_gender
from surrogateshield.core.generation import places
from surrogateshield.core.generation.identity import FREEMAIL, given_gender, name_gender
from surrogateshield.core.storage.shadow_map import ShadowMap

RUNS = 300


def _ent(text, etype, start=0):
    return DetectedEntity(text=text, start=start, end=start + len(text), type=etype, score=1.0,
                          source="t")


def _all(pairs, seed, text=None):
    """generate_all over (value, type) pairs of one message."""
    text = text or " | ".join(v for v, _ in pairs)
    ents, pos = [], 0
    for v, t in pairs:
        i = text.index(v, pos)
        ents.append(_ent(v, t, i))
    return MimicGen(seed=seed).generate_all(ents, text=text)


def _words(value):
    return re.findall(r"[^\W\d_]+", value.lower())


# ── D1: names and e-mails are one person ─────────────────────────────────────

@pytest.mark.parametrize("email, pattern", [
    ("sarah.mitchell@gmail.com", "{g}.{f}"),
    ("smitchell@acme-dental.com", "{gi}{f}"),
    ("sarah@acme-dental.com", "{g}"),
    ("sarah_mitchell92@yahoo.com", "{g}_{f}\\d\\d"),
    ("mitchell.sarah@outlook.com", "{f}.{g}"),
    ("SarahMitchell@hotmail.com", "{G}{F}"),
])
def test_J5_D1_name_and_email_linked_every_run(email, pattern):
    for seed in range(RUNS if email == "sarah.mitchell@gmail.com" else 40):
        m = _all([("Sarah Mitchell", "PERSON"), (email, "email")], seed)
        g, f = m["Sarah Mitchell"].split()
        local, _, domain = m[email].partition("@")
        expect = pattern.format(g=g.lower(), f=f.lower(), gi=g[0].lower(), G=g, F=f)
        assert re.fullmatch(expect, local), (seed, m)


def test_J5_D1_email_first_still_links_to_later_name():
    # the e-mail is seen first (another message); the person named later is
    # the same surrogate person
    gen = MimicGen(seed=5)
    e = gen.generate_all([_ent("sarah.mitchell@gmail.com", "email")])["sarah.mitchell@gmail.com"]
    p = gen.generate_all([_ent("Sarah Mitchell", "PERSON")])["Sarah Mitchell"]
    assert e.split("@")[0] == p.lower().replace(" ", ".")


@pytest.mark.parametrize("seed", range(40))
def test_J5_D1_domains_real_looking_and_consistent(seed):
    m = _all([("Sarah Mitchell", "PERSON"), ("sarah@acme-dental.co.uk", "email"),
              ("info@acme-dental.co.uk", "email"), ("j.doe@gmail.com", "email")], seed)
    d1 = m["sarah@acme-dental.co.uk"].split("@")[1]
    assert d1.endswith(".co.uk") and d1 != "acme-dental.co.uk" and "example" not in d1
    assert m["info@acme-dental.co.uk"] == "info@" + d1           # role address, same domain
    assert m["j.doe@gmail.com"].endswith("@gmail.com")             # free-mail kept
    assert m["j.doe@gmail.com"] != "j.doe@gmail.com"


# ── D2: gender ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, gender", [
    ("Sarah Mitchell", "f"), ("David Escobar", "m"), ("Emily Chen", "f"),
    ("Ms. Taylor", "f"), ("Mr. Okafor", "m"), ("Mrs. Jordan Lee", "f"),
])
def test_J5_D2_surrogate_keeps_gender(name, gender):
    for seed in range(RUNS if name == "Sarah Mitchell" else 60):
        s = _all([(name, "PERSON")], seed)[name]
        assert name_gender(s) == gender, (seed, s)


def test_J5_D2_pronouns_decide_an_unknown_name():
    for seed in range(30):
        text = "Kerensa Vantree said she would call."
        s = MimicGen(seed=seed).generate_all([_ent("Kerensa Vantree", "PERSON")], text=text)
        assert given_gender(s["Kerensa Vantree"].split()[0]) in ("f", "x"), s


@pytest.mark.parametrize("seed", range(60))
def test_J5_D2_no_title_or_suffix_added(seed):
    s = _all([("Sarah Mitchell", "PERSON")], seed)["Sarah Mitchell"]
    assert len(s.split()) == 2 and not re.search(r"(?i)\b(dr|mr|ms|mrs|jr|sr|phd|md|dvm)\b", s)


def test_J5_D2_form_kept():
    m = _all([("Dr. Sarah J. Mitchell-Okafor Jr.", "PERSON")], 3)
    s = m["Dr. Sarah J. Mitchell-Okafor Jr."]
    assert re.fullmatch(r"Dr\. [A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+-[A-Z][a-z]+ Jr\.", s), s
    assert name_gender(s) == "f"


@pytest.mark.parametrize("term, follows", [
    ("female", True), ("gender: F", True), ("she/her", True), ("male", False), ("he/him", False),
    ("non-binary", False),
])
def test_J5_D2_gender_term_follows_named_person(term, follows):
    ents = [_ent("Sarah Mitchell", "PERSON"), _ent(term, "gender_indicator", 20)]
    kept, kept_verbatim = gender_follows_name(ents)
    assert bool(kept_verbatim) is follows
    assert any(e.type == "gender_indicator" for e in kept) is not follows


def test_J5_D2_gender_term_alone_is_masked():
    kept, verbatim = gender_follows_name([_ent("female", "gender_indicator")])
    assert verbatim == [] and len(kept) == 1
    assert term_gender("Gender: female") == "f" and term_gender("they/them") is None


# ── D3: shapes ───────────────────────────────────────────────────────────────

D3_SHAPES = [
    ("(312) 849-2031", "phone_us", r"\([2-9]\d\d\) [2-9]\d\d-\d{4}"),
    ("480-555-1234", "phone_us", r"[2-9]\d\d-[2-9]\d\d-\d{4}"),
    ("+1 800 555 0199", "phone_us", r"\+1 800 [2-9]\d\d \d{4}"),
    ("+44 7911 123456", "phone_uk", r"\+44 7\d{3} \d{6}"),
    ("07911 123456", "phone_uk", r"07\d{3} \d{6}"),
    ("+49 30 1234567", "phone_intl", r"\+49 3\d \d{7}"),
    ("+91 98765 43210", "phone_intl", r"\+91 9\d{4} \d{5}"),
    ("0049 89 123456", "phone_intl", r"0049 8\d \d{6}"),
    ("1990-03-15", "dob", r"\d{4}-\d{2}-\d{2}"),
    ("15 March 1990", "dob", r"\d{1,2} [A-Z][a-z]+ \d{4}"),
    ("15/03/1990", "dob", r"\d{2}/\d{2}/\d{4}"),
    ("NW1 6XE", "postcode_uk", r"[A-Z]{2}\d \d[A-Z]{2}"),
    ("85281-1234", "zip_us", r"\d{5}-\d{4}"),
    ("AKIAIOSFODNN7EXAMPLE", "api_key", r"AKIA[A-Z0-9]{16}"),
    ("ghp_16C7e42F292c6912E7710c838347Ae178B4a", "api_key", r"ghp_[A-Za-z0-9]{36}"),
    ("0x742d35Cc6634C0532925a3b844Bc454e4438f44e", "crypto", r"0x[0-9a-f]{40}"),
    ("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq", "crypto", r"bc1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]{39}"),
    ("4111-1111-1111-1111", "credit_card", r"41\d\d-\d{4}-\d{4}-\d{4}"),
    ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "ip_address", r"([0-9a-f]{4}:){7}[0-9a-f]{4}"),
    ("078051120", "ssn", r"\d{9}"),
    ("078-05-1120", "ssn", r"\d{3}-\d{2}-\d{4}"),
]


@pytest.mark.parametrize("value, etype, shape", D3_SHAPES)
def test_J5_D3_shape_preserved(value, etype, shape):
    for seed in range(30):
        s = MimicGen(seed=seed).generate(_ent(value, etype))
        assert re.fullmatch(shape, s) and s != value, (seed, s)


def _luhn_ok(number):
    ds = [int(c) for c in number if c.isdigit()][::-1]
    return sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(ds)) % 10 == 0


@pytest.mark.parametrize("seed", range(30))
def test_J5_D3_numbers_stay_valid(seed):
    gen = MimicGen(seed=seed)
    assert _luhn_ok(gen.generate(_ent("4111 1111 1111 1111", "credit_card")))
    ssn = gen.generate(_ent("078-05-1120", "ssn"))
    area, group, serial = ssn.split("-")
    assert area not in ("000", "666") and int(area) < 900 and group != "00" and serial != "0000"
    ip = gen.generate(_ent("192.168.1.20", "ip_address"))
    assert ip.startswith("192.168.") and ip != "192.168.1.20"
    pub = [int(o) for o in gen.generate(_ent("8.8.4.4", "ip_address")).split(".")]
    assert pub[0] not in (10, 127, 192, 172) and 1 <= pub[0] <= 223


@pytest.mark.parametrize("seed", range(30))
def test_J5_D3_nanp_valid(seed):
    s = MimicGen(seed=seed).generate(_ent("480-555-1234", "phone_us"))
    area, exch = s[:3], s[4:7]
    assert area[0] in "23456789" and area[1:] != "11" and area[1] != "9"
    assert exch[0] in "23456789" and exch[1:] != "11" and exch != "555"


# ── D4: one person, one identity ─────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(60))
def test_J5_D4_mentions_share_one_identity(seed):
    text = "Sarah Mitchell (sarah.mitchell@gmail.com). Sarah said Ms. Mitchell is SARAH MITCHELL."
    m = assign_surrogates([_ent(v, "email" if "@" in v else "PERSON", text.index(v)) for v in
                           ("Sarah Mitchell", "sarah.mitchell@gmail.com", "Sarah", "Ms. Mitchell",
                            "SARAH MITCHELL")], text, MimicGen(seed=seed))
    g, f = m["Sarah Mitchell"].split()
    assert m["Sarah"] == g                                    # single token stays single
    assert m["Ms. Mitchell"] == f"Ms. {f}"
    assert m["SARAH MITCHELL"] == f"{g} {f}".upper()
    assert m["sarah.mitchell@gmail.com"] == f"{g}.{f}@gmail.com".lower()


def test_J5_D4_later_message_links_to_earlier_person():
    gen, shadow = MimicGen(seed=8), ShadowMap("j5-d4", storage_dir=None)
    t1 = "I am Sarah Mitchell."
    m1 = assign_surrogates([_ent("Sarah Mitchell", "PERSON", 5)], t1, gen, [shadow])
    shadow.update({v: k for k, v in m1.items()})
    t2 = "Sarah forgot. Email Ms. Mitchell."
    m2 = assign_surrogates([_ent("Sarah", "PERSON", 0), _ent("Ms. Mitchell", "PERSON", 20)],
                           t2, gen, [shadow], forbidden=set(shadow.originals()))
    g, f = m1["Sarah Mitchell"].split()
    assert m2 == {"Sarah": g, "Ms. Mitchell": f"Ms. {f}"}


def test_J5_D4_reopened_session_relinks():
    from surrogateshield.core.generation.identity import People
    import random
    p = People(random.Random(1))
    p.learn("Mia Lopez", "Sarah Mitchell")
    p.observe(["Sarah"], "Sarah")
    assert p.name("Sarah") == "Mia" and p.name("Ms. Mitchell") == "Ms. Lopez"


@pytest.mark.parametrize("seed", range(30))
def test_J5_D4_distinct_people_distinct_surrogates(seed):
    m = _all([("Sarah Mitchell", "PERSON"), ("Sarah Jones", "PERSON"), ("Tom Mitchell", "PERSON")],
             seed)
    s1, s2, s3 = (m[k].split() for k in ("Sarah Mitchell", "Sarah Jones", "Tom Mitchell"))
    assert s1[0] == s2[0] and s1[1] == s3[1]                  # shared given / family name
    assert s1[1] != s2[1] and s1[0] != s3[0]


# ── I7: the meaning the answer needs survives ─────────────────────────────────

@pytest.mark.parametrize("value, fmt", [
    ("15/03/1990", "%d/%m/%Y"), ("1990-03-15", "%Y-%m-%d"), ("March 15, 1990", "%B %d, %Y"),
    ("15 Mar 1990", "%d %b %Y"), ("03.15.1990", "%m.%d.%Y"),
])
def test_J5_I7_dob_within_two_years_same_format(value, fmt):
    orig = datetime.datetime.strptime(value, fmt).date()
    for seed in range(RUNS // 3):
        s = MimicGen(seed=seed).generate(_ent(value, "dob"))
        new = datetime.datetime.strptime(s, fmt).date()          # same format
        assert 30 <= abs((new - orig).days) <= 731 and new <= datetime.date.today(), (seed, s)


def test_J5_I7_dob_partial_forms():
    for seed in range(40):
        g = MimicGen(seed=seed)
        assert abs(int(g.generate(_ent("1990", "dob"))) - 1990) in (1, 2)
        assert re.fullmatch(r"(January|February|March|April|May|June|July|August|September|"
                            r"October|November|December) (198[89]|199[0-2])",
                            g.generate(_ent("March 1990", "dob")))
        assert re.fullmatch(r"\d{1,2}(st|nd|rd|th) of [A-Z][a-z]+, \d{4}",
                            g.generate(_ent("3rd of March, 1990", "dob")))


def test_J5_I7_weekday_matches_new_date():
    for seed in range(20):
        s = MimicGen(seed=seed).generate(_ent("Thursday, March 15, 1990", "dob"))
        d = datetime.datetime.strptime(s, "%A, %B %d, %Y")
        assert d.strftime("%A") == s.split(",")[0]


@pytest.mark.parametrize("value, check", [
    ("Tempe", lambda s: s.casefold() in places.TOWN_NAMES),
    ("Leeds", lambda s: s in places.TOWNS["GB"]),
    ("AZ", lambda s: s in US_STATE_ABBREVS and s != "AZ"),
    ("Arizona", lambda s: s.casefold() in US_STATES and s != "Arizona"),
    ("Lake Pleasant", lambda s: s in places.FEATURES["lake"]),
    ("Camelback Mountain", lambda s: s in places.FEATURES["mount"]),
])
def test_J5_I7_places_are_real(value, check):
    for seed in range(RUNS // 3):
        s = MimicGen(seed=seed).generate(_ent(value, "GPE"))
        assert check(s), (seed, s)


def test_J5_I7_town_follows_country_in_message():
    text = "I moved to Smallbridge, Ontario, Canada last year."
    for seed in range(20):
        m = MimicGen(seed=seed).generate_all([_ent("Smallbridge", "GPE", 11)], text=text)
        assert m["Smallbridge"] in places.TOWNS["CA"]


@pytest.mark.parametrize("value, keep", [
    ("Google", ""), ("Mitchell Family Dental LLC", "Family Dental LLC"),
    ("Acme Holdings, Inc.", "Holdings, Inc."), ("University of Arizona", "University of"),
    ("St. Joseph's Hospital", "Hospital"), ("Empire State Building", "Building"),
])
def test_J5_I7_org_without_commas_keeps_kind(value, keep):
    for seed in range(RUNS // 6):
        s = MimicGen(seed=seed).generate(_ent(value, "ORG"))
        assert s.count(",") == value.count(",") and " and " not in s, s
        assert keep in s and s != value, s


@pytest.mark.parametrize("name, locale_names", [
    ("Zhou Yan", ("zh_Latn",)), ("Adebayo Okafor", ("yo_NG", "ig_NG")),   # Okafor is Igbo
    ("Rahul Verma", ("en_IN",)), ("Иван Петров", ("ru_RU",)), ("李娜", ("zh_CN",)),
])
def test_J5_I7_person_keeps_locale_and_script(name, locale_names):
    from surrogateshield.core.generation.identity import _pools
    pools = [_pools()[k] for k in locale_names]
    for seed in range(30):
        s = _all([(name, "PERSON")], seed)[name]
        if locale_names == ("zh_CN",):
            assert len(s) == len(name) and all("一" <= c <= "鿿" for c in s), s
            continue
        assert len(s.split()) == len(name.split())
        assert any(w.casefold() in pool.family_set or w.casefold() in pool.gender_of
                   for w in s.split() for pool in pools), (seed, s)


# ── I7: surrogate first names are not ordinary words ─────────────────────────

def test_J5_I7_no_common_word_names():
    for seed in range(RUNS):
        m = _all([("Sarah Mitchell", "PERSON"), ("Jack Hill", "PERSON"), ("Grace Brown", "PERSON")],
                 seed)
        for s in m.values():
            assert not set(_words(s)) & COMMON_WORD_NAMES, (seed, m)
            assert all(len(w) >= 3 for w in s.split())


# ── Determinism under a session seed (P3-2) ───────────────────────────────────

ALL_TYPES = [("Sarah Mitchell", "PERSON"), ("sarah.mitchell@acme.com", "email"),
             ("480-555-1234", "phone_us"), ("123-45-6789", "ssn"), ("15/03/1990", "dob"),
             ("Tempe", "GPE"), ("Acme Corp", "ORG"), ("4111 1111 1111 1111", "credit_card"),
             ("10.0.0.7", "ip_address"), ("85281", "zip_us"), ("sk-abc123def456ghi789", "api_key"),
             ("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "crypto"), ("021000021", "us_bank_number"),
             ("D1234567", "us_driver_license"), ("female", "gender_indicator"),
             ("123 Main St, Tempe, AZ 85281", "address")]


def test_J5_deterministic_under_seed():
    a, b = _all(ALL_TYPES, 1234), _all(ALL_TYPES, 1234)
    assert a == b
    assert a != _all(ALL_TYPES, 1235)


def test_J5_session_seed_is_deterministic():
    import surrogateshield as ss
    ents = [_ent(v, t) for v, t in ALL_TYPES[:6]]
    outs = []
    for _ in range(2):
        s = ss.Session(seed=99)
        outs.append(assign_surrogates(ents, "x", s._mimic, [s._shadow]))
    assert outs[0] == outs[1]


def test_J5_no_global_faker_state():
    import random
    import surrogateshield.core.generation.mimic as mimic
    assert not hasattr(mimic, "_fake")
    random.seed(0)
    a = _all(ALL_TYPES, 7)
    random.seed(1)                                   # the global RNG has no effect
    assert _all(ALL_TYPES, 7) == a


@pytest.mark.parametrize("seed", range(20))
def test_J5_D4_new_original_never_gets_an_issued_surrogate(seed):
    # Before: "shared37@…" got the surrogate already issued to "shared21@…",
    # so the shadow map restored the wrong address (found by I9's 8-thread test)
    gen, issued = MimicGen(seed=seed), {}
    for k in range(80):
        v = f"shared{k}@corp.example"
        s = gen.generate_all([_ent(v, "email")], text=v)[v]
        assert s not in issued, (k, s, issued[s] if s in issued else None)
        issued[s] = v


@pytest.mark.parametrize("value, shape", [
    ("mill ave", r"[a-z]+ ave"),                      # Before: RuntimeError (rw-dev-0048)
    ("Medical Center", r"[A-Z][a-z]+ Center"),
    ("the dermatology clinic", r"the [a-z]+ clinic"),
    ("st joseph hospital", r"st [a-z]+ hospital"),
])
def test_J5_I7_all_generic_facility(value, shape):
    for seed in range(20):
        m = MimicGen(seed=seed).generate_all([_ent(value, "FAC")], text=f"at {value} today")
        assert re.fullmatch(shape, m[value]) and m[value] != value, m


def test_J5_D3_evaluator_credits_kept_country_code_only():
    # J1 key 860 labels "+7" (a calling code) as GPE; the phone surrogate keeps it
    import eval_metrics as em
    q = "Mikhail Belov, mobile +7 916 234 5678, lives in Kazan, +7."
    ph = "+7 916 234 5678"
    sur = "+7 925 871 3304"
    sent = q.replace(ph, sur)
    spans = [{"text": ph, "start": q.index(ph), "end": q.index(ph) + len(ph),
              "type": "phone_intl", "replaced": True}]
    key = {"phone": ["916 234 5678"], "gpe": ["+7"]}
    out = em.classify_sent_leaks(q, key, sent, spans, None, {ph: sur})
    leak, = out["leaks"]
    assert leak["value"] == "+7" and leak["reasons"] == ["phone_country_code_kept"]
    assert leak["deliberate"]
    # the bare "+7" outside the phone is not covered by the phone
    q2 = "Code +7 for Russia; call 916 234 5678."
    spans2 = [{"text": "916 234 5678", "start": q2.index("916"), "end": len(q2) - 1,
               "type": "phone_intl", "replaced": True}]
    out2 = em.classify_sent_leaks(q2, key, q2.replace("916 234 5678", "925 871 3304"), spans2,
                                  None, {"916 234 5678": "925 871 3304"})
    assert [l["deliberate"] for l in out2["leaks"]] == [False]


@pytest.mark.parametrize("town, text, country", [
    ("Sevilla", "Me llamo Carlos, vivo en Sevilla.", "ES"),      # Before: "Ann Arbor" (echo J4)
    ("Nürnberg", "I live in Nürnberg.", "DE"),
    ("Bourg", "Je m'appelle Léa et j'habite à Bourg.", "FR"),
])
def test_J5_I7_town_country_from_local_name_or_language(town, text, country):
    for seed in range(15):
        m = MimicGen(seed=seed).generate_all([_ent(town, "GPE", text.index(town))], text=text)
        assert m[town] in places.TOWNS[country], m


def test_J4_surrogate_unique_across_case():
    # Before: "denver" → "shreveport", later "पुणे" → "Shreveport"; the
    # case-insensitive restore sent the second back to "denver" (echo seed 0)
    gen = MimicGen(seed=0)
    gen.used_surrogates.add("shreveport")
    for _ in range(200):
        assert gen.generate(_ent("Pune", "GPE")).casefold() != "shreveport"
    assert gen._taken("SHREVEPORT") and not gen._taken("Ogden")
