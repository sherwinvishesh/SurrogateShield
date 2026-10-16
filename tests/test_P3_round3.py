"""Phase 3, round 3: the injected-slice misses of dev, one rule each.
Phone words in more scripts, the traditional-Chinese age unit, a tagger
name cut at full-width punctuation, institution names that stay on their
line, and a URL on a contact strip. Texts and values are invented."""

import pytest

from surrogateshield.core.detection import pattern_scan, pii_tagger, structural


def _types(text):
    return {(e.type, e.text) for e in pattern_scan.scan(text)}


# ── phone words in more scripts ──────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("Мои данные: телефон: 912 34 56 78, город Казань", "912 34 56 78"),
    ("Тел. 912 345 67 89", "912 345 67 89"),
    ("Τηλέφωνο: 694 123 4567", "694 123 4567"),
    ("رقم هاتف 055 123 4567", "055 123 4567"),
    ("شماره تلفن 912 345 6789", "912 345 6789"),
    ("טלפון 052 123 4567", "052 123 4567"),
    ("電話 0912 345 678", "0912 345 678"),
    ("số điện thoại 912 345 678", "912 345 678"),
])
def test_a_phone_word_in_another_script(text, value):
    # a 3-3-4 number may be claimed first as phone_us; any phone type will do
    assert value in {v for t, v in _types(text) if t.startswith("phone")}


def test_no_phone_word_no_phone():
    assert not [e for e in pattern_scan.scan("Заказ 912 34 56 78 отправлен") if e.type.startswith("phone")]


# ── traditional-Chinese age ──────────────────────────────────────────────────

def test_the_traditional_age_unit():
    assert ("age", "34 歲") in _types("我今年 34 歲，住在高雄")
    assert ("age", "34 岁") in _types("我今年 34 岁")


# ── a tagger name cut at full-width punctuation ──────────────────────────────

def _tokens(text, words, tag):
    labels = ["O", f"B-{tag}", f"I-{tag}"]
    out, pos = [], 0
    for i, w in enumerate(words):
        s = text.index(w, pos)
        pos = s + len(w)
        p = [0.0, 0.0, 0.0]
        p[1 if i == 0 else 2] = 1.0
        out.append((s, pos, p))
    return out, labels


@pytest.mark.parametrize("text, words, tag, want", [
    ("我是 Ilse K. Varga，34 歲", ["Ilse", "K.", "Varga，34"], "PERSON", "Ilse K. Varga"),
    ("我在 Kessmar Labs，工作", ["Kessmar", "Labs，工作"], "ORG", "Kessmar Labs"),
    ("华为技术（深圳）有限公司", ["华为技术（深圳）有限公司"], "ORG", "华为技术（深圳）有限公司"),
])
def test_a_name_ends_at_full_width_punctuation(text, words, tag, want):
    tokens, labels = _tokens(text, words, tag)
    got = [text[c.start:c.end] for c in pii_tagger.decode(text, tokens, labels, merge_address=False)]
    assert got == [want]


def test_other_types_keep_the_span():
    text = "位置：台北市，信义区"
    tokens, labels = _tokens(text, ["台北市，信义区"], "LOCATION")
    got = [text[c.start:c.end] for c in pii_tagger.decode(text, tokens, labels, merge_address=False)]
    assert got == ["台北市，信义区"]


# ── institution names stay on their line ─────────────────────────────────────

@pytest.mark.parametrize("text, bad", [
    ("Name: Ilse Varga\nSchool: Kessmar Riverside College\nPhone: 06 12 34 56 78", "College\nPhone"),
    ("Name: Ilse Varga\nHospital\nNotes: none", "Hospital\nNotes"),
    ("Contact: Kim Tavora\nPharmacy: on 5th street", "Tavora\nPharmacy"),
])
def test_an_institution_never_crosses_a_line(text, bad):
    added, _removed = structural._institutions(text, [])
    assert bad not in [a.text for a in added]


def test_an_institution_on_one_line_is_still_found():
    added, _ = structural._institutions("She studied at Collège Saint-Michel last year.", [])
    assert [a.text for a in added] == ["Collège Saint-Michel"]
    added, _ = structural._institutions("We go to Riverside Dental Practice on Mondays.", [])
    assert [a.text for a in added] == ["Riverside Dental Practice"]


# ── a URL on a contact strip ─────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Thanks.\n\nPolicy: QXR-381205774 | https://brenhavenweb.com",
    "Ilse Varga · ilse@brenhaven.org · https://brenhaven.org",
    "Ilse Varga | 555-201-3344 | https://brenhaven.org",
    "https://brenhaven.org | +44 20 7946 0958",
])
def test_a_url_on_a_contact_strip_is_personal(text):
    assert [p for _s, _e, p in pattern_scan.find_urls(text)] == [True]


@pytest.mark.parametrize("text", [
    "Thanks.\n\nhttps://brenhavenweb.com",
    "Released 2024-01-15 | https://brenhavenweb.com/notes",
    "Version: 3.12 | https://docs.python.org/3/",
    "| 2024-01-15 | QXR-381205774 | https://brenhavenweb.com/notes |",
    "Policy: QXR-381205774\nhttps://brenhavenweb.com",
])
def test_a_cited_url_stays_opaque(text):
    assert [p for _s, _e, p in pattern_scan.find_urls(text)] == [False]
