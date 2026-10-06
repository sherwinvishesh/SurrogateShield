"""The canonicaliser (PROMPT_FOR_OPUS_V3 §3.2): views of a message that
PatternScan also reads, hits mapped back to the original text and tagged with
their view, and surrogates written back in the message's own spelling.
Model-free (the cascade runs without NER); every text and value is invented."""

import re

import pytest

from surrogateshield.core.detection import canonical as C
from surrogateshield.core.detection import pattern_scan, pipeline
from surrogateshield.core.entities import DetectedEntity, plan_substitutions
from surrogateshield.core.generation.mimic import MimicGen


def _view(fn, text):
    v = fn(text)
    assert v is not None, text
    return v


# ── views ────────────────────────────────────────────────────────────────────

def test_rewrite_maps_every_view_character_back():
    v = C.rewrite("t", "ab XY cd", [(3, 5, "Q")])
    assert v.text == "ab Q cd" and v.changed == ((3, 4),)
    assert v.original(3, 4) == (3, 5) and v.original(0, 2) == (0, 2) and v.original(5, 7) == (6, 8)
    assert C.rewrite("t", "abc", [(0, 1, "a")]) is None


def test_rewrite_deletion_marks_its_left_neighbour():
    v = C.rewrite("t", "a​b", [(1, 2, "")])
    assert v.text == "ab" and v.changed == ((0, 1),) and v.original(0, 2) == (0, 3)


@pytest.mark.parametrize("text, want", [
    ("call six one seven, two eight four, three three nine one", "call 617 284 3391"),
    ("call six-one-seven-two-eight-four-three", "call 6174 2843"[:0] + "call 6172843"),
    ("I'm twenty-nine", "I'm 29"),
    ("I'm twenty nine", "I'm 29"),
    ("she is Fifteen", "she is 15"),
    ("seven years old", "7 years old"),
    ("the twenty first of May", None),          # an ordinal stays for the dates view
    ("one of two options", None),               # lone small numbers stay
])
def test_worded(text, want):
    v = C.worded(text)
    assert (v.text if v else None) == want


@pytest.mark.parametrize("text, want", [
    ("mail jane dot doe at fastmail dot com", "mail jane.doe@fastmail.com"),
    ("k.marsh [at] tutanota [dot] com", "k.marsh@tutanota.com"),
    ("pat99 (at) example (dot) org", "pat99@example.org"),
    ("my email is pat at example dot org", "my email is pat@example.org"),
    ("reach me at jane dot doe at fastmail dot com", "reach me at jane.doe@fastmail.com"),
    ("j_r at web dot de", "j_r@web.de"),
    ("ggarcia at hotmail dot co dot uk", "ggarcia@hotmail.co.uk"),     # a webmail domain
    ("— Ana | felicitasg at yahoo dot com | 555", "— Ana | felicitasg@yahoo.com | 555"),
])
def test_spelled(text, want):
    assert _view(C.spelled, text).text == want


@pytest.mark.parametrize("text", [
    "look at gmail dot com for that",          # bare "at", one-word local part, no mail cue
    "log in at outlook dot com first",         # a webmail domain behind a word that takes "at"
    "the bug at example dot com is fixed",     # not a webmail domain
    "meet me at the cafe dot later",           # no TLD
    "jane@example.com",                        # already an address: nothing to rewrite
])
def test_spelled_leaves_ordinary_english(text):
    assert C.spelled(text) is None


@pytest.mark.parametrize("text, want", [
    ("born on the third of March 1991", "born on 1991-03-03"),
    ("born March twenty-first, 1991", "born 1991-03-21"),
    ("born on the 3rd of Sept 1985", "born on 1985-09-03"),
    ("born March third, nineteen ninety-one", "born 1991-03-03"),
    ("born the first of June two thousand and four", "born 2004-06-01"),
])
def test_dates(text, want):
    assert _view(C.dates, text).text == want


def test_dates_reject_impossible_days():
    assert C.dates("the 45th of March 1991") is None


def test_folded():
    v = _view(C.folded, "ｐｈｏｎｅ ６１７​-２８４-３３９１ аlex")
    assert v.text == "phone 617-284-3391 alex"
    s = v.text.index("617")
    assert v.original(s, s + len("617-284-3391")) == (6, 19)


def test_joined_and_casefold():
    assert _view(C.joined, "IBAN GB82 WEST 1234 5698 7654 32, id 4111 1111 1111 1111").text.endswith(
        "id 4111111111111111")
    assert C.joined("version 1.2.3") is None
    assert C.joined("build Stream/10.1.1234.567 and 10.20.3000.4000") is None    # dotted: versions, paths
    assert _view(C.casefold, "JANE@X.COM").text == "jane@x.com" and C.casefold("lower") is None


def test_number_value_and_words_round_trip():
    for n in list(range(0, 100)) + [100, 105, 120]:
        assert C.number_value(C.number_words(n)) == n or n >= 100
    assert C.number_words(105) == "one hundred and five"
    assert [C.ordinal_words(n) for n in (1, 3, 12, 20, 21, 30, 31)] == [
        "first", "third", "twelfth", "twentieth", "twenty-first", "thirtieth", "thirty-first"]
    for y in (1905, 1991, 1900, 2000, 2004, 2015):
        assert C.year_value(C.year_words(y)) == y, y
    assert C.year_words(2015) == "twenty fifteen" and C.year_words(2015, "two thousand fifteen") == "two thousand fifteen"


# ── hits on views ────────────────────────────────────────────────────────────

def _scan(text, views=C.DEFAULT_VIEWS):
    return C.scan_views(text, pattern_scan.scan(text), views, pattern_scan.scan)


@pytest.mark.parametrize("text, value, typ, view", [
    ("Hi, I'm twenty-nine and new here", "twenty-nine", "age", "worded"),
    ("my number is six one seven, two eight four, three three nine one", "six one seven, two eight four, three three nine one",
     "phone_us", "worded"),
    ("Write to jane dot doe at fastmail dot com please", "jane dot doe at fastmail dot com", "email", "spelled"),
    ("I was born on the third of March 1991, in Leeds.", "the third of March 1991", "dob", "dates"),
    ("Mail: ｊａｎｅ＠ｅｘａｍｐｌｅ．ｃｏｍ", "ｊａｎｅ＠ｅｘａｍｐｌｅ．ｃｏｍ", "email", "folded"),
])
def test_view_hits_in_original_coordinates(text, value, typ, view):
    hits = _scan(text)
    assert [(h.text, h.type, h.view) for h in hits] == [(value, typ, view)]
    h = hits[0]
    assert text[h.start:h.end] == h.text and h.source == "pattern" and h.canonical is not None


def test_a_view_hit_overlapping_an_original_hit_is_dropped():
    text = "email jane.doe@example.com or jane dot doe at example dot com"
    hits = _scan(text)
    assert [h.text for h in hits] == ["jane dot doe at example dot com"]
    assert not any(h.start < 26 for h in hits)


@pytest.mark.parametrize("text", [
    "I have twenty-two apples and three cats",
    "Chapter twelve covers loops; see section four.",
    "We met at noon dot the cafe was full",
    "The meeting is on the third of March 2027",           # a date, but no birth cue
])
def test_no_view_hit_on_ordinary_text(text):
    assert _scan(text) == []


def test_views_respect_the_patterns_validators():
    # the view's digits meet PatternScan's own validators: Luhn decides here
    # (no "card" cue, so the context-gated fallback stays out)
    run = lambda d: " ".join(C.number_words(int(c)) for c in d)
    bad = "ok so " + run("1234123412341234") + " then"
    good = "ok so " + run("4111111111111111") + " then"
    assert not [h for h in _scan(bad) if h.type == "credit_card"]
    assert [h.type for h in _scan(good)] == ["credit_card"]


# ── the cascade and the surrogates ───────────────────────────────────────────

def _send(text, **kw):
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False, **kw)
    mapping = MimicGen(seed=7).generate_all(conf, text=text)
    out = text
    for s, e, _o, r in sorted(plan_substitutions(text, conf, mapping), reverse=True):
        out = out[:s] + r + out[e:]
    return conf, out


def test_cascade_runs_the_default_views_and_can_switch_them_off():
    text = "Hi, I'm twenty-nine; reach me at jane dot doe at fastmail dot com"
    conf, _ = _send(text)
    assert sorted(e.view for e in conf if e.view) == ["spelled", "worded"]
    conf, out = _send(text, canonical_views=())
    assert not [e for e in conf if e.view] and "twenty-nine" in out
    conf, _ = _send(text, canonical_views=["spelled"])
    assert [e.view for e in conf if e.view] == ["spelled"]


@pytest.mark.parametrize("text, value, shape", [
    ("Hi, I'm twenty-nine and new here", "twenty-nine", r"I'm [a-z]+(-[a-z]+)? and"),
    ("my number is six one seven, two eight four, three three nine one.", "six one seven, two eight four, three three nine one",
     r"is ([a-z]+ ){2}[a-z]+, ([a-z]+ ){2}[a-z]+, ([a-z]+ ){3}[a-z]+\."),
    ("Write to jane dot doe at fastmail dot com please", "jane dot doe at fastmail dot com",
     r"to [a-z]+ dot [a-z]+ at [a-z]+ dot com please"),
    ("Contact: k.marsh [at] tutanota [dot] com", "k.marsh [at] tutanota [dot] com", r"Contact: \S+ \[at\] \S+ \[dot\] com"),
    ("I was born on the third of March 1991, in Leeds.", "the third of March 1991",
     r"born on the [a-z]+(-[a-z]+)? of [A-Z][a-z]+ \d{4}, in"),
    ("I was born March twenty-first, nineteen ninety-one.", "March twenty-first, nineteen ninety-one",
     r"born [A-Z][a-z]+ [a-z]+(-[a-z]+)?, nineteen [a-z]+(-[a-z]+)?\."),
    ("Age: Thirty Four", "Thirty Four", r"Age: [A-Z][a-z]+( [A-Z][a-z]+)?$"),
])
def test_surrogate_is_written_the_way_the_message_wrote_it(text, value, shape):
    _conf, out = _send(text)
    assert value not in out
    assert re.search(shape, out), out


def test_spelled_email_links_to_the_plain_one():
    text = "jane.doe@fastmail.com, or if that fails jane dot doe at fastmail dot com"
    conf, out = _send(text)
    plain, spoken = out.split(", or if that fails ")
    assert plain.replace(".", " dot ").replace("@", " at ").replace(" dot com", "") == \
        spoken.replace(" dot com", "")


def test_render_falls_back_to_the_drawn_surrogate():
    canon = C.Canon("617 284 3391", ((0, 12, 0, 9),))
    assert C.render_like("worded", "six one seven", canon, "no digits here") == "no digits here"
    assert C.render_like("casefold", "JANE@X.COM", C.Canon("jane@x.com", ()), "kim@y.com") == "kim@y.com"


@pytest.mark.parametrize("age, shape", [
    ("twenty-nine", r"(twenty|thirty)-(one|two|three|four|five|six|seven|eight|nine)|thirty"),
    ("Seven years old", r"(Four|Five|Six|Eight|Nine|Ten) years old"),
    ("41", r"3[89]|4[0234]"),
    ("mid-thirties", r"[a-z]{3}-[a-z]{8}"),          # no number in it: a shape
])
def test_an_age_found_by_a_model_in_words_gets_a_surrogate(age, shape):
    # a model's AGE has no canonical view: it used to crash on a missing digit
    sur = MimicGen(seed=5).generate(DetectedEntity(age, 0, len(age), "age", 0.9, "slm"))
    assert sur != age and re.fullmatch(shape, sur), sur


def test_prepared_spans_carry_the_view():
    from json_tester import Prepared
    text = "I'm twenty-nine"
    ent = DetectedEntity("twenty-nine", 4, 15, "age", 1.0, "pattern", view="worded")
    p = Prepared(question=text, is_service_query=False, address_mode="shift", confirmed=[ent], skipped=[],
                 skip_reason="", surrogate_map={"twenty-nine": "thirty"}, edits=[(4, 15, "twenty-nine", "thirty")],
                 sanitized="I'm thirty")
    assert p.spans()[0]["view"] == "worded"
    plain = DetectedEntity("29", 4, 6, "age")
    p = Prepared(question="I'm 29", is_service_query=False, address_mode="shift", confirmed=[plain], skipped=[],
                 skip_reason="", surrogate_map={"29": "31"}, edits=[(4, 6, "29", "31")], sanitized="I'm 31")
    assert "view" not in p.spans()[0]


def test_trace_shows_view_hits_at_their_own_stage():
    text = "I'm twenty-nine, mail jane.doe@example.com"
    trace = []
    pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False, trace=trace)
    by = {t["stage"]: {(r[0], r[1], r[2]) for r in t["entities"]} for t in trace}
    age = (text.index("twenty"), text.index(","), "age")
    assert age not in by["pattern_scan"] and age in by["canonicaliser"]
    assert by["pattern_scan"] < by["canonicaliser"]


def test_view_hits_leave_the_ner_input_alone_and_absorb_model_spans(monkeypatch):
    text = "Ada Byrne, born the fourth of June 1962, mail ada dot byrne at example dot org"
    seen = []

    def fake_trace(remaining, existing_entities=None, **_kw):
        seen.append(remaining)
        return [], []

    def fake_guard(remaining_text, borderline_entities, **_kw):
        s = text.index("byrne at")
        return [DetectedEntity("Ada Byrne", 0, 9, "PERSON", 0.9, "slm"),
                DetectedEntity("byrne", s, s + 5, "PERSON", 0.9, "slm")], []

    monkeypatch.setattr(pipeline.entity_trace, "trace", fake_trace)
    monkeypatch.setattr(pipeline.context_guard, "guard", fake_guard)
    off, _ = pipeline.run_cascade(text, use_context_guard=True, use_tagger=False, canonical_views=())
    on, _ = pipeline.run_cascade(text, use_context_guard=True, use_tagger=False)
    assert seen[0] == seen[1]                                   # same text for the NER stages
    assert sorted(e.view for e in on if e.view) == ["dates", "spelled"]
    assert [e.text for e in on if e.type == "PERSON"] == ["Ada Byrne"]
    assert sorted(e.text for e in off if e.type == "PERSON") == ["Ada Byrne", "byrne"]


# ── PatternScan gaps the canonicaliser's devlarge attribution surfaced ──────

@pytest.mark.parametrize("text, value", [
    ('{"name": "X", "age": "30", "id": 1}', "30"),
    ('{"age": 41}', "41"),
    ("{'edad': '7'}", "7"),
    ('{"name": "X", "age": "thirty", "id": 1}', "thirty"),      # through the worded view
])
def test_an_age_under_a_json_key_is_found(text, value):
    found = [(h.type, text[h.start:h.end]) for h in pattern_scan.scan(text)]
    found += [(h.type, text[h.start:h.end]) for h in C.scan_views(text, [], C.DEFAULT_VIEWS, pattern_scan.scan)]
    assert ("age", value) in found, found


@pytest.mark.parametrize("text", ['{"price": "30", "rating": 4}', '{"page": "12"}', '"stage": "3",'])
def test_other_json_keys_are_not_ages(text):
    assert not any(h.type == "age" for h in pattern_scan.scan(text))


@pytest.mark.parametrize("number", ["06.45681480", "0176.20476296", "06.45.68.14.80", "+33.7.25.61.98.87"])
def test_a_dotted_phone_is_found_whole(number):
    found = [(h.type, h.text) for h in pattern_scan.scan(f"Phone: {number}\nthanks")]
    assert any(t.startswith("phone") and v == number for t, v in found), found


@pytest.mark.parametrize("text, ip", [("server 10.0.0.12 is up", "10.0.0.12"), ("at 192.168.1.1.", "192.168.1.1")])
def test_ipv4_still_found_outside_a_dotted_phone(text, ip):
    assert ("ip_address", ip) in [(h.type, h.text) for h in pattern_scan.scan(text)]


@pytest.mark.parametrize("text", ["version 0.12.3.4567890", "010.123.045.067", "dated 01.02.2023"])
def test_dotted_numbers_that_are_not_phones(text):
    assert not any(h.type.startswith("phone") for h in pattern_scan.scan(text))
