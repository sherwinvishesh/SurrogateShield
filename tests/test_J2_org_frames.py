"""Gate J2 — organisations named by a frame: a labelled employer, a legal
suffix, a payroll line, "a shop called X", a signature's "Title | Org",
Romance and lowercase institutions, an English parish or practice, and
Japanese / Chinese schools and companies; Hindi names after a kin word.
Surrogates keep the kind word and the script. Model-free.
"""

import re

import pytest

from surrogateshield.core.detection import pattern_scan, structural
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation import identity
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    added, _ = structural.detect(text, pattern_scan.scan(text))
    return {(e.text, e.type) for e in added}


def orgs(text):
    return {t for t, typ in found(text) if typ == "ORG"}


@pytest.mark.parametrize("text, org", [
    ("Applicant: Ana Puerta\nEmployer: Grupo Andino S.A.C.\nMonthly income", "Grupo Andino S.A.C."),
    ("2026-09-05,PAYROLL HEXAGON DENTAL,2140.55", "HEXAGON DENTAL"),
    ("the coffee shop where i work its called bean there done that on 5th street",
     "bean there done that"),
    ("Thanks,\nHenryk\nEvents Coordinator | Lantern Hall Collective\nm. +48", "Lantern Hall Collective"),
    ("Need a bulletin: St. Brigid's Parish in Kilrush is hosting", "St. Brigid's Parish"),
    ("hi it's dr joey from lakeside family practice calling", "lakeside family practice"),
    ("il dottor Bellandi dello Studio Medico San Rocco a Cortona.", "Studio Medico San Rocco"),
    ("l'absence de mon fils au collège Jean-Moulin de Saint-Flour. Il", "collège Jean-Moulin de Saint-Flour"),
    ("娘が通っている青葉台さくら小学校の面談があります", "青葉台さくら小学校"),
    ("在合肥一家叫“蓝鲸数智”的小公司做前端", "蓝鲸数智"),
])
def test_J2_org_frame_finds_the_org(text, org):
    assert org in orgs(text)


@pytest.mark.parametrize("text", [
    "Employer: N/A\nIncome: none",
    "the shop is called out of stock right now",
    "I love the band called Queen",
    "Senior Engineer | Remote",
    "Our High School reunion is Friday",
    "The General Practice guidelines changed",
    "我们公司叫什么不重要",
    "皆さん、こんにちは。先生に伝えたい",
])
def test_J2_org_frame_leaves_non_orgs(text):
    assert not orgs(text)


@pytest.mark.parametrize("text, name", [
    ("मेरे पिताजी रमेश चंद्र त्रिपाठी 68 साल के हैं", "रमेश चंद्र त्रिपाठी"),
    ("最近、同級生の田村くんとトラブルがあったようです", "田村"),
])
def test_J2_cjk_and_hindi_names(text, name):
    assert (name, "PERSON") in found(text)


@pytest.mark.parametrize("text", ["मेरी माँ बीमार हैं", "मेरे पिता बहुत बीमार हैं"])
def test_J2_hindi_kin_word_then_a_predicate_is_not_a_name(text):
    assert not [t for t, typ in found(text) if typ == "PERSON"]


def test_J2_devanagari_name_words_keep_their_vowel_signs():
    assert [w.text for w in identity._parse("रमेश चंद्र त्रिपाठी")] == ["रमेश", "चंद्र", "त्रिपाठी"]
    out = MimicGen(seed=4).generate(DetectedEntity("रमेश त्रिपाठी", 0, 13, "PERSON", 0.9, "ner"))
    words = out.split()
    assert len(words) == 2 and "रमेश" not in out and "त्रिपाठी" not in out
    assert all(re.fullmatch(r"[ऀ-ॿ]+", w) for w in words), out


def gen(text, seed=5):
    return MimicGen(seed=seed).generate(DetectedEntity(text, 0, len(text), "ORG", 0.9, "structural"))


@pytest.mark.parametrize("text, keep, gone", [
    ("lakeside family practice", "family practice", "lakeside"),
    ("Studio Medico San Rocco", "Studio Medico San", "Rocco"),
    ("Our Lady of Lourdes Parish", "Our Lady of", "Lourdes"),
])
def test_J2_institution_surrogate_keeps_its_kind_words(text, keep, gone):
    out = gen(text)
    assert keep in out and gone not in out, out


@pytest.mark.parametrize("text, suffix, script", [
    ("青葉台さくら小学校", "小学校", r"[぀-ヿ一-鿿々ヶ]+"),
    ("蓝鲸数智", "", r"[一-鿿]+"),
])
def test_J2_cjk_org_surrogate_keeps_kind_and_script(text, suffix, script):
    out = gen(text)
    assert out != text and out.endswith(suffix) and re.fullmatch(script, out), out
