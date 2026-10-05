"""Gate J2 — handles named by a platform, a stream, a seller or an account;
web-log users; device identifiers (IMEI, letter-and-digit serials, health
plan group numbers); a phone's extension, an age's unit and a birthday
without its year; a device named after its owner. Model-free.
"""

import re

import pytest

from surrogateshield.core.detection import pattern_scan as ps, structural
from surrogateshield.core.detection.pipeline import _org_is_plausible
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    return [(e.text, e.type) for e in ps.scan(text)]


@pytest.mark.parametrize("text, value", [
    ("my minecraft name is PixelMoth_77 and I got banned", "PixelMoth_77"),
    ("ban appeal please. discord: pixelmoth", "pixelmoth"),
    ("I'm 17 and stream on twitch as lunarkoi_tv, a viewer", "lunarkoi_tv"),
    ("I SENT $600 TO A SELLER CALLED GREENLEAF_GEAR AND", "GREENLEAF_GEAR"),
    ("Find My Device was on with the account r.mwangi.dev. What", "r.mwangi.dev"),
    ("my duo partner xX_VoidRunner_Xx just got banned", "xX_VoidRunner_Xx"),
    ('102.89.14.66 - mbanda_r [12/Mar/2026:09:14:09 +0000] "GET /admin" 403', "mbanda_r"),
])
def test_J2_handle_frames(text, value):
    assert (value, "handle") in found(text)


@pytest.mark.parametrize("text", [
    "discord: great for chatting with friends",
    "set MAX_RETRIES and call get_UserById",
    "log into the account settings.json file",
    "log into your account at www.chase.com now",
    "run `my_Var_Name` in the shell",
    '1.2.3.4 - - [12/Mar/2026:09:14:09 +0000] "GET /" 200',
    '1.2.3.4 - root [12/Mar/2026:09:14:09 +0000] "GET /" 200',
])
def test_J2_handle_frames_leave_non_handles(text):
    assert not [t for t, typ in found(text) if typ == "handle"]


@pytest.mark.parametrize("text, value", [
    ("IMEI 356938107241598, serial F2LXQ9K8N72P.", "356938107241598"),
    ("It's a ThinkPad X1 Carbon, serial PF3K8L2M, and", "PF3K8L2M"),
    ("Member ID W2281-5530-07, group 44190, patient", "44190"),
])
def test_J2_device_and_plan_identifiers(text, value):
    assert (value, "id_number") in found(text)


def test_J2_group_number_needs_plan_context():
    assert found("the ops group 2231 meets on Friday") == []


def test_J2_phone_extension_age_unit_and_birthday():
    assert ("0161 496 0732 extension 214", "phone_uk") in found(
        "call us back on 0161 496 0732 extension 214, thanks")
    assert ("Age – 74 years", "age") in found("Age – 74 years, DOB – 05/01/1952")
    assert ("9 SEPTEMBER", "dob") in found("MY SON TURNED 18 ON 9 SEPTEMBER AND THE BANK")
    assert ("March 3rd", "dob") in found("her birthday is March 3rd!")


def test_J2_phone_extension_surrogate_keeps_its_label():
    out = MimicGen(seed=1).generate(DetectedEntity("0161 496 0732 extension 214", 0, 27,
                                                   "phone_uk", 1.0, "pattern"))
    assert re.fullmatch(r"0\d{3} \d{3} \d{4} extension \d{3}", out), out
    assert "496 0732" not in out


def hosts(text):
    added, _ = structural.detect(text, ps.scan(text))
    return [e.text for e in added if e.type == "hostname"]


def test_J2_device_named_after_its_owner():
    assert hosts("DHCP table: 'Galaxy-S23-Ritika' 192.168.1.88") == ["Galaxy-S23-Ritika"]
    for text in ["the iPhone-15-Pro and the Galaxy-S23-Ultra compare well",
                 "my MacBook's serial is on the box", "we formed an S-corp last year"]:
        assert hosts(text) == []
    out = MimicGen(seed=2).generate(DetectedEntity("Galaxy-S23-Ritika", 0, 17, "hostname",
                                                   0.9, "structural"))
    assert out.startswith("Galaxy-S23-") and "Ritika" not in out


def test_J2_product_with_model_code_is_not_an_org():
    text = "It's a ThinkPad X1 Carbon, serial"
    ent = DetectedEntity("ThinkPad X1 Carbon", 7, 25, "ORG", 0.85, "ner")
    assert not _org_is_plausible(ent, text)


def test_J2_file_name_words_do_not_spread_as_names():
    text = "named Kateryna_Bondar_Resume_2026 — tailor my resume please"
    added, _ = structural.detect(text, ps.scan(text))
    assert "resume" not in [e.text for e in added]
