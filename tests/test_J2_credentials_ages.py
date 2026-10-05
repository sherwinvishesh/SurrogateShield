"""Gate J2 — credentials, ages and birth dates in the general forms people
type: misspelt "pasword", "pw", YAML "pass:", grouped backup codes, base32
2FA keys, SMS codes as pasted, quoted passwords, seed phrases; ages in a CSV
column, "(29F)", "Esther is 13 and…", "Nana's 80th"; birth dates with
non-English month names and bare birth years.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


@pytest.mark.parametrize("text, value", [
    ("he knows my pasword is Sunflower!882 bc", "Sunflower!882"),
    ("my wifi pw is Ghost-Lantern-41 how", "Ghost-Lantern-41"),
    ("my wifi password is tangerine-otter-57 and", "tangerine-otter-57"),
    ("your Kloverbank code is 482 913, never share it", "482 913"),
    ('  pass: "m0ss&Quill!Sept"\n', "m0ss&Quill!Sept"),
    ("curl -H 'X-Api-Key: nimbus_9c41f0e2b7d84a6f93e1' https://x", "nimbus_9c41f0e2b7d84a6f93e1"),
    ("enabled 2FA: MZXW 6YTB OI3K 5QPL R2DN 4HVE. can", "MZXW 6YTB OI3K 5QPL R2DN 4HVE"),
    ("G-482913 is your Google verification code", "G-482913"),
    ("529 114 is your Instagram code", "529 114"),
    ("my wifi key is k7Hd92mPq", "k7Hd92mPq"),
    ('wifi password: "sunshine"', "sunshine"),
    ("seed phrase: apple banana cherry dog eagle fork grape house igloo jelly kite lemon",
     "apple banana cherry dog eagle fork grape house igloo jelly kite lemon"),
])
def test_J2_credential_is_masked(text, value):
    assert value in [e.text for e in ps.scan(text) if e.type == "credential"]


def test_J2_every_backup_code_in_a_list_is_masked():
    codes = [e.text for e in ps.scan("backup codes: 4829 1173, 6630 2981, 0915 4472. Where")]
    assert codes == ["4829 1173", "6630 2981", "0915 4472"]


@pytest.mark.parametrize("text, values", [
    ("who's bringing the cake for Nana's 80th\n[19:03]", ["80th"]),
    ("Name,Age,Allergy,Parent phone\nKai Nakamura,8,peanuts,0412 773 519\n"
     "Isla Ferguson,10,none,0400 111 222", ["8", "10"]),
    ("My grandad Tadeusz is 87 and lives alone", ["87"]),
    ("he farmed sheep for 50 years and turns 80 in december", ["turns 80"]),
    ("Age – 74 years, DOB", ["Age – 74 years"]),
    ("My twins Noa and Eitan are 11 and my youngest, Talia, is 4. Is", ["11", "4"]),
    ("Esther is 13 and has dyslexia", ["13"]),
    ("so my mum (gloria, 63) has been", ["63"]),
    ("name 'Dmitri, 39' with the bio", ["39"]),
    ("I (29F) and my bf (31M) have", ["29", "31"]),
    ("[F/24] need advice", ["24"]),
    ("my kids are 7 and 9", ["7", "9"]),
])
def test_J2_age_is_masked(text, values):
    assert [e.text for e in ps.scan(text) if e.type == "age"] == values


@pytest.mark.parametrize("text, value", [
    ("Sono Chiara, nata il 4 aprile 1987, codice", "4 aprile 1987"),
    ("nacido el 12 de marzo de 1990", "12 de marzo de 1990"),
    ("Born: 1997 · Nationality", "1997"),
    ("born in March 1987", "March 1987"),
    ("DOB 1987/04/05", "1987/04/05"),
])
def test_J2_birth_date_is_masked(text, value):
    assert value in [e.text for e in ps.scan(text) if e.type == "dob"]


@pytest.mark.parametrize("text", [
    "reset my password for Gmail",
    "THIS IS VERY GOOD NEWS FROM THE TEAM TODAY",
    "the zip code is 852 810",
    "error code is 500 123",
    "the license key is in the email",
    "the pass is closed",
    "Python is 3.",
    "It is 5 and",
    "Version is 2.",
    "the company's 50th anniversary",
    "(Smith, 2019)",
    "see (Fig, 3)",
    "press F5 (5F) floor",
    "my 3rd attempt",
    "Order,Qty,Price\nA1,3,4.50",
    "the 4 aprile meeting",
    "in 1997 we moved",
    "the Born Identity",
])
def test_J2_credential_age_rules_leave_other_text(text):
    assert not [e for e in ps.scan(text) if e.type in ("credential", "age", "dob")]
