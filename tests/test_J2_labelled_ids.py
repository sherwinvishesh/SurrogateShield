"""Gate J2 — labelled identifiers that went out unmasked on the blind dev3
split: the label or the value shape was not known. Each one is a general
form (an NI number, a plate after "rego"/"kenteken", a dashed licence
number, a national-ID acronym), not a phrase from the corpus.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


@pytest.mark.parametrize("text, value", [
    ("National Insurance no.: JT 48 21 73 C\n", "JT 48 21 73 C"),
    ("Mitgliedsnummer KF-207733.", "KF-207733"),
    ("o código de cliente 3001 7745 21.", "3001 7745 21"),
    ("my SIN is 512 448 903.", "512 448 903"),
    ("my travel insurance (policy TRV-88-2039115)", "TRV-88-2039115"),
    ("MY LICENSE NUMBER IS D482-7715-9036 AND", "D482-7715-9036"),
    ("Plate was KX19 RZT, VIN", "KX19 RZT"),
    ("voor kenteken 52-RTK-9 ingetrokken", "52-RTK-9"),
    ("i got there plate its BRT 4471 (silver", "BRT 4471"),
    ("rego 1XY-4KT (VIC)", "1XY-4KT"),
    ("Reg: EL 48213\n", "EL 48213"),
    ("my TFN is 123 456 782", "123 456 782"),
    ("CURP: GOMR850101HDFRRN09", "GOMR850101HDFRRN09"),
    ("RUT 12.345.678-5", "12.345.678-5"),
    ("NRIC S1234567D", "S1234567D"),
    ("Personnummer 19850101-1234", "19850101-1234"),
    ("licence # 7741-209-88", "7741-209-88"),
])
def test_J2_labelled_id_is_masked_whole(text, value):
    assert value in [e.text for e in ps.scan(text)]


@pytest.mark.parametrize("text", [
    "our privacy policy 2024 update",          # a year is not an ID
    "the reg ex is fine",
    "price tag is 45000",                      # a plate has letters
    "sin 2000 euros",                          # Spanish "sin", lower case
    "the PAN 2024 report",
    "Claim ref C-449201.",                     # a case reference is kept (GUIDE)
])
def test_J2_id_labels_do_not_claim_other_numbers(text):
    assert not ps.scan(text)
