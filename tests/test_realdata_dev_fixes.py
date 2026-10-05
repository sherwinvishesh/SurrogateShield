"""Bugs found while diagnosing SurrogateShield's misses on the real-data dev
split (E5, Phase 5). Model-free; numbers are in Ofcom's drama ranges
(07700 900xxx, 020 7946 0xxx); card numbers are the
issuers' published test numbers."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    return {(e.text, e.type) for e in ps.scan(text)}


# phone_intl hands every "+44" followed by " ", "-" or "." to phone_uk, which
# took only the space: "+44.7700.900123" matched no rule at all.
@pytest.mark.parametrize("value", [
    "+44.7700.900123", "+44-7700-900123", "+44.20.7946.0958", "+44-20-7946-0958",
    "+44.117.496.0123", "+44 7700 900123", "+447700900123", "+44.7700 900123",
])
def test_uk_plus44_takes_the_separators_phone_intl_leaves_to_it(value):
    assert (value, "phone_uk") in found(f"Reach me at {value} today.")


@pytest.mark.parametrize("value", ["07700 900123", "07700-900123", "020 7946 0958"])
def test_uk_trunk_form_is_unchanged(value):
    assert (value, "phone_uk") in found(f"Call {value} please.")


@pytest.mark.parametrize("text", [
    "Version 07700.900123 of the build",     # the trunk-0 form still takes no dots
    "ratio 0.7700900123",
    "id x+44.7700.900123",                   # glued to a word
])
def test_dots_stay_out_of_the_trunk_form(text):
    assert not any(t == "phone_uk" for _v, t in found(text))


def test_dotted_uk_surrogate_keeps_country_code_and_dots():
    value = "+44.7700.900123"
    for seed in range(10):
        out = MimicGen(seed=seed).generate(DetectedEntity(value, 0, len(value), "phone_uk", 1.0, "pattern"))
        assert out.startswith("+44.7") and out.count(".") == 2 and len(out) == len(value) and out != value


# E6's wrong-credit_card condition swaps the card rule's separators for
# "./_"; the validator stripped only spaces and dashes and int(".") raised.
# It now Luhn-checks the digits of whatever matched.
def test_card_validator_takes_the_digits_whatever_separates_them():
    import re
    for value in ("4111.1111.1111.1111", "4111_1111_1111_1111", "4111/1111/1111/1111"):
        m = re.search(r"(?:\d[./_]?){12,18}\d", f"paid with {value} today")
        assert ps._card_validator(m)
    m = re.search(r"(?:\d[./_]?){12,18}\d", "paid with 4111.1111.1111.1112 today")
    assert not ps._card_validator(m)                      # Luhn still decides


@pytest.mark.parametrize("value", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111"])
def test_card_rule_is_unchanged(value):
    assert (value, "credit_card") in found(f"Card {value} on file.")


# An e-mail whose domain has an empty label ("x@y..com", a typo PatternScan
# accepts) gave _draw an empty original; "" is in every string, so no
# candidate was ever free and the surrogate generator never returned.
@pytest.mark.parametrize("value", ["ann@example..org", "bo@..net", "c.d@mail..co.uk"])
def test_email_with_an_empty_domain_label_gets_a_surrogate(value):
    import signal

    def timeout(*_):
        raise TimeoutError(value)

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(5)
    try:
        for seed in range(5):
            out = MimicGen(seed=seed).generate(DetectedEntity(value, 0, len(value), "email", 1.0, "pattern"))
            assert "@" in out and out != value
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
