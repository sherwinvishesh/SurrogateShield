"""J1 / I14 — secrets with a published provider prefix are API keys
without any label next to them. All values below are made up, and each
is split in two so that no key-shaped literal is in the source (push
protection flags made-up tokens too)."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


def keys(text):
    return [e.text for e in ps.scan(text) if e.type == "api_key"]


@pytest.mark.parametrize("value", [
    "xo" + "xb-1234567890-0987654321-AbCdEfGhIj",
    "xo" + "xp-2222222222-3333333333-abcdef0123",
    "xa" + "pp-1-A0123456789-abcdef0123",
    "gh" + "s_abcdefghijklmnopqrstuvwx",
    "gi" + "thub_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    "gl" + "pat-abcdefghij0123456789",
    "np" + "m_abcdefghijklmnopqrstuvwxyz0123456789",
    "hf" + "_abcdefghijklmnopqrstuvwxyz012345",
    "SG" + ".abcdefghijklmnopqrst.abcdefghijklmnopqrstuvwxyz0123",
    "sh" + "pat_0123456789abcdef0123456789abcdef",
    "ya" + "29.a0AfH6SMBabcdefghijklmnopqrstu",
    "AS" + "IAABCDEFGHIJKLMNOP",
    "12" + "3456789:AAHfj3kd93JDkd93kdJD93kdjd93kdJDk3k",
])
def test_J1_I14_provider_token_prefix_is_an_api_key(value):
    assert keys(f"why does this fail with {value} in the header?") == [value]


@pytest.mark.parametrize("text", [
    "the xoxo at the end of the letter", "my hf_ratio variable", "the SG. office",
    "npm_config is fine", "rk_live is a flag name", "ASIA is a continent",
])
def test_J1_I14_prefix_alone_is_not_a_key(text):
    assert keys(text) == []
