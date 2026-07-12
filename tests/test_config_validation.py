"""Config knobs: defaults, validation errors, both trees."""

import pytest

import surrogateshield as ss
from surrogateshield._state import cfg


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    ss.config(detailed_view=False)  # restore defaults after each test


# ── Defaults (the high-accuracy path out of the box) ──────────────────────────

def test_library_defaults():
    ss.config(detailed_view=False)
    assert cfg.address_mode == "shift"
    assert cfg.address_shift_range == 1
    assert cfg.verify_addresses is False          # network is opt-in
    assert cfg.fuzzy_threshold == 85
    assert cfg.spacy_model == "en_core_web_lg"
    assert cfg.context_guard_enabled is True
    assert cfg.context_guard_model == "dslim/distilbert-NER"
    assert cfg.context_guard_device == -1
    assert cfg.entity_trace_high_threshold == 0.85
    assert cfg.entity_trace_low_threshold == 0.60
    assert cfg.context_guard_threshold == 0.70
    assert cfg.entity_trace_fallback_threshold == 0.65
    assert cfg.service is True
    assert cfg.pii_off == []
    assert cfg.pii_mem == "temp"


def test_root_config_defaults_and_validation():
    import config as root_config
    assert root_config.ADDRESS_MODE == "shift"
    assert root_config.ADDRESS_SHIFT_RANGE == 1
    assert root_config.SERVICE_QUERY_VERIFY_ADDRESSES is False
    assert root_config.FUZZY_MATCH_THRESHOLD == 85
    assert root_config.CONTEXT_GUARD_DEVICE == -1
    assert root_config.VERSION == "2.0.0"
    root_config.validate_config()  # must not raise on shipped defaults


# ── Valid customization round-trips ───────────────────────────────────────────

def test_custom_values_accepted():
    ss.config(
        detailed_view=False,
        address_mode="auto",
        address_shift_range=4,
        verify_addresses=True,
        fuzzy_threshold=70,
        pii_off=["phone", "name", "location"],
        context_guard_model="dslim/bert-base-NER",
        context_guard_device=0,
    )
    assert cfg.address_mode == "auto"
    assert cfg.address_shift_range == 4
    assert cfg.verify_addresses is True
    assert cfg.fuzzy_threshold == 70
    assert cfg.pii_off == ["phone", "name", "location"]
    assert cfg.context_guard_model == "dslim/bert-base-NER"
    assert cfg.context_guard_device == 0


@pytest.mark.parametrize("mode", ["shift", "replace", "auto"])
def test_all_address_modes_accepted(mode):
    ss.config(detailed_view=False, address_mode=mode)
    assert cfg.address_mode == mode


# ── Invalid values raise ValueError with actionable messages ──────────────────

@pytest.mark.parametrize("kwargs, fragment", [
    (dict(address_mode="wobble"), "address_mode"),
    (dict(address_mode="SHIFT"), "address_mode"),
    (dict(address_shift_range=0), "address_shift_range"),
    (dict(address_shift_range=-2), "address_shift_range"),
    (dict(address_shift_range=1.5), "address_shift_range"),
    (dict(address_shift_range=True), "address_shift_range"),
    (dict(fuzzy_threshold=101), "fuzzy_threshold"),
    (dict(fuzzy_threshold=-1), "fuzzy_threshold"),
    (dict(entity_trace_high_threshold=1.5), "entity_trace_high_threshold"),
    (dict(entity_trace_low_threshold=-0.1), "entity_trace_low_threshold"),
    (dict(context_guard_threshold=2), "context_guard_threshold"),
    (dict(entity_trace_fallback_threshold=7), "entity_trace_fallback_threshold"),
    (dict(spacy_model=""), "spacy_model"),
    (dict(context_guard_model="  "), "context_guard_model"),
    (dict(context_guard_device="gpu"), "context_guard_device"),
    (dict(pii_off=["nonsense_type"]), "pii_off"),
    (dict(pii_mem="/definitely/not/a/real/dir"), "pii_mem"),
])
def test_invalid_values_rejected(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment.replace("/", ".")):
        ss.config(detailed_view=False, **kwargs)


def test_invalid_config_leaves_previous_state_intact():
    ss.config(detailed_view=False, address_mode="replace")
    with pytest.raises(ValueError):
        ss.config(detailed_view=False, address_mode="bogus")
    assert cfg.address_mode == "replace"  # validation happens before mutation


@pytest.mark.parametrize("alias", [
    "phone", "name", "names", "location", "org", "email", "ssn", "dob",
    "address", "zip", "postcode", "credit_card", "ip_address", "api_key",
    "crypto", "bank", "license", "gender_indicator",
])
def test_documented_pii_off_aliases_accepted(alias):
    ss.config(detailed_view=False, pii_off=[alias])
    assert cfg.pii_off == [alias]
