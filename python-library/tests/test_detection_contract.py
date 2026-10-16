"""Detection correctness contract (pre-publish audit, pass 4).

The README tables are derived from the code here: the PatternScan table, the
public-type map, the evaluator's label map, the quasi-identifier table and the
module count. The presets, the address modes, the service-query rules and the
quasi-identifier scorer are tested against their documented behaviour.
Everything runs model-free except the tests marked heavy, which run the real
presets:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest python-library/tests/test_detection_contract.py -m heavy
"""

import importlib.util
import random
import re
import sys
from pathlib import Path

import pytest

import surrogateshield as ss
from surrogateshield import Session
from surrogateshield.core.detection import address_parser, pattern_scan, pipeline, quasi_identifier, service_query
from surrogateshield.core.detection.config import INTERNAL_TYPES, PUBLIC_TYPES, preset
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import shift_house_number

heavy = pytest.mark.heavy

LIB_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = LIB_ROOT.parent
LIB_README = LIB_ROOT / "README.md"
ROOT_README = REPO_ROOT / "README.md"

# Twenty messages with every family of PII, shared by the preset tests
MESSAGES = [
    "Hi, I'm Priya Raman and my email is priya.raman@example.com.",
    "Call me at +1-480-555-1234 after 5pm, or text (480) 555-9876.",
    "My SSN is 512-44-7810 and my card is 4111 1111 1111 1111.",
    "I live at 789 Crescent Row, Tempe, AZ 85281 and need a plumber.",
    "Find a coffee shop near 42 Baker Street, London.",
    "Daniel Kowalczyk works at Northwind Traders in Berlin.",
    "I was born on 03/14/1990 and I am 34 years old.",
    "My server is at 192.168.1.100 and its MAC is 00:1A:2B:3C:4D:5E.",
    "Use the key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789 for the demo.",
    "My username is priya_r and my site is https://example.com/u/priya.",
    "Send the parcel to PO Box 1234, Tempe, AZ 85281.",
    "Is there a rehab clinic near 316 Citrus Blvd, Orlando, FL 32801?",
    "Our IBAN is GB29 NWBK 6016 1331 9268 19 and the routing number 021000021.",
    "Passport number 512345678, driver's license number B7654321.",
    "Patient MRN 00482913, employee ID 48213977, VIN 1HGCM82633A004352.",
    "Dr. Alice Nguyen at Deloitte reviewed the contract with Omar Haddad.",
    "I did a 3 point turn and gave the instructor 5 stars.",
    "The meeting is at 10am; bring the quarterly report and a laptop.",
    "My daughter is 7 years old and her school is in New York.",
    "Postcode SW1A 1AA, phone +44 7911 123456, wallet 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2.",
]

# The documented stage table: EntityTrace / ContextGuard / PII tagger per preset
STAGE_TABLE = {
    "balanced": {"entity_trace": True, "context_guard": False, "pii_tagger": True},
    "strict":   {"entity_trace": True, "context_guard": True, "pii_tagger": True},
    "fast":     {"entity_trace": False, "context_guard": False, "pii_tagger": True},
    "classic":  {"entity_trace": True, "context_guard": True, "pii_tagger": False},
}

MODEL_FREE = (preset("balanced")
              .with_stage("entity_trace", enabled=False)
              .with_stage("pii_tagger", enabled=False)
              .with_stage("context_guard", enabled=False))


def _table(text, header):
    """The rows of the markdown table whose header line is *header*, as lists
    of cells, up to the first blank line."""
    assert header in text, header
    rows = []
    for line in text.split(header, 1)[1].splitlines()[1:]:
        if not line.strip():
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and not set(cells[0]) <= {"-", ":"}:
            rows.append(cells)
    assert rows, header
    return rows


def _root_readme():
    if not ROOT_README.exists():
        pytest.skip("the repository README is not available here")
    return ROOT_README.read_text(encoding="utf-8")


def _lib_readme():
    if not LIB_README.exists():
        pytest.skip("the library README is not available here")
    return LIB_README.read_text(encoding="utf-8")


def _mask(mode, text):
    ss.config(detection=MODEL_FREE, address_mode=mode)
    return Session().mask(text)


# ─────────────────────────────────────────────────────────────────────────────
# README tables derived from the code
# ─────────────────────────────────────────────────────────────────────────────

PATTERN_HEADER = "| Internal type | Examples (each is found when scanned on its own) | Validator |"


def _pattern_table():
    rows = {}
    for cells in _table(_root_readme(), PATTERN_HEADER):
        typ = re.findall(r"`([^`]+)`", cells[0])[0]
        rows[typ] = re.findall(r"`([^`]+)`", cells[1])
    return rows


def _scanner_types():
    """Every type pattern_scan.scan can emit: the pattern table, the personal
    URL regex (_URL_RE) and the structured address parser it calls first."""
    return {t for t, _, _ in pattern_scan._PATTERNS} | {"url"}


def test_pattern_table_lists_every_pattern_type_once():
    assert set(_pattern_table()) == _scanner_types() | {"address"}
    assert len(_table(_root_readme(), PATTERN_HEADER)) == len(_scanner_types()) + 1


def test_every_pattern_table_example_scans_to_its_row():
    for typ, examples in _pattern_table().items():
        assert examples, typ
        for example in examples:
            found = {e.type for e in pattern_scan.scan(example)}
            assert typ in found, (typ, example, found)


def test_pattern_scan_comment_states_the_real_type_count():
    assert f"# regex scanner ({len(_scanner_types())} regex types plus the address parser)" in _root_readme()


def test_readme_pii_types_table_matches_the_code():
    documented = {}
    for cells in _table(_root_readme(), "| Public type | Internal names |"):
        documented[cells[0].strip("`")] = set(re.findall(r"`([^`]+)`", cells[1]))
    assert documented == {p: set(INTERNAL_TYPES[p]) for p in PUBLIC_TYPES}
    assert f"the {len(PUBLIC_TYPES)} public types" in _root_readme()


def test_evaluator_label_table_matches_eval_metrics():
    path = REPO_ROOT / "eval_metrics.py"
    if not path.exists():
        pytest.skip("eval_metrics.py is not available here")
    spec = importlib.util.spec_from_file_location("_eval_metrics_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look their module up by name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    documented = {}
    for cells in _table(_root_readme(), "| Label(s) in key file | Internal type |"):
        typ = cells[1].strip("`")
        for label in re.findall(r"`([^`]+)`", cells[0]):
            documented[label] = typ
    assert documented == module.KEY_TYPE_MAP


def test_readme_module_counts_are_real():
    n = len(list((REPO_ROOT / "tests").glob("test_*.py")))
    if not n:
        pytest.skip("the repository test tree is not available here")
    text = _root_readme()
    assert f"`tests/` holds {n} pytest modules" in text
    assert f"# {n} pytest modules" in text
    assert len(re.findall(r"\d+ pytest modules", text)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Quasi-identifier scorer
# ─────────────────────────────────────────────────────────────────────────────

def _combo_levels():
    return {c["name"]: c["risk_level"] for c in quasi_identifier.QUASI_ID_COMBINATIONS}


def _quasi_table(text, header):
    return {cells[0]: cells[1].split(":")[0].strip().lower() for cells in _table(text, header)}


def test_quasi_identifier_tables_match_the_code():
    assert _quasi_table(_root_readme(), "| Combination | Risk |") == _combo_levels()
    assert _quasi_table(_lib_readme(), "| Combination | Risk | Basis |") == _combo_levels()


def _entities(*types):
    return [DetectedEntity(f"v{i}", i * 10, i * 10 + 2, t, 0.95, "test") for i, t in enumerate(types)]


def test_zip_alone_triggers_nothing():
    assert quasi_identifier.score(_entities("zip_us")) == []
    assert quasi_identifier.score(_entities("zip_us", "email")) == []
    assert quasi_identifier.score([]) == []


@pytest.mark.parametrize("combo", quasi_identifier.QUASI_ID_COMBINATIONS, ids=lambda c: c["name"])
def test_each_combination_triggers_at_its_documented_level(combo):
    fields = list(combo["fields"])
    full = {m.combination_name: m for m in quasi_identifier.score(_entities(*fields))}
    match = full[combo["name"]]
    assert match.risk_level == combo["risk_level"] == _combo_levels()[combo["name"]]
    assert match.all_fields_matched and set(match.matched_fields) == set(fields)
    assert match.risk_level in {"high", "medium"}
    # exactly required_count fields present: triggered, not all fields matched
    partial = {m.combination_name: m for m in quasi_identifier.score(_entities(*fields[:combo["required_count"]]))}
    assert combo["name"] in partial
    assert partial[combo["name"]].all_fields_matched == (combo["required_count"] == len(fields))
    # one short of required_count: this combination is silent
    below = {m.combination_name for m in quasi_identifier.score(_entities(*fields[:combo["required_count"] - 1]))}
    assert combo["name"] not in below


def test_sweeney_triple_is_high_and_name_employer_city_is_medium():
    high = quasi_identifier.score(_entities("zip_us", "dob", "gender_indicator"))
    assert [m.risk_level for m in high if m.combination_name == "ZIP + DOB + Gender"] == ["high"]
    medium = quasi_identifier.score(_entities("PERSON", "ORG", "GPE"))
    assert [m.risk_level for m in medium if m.combination_name == "Name + Employer + City"] == ["medium"]


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(STAGE_TABLE))
def test_preset_stage_table_is_as_documented(name):
    cfg = preset(name)
    assert {s: cfg.stage(s).enabled for s in STAGE_TABLE[name]} == STAGE_TABLE[name]
    assert cfg.stage("pattern_scan").enabled
    assert ("pii_tagger" in [s.name for s in cfg.plugin_stages()]) == STAGE_TABLE[name]["pii_tagger"]


def test_classic_preset_declares_no_detector_plugin():
    assert [s.name for s in preset("classic").plugin_stages()] == []


@heavy
@pytest.mark.parametrize("name", sorted(STAGE_TABLE))
def test_preset_runs_exactly_the_documented_stages(name, monkeypatch):
    cfg = preset(name)
    guard_calls, trace_calls, plugin_names = [], [], []
    real_guard, real_trace, real_get = pipeline.context_guard.guard, pipeline.entity_trace.trace, pipeline.plugins.get_detector

    def counting_guard(*args, **kwargs):
        guard_calls.append(kwargs.get("enabled", True))
        return real_guard(*args, **kwargs)

    def counting_trace(*args, **kwargs):
        trace_calls.append(1)
        return real_trace(*args, **kwargs)

    def counting_get(stage):
        plugin_names.append(stage.name)
        return real_get(stage)

    monkeypatch.setattr(pipeline.context_guard, "guard", counting_guard)
    monkeypatch.setattr(pipeline.entity_trace, "trace", counting_trace)
    monkeypatch.setattr(pipeline.plugins, "get_detector", counting_get)
    for text in MESSAGES:
        trace = []
        confirmed, _ = pipeline.run_cascade(text, config=cfg, trace=trace)
        assert [row["stage"] for row in trace][0] == "pattern_scan"
        assert all(0 <= e.start < e.end <= len(text) for e in confirmed)
    documented = STAGE_TABLE[name]
    assert bool(trace_calls) == documented["entity_trace"], name
    assert (True in guard_calls) == documented["context_guard"], name
    assert ("pii_tagger" in plugin_names) == documented["pii_tagger"], name
    if name == "classic":
        assert plugin_names == []


@heavy
def test_every_preset_masks_the_structured_values_in_the_fixture():
    for name in STAGE_TABLE:
        ss.config(detection=preset(name))
        session = Session()
        for text in MESSAGES:
            out = session.mask(text)
            for value in ("priya.raman@example.com", "512-44-7810", "4111 1111 1111 1111", "192.168.1.100",
                          "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789", "GB29 NWBK 6016 1331 9268 19"):
                assert value not in out or value not in text, (name, value)


# ─────────────────────────────────────────────────────────────────────────────
# Address modes and service queries (model-free)
# ─────────────────────────────────────────────────────────────────────────────

def test_auto_shifts_the_house_number_in_a_service_query():
    out = _mask("auto", "Find parking near 42 Baker Street, London.")
    assert re.fullmatch(r"Find parking near (41|43) Baker Street, London\.", out), out


def test_auto_replaces_the_whole_address_outside_a_service_query():
    out = _mask("auto", "I live at 789 Crescent Row, Tempe, AZ 85281 and need a plumber.")
    assert re.fullmatch(r"I live at \d+ [A-Z][A-Za-z']+ Row, [A-Z][A-Za-z .']+, [A-Z]{2} \d{5} and need a plumber\.", out), out
    for real in ("789", "Crescent", "Tempe", "85281"):
        assert real not in out


def test_auto_coarsens_a_sensitive_service_query_and_keeps_the_city():
    out = _mask("auto", "Is there a rehab clinic near 316 Citrus Blvd, Orlando, FL 32801?")
    assert out == "Is there a rehab clinic near my area, Orlando, FL?"


def test_shift_only_moves_the_house_number_by_one():
    out = _mask("shift", "I live at 789 Crescent Row, Tempe, AZ 85281.")
    assert re.fullmatch(r"I live at (788|790) Crescent Row, Tempe, AZ 85281\.", out), out


def test_replace_replaces_even_inside_a_service_query():
    out = _mask("replace", "Find parking near 42 Baker Street, London.")
    assert "42 Baker Street" not in out and "Baker" not in out
    assert re.fullmatch(r"Find parking near \d+ [A-Z][A-Za-z']+ Street, [A-Z][A-Za-z .']+\.", out), out


def test_po_boxes_are_replaced_in_auto_and_shifted_in_shift():
    out = _mask("auto", "Send it to PO Box 1234, Tempe, AZ 85281.")
    assert re.fullmatch(r"Send it to PO Box \d+, [A-Z][A-Za-z .']+, [A-Z]{2} \d{5}\.", out), out
    assert "1234" not in out and "Tempe" not in out
    out = _mask("shift", "Find a locksmith near PO Box 1234, Tempe, AZ 85281.")
    assert re.fullmatch(r"Find a locksmith near PO Box (1233|1235), Tempe, AZ 85281\.", out), out


def test_highway_addresses_parse_and_shift():
    text = "Find a gas station near 1500 Highway 50, Carson City, NV."
    parsed = address_parser.find_addresses(text)
    assert [(p.house_number, p.street_name, p.city, p.state) for p in parsed] == [("1500", "Highway 50", "Carson City", "NV")]
    for mode in ("auto", "shift"):
        out = _mask(mode, text)
        assert re.fullmatch(r"Find a gas station near (1499|1501) Highway 50, Carson City, NV\.", out), (mode, out)


def test_counts_that_look_like_house_numbers_are_rejected():
    text = "I did a 3 point turn and gave the instructor 5 stars."
    assert address_parser.find_addresses(text) == []
    assert _mask("auto", text) == text
    assert _mask("shift", text) == text


def test_house_number_one_never_shifts_to_zero():
    parsed = address_parser.find_addresses("1 Elm Street, Tempe, AZ")[0]
    assert {shift_house_number(parsed, rng=random.Random(seed)) for seed in range(40)} == {"2 Elm Street, Tempe, AZ"}
    assert _mask("shift", "Find a cafe near 1 Elm Street, Tempe, AZ.") == "Find a cafe near 2 Elm Street, Tempe, AZ."


def test_service_query_classification_and_effective_modes():
    assert service_query.classify("Find parking near 42 Baker Street, London.") == "service"
    assert service_query.classify("I live at 789 Crescent Row, Tempe, AZ 85281 and need a plumber.") == "none"
    assert service_query.classify("Is there a rehab clinic near 316 Citrus Blvd, Orlando, FL 32801?") == "service_coarse"
    assert service_query.resolve("Find parking near 42 Baker Street, London.", "auto") == (True, "shift")
    assert service_query.resolve("I live at 789 Crescent Row.", "auto") == (False, "replace")
    assert service_query.resolve("Is there a rehab clinic near 316 Citrus Blvd, Orlando?", "auto") == (True, "coarse")
    assert service_query.resolve("Find parking near 42 Baker Street, London.", "auto", enabled=False) == (False, "replace")
    for mode in ("shift", "replace"):          # explicit modes pass through untouched
        assert service_query.resolve_address_mode(mode, True, True) == mode
        assert service_query.resolve_address_mode(mode, False) == mode


def _code_sensitive_keywords():
    pattern = service_query._SENSITIVE_OVERRIDES[0].pattern
    return [k.replace(".", "-") for k in pattern.strip("()").split("|")]


def test_every_sensitive_keyword_in_the_code_coarsens_a_service_query():
    keywords = _code_sensitive_keywords()
    assert len(keywords) >= 20
    for keyword in keywords:
        text = f"Is there a {keyword} clinic near 316 Citrus Blvd, Orlando, FL 32801?"
        assert service_query.is_sensitive_topic(text), keyword
        assert service_query.classify(text) == "service_coarse", keyword
        assert service_query.is_sensitive_topic(text.upper()), keyword


def test_the_documented_sensitive_examples_are_in_the_code_and_legal_topics_are_not():
    documented = ["HIV", "abortion", "rehab", "therapy", "domestic violence", "homeless", "immigration", "shelter"]
    code = set(_code_sensitive_keywords())
    for keyword in documented:
        assert keyword.lower() in code, keyword
        assert service_query.is_sensitive_topic(f"a {keyword} centre near me"), keyword
    for keyword in ("legal", "lawyer", "attorney"):     # the READMEs no longer claim a legal override
        assert not service_query.is_sensitive_topic(f"Is there a {keyword} near 42 Baker Street, London?"), keyword
        assert keyword not in code
    for readme in (_root_readme(), _lib_readme()):
        for sentence in re.findall(r"[^.\n]*(?:[Ss]ensitive[- ]topic|Sensitive topics)[^.\n]*", readme):
            assert "legal" not in sentence, sentence


def test_city_and_state_survive_in_a_service_query_but_not_elsewhere(monkeypatch):
    def cascade(text, skip_values=None, skip_location_entities=False, **_kw):
        found = pattern_scan.scan(text, skip_values=skip_values)
        if not skip_location_entities:
            for m in re.finditer(r"\bTempe\b", text):
                found.append(DetectedEntity(m.group(), m.start(), m.end(), "GPE", 0.95, "stub"))
        return pipeline.deduplicate(found), []
    monkeypatch.setattr(pipeline, "run_cascade", cascade)
    ss.config(address_mode="auto")
    assert Session().mask("Find a cafe in Tempe.") == "Find a cafe in Tempe."
    out = Session().mask("I moved to Tempe last year.")
    assert "Tempe" not in out and out != "I moved to Tempe last year."


def test_no_geocoder_anywhere_in_address_handling():
    for module in (service_query, address_parser):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for name in ("requests", "urllib", "httpx", "socket", "geopy", "nominatim.openstreetmap.org"):
            assert name not in vars(module), (module.__name__, name)
            assert name not in source.replace("Nominatim existence check", ""), (module.__name__, name)
    assert not hasattr(service_query, "verify_addresses")
    with pytest.warns(DeprecationWarning):
        verified = service_query.fuzz_addresses("Find parking near 42 Baker Street, London.", verify=True)
    assert verified[1] == {"42 Baker Street, London": verified[0][len("Find parking near "):-1]}
    assert re.fullmatch(r"Find parking near (41|43) Baker Street, London\.", verified[0])
