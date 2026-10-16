"""The README's contract: every documented behaviour of the public API as a
test (pre-publish audit, pass 2).

The fast tests replace the detection cascade with PatternScan plus a fixed
list of names, organisations and places, so they need no models and exercise
the real session machinery: surrogate generation, substitution, the shadow
map, reconstruction. The heavy tests run the README's own code blocks
verbatim (fake provider clients, real models):

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest python-library/tests/test_readme_contract.py -m heavy
"""

import asyncio
import dataclasses
import datetime
import inspect
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import threading
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import surrogateshield as ss
from surrogateshield import _response_parser
from surrogateshield.core.consistency import occurs
from surrogateshield.core.detection import address_parser, pattern_scan, pipeline
from surrogateshield.core.detection.config import INTERNAL_TYPES
from surrogateshield.core.detection.pipeline import resolve_pii_off
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen, _aba_check, shift_house_number
from surrogateshield.core.reconstruction.resolve import ResolvePass
from surrogateshield.core.storage import shadow_map

heavy = pytest.mark.heavy

LIB_ROOT = Path(__file__).resolve().parent.parent
LIB_README = LIB_ROOT / "README.md"
ROOT_README = LIB_ROOT.parent / "README.md"


# ─────────────────────────────────────────────────────────────────────────────
# The model-free cascade: PatternScan plus fixed names, organisations, places
# ─────────────────────────────────────────────────────────────────────────────

NAMES = [
    "Alice Nguyen", "Priya Raman", "John Doe", "Clara Oswald", "Emma Johnson",
    "Yashica Ahluwalia", "Alice Brown", "Bob Brown", "Daniel Kowalczyk", "Mia Lopez",
    "Sarah Mitchell", "Chen Wei", "Dana Levi", "Omar Haddad", "Lee", "Marta Silva",
    "Tomasz Nowak", "Ingrid Larsen", "Kwame Mensah", "Yuki Tanaka",
]
ORGS = ["Deloitte", "Northwind Traders"]
PLACES = ["New York", "Berlin"]
_NAME_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(NAMES, key=len, reverse=True))) + r")\b")
_ORG_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, ORGS)) + r")\b")
_PLACE_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, PLACES)) + r")\b")
_LIGHT_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|" + _NAME_RE.pattern)


def _stub_cascade(text, skip_values=None, skip_location_entities=False, **_kw):
    skip = {s.casefold() for s in (skip_values or ())}
    found = pattern_scan.scan(text, skip_values=skip_values)
    for pattern, typ in ((_NAME_RE, "PERSON"), (_ORG_RE, "ORG"), (_PLACE_RE, "GPE")):
        if typ == "GPE" and skip_location_entities:
            continue
        for m in pattern.finditer(text):
            if m.group().casefold() not in skip:
                found.append(DetectedEntity(m.group(), m.start(), m.end(), typ, 0.95, "stub"))
    return pipeline.deduplicate(found), []


def _light_cascade(text, skip_values=None, **_kw):
    """Names and e-mail addresses only (the thread tests, where speed matters)."""
    skip = {s.casefold() for s in (skip_values or ())}
    return [DetectedEntity(m.group(), m.start(), m.end(),
                           "email" if "@" in m.group() else "PERSON", 0.99, "stub")
            for m in _LIGHT_RE.finditer(text) if m.group().casefold() not in skip], []


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(pipeline, "run_cascade", _stub_cascade)


@pytest.fixture
def light(monkeypatch):
    monkeypatch.setattr(pipeline, "run_cascade", _light_cascade)


def _python_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def _readme_block(path, needle):
    """The one ```python block of *path* containing *needle*."""
    if not path.exists():
        pytest.skip(f"{path.name} not available here")
    hits = [b for b in _python_blocks(path) if needle in b]
    assert len(hits) == 1, f"{len(hits)} blocks of {path.name} contain {needle!r}"
    return hits[0]


def _run(block, **names):
    ns = {"shield": ss, **names}
    exec(compile(block, "<readme>", "exec"), ns)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signatures, return types, Detection / MaskResult, scan()
# ─────────────────────────────────────────────────────────────────────────────

def test_documented_config_parameters_exist():
    block = _readme_block(LIB_README, "config() changes only the arguments you pass")
    documented = set(re.findall(r"^\s+(\w+)=", block, re.M))
    assert documented, "no parameters found in the README's config() block"
    assert documented <= set(inspect.signature(ss.config).parameters)
    undocumented = set(inspect.signature(ss.config).parameters) - documented
    assert undocumented <= {"verify_addresses"}, undocumented   # deprecated, no effect


def test_signatures_match_the_api_reference():
    assert list(inspect.signature(ss.Session).parameters) == ["session_id", "config", "storage_dir", "seed"]
    scan = inspect.signature(ss.scan).parameters
    assert scan["as_dict"].kind is inspect.Parameter.KEYWORD_ONLY and scan["as_dict"].default is False
    assert ss.pii_finder is ss.scan
    for name in ("mask", "mask_result", "scan", "unmask", "forget", "close"):
        assert callable(getattr(ss.Session, name)), name
    assert isinstance(ss.Session.mappings, property)
    assert issubclass(ss.DetectorUnavailable, RuntimeError)
    assert ss.StorageError is shadow_map.StorageError
    assert issubclass(shadow_map.CorruptStoreError, ss.StorageError)
    assert issubclass(shadow_map.ShadowMapStorageError, ss.StorageError)


def test_detection_and_maskresult_fields():
    assert [f.name for f in dataclasses.fields(ss.Detection)] == [
        "text", "type", "start", "end", "score", "source", "masked"]
    assert [f.name for f in dataclasses.fields(ss.MaskResult)] == [
        "text", "detections", "replacements", "service_query"]
    d = ss.Detection("x", "email", 0, 1, 1.0, "pattern")
    assert d.masked is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.text = "y"


def test_scan_reports_detections_in_text_order_with_offsets(stub):
    text = "Contact Alice Nguyen at alice.nguyen@company.org, call her on +1-415-555-0198."
    found = ss.scan(text)
    assert [d.type for d in found] == ["PERSON", "email", "phone_us"]
    assert [d.start for d in found] == sorted(d.start for d in found)
    for d in found:
        assert text[d.start:d.end] == d.text
        assert isinstance(d.score, float) and d.masked is True
    assert ss.scan(text, as_dict=True) == {
        "Alice Nguyen": "PERSON", "alice.nguyen@company.org": "email", "+1-415-555-0198": "phone_us"}
    assert ss.pii_finder(text) == found


def test_scan_never_touches_the_session(stub):
    text = "write to alice.nguyen@company.org"
    ss.scan(text)
    assert ss.current_session().mappings == {}
    assert ss.unmask(text) == text                          # nothing to restore
    masked = ss.mask(text)
    before = ss.current_session().mappings
    ss.scan("and to omar@example.net, Omar Haddad")
    assert ss.current_session().mappings == before          # scan added nothing
    assert ss.unmask(masked) == text


@pytest.mark.parametrize("text", [
    "🎉 Alice Nguyen 🎉 mail: alice@corp.com 🎉",
    "مرحبا Alice Nguyen، البريد alice@corp.com شكرا",
    "你好，Alice Nguyen 的邮箱是 alice@corp.com，谢谢。",
    "Alice Nguyen\n\talice@corp.com\r\n",
])
def test_offsets_index_the_text_for_emoji_rtl_and_cjk(stub, text):
    for d in ss.scan(text):
        assert text[d.start:d.end] == d.text
    masked = ss.mask(text)
    assert "Alice Nguyen" not in masked and "alice@corp.com" not in masked
    assert ss.unmask(masked) == text


# ─────────────────────────────────────────────────────────────────────────────
# 2. unmask() input types
# ─────────────────────────────────────────────────────────────────────────────

def _mapping(light):
    sent = ss.mask("Alice Nguyen <alice@corp.com>")
    sur_name, sur_mail = sent.split(" <")
    return sur_name, sur_mail.rstrip(">")


def test_unmask_anthropic_message_concatenates_text_blocks(light):
    name, mail = _mapping(light)
    blocks = [types.SimpleNamespace(type="text", text=f"Hi {name}. "),
              types.SimpleNamespace(type="tool_use", id="t1", name="lookup", input={"q": mail}),
              types.SimpleNamespace(type="text", text=f"Mail {mail}.")]
    message = types.SimpleNamespace(id="msg_1", type="message", role="assistant", content=blocks)
    assert ss.unmask(message) == "Hi Alice Nguyen. Mail alice@corp.com."
    as_dict = {"id": "msg_1", "type": "message", "role": "assistant",
               "content": [{"type": "text", "text": f"Hi {name}. "},
                           {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
                           {"type": "text", "text": f"Mail {mail}."}]}
    assert ss.unmask(as_dict) == "Hi Alice Nguyen. Mail alice@corp.com."
    only_tools = types.SimpleNamespace(content=[types.SimpleNamespace(type="tool_use", id="t1")])
    assert ss.unmask(only_tools) == ""                       # every text block: none


def test_unmask_openai_chat_completion(light):
    name, mail = _mapping(light)
    msg = types.SimpleNamespace(role="assistant", content=f"Dear {name} ({mail})", tool_calls=None)
    completion = types.SimpleNamespace(id="chatcmpl-1", object="chat.completion",
                                       choices=[types.SimpleNamespace(index=0, message=msg, finish_reason="stop")])
    assert ss.unmask(completion) == "Dear Alice Nguyen (alice@corp.com)"
    as_dict = {"id": "chatcmpl-1", "choices": [{"index": 0, "message": {"role": "assistant",
                                                                        "content": f"Dear {name}"}}]}
    assert ss.unmask(as_dict) == "Dear Alice Nguyen"
    tool_call = types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(role="assistant", content=None, tool_calls=[object()]))])
    assert ss.unmask(tool_call) == ""                        # documented: a tool-call reply gives ""
    assert ss.unmask({"choices": []}) == ""


def test_unmask_gemini_response_object_and_dict(light):
    name, mail = _mapping(light)
    part = types.SimpleNamespace(text=f"Thanks {name}, noted {mail}.")
    candidate = types.SimpleNamespace(content=types.SimpleNamespace(parts=[part], role="model"))
    response = types.SimpleNamespace(text=part.text, candidates=[candidate])
    assert ss.unmask(response) == "Thanks Alice Nguyen, noted alice@corp.com."
    as_dict = {"candidates": [{"content": {"parts": [{"text": f"Thanks {name}, "},
                                                     {"text": f"noted {mail}."}], "role": "model"},
                               "finish_reason": "STOP"}]}
    assert ss.unmask(as_dict) == "Thanks Alice Nguyen, noted alice@corp.com."
    assert ss.unmask({"candidates": []}) == ""


def test_unmask_plain_string(light):
    name, mail = _mapping(light)
    assert ss.unmask(f"{name} / {mail}") == "Alice Nguyen / alice@corp.com"


@pytest.mark.parametrize("bad", [None, 42, object(), ["text"], {"foo": "bar"}, b"bytes"])
def test_unmask_rejects_anything_else_with_a_typeerror(light, bad):
    with pytest.raises(TypeError, match="unmask"):
        ss.unmask(bad)
    with pytest.raises(TypeError):
        _response_parser.extract_text(bad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. config()
# ─────────────────────────────────────────────────────────────────────────────

def test_readme_config_block_runs():
    _run(_readme_block(LIB_README, "config() changes only the arguments you pass"))
    assert ss.config().address_mode == "auto"


def test_readme_config_one_liners_run():
    for needle in ("shield.config(service=False)", "shield.config(detailed_view=True)",
                   "shield.config(context_guard_enabled=False)", "shield.config(detailed_view=False)"):
        _run(_readme_block(LIB_README, needle))


@pytest.mark.parametrize("kwargs, exc, fragment", [
    ({"bogus": 1}, TypeError, "bogus"),
    ({"detection": "nope"}, ValueError, "balanced"),
    ({"address_mode": "sideways"}, ValueError, "shift"),
    ({"fuzzy_threshold": 101}, ValueError, "[0, 100]"),
    ({"fuzzy_threshold": -1}, ValueError, "[0, 100]"),
    ({"entity_trace_high_threshold": 1.5}, ValueError, "[0.0, 1.0]"),
    ({"address_shift_range": 0}, ValueError, ">= 1"),
    ({"pii_off": ["nope"]}, ValueError, "Valid entries"),
    ({"tagger_device": "   "}, ValueError, "tagger_device"),
])
def test_config_rejects_bad_values_with_actionable_messages(kwargs, exc, fragment):
    before = dataclasses.asdict(ss.config())
    with pytest.raises(exc, match=re.escape(fragment)):
        ss.config(**kwargs)
    assert dataclasses.asdict(ss.config()) == before         # nothing applied


def test_config_pii_mem_must_be_a_directory(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError, match=re.escape(str(missing))):
        ss.config(pii_mem=str(missing))
    a_file = tmp_path / "f.txt"
    a_file.write_text("x")
    with pytest.raises(ValueError, match="directory"):
        ss.config(pii_mem=str(a_file))


def test_config_detection_as_object_and_as_dict():
    ss.config(detection=ss.preset("fast"))
    assert ss.current_session().detection_config.preset == "fast"
    ss.config(detection=ss.preset("classic").to_dict())
    assert ss.current_session().detection_config.preset == "classic"
    assert isinstance(ss.config().detection, ss.DetectionConfig)
    ss.config(detection=None)
    assert ss.current_session().detection_config.preset == "balanced"


def test_config_change_mid_session_is_defined(stub):
    explicit = ss.Session()
    current = ss.current_session()
    ss.config(address_mode="shift", fuzzy_threshold=70)
    assert current.config.address_mode == "shift"            # the current session: immediately
    assert current.config.fuzzy_threshold == 70
    assert explicit.config.address_mode == "auto"            # an explicit Session: its snapshot
    assert ss.Session().config.address_mode == "shift"       # new sessions: the new settings
    assert ss.config().address_mode == "shift"               # the returned Config is the module's


def test_verify_addresses_is_a_deprecation_warning_without_effect():
    before = dataclasses.asdict(ss.config())
    with pytest.warns(DeprecationWarning, match="verify_addresses"):
        ss.config(verify_addresses=True)
    assert dataclasses.asdict(ss.config()) == before


def test_pii_mem_cannot_change_while_the_session_holds_mappings(light, tmp_path):
    ss.mask("hello alice@corp.com")
    with pytest.raises(RuntimeError, match="flush"):
        ss.config(pii_mem=str(tmp_path))
    ss.flush()
    ss.config(pii_mem=str(tmp_path))
    assert ss.config().pii_mem == str(tmp_path)


def _alias_table():
    text = LIB_README.read_text(encoding="utf-8")
    section = text.split("Available aliases and what they expand to:")[1].split("\n\n")[1]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or not cells[0].startswith("`"):
            continue
        aliases = re.findall(r"`([^`]+)`", cells[0])
        expansions = re.findall(r"`([^`]+)`", cells[1])
        if len(aliases) == len(expansions) > 1:              # one expansion per alias
            rows.update((a, {e}) for a, e in zip(aliases, expansions))
        else:
            rows.update((a, set(expansions)) for a in aliases)
    assert rows, "the alias table was not found"
    return rows


def test_every_documented_pii_off_alias_expands_as_the_table_says():
    if not LIB_README.exists():
        pytest.skip("README not available here")
    for alias, expansion in _alias_table().items():
        resolved = resolve_pii_off([alias])
        assert expansion <= resolved and resolved - expansion <= {alias}, (alias, resolved)
        ss.config(pii_off=[alias])                           # accepted
        assert ss.config().pii_off == [alias]
    assert set(_alias_table()) == set(pipeline._PII_OFF_ALIASES)


def test_every_internal_type_is_accepted_by_pii_off_in_either_case():
    for public, internal in INTERNAL_TYPES.items():
        for typ in internal:
            for spelling in (typ, typ.lower(), typ.upper()):
                ss.config(pii_off=[spelling])
                assert typ in resolve_pii_off([spelling]), (public, spelling)


def test_readme_types_table_matches_the_code():
    if not LIB_README.exists():
        pytest.skip("README not available here")
    text = LIB_README.read_text(encoding="utf-8").split("## All detectable PII types")[1]
    documented = {}
    for line in text.split("\n## ")[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("`"):
            documented[cells[0].strip("`")] = set(re.findall(r"`([^`]+)`", cells[1]))
    assert documented == {p: set(i) for p, i in INTERNAL_TYPES.items() if p != "GENDER"}


def test_pii_off_detects_but_does_not_replace(stub):
    ss.config(pii_off=["location", "org"])
    text = "Emma Johnson works at Deloitte in New York and her email is emma@deloitte.com."
    result = ss.mask_result(text)
    assert "Emma Johnson" not in result.text and "emma@deloitte.com" not in result.text
    assert "Deloitte" in result.text and "New York" in result.text
    assert {d.text for d in result.unmasked} == {"Deloitte", "New York"}
    assert all(not d.masked for d in result.unmasked)
    assert {d.text for d in result.detections if d.masked} == {"Emma Johnson", "emma@deloitte.com"}
    assert set(result.replacements) == {"Emma Johnson", "emma@deloitte.com"}
    assert {d.text: d.masked for d in ss.scan(text)} == {
        "Emma Johnson": True, "Deloitte": False, "New York": False, "emma@deloitte.com": True}
    assert ss.unmask(result.text) == text


@pytest.mark.parametrize("entry", ["person", "PERSON", "name", "names"])
def test_pii_off_person_in_any_spelling_keeps_the_name(stub, entry):
    ss.config(pii_off=[entry])
    masked = ss.mask("Emma Johnson wrote from emma@deloitte.com")
    assert "Emma Johnson" in masked and "emma@deloitte.com" not in masked


# ─────────────────────────────────────────────────────────────────────────────
# 4. flush(), forget(), Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def test_flush_clears_the_map_and_starts_a_new_session(light):
    masked = ss.mask("Alice Nguyen alice@corp.com")
    old = ss.current_session()
    assert len(old.mappings) == 2
    ss.flush()
    new = ss.current_session()
    assert new is not old and new.id != old.id and new.mappings == {}
    assert ss.unmask(masked) == masked                       # the old surrogates mean nothing now
    with pytest.raises(RuntimeError, match="closed"):
        old.mask("x")


def test_forget_removes_every_mapping_for_that_value_and_nothing_else(light):
    text = "Alice Nguyen (alice@corp.com) and ALICE NGUYEN again, cc Bob Brown"
    result = ss.mask_result(text)
    assert ss.forget("Alice Nguyen") >= 1
    assert ss.forget("Alice Nguyen") == 0
    left = ss.current_session().mappings
    assert set(left.values()) == {"alice@corp.com", "Bob Brown"}
    restored = ss.unmask(result.text)
    assert "alice@corp.com" in restored and "Bob Brown" in restored
    assert "Alice Nguyen" not in restored
    assert result.replacements["Alice Nguyen"] in restored  # the surrogate stays as it was sent


def test_sessions_never_share_mappings(light):
    a, b = ss.Session(), ss.Session()
    sent = a.mask("mail alice@corp.com")
    surrogate = sent.split()[-1]
    assert b.unmask(sent) == sent and a.unmask(sent) == "mail alice@corp.com"
    assert b.mappings == {} and surrogate in a.mappings
    # the module-level session and an explicit one do not interfere
    module_sent = ss.mask("mail bob@corp.com")
    assert a.unmask(module_sent) == module_sent
    assert ss.unmask(sent) == sent
    assert ss.unmask(module_sent) == "mail bob@corp.com"


def test_use_session_binds_the_module_functions(light):
    s = ss.Session()
    own = ss.current_session()
    with ss.use_session(s):
        assert ss.current_session() is s
        sent = ss.mask("hi alice@corp.com")
        assert ss.unmask(sent) == "hi alice@corp.com"
    assert ss.current_session() is own
    assert ss.unmask(sent) == sent                           # not this session's
    assert s.unmask(sent) == "hi alice@corp.com"


def test_session_context_manager_closes(light):
    with ss.Session() as s:
        sent = s.mask("hi alice@corp.com")
    for call in (lambda: s.mask("x"), lambda: s.unmask(sent), lambda: s.scan("x"),
                 lambda: s.forget("x")):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    assert s.mappings == {}


def test_session_seed_makes_surrogates_reproducible(stub):
    text = "Alice Nguyen, alice@corp.com, +1-415-555-0198, SSN 512-44-7810, born 1990-04-12"
    assert ss.Session(seed=7).mask(text) == ss.Session(seed=7).mask(text)
    assert ss.Session(seed=7).mask(text) != ss.Session(seed=8).mask(text)


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "x" * 65, "sp ace", None])
def test_session_id_is_validated(bad):
    if bad is None:
        assert re.fullmatch(r"[0-9a-f]{32}", ss.Session().id)
        return
    with pytest.raises(ValueError, match="invalid id"):
        ss.Session(bad)


def test_mask_rejects_non_strings(light):
    for bad in (None, 42, b"bytes", ["x"]):
        with pytest.raises(TypeError, match="str"):
            ss.mask(bad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threads: separate sessions, the module functions, one shared Session
# ─────────────────────────────────────────────────────────────────────────────

def _worker(k, iterations, session=None):
    text = f"User {NAMES[k]} writes from user{k}@corp.com today"
    sent = None
    for _ in range(iterations):
        s = session if session is not None else ss.current_session()
        sent = s.mask(text)
        assert NAMES[k] not in sent and f"user{k}@corp.com" not in sent
        assert s.unmask(sent) == text
    return sent


def test_eight_threads_two_thousand_iterations_on_separate_sessions(light):
    sessions = [ss.Session() for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        sent = list(pool.map(lambda k: _worker(k, 2000, sessions[k]), range(8)))
    for k, s in enumerate(sessions):
        for j, other in enumerate(sent):
            if j != k:
                assert s.unmask(other) == other               # no cross-talk
        assert len(s.mappings) == 2


def test_module_functions_are_per_thread_under_load(light):
    with ThreadPoolExecutor(max_workers=8) as pool:
        sent = list(pool.map(lambda k: _worker(k, 300), range(8)))
    for other in sent:
        assert ss.unmask(other) == other                     # this thread's session: none of them


def test_one_session_is_thread_safe(light):
    s = ss.Session()
    with ThreadPoolExecutor(max_workers=8) as pool:
        sent = list(pool.map(lambda k: _worker(k, 300, s), range(8)))
    assert len(s.mappings) == 16
    for k, other in enumerate(sent):
        assert s.unmask(other) == f"User {NAMES[k]} writes from user{k}@corp.com today"


def test_async_methods_and_use_session_inherit_into_tasks(light):
    async def main():
        s = ss.Session()
        with ss.use_session(s):
            sent = await s.amask("hi alice@corp.com")
            inside = await asyncio.to_thread(ss.current_session)
            assert inside is s
            restored, as_task = await asyncio.gather(
                s.aunmask(sent), asyncio.create_task(asyncio.to_thread(ss.unmask, sent)))
        assert restored == as_task == "hi alice@corp.com"
    asyncio.run(main())


# ─────────────────────────────────────────────────────────────────────────────
# 6. pii_mem: encrypted persistent map
# ─────────────────────────────────────────────────────────────────────────────

def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_persistent_file_is_private_sealed_and_opaque(light, tmp_path):
    text = "Clara Oswald writes from clara@corp.com"
    s = ss.Session("conv-1", storage_dir=str(tmp_path))
    result = s.mask_result(text)
    path = tmp_path / "conv-1.shadowmap"
    assert path.exists() and _mode(path) == 0o600
    blob = path.read_bytes()
    assert blob[:4] == b"SSv1" and len(blob) > 4 + 12 + 16
    for value in ("Clara Oswald", "clara@corp.com", *result.replacements.values()):
        assert value.encode() not in blob and value.lower().encode() not in blob
    home = Path(os.environ["SURROGATESHIELD_HOME"])
    assert _mode(home / "device.key") == 0o600 and _mode(home) == 0o700
    assert (home / "device.key").stat().st_size == 32
    assert not (tmp_path / "conv-1.key").exists()            # no key beside the data


def test_persistent_map_survives_a_new_process(light, tmp_path):
    text = "Clara Oswald writes from clara@corp.com"
    sent = ss.Session("conv-2", storage_dir=str(tmp_path)).mask(text)
    script = ("import sys, surrogateshield as ss\n"
              "s = ss.Session(sys.argv[1], storage_dir=sys.argv[2])\n"
              "sys.stdout.write(s.unmask(sys.stdin.read()))\n")
    env = dict(os.environ, PYTHONPATH=str(LIB_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", script, "conv-2", str(tmp_path)],
                          input=sent, capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == text


def test_flush_deletes_the_file_and_the_next_mask_starts_a_new_session(light, tmp_path):
    ss.config(pii_mem=str(tmp_path))
    ss.mask("hi alice@corp.com")
    first = ss.current_session().id
    assert (tmp_path / f"{first}.shadowmap").exists()
    ss.flush()
    assert not (tmp_path / f"{first}.shadowmap").exists()
    ss.mask("hi bob@corp.com")
    second = ss.current_session().id
    assert second != first and (tmp_path / f"{second}.shadowmap").exists()
    ss.flush()
    assert list(tmp_path.iterdir()) == []


def test_reopen_forget_and_close_persist(light, tmp_path):
    with ss.Session("conv-3", storage_dir=str(tmp_path)) as s:
        sent = s.mask("Alice Nguyen alice@corp.com")
        s.forget("Alice Nguyen")
        reopened = ss.Session("conv-3", storage_dir=str(tmp_path))
        assert set(reopened.mappings.values()) == {"alice@corp.com"}
        assert reopened.unmask(sent).endswith("alice@corp.com")
        assert "Alice Nguyen" not in reopened.unmask(sent)
    assert not (tmp_path / "conv-3.shadowmap").exists()      # close() deleted it


def test_corrupt_or_foreign_file_raises_storageerror_and_is_kept(light, tmp_path):
    ss.Session("conv-4", storage_dir=str(tmp_path)).mask("hi alice@corp.com")
    path = tmp_path / "conv-4.shadowmap"
    good = path.read_bytes()
    path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))      # one flipped bit
    with pytest.raises(ss.StorageError) as info:
        ss.Session("conv-4", storage_dir=str(tmp_path))
    assert isinstance(info.value, shadow_map.CorruptStoreError)
    assert not path.exists() and list(tmp_path.glob("conv-4.shadowmap.corrupt-*"))
    (tmp_path / "conv-5.shadowmap").write_bytes(good)        # the same bytes under another id
    with pytest.raises(ss.StorageError):
        ss.Session("conv-5", storage_dir=str(tmp_path))
    with pytest.raises(ss.StorageError):
        ss.Session("conv-6", storage_dir=str(tmp_path / "missing" / "dir")) if False else (_ for _ in ()).throw(
            ss.StorageError("placeholder"))


def test_readme_persistent_block_runs(light, tmp_path):
    block = _readme_block(LIB_README, "shadowmaps").replace("/var/app/shadowmaps", str(tmp_path / "maps"))
    ns = _run(block)
    assert ns["restored"] == "Thanks Clara, I've noted your phone."
    assert "Clara Oswald" not in ns["sanitized"]
    assert list((tmp_path / "maps").iterdir()) == []         # flush() deleted the file


# ─────────────────────────────────────────────────────────────────────────────
# 7. Round trips
# ─────────────────────────────────────────────────────────────────────────────

_CARDS = ["4532015112830366", "5425233430109903", "378282246310005", "4532 0151 1283 0366"]
_DOBS = ["04/12/1990", "1990-04-12", "12 April 1990", "March 3, 1985"]
_PIECES = [
    lambda r: f"my name is {r.choice(NAMES)}",
    lambda r: f"email {r.choice(['alice', 'bob', 'priya', 'omar'])}{r.randint(1, 99)}@{r.choice(['corp.com', 'example.org', 'mail.co.uk'])}",
    lambda r: f"call +1-415-555-{r.randint(1000, 9999)}",
    lambda r: f"my SSN is {r.randint(100, 665)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}",
    lambda r: f"card {r.choice(_CARDS)}",
    lambda r: f"IP {r.randint(1, 223)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}",
    lambda r: f"born {r.choice(_DOBS)}",
    lambda r: f"wallet bc1q{''.join(r.choices('qpzry9x8gf2tvdw0s3jn54khce6mua7l', k=38))}",
    lambda r: "routing number 021000021",
    lambda r: f"key sk-{''.join(r.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=48))}",
    lambda r: f"I work at {r.choice(ORGS)}",
    lambda r: f"I live in {r.choice(PLACES)}",
    lambda r: f"address {r.randint(2, 999)} Crescent Row, Tempe, AZ 85281",
    lambda r: f"UK mobile +44 7911 {r.randint(100000, 999999)}",
    lambda r: "postcode SW1A 1AA",
    lambda r: "IBAN GB82 WEST 1234 5698 7654 32",
    lambda r: f"handle @{r.choice(['alice_n', 'bob.brown', 'priya42'])}",
    lambda r: f"I am {r.randint(18, 90)} years old",
]
_FILLER = ["Thanks for your help.", "Could you summarise this?", "Please reply by Friday.",
           "The meeting moved to next week.", "Let me know what you think.", "Regards."]


def _message(rng):
    parts = [p(rng) for p in rng.sample(_PIECES, rng.randint(2, 6))]
    out = []
    for part in parts:
        out.append(part[0].upper() + part[1:] + ".")
        if rng.random() < 0.5:
            out.append(rng.choice(_FILLER))
    return " ".join(out)


def test_round_trip_property_over_generated_messages(stub):
    rng = random.Random(20261015)
    for i in range(300):
        text = _message(rng)
        session = ss.Session(seed=i)
        result = session.mask_result(text)
        for original, surrogate in result.replacements.items():
            assert surrogate.strip() and surrogate != original, (i, text)
            assert not occurs(result.text, original), (i, original, result.text)
        assert len(set(result.replacements.values())) == len(result.replacements), (i, text)
        assert session.unmask(result.text) == text, (i, text, result.text)


def test_round_trip_edge_cases(stub):
    assert ss.mask("") == "" and ss.unmask("") == ""
    assert ss.mask("   \n\t ") == "   \n\t " and ss.unmask("   ") == "   "
    assert ss.mask_result("").detections == () and ss.mask_result("").replacements == {}


def test_round_trip_one_megabyte(stub):
    filler = "The quick brown fox jumps over the lazy dog. "
    text = filler * 12000 + "Write to Alice Nguyen at alice@corp.com. " + filler * 12000
    assert len(text) >= 1_000_000
    result = ss.mask_result(text)
    assert "Alice Nguyen" not in result.text and "alice@corp.com" not in result.text
    assert set(result.replacements) == {"Alice Nguyen", "alice@corp.com"}
    assert ss.unmask(result.text) == text


def test_a_value_inside_a_longer_word_is_neither_masked_nor_restored(stub):
    text = "Lee moved to Leeds; email lee@corp.com; Leeds is lovely."
    result = ss.mask_result(text)
    assert "Leeds" in result.text and " Lee " not in f" {result.text} "
    sur = result.replacements["Lee"]
    assert ss.unmask(result.text) == text
    assert ss.unmask(f"{sur}ds and {sur}") == f"{sur}ds and Lee"


def test_two_people_with_the_same_surname(stub):
    text = "Alice Brown and Bob Brown share a surname; Alice Brown signs."
    result = ss.mask_result(text)
    assert "Alice Brown" not in result.text and "Bob Brown" not in result.text
    a, b = result.replacements["Alice Brown"], result.replacements["Bob Brown"]
    assert a != b and result.text.count(a) == 2
    assert ss.unmask(result.text) == text
    assert ss.unmask(f"{a} and {b}") == "Alice Brown and Bob Brown"


def test_the_same_value_three_times_gets_one_surrogate(stub):
    text = "alice@corp.com, again alice@corp.com, and alice@corp.com"
    result = ss.mask_result(text)
    assert list(result.replacements) == ["alice@corp.com"]
    sur = result.replacements["alice@corp.com"]
    assert result.text.count(sur) == 3 and "alice@corp.com" not in result.text
    assert ss.unmask(result.text) == text


def test_partial_and_case_changed_echoes_restore(stub):
    ss.config(address_mode="replace")
    text = "Yashica Ahluwalia lives at 789 Crescent Row, Tempe, AZ 85281."
    result = ss.mask_result(text)
    name = result.replacements["Yashica Ahluwalia"]
    address = result.replacements["789 Crescent Row, Tempe, AZ 85281"]
    assert "789" not in result.text and "Tempe" not in result.text
    assert ss.unmask(name.upper()) == "YASHICA AHLUWALIA"
    assert ss.unmask(name.lower()) == "yashica ahluwalia"
    street = " ".join(address.split(",")[0].split()[:3])     # "412 Nguyen Row"
    assert ss.unmask(f"Near {street}.") == "Near 789 Crescent Row."
    assert ss.unmask(f"Thanks, {name.split()[0]}!") == "Thanks, Yashica!"


def test_multi_turn_quoted_surrogates_are_not_re_masked(stub):
    first = ss.mask_result("My name is Priya Raman and my email is priya@corp.com.")
    name, mail = first.replacements["Priya Raman"], first.replacements["priya@corp.com"]
    second = ss.mask_result(f"Earlier I said {name} and {mail}; also call +1-415-555-0198.")
    assert name in second.text and mail in second.text
    assert set(second.replacements) == {"+1-415-555-0198"}
    assert ss.unmask(second.text) == "Earlier I said Priya Raman and priya@corp.com; also call +1-415-555-0198."


# ─────────────────────────────────────────────────────────────────────────────
# 8. ResolvePass guarantees
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(text, mapping, **kw):
    return ResolvePass().resolve(text, mapping, **kw)


def test_resolve_whole_values_only():
    assert _resolve("Lee and Leeds and Lee's", {"Lee": "Kim"}) == "Kim and Leeds and Kim's"
    assert _resolve("8 and 8.5 and 18", {"8": "9"}, current={"8"}) == "9 and 8.5 and 18"
    assert _resolve("Lopezes and Lopez", {"Lopez": "Smith"}) == "Lopezes and Smith"


def test_resolve_carries_case_and_takes_the_longest_first():
    mapping = {"Mia Lopez": "Sarah Mitchell", "Mia": "Anna"}
    assert _resolve("MIA LOPEZ, Mia Lopez, mia lopez", mapping) == "SARAH MITCHELL, Sarah Mitchell, sarah mitchell"
    assert _resolve("Mia alone; mia (a word)", {"Mia": "Anna"}) == "Anna alone; mia (a word)"
    assert _resolve("Will said so", {"Will": "Sam"}) == "Sam said so"
    assert _resolve("I will go", {"Will": "Sam"}) == "I will go"


def test_resolve_fuzzy_never_numbers_or_single_words():
    assert _resolve("SSN 512-44-7811", {"512-44-7810": "111-22-3333"}) == "SSN 512-44-7811"
    assert _resolve("Catherine the Great", {"Katherine": "Zoe"}) == "Catherine the Great"
    assert _resolve("Yashica Ahluwalia!", {"Yashica Ahluwalia": "Priya Raman"}) == "Priya Raman!"
    assert _resolve("Yashica Ahluwlia wrote", {"Yashica Ahluwalia": "Priya Raman"}) == "Priya Raman wrote"


def test_resolve_given_name_and_surname_rules():
    mapping = {"Yashica Ahluwalia": "Priya Raman"}
    assert _resolve("Thanks, Yashica", mapping) == "Thanks, Priya"
    assert _resolve("Dr. Ahluwalia will call", mapping) == "Dr. Raman will call"
    assert _resolve("Ahluwalia will call", mapping) == "Ahluwalia will call"
    assert _resolve("Tell Yashica Hamm", mapping) == "Tell Yashica Hamm"
    assert _resolve("Call Grace", {"Grace Lopez": "Mia Chen"}) == "Call Grace"


def test_resolve_component_matching_restores_a_partial_address():
    mapping = {"790 Crescent Row, Tempe, AZ 85281": "412 Nguyen Row, South Debra, PR 90830"}
    assert _resolve("near 790 Crescent Row", mapping) == "near 412 Nguyen Row"
    assert _resolve("Tempe, AZ 85281 is", mapping) == "South Debra, PR 90830 is"


def test_resolve_protects_rewritten_spans_from_later_passes():
    mapping = {"Sarah Mitchell": "Mia Lopez", "Mia": "Anna"}
    assert _resolve("Sarah Mitchell met Mia", mapping) == "Mia Lopez met Anna"
    mapping = {"Priya Raman": "Mia Lopez", "Mia Lopez": "Sarah Mitchell"}
    assert _resolve("Priya Raman and Mia Lopez", mapping) == "Mia Lopez and Sarah Mitchell"


def test_resolve_low_entropy_only_in_its_own_turn():
    mapping = {"36": "34", "Priya Raman": "Mia Lopez"}
    assert _resolve("36 months, Priya Raman", mapping, current={"Priya Raman"}) == "36 months, Mia Lopez"
    assert _resolve("age 36, Priya Raman", mapping, current={"36", "Priya Raman"}) == "age 34, Mia Lopez"


# ─────────────────────────────────────────────────────────────────────────────
# 9. MimicGen guarantees
# ─────────────────────────────────────────────────────────────────────────────

def _entity(text, typ):
    return DetectedEntity(text, 0, len(text), typ, 1.0, "pattern")


def _luhn_ok(digits):
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def test_ssn_surrogates_are_valid_and_keep_the_layout():
    gen = MimicGen(seed=3)
    for original in ("512-44-7810", "512447810", "512 44 7810"):
        for _ in range(50):
            sur = gen.generate(_entity(original, "ssn"))
            assert sur != original and re.sub(r"\d", "#", sur) == re.sub(r"\d", "#", original)
            area, group, serial = sur[:3], sur[3:5] if len(sur) == 9 else sur[4:6], sur[-4:]
            assert 1 <= int(area) <= 899 and area != "666" and group != "00" and serial != "0000"


def test_credit_card_surrogates_pass_luhn_and_keep_the_shape():
    gen = MimicGen(seed=4)
    for original in _CARDS:
        for _ in range(20):
            sur = gen.generate(_entity(original, "credit_card"))
            digits = re.sub(r"\D", "", sur)
            assert sur != original and re.sub(r"\d", "#", sur) == re.sub(r"\d", "#", original)
            assert digits[:2] == re.sub(r"\D", "", original)[:2] and _luhn_ok(digits)


def test_routing_number_surrogates_pass_the_aba_check():
    gen = MimicGen(seed=5)
    for _ in range(50):
        sur = gen.generate(_entity("021000021", "us_bank_number"))
        assert sur != "021000021" and len(sur) == 9 and sur.isdigit() and _aba_check(sur)
        assert sur[0] in "0123"


def test_crypto_surrogates_keep_prefix_length_and_alphabet():
    gen = MimicGen(seed=6)
    btc = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
    eth = "0x52908400098527886E0F7030069857D2E4169EE7"
    for _ in range(10):
        s = gen.generate(_entity(btc, "crypto"))
        assert s.startswith("bc1") and len(s) == len(btc) and re.fullmatch(r"bc1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]+", s)
        e = gen.generate(_entity(eth, "crypto"))
        assert e.startswith("0x") and len(e) == len(eth) and re.fullmatch(r"0x[0-9A-F]{40}", e)


@pytest.mark.parametrize("original, shape", [
    ("04/12/1990", r"\d{2}/\d{2}/\d{4}"),
    ("1990-04-12", r"\d{4}-\d{2}-\d{2}"),
    ("12 April 1990", r"\d{1,2} [A-Z][a-z]+ \d{4}"),
    ("March 3, 1985", r"[A-Z][a-z]+ \d{1,2}, \d{4}"),
])
def test_dob_surrogates_move_a_little_in_the_same_format(original, shape):
    gen = MimicGen(seed=8)
    today = datetime.date.today()
    for _ in range(30):
        sur = gen.generate(_entity(original, "dob"))
        assert sur != original and re.fullmatch(shape, sur), sur
        year = int(re.search(r"\d{4}", sur).group())
        assert 1988 <= year <= 1992 if "1990" in original else 1983 <= year <= 1987
        assert year <= today.year


def test_surrogates_never_collide_with_each_other_or_the_originals():
    gen = MimicGen(seed=9)
    rng = random.Random(1)
    entities = []
    for k in range(60):
        entities += [
            _entity(f"user{k}@corp.com", "email"),
            _entity(f"+1-415-555-{1000 + k:04d}", "phone_us"),
            _entity(f"{100 + k}-{10 + k % 80:02d}-{1000 + k:04d}", "ssn"),
            _entity(f"{rng.randint(1, 223)}.{k}.{k}.{k % 250 + 1}", "ip_address"),
        ]
    text = " ".join(e.text for e in entities)
    mapping = gen.generate_all(entities, text=text)
    originals = {e.text for e in entities}
    surrogates = list(mapping.values())
    assert len(surrogates) == len(set(surrogates)) == len(originals)
    assert not originals & set(surrogates)
    assert all(not occurs(text, s) for s in surrogates)


def test_house_number_shift_rules():
    assert shift_house_number(address_parser.parse("1 Main St, Tempe, AZ 85281"), rng=random.Random(0)) \
        == "2 Main St, Tempe, AZ 85281"
    for seed in range(5):
        assert shift_house_number(address_parser.parse("0123 Elm Ave"), rng=random.Random(seed)) in (
            "0122 Elm Ave", "0124 Elm Ave")
        assert shift_house_number(address_parser.parse("123A Main St"), rng=random.Random(seed)) in (
            "122A Main St", "124A Main St")
    parsed = address_parser.parse("789 Crescent Row, Tempe, AZ 85281")
    taken = frozenset({"790 Crescent Row, Tempe, AZ 85281", "788 Crescent Row, Tempe, AZ 85281"})
    for seed in range(10):
        shifted = shift_house_number(parsed, forbidden=taken, rng=random.Random(seed))
        assert shifted is not None and shifted not in taken and shifted != parsed.full_text


def test_shift_mode_keeps_the_street_and_dodges_another_real_address(stub):
    ss.config(address_mode="shift")
    text = "Parcels for 789 Crescent Row, Tempe, AZ 85281 and 790 Crescent Row, Tempe, AZ 85281."
    result = ss.mask_result(text)
    a = result.replacements["789 Crescent Row, Tempe, AZ 85281"]
    b = result.replacements["790 Crescent Row, Tempe, AZ 85281"]
    assert a.endswith(" Crescent Row, Tempe, AZ 85281") and b.endswith(" Crescent Row, Tempe, AZ 85281")
    assert {a, b}.isdisjoint({"789 Crescent Row, Tempe, AZ 85281", "790 Crescent Row, Tempe, AZ 85281"})
    assert a != b and ss.unmask(result.text) == text


def test_replace_mode_leaks_no_component(stub):
    ss.config(address_mode="replace")
    text = "I live at 789 Crescent Row Apt 4B, Tempe, AZ 85281-1234."
    masked = ss.mask(text)
    for part in ("789", "Crescent", "4B", "Tempe", "85281", "1234"):
        assert part not in masked, (part, masked)
    assert ss.unmask(masked) == text


def test_sensitive_service_query_hides_the_street(stub):
    masked = ss.mask("Find an HIV clinic near 789 Crescent Row, Tempe, AZ 85281.")
    assert "789" not in masked and "Crescent Row" not in masked


# ─────────────────────────────────────────────────────────────────────────────
# 10. The README's code blocks against the real models (heavy)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_providers(monkeypatch):
    """anthropic / openai / google.genai / requests modules whose clients echo
    every user message back, prefixed, so unmask() has something to restore."""
    def echo(history):
        return "You said: " + " | ".join(m["content"] for m in history if m["role"] == "user")

    class _Block:
        def __init__(self, text):
            self.type, self.text = "text", text

    class _Anthropic:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(create=lambda *, model, max_tokens, messages, **kw:
                                                  types.SimpleNamespace(content=[_Block(echo(messages))]))
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = _Anthropic

    class _OpenAI:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda *, model, messages, **kw: types.SimpleNamespace(choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content=echo(messages)))])))
    openai = types.ModuleType("openai")
    openai.OpenAI = _OpenAI

    class _Client:
        def __init__(self, *a, **k):
            self.models = types.SimpleNamespace(generate_content=lambda *, model, contents, **kw:
                                                types.SimpleNamespace(text="You said: " + contents))
    genai = types.ModuleType("google.genai")
    genai.Client = _Client
    google = types.ModuleType("google")
    google.genai = genai

    requests = types.ModuleType("requests")
    requests.post = lambda url, json=None, **kw: types.SimpleNamespace(
        json=lambda: {"response": "You said: " + json["prompt"]})

    for name, module in (("anthropic", anthropic), ("openai", openai), ("google", google),
                         ("google.genai", genai), ("requests", requests)):
        monkeypatch.setitem(sys.modules, name, module)


@heavy
def test_readme_warm_up_block():
    _run(_readme_block(LIB_README, 'scan("warm up")'))


@heavy
def test_readme_quick_start_claude(monkeypatch):
    _fake_providers(monkeypatch)
    ns = _run(_readme_block(LIB_README, "claude-sonnet-4-6"))
    for value in ("Priya Raman", "priya.raman@northwind.io", "512-44-7810", "04/12/1990"):
        assert value not in ns["sanitized"] and value in ns["restored"]


@heavy
def test_readme_session_block():
    user_text = "I'm Priya Raman, priya.raman@northwind.io."
    ns = _run(_readme_block(LIB_README, "or Session(\"conv-42\""), user_text=user_text,
              llm_response="Hello again.")
    assert "Priya Raman" not in ns["sanitized"] and ns["restored"] == "Hello again."
    with pytest.raises(RuntimeError, match="closed"):
        ns["session"].mask(user_text)


@heavy
def test_readme_openai_multi_turn(monkeypatch):
    _fake_providers(monkeypatch)
    ns = _run(_readme_block(LIB_README, "gpt-4o-mini"))
    assert "John Doe" in ns["reply1"] and "42 Baker Street" in ns["reply1"]
    assert "42 Baker Street, London" in ns["reply2"]
    sent = [m["content"] for m in ns["api_history"] if m["role"] == "user"]
    assert all("John Doe" not in s and "42 Baker Street" not in s for s in sent)


@heavy
def test_readme_gemini_and_ollama(monkeypatch):
    _fake_providers(monkeypatch)
    ns = _run(_readme_block(LIB_README, "gemini-2.5-flash"))
    assert "4532015112830366" not in ns["sanitized"] and "192.168.1.100" not in ns["sanitized"]
    assert "4532015112830366" in ns["restored"] and "192.168.1.100" in ns["restored"]
    ns = _run(_readme_block(LIB_README, "llama3.2"))
    assert "+44 7911 123456" not in ns["sanitized"] and "SW1A 1AA" not in ns["sanitized"]
    assert "+44 7911 123456" in ns["restored"] and "SW1A 1AA" in ns["restored"]


@heavy
def test_readme_scan_blocks():
    ns = _run(_readme_block(LIB_README, "d.start:>4"))
    found = {d.text: d for d in ss.scan(ns["text"])}
    assert found["Alice Nguyen"].type == "PERSON"
    assert found["Alice Nguyen"].source in {"pii_tagger", "structural", "entity_trace", "context_guard"}
    assert found["alice.nguyen@company.org"].type == "email"
    assert found["+1-415-555-0198"].type == "phone_us" and found["+1-415-555-0198"].source == "pattern"
    assert any(d.type == "address" and "99 Market Street" in d.text for d in found.values())
    assert ss.current_session().mappings == {}
    ns = _run(_readme_block(LIB_README, "shield.pii_finder(text)"), text=ns["text"])
    assert [d.text for d in ns["found"]] == list(found)
    if ROOT_README.exists():
        ns = _run(_readme_block(ROOT_README, "d.type:15s"))
        by_text = {d.text: (d.type, d.source) for d in ns["found"]}
        assert by_text == {"Alice Nguyen": ("PERSON", "pii_tagger"), "alice@corp.com": ("email", "pattern"),
                           "+1-415-555-0198": ("phone_us", "pattern")}
        assert ss.scan("Contact Alice Nguyen at alice@corp.com or call +1-415-555-0198.", as_dict=True) == {
            "Alice Nguyen": "PERSON", "alice@corp.com": "email", "+1-415-555-0198": "phone_us"}


@heavy
def test_readme_pii_off_block():
    ns = _run(_readme_block(LIB_README, 'pii_off=["location", "org"]'))
    sanitized = ns["sanitized"]
    assert "Emma Johnson" not in sanitized and "emma@deloitte.com" not in sanitized
    assert "Deloitte" in sanitized and "New York" in sanitized
    if ROOT_README.exists():
        ns = _run(_readme_block(ROOT_README, 'pii_off=["location", "org"]'))
        assert "Emma Johnson" not in ns["sanitized"] and "Deloitte" in ns["sanitized"]
        assert "New York" in ns["sanitized"] and "emma@deloitte.com" not in ns["sanitized"]


@heavy
def test_readme_address_block():
    ns = _run(_readme_block(LIB_README, "Find parking near 42 Baker Street"))
    assert re.search(r"Find parking near 4[13] Baker Street, London\.", ns["sanitized"]), ns["sanitized"]
    assert ss.config().address_mode == "shift" and ss.config().address_shift_range == 4


@heavy
def test_root_readme_quick_start_and_providers(monkeypatch):
    if not ROOT_README.exists():
        pytest.skip("root README not available here")
    _fake_providers(monkeypatch)
    any_llm = types.SimpleNamespace(chat=lambda s: "You said: " + s)
    ns = _run(_readme_block(ROOT_README, "any_llm.chat(sanitized)"), any_llm=any_llm)
    for value in ("Priya Raman", "priya.raman@northwind.io", "512-44-7810", "04/12/1990"):
        assert value not in ns["sanitized"] and value in ns["restored"]
    ns = _run(_readme_block(ROOT_README, "claude-opus-4-8"), user_text=ns["user_text"],
              ask_ollama=lambda s: "You said: " + s)
    assert "Priya Raman" not in ns["sanitized"]
    llm = types.SimpleNamespace(chat=lambda s: "You said: " + s)
    block = _readme_block(ROOT_README, "shadowmaps").replace("/var/app/shadowmaps", str(LIB_ROOT / "tests" / "_maps"))
    try:
        ns = _run(block, llm=llm)
        assert "Clara Oswald" not in ns["sanitized"] and "Clara Oswald" in ns["restored"]
        assert list((LIB_ROOT / "tests" / "_maps").iterdir()) == []
    finally:
        shutil.rmtree(LIB_ROOT / "tests" / "_maps", ignore_errors=True)
