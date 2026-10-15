"""The security contract (pre-publish audit, pass 3): no egress, model
weights pinned and failing closed, the shadow map as the README describes
it, value-free logs at every level, fail-closed on every failure mode, and
bounded time on adversarial input.

The fast tests need no model. The heavy ones run the real models from the
local cache:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest python-library/tests/test_security_contract.py -m heavy
"""

import ast
import hashlib
import json
import logging
import os
import random
import re
import socket
import stat
import sys
import time
from pathlib import Path

import pytest

import surrogateshield as ss
from surrogateshield._state import Config
from surrogateshield.core.detection import config as dcfg
from surrogateshield.core.detection import context_guard as cg
from surrogateshield.core.detection import entity_trace as et
from surrogateshield.core.detection import pii_tagger as T
from surrogateshield.core.detection import plugins
from surrogateshield.core.errors import DetectorUnavailable
from surrogateshield.core.generation.mimic import MimicGen
from surrogateshield.core.storage import shadow_map as sm

heavy = pytest.mark.heavy
PACKAGE = Path(ss.__file__).resolve().parent
ZERO_PIN = "sha256:" + "0" * 64

_MESSAGE = ("Hi, I'm Priyanka Venkataraghavan (priyanka.v@example.com, +1 415 555 0134). "
            "SSN 123-45-6789, card 4111 1111 1111 1111.")


@pytest.fixture
def no_network(monkeypatch):
    """Every socket operation raises: a test that passes here made no request."""
    def refuse(*a, **k):
        raise AssertionError("network access")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def fresh_models(monkeypatch):
    """No loaded model or detector instance survives from another test (and
    none loaded here survives into one)."""
    monkeypatch.setattr(T, "_models", {})
    monkeypatch.setattr(plugins, "_instances", {})
    monkeypatch.setattr(cg, "_ner_pipelines", {})
    monkeypatch.setattr(et, "_nlp", {})


def _tagger_only():
    """``balanced`` with the tagger as its only model stage (no spaCy to load)."""
    return (dcfg.preset("balanced").with_stage("entity_trace", enabled=False)
            .with_stage("context_guard", enabled=False))


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


# ─────────────────────────────────────────────────────────────────────────────
# 1. Egress
# ─────────────────────────────────────────────────────────────────────────────

def test_the_package_imports_no_network_module():
    banned = {"socket", "ssl", "requests", "urllib", "urllib3", "httpx", "aiohttp", "http", "ftplib", "smtplib"}
    found = [f"{p.relative_to(PACKAGE)}: {m}" for p in PACKAGE.rglob("*.py")
             for m in _imports(p) if m.split(".")[0] in banned]
    assert found == []
    # the only downloader, and only where a model is fetched
    hub = sorted(str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*.py")
                 if any(m.split(".")[0] == "huggingface_hub" for m in _imports(p)))
    assert hub == ["cli.py", "core/detection/context_guard.py", "core/detection/pii_tagger.py"]


def test_no_unsafe_deserialiser_in_the_package():
    loaders = {"pickle", "cPickle", "dill", "shelve", "marshal"}
    found = [f"{p.relative_to(PACKAGE)}: {m}" for p in PACKAGE.rglob("*.py")
             for m in _imports(p) if m.split(".")[0] in loaders]
    text_hits = [f"{p.relative_to(PACKAGE)}:{i}" for p in PACKAGE.rglob("*.py")
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if re.search(r"torch\.load\(|weights_only\s*=\s*False|yaml\.(unsafe_)?load\(|(?<![\w.])(eval|exec)\(", line)]
    assert found == [] and text_hits == []


def test_masking_makes_no_network_request_without_models(no_network, tmp_path):
    ss.config(detection="no_models", pii_mem=str(tmp_path))
    found = {d.text for d in ss.scan(_MESSAGE)}
    assert {"priyanka.v@example.com", "123-45-6789"} <= found
    out = ss.mask(_MESSAGE)
    assert "priyanka.v@example.com" not in out and "123-45-6789" not in out
    back = ss.unmask("Noted: " + out)
    assert "priyanka.v@example.com" in back and "123-45-6789" in back
    ss.flush()
    with ss.Session("egress", storage_dir=str(tmp_path)) as s:
        s.mask(_MESSAGE)
        assert (tmp_path / "egress.shadowmap").exists()


@heavy
def test_masking_makes_no_network_request_with_the_models(no_network, fresh_models, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)        # even when a download would be allowed
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    for preset in ("balanced", "classic"):
        ss.config(detection=preset)
        out = ss.mask(_MESSAGE)
        assert "Priyanka" not in out and "Venkataraghavan" not in out and "123-45-6789" not in out, preset
        ss.flush()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_the_default_tagger_is_pinned_by_the_sha256_of_its_weights():
    assert T._PIN.fullmatch(dcfg.PII_TAGGER_REVISION)
    assert Config().tagger_revision == dcfg.PII_TAGGER_REVISION
    assert Config().tagger_model == dcfg.PII_TAGGER_MODEL
    repo, commit = dcfg.TAGGER_HUB_SOURCES[dcfg.PII_TAGGER_MODEL]
    assert repo == dcfg.PII_TAGGER_HUB and re.fullmatch(r"[0-9a-f]{40}", commit)
    for preset in ("balanced", "fast", "strict"):
        stage = dcfg.preset(preset).stage("pii_tagger")
        assert stage.enabled and stage.model == dcfg.PII_TAGGER_MODEL, preset
        assert stage.revision == dcfg.PII_TAGGER_REVISION, preset
    assert T._hub(dcfg.PII_TAGGER_MODEL, dcfg.PII_TAGGER_REVISION) == (repo, commit, True)


@pytest.mark.parametrize("name", ["../no-such-model", "/abs/model", "~/model", "a/b/c", "nohub", ".hidden/x", ""])
def test_a_model_name_that_is_neither_a_folder_nor_a_hub_id_is_refused(name, tmp_path, monkeypatch, no_network):
    monkeypatch.setenv(T.MODELS_ENV, str(tmp_path))
    with pytest.raises(DetectorUnavailable, match="not a Hugging Face model id"):
        T.model_path(name, None, download=False)
    assert T.local_dir(name) is None


def test_tampered_weights_fail_closed_before_torch_is_imported(tmp_path, monkeypatch, no_network, fresh_models):
    folder = tmp_path / dcfg.PII_TAGGER_MODEL
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"not the model")
    monkeypatch.setenv(T.MODELS_ENV, str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)             # importing torch would raise
    ss.config(detection=_tagger_only())
    with pytest.raises(DetectorUnavailable, match="not the pinned ones") as info:
        ss.mask("Priyanka Venkataraghavan, priyanka.v@example.com")
    assert "Priyanka" not in str(info.value)
    with pytest.raises(DetectorUnavailable, match="not the pinned ones"):
        ss.scan("Priyanka Venkataraghavan")
    assert ss.Session(config=ss.config()).mappings == {}
    # the pin of these bytes opens the folder: the next step is the (blocked) torch import
    pin = "sha256:" + hashlib.sha256(b"not the model").hexdigest()
    ss.config(detection=_tagger_only().with_stage("pii_tagger", revision=pin))
    with pytest.raises(DetectorUnavailable, match="needs transformers and torch"):
        ss.mask("Priyanka Venkataraghavan")


def test_a_missing_or_malformed_pin_is_refused(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"w")
    good = "sha256:" + hashlib.sha256(b"w").hexdigest()
    T.check_pin(str(tmp_path), good)
    assert T.weights_sha256(str(tmp_path)) == good.split(":")[1]
    for bad in (good.upper(), good[:-1], good + "0", "md5:" + "0" * 32, "main", "sha256:"):
        with pytest.raises(DetectorUnavailable, match="sha256:<64 hex>"):
            T.check_pin(str(tmp_path), bad)
    with pytest.raises(DetectorUnavailable, match="not the pinned ones"):
        T.check_pin(str(tmp_path), ZERO_PIN)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(DetectorUnavailable, match="no weights file"):
        T.check_pin(str(tmp_path), good)


def test_the_cached_default_tagger_is_checked_against_its_pin(no_network):
    with pytest.raises(DetectorUnavailable, match="not the pinned ones|not downloaded yet"):
        T.model_path(dcfg.PII_TAGGER_MODEL, ZERO_PIN, download=False)
    path = T.cached_path(dcfg.PII_TAGGER_MODEL, dcfg.PII_TAGGER_REVISION)
    if path is None:
        pytest.skip("the default tagger is not downloaded here")
    assert T.model_path(dcfg.PII_TAGGER_MODEL, dcfg.PII_TAGGER_REVISION) == path
    assert T.weights_sha256(path) == dcfg.PII_TAGGER_REVISION.split(":")[1]


def test_an_uncached_hub_model_fails_closed_offline_and_without_download(monkeypatch, no_network):
    calls = []

    def fake(repo, revision=None, local_files_only=False):
        calls.append((repo, revision, local_files_only))
        raise OSError("LocalEntryNotFoundError: not cached")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(DetectorUnavailable, match="not downloaded yet and HF_HUB_OFFLINE is set"):
        T.model_path("org/model", None)
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(DetectorUnavailable, match="not downloaded yet") as info:
        T.model_path("org/model", None, download=False)
    assert "HF_HUB_OFFLINE" not in str(info.value)
    assert calls and all(local for _r, _v, local in calls)      # only the cache was asked
    with pytest.raises(DetectorUnavailable, match="could not download"):
        T.model_path("org/model", "main")
    assert calls[-1] == ("org/model", "main", False)


def test_a_download_without_the_pinned_weights_fails_closed(monkeypatch, tmp_path):
    def fake(repo, revision=None, local_files_only=False):
        if local_files_only:
            raise OSError("not cached")
        return str(tmp_path)                                     # a download with no weights file
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(DetectorUnavailable, match="no weights file"):
        T.model_path("org/model", ZERO_PIN)
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(DetectorUnavailable, match="not the pinned ones"):
        T.model_path("org/model", ZERO_PIN)
    assert T.model_path("org/model", "sha256:" + hashlib.sha256(b"other weights").hexdigest()) == str(tmp_path)


@heavy
def test_garbage_weights_with_a_matching_pin_still_fail_to_load(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not a model")
    pin = "sha256:" + hashlib.sha256(b"not a model").hexdigest()
    with pytest.raises(DetectorUnavailable, match="could not load"):
        T.TaggerModel(str(tmp_path), pin, "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 3. A model's malformed output fails closed
# ─────────────────────────────────────────────────────────────────────────────

class _Fake:
    labels = T.LABELS

    def __init__(self, rows):
        self.rows = rows

    def token_probs(self, texts, window, stride, batch):
        return self.rows


def _tagger(model):
    t = T.PIITagger.__new__(T.PIITagger)
    t.model, t.window, t.stride, t.batch, t.merge_address = model, 256, 64, 8, True
    return t


def _row(label, s, e):
    p = [0.0] * len(T.LABELS)
    p[T.LABELS.index(label)] = 1.0
    return (s, e, p)


def test_a_well_formed_output_is_decoded():
    found = _tagger(_Fake([[_row("B-PERSON", 0, 5), _row("I-PERSON", 6, 12)]])).detect("Alice Nguyen", None)
    assert [(c.start, c.end, c.type) for c in found] == [(0, 12, "PERSON")]


@pytest.mark.parametrize("rows", [
    None,                                                       # not a list
    [],                                                         # no row for the text
    [[(0, 12, [])]],                                            # no probabilities
    [[(0, 12, [0.0] * (len(T.LABELS) + 1) + [1.0])]],           # more probabilities than labels
    [[_row("B-PERSON", 3, 999)]],                               # offsets past the end
    [[("a", "b", _row("B-PERSON", 0, 1)[2])]],                  # offsets that are not numbers
    [[_row("B-PERSON", 0, 5), None]],                           # a row that is not a triple
])
def test_malformed_output_is_detector_unavailable_without_the_text(rows):
    with pytest.raises(DetectorUnavailable) as info:
        _tagger(_Fake(rows)).detect("Alice Nguyen", None)
    assert "Alice" not in str(info.value) and "not masked" in str(info.value)


def test_detect_many_refuses_fewer_results_than_texts():
    with pytest.raises(DetectorUnavailable, match="1 results for 2 texts"):
        _tagger(_Fake([[_row("B-PERSON", 0, 5)]])).detect_many(["Alice", "Bob"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fail-closed per failure mode (mocks)
# ─────────────────────────────────────────────────────────────────────────────

def test_a_missing_spacy_model_fails_closed(monkeypatch, fresh_models):
    monkeypatch.setattr(et, "_load_nlp", lambda name: DetectorUnavailable(f"EntityTrace needs the spaCy model {name!r}"))
    ss.config(detection="classic")
    with pytest.raises(DetectorUnavailable, match="spaCy model"):
        ss.mask("Priyanka Venkataraghavan, priyanka.v@example.com")


def test_a_context_guard_that_cannot_load_fails_closed(monkeypatch, fresh_models):
    monkeypatch.setattr(cg, "_load_ner", lambda *a: DetectorUnavailable("ContextGuard could not load"))
    ss.config(detection=dcfg.preset("no_models").with_stage("context_guard", enabled=True))
    with pytest.raises(DetectorUnavailable, match="ContextGuard could not load"):
        ss.mask("Priyanka Venkataraghavan, priyanka.v@example.com")


def test_a_context_guard_that_fails_on_the_input_fails_closed(monkeypatch, fresh_models):
    def boom(text):
        raise RuntimeError("CUDA error on " + text)
    monkeypatch.setattr(cg, "_load_ner", lambda *a: boom)
    ss.config(detection=dcfg.preset("no_models").with_stage("context_guard", enabled=True))
    with pytest.raises(DetectorUnavailable, match="ContextGuard failed on this input") as info:
        ss.mask("Priyanka Venkataraghavan, priyanka.v@example.com")
    assert "Priyanka" not in str(info.value) and "CUDA" not in str(info.value)


def test_a_detector_that_raises_fails_closed_without_the_text(fresh_models):
    class _Boom:
        def detect(self, text, view=None):
            raise RuntimeError("crashed on " + text)
    plugins.register_detector("pii_tagger", lambda stage: _Boom(), replace=True)
    try:
        ss.config(detection=_tagger_only())
        s = ss.Session()
        with pytest.raises(DetectorUnavailable, match="'pii_tagger' detector failed") as info:
            s.mask("Priyanka Venkataraghavan, priyanka.v@example.com")
        assert "Priyanka" not in str(info.value) and "crashed" not in str(info.value)
        assert s.mappings == {}                                  # nothing was issued
    finally:
        plugins.unregister_detector("pii_tagger")


def test_a_detector_with_bad_offsets_cannot_mislabel_the_text(fresh_models):
    from surrogateshield.core.detection.plugins import Candidate

    class _Odd:
        def detect(self, text, view=None):
            return [Candidate(-3, 4, "PERSON", 0.9), Candidate(0, len(text) + 5, "PERSON", 0.9),
                    Candidate(5, 5, "PERSON", 0.9), Candidate("0", "4", "PERSON", 0.9),
                    Candidate(0, 8, "PERSON", 0.9)]
    plugins.register_detector("pii_tagger", lambda stage: _Odd(), replace=True)
    try:
        ss.config(detection=_tagger_only())
        text = "Priyanka Venkataraghavan writes."
        found = ss.scan(text)
        # the out-of-range candidates are dropped, not clamped; the one good span stays (the
        # structural pass may complete the name from it)
        assert len(found) == 1 and found[0].start == 0 and found[0].end <= len("Priyanka Venkataraghavan")
        assert all(text[d.start:d.end] == d.text and 0 <= d.start < d.end <= len(text) for d in found)
    finally:
        plugins.unregister_detector("pii_tagger")


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a directory this user cannot write")
def test_a_store_that_cannot_be_written_fails_closed(tmp_path, monkeypatch):
    ss.config(detection="no_models")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(ss.StorageError):
            ss.Session("s", storage_dir=str(locked)).mask("priyanka.v@example.com")
        monkeypatch.setenv("SURROGATESHIELD_HOME", str(locked / "home"))
        monkeypatch.setattr(sm, "_secret_cache", {})
        with pytest.raises(ss.StorageError, match="device secret"):
            sm.ShadowMap("s", tmp_path)
    finally:
        locked.chmod(0o700)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Side channels: no value in a log at any level
# ─────────────────────────────────────────────────────────────────────────────

_CANARY = "Zyxquvial"
_LOG_MESSAGE = (f"Hi, I'm Priyanka Venkataraghavan from {_CANARY} Ltd, priyanka.v@example.com, "
                "+1 415 555 0134, SSN 123-45-6789, card 4111 1111 1111 1111, IP 10.1.2.3, "
                "I live at 12 Oak Street, Springfield IL 62704, born 03/04/1990.")
_VALUES = ("Priyanka", "Venkataraghavan", _CANARY, "priyanka.v", "415 555 0134", "123-45-6789",
           "4111 1111 1111 1111", "10.1.2.3", "Oak Street", "62704", "03/04/1990")


def _exercise(tmp_path):
    s = ss.Session("hygiene", storage_dir=str(tmp_path))
    out = s.mask(_LOG_MESSAGE)
    s.unmask("Thanks, " + out)
    s.scan(_LOG_MESSAGE)
    s.forget("123-45-6789")
    again = ss.Session("hygiene", storage_dir=str(tmp_path))     # reopened: the map is loaded
    again.mask(_LOG_MESSAGE)
    again.close()
    ss.config(pii_mem=str(tmp_path))
    ss.mask_result(_LOG_MESSAGE)
    ss.unmask("ok")
    ss.flush()


def test_no_value_reaches_a_debug_log_without_models(caplog, tmp_path):
    ss.config(detection="no_models")
    with caplog.at_level(logging.DEBUG, logger="surrogateshield"):
        _exercise(tmp_path)
    assert any(r.levelno == logging.DEBUG for r in caplog.records), "no DEBUG line: the test proves nothing"
    for value in _VALUES:
        assert value not in caplog.text, value


@heavy
def test_no_value_reaches_a_debug_log_with_the_models(caplog, tmp_path):
    for preset in ("balanced", "classic", "strict"):
        caplog.clear()
        ss.config(detection=preset)
        with caplog.at_level(logging.DEBUG, logger="surrogateshield"):
            _exercise(tmp_path)
        assert any(r.levelno == logging.DEBUG for r in caplog.records), preset
        for value in _VALUES:
            assert value not in caplog.text, (preset, value)


# ─────────────────────────────────────────────────────────────────────────────
# 6. The shadow map, as the README describes it
# ─────────────────────────────────────────────────────────────────────────────

def test_the_shadow_map_file_is_what_the_readme_says(tmp_path):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    key_file = tmp_path / "home" / "device.key"
    secret = sm.device_secret(key_file)
    assert len(secret) == 32 and sm.device_secret(key_file) == secret      # made once
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o700
    store = tmp_path / "maps"
    m = sm.ShadowMap("sess-1", store, secret_path=key_file)
    m.update({"Mia Lopez": "Alice Nguyen"})
    path = store / "sess-1.shadowmap"
    blob = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(store.stat().st_mode) == 0o700
    assert not (store / "sess-1.key").exists()                             # no key beside the data
    assert blob[:4] == b"SSv1" and b"Alice" not in blob and b"Mia" not in blob
    # the README's recipe, by hand: HKDF-SHA256(secret, salt=id, info=kind), AES-256-GCM, aad=kind|id
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"sess-1", info=b"shadowmap-v1").derive(secret)
    plain = AESGCM(key).decrypt(blob[4:16], blob[16:], b"shadowmap-v1|sess-1")
    assert json.loads(plain) == {"Mia Lopez": "Alice Nguyen"}
    m.save()
    blob2 = path.read_bytes()
    assert blob2 != blob and blob2[4:16] != blob[4:16]                    # a fresh nonce per write
    assert sm.unseal(key, "sess-1", sm.KIND_SHADOWMAP, blob2) == plain
    other = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"sess-2", info=b"shadowmap-v1").derive(secret)
    for args in ((key, "sess-2", sm.KIND_SHADOWMAP, blob), (key, "sess-1", sm.KIND_TRANSCRIPT, blob),
                 (other, "sess-1", sm.KIND_SHADOWMAP, blob), (key, "sess-1", sm.KIND_SHADOWMAP, blob[:14]),
                 (key, "sess-1", sm.KIND_SHADOWMAP, blob[:-1] + bytes([blob[-1] ^ 1])),
                 (key, "sess-1", sm.KIND_SHADOWMAP, blob[:20] + bytes([blob[20] ^ 1]) + blob[21:])):
        with pytest.raises(sm.CorruptStoreError):
            sm.unseal(*args)
    assert sm.ShadowMap("sess-1", store, secret_path=key_file).get_all() == {"Mia Lopez": "Alice Nguyen"}


def test_a_session_file_opens_with_the_device_key_alone(tmp_path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    ss.config(detection="no_models")
    s = ss.Session("sess-9", storage_dir=str(tmp_path))
    out = s.mask("Mail priyanka.v@example.com")
    blob = (tmp_path / "sess-9.shadowmap").read_bytes()
    secret = (sm.home() / "device.key").read_bytes()
    key = sm.derive_key(secret, "sess-9", sm.KIND_SHADOWMAP)
    mappings = json.loads(AESGCM(key).decrypt(blob[4:16], blob[16:], b"shadowmap-v1|sess-9"))
    assert mappings == s.mappings and "priyanka.v@example.com" in mappings.values()
    assert all(v in out for v in mappings)
    s.close()
    assert not (tmp_path / "sess-9.shadowmap").exists()                   # close discards the file


def test_a_tampered_shadow_map_is_refused_and_moved_aside(tmp_path):
    ss.config(detection="no_models")
    ss.Session("tamper", storage_dir=str(tmp_path)).mask("priyanka.v@example.com")
    path = tmp_path / "tamper.shadowmap"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(sm.CorruptStoreError) as info:
        ss.Session("tamper", storage_dir=str(tmp_path))
    assert isinstance(info.value, ss.StorageError) and "priyanka" not in str(info.value)
    assert not path.exists()
    moved = list(tmp_path.glob("tamper.shadowmap.corrupt-*"))
    assert len(moved) == 1 and moved[0].read_bytes() == bytes(data)       # moved aside, not deleted
    fresh = ss.Session("tamper", storage_dir=str(tmp_path))                # the id is usable again
    assert fresh.mappings == {}


@pytest.mark.parametrize("bad", ["../x", "/abs", "a\0b", "ünïcode", "a" * 65, "", "a b", "x.shadowmap", "a/b"])
def test_an_unsafe_session_id_is_refused_before_any_file_is_touched(bad, tmp_path):
    with pytest.raises(ValueError):
        sm.ShadowMap(bad, tmp_path)
    with pytest.raises(ValueError):
        ss.Session(bad, storage_dir=str(tmp_path))
    with pytest.raises(ValueError):
        sm.ShadowMap.erase(bad, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert sm.validate_id("a" * 64) == "a" * 64 and sm.validate_id("A-z_09") == "A-z_09"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Surrogates are not predictable
# ─────────────────────────────────────────────────────────────────────────────

def test_surrogates_do_not_follow_the_global_random_state():
    ss.config(detection="no_models")
    text = "Mail priyanka.v@example.com or call 415-555-0134, card 4111 1111 1111 1111."
    random.seed(0)
    a = ss.Session().mask(text)
    random.seed(0)
    b = ss.Session().mask(text)
    assert a != b and "priyanka.v@example.com" not in a + b
    assert ss.Session(seed=7).mask(text) == ss.Session(seed=7).mask(text) != a
    random.seed(0)
    g1 = MimicGen()
    random.seed(0)
    g2 = MimicGen()
    assert g1._rng.random() != g2._rng.random() and g1._fake.name() != g2._fake.name()
    assert MimicGen(3)._fake.name() == MimicGen(3)._fake.name()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Input hardening
# ─────────────────────────────────────────────────────────────────────────────

_UNITS = {"newline": "\n", "space": " ", "tab": "\t", "dash": "-", "at": "@", "digit": "7", "plus": "+",
          "comma": ",", "dot": ".", "a@": "a@", "(1": "(1", "+1 ": "+1 ", "12/": "12/", "ssn-ish": "123-45-"}
_FRAMES = ("123 Main St", "I am 34", "Name: Alice", "12:00", "Dear", "Mr.", "Dr.", "Apt", "")
_PADS = {"space": " ", "tab": "\t", "newline": "\n", "comma": ", "}


def _mask_time(text):
    t = time.perf_counter()
    ss.mask(text)
    ss.flush()
    return time.perf_counter() - t


def test_adversarial_input_is_masked_in_bounded_time():
    """Every shape at 10k characters within 3 s (measured 0.04-0.13 s; the
    pre-pass-3 patterns took 7-25 s on the whitespace runs).

    Not in the grid, and still quadratic (deferred, P2): "sk-" repeated
    (1.9 s at 10k, 16.9 s at 30k), "1-" repeated (2.3 s / 20.1 s) and "a."
    repeated (1.8 s / 16.1 s). The e-mail local part, the spelled-number
    canonicaliser and the name-token frames are (?:X+[-.])*X+ shapes whose
    unambiguous form needs a possessive quantifier (Python 3.11; the floor
    is 3.10). A host that masks untrusted text should cap the length.
    """
    ss.config(detection="no_models")
    ss.mask("warm up")
    slow = {}
    for name, unit in _UNITS.items():
        if (dt := _mask_time(unit * (10_000 // len(unit)))) > 3:
            slow[name] = round(dt, 2)
    for frame in _FRAMES:
        for pname, pad in _PADS.items():
            if (dt := _mask_time(frame + pad * (10_000 // len(pad)) + "x.")) > 3:
                slow[f"{frame!r}+{pname}"] = round(dt, 2)
    assert not slow, slow


def test_whitespace_runs_scale_linearly():
    """Three times the input, at most five times the time (linear is 3x,
    the quadratic patterns of before were 9x)."""
    ss.config(detection="no_models")
    ss.mask("warm up")
    for frame, pad in (("", " "), ("", "\t"), ("", "\n"), ("123 Main St", " "), ("I am 34", " "),
                       ("Name: Alice", " "), ("12:00", "\t")):
        small = min(_mask_time(frame + pad * 10_000 + "x.") for _ in range(2))
        large = min(_mask_time(frame + pad * 30_000 + "x.") for _ in range(2))
        assert large <= max(5 * small, 0.4), (frame, pad, small, large)


def test_a_long_message_is_scanned_to_the_end():
    ss.config(detection="no_models")
    text = "The quick brown fox jumps over the lazy dog. " * 700
    text += "My SSN is 123-45-6789 and my email is priyanka.v@example.com."
    assert len(text) > 30_000
    found = {(d.text, d.start) for d in ss.scan(text)}
    assert ("123-45-6789", text.index("123-45-6789")) in found
    assert ("priyanka.v@example.com", text.index("priyanka.v@example.com")) in found
    out = ss.mask(text)
    assert "123-45-6789" not in out and "priyanka.v@example.com" not in out
    assert ss.unmask(out) == text


@heavy
def test_a_long_message_is_scanned_to_the_end_with_the_models():
    ss.config()
    text = "Priyanka Venkataraghavan here. " + "The quick brown fox jumps over the lazy dog. " * 700
    text += "Again, this is Priyanka Venkataraghavan; my SSN is 123-45-6789."
    out = ss.mask(text)
    assert "Priyanka" not in out and "Venkataraghavan" not in out and "123-45-6789" not in out
    assert ss.unmask(out) == text
