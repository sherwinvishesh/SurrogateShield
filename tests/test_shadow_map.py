"""ShadowMap: forward index, encryption round-trip, flush semantics, and the
storage hardening of audit I11 / F1 / I19 (gate J6)."""

import json
import os
import stat
import threading
import uuid
from pathlib import Path

import pytest

import surrogateshield.core.storage.shadow_map as store

from surrogateshield.core.storage.shadow_map import ShadowMap as LibShadowMap
from storage.logic import ShadowMap as RootShadowMap


# ── Library ShadowMap (memory mode) ───────────────────────────────────────────

def test_memory_mode_forward_index():
    sm = LibShadowMap("session-a")
    sm.update({"790 Crescent Row": "789 Crescent Row", "FakeCo": "RealCo"})
    assert sm.get_all() == {"790 Crescent Row": "789 Crescent Row", "FakeCo": "RealCo"}
    assert sm.lookup_original("789 Crescent Row") == "790 Crescent Row"
    assert sm.lookup_original("RealCo") == "FakeCo"
    assert sm.lookup_original("unknown") is None
    assert set(sm.originals()) == {"789 Crescent Row", "RealCo"}


def test_memory_mode_flush_clears_both_indexes():
    sm = LibShadowMap("session-b")
    sm.update({"x": "y"})
    sm.flush()
    assert len(sm) == 0
    assert sm.lookup_original("y") is None
    assert not list(sm.originals())


# ── Library ShadowMap (persistent, AES-256-GCM) ───────────────────────────────

def test_persistent_encryption_roundtrip(tmp_path):
    session = str(uuid.uuid4())
    sm = LibShadowMap(session, storage_dir=str(tmp_path))
    sm.update({"790 Crescent Row, Tempe, AZ": "789 Crescent Row, Tempe, AZ"})

    # file exists and is NOT plaintext
    map_file = tmp_path / f"{session}.shadowmap"
    assert map_file.exists()
    raw = map_file.read_bytes()
    assert b"Crescent" not in raw, "shadow map must be encrypted on disk"

    # a new instance for the same session reads it back — including the index
    sm2 = LibShadowMap(session, storage_dir=str(tmp_path))
    assert sm2.get_all() == {"790 Crescent Row, Tempe, AZ": "789 Crescent Row, Tempe, AZ"}
    assert sm2.lookup_original("789 Crescent Row, Tempe, AZ") == "790 Crescent Row, Tempe, AZ"


def test_persistent_flush_deletes_files(tmp_path):
    session = str(uuid.uuid4())
    sm = LibShadowMap(session, storage_dir=str(tmp_path))
    sm.update({"a": "b"})
    assert (tmp_path / f"{session}.shadowmap").exists()
    sm.flush()
    assert not (tmp_path / f"{session}.shadowmap").exists()
    assert not (tmp_path / f"{session}.key").exists()


# ── Root ShadowMap (in-memory operations) ─────────────────────────────────────

def test_root_forward_index_add_and_update():
    sm = RootShadowMap(str(uuid.uuid4()))
    sm.add("surrogate-1", "original-1")
    sm.update({"surrogate-2": "original-2"})
    assert sm.lookup_original("original-1") == "surrogate-1"
    assert sm.lookup_original("original-2") == "surrogate-2"
    assert sm.get("surrogate-1") == "original-1"
    assert set(sm.originals()) == {"original-1", "original-2"}
    assert sm.lookup_original("nope") is None


# ── J6 / I11: storage hardening ───────────────────────────────────────────────

def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def ss_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("SURROGATESHIELD_HOME", str(home))
    monkeypatch.setattr(store, "_secret_cache", {})
    return home


def test_I11_permissions_files_0600_dirs_0700(ss_home, tmp_path):
    d = tmp_path / "maps"
    sm = LibShadowMap("perm-test", storage_dir=d)
    sm.update({"s": "o"})
    assert _mode(ss_home) == 0o700
    assert _mode(ss_home / "device.key") == 0o600
    assert _mode(d) == 0o700
    assert _mode(d / "perm-test.shadowmap") == 0o600
    assert not list(d.glob(".tmp-*")), "temp files left behind"


def test_I11_loose_permissions_tightened(ss_home):
    ss_home.mkdir(mode=0o755)
    os.chmod(ss_home, 0o755)
    key = ss_home / "device.key"
    key.write_bytes(os.urandom(32))
    os.chmod(key, 0o644)
    store.device_secret()
    assert _mode(key) == 0o600 and _mode(ss_home) == 0o700


def test_I11_default_store_is_home_anchored(ss_home, tmp_path, monkeypatch):
    import config
    from storage.logic import ShadowMap as AppShadowMap, conversations_dir

    monkeypatch.setattr(config, "SHADOWMAP_DIR", None)
    monkeypatch.chdir(tmp_path)
    assert conversations_dir() == ss_home / "conversations"
    sm = AppShadowMap("home-test")
    sm.add("s", "o")
    sm.save()
    assert (ss_home / "conversations" / "home-test.shadowmap").exists()
    assert not (tmp_path / "conversations").exists()


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "x" * 65, "a.b", "évil", None])
def test_I11_id_validation(bad, tmp_path):
    with pytest.raises(ValueError):
        LibShadowMap(bad, storage_dir=tmp_path)
    with pytest.raises(ValueError):
        LibShadowMap.erase(bad, tmp_path)


def test_I11_corrupt_file_raises_and_is_kept(ss_home, tmp_path):
    sm = LibShadowMap("corrupt", storage_dir=tmp_path)
    sm.update({"a": "b"})
    path = tmp_path / "corrupt.shadowmap"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(bytes(blob))
    with pytest.raises(store.CorruptStoreError, match="moved to"):
        LibShadowMap("corrupt", storage_dir=tmp_path)
    assert not path.exists()
    (aside,) = tmp_path.glob("corrupt.shadowmap.corrupt-*")
    assert aside.read_bytes() == bytes(blob), "the evidence is moved, never deleted"


def test_I11_truncated_file_raises(ss_home, tmp_path):
    (tmp_path / "trunc.shadowmap").write_bytes(store.MAGIC + b"\0" * 5)
    with pytest.raises(store.CorruptStoreError):
        LibShadowMap("trunc", storage_dir=tmp_path)


def test_I11_aad_binds_kind_and_id(ss_home):
    secret = store.device_secret()
    k = store.derive_key(secret, "conv-a", store.KIND_SHADOWMAP)
    blob = store.seal(k, "conv-a", store.KIND_SHADOWMAP, b"{}")
    assert store.unseal(k, "conv-a", store.KIND_SHADOWMAP, blob) == b"{}"
    with pytest.raises(store.CorruptStoreError):      # same key, other id
        store.unseal(k, "conv-b", store.KIND_SHADOWMAP, blob)
    with pytest.raises(store.CorruptStoreError):      # same key, other kind
        store.unseal(k, "conv-a", store.KIND_TRANSCRIPT, blob)
    assert store.derive_key(secret, "conv-a", store.KIND_TRANSCRIPT) != k


def test_I11_copied_file_does_not_open_under_another_id(ss_home, tmp_path):
    LibShadowMap("victim", storage_dir=tmp_path).update({"s": "o"})
    (tmp_path / "victim.shadowmap").rename(tmp_path / "other.shadowmap")
    with pytest.raises(store.CorruptStoreError):
        LibShadowMap("other", storage_dir=tmp_path)


def test_I11_atomic_write_keeps_old_file_on_failure(ss_home, tmp_path, monkeypatch):
    sm = LibShadowMap("atomic", storage_dir=tmp_path)
    sm.update({"s1": "o1"})
    before = (tmp_path / "atomic.shadowmap").read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        sm.update({"s2": "o2"})
    assert (tmp_path / "atomic.shadowmap").read_bytes() == before
    assert not list(tmp_path.glob(".tmp-*"))


def test_I11_per_value_erasure(ss_home, tmp_path):
    sm = LibShadowMap("forget", storage_dir=tmp_path)
    sm.update({"Zoe Park": "Ann Lee", "Zoe": "Ann Lee", "X Corp": "Acme"})
    assert sm.forget("Ann Lee") == 2
    assert sm.lookup_original("Ann Lee") is None
    again = LibShadowMap("forget", storage_dir=tmp_path)
    assert again.get_all() == {"X Corp": "Acme"}


def test_I11_erase_works_on_unreadable_file(tmp_path):
    (tmp_path / "gone.shadowmap").write_bytes(b"junk")
    LibShadowMap.erase("gone", tmp_path)
    assert not (tmp_path / "gone.shadowmap").exists()


def test_I11_library_key_not_beside_ciphertext(ss_home, tmp_path):
    LibShadowMap("lib", storage_dir=tmp_path / "maps").update({"s": "o"})
    assert sorted(p.name for p in (tmp_path / "maps").iterdir()) == ["lib.shadowmap"]


def test_I11_pre_v1_files_still_open_and_upgrade(ss_home, tmp_path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    secret = store.device_secret()
    nonce = os.urandom(12)
    legacy = nonce + AESGCM(store.legacy_key(secret, "old")).encrypt(
        nonce, json.dumps({"s": "o"}).encode(), None)
    (tmp_path / "old.shadowmap").write_bytes(legacy)
    sm = LibShadowMap("old", storage_dir=tmp_path)
    assert sm.get_all() == {"s": "o"}
    sm.save()
    assert (tmp_path / "old.shadowmap").read_bytes().startswith(store.MAGIC)
    # pre-v1 library layout: per-session key file beside the map
    sess_key = os.urandom(32)
    (tmp_path / "lib-old.key").write_bytes(sess_key)
    (tmp_path / "lib-old.shadowmap").write_bytes(nonce + AESGCM(
        store.legacy_key(sess_key, "lib-old")).encrypt(nonce, b'{"a": "b"}', None))
    sm = LibShadowMap("lib-old", storage_dir=tmp_path)
    assert sm.get_all() == {"a": "b"}
    sm.save()
    assert not (tmp_path / "lib-old.key").exists()


# ── F1: no silent ephemeral key ───────────────────────────────────────────────

def test_F1_no_silent_ephemeral_key(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_secret_cache", {})
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(store.ShadowMapStorageError):
            store.device_secret(ro / "sub" / "device.key")
        with pytest.raises(store.ShadowMapStorageError):
            LibShadowMap("f1", storage_dir=tmp_path / "maps", secret_path=ro / "sub" / "device.key")
        # opt-in: one secret for the whole process, not one per call
        a = store.device_secret(ro / "sub" / "device.key", allow_ephemeral=True)
        b = store.device_secret(ro / "sub" / "device.key", allow_ephemeral=True)
        assert a == b and len(a) == 32
    finally:
        os.chmod(ro, 0o700)


def test_F1_truncated_device_key_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_secret_cache", {})
    key = tmp_path / "device.key"
    key.write_bytes(b"")
    with pytest.raises(store.ShadowMapStorageError, match="corrupt"):
        store.device_secret(key)
    with pytest.raises(store.ShadowMapStorageError, match="corrupt"):
        store.device_secret(key, allow_ephemeral=True)


def test_F1_concurrent_first_run_agrees_on_one_secret(tmp_path, monkeypatch):
    key = tmp_path / "k" / "device.key"
    results = []

    def worker():
        results.append(store._create_secret(key) or key.read_bytes())

    (tmp_path / "k").mkdir()
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1 and len(results[0]) == 32


# ── I19: transcripts ──────────────────────────────────────────────────────────

def _chat_with(conv_id, text):
    from chatbot.chat import ClaudeChat
    from util import Conversation, ConversationMessage

    chat = ClaudeChat.__new__(ClaudeChat)
    chat.conversation = Conversation(id=conv_id)
    chat.conversation.messages.append(ConversationMessage(role="user", content=text))
    chat.conversation.api_messages.append(ConversationMessage(role="user", content="Zoe"))
    return chat


@pytest.fixture
def app_store(ss_home, monkeypatch):
    import config
    from chatbot.chat import ClaudeChat

    # no provider client (and no API key) needed to read a transcript
    monkeypatch.setattr(ClaudeChat, "__init__",
                        lambda self, conversation=None: setattr(self, "conversation", conversation))
    monkeypatch.setattr(config, "SHADOWMAP_DIR", None)
    monkeypatch.setattr(config, "DEVICE_KEY_PATH", None)
    return ss_home / "conversations"


def test_I19_transcript_roundtrip_sealed(app_store):
    from chatbot.chat import ClaudeChat

    _chat_with("t-1", "Ann Lee").save()
    path = app_store / "t-1.json"
    assert _mode(path) == 0o600 and b"Ann" not in path.read_bytes()
    assert path.read_bytes().startswith(store.MAGIC)
    loaded = ClaudeChat.load("t-1")
    assert loaded.conversation.messages[0].content == "Ann Lee"
    # a transcript is not a shadow map: kind is bound into the AAD
    (app_store / "t-1.json").rename(app_store / "t-1.shadowmap")
    with pytest.raises(store.CorruptStoreError):
        from storage.logic import ShadowMap
        ShadowMap("t-1")


def test_I19_undecryptable_transcript_is_valueerror_not_legacy(app_store):
    from chatbot.chat import ClaudeChat

    _chat_with("t-2", "Ann Lee").save()
    path = app_store / "t-2.json"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="cannot be decrypted"):
        ClaudeChat.load("t-2")
    (row,) = ClaudeChat.list_conversations()
    assert row["readable"] is False and row["id"] == "t-2"


def test_I19_plaintext_legacy_transcript_loads(app_store):
    from chatbot.chat import ClaudeChat

    app_store.mkdir(parents=True)
    (app_store / "t-3.json").write_text(json.dumps(
        {"id": "t-3", "messages": [{"role": "user", "content": "hi"}],
         "api_messages": [{"role": "user", "content": "hi"}]}))
    assert ClaudeChat.load("t-3").conversation.messages[0].content == "hi"


def test_I11_load_and_delete_reject_bad_ids(app_store):
    from chatbot.chat import ClaudeChat
    import main

    for bad in ("../evil", "a/b"):
        with pytest.raises(ValueError):
            ClaudeChat.load(bad)
        with pytest.raises(ValueError):
            main._delete_conversation(bad)


def test_I11_delete_removes_unreadable_conversation(app_store):
    import main
    from storage.logic import ShadowMap

    _chat_with("t-4", "Ann").save()
    sm = ShadowMap("t-4")
    sm.add("s", "o")
    sm.save()
    for p in app_store.iterdir():
        p.write_bytes(b"garbage")
    main._delete_conversation("t-4")
    assert list(app_store.iterdir()) == []
