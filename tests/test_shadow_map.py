"""ShadowMap: forward index, encryption round-trip, flush semantics."""

import uuid

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


def test_corrupt_file_degrades_to_empty(tmp_path):
    session = str(uuid.uuid4())
    sm = LibShadowMap(session, storage_dir=str(tmp_path))
    sm.update({"a": "b"})
    (tmp_path / f"{session}.shadowmap").write_bytes(b"garbage-not-encrypted")
    sm2 = LibShadowMap(session, storage_dir=str(tmp_path))
    assert sm2.get_all() == {}
    assert sm2.lookup_original("b") is None


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
