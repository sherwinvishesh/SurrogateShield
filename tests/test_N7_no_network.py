"""N7 / C5 — masking makes no network request: a cached ContextGuard model
is loaded from disk without asking the Hub, and only a model that is not
cached yet is fetched."""

import socket

import pytest

from surrogateshield.core.detection import context_guard as cg


def test_N7_cached_model_is_resolved_locally(monkeypatch):
    calls = []

    def fake(repo, local_files_only=False):
        calls.append(local_files_only)
        return "/cache/" + repo
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert cg._local_model("org/model") == "/cache/org/model"
    assert calls == [True]


def test_N7_missing_model_is_fetched_once_unless_offline(monkeypatch):
    calls = []

    def fake(repo, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError(repo)
        return "/cache/" + repo
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert cg._local_model("org/model") == "/cache/org/model"
    assert calls == [True, False]
    calls.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert cg._local_model("org/model") == "org/model"     # the pipeline then fails closed
    assert calls == [True]


def test_N7_local_directory_is_used_as_is(tmp_path):
    assert cg._local_model(str(tmp_path)) == str(tmp_path)


@pytest.mark.heavy
def test_N7_C5_mask_opens_no_socket(monkeypatch):
    import surrogateshield as sh

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    cg._ner_pipelines.clear()

    def refuse(self, *a, **k):
        raise AssertionError("network access during masking")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    with sh.Session(seed=0) as s:
        out = s.mask("Hi, I'm Priya Raman, call me on 0161 496 0732 about Lakeside Dental.")
    assert "Priya" not in out and "496 0732" not in out
