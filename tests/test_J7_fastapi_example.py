"""Gate J7 — python-library/examples/fastapi_app.py works as documented:
mask → unmask round trip per conversation, server-created ids only, 503 and
no text when detection fails, LRU eviction closes the oldest session, and no
response contains an original. Skipped without fastapi/httpx; model-free.
"""

import dataclasses
import importlib.util
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient   # noqa: E402

from surrogateshield import Config, DetectorUnavailable   # noqa: E402
from surrogateshield.core.detection import entity_trace   # noqa: E402

pytestmark = pytest.mark.usefixtures("stub_tagger")    # balanced runs the tagger

_path = Path(__file__).resolve().parent.parent / "python-library" / "examples" / "fastapi_app.py"
_spec = importlib.util.spec_from_file_location("fastapi_app", _path)

TEXT = "Mail me at dana.w@example.com, SSN 219-09-9999."
CFG = dataclasses.replace(Config(), context_guard_enabled=False)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(entity_trace, "_get_nlp", lambda *_a, **_k: (lambda t: types.SimpleNamespace(ents=[])))
    mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    return mod


def test_J7_fastapi_round_trip(example):
    with TestClient(example.make_app(CFG)) as c:
        r = c.post("/mask", json={"text": TEXT}).json()
        assert "dana.w@example.com" not in r["text"] and "219-09-9999" not in r["text"]
        cid = r["conversation"]
        r2 = c.post("/mask", json={"text": "Again: 219-09-9999", "conversation": cid}).json()
        assert r2["conversation"] == cid and r["text"].split("SSN ")[1][:11] in r2["text"]
        back = c.post("/unmask", json={"conversation": cid, "text": "Got it: " + r["text"]}).json()
        assert back == {"text": "Got it: " + TEXT}
        assert c.delete(f"/conversations/{cid}").status_code == 204
        assert c.post("/unmask", json={"conversation": cid, "text": "x"}).status_code == 404


def test_J7_fastapi_client_cannot_create_ids(example):
    with TestClient(example.make_app(CFG)) as c:
        assert c.post("/mask", json={"text": TEXT, "conversation": "mine"}).status_code == 404


def test_J7_fastapi_fails_closed(example, monkeypatch):
    with TestClient(example.make_app(CFG)) as c:
        def boom(*_a, **_k):
            raise DetectorUnavailable("model gone")
        monkeypatch.setattr(entity_trace, "_get_nlp", boom)
        r = c.post("/mask", json={"text": TEXT})
        assert r.status_code == 503 and "219-09-9999" not in r.text and "dana" not in r.text


def test_J7_fastapi_startup_fails_without_models(example, monkeypatch):
    def boom(*_a, **_k):
        raise DetectorUnavailable("model gone")
    monkeypatch.setattr(entity_trace, "_get_nlp", boom)
    with pytest.raises(DetectorUnavailable):
        with TestClient(example.make_app(CFG)):
            pass


def test_J7_fastapi_evicts_oldest(example):
    with TestClient(example.make_app(CFG, max_conversations=2)) as c:
        ids = [c.post("/mask", json={"text": TEXT}).json()["conversation"] for _ in range(3)]
        assert c.post("/unmask", json={"conversation": ids[0], "text": "x"}).status_code == 404
        assert c.post("/unmask", json={"conversation": ids[2], "text": "x"}).status_code == 200
