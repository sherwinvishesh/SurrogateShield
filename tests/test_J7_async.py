"""Gate J7 — the asyncio API: Session.amask / amask_result / ascan / aunmask
and ``async with``. Same results and exceptions as the sync methods, and a
slow detector does not block the event loop. Model-free: spaCy is stubbed
(optionally with a delay) and ContextGuard is off.
"""

import asyncio
import dataclasses
import time
import types

import pytest

from surrogateshield import Config, DetectorUnavailable, Session
from surrogateshield.core.detection import entity_trace

pytestmark = pytest.mark.usefixtures("stub_tagger")    # balanced runs the tagger

TEXT = "Mail me at dana.w@example.com, SSN 219-09-9999."
CFG = dataclasses.replace(Config(), context_guard_enabled=False)


def stub_nlp(monkeypatch, delay=0.0):
    def nlp(_t):
        time.sleep(delay)
        return types.SimpleNamespace(ents=[])
    monkeypatch.setattr(entity_trace, "_get_nlp", lambda *_a, **_k: nlp)


def test_J7_async_matches_sync(monkeypatch):
    stub_nlp(monkeypatch)
    with Session(config=CFG, seed=5) as s:
        sync = s.mask_result(TEXT)

    async def go():
        async with Session(config=CFG, seed=5) as s:
            r = await s.amask_result(TEXT)
            assert await s.amask(TEXT) == r.text                 # same session, same surrogates
            assert [d.text for d in await s.ascan(TEXT)] == [d.text for d in r.detections]
            back = await s.aunmask(f"Noted: {r.text}")
        return r, back, s
    r, back, s = asyncio.run(go())
    assert (r.text, r.replacements) == (sync.text, sync.replacements)
    assert back == f"Noted: {TEXT}"
    with pytest.raises(RuntimeError, match="closed"):                # async with closed it
        s.mask(TEXT)


def test_J7_async_does_not_block_the_loop(monkeypatch):
    stub_nlp(monkeypatch, delay=0.15)

    async def go():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1
        t = asyncio.create_task(ticker())
        sessions = [Session(config=CFG, seed=i) for i in range(4)]
        t0 = time.perf_counter()
        out = await asyncio.gather(*(s.amask(TEXT) for s in sessions))
        took = time.perf_counter() - t0
        t.cancel()
        return out, took, ticks
    out, took, ticks = asyncio.run(go())
    assert all("219-09-9999" not in o for o in out)
    assert ticks >= 5               # the loop ran while detection slept
    assert took < 4 * 0.15          # the four masks overlapped


def test_J7_async_raises_like_sync(monkeypatch):
    def boom(*_a, **_k):
        raise DetectorUnavailable("spaCy model missing")
    monkeypatch.setattr(entity_trace, "_get_nlp", boom)

    async def go():
        s = Session(config=CFG)
        with pytest.raises(DetectorUnavailable):
            await s.amask(TEXT)
        with pytest.raises(TypeError):
            await s.aunmask(None)
        with pytest.raises(TypeError):
            await s.ascan(b"bytes")
        assert s.mappings == {}                                      # fail closed
    asyncio.run(go())
