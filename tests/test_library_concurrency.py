"""Gate J7 — the pip library: Session objects, no shared PII state, thread
safety, explicit errors, offsets, silent by default (audit I9, I10).

The fast tests replace the detection cascade with a deterministic regex stub
(e-mail addresses and a fixed name list), so they need no models and exercise
only the session machinery: surrogate maps, locks, context binding. The heavy
test at the end repeats the audit's cross-user scenario with the real models.
"""

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import surrogateshield as ss
from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity

NAMES = ["Alice Brown", "Bob Okafor", "Chen Wei", "Dana Levi"]
_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|" + "|".join(map(re.escape, NAMES)))


def _stub_cascade(text, **_kw):
    found = [DetectedEntity(m.group(), m.start(), m.end(),
                            "email" if "@" in m.group() else "person", 0.99, "stub")
             for m in _PII.finditer(text)]
    return found, []


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(pipeline, "run_cascade", _stub_cascade)


# ── isolation ────────────────────────────────────────────────────────────────

def test_I9_sessions_do_not_restore_each_others_values(stub):
    # The audit's repro: user B's unmask() restored user A's real e-mail.
    a, b = ss.Session(), ss.Session()
    sent_a = a.mask("My email is alice@corp.com")
    surrogate = sent_a.split()[-1]
    assert "alice@corp.com" not in sent_a
    assert b.unmask(f"Reply to {surrogate}") == f"Reply to {surrogate}"
    assert a.unmask(f"Reply to {surrogate}") == "Reply to alice@corp.com"


def test_I9_module_functions_are_per_thread(stub):
    sent = ss.mask("write to alice@corp.com")
    surrogate = sent.split()[-1]
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("other", ss.unmask(surrogate)))
    t.start(); t.join()
    assert out["other"] == surrogate                  # another thread: its own session
    assert ss.unmask(surrogate) == "alice@corp.com"   # this thread: restored


def test_I9_warm_up_session_is_not_inherited_by_tasks(stub):
    # A mask() at start-up must not become one map shared by every request.
    sent = ss.mask("warm-up for alice@corp.com")
    surrogate = sent.split()[-1]

    async def request():
        return ss.unmask(surrogate), ss.current_session()

    async def server():
        return await asyncio.gather(request(), request())

    (r1, s1), (r2, s2) = asyncio.run(server())
    assert r1 == r2 == surrogate
    assert s1 is not s2 and ss.current_session() not in (s1, s2)


def test_I9_use_session_is_inherited_by_tasks_and_to_thread(stub):
    s = ss.Session()
    surrogate = s.mask("alice@corp.com")

    async def handler():
        loop = asyncio.get_running_loop()
        in_task = await asyncio.create_task(asyncio.to_thread(ss.unmask, surrogate))
        # run_in_executor does not copy the context: the worker has no bound
        # session, so it gets its own and restores nothing (isolated, not shared).
        in_executor = await loop.run_in_executor(None, ss.unmask, surrogate)
        return in_task, in_executor

    async def main():
        with ss.use_session(s):
            return await handler()

    assert asyncio.run(main()) == ("alice@corp.com", surrogate)


def test_I9_eight_threads_own_sessions_no_cross_talk(stub):
    """8 threads × 1000 mask/unmask round trips, each on its own session."""
    errors, maps = [], {}

    def worker(t):
        s = ss.Session()
        values = [f"user{t}x{k}@corp{t}.example" for k in range(25)]
        first = {}
        for i in range(1000):
            v = values[i % len(values)]
            sent = s.mask(f"contact {v} today")
            sur = sent[len("contact "):-len(" today")]
            if v in sent:
                errors.append((t, i, "leaked"))
            if first.setdefault(v, sur) != sur:
                errors.append((t, i, "surrogate changed"))
            if s.unmask(f"ok, {sur}.") != f"ok, {v}.":
                errors.append((t, i, "bad restore"))
        maps[t] = s.mappings

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(worker, range(8)))
    assert errors == []
    for t, m in maps.items():
        assert len(m) == 25
        assert all(f"corp{t}." in original for original in m.values())


def test_I9_one_session_shared_by_eight_threads(stub):
    """8 threads × 1000 ops on ONE session: one surrogate per value, no clashes."""
    s = ss.Session()
    values = [f"shared{k}@corp.example" for k in range(40)]
    seen, errors, lock = {}, [], threading.Lock()

    def worker(t):
        for i in range(1000):
            v = values[(t * 7 + i) % len(values)]
            sur = s.mask(v)
            with lock:
                if seen.setdefault(v, sur) != sur:
                    errors.append((v, sur))
            if s.unmask(sur) != v:
                errors.append((t, i, "bad restore"))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(worker, range(8)))
    assert errors == []
    m = s.mappings
    assert len(m) == len(values) and sorted(m.values()) == sorted(values)


# ── API shape (I10) ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, object(), 42, {"unexpected": 1}])
def test_I10_unmask_rejects_non_text(bad):
    with pytest.raises(TypeError):
        ss.unmask(bad)


def test_I10_unmask_sdk_shapes():
    class Block:
        def __init__(self, type_, text=None):
            self.type, self.text = type_, text

    class Anthropic:
        content = [Block("thinking"), Block("text", "a "), Block("tool_use"), Block("text", "b")]

    assert ss.unmask(Anthropic()) == "a b"
    assert ss.unmask({"content": [{"type": "text", "text": "dict form"}]}) == "dict form"
    assert ss.unmask({"choices": [{"message": {"content": None}}]}) == ""   # tool-call reply
    assert ss.unmask({"choices": [{"message": {"content": "oai"}}]}) == "oai"


def test_I10_mask_rejects_non_text(stub):
    with pytest.raises(TypeError):
        ss.mask(None)


def test_I10_scan_returns_detections_with_offsets(stub):
    text = "Alice Brown wrote; later Alice Brown and alice@corp.com wrote again"
    found = ss.scan(text)
    assert all(isinstance(d, ss.Detection) for d in found)
    assert [(d.text, d.start) for d in found] == [
        ("Alice Brown", 0), ("Alice Brown", 25), ("alice@corp.com", 41)]
    assert all(text[d.start:d.end] == d.text for d in found)
    assert ss.current_session().mappings == {}            # scan() masks nothing
    assert ss.scan(text, as_dict=True) == {"Alice Brown": "person", "alice@corp.com": "email"}


def test_I10_pii_off_is_reported_not_silent(stub):
    ss.config(pii_off=["email"])
    r = ss.mask_result("Alice Brown, alice@corp.com")
    assert "alice@corp.com" in r.text and "Alice Brown" not in r.text
    assert [d.text for d in r.unmasked] == ["alice@corp.com"]
    assert [d.masked for d in ss.scan("alice@corp.com")] == [False]


def test_I10_silent_by_default(stub, capsys):
    s = ss.Session()
    s.unmask(s.mask("Alice Brown"))
    s.scan("Alice Brown")
    assert capsys.readouterr() == ("", "")
    assert ss.Config().detailed_view is False
    assert not hasattr(ss.Config(), "verify_addresses")   # F3: removed


def test_I10_config_changes_only_what_is_passed():
    ss.config(fuzzy_threshold=70)
    ss.config(address_mode="replace")
    assert ss.config().fuzzy_threshold == 70
    with pytest.raises(ValueError):
        ss.config(address_mode="bogus")
    assert ss.config().address_mode == "replace"


def test_I10_config_reaches_current_session_and_new_ones(stub):
    s = ss.current_session()
    ss.config(pii_off=["email"])
    assert s.config.pii_off == ["email"] and ss.Session().config.pii_off == ["email"]
    explicit = ss.Session(config=ss.Config())
    assert explicit.config.pii_off == []


def test_I10_pii_mem_cannot_change_under_live_mappings(stub, tmp_path):
    ss.mask("alice@corp.com")
    with pytest.raises(RuntimeError):
        ss.config(pii_mem=str(tmp_path))
    ss.flush()
    ss.config(pii_mem=str(tmp_path))
    assert ss.current_session()._shadow.persistent


def test_I9_persistent_session_resumes_by_id(stub, tmp_path):
    a = ss.Session("conv-42", storage_dir=str(tmp_path))
    surrogate = a.mask("alice@corp.com")
    b = ss.Session("conv-42", storage_dir=str(tmp_path))
    assert b.unmask(surrogate) == "alice@corp.com"
    assert b.mask("alice@corp.com") == surrogate          # same value, same surrogate
    b.close()
    assert ss.Session("conv-42", storage_dir=str(tmp_path)).mappings == {}


def test_I9_forget_and_closed_session(stub):
    s = ss.Session()
    sur = s.mask("alice@corp.com")
    assert s.forget("alice@corp.com") == 1
    assert s.unmask(sur) == sur
    s.close()
    with pytest.raises(RuntimeError):
        s.mask("x")
    with pytest.raises(ValueError):
        ss.Session("../evil")


# ── real models ──────────────────────────────────────────────────────────────

@pytest.mark.heavy
def test_J7_cross_user_with_real_models():
    a, b = ss.Session(), ss.Session()
    sent = a.mask("Hi, I'm Priya Raman, email priya.raman@gmail.com")
    assert "Priya Raman" not in sent and "priya.raman@gmail.com" not in sent
    assert b.unmask(sent) == sent
    assert a.unmask(sent) == "Hi, I'm Priya Raman, email priya.raman@gmail.com"
    found = a.scan("Paris Hilton met Paris Hilton")
    assert all("Paris Hilton met Paris Hilton"[d.start:d.end] == d.text for d in found)
