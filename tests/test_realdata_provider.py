"""Model-free tests for bench/realdata/provider.py: the call ledger and its cap,
and the resumable Message Batches runner (fake client, no network)."""

import json
from types import SimpleNamespace

import pytest

from bench.realdata import provider as pv


def test_ledger_counts_across_processes_and_refuses_the_call_that_crosses_the_cap(tmp_path):
    a = pv.Ledger(tmp_path, cap=10, run="a")
    a.take(6, "2-labels", "label", "m")
    b = pv.Ledger(tmp_path, cap=10, run="b")
    b.path = tmp_path / "other.jsonl"                     # a second process writes its own file
    b.take(4, "3-inject", "place", "m")
    assert a.total() == b.total() == 10
    before = a.rows()
    with pytest.raises(pv.BudgetExceeded):
        a.take(1, "6-live", "responder", "m")
    assert a.rows() == before                             # nothing reserved, nothing sent
    with pytest.raises(ValueError):
        a.take(0, "x", "y", "m")


def test_ledger_usage_rows_cost_no_calls_and_sum_by_phase(tmp_path):
    led = pv.Ledger(tmp_path, cap=5, run="u")
    led.take(2, "2-labels", "label", "m")
    led.usage("2-labels", "label", "m", {"input_tokens": 100, "output_tokens": 7, "cache_read_input_tokens": None})
    led.usage("2-labels", "label", "m", {"input_tokens": 1, "output_tokens": 3})
    assert led.total() == 2
    s = led.summary()["2-labels"]
    assert s["calls"] == 2 and s["input_tokens"] == 101 and s["output_tokens"] == 10
    assert oct(led.path.stat().st_mode)[-3:] == "600"
    assert all("text" not in json.loads(l) for l in led.path.read_text().splitlines())


class _Msg:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return self.d


class FakeClient:
    def __init__(self, polls_until_end=2):
        self.created, self.polls, self.until = [], 0, polls_until_end
        self.messages = SimpleNamespace(batches=self)

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch-{len(self.created)}")

    def retrieve(self, bid):
        self.polls += 1
        status = "ended" if self.polls >= self.until else "in_progress"
        return SimpleNamespace(processing_status=status,
                               request_counts=SimpleNamespace(processing=1, succeeded=0, errored=0))

    def results(self, bid):
        for r in self.created[-1]:
            if r["custom_id"] == "bad":
                yield SimpleNamespace(custom_id="bad", result=SimpleNamespace(type="errored", error=None))
            else:
                yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                    type="succeeded", message=_Msg({"content": [], "usage": {"input_tokens": 10, "output_tokens": 2}})))


def _reqs(*ids):
    return pv.requests_of((i, {"model": "m", "max_tokens": 5, "messages": []}) for i in ids)


def test_run_batch_counts_polls_collects_and_resumes_offline(tmp_path):
    led = pv.Ledger(tmp_path / "ledger", cap=10, run="t")
    cl = FakeClient()
    out = pv.run_batch(cl, led, "b", "2-labels", "label", _reqs("x", "y", "bad"), poll_s=0,
                       log=lambda s: None, directory=tmp_path / "batches", sleep=lambda s: None)
    assert out["x"]["status"] == "succeeded" and out["bad"]["status"] == "errored"
    assert led.total() == 3 and cl.polls == 2
    assert led.summary()["2-labels"]["input_tokens"] == 20
    # resume: saved results, no client, no new calls, no double-counted usage
    again = pv.run_batch(None, led, "b", "2-labels", "label", _reqs("x", "y", "bad"),
                         directory=tmp_path / "batches")
    assert again == out and led.total() == 3 and led.summary()["2-labels"]["input_tokens"] == 20
    with pytest.raises(ValueError):
        pv.run_batch(None, led, "b", "2-labels", "label", _reqs("x", "z"), directory=tmp_path / "batches")
    for f in (tmp_path / "batches").iterdir():
        assert oct(f.stat().st_mode)[-3:] == "600"


def test_run_batch_over_the_cap_sends_nothing(tmp_path):
    led = pv.Ledger(tmp_path / "ledger", cap=2, run="t")
    cl = FakeClient()
    with pytest.raises(pv.BudgetExceeded):
        pv.run_batch(cl, led, "b", "p", "k", _reqs("a", "b", "c"), directory=tmp_path / "batches")
    assert cl.created == [] and not (tmp_path / "batches" / "b.json").exists()


def test_run_batch_needs_a_client_for_unsent_or_unfinished_batches(tmp_path):
    led = pv.Ledger(tmp_path / "ledger", cap=5, run="t")
    with pytest.raises(RuntimeError):
        pv.run_batch(None, led, "b", "p", "k", _reqs("a"), directory=tmp_path / "batches")
    with pytest.raises(ValueError):
        pv.run_batch(FakeClient(), led, "b", "p", "k", _reqs("a", "a"), directory=tmp_path / "batches")


def test_tool_input_picks_the_named_tool_use_block():
    msg = {"content": [{"type": "text", "text": "x"}, {"type": "tool_use", "name": "other", "input": {"a": 1}},
                       {"type": "tool_use", "name": "record_labels", "input": {"messages": []}}]}
    assert pv.tool_input(msg, "record_labels") == {"messages": []}
    assert pv.tool_input({"content": []}, "record_labels") is None
