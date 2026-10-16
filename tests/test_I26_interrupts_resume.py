"""Audit I26 — interrupts, multi-line input and resume. Model-free, no network."""

import json
import os
import stat

import pytest

import attacker as atk
import main
from tests.test_attacker import FakeClient, Q, _files
from tests.test_runner import FIELDS, FakeChat, exp  # noqa: F401  (fixture)
import json_tester as jt


def lines(*items):
    it = iter(items)
    return lambda prompt: next(it)


# ── multi-line input ─────────────────────────────────────────────────────────

def test_I26_single_line():
    assert main.read_message(lines("  hello  ")) == "hello"


def test_I26_fenced_block_is_one_message():
    msg = main.read_message(lines('"""', "Dear team,", "", "Ann Lee is out.", '"""'))
    assert msg == "Dear team,\n\nAnn Lee is out."
    assert main.read_message(lines('"""first', "second\"\"\"")) == "first\nsecond"
    assert main.read_message(lines('"""one line"""')) == "one line"


def test_I26_backslash_continuation():
    assert main.read_message(lines("a \\", "b \\", "c")) == "a \nb \nc"


# ── batch runner ─────────────────────────────────────────────────────────────

class InterruptingChat(FakeChat):
    def _send_to_api(self, messages):
        if len(self.sent) == 2:
            raise KeyboardInterrupt
        return super()._send_to_api(messages)


def test_I26_batch_interrupt_keeps_finished_rows(exp):  # noqa: F811
    chat = InterruptingChat()
    with pytest.raises(KeyboardInterrupt):
        jt.run_batch("qs.json", FIELDS, chat=chat, experiment_dir=exp)
    (out,) = [p for p in exp.iterdir() if p.name.endswith("_answers.json")]
    rows = json.loads(out.read_text())
    assert len(rows) == 2 and stat.S_IMODE(os.stat(out).st_mode) == 0o600
    jt.run_batch("qs.json", FIELDS, chat=FakeChat(), experiment_dir=exp)   # resumes
    assert len(json.loads(out.read_text())) == 3


def test_I26_batch_corrupt_output_raises_not_restarts(exp):  # noqa: F811
    jt.run_batch("qs.json", FIELDS, chat=FakeChat(), experiment_dir=exp)
    (out,) = [p for p in exp.iterdir() if p.name.endswith("_answers.json")]
    out.write_text('[{"question": "Alice asked')                       # truncated
    with pytest.raises(ValueError):
        jt.run_batch("qs.json", FIELDS, chat=FakeChat(), experiment_dir=exp)
    assert out.read_text() == '[{"question": "Alice asked'            # not overwritten


# ── attacker runner ──────────────────────────────────────────────────────────

REPLY = ('{"estimates": [{"seen": "Zoe Park", "original_estimate": "Ann Lee"}]}', "end_turn")


def test_I26_attacker_reruns_provider_errors_only(tmp_path):
    _files(tmp_path, n=3)
    first = FakeClient([REPLY, REPLY, OSError("reset"), REPLY, ("not json", "end_turn"), REPLY])
    path = atk.run_experiment("a.json", "k.json", model="atk", client=first,
                              experiment_dir=tmp_path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    again = FakeClient([REPLY, REPLY])
    atk.run_experiment("a.json", "k.json", model="atk", client=again, experiment_dir=tmp_path)
    assert len(again.calls) == 2                       # only the row with OSError
    rows = json.loads(path.read_text())["rows"]
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert rows[1]["ss"]["available"] and rows[1]["presidio"]["available"]
    assert rows[2]["ss"]["error"] == "json_parse_error"   # a result, kept


def test_I26_attacker_interrupt_flushes(tmp_path):
    _files(tmp_path, n=3)
    class Interrupting(FakeClient):
        def create(self, **kw):
            if len(self.calls) == 2:
                raise KeyboardInterrupt
            return super().create(**kw)
    client = Interrupting([REPLY, REPLY])
    with pytest.raises(KeyboardInterrupt):
        atk.run_experiment("a.json", "k.json", model="atk", client=client,
                           experiment_dir=tmp_path)
    out = json.loads((tmp_path / "a_Attacker_Experiment.json").read_text())
    assert [r["index"] for r in out["rows"]] == [0]
