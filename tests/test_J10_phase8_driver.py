"""J10 / A8 / A10 — the Phase 8 driver: seeded sample of PII-bearing rows,
a hard cap on every provider attempt, no call in a dry run. Model-free."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import phase8  # noqa: E402


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    # The driver reads .env for a live run; tests must not load any key.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


def test_J10_sample_is_seeded_and_has_gold_pii():
    a, b = phase8.sample_rows(50, 0), phase8.sample_rows(50, 0)
    assert a == b and len(set(a)) == 50 and a == sorted(a)
    assert phase8.sample_rows(50, 1) != a
    key = json.loads(phase8.SYNTH_KEY.read_text(encoding="utf-8"))
    assert all(key[i]["Answer-Key"] for i in a)


def test_J10_counter_refuses_the_call_that_crosses_the_cap():
    c = phase8.CallCounter(3)
    sent = []
    adapter = phase8.CountedAdapter(lambda p, s: sent.append(p) or "ok", c)
    for _ in range(3):
        adapter([{"role": "user", "content": "x"}], "")
    with pytest.raises(phase8.BudgetExceeded):
        adapter([{"role": "user", "content": "x"}], "")
    assert len(sent) == 3 and c.calls == {"responder": 3, "attacker": 0}


def test_J10_attacker_client_is_counted_too():
    class Fake:
        class messages:
            @staticmethod
            def create(**kw):
                return kw
    c = phase8.CallCounter(1)
    client = phase8.CountedAnthropic(Fake(), c)
    assert client.messages.create(model="m") == {"model": "m"}
    with pytest.raises(phase8.BudgetExceeded):
        client.messages.create(model="m")


def test_J10_dry_run_and_over_cap_send_nothing(monkeypatch, capsys):
    monkeypatch.setattr(phase8, "write_sample", lambda *_: pytest.fail("wrote a sample"))
    assert phase8.main(["--sample", "50", "--attacker", "attacker-x", "--dry-run"]) == 0
    assert phase8.main(["--sample", "50", "--attacker", "attacker-x", "--max-calls", "200"]) == 2
    assert "nothing sent" in capsys.readouterr().err


def test_J10_attacker_must_differ_from_responder():
    import config
    from attacker import AttackerConfigError
    with pytest.raises(AttackerConfigError):
        phase8.main(["--attacker", config.CLAUDE_MODEL, "--dry-run"])


def test_J10_phase8_outputs_are_ignored():
    assert "experiment/phase8/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
