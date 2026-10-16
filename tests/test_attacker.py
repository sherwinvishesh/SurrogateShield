"""Attacker experiment protocol (attacker.py) — audit A10, I30, J10.

No network: the provider client is a fake.
"""

import json
from types import SimpleNamespace

import pytest

import attacker as atk

Q = "Ann Lee (ann@corp.com, 480-555-0199) lives at 12 Elm St, Tempe. Ask Alexander Fleming."
KEY = {"name": "Ann Lee", "email": "ann@corp.com", "phone": "480-555-0199",
       "address": "12 Elm St, Tempe"}


# ── A10: neutral prompts ─────────────────────────────────────────────────────

def test_A10_neutral_prompt():
    for p in (atk.SS_PROMPT, atk.PRESIDIO_PROMPT):
        low = p.lower()
        for primed in ("no cryptographic", "no statistical", "correct and expected",
                       "unrecoverable", "impossible", "do not fabricate"):
            assert primed not in low, primed
        assert "{text}" in p
    assert "<PERSON>" in atk.PRESIDIO_PROMPT          # placeholder-aware
    assert "<PERSON>" not in atk.SS_PROMPT
    atk.SS_PROMPT.format(text="x")                     # braces escaped
    atk.PRESIDIO_PROMPT.format(text="x")


# ── I30: attacker must differ from responder ─────────────────────────────────

def test_I30_attacker_is_not_responder(monkeypatch):
    with pytest.raises(atk.AttackerConfigError):
        atk.resolve_model("claude-x", responder="claude-x")
    monkeypatch.setattr(atk, "ATTACKER_MODEL", None)
    with pytest.raises(atk.AttackerConfigError, match="No attacker model"):
        atk.resolve_model(None, responder="claude-x")
    assert atk.resolve_model("claude-big", responder="claude-small") == "claude-big"


# ── I30: per-value scoring, leak-through and non-gold separate ───────────────

def test_I30_match_levels():
    assert atk.match_level("phone", "(480) 555-0199", "480-555-0199") == "exact"
    assert atk.match_level("phone", "480-111-2222", "480-555-0199") == "partial"
    assert atk.match_level("email", "x@corp.com", "ann@corp.com") == "partial"
    assert atk.match_level("PERSON", "Ann Smith", "Ann Lee") == "partial"
    assert atk.match_level("PERSON", "Bob Smith", "Ann Lee") is None
    assert atk.match_level("dob", "1990-01-01", "March 3, 1990") == "partial"
    assert atk.match_level("ssn", "123-45-6789", "123456789") == "exact"
    assert atk.match_level("ssn", "123-45-0000", "123-45-6789") is None


def test_I30_score_arm_per_value_units():
    gold = atk.gold_targets(Q, KEY)
    assert len(gold) == 4
    sent = "Zoe Park (zoe@mail.org, 480-555-0199) lives at 13 Elm St, Tempe. Ask <PERSON>."
    parsed = {"estimates": [
        {"seen": "Zoe Park", "original_estimate": "Ann Lee"},
        {"seen": "zoe@mail.org", "original_estimate": None},
        {"seen": "13 Elm St, Tempe", "original_estimate": "12 Elm St, Tempe"},
        {"seen": "<PERSON>", "original_estimate": "Alexander Fleming"},
    ]}
    s = atk.score_arm(parsed, gold, sent, ["Ann Lee", "ann@corp.com", "Alexander Fleming"])
    out = {v["type"]: v["outcome"] for v in s["values"]}
    assert out == {"PERSON": "exact", "email": "not_recovered",
                   "phone": "leaked_verbatim", "address": "exact"}
    # the public figure is a non-gold redaction, scored on its own
    assert s["non_gold"] == [{"value": "Alexander Fleming", "recovered": True}]


def test_I30_analysis_same_units_and_unavailable_excluded():
    gold = atk.gold_targets(Q, KEY)
    ok = {"available": True, "usage": {"input_tokens": 10, "output_tokens": 5},
          "score": atk.score_arm({"estimates": [{"original_estimate": "Ann Lee"},
                                                {"original_estimate": "Ann Lee"},
                                                {"original_estimate": "Ann Lee"}]},
                                 gold, "nothing visible")}
    bad = {"available": False, "error": "json_parse_error", "usage": None}
    a = atk.compute_analysis([{"index": 0, "ss": ok, "presidio": bad},
                              {"index": 1, "ss": bad, "presidio": bad}])
    ss = a["ss"]
    assert ss["rows_available"] == 1 and ss["rows_unavailable"] == 1
    assert ss["gold_values"] == 4 and ss["inference_targets"] == 4
    assert ss["exact"] == 1                           # three identical guesses, one value
    assert ss["by_type"]["PERSON"]["exact_rate"] == 1.0
    assert all(bt["exact_rate"] <= 1.0 for bt in ss["by_type"].values())
    assert ss["exact_rate_ci95"][0] < 0.25 < ss["exact_rate_ci95"][1]
    assert a["presidio"]["rows_available"] == 0 and a["presidio"]["exact_rate"] is None
    assert ss["tokens"] == {"input_tokens": 10, "output_tokens": 5}


# ── I30: failures are unavailable; runner with a fake client ─────────────────

class _Resp(SimpleNamespace):
    pass


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        text, stop = r
        return _Resp(content=[SimpleNamespace(text=text)], stop_reason=stop,
                     usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def test_I30_call_attacker_failures_unavailable():
    good = FakeClient([('```json\n{"estimates": []}\n```', "end_turn")])
    assert atk.call_attacker(good, "m", "ss", "t")["available"] is True
    trunc = FakeClient([('{"estimates": [', "max_tokens")])
    r = atk.call_attacker(trunc, "m", "ss", "t")
    assert r["available"] is False and "truncated" in r["error"]
    junk = FakeClient([("I cannot help", "end_turn")])
    assert atk.call_attacker(junk, "m", "ss", "t")["error"] == "json_parse_error"
    down = FakeClient([OSError("connection reset")])
    r = atk.call_attacker(down, "m", "presidio", "t")
    assert r["available"] is False and "OSError" in r["error"]


def _files(tmp_path, n=4):
    answers, keys = [], []
    for i in range(n):
        answers.append({"question": Q, "sanitized_input": "Zoe Park lives somewhere.",
                        "presidio_sanitized_input": "<PERSON> lives at 12 Elm St, Tempe.",
                        "surrogate_map": {"Ann Lee": "Zoe Park"},
                        "presidio_found_piis": [{"value": "Ann Lee", "type": "PERSON"}]})
        keys.append({"Question": Q, "Answer-Key": KEY})
    answers.append({"question": Q, "error": "boom"})
    keys.append({"Question": Q, "Answer-Key": KEY})
    (tmp_path / "a.json").write_text(json.dumps(answers))
    (tmp_path / "k.json").write_text(json.dumps(keys))


def test_I30_runner_sample_budget_and_meta(tmp_path):
    _files(tmp_path)
    reply = ('{"estimates": [{"seen": "Zoe Park", "original_estimate": "Ann Lee"}]}', "end_turn")
    client = FakeClient([reply] * 4)
    with pytest.raises(ValueError, match="max_calls"):
        atk.run_experiment("a.json", "k.json", sample=2, seed=3, model="atk", max_calls=3,
                           client=client, experiment_dir=tmp_path)
    assert client.calls == []                              # refused before any call
    path = atk.run_experiment("a.json", "k.json", sample=2, seed=3, model="atk",
                              max_calls=4, client=client, experiment_dir=tmp_path)
    out = json.loads(path.read_text())
    assert out["meta"]["attacker_model"] == "atk" and out["meta"]["planned_calls"] == 4
    assert len(out["meta"]["indices"]) == 2 and 4 not in out["meta"]["indices"]   # error row
    assert all(c["max_tokens"] == 4096 for c in client.calls)
    an = out["analysis"]
    assert an["ss"]["exact"] == 2                          # the name, per row
    assert an["presidio"]["leaked_verbatim"] == 2          # address visible in Presidio text
    assert an["ss"]["tokens"]["input_tokens"] == 200
    # same settings resume (nothing left to do); different settings refuse
    atk.run_experiment("a.json", "k.json", sample=2, seed=3, model="atk",
                       client=FakeClient([]), experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="different settings"):
        atk.run_experiment("a.json", "k.json", sample=2, seed=4, model="atk",
                           client=FakeClient([]), experiment_dir=tmp_path)


def test_I30_plan_is_deterministic():
    answers = [{"sanitized_input": "x"} for _ in range(50)]
    a = atk.plan(answers, 10, seed=7)
    assert a == atk.plan(answers, 10, seed=7) and a[1] == 10
    assert a != atk.plan(answers, 10, seed=8)


# ── A11/I23: attacker screen prints computed values only ─────────────────────

def test_A11_attacker_screen_no_hardcoded_claims():
    src = open("main.py", encoding="utf-8").read()
    body = src[src.index("def _run_attacker_experiment"):src.index("# ─── Settings")]
    for phrase in ("Security Analysis", "Expected result", "proves", "BERTScore advantage",
                   "inference resistance confirmed", "unable to recover"):
        assert phrase not in body, phrase


def test_A11_render_attacker_from_dict():
    from rich.console import Console

    from eval_report import render_attacker

    gold = atk.gold_targets(Q, KEY)
    ok = {"available": True, "usage": None,
          "score": atk.score_arm({"estimates": []}, gold, "480-555-0199")}
    an = atk.compute_analysis([{"index": 0, "ss": ok,
                                "presidio": {"available": False, "error": "json_parse_error"}}])
    con = Console(record=True, width=160)
    render_attacker(an, {"attacker_model": "atk", "responder_model": "resp", "seed": 0}, con)
    out = con.export_text()
    assert "atk" in out and "resp" in out
    assert "json_parse_error 1" in out
    assert "n/a" in out                                   # Presidio arm has no rates


def test_I30_old_format_results_file_refused(tmp_path):
    _files(tmp_path)
    (tmp_path / "a_Attacker_Experiment.json").write_text("[]")
    with pytest.raises(ValueError, match="pre-v2"):
        atk.run_experiment("a.json", "k.json", model="atk", client=FakeClient([]),
                           experiment_dir=tmp_path)
