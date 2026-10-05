"""Batch runner (json_tester.py) — audit I22, I31, A8, J12.

Model-free: detection is replaced by a fake ``prepare_send`` that plans a
single PERSON substitution, and the provider by a recording fake chat.
"""

import json
import os
import stat

import pytest

import json_tester as jt
import util


class FakeChat:
    def __init__(self, fail_on=None):
        self.sent = []
        self.fail_on = fail_on

    def _send_to_api(self, messages):
        text = messages[0]["content"]
        self.sent.append(text)
        if self.fail_on and self.fail_on in text:
            raise ConnectionError("provider down")
        return f"Reply to: {text}"


def _fake_prepare(question, mimic):
    name = question.split()[0]
    ents = [util.DetectedEntity(text=name, start=0, end=len(name), type="PERSON")]
    surrogate = f"Zed{mimic._rng.randint(100, 999)}"
    mapping = {name: surrogate}
    edits = util.plan_substitutions(question, ents, mapping)
    return jt.Prepared(
        question=question, is_service_query=False, address_mode="replace",
        confirmed=ents, skipped=[], skip_reason="topical_geo_filtered",
        surrogate_map=mapping, edits=edits, sanitized=util.splice(question, edits),
        timings={"pattern_scan_ms": 1.5, "entity_trace_ms": 2.5},
    )


FIELDS = {
    "question": True, "sanitized_input": True, "surrogate_map": True,
    "llm_response": True, "final_output": True, "stage_timings_ms": True,
    "pii_spans": True,
}


@pytest.fixture
def exp(tmp_path, monkeypatch):
    monkeypatch.setattr(jt, "prepare_send", _fake_prepare)
    import settings_manager
    monkeypatch.setattr(settings_manager, "load_settings",
                        lambda: {"llm_provider": "fake", "presidio_comparison": False})
    qs = [{"input": "Alice asked about taxes."},
          {"input": "Bob asked about rent. Bob is late."},
          {"input": "Carol asked about loans."}]
    (tmp_path / "qs.json").write_text(json.dumps(qs))
    return tmp_path


def _run(exp, chat, **kw):
    path = jt.run_batch("qs.json", FIELDS, chat=chat, experiment_dir=exp, **kw)
    return json.loads(open(path).read())


def test_I22_runner_sends_sanitized_text(exp):
    chat = FakeChat()
    rows = _run(exp, chat)
    assert chat.sent == [r["sanitized_input"] for r in rows]
    for sent, name in zip(chat.sent, ("Alice", "Bob", "Carol")):
        assert name not in sent
    assert rows[1]["sanitized_input"].count("Zed") == 2      # both occurrences


def test_I22_send_mismatch_raises_before_provider_call():
    prep = _fake_prepare("Alice asked about taxes.", jt_mimic(1))
    chat = FakeChat()
    with pytest.raises(jt.SendMismatch):
        jt._send_checked(chat, prep.question, prep)           # un-sanitised text
    tampered = jt.Prepared(**{**prep.__dict__, "sanitized": "Alice " + prep.sanitized})
    with pytest.raises(jt.SendMismatch):
        jt._send_checked(chat, tampered.sanitized, tampered)
    assert chat.sent == []


def jt_mimic(seed):
    from generation.logic import MimicGen
    return MimicGen(seed=seed)


def test_I22_final_output_restores_originals(exp):
    rows = _run(exp, FakeChat())
    assert rows[0]["final_output"] == "Reply to: Alice asked about taxes."


def test_J12_timings_from_the_real_pass(exp):
    rows = _run(exp, FakeChat())
    t = rows[0]["stage_timings_ms"]
    assert t["pattern_scan_ms"] == 1.5 and t["entity_trace_ms"] == 2.5
    assert {"llm_call_ms", "total_ms"} <= set(t)
    assert t["total_ms"] >= t["llm_call_ms"] >= 0


def test_I31_seeded_and_deterministic(exp, tmp_path_factory):
    a = _run(exp, FakeChat())
    other = tmp_path_factory.mktemp("again")
    (other / "qs.json").write_text((exp / "qs.json").read_text())
    b = _run(other, FakeChat())
    assert [r["surrogate_map"] for r in a] == [r["surrogate_map"] for r in b]
    assert len({next(iter(r["surrogate_map"].values())) for r in a}) == 3   # seed + i


def test_I31_resume_reprocesses_error_rows(exp):
    chat = FakeChat(fail_on="rent")
    rows = _run(exp, chat)
    assert "error" in rows[1] and rows[1]["error_type"] == "ConnectionError"
    assert "error" not in rows[0] and "error" not in rows[2]

    chat2 = FakeChat()
    rows2 = _run(exp, chat2)
    assert len(chat2.sent) == 1                                # only the error row
    assert "error" not in rows2[1]
    assert rows2[0] == rows[0] and rows2[2] == rows[2]


def test_I31_atomic_private_output(exp):
    _run(exp, FakeChat())
    out = exp / "qs_answers.json"
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert not (exp / "qs_answers.json.tmp").exists()


def test_I31_stale_answers_file_raises(exp):
    (exp / "qs_answers.json").write_text(json.dumps([{"question": "something else"}]))
    with pytest.raises(ValueError):
        _run(exp, FakeChat())


def test_I31_meta_records_generator_commit_and_seed(exp):
    _run(exp, FakeChat(), seed=123)
    meta = json.loads((exp / "qs_answers.meta.json").read_text())
    assert meta["base_seed"] == 123
    assert set(meta["generator"]) == {"commit", "dirty"}
    assert meta["presidio"] is None                            # gated off by the setting


def test_I31_presidio_setting_gates_every_presidio_field(exp):
    fields = {**FIELDS, "presidio_sanitized_input": True, "presidio_found_piis": True,
              "bertscore_presidio": True}
    path = jt.run_batch("qs.json", fields, chat=FakeChat(), experiment_dir=exp)
    rows = json.loads(open(path).read())
    assert not any(k.startswith("presidio") or k.startswith("bertscore") for k in rows[0])


# ── A8: BERTScore rescaled, errors stored ────────────────────────────────────

def test_A8_bertscore_rescaled_with_fake_scorer():
    answers = [{"sanitized_input": "Zed asked", "final_output": "a", "clean_llm_response": "b"},
               {"error": "x"}]
    questions = [{"input": "Ann asked"}, {"input": "Bob asked"}]
    calls = []

    def scorer(c, r):
        calls.append((c, r))
        return [0.5] * len(c), [0.6] * len(c), [0.55] * len(c)

    cfg = jt._run_bertscore_batch(answers, questions, True, False, None, 2, scorer=scorer)
    assert cfg["rescale_with_baseline"] is True and cfg["available"] is True
    assert answers[0]["bertscore_ss"] == {"precision": 0.5, "recall": 0.6, "f1": 0.55, "rescaled": True}
    assert answers[0]["bertscore_ss_output"]["f1"] == 0.55
    assert answers[1]["bertscore_ss"] is None                   # error rows not scored
    assert calls[0] == (["Zed asked"], ["Ann asked"])


def test_A8_bertscore_errors_stored_not_swallowed():
    answers = [{"sanitized_input": "Zed asked"}]

    def scorer(c, r):
        raise RuntimeError("cuda oom")

    jt._run_bertscore_batch(answers, [{"input": "Ann asked"}], True, False, None, 1, scorer=scorer)
    assert answers[0]["bertscore_ss"] is None
    assert "cuda oom" in answers[0]["bertscore_error"]
