"""Arm subprocesses made deterministic while rescoring the real-data dev
split (E5, Phase 5): reruns now give byte-identical spans. Model-free."""

import pytest


# Presidio breaks ties between same-span, same-score entities in set order, so
# a rerun under another string-hash seed relabelled one span (NRP / URL) and
# drew another Faker value for it. Every arm subprocess gets a fixed seed.
def test_arm_subprocesses_run_with_a_fixed_hash_seed(monkeypatch, tmp_path):
    import subprocess
    from bench.arms import base, run as arm_run
    from bench import perf_arms
    from bench.realdata import regex_ablation

    env = base.arm_env()
    assert env["PYTHONHASHSEED"] == "0" and env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"

    seen = []

    class Stop(Exception):
        pass

    def fake_run(cmd, **kw):
        seen.append(kw["env"])
        raise Stop

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(arm_run, "PRIVATE", tmp_path / "spans")
    for call in (lambda: arm_run.run_arm("ss", tmp_path / "in.jsonl", "x"),
                 lambda: regex_ablation.run_ablate(tmp_path / "in.jsonl", tmp_path / "out", 0)):
        with pytest.raises(Stop):
            call()

    def fake_popen(cmd, **kw):
        seen.append(kw["env"])
        raise Stop

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with pytest.raises(Stop):
        perf_arms.spawn(["true"])
    assert len(seen) == 3 and all(e["PYTHONHASHSEED"] == "0" for e in seen)


# LLM Guard placed its NER pipeline on Apple's GPU (mps), whose float results
# vary between runs: one dev message gained or lost six IP_ADDRESS spans.
def test_llm_guard_arm_pins_its_pipeline_to_the_cpu(monkeypatch):
    import sys
    import types
    from bench.arms import llm_guard as lg

    class Model:
        pipeline_kwargs = {"batch_size": 1, "device": "mps"}

    class Torch(types.ModuleType):
        @staticmethod
        def device(name):
            return f"device({name})"

    helpers = types.ModuleType("llm_guard.input_scanners.anonymize_helpers")
    helpers.DEBERTA_AI4PRIVACY_v2_CONF = {"DEFAULT_MODEL": Model}
    for name in ("llm_guard", "llm_guard.input_scanners"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "llm_guard.input_scanners.anonymize_helpers", helpers)
    monkeypatch.setitem(sys.modules, "torch", Torch("torch"))
    lg.pin_cpu()
    assert Model.pipeline_kwargs == {"batch_size": 1, "device": "device(cpu)"}

