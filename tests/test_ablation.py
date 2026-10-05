"""Ablation re-runs the cascade per configuration (audit A9).

Heavy: loads spaCy (and ContextGuard for the full configuration).
"""

import json

import pytest

pytestmark = pytest.mark.heavy


def test_A9_stage_switches_change_detection():
    from detection.logic import run_cascade

    text = "Please ask Margaret Okonkwo to email me at m.ok@example.com."
    full = {e.text for e in run_cascade(text)[0]}
    pattern_only = {e.text for e in run_cascade(text, use_entity_trace=False,
                                                use_context_guard=False)[0]}
    assert "m.ok@example.com" in full and "m.ok@example.com" in pattern_only
    assert "Margaret Okonkwo" in full
    assert "Margaret Okonkwo" not in pattern_only


def test_A9_run_ablation_signed_deltas(tmp_path):
    from offline_eval import ABLATION_CONFIGS, run_ablation

    key = [
        {"Question": "Margaret Okonkwo can be reached at 480-555-0199.",
         "Answer-Key": {"name": "Margaret Okonkwo", "phone": "480-555-0199"}},
        {"Question": "How do volcanoes form?", "Answer-Key": {}},
    ]
    p = tmp_path / "k.json"
    p.write_text(json.dumps(key))
    out = run_ablation(p, limit=None, seed=0)
    cfg = out["configurations"]
    assert set(cfg) == {name for name, _label, _opts in ABLATION_CONFIGS}
    assert "delta_micro_f1_vs_full" not in cfg["full"]
    assert cfg["pattern_only"]["delta_micro_f1_vs_full"] < 0     # name missed
    assert cfg["pattern_only"]["micro"]["recall"] == 0.5
