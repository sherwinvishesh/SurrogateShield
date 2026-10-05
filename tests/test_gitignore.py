"""Run outputs that hold original PII are never picked up by `git add` (audit I25)."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    return r.returncode == 0


@pytest.mark.parametrize("path", [
    "experiment/mydata_answers.json",
    "experiment/mydata_answers.meta.json",
    "experiment/mydata_answers_Attacker_Experiment.json",
    "experiment/mydata_eval_results.json",
    "experiment/mydata_answers.json.tmp",
    "conversations/abc.json",
    "conversations/abc.shadowmap",
    ".env",
])
def test_I25_pii_outputs_ignored(path):
    assert _ignored(path)


@pytest.mark.parametrize("path", [
    "experiment/mydata.json",               # inputs and keys are authored data
    "experiment/mydata_key.json",
    "bench/realworld/dev.jsonl",
    "bench/realworld.py",
])
def test_I25_inputs_not_ignored(path):
    assert not _ignored(path)
