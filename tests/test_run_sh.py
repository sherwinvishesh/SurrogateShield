"""run.sh launcher (audit I24): no shell-side .env parsing, no Anthropic-only
key gate, never dumps the environment, warns on a world-readable .env."""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STUB_PY = """#!/bin/sh
# stand-in interpreter: answers run.sh's two pre-flight checks, echoes the launch
case "$2" in
  *config.SPACY_MODEL*) echo en_core_web_lg; exit 0 ;;
  *is_package*) exit ${SPACY_MISSING:-0} ;;
esac
echo "LAUNCH $*"
"""


def _setup(tmp_path, env_text="# only a comment\n", env_mode=0o644):
    shutil.copy(ROOT / "run.sh", tmp_path / "run.sh")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    py = tmp_path / ".venv" / "bin" / "python"
    py.write_text(STUB_PY)
    py.chmod(0o755)
    env = tmp_path / ".env"
    env.write_text(env_text)
    env.chmod(env_mode)
    return tmp_path


def _run(d, *args, **extra_env):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["SS_CANARY_VAR"] = "canary-value"
    env.update(extra_env)
    return subprocess.run(["bash", str(d / "run.sh"), *args], cwd=d, env=env,
                          capture_output=True, text=True, timeout=30)


def test_I24_launches_without_anthropic_key(tmp_path):
    r = _run(_setup(tmp_path), "chat")
    assert r.returncode == 0, r.stderr
    assert "LAUNCH main.py chat" in r.stdout
    assert "ANTHROPIC_API_KEY is not set" not in r.stdout


def test_I24_comment_only_env_does_not_print_environment(tmp_path):
    r = _run(_setup(tmp_path))
    assert "canary-value" not in r.stdout + r.stderr


def test_I24_world_readable_env_warns(tmp_path):
    r = _run(_setup(tmp_path, env_mode=0o644))
    assert "chmod 600 .env" in r.stdout
    private = tmp_path / "private"
    private.mkdir()
    r = _run(_setup(private, env_mode=0o600))
    assert "chmod 600 .env" not in r.stdout


def test_I24_missing_spacy_model_stops(tmp_path):
    r = _run(_setup(tmp_path), SPACY_MISSING="1")
    assert r.returncode == 1
    assert "spacy download en_core_web_lg" in r.stdout
    assert "LAUNCH" not in r.stdout


def test_I24_script_never_parses_env():
    text = (ROOT / "run.sh").read_text()
    assert "export $(" not in text and "xargs" not in text
    assert "exec " in text
