"""Run the legacy check()-style scripts (tests/test1-7.py) and fail loudly.

The legacy scripts print "Results: X/Y passed" but always exit 0, so a plain
runner could never gate on them. This wrapper executes each script in a
subprocess and asserts every check passed. They load spaCy/HF models, so the
whole module is marked heavy:  python -m pytest tests/ -m heavy
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY = [f"test{i}.py" for i in range(1, 8)]

_RESULT_RE = re.compile(r"Results?:\s*(\d+)\s*/\s*(\d+)")


@pytest.mark.parametrize("script", LEGACY)
def test_legacy_script(script):
    proc = subprocess.run(
        [sys.executable, str(Path("tests") / script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{script} crashed:\n{output[-3000:]}"

    matches = _RESULT_RE.findall(output)
    assert matches, f"{script} produced no 'Results: X/Y passed' line:\n{output[-3000:]}"
    for passed, total in matches:
        assert passed == total, (
            f"{script}: only {passed}/{total} checks passed\n{output[-3000:]}"
        )
