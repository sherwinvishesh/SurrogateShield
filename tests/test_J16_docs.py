"""Gate J16 — docs: both READMEs carry a Limitations section, and every
committed result file under bench/results is listed with its command.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


@pytest.mark.parametrize("readme", ["README.md", "python-library/README.md"])
def test_J16_limitations_section_present(readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    m = re.search(r"^## Limitations\n(.*?)(?=^## )", text, re.S | re.M)
    assert m, f"{readme} has no Limitations section"
    body = m.group(1)
    for topic in ("unseen", "English", "Special-category", "public", "Service queries"):
        assert topic in body, topic
    assert "bench/results/README.md" in body


def test_J16_every_result_file_has_its_command():
    index = (RESULTS / "README.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `([^`]+\.json)` \| `([^`]+)` \|", index, re.M))
    files = sorted(p.name for p in RESULTS.glob("*.json"))
    assert files and set(files) == set(rows), set(files) ^ set(rows)
    for name, command in rows.items():
        assert name in command, name
        stored = json.loads((RESULTS / name).read_text(encoding="utf-8")).get("command")
        if stored is not None:
            assert stored.split()[-1].endswith(name)
