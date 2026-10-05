"""Gate J16 — docs: both READMEs carry a Limitations section, and every
committed result file under bench/results is listed with its command.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
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


def test_J16_quickstart_install_is_accurate():
    # The spaCy model is a separate download, and PyPI has 1.0.0, not this branch.
    text = (ROOT / "python-library" / "README.md").read_text(encoding="utf-8")
    assert "installs automatically" not in text
    assert "python -m spacy download en_core_web_lg" in text
    assert "pip install ./python-library" in text


CLAIM_SURFACES = ["README.md", "python-library/README.md", "website/index.html",
                  "website/docs.html", "main.py", "help_screen.py"]
UNSUPPORTED = [
    # A1–A4: the detector misses some values, so nothing can promise "never".
    r"PII never leaves", r"data never leaves", r"never crosses the API boundary",
    r"real values never (leave|transmitted)", r"No PII crosses",
    # A5, A9: a prediction stated as a proof.
    r"never enters the vector store", r"This proves",
    # A6, A7: BERTScore ranges that were never measured on answers.
    r"~ ?95", r"~ ?84", r"92–97", r"80–88",
    # A8: the withdrawn attacker run (it may only be named where it is withdrawn).
    r'stat-value">0\.00%', r"recovers <em>nothing", r"Table 7",
    # A10: legacy ablation figures.
    r"representative ranges", r">0\.62<",
]


@pytest.mark.parametrize("surface", CLAIM_SURFACES)
def test_J16_no_unsupported_claims(surface):
    text = (ROOT / surface).read_text(encoding="utf-8")
    found = [p for p in UNSUPPORTED if re.search(p, text)]
    assert not found, f"{surface}: {found}"


def test_J16_withdrawn_attacker_figures_only_in_withdrawal():
    text = (ROOT / "website" / "index.html").read_text(encoding="utf-8")
    for figure in ("0.00%", "1.53%"):
        for m in re.finditer(re.escape(figure), text):
            window = text[max(0, m.start() - 300): m.end() + 300]
            assert "withdrawn" in window, figure


def test_J16_published_numbers_match_results():
    dev4 = json.loads((RESULTS / "compare_final_dev4.json").read_text(encoding="utf-8"))["realworld"]
    ss, prs = dev4["ss"], dev4["presidio_result"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    site = (ROOT / "website" / "index.html").read_text(encoding="utf-8")
    for pct in (ss["leak_rate"], prs["leak_rate"], ss["spurious_rate"], prs["spurious_rate"]):
        assert f"{pct * 100:.1f}" in readme, pct
    assert f"{ss['leaked']} of" in readme
    assert f"{ss['negatives_untouched']} / {ss['negatives']}" in site
    assert f"{prs['negatives_untouched']} / {prs['negatives']}" in site
    configs = json.loads((RESULTS / "ablation_synth_dev.json").read_text(encoding="utf-8"))["configurations"]
    for cfg in configs.values():
        f1 = str(Decimal(str(cfg["micro"]["f1"])).quantize(Decimal("0.001"), ROUND_HALF_UP))
        assert f1 in readme and f1 in site, f1
