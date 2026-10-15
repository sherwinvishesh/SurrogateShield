"""Gate J16 — docs: both READMEs and the website carry the one-line
Limitations, every committed result file under bench/results is listed with
its command, and the published test-3 numbers match the result files.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"

LIMITATIONS = (
    "it trailed on the 96 real values in natural text (29 leaked; GLiNER-PII 16–20, Presidio 27), "
    "on some single types (3 of 13 on test-3, 8 of 13 on Nemotron-PII, URL most of all) and on speed "
    "and memory (plain Presidio is about 5× faster and lighter); it came out ahead on every other test."
)
SITE = ["website/index.html", "website/docs.html"]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("readme", ["README.md", "python-library/README.md"])
def test_J16_limitations_section_is_one_line(readme):
    m = re.search(r"^## Limitations\n(.*?)(?=^## )", read(readme), re.S | re.M)
    assert m, f"{readme} has no Limitations section"
    body = m.group(1).strip()
    assert "\n" not in body, "Limitations is one line"
    assert LIMITATIONS in body
    assert "bench/results/README.md" in body


@pytest.mark.parametrize("page", SITE)
def test_J16_site_limitations_is_one_line(page):
    text = read(page)
    assert 'id="limitations"' in text
    assert LIMITATIONS in text
    assert "not perfect" not in text


def test_J16_limitations_line_matches_results():
    natural = load("realdata_test3.json")["results"]["all"]["natural"]
    leaked = {arm: natural[arm]["leak"]["k"] for arm in natural}
    assert natural["ss"]["leak"]["n"] == 96
    assert f"({leaked['ss']} leaked; GLiNER-PII {leaked['gliner_pii_tuned']}–{leaked['gliner_pii']}, " \
           f"Presidio {leaked['presidio_default']})" in LIMITATIONS
    behind = {}
    for name, results in (("test-3", load("realdata_test3.json")["results"]["all"]["injected"]),
                          ("Nemotron-PII", load("external_nemotron_pii.json")["results"]["all"])):
        types = results["ss"]["by_type"]
        behind[name] = sum(results["ss"]["by_type"][t]["leak_rate"]
                           > min(arm["by_type"][t]["leak_rate"] for arm in results.values())
                           for t in types)
        assert len(types) == 13
    assert f"{behind['test-3']} of 13 on test-3, {behind['Nemotron-PII']} of 13 on Nemotron-PII" in LIMITATIONS
    perf = {a["arm"]: a for a in load("perf_arms_v4.json")["arms"]}
    assert round(perf["ss"]["p50_ms"] / perf["presidio_default"]["p50_ms"]) == 5
    assert perf["ss"]["peak_rss_mb"] > perf["presidio_default"]["peak_rss_mb"]


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
    # The spaCy model is a separate download, and PyPI has 1.0.0, not this
    # branch: the root README never says "pip install surrogateshield". The
    # library README is what PyPI renders, so its first screen may, provided
    # it states which version it describes and gives the checkout install.
    for readme in ("README.md", "python-library/README.md"):
        text = read(readme)
        assert "installs automatically" not in text
        assert "python -m spacy download en_core_web_lg" in text
    assert "pip install surrogateshield" not in read("README.md")
    lib = read("python-library/README.md")
    version = re.search(r'^version = "([^"]+)"$', read("python-library/pyproject.toml"), re.M).group(1)
    first_screen = lib[:lib.index("## Why SurrogateShield exists")]
    assert f"This README describes {version}" in first_screen
    for phrase in ("pip install surrogateshield", "pip install ./python-library",
                   "python -m spacy download en_core_web_lg", "surrogateshield download",
                   "DetectorUnavailable"):
        assert phrase in first_screen, phrase


def test_J16_library_readme_stands_alone():
    # PyPI renders python-library/README.md by itself: a relative link or a
    # bare reference to a repo-only file leads nowhere there.
    lib = read("python-library/README.md")
    targets = re.findall(r"\]\(([^)]+)\)", lib)
    relative = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert not relative, relative
    repo = "https://github.com/sherwinvishesh/SurrogateShield/blob/main/"
    for name in ("CONFIGURATION.md", "CHANGELOG.md", "bench/results/README.md"):
        for m in re.finditer(re.escape(name), lib):
            assert lib[:m.start()].endswith(repo) or lib[m.end():].startswith("`](" + repo), \
                f"{name} without its URL"


CLAIM_SURFACES = ["README.md", "python-library/README.md", "website/index.html",
                  "website/docs.html", "website/docs.md", "main.py", "help_screen.py"]
UNSUPPORTED = [
    # A1–A4: nothing can promise "never".
    r"PII never leaves", r"data never leaves", r"never crosses the API boundary",
    r"real values never (leave|transmitted)", r"No PII crosses",
    # A5, A9: a prediction stated as a proof.
    r"never enters the vector store", r"This proves",
    # A6, A7: BERTScore ranges that were never measured on answers.
    r"~ ?95", r"~ ?84", r"92–97", r"80–88",
    # A8: the withdrawn attacker run.
    r'stat-value">0\.00%', r"recovers <em>nothing", r"Table 7",
    # A10: legacy ablation figures.
    r"representative ranges", r">0\.62<",
    # Earlier benchmark figures, replaced by test-3.
    r"21\.6 ?%", r"47\.4 ?%", r"0\.563", r"0\.384", r"17\.4 ?%", r"25\.5 ?%",
]


@pytest.mark.parametrize("surface", CLAIM_SURFACES)
def test_J16_no_unsupported_claims(surface):
    # The home page's headline is the author's tagline; the text under it is checked.
    text = re.sub(r'<h1 class="hero-headline">.*?</h1>', "", read(surface), flags=re.S)
    found = [p for p in UNSUPPORTED if re.search(p, text)]
    assert not found, f"{surface}: {found}"


@pytest.mark.parametrize("surface", ["README.md", *SITE])
def test_J16_published_numbers_match_results(surface):
    text = read(surface)
    injected = load("realdata_test3.json")["results"]["all"]["injected"]
    for arm in ("ss", "gliner_pii", "presidio_default"):
        pct = f"{injected[arm]['leak']['rate'] * 100:.1f}"
        assert re.search(re.escape(pct) + r" ?%", text), (arm, pct)
    diffs = load("utility_realdata_test3.json")["e5b"]["all"]["bertscore_differences"]
    for other in ("ss-presidio_faker", "ss-gliner_pii"):
        assert f"+{diffs[other]['diff']:.3f}" in text, other
    restored = load("multiturn_realdata_test3.json")["results"]["all"]["ss"]["restored"]
    assert re.search(rf"{restored['k']} ?(/|of) ?{restored['n']}", text)
    exact = load("attacker_exact_test3.json")["results"]["all"]["ss"]["exact"]
    assert re.search(rf"{exact['k']} ?(/|of) ?{exact['n']}", text)


def test_J16_security_design_gitignore_row_is_real(tmp_path):
    """The Security Design table says `.gitignore` covers `*.shadowmap`,
    `conversations/`, `device.key`, `.env` and new experiment outputs. Ask
    git, not the file's text."""
    import shutil
    import subprocess
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    row = next(l for l in read("README.md").splitlines() if l.startswith("| `.gitignore` |"))
    for claim in ("`*.shadowmap`", "`conversations/`", "`device.key`", "`.env`",
                  "`experiment/*_answers*.json`", "`*_eval_results*.json`", "`*_Attacker_Experiment*.json`"):
        assert claim in row, claim
    ignored = ["a.shadowmap", "store/session-1.shadowmap", "conversations/abc.json",
               "conversations/abc.shadowmap", ".surrogateshield/device.key", ".env",
               "experiment/run_answers.json", "experiment/run_eval_results.json",
               "experiment/run_Attacker_Experiment.json"]
    tracked = ["experiment/synth_dev_key.json", "python-library/README.md"]
    proc = subprocess.run([git, "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index",
                           "--", *ignored, *tracked],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    assert proc.stdout.split("\n")[:-1] == ignored, proc.stdout + proc.stderr

