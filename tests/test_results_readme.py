"""The benchmark numbers the README quotes are arithmetic on the result files
in ``bench/results/``. Every quoted number is recomputed here from the JSON
it comes from, so a result file cannot be regenerated, or a sentence edited,
without the README following. Rates are compared at one unit in the last
printed digit (a stored rate is rounded to four places before it is printed
to one, e.g. 2820/4627 = 60.947 % prints as 61.0 %). Read-only and fast:
nothing here runs detection."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")
# minus signs and thin spaces as the README writes them
NORM = README.replace("−", "-").replace(" ", " ").replace(" ", " ")

ARMS = {"SurrogateShield": "ss", "GLiNER-PII (threshold 0.3)": "gliner_pii_tuned",
        "GLiNER-PII (threshold 0.5)": "gliner_pii", "Presidio + RoBERTa": "presidio_transformers",
        "LLM Guard": "llm_guard", "Presidio + Faker": "presidio_faker", "Presidio": "presidio_default",
        "Presidio (placeholders)": "presidio_default", "GLiNER-PII": "gliner_pii"}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def t3():
    return load("realdata_test3.json")


@pytest.fixture(scope="module")
def ext():
    return load("external_nemotron_pii.json")


def num(s: str) -> float:
    return float(s.replace(",", "").replace("−", "-").replace("+", ""))


def pct(k: int, n: int) -> float:
    return 100.0 * k / n


def assert_pct(shown: float, k: int, n: int, what: str) -> None:
    assert abs(shown - pct(k, n)) <= 0.1 + 1e-9, f"{what}: README {shown} %, file {k}/{n} = {pct(k, n):.2f} %"


def table(header_start: str) -> dict:
    """Rows of the README table whose header line starts with *header_start*,
    keyed by the first cell (bold stripped)."""
    lines = NORM.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith(header_start)]
    assert len(starts) == 1, f"{header_start!r}: {len(starts)} tables"
    rows = {}
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def found(pattern: str) -> re.Match:
    m = re.search(pattern, NORM)
    assert m, f"README sentence not found: {pattern!r}"
    return m


# ── values let through (realdata_test3.json) ─────────────────────────────────

def test_leak_table(t3):
    R = t3["results"]["all"]
    rows = table("| arm | values leaked | prompts sent clean |")
    assert set(rows) == {"SurrogateShield", "GLiNER-PII (threshold 0.3)", "GLiNER-PII (threshold 0.5)",
                         "Presidio / Presidio + Faker", "Presidio + RoBERTa", "LLM Guard"}
    for label, (leak, clean, odd, edited) in rows.items():
        arms = ["presidio_default", "presidio_faker"] if label.startswith("Presidio /") else [ARMS[label]]
        for arm in arms:
            inj, shift = R["injected"][arm], R["shift"][arm]
            assert_pct(num(leak.split(" %")[0]), inj["leak"]["k"], inj["leak"]["n"], f"{arm} leak")
            assert_pct(num(clean.split(" %")[0]), inj["message_leak"]["n"] - inj["message_leak"]["k"],
                       inj["message_leak"]["n"], f"{arm} clean prompts")
            assert_pct(num(odd.split(" %")[0]), shift["leak"]["k"], shift["leak"]["n"], f"{arm} odd-format leak")
            assert_pct(num(edited.split(" %")[0]), inj["spurious"]["k"], inj["spurious"]["n"], f"{arm} harmless edited")
            for cell, stat in ((leak, inj["leak"]), (odd, shift["leak"])):
                m = re.search(r"\((\d[\d,]*) of (\d[\d,]*)\)", cell)
                if m:
                    assert (int(num(m[1])), int(num(m[2]))) == (stat["k"], stat["n"]), f"{arm}: {cell}"
    # the row covers both Presidio arms: they must agree to the printed digit
    for sub in ("injected", "shift"):
        assert R[sub]["presidio_default"]["leak"] == R[sub]["presidio_faker"]["leak"]


def test_leak_headline_and_ratios(t3):
    R = t3["results"]["all"]["injected"]
    m = found(r"it let ([\d.]+)% of ([\d,]+) personal values")
    assert_pct(num(m[1]), R["ss"]["leak"]["k"], R["ss"]["leak"]["n"], "headline leak")
    assert int(num(m[2])) == R["ss"]["leak"]["n"]
    m = found(r"leaked (\d+)× fewer values than GLiNER-PII and (\d+)× fewer\s+than Presidio")
    assert int(m[1]) == round(R["gliner_pii"]["leak"]["k"] / R["ss"]["leak"]["k"])
    assert int(m[2]) == round(R["presidio_default"]["leak"]["k"] / R["ss"]["leak"]["k"])
    m = found(r"the gap to GLiNER-PII at 0\.5 is (-?[\d.]+) points \(95 % CI\s+(-?[\d.]+) to (-?[\d.]+)\)")
    d = t3["differences"]["all"]["injected"]["gliner_pii"]["leak_rate"]
    for shown, exact in zip(m.groups(), (d["diff"], d["ci95"][0], d["ci95"][1])):
        assert abs(num(shown) - 100 * exact) <= 0.1 + 1e-9, (shown, exact)


def test_natural_negatives_untouched(t3):
    N = t3["results"]["all"]["natural"]
    m = found(r"Of ([\d,]+) real messages with nothing to protect, it left ([\d,]+) \(([\d.]+) %\)\s+untouched, "
              r"against ([\d,]+) for LLM Guard, ([\d,]+) for Presidio \+ RoBERTa,\s+([\d,]+) for Presidio "
              r"and ([\d,]+) for GLiNER-PII at 0\.5")
    ss = N["ss"]["negatives_untouched"]
    assert (int(num(m[1])), int(num(m[2]))) == (ss["n"], ss["k"])
    assert_pct(num(m[3]), ss["k"], ss["n"], "negatives untouched")
    for shown, arm in zip(m.groups()[3:], ("llm_guard", "presidio_transformers", "presidio_default", "gliner_pii")):
        assert int(num(shown)) == N[arm]["negatives_untouched"]["k"], arm
        assert N[arm]["negatives_untouched"]["n"] == ss["n"]


def test_limitations_line(t3, ext):
    m = found(r"trailed on the (\d+) real values in natural text \((\d+) leaked; GLiNER-PII (\d+)–(\d+), "
              r"Presidio (\d+)\), on some single types \((\d+) of (\d+) on test-3, (\d+) of (\d+) on Nemotron-PII, "
              r"(\w+) most of all\) and on speed and memory \(plain Presidio is about (\d+)× faster")
    N = t3["results"]["all"]["natural"]
    assert int(m[1]) == N["ss"]["leak"]["n"] and int(m[2]) == N["ss"]["leak"]["k"]
    assert (int(m[3]), int(m[4])) == (N["gliner_pii_tuned"]["leak"]["k"], N["gliner_pii"]["leak"]["k"])
    assert int(m[5]) == N["presidio_default"]["leak"]["k"] == N["presidio_faker"]["leak"]["k"]

    def trailed(results):     # types on which a baseline leaked a lower share than SS
        ss = results["ss"]["by_type"]
        best = {t: min(results[a]["by_type"][t]["leak_rate"] for a in results if a != "ss" and t in results[a]["by_type"])
                for t in ss}
        return {t for t in ss if ss[t]["leak_rate"] > best[t]}, len(ss)
    worse3, n3 = trailed(t3["results"]["all"]["injected"])
    worse_ext, n_ext = trailed(ext["results"]["all"])
    assert (int(m[6]), int(m[7])) == (len(worse3), n3), sorted(worse3)
    assert (int(m[8]), int(m[9])) == (len(worse_ext), n_ext), sorted(worse_ext)
    ext_ss = ext["results"]["all"]["ss"]["by_type"]
    assert m[10] == max(worse_ext, key=lambda t: ext_ss[t]["leak_rate"])
    perf = {a["arm"]: a for a in load("perf_arms_v4.json")["arms"]}
    assert int(m[11]) == round(perf["ss"]["p50_ms"] / perf["presidio_default"]["p50_ms"])


# ── the outside dataset (external_nemotron_pii.json) ────────────────────────

def test_external_sentence(ext):
    E = ext["results"]["all"]
    m = found(r"Nemotron-PII, ([\d,]+) records, ([\d,]+) values")
    assert int(num(m[1])) == ext["corpus"]["records"]
    assert int(num(m[2])) == E["ss"]["leak"]["n"] == E["ss"]["protect_values"]
    m = found(r"Leaked ([\d.]+) % with F1 ([\d.]+), against GLiNER-PII ([\d.]+)–([\d.]+) % \(F1\s+([\d.]+)–([\d.]+)\), "
              r"Presidio ([\d.]+) % \(([\d.]+)\), Presidio \+ RoBERTa ([\d.]+) % \(([\d.]+)\)\s+and LLM Guard "
              r"([\d.]+) % \(([\d.]+)\)")
    pairs = [("ss", m[1], m[2]), ("gliner_pii_tuned", m[3], m[5]), ("gliner_pii", m[4], m[6]),
             ("presidio_default", m[7], m[8]), ("presidio_transformers", m[9], m[10]), ("llm_guard", m[11], m[12])]
    for arm, leak, f1 in pairs:
        assert_pct(num(leak), E[arm]["leak"]["k"], E[arm]["leak"]["n"], f"external {arm}")
        assert abs(num(f1) - E[arm]["span_f1"]) <= 0.0005 + 1e-9, (arm, f1, E[arm]["span_f1"])


# ── answer quality (utility_realdata_test3.json) ────────────────────────────

def test_utility_numbers():
    e = load("utility_realdata_test3.json")["e5b"]["all"]
    rows = table("| arm | utility: answer vs answer to the original")
    assert set(rows) == {"SurrogateShield", "GLiNER-PII", "Presidio + Faker", "Presidio (placeholders)"}
    for label, (score,) in rows.items():
        assert abs(num(score) - e["bertscore"][ARMS[label]]["mean"]) <= 0.0005 + 1e-9, label
    for m in (found(r"Measured on (\d+) real chat prompts"), found(r"Answer quality \((\d+) prompts")):
        assert int(m[1]) == e["bertscore"]["ss"]["n"]
    m = found(r"Paired difference in utility: (\+?[\d.]+) \(95 % CI (\+?[\d.]+) to (\+?[\d.]+)\) against Presidio "
              r"\+ Faker and (\+?[\d.]+) \((\+?[\d.]+) to (\+?[\d.]+)\) against GLiNER-PII\. A pairwise judge "
              r"preferred SurrogateShield's answer by (\+?[\d.]+) against Presidio \+ Faker and (\+?[\d.]+) "
              r"against GLiNER-PII \((\d+) wins to (\d+)\)")
    for shown, (key, field) in zip(m.groups()[:6], [("ss-presidio_faker", "diff"), ("ss-presidio_faker", 0),
                                                     ("ss-presidio_faker", 1), ("ss-gliner_pii", "diff"),
                                                     ("ss-gliner_pii", 0), ("ss-gliner_pii", 1)]):
        d = e["bertscore_differences"][key]
        exact = d["diff"] if field == "diff" else d["ci95"][field]
        assert abs(num(shown) - exact) <= 0.0005 + 1e-9, (key, field, shown, exact)
    j = e["judges"]
    assert abs(num(m[7]) - j["presidio_faker"]["score"]["diff"]) <= 0.0005 + 1e-9
    assert abs(num(m[8]) - j["gliner_pii"]["score"]["diff"]) <= 0.0005 + 1e-9
    assert (int(m[9]), int(m[10])) == (j["gliner_pii"]["ss_wins"], j["gliner_pii"]["gliner_pii_wins"])
    m = found(r"BERTScore of the answer against the answer to the original: ([\d.]+),\s+against ([\d.]+) for "
              r"GLiNER-PII, ([\d.]+) for Presidio \+ Faker and ([\d.]+) for\s+Presidio")
    for shown, arm in zip(m.groups(), ("ss", "gliner_pii", "presidio_faker", "presidio_default")):
        assert abs(num(shown) - e["bertscore"][arm]["mean"]) <= 0.0005 + 1e-9, arm


# ── multi-turn (multiturn_realdata_test3.json) ──────────────────────────────

def test_multiturn_numbers():
    M = load("multiturn_realdata_test3.json")["results"]["all"]
    assert int(found(r"Multi-turn conversations \((\d+) conversations")[1]) == M["ss"]["conversations"]
    m = found(r"restored in the answer: (\d+) of (\d+), against (\d+) of (\d+) for\s+Presidio \+ Faker and "
              r"(\d+) of (\d+) for GLiNER-PII")
    for i, arm in enumerate(("ss", "presidio_faker", "gliner_pii")):
        r = M[arm]["restored"]
        assert (int(m[2 * i + 1]), int(m[2 * i + 2])) == (r["k"], r["n"]), arm
    m = found(r"sent upstream over the conversation: (\d+) of (\d+), against (\d+)\s+for Presidio \+ Faker and "
              r"(\d+) for GLiNER-PII")
    assert (int(m[1]), int(m[2])) == (M["ss"]["payload_gold"]["k"], M["ss"]["payload_gold"]["n"])
    assert int(m[3]) == M["presidio_faker"]["payload_gold"]["k"]
    assert int(m[4]) == M["gliner_pii"]["payload_gold"]["k"]
    assert M["presidio_faker"]["payload_gold"]["n"] == M["gliner_pii"]["payload_gold"]["n"] == M["ss"]["payload_gold"]["n"]


# ── attacker (attacker_exact_test3.json) ────────────────────────────────────

def test_attacker_numbers():
    A = load("attacker_exact_test3.json")["results"]["all"]
    rows = table("| arm | left in plain text | recovered exactly |")
    assert set(rows) == {"SurrogateShield", "GLiNER-PII", "Presidio + Faker", "Presidio", "LLM Guard"}
    for label, (left, exact, known) in rows.items():
        a = A[ARMS[label]]
        assert int(left) == a["left_in"], label
        assert tuple(int(x) for x in re.fullmatch(r"(\d+) of (\d+)", exact).groups()) == (a["exact"]["k"], a["exact"]["n"]), label
        k, n, shown = re.fullmatch(r"(\d+) of (\d+) \(([\d.]+) %\)", known).groups()
        assert (int(k), int(n)) == (a["known_letter_for_letter"]["k"], a["known_letter_for_letter"]["n"]), label
        assert_pct(num(shown), int(k), int(n), f"{label} known letter for letter")
    for m in (found(r"Attacker \((\d+) values"), found(r"sealed test-3 split: (\d+) values in real chat messages")):
        assert int(m[1]) == A["ss"]["values"]
    m = found(r"knows letter for letter: ([\d.]+) % \((\d+) of (\d+)\), against\s+([\d.]+) % for GLiNER-PII, "
              r"([\d.]+) % for Presidio \+ Faker, ([\d.]+) % for Presidio\s+and ([\d.]+) % for LLM Guard")
    k = A["ss"]["known_letter_for_letter"]
    assert (int(m[2]), int(m[3])) == (k["k"], k["n"])
    for shown, arm in zip((m[1], m[4], m[5], m[6], m[7]), ("ss", "gliner_pii", "presidio_faker", "presidio_default", "llm_guard")):
        s = A[arm]["known_letter_for_letter"]
        assert_pct(num(shown), s["k"], s["n"], f"{arm} known letter for letter")


# ── speed and memory (perf_arms_v4.json) ────────────────────────────────────

def test_perf_table():
    perf = {a["arm"]: a for a in load("perf_arms_v4.json")["arms"]}
    rows = table("| arm | median latency | peak memory |")
    labels = {"Presidio": ["presidio_default"], "Presidio + Faker": ["presidio_faker"], "SurrogateShield": ["ss"],
              "LLM Guard": ["llm_guard"], "GLiNER-PII (0.3 / 0.5)": ["gliner_pii_tuned", "gliner_pii"],
              "Presidio + RoBERTa": ["presidio_transformers"]}
    assert set(rows) == set(labels)
    for label, (latency, memory) in rows.items():
        shown_ms = [num(x) for x in latency.replace(" ms", "").split(" / ")]
        shown_mb = [num(x) for x in memory.replace(" MB", "").split(" / ")]
        for arm, ms, mb in zip(labels[label], shown_ms, shown_mb, strict=True):
            assert abs(ms - perf[arm]["p50_ms"]) <= 0.05 + 1e-9, (arm, ms, perf[arm]["p50_ms"])
            assert abs(mb - perf[arm]["peak_rss_mb"]) <= 0.5 + 1e-9, (arm, mb, perf[arm]["peak_rss_mb"])


# ── the verdict, the freeze and the record of each run ──────────────────────

def test_verdict_all_hold():
    v = load("verdict_test3.json")
    assert "All ten checks in" in NORM
    assert len(v["hypotheses"]) == 10
    assert all(h["holds"] for h in v["hypotheses"].values())
    assert not v["not_held"] and not v["not_run"]


def test_sealed_results_record_the_freeze(t3, ext):
    frozen = hashlib.sha256((ROOT / "bench" / "realdata" / "test3" / "FREEZE.json").read_bytes()).hexdigest()
    for doc in (t3, ext, load("verdict_test3.json")):
        assert doc["freeze_sha256"] == frozen


@pytest.mark.parametrize("name", ["realdata_test3", "external_nemotron_pii", "utility_realdata_test3",
                                  "multiturn_realdata_test3", "attacker_realdata_test3", "perf_arms_v4",
                                  "verdict_test3"])
def test_results_readme_names_the_commit(name):
    """bench/results/README.md says at which commit each file was made; the
    file records it. (The attacker_exact_test3.json row names none.)"""
    doc = load(f"{name}.json")
    rows = [l for l in (RESULTS / "README.md").read_text(encoding="utf-8").splitlines()
            if l.startswith(f"| `{name}.json` |")]
    assert len(rows) == 1, name
    assert doc["git"]["commit"][:7] in rows[0], (name, doc["git"]["commit"][:7])


def test_every_result_file_the_readme_names_exists():
    names = set(re.findall(r"bench/results/([\w.-]+\.json)", README))
    assert names
    missing = sorted(n for n in names if not (RESULTS / n).exists())
    assert not missing, missing
