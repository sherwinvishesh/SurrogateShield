"""Gate J2 corpus hygiene (audit B-series, I1/I8/I12/I14): both splits lint
clean, ids are unique across splits, and the shape the gate relies on holds.
Reads only annotations and counts; no model, no network."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("realworld", ROOT / "bench" / "realworld.py")
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)


@pytest.mark.parametrize("split, minimum", [("dev", 100), ("dev2", 200), ("dev3", 200),
                                            ("dev4", 200), ("test", 200)])
def test_J2_corpus_lints_clean(split, minimum):
    records = rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")
    assert rw.lint(records) == []
    s = rw.stats(records)
    assert s["messages"] >= minimum
    assert s["negative_share"] >= 0.20


def test_J2_ids_unique_across_splits():
    ids = [r["id"] for split in rw.SPLITS
           for r in rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("rid, ok", [("rw-dev-0001", True), ("rw-dev3-0200", True), ("rw-test-0001", True),
                                     ("rd-oasst1-dev-0001", True), ("rd-wildchat-test-0250", True),
                                     ("rw-dev5-0001", False), ("rd-oasst1-dev2-0001", False),
                                     ("rd-Oasst1-dev-0001", False), ("rd-oasst1-test-001", False), ("x-0001", False)])
def test_J2_lint_accepts_the_real_data_ids_and_nothing_else(rid, ok):
    rec = {"id": rid, "text": "hi", "category": "qa", "lang": "en", "service_query": False}
    assert (rw.lint([rec]) == []) is ok


def test_J2_cjk_values_are_matched():
    assert rw.occurrences("请联系王伟，电话13812345678", "王伟") == [(3, 5)]


def test_J2_no_message_is_in_two_splits():
    seen = {}
    for split in rw.SPLITS:
        for r in rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl"):
            key = " ".join(r["text"].lower().split())
            assert seen.setdefault(key, split) == split, (r["id"], seen[key])


@pytest.mark.parametrize("split", ["test", "dev4"])
def test_J2_show_refuses_held_out_splits(split, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["realworld.py", "--split", split, "--show"])
    with pytest.raises(SystemExit) as e:
        rw.main()
    assert e.value.code == 2 and "not inspected" in capsys.readouterr().err


def test_J2_run_scores_any_system_with_the_same_code():
    """bench/compare.py scores Presidio through run(prepare=…): an empty edit
    list leaks every non-policy protect value and makes no spurious edit."""
    from types import SimpleNamespace
    s = rw.run("dev", show=False, prepare=lambda text, i: SimpleNamespace(edits=[], sanitized=text))
    assert s["leaked"] == s["protect_values"] - s["policy"] and s["leak_rate"] == 1.0
    assert s["edits"] == s["spurious"] == 0
    assert s["negatives_untouched"] == s["negatives"]


# Phase 4b (E4): the scorer takes --file and any id prefix without changing
# what it computes. These sha256s are of the five split summaries under a fixed
# stub system, computed with the scorer as of 8d2dccf, before --file existed.
GOLDEN = {"dev": "874931514c96d2699ee172ae082b1a14f9cf50d825f4551218a41ff92bef3779",
          "dev2": "fec8c27c88b303a5a4ad3e22454e8ceb21974a6465a4c2167d26ded31b793225",
          "dev3": "713638dd745062401d4edc48cc5a8812e18c9ba662ee1394b999d5e89ad9ef01",
          "dev4": "f71e04d3f6aed581adf1c3528703e65c01965c1591260e5a0f27f6714f8a4a59",
          "test": "3347fbad3fe6b36364c6e660a74400e769e4b04f6ac411bbc1a5117416ae71ed"}


def _stub(text, i):
    from types import SimpleNamespace
    edits = [(m.start(), m.end(), m.group(), "Q" * (len(m.group()) + i % 3))
             for m in re.finditer(r"\b[A-Z][a-z]+\b|\d+|\S+@\S+", text)]
    return SimpleNamespace(edits=edits, sanitized=text)


@pytest.mark.parametrize("split", rw.SPLITS)
def test_J2_file_mode_gives_byte_identical_summaries(split, monkeypatch):
    monkeypatch.setattr(rw.time, "perf_counter", lambda: 0.0)
    by_split = json.dumps(rw.run(split, show=False, prepare=_stub), indent=2)
    by_file = json.dumps(rw.run(None, show=False, prepare=_stub,
                                path=ROOT / "bench" / "realworld" / f"{split}.jsonl"), indent=2)
    assert by_split == by_file
    assert hashlib.sha256(by_split.encode()).hexdigest() == GOLDEN[split]


def test_J2_any_id_prefix(tmp_path):
    rec = {"id": "zz-pilot-0007", "text": "Write to Ada Byrne today.", "category": "qa", "lang": "en",
           "service_query": False, "protect": [{"value": "Ada Byrne", "type": "PERSON"}]}
    assert rw.lint([rec]) == [("zz-pilot-0007", "bad id")]
    assert rw.lint([rec], rw.id_pattern("zz-")) == []
    assert rw.lint([dict(rec, id="zz-pilot-07")], rw.id_pattern("zz-")) == [("zz-pilot-07", "bad id")]
    f = tmp_path / "pilot.jsonl"
    f.write_text(json.dumps(rec) + "\n")
    s = rw.run(None, show=False, prepare=_stub, path=f, id_re=rw.id_pattern("zz-"))
    assert s["split"] == "pilot" and s["protect_values"] == 1 and s["leaked"] == 0


@pytest.mark.parametrize("argv, message", [(["--file", "x.jsonl", "--show"], "never shown"),
                                           (["--file", "x.jsonl", "--split", "dev"], "exclusive")])
def test_J2_file_mode_refuses_show_and_split(argv, message, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["realworld.py", *argv])
    with pytest.raises(SystemExit) as e:
        rw.main()
    assert e.value.code == 2 and message in capsys.readouterr().err
