"""V3 §5.3.1 — the training pools and test-2's are disjoint, and the report
says so in counts and hashes only (``bench/tagger/pools.py``). Model-free and
provider-free: identities are drawn from the generator with a stub plan."""

import json
import random

from bench.realdata import identities as I
from bench.tagger import pools as P

TASKS = ("advice", "writing", "coding")


def _ident(seed, pool):
    return I.identity(random.Random(seed), I.TYPES, TASKS[seed % 3], shift=bool(seed % 2), pool=pool)


def _train_file(tmp_path, extra=()):
    path = tmp_path / "train.jsonl"
    rows = [{"id": f"t{i}", "text": "", "spans": [], "meta": {"pool_tokens": _ident(i, "train")["pool_tokens"]}}
            for i in range(20)]
    rows[0]["meta"]["pool_tokens"] += list(extra)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _plan(ds, coll):
    assert coll.name == "test2"
    return [{"identity": _ident(1000 + 100 * len(ds) + i, "eval")} for i in range(15)]


def test_training_and_test2_pools_are_disjoint_and_reported_without_words(tmp_path, capsys):
    out = tmp_path / "pools.json"
    assert P.main(["--data", str(_train_file(tmp_path)), "--datasets", "oasst1", "wildchat", "--out", str(out)],
                  plan=_plan) == 0
    doc = json.loads(out.read_text())
    (name, train), = doc["training"].items()
    assert doc["ok"] and train["overlap_with_test2"] == 0 and train["other_half"] == 0 and train["words"] > 50
    t2 = doc["test2"]
    assert set(t2["datasets"]) == {"oasst1", "wildchat"} and t2["all"]["other_half"] == 0
    assert t2["all"]["words"] >= max(v["words"] for v in t2["datasets"].values())
    words = {w for i in range(20) for w in _ident(i, "train")["pool_tokens"]}
    assert not any(f'"{w}"' in out.read_text() for w in words)                    # counts and hashes, no words
    assert "0 shared with test-2" in capsys.readouterr().out


def test_a_shared_word_fails_the_report(tmp_path):
    eval_word = _plan("oasst1", P.COLLECTIONS["test2"])[0]["identity"]["pool_tokens"][0]
    out = tmp_path / "pools.json"
    assert P.main(["--data", str(_train_file(tmp_path, [eval_word])), "--datasets", "oasst1", "--out", str(out)],
                  plan=_plan) == 1
    (train,) = json.loads(out.read_text())["training"].values()
    assert train["overlap_with_test2"] == 1 and train["other_half"] == 1


def _plan23(ds, coll):
    off = {"test2": 1000, "test3": 7000}[coll.name]
    return [{"identity": _ident(off + 100 * len(ds) + i, "eval")} for i in range(15)]


def test_test2_and_test3_are_each_checked_against_the_training_pool(tmp_path, capsys):
    out = tmp_path / "pools.json"
    assert P.main(["--data", str(_train_file(tmp_path)), "--datasets", "oasst1", "--collection", "test2", "test3",
                   "--out", str(out)], plan=_plan23) == 0
    doc = json.loads(out.read_text())
    (train,) = doc["training"].values()
    assert doc["ok"] and train["overlap_with_test2"] == train["overlap_with_test3"] == 0
    assert doc["test2"]["all"]["sha256"] != doc["test3"]["all"]["sha256"] and doc["test3"]["all"]["other_half"] == 0
    printed = capsys.readouterr().out
    assert "0 shared with test-2's" in printed and "0 shared with test-3's" in printed
    eval_word = _plan23("oasst1", P.COLLECTIONS["test3"])[0]["identity"]["pool_tokens"][0]
    assert P.main(["--data", str(_train_file(tmp_path, [eval_word])), "--datasets", "oasst1",
                   "--collection", "test2", "test3", "--out", str(out)], plan=_plan23) == 1
    (train,) = json.loads(out.read_text())["training"].values()
    assert train["overlap_with_test3"] >= 1 and not json.loads(out.read_text())["ok"]
