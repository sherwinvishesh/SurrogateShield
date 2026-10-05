"""Model-free tests for bench/realdata/live.py (Phase 6): the reply cache and
call counter, the samples, the attacker's matching, and each live phase end to
end with a stub batch runner, a stub BERTScore and, for the multi-turn replay,
the app's Pipeline with a regex cascade. No network, no model."""

import json
import re

import pytest

from bench.realdata import live as L
from bench.realdata import provider as pv
from tests.test_I3_I5_consistency import fake_cascade


# ── stubs ────────────────────────────────────────────────────────────────────

def _text_of(params):
    return params["messages"][-1]["content"]


def _block(content, label):
    """The text between ``<label>\n<<<\n`` and ``\n>>>`` of a judge prompt."""
    return content.split(f"{label}\n<<<\n", 1)[1].split("\n>>>", 1)[0]


class Stub:
    """A batch runner: takes the calls from the ledger like ``run_batch`` and
    answers each request with ``reply(params)`` (a message dict, or None for an
    errored request)."""

    def __init__(self, reply):
        self.reply, self.batches = reply, []

    def __call__(self, client, ledger, name, phase, kind, requests, log=print, directory=None):
        ledger.take(len(requests), phase, kind, requests[0]["params"]["model"])
        self.batches.append((name, phase, kind, [r["params"] for r in requests]))
        out = {}
        for r in requests:
            msg = self.reply(r["params"])
            out[r["custom_id"]] = ({"status": "errored", "error": "stub error"} if msg is None
                                   else {"status": "succeeded", "message": msg})
        return out


def text_msg(text, stop="end_turn"):
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop, "model": "stub",
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def tool_msg(name, **inp):
    return {"content": [{"type": "tool_use", "name": name, "input": inp}], "stop_reason": "tool_use",
            "model": "stub", "usage": {"input_tokens": 20, "output_tokens": 3}}


def respond(params):
    """Responder: echo the request; judges: prefer answers without a
    ``[`` placeholder, grade by whether Sarah Mitchell is named; attacker: guess
    Sarah Mitchell and a wrong e-mail at the right domain."""
    content = _text_of(params)
    if params.get("tools"):
        if params["tools"][0]["name"] == "verdict":
            a, b = _block(content, "Answer A:"), _block(content, "Answer B:")
            better = "tie" if ("[" in a) == ("[" in b) else ("A" if "[" in b else "B")
            return tool_msg("verdict", better=better, reason="r")
        ans = _block(content, "Final answer:")
        return tool_msg("grade", grade="correct" if "Sarah Mitchell" in ans else "incorrect", reason="r")
    if params["model"] == pv.OPUS:
        return text_msg(json.dumps({"estimates": [
            {"seen": "x", "type": "PERSON", "original_estimate": "Sarah Mitchell", "confidence": 0.9},
            {"seen": "y", "type": "EMAIL", "original_estimate": "someone@corp.com", "confidence": 0.5}]}))
    return text_msg("You said: " + content)


def stub_scorer(cands, refs):
    return [1.0 if c == r else 0.5 for c, r in zip(cands, refs)]


@pytest.fixture
def run(tmp_path):
    def make(reply=respond, **kw):
        stub = Stub(reply)
        r = L.Run("t", root=tmp_path / "runs", ledger=pv.Ledger(tmp_path / "ledger", cap=1000, run="t"),
                  batch_runner=stub, log=lambda *_: None, **kw)
        r.stub = stub
        return r
    return make


# ── data ─────────────────────────────────────────────────────────────────────

NAME, EMAIL = "Sarah Mitchell", "sarah@corp.com"


def _unit(ds, mid, text, protect, slices, conv=None, turn=0):
    return {"mid": mid, "conv": conv or mid, "dataset": ds, "task": "advice", "slices": slices, "turn": turn,
            "gold": {"id": mid, "text": text, "service_query": False, "protect": [dict(p) for p in protect],
                     "sensitive": [], "optional": [], "keep": []}}


def _edit(text, value, rep, typ):
    i = text.index(value)
    return [i, i + len(value), typ, rep]


def build(datasets=("oasst1", "sharegpt"), single=3, natural=2):
    """Per dataset: *single* injected prompts naming Sarah Mitchell and her
    e-mail, *natural* prompts with nothing to protect, and one two-turn
    conversation; span rows for every arm."""
    data = {}
    for ds in datasets:
        units, spans = {}, {a: {} for a in L.SPAN_ARMS}
        for i in range(single):
            mid = f"{ds}-s{i}"
            text = f"Write to {NAME} at {EMAIL} about item {i}."
            units[mid] = _unit(ds, mid, text, [{"type": "PERSON", "value": NAME}, {"type": "EMAIL", "value": EMAIL}],
                               ("injected", "injected_single"))
            spans["ss"][mid] = {"id": mid, "edits": [_edit(text, NAME, "Kara Doyle", "PERSON"),
                                                     _edit(text, EMAIL, "kara@corp.com", "EMAIL")]}
            spans["presidio_default"][mid] = {"id": mid, "edits": [_edit(text, NAME, "[PERSON]", "PERSON")]}
            spans["presidio_faker"][mid] = {"id": mid, "edits": [_edit(text, NAME, "Jo Bloggs", "PERSON"),
                                                                 _edit(text, EMAIL, "jo@mail.net", "EMAIL")]}
            spans["llm_guard"][mid] = ({"id": mid, "edits": [], "refused": "stub"} if i == 0 else
                                       {"id": mid, "edits": [_edit(text, NAME, "Al Brown", "PERSON"),
                                                             _edit(text, EMAIL, "al@b.org", "EMAIL")]})
        for i in range(natural):
            mid = f"{ds}-n{i}"
            text = f"How do I sort a list in Python, version {i}?"
            units[mid] = _unit(ds, mid, text, [], ("natural",))
            j = text.index("Python")
            spans["ss"][mid] = {"id": mid, "edits": [[j, j + 6, "ORG", "Rython"]] if i == 0 else []}
            for a in ("presidio_default", "presidio_faker", "llm_guard"):
                spans[a][mid] = {"id": mid, "edits": []}
        conv = f"{ds}-c0"
        for t, text in enumerate([f"I am {NAME}, mail me at {EMAIL}.", f"Remind {NAME} of the plan."]):
            mid = f"{conv}#t{t}"
            protect = [{"type": "PERSON", "value": NAME}] + ([{"type": "EMAIL", "value": EMAIL}] if t == 0 else [])
            units[mid] = _unit(ds, mid, text, protect, ("injected", "multi"), conv=conv, turn=t)
            edits = [_edit(text, NAME, f"Pat Fake{t}", "PERSON")]          # a new fake every turn
            if t == 0:
                edits.append(_edit(text, EMAIL, "pat@fake.org", "EMAIL"))
            spans["presidio_faker"][mid] = {"id": mid, "edits": sorted(edits)}
            for a in ("ss", "presidio_default", "llm_guard"):
                spans[a][mid] = {"id": mid, "edits": []}
        data[ds] = {"units": units, "spans": spans}
    return data


# ── the run ──────────────────────────────────────────────────────────────────

def test_fetch_sends_only_missing_requests_counts_them_and_keeps_errors(run, tmp_path):
    fails = {"bad"}
    r = run(lambda p: None if _text_of(p) in fails else text_msg("ok " + _text_of(p)))
    items = [("a", L.responder_params([{"role": "user", "content": t}])) for t in ("x", "bad", "x")]
    rows = r.fetch("s", "6-utility", "responder", items)
    assert [x["status"] for x in rows] == ["ok", "errored", "ok"] and rows[0]["text"] == "ok x"
    assert r.sent == 2 and r.ledger.total() == 2 and len(r.stub.batches) == 1   # "x" asked twice, sent once
    assert r.fetch("s", "6-utility", "responder", items) == rows and r.sent == 2  # all cached, errors too

    again = L.Run("t", root=tmp_path / "runs", ledger=r.ledger, batch_runner=r.stub, retry_errors=True,
                  log=lambda *_: None)
    fails.clear()
    rows2 = again.fetch("s", "6-utility", "responder", items)
    assert rows2[1]["status"] == "ok" and rows2[1]["attempt"] == 2 and again.sent == 1
    assert r.stub.batches[-1][0].endswith("-a2")
    assert again.fetch("s", "6-utility", "responder", items) == rows2 and again.sent == 1

    f = tmp_path / "runs" / "t" / "results.jsonl"
    assert oct(f.stat().st_mode)[-3:] == "600" and oct(f.parent.stat().st_mode)[-3:] == "700"
    assert L.Run("t", root=tmp_path / "runs").get("a", items[1][1])["status"] == "ok"   # the retry wins on reload


def test_fetch_refuses_before_sending_past_max_calls(run):
    r = run(max_calls=1)
    items = [("a", L.responder_params([{"role": "user", "content": t}])) for t in ("x", "y")]
    with pytest.raises(pv.BudgetExceeded):
        r.fetch("s", "6-utility", "responder", items)
    assert r.ledger.total() == 0 and not r.stub.batches


def test_reply_row_reads_text_tool_truncation_and_errors():
    k = "0" * 64
    assert L.reply_row(k, "a", "s", "r", 1, {"status": "succeeded", "message": text_msg("hi", "max_tokens")})[
        "status"] == "truncated"
    row = L.reply_row(k, "j", "s", "judge", 1, {"status": "succeeded", "message": tool_msg("verdict", better="A")})
    assert row["status"] == "ok" and row["tool"] == {"better": "A"} and row["text"] == ""
    bad = L.reply_row(k, "a", "s", "r", 1, {"status": "expired", "error": "late"})
    assert bad["status"] == "errored" and bad["error"] == "late"
    assert L.reply_row(k, "a", "s", "r", 1, None)["error"] == "no result"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", L._cid(L.cache_key("a", {"x": 1})))


def test_cache_key_depends_on_slot_and_every_parameter():
    p = L.responder_params([{"role": "user", "content": "x"}])
    assert L.cache_key("orig", p) == L.cache_key("orig", json.loads(json.dumps(p)))
    assert L.cache_key("orig", p) != L.cache_key("rerun", p)
    assert L.cache_key("orig", p) != L.cache_key("orig", {**p, "max_tokens": 1})


def test_roles_and_prompts():
    import attacker
    from chatbot.chat import SYSTEM_PROMPT
    L.check_roles()
    assert pv.OPUS != pv.SONNET
    p = L.responder_params([{"role": "user", "content": "x", "extra": 1}])
    assert p["model"] == pv.SONNET and p["system"] == SYSTEM_PROMPT and p["max_tokens"] == 4096
    assert p["messages"] == [{"role": "user", "content": "x"}]
    t = L.attack_prompts()
    assert t["substitute"] == attacker.SS_PROMPT
    assert "[PERSON], [LOCATION] or [EMAIL_ADDRESS]" in t["placeholder"] and "<PERSON>" not in t["placeholder"]
    for tmpl in t.values():
        assert "{x}" in tmpl.format(text="{x}")                 # text with braces survives
        assert '"estimates"' in tmpl.format(text="")
    rec = L.prompt_record()
    assert rec["version"] == L.PROMPT_VERSION and set(rec["sha256"]) == set(rec["templates"])
    jp = L.judge_pair_params("req", "one", "two")
    assert jp["model"] == pv.OPUS and jp["tool_choice"]["name"] == "verdict"
    assert _block(_text_of(jp), "Answer B:") == "two"


# ── samples ──────────────────────────────────────────────────────────────────

def test_samples_are_seeded_by_dataset_and_select_their_slices():
    data = build(single=12, natural=6)
    for ds in data:
        u = L.utility_sample(data, ds, 5)
        assert u == L.utility_sample(data, ds, 5) and len(set(u)) == 5
        assert all("-s" in m for m in u) and L.attack_sample(data, ds, 3) == L.utility_sample(data, ds)[:3]
        assert all("-n" in m for m in L.natural_sample(data, ds, 4))
        assert L.multiturn_sample(data, ds) == [f"{ds}-c0"]
    order = [m.split("-", 1)[1] for m in L.utility_sample(data, "oasst1", 12)]
    assert order != [m.split("-", 1)[1] for m in L.utility_sample(data, "sharegpt", 12)]   # seeded per dataset


def test_pilot_interleaves_datasets_and_none_keeps_everything():
    s = {"a": [1, 2, 3], "b": [4, 5], "c": [6]}
    assert L.pilot_of(s, 4) == {"a": [1, 2], "b": [4], "c": [6]}
    assert L.pilot_of(s, None) is s and L.pilot_of(s, 0) == {"a": [], "b": [], "c": []}


# ── attacker matching ────────────────────────────────────────────────────────

@pytest.mark.parametrize("typ,guess,gold,level,cls", [
    ("PERSON", "Mitchell, Sarah", "Sarah Mitchell", "exact", None),
    ("PERSON", "Sarah Jones", "Sarah Mitchell", "partial", "name_token"),
    ("EMAIL", "SARAH (at) corp (dot) com", "sarah@corp.com", "exact", None),
    ("EMAIL", "kara@corp.com", "sarah@corp.com", "partial", "email_domain"),
    ("EMAIL", "sarah@other.org", "sarah@corp.com", "partial", "email_local"),
    ("PHONE", "+1 (602) 555-0143", "602.555.0143", "exact", None),
    ("PHONE", "six oh two five five five oh one four three", "602-555-0143", "exact", None),
    ("PHONE", "602-555-9999", "602-555-0143", "partial", "phone_area"),
    ("PHONE", "07700 900123", "+44 7700 900123", "exact", None),
    ("AGE", "thirty-four", "34", "exact", None),
    ("AGE", "35", "34", None, None),
    ("DATE_OF_BIRTH", "3 March 1985", "March 3rd, 1985", "exact", None),
    ("DATE_OF_BIRTH", "June 1985", "March 3rd, 1985", "partial", "birth_year"),
    ("NETWORK", "10.0.0.7", "10.0.0.9", "partial", "ipv4_24"),
    ("NETWORK", "2001:db8:0:0::1", "2001:db8::1", "exact", None),
    ("NETWORK", "AA-BB-CC-DD-EE-FF", "aa:bb:cc:dd:ee:ff", "exact", None),
    ("URL", "http://www.site.example/", "https://site.example", "exact", None),
    ("URL", "site.example/other", "https://site.example/page", "partial", "url_host"),
    ("HANDLE", "sarahm", "@SarahM", "exact", None),
    ("ID", "gb82 west 1234", "GB82-WEST-1234", "exact", None),
    ("CREDENTIAL", "zqk-abc", "zqk-ABC", None, None),
    ("ADDRESS", "9 Elm Street", "12 Oak Street", None, None),
    ("ADDRESS", "4 Oak Road", "12 Oak Street", "partial", "address_part"),
    ("ORG", "Leeds Bank", "Northern Bank", "partial", "org_word"),
    ("LOCATION", "", "Leeds", None, None),
])
def test_recovery_levels_and_partial_classes(typ, guess, gold, level, cls):
    got = L.recovery(typ, guess, gold)
    assert got[:2] == (level, cls)


def test_attack_outcomes_count_left_in_and_unavailable_apart():
    gold = [("PERSON", NAME), ("EMAIL", EMAIL), ("PHONE", "602-555-0143")]
    parsed = {"estimates": [{"original_estimate": "Sarah Mitchell"}, {"original_estimate": "x@corp.com"},
                            {"original_estimate": None}]}
    rows = L.attack_outcomes(gold, {"602-555-0143": "leaked"}, parsed, "Write to Kara Doyle at kara@corp.com")
    assert [r["outcome"] for r in rows] == ["exact", "partial", "leaked"]
    assert rows[1]["class"] == "email_domain" and rows[1]["visible"] is True
    assert {r["outcome"] for r in L.attack_outcomes(gold, {}, None, "")} == {"unavailable"}
    s = L.summarise_attack(rows + [{"type": "PERSON", "outcome": "refused", "class": None, "visible": None}])
    assert s["attacked"] == 2 and s["exact"]["k"] == 1 and s["leaked"] == 1 and s["refused"] == 1
    assert s["exposed"]["k"] == 3 and s["exposed"]["n"] == 3
    assert s["partial_classes"] == {"email_domain": {"n": 1, "visible_in_text": 1}}


def test_left_in_uses_the_scorer_and_message_gold_is_distinct():
    data = build(datasets=("oasst1",))
    u = data["oasst1"]["units"]["oasst1-s1"]
    assert L.left_in_by_arm(u, L.edits_of(data, "oasst1", "presidio_default", "oasst1-s1")) == {EMAIL: "leaked"}
    assert L.left_in_by_arm(u, L.edits_of(data, "oasst1", "ss", "oasst1-s1")) == {}
    u2 = {"gold": {"protect": [{"type": "PERSON", "value": "Ann"}, {"type": "PERSON", "value": "ANN"}]}}
    assert L.message_gold(u2) == [("PERSON", "Ann")]


# ── statistics ───────────────────────────────────────────────────────────────

def test_bertscore_is_cached_private_and_skips_empty_texts(tmp_path):
    calls = []

    def scorer(c, r):
        calls.append(len(c))
        return [0.25] * len(c)
    cache = tmp_path / "bs.jsonl"
    assert L.bertscore([("a", "b"), ("", "b"), ("a", "b")], cache, scorer) == [0.25, None, 0.25]
    assert L.bertscore([("a", "b"), ("c", "d")], cache, scorer) == [0.25, 0.25]
    assert calls == [1, 1] and oct(cache.stat().st_mode)[-3:] == "600"


def test_mean_diff_is_paired_and_handles_empty():
    d = L.mean_diff(["x", "y", "z"], [1.0, 0.8, 0.9], [0.5, 0.4, 0.4], seed=1)
    assert d["diff"] == pytest.approx(0.4667, abs=1e-3) and d["ci95"][0] > 0 and d["n"] == 3
    assert L.mean_diff([], [], [], seed=1)["diff"] is None


# ── the phases, end to end ───────────────────────────────────────────────────

def test_utility_restores_ss_judges_blind_and_checks_h3(run):
    data = build()
    r = run()
    out = L.utility(r, data, list(data), scorer=stub_scorer)
    for ds in data:
        assert sorted(out["sample"][ds]) == sorted(m for m in data[ds]["units"] if "-s" in m)
    res = out["results"]["all"]
    assert res["n"] == 6 and res["refused"]["original"] == 0
    assert res["bertscore"]["ss"]["mean"] == 1.0          # restored: the original's answer
    assert res["bertscore"]["ss_raw"]["mean"] == 0.5 and res["bertscore"]["presidio_default"]["mean"] == 0.5
    j = res["judge"]
    assert j["ss_wins"] == 6 and j["presidio_default_wins"] == 0 and j["n"] == 6
    assert 0 < j["first_position_chosen"]["k"] < 6        # positions vary, the verdict follows the arm
    assert j["agreement_with_bertscore"]["rate"] == 1.0
    assert res["H3"] == {"bertscore_above_presidio_default": True, "judge_above_presidio_default": True,
                         "not_below_presidio_faker": True, "supported": True}
    for _n, _p, kind, params in r.stub.batches:
        if kind == "judge":
            assert all("presidio" not in _text_of(p) and "SurrogateShield" not in _text_of(p) for p in params)
            assert all(p["model"] == pv.OPUS for p in params)
        else:
            assert all(p["model"] == pv.SONNET for p in params)
    sent = r.sent
    L.utility(r, data, list(data), scorer=stub_scorer)
    assert r.sent == sent                                 # a rerun sends nothing
    assert "| all | 6 |" in L.markdown("utility", {"command": "c", "git": {"commit": "abc", "modified": []},
                                                   "e5b": out["results"], "natural": L.natural(
                                                       r, data, list(data), scorer=stub_scorer)["results"]})


def test_natural_sends_edited_prompts_only_and_reuses_the_rerun(run):
    data = build()
    r = run()
    out = L.natural(r, data, list(data), scorer=stub_scorer)
    res = out["results"]["all"]
    assert res["n"] == 4 and res["edited"]["ss"]["k"] == 2 and res["edited"]["presidio_default"]["k"] == 0
    assert r.sent == 4 * 2 + 2                            # orig + rerun each, ss on the two it edited
    assert res["bertscore"]["rerun"]["mean"] == 1.0       # the stub answers identically
    assert res["bertscore"]["presidio_default"]["mean"] == 1.0
    sent_texts = [_text_of(p) for b in r.stub.batches for p in b[3]]
    assert sum("Rython" in t for t in sent_texts) == 2


def test_multiturn_replays_the_app_restores_and_compares_with_presidio_faker(run, tmp_path, monkeypatch):
    monkeypatch.setattr(L, "TMP", tmp_path / "live-tmp")
    data = build()
    r = run()
    out = L.multiturn(r, data, list(data), cascade=fake_cascade, log=lambda *_: None)
    res = out["results"]["all"]
    ss, pf = res["ss"], res["presidio_faker"]
    assert ss["complete"] == pf["complete"] == 2
    assert ss["restored"]["rate"] == 1.0 and pf["restored"]["rate"] == 0.0
    assert ss["history_original"]["k"] == 0 and ss["payload_gold"]["k"] == 0
    assert ss["consistent"]["rate"] == 1.0 and pf["consistent"]["rate"] == 0.0
    assert pf["replaced_every_turn"]["rate"] == 1.0
    assert ss["grades"] == {"correct": 2} and pf["grades"] == {"incorrect": 2}
    assert res["H4"]["supported"] is True
    assert not any((tmp_path / "live-tmp").iterdir())     # every temporary home removed
    # turn 2 of each SS conversation carries turn 1's surrogate reply as history, never the original
    hist = [b for b in r.stub.batches if b[1] == "6-multiturn"]
    assert len(hist) == 2 and len(hist[1][3]) == 4
    for p in hist[1][3]:
        assert len(p["messages"]) == 3 and all(NAME not in m["content"] for m in p["messages"])
    sent = r.sent
    L.multiturn(r, data, list(data), cascade=fake_cascade, log=lambda *_: None)
    assert r.sent == sent
    h = L.read_jsonl(r.dir / L.HISTORIES)
    assert len(h) == 2 and oct((r.dir / L.HISTORIES).stat().st_mode)[-3:] == "600"
    assert "| all | ss |" in L.markdown("multiturn", {"command": "c", "git": {"commit": "abc", "modified": []},
                                                     "results": out["results"]})


def test_multiturn_stops_a_conversation_at_an_errored_turn(run, tmp_path, monkeypatch):
    monkeypatch.setattr(L, "TMP", tmp_path / "live-tmp")
    data = build(datasets=("oasst1",))
    r = run(lambda p: None if len(p["messages"]) > 1 else respond(p))
    out = L.multiturn(r, data, ["oasst1"], cascade=fake_cascade, log=lambda *_: None)
    rows = {x["arm"]: x for x in out["rows"]}
    assert rows["ss"]["complete"] is False and rows["ss"]["turns"] == 2 and rows["ss"]["grade"] is None
    assert out["results"]["oasst1"]["H4"]["supported"] is False


def test_attack_uses_the_placeholder_prompt_for_presidio_default_and_scores_the_conversations(run, tmp_path,
                                                                                            monkeypatch):
    monkeypatch.setattr(L, "TMP", tmp_path / "live-tmp")
    data = build()
    r = run()
    L.multiturn(r, data, list(data), cascade=fake_cascade, log=lambda *_: None)
    out = L.attack(r, data, list(data))
    batches = [b for b in r.stub.batches if b[1] == "6-attacker"]
    single = [p for b in batches if b[0].startswith("attack-") and not b[0].startswith("attack-conv") for p in b[3]]
    assert len(single) == 6 * 4 - 2                       # llm_guard refused one prompt per dataset
    assert sum("[PERSON], [LOCATION]" in _text_of(p) for p in single) == 6
    assert all(p["model"] == pv.OPUS and "system" not in p for p in single)
    res = out["results"]["all"]
    assert res["ss"]["exact"]["k"] == 6 and res["ss"]["partial"]["k"] == 6        # the stub guesses the name
    assert res["ss"]["partial_classes"]["email_domain"] == {"n": 6, "visible_in_text": 6}
    assert res["presidio_default"]["leaked"] == 6 and res["llm_guard"]["refused"] == 4
    assert res["H5"]["supported"] is False
    conv = res["ss-conversation"]
    assert conv["values"] == 4 and conv["exact"]["k"] == 2 and res["H5-conv"]["supported"] is False
    conv_batches = [b for b in batches if b[0].startswith("attack-conv")]
    assert len(conv_batches) == 1 and all("Assistant:" in _text_of(p) and NAME not in _text_of(p)
                                          for p in conv_batches[0][3])
    assert "| all | ss-conversation |" in L.markdown("attacker", {"command": "c", "git": {"commit": "a", "modified": []},
                                                                 "results": out["results"]})


def test_estimate_counts_what_is_missing_and_tokens_from_cached_replies(run):
    data = build()
    r = run()
    before = L.estimate(r, data, list(data))
    assert before["plan"]["utility responder"] == 6 * 4 and before["plan"]["attacker single"] == 6 * 4 - 2
    assert before["plan"]["natural responder"] == 4 * 2 + 2 and before["plan"]["multiturn judge"] == 4
    L.natural(r, data, list(data), scorer=stub_scorer)
    after = L.estimate(r, data, list(data))
    assert after["plan"]["natural responder"] == 0 and after["ledger_total"] == 10
    assert after["tokens_per_call"]["responder"] == {"in": 10, "out": 5, "n": 10}
