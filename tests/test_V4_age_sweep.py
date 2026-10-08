"""V4 §3.1 A — the AGE threshold sweep (``bench/tagger/age_sweep.py``): its
variants as configs, the edit labels it counts as ages, and the §3.3 rule it
chooses by. Model-free; every count is invented."""

from bench.tagger import age_sweep as S
from bench.tagger import evaluate as E
from surrogateshield.core.detection import config as C


def _cfg(t):
    return C.from_partial(E.tagger_config("pii-tagger-stub", "cpu", extra=S.extra(t)), C.benchmark())


def test_each_variant_routes_age_to_the_tagger_at_its_threshold_or_not_at_all():
    off = _cfg(None)
    assert off.allows("pattern_scan", "AGE") and not off.allows("pii_tagger", "AGE")
    for t in S.THRESHOLDS:
        cfg = _cfg(t)
        assert cfg.allows("pattern_scan", "AGE") and cfg.allows("pii_tagger", "AGE")
        assert cfg.min_score("pii_tagger", "AGE") == t
        # every other type keeps its own threshold
        assert cfg.min_score("pii_tagger", "PERSON") == C.PII_TAGGER_THRESHOLDS["PERSON"]
    assert [S.variant(t) for t in (None, 0.5, 0.85)] == ["age-off", "age50", "age85"]


def test_ages_are_counted_by_label_worded_ones_included():
    assert S.is_age("age") and S.is_age("age:worded")
    assert not any(S.is_age(x) for x in ("dob", "PERSON", "phone_us:worded", "GPE"))


def _part(age=(0, 33), spurious=100, neg=(130, 120, 110), other=None):
    by = {"AGE": age[0], "PERSON": 2, **(other or {})}
    vals = {"AGE": age[1], "PERSON": 200}
    per = {ds: {"negatives_untouched": {"k": k}} for ds, k in zip(("oasst1", "sharegpt", "wildchat"), neg)}
    return {**per, "all": {"leaked_by_type": by, "values_by_type": vals, "natural_spurious": spurious,
                           "negatives_untouched": {"k": sum(neg)}, "natural_age_edits": {}}}


def test_g1_g2_g3_on_one_part():
    ref, gl = _part(age=(1, 33)), _part(age=(0, 33))
    ok = S.checks(_part(), ref, gl)
    assert all(c["ok"] for c in ok.values()) and ok["G1"]["bound"] == 0.02
    assert not S.checks(_part(age=(1, 33)), ref, gl)["G1"]["ok"]            # 1/33 > 0.02
    assert S.checks(_part(age=(1, 68)), ref, gl)["G1"]["ok"]                # 1/68 <= 0.02
    assert S.checks(_part(age=(2, 33)), ref, _part(age=(3, 33)))["G1"]["ok"]   # GLiNER's own rate bounds it
    g2 = S.checks(_part(other={"PERSON": 3}), ref, gl)["G2"]
    assert not g2["ok"] and g2["risen"] == {"PERSON": [2, 3]}
    assert not S.checks(_part(spurious=101), ref, gl)["G3"]["ok"]
    g3 = S.checks(_part(neg=(129, 125, 115)), ref, gl)["G3"]                # OASST1 falls, the pool does not
    assert not g3["ok"] and g3["negatives_fallen"] == {"oasst1": [130, 129]}


def test_the_highest_threshold_passing_everywhere_is_chosen():
    yes = {"G1": {"ok": True}, "G2": {"ok": True}, "G3": {"ok": True}}
    no = {**yes, "G3": {"ok": False}}
    sweep = {"off": {"checks": {"dev": yes}}, "0.5": {"checks": {"dev": yes, "devlarge-calib": yes}},
             "0.8": {"checks": {"dev": yes, "devlarge-calib": yes}},
             "0.9": {"checks": {"dev": yes, "devlarge-calib": no}}}
    assert S.choose(sweep) == {"threshold": 0.8, "passing": ["0.5", "0.8"]}
    assert S.choose({"0.9": sweep["0.9"]}) == {"threshold": None, "passing": []}
