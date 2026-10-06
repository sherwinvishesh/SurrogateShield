"""Model-free tests for the V3 §3.8 GLiNER-PII sweep (bench/realdata/gliner_sweep.py)
and the gliner_pii_tuned arm's label sets."""

from bench import realworld as rw
from bench.arms import gliner_pii, gliner_pii_tuned as tuned
from bench.realdata.gliner_sweep import choose, edits_at, f32


def test_threshold_keeps_scores_strictly_above_it_in_float32():
    # GLiNER keeps probs > threshold with both in float32: a score equal to
    # float32(0.4) is dropped at 0.4 although it exceeds the double 0.4.
    spans = [(0, 4, "person", f32(0.4)), (10, 14, "email", 0.41), (20, 24, "age", 0.39)]
    assert f32(0.4) > 0.4
    assert edits_at(spans, 0.3) == [[0, 4, "person", "[PERSON]"], [10, 14, "email", "[EMAIL]"],
                                    [20, 24, "age", "[AGE]"]]
    assert edits_at(spans, 0.4) == [[10, 14, "email", "[EMAIL]"]]
    assert edits_at(spans, 0.6) == []


def test_overlaps_between_windows_resolve_after_the_filter():
    # Two windows decoded the same name differently: longest wins while it
    # passes the threshold, the shorter one takes over when it does not.
    spans = [(0, 10, "person", 0.35), (0, 5, "person", 0.9), (3, 8, "location", 0.85)]
    assert edits_at(spans, 0.3) == [[0, 10, "person", "[PERSON]"]]
    assert edits_at(spans, 0.4) == [[0, 5, "person", "[PERSON]"]]


def test_edit_labels_follow_the_arm():
    spans = [(0, 3, "driver's license number", 0.7), (5, 9, "phone number", 0.7)]
    assert edits_at(spans, 0.5) == gliner_pii.to_edits(spans) == [
        [0, 3, "driver's license number", "[DRIVERS_LICENSE_NUMBER]"], [5, 9, "phone number", "[PHONE_NUMBER]"]]


def _c(ls, t, leak, spurious):
    return {"label_set": ls, "threshold": t,
            "all": {"injected": {"leak": {"rate": leak}}, "natural": {"spurious": {"k": spurious}}}}


def test_choice_is_lowest_leak_then_fewer_spurious_then_higher_threshold():
    assert choose([_c("a", 0.3, 0.02, 900), _c("b", 0.5, 0.03, 10)])["label_set"] == "a"
    assert choose([_c("a", 0.3, 0.02, 900), _c("b", 0.5, 0.02, 10)])["label_set"] == "b"
    assert choose([_c("a", 0.3, 0.02, 10), _c("a", 0.4, 0.02, 10)])["threshold"] == 0.4


def test_label_sets_cover_every_protect_type():
    assert tuned.LABEL_SETS["published"] == gliner_pii.LABELS
    for name, types in tuned.LABEL_TYPES.items():
        assert list(types) == tuned.LABEL_SETS[name]
        assert set(types.values()) == set(rw.PROTECT_TYPES), name


def test_frozen_choice_is_a_grid_point():
    assert (tuned.LABEL_SET is None) == (tuned.THRESHOLD is None)
    if tuned.LABEL_SET is not None:
        assert tuned.LABEL_SET in tuned.LABEL_SETS
        assert tuned.THRESHOLD in tuned.THRESHOLDS
