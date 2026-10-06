"""
surrogateshield/core/detection/resolver.py — one candidate per value
(V3 §3.5).

Several stages can report the same value (its text, stripped). For example,
spaCy and distilbert both read "Mercy General", or PatternScan and a plugin
both read an ID. The surrogate is drawn per value, so one candidate has to
stand for it. :func:`resolve` keeps, for each value:

1. the candidates from the best-ranked source. Ranks come from
   ``config.source_priority``, using :func:`stage_of` for a candidate's
   stage; stages not listed share the last rank;
2. among those, when they read the value as different public types, the
   type that the ``type_conflicts`` entry for the pair names. A type that
   loses to another type present is out, and "score" means no preference;
3. the highest score, and on a tie the first reported.

With an empty ``source_priority`` and every conflict set to "score", this is
the rule that ``deduplicate`` used before (highest score, first on a tie).

The other parts of §3.5 sit where they act:

- overlapping spans of different values are planned longest first
  (``entities.plan_substitutions``), and the shorter value keeps its
  surrogate for its other occurrences;
- adjacent PERSON pieces are merged in the cascade
  (``pipeline._merge_adjacent_persons``);
- inside a URL, only pattern entities are kept (``pipeline._outside_opaque``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..entities import DetectedEntity
from .config import DetectionConfig, public_type

_STAGE_OF_SOURCE = {"ner": "entity_trace", "slm": "context_guard", "structural": "structural"}


def stage_of(ent: DetectedEntity) -> str:
    """The config stage an entity came from (``DetectionConfig.detectors``):
    a PatternScan hit, one on a canonical view, a structural pass's PERSON
    or ORG (Passes A, E, H mark theirs "pattern"), a model, or a plugin
    (its source is the stage name)."""
    src = ent.source
    if src == "pattern":
        if ent.view is not None:
            return "canonicaliser"
        return "structural" if ent.type in ("PERSON", "ORG") else "pattern_scan"
    return _STAGE_OF_SOURCE.get(src, src)


def pick(group: List[DetectedEntity], config: DetectionConfig) -> DetectedEntity:
    """The candidate that stands for one value (*group*: every candidate
    whose stripped text is that value, in the order they were reported)."""
    if len(group) == 1:
        return group[0]
    ranks = [config.rank(stage_of(e)) for e in group]
    best = min(ranks)
    top = [e for e, r in zip(group, ranks) if r == best]
    types = list(dict.fromkeys(public_type(e.type) for e in top))
    if len(types) > 1:
        out = {t for t in types for u in types
               if u != t and config.conflict_winner(t, u) == u}
        if len(out) < len(types):          # a cycle would rule every type out: keep all
            top = [e for e in top if public_type(e.type) not in out]
    return max(top, key=lambda e: e.score)  # max keeps the first on a tie


def resolve(entities: List[DetectedEntity],
            config: Optional[DetectionConfig] = None) -> List[DetectedEntity]:
    """One candidate per value (see the module docstring), in the order
    each value was first reported. *config*: the cascade's (default: the
    environment's, else ``balanced``)."""
    if config is None:
        from .pipeline import detection_config
        config = detection_config()
    groups: Dict[str, List[DetectedEntity]] = {}
    for ent in entities:
        groups.setdefault(ent.text.strip(), []).append(ent)
    return [pick(g, config) for g in groups.values()]
