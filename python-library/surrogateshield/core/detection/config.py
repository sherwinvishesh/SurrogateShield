"""
surrogateshield/core/detection/config.py — what the detector runs and what
happens to each type it finds (V3 §3.6).

A :class:`DetectionConfig` names:

* ``detectors``: the stages, each a :class:`Stage` (``enabled``, ``model``
  and ``revision``, ``device``, ``thresholds``, ``window`` / ``stride``,
  ``max_latency_ms``, ``options``). The built-in stages run in a fixed
  order (``BUILTIN_STAGES``: each reads what the one before it masked); a
  stage that is not built in is a :mod:`.plugins` detector and runs after
  the model stages, in list order.
* ``type_sources``: which stages may report each public type. Structured
  types come from the pattern stages (and the structural passes, which read
  addresses and host names off the text's layout; the tagger reads phones,
  IDs, addresses, handles and credentials too); names and organisations
  from the tagger (and ContextGuard when it is on), places from the tagger
  and spaCy; ``"*"`` lets every stage report a type (``OTHER``, and the
  free-text types in ``strict`` and ``classic``). A plugin is added with
  :meth:`DetectionConfig.with_plugin`, which lists it as a source.
* ``type_actions``: per type ``replace``, ``shift``, ``redact`` (a
  ``[PERSON_1]`` placeholder) or ``keep`` (detected, sent as typed). An
  address may also be ``auto`` (shift inside a service query, replace
  otherwise). A key is a public type ("PHONE") or an internal one
  ("phone_uk"); the internal key wins.
* ``gate`` / ``gate_bypass``: the relation gate (names not tied to a person
  stay as typed) and the types it never drops. A stage's ``gate_above``
  option: its places and organisations scored at least that count as tied
  to the writer at the gate, which still drops public names, junk and common
  nouns (for a detector trained to tell the writer's names from topics,
  like ``pii_tagger``; V3 §3.4a).
* ``source_priority`` / ``type_conflicts``: which candidate stands for a
  value two stages read differently (the resolver, V3 §3.5).

``config_hash()`` is the SHA-256 of the canonical JSON; the benchmark's SS
arm writes it into every ``.meta.json``. Presets: ``fast``, ``balanced``
(the default) and ``strict``; ``preset(name)``. ``from_env()`` reads
``SURROGATESHIELD_PRESET`` and ``SURROGATESHIELD_DETECTION_CONFIG`` (a JSON
file merged onto the preset).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

PUBLIC_TYPES = ("PERSON", "ORG", "LOCATION", "ADDRESS", "EMAIL", "PHONE", "URL",
                "HANDLE", "ID", "CREDENTIAL", "NETWORK", "AGE", "DATE_OF_BIRTH")
EXTRA_TYPES = ("GENDER", "OTHER")
ALL_TYPES = PUBLIC_TYPES + EXTRA_TYPES
FREE_TEXT_TYPES = ("PERSON", "ORG", "LOCATION", "OTHER")

# internal type names (PatternScan, EntityTrace, ContextGuard, the passes)
INTERNAL_TYPES: Dict[str, Tuple[str, ...]] = {
    "PERSON":        ("PERSON", "person"),
    "ORG":           ("ORG",),
    "LOCATION":      ("GPE", "LOC", "FAC", "implicit_location"),
    "ADDRESS":       ("address", "zip_us", "postcode_uk"),
    "EMAIL":         ("email",),
    "PHONE":         ("phone_us", "phone_uk", "phone_intl"),
    "URL":           ("url", "hostname"),
    "HANDLE":        ("handle",),
    "ID":            ("id_number", "ssn", "passport", "us_driver_license", "license_plate",
                      "vin", "credit_card", "iban", "us_bank_number", "crypto"),
    "CREDENTIAL":    ("credential", "api_key"),
    "NETWORK":       ("ip_address", "mac_address"),
    "AGE":           ("age",),
    "DATE_OF_BIRTH": ("dob",),
    "GENDER":        ("gender_indicator",),
}
_PUBLIC_OF = {i: p for p, internal in INTERNAL_TYPES.items() for i in internal}

# the internal type a candidate reported under a public name is generated as
GENERATOR_TYPE: Dict[str, str] = {
    "LOCATION": "GPE", "ADDRESS": "address", "EMAIL": "email", "PHONE": "phone_intl",
    "URL": "url", "HANDLE": "handle", "ID": "id_number", "CREDENTIAL": "credential",
    "NETWORK": "ip_address", "AGE": "age", "DATE_OF_BIRTH": "dob",
    "GENDER": "gender_indicator",
}


def public_type(typ: str) -> str:
    """The public type of an internal (or public) type name; OTHER for any
    name the map does not know (spaCy's MISC, a plugin's own label)."""
    if typ in _PUBLIC_OF:
        return _PUBLIC_OF[typ]
    return typ if typ in ALL_TYPES else "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

# run order of the built-in stages. ``structural`` is the post passes (A, E,
# F, G, O, H, B, C) and Pass S; ``pii_tagger`` is resolved through the
# detector registry like a plugin.
BUILTIN_STAGES = ("pattern_scan", "canonicaliser", "entity_trace", "context_guard",
                  "pii_tagger", "structural")
MODEL_STAGES = ("entity_trace", "context_guard", "pii_tagger")

# threshold keys a stage reads itself; any other key is a public type and
# sets the lowest score at which that stage's candidates of the type are kept
STAGE_THRESHOLDS = {
    "entity_trace": ("high", "low", "fallback"),
    "context_guard": ("accept",),
}

ACTIONS = ("replace", "shift", "redact", "keep", "auto")
ADDRESS_MODES = ("replace", "shift", "auto")


@dataclass(frozen=True)
class Stage:
    """One detector stage. ``thresholds``: the stage's own keys
    (``STAGE_THRESHOLDS``) and per public type the lowest score kept.
    ``max_latency_ms``: a stage that takes longer is logged and reported in
    the timings as over budget (its candidates are kept: dropping them after
    the time is spent would only lose detections)."""
    name: str
    enabled: bool = True
    model: Optional[str] = None
    revision: Optional[str] = None
    device: int = -1
    thresholds: Mapping[str, float] = field(default_factory=dict)
    window: Optional[int] = None
    stride: Optional[int] = None
    max_latency_ms: Optional[float] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["thresholds"] = dict(sorted(self.thresholds.items()))
        d["options"] = _jsonable(self.options)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stage":
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown stage field(s) {sorted(unknown)} in {d.get('name')!r}")
        d = dict(d)
        d["thresholds"] = dict(d.get("thresholds") or {})
        d["options"] = dict(d.get("options") or {})
        return cls(**d)


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# The models the defaults load, with the revision every benchmark number was
# produced with (doctor reports them; a revision pins the download).
SPACY_MODEL = "en_core_web_lg"
SPACY_REVISION = "3.8.0"
CONTEXT_GUARD_MODEL = "dslim/distilbert-NER"
CONTEXT_GUARD_REVISION = "dfa2838a127384aabb82ed7719e16dab84c42a2a"
# The project's tagger (bench/tagger/, V3 §3.3), fine-tuned from
# microsoft/deberta-v3-small (MIT) on generated text only, chosen over
# deberta-v3-xsmall and ModernBERT-base by the rule in bench/tagger/select.py
# (bench/results/tagger_selection.json). Its weights are
# not on a hub: the stage names a folder under $SURROGATESHIELD_MODELS
# (default ~/.cache/surrogateshield/models; ``python -m bench.tagger.install``
# puts it there) and pins the weights file by its SHA-256.
PII_TAGGER_MODEL = "pii-tagger-dv3s-40k"
PII_TAGGER_REVISION = "sha256:f9e5821151ac59ad6871bf53ffc91ab370a6f8704659a2d3007adf67a920a173"
# Their licences, as recorded (doctor prints them; a spaCy package's own
# meta.json wins when it is installed). Permissive only (V3 §0).
MODEL_LICENCES = {SPACY_MODEL: "MIT", CONTEXT_GUARD_MODEL: "Apache-2.0", PII_TAGGER_MODEL: "MIT"}

# The tagger's lowest kept score per type, as first chosen on dev-large's
# calib half (bench/tagger/configs/sel3ga9-spacyloc.json): names and places
# at 0.5, the structured types it adds to the patterns only when it is near
# certain. Its places and organisations scored 0.9 or more count as tied to
# the writer at the relation gate (``gate_above``). ``fast`` keeps these;
# with them spaCy's per-label scores pass at SPACY_TYPE_GATES.
SEL3_TAGGER_THRESHOLDS = {"PERSON": 0.5, "ORG": 0.5, "LOCATION": 0.5, "PHONE": 0.9, "ID": 0.9,
                          "ADDRESS": 0.8, "HANDLE": 0.9, "CREDENTIAL": 0.95}
SEL3_GATE_ABOVE = 0.9
SPACY_TYPE_GATES = {"high": 0.85, "low": 0.60, "fallback": 0.65}
# ``balanced``: the operating curve's point that was op4 (the values above
# are now its op3, bench/realdata/configs/curve/). With deberta-v3-small it
# leaks no more than they do on dev and edits less harmless text, and it
# passes the §3.3 acceptance: the default rule fixed before the curve was
# run (bench/results/operating_curve_dev.json, tagger_acceptance_dev.json).
# spaCy's labels score PERSON 0.88, GPE/ORG 0.85, LOC 0.74, FAC 0.70: at
# these thresholds none is confirmed outright, PERSON, GPE and ORG are
# promoted at the fallback (ContextGuard off), LOC and FAC are dropped.
PII_TAGGER_THRESHOLDS = {"PERSON": 0.7, "ORG": 0.7, "LOCATION": 0.7, "PHONE": 0.95, "ID": 0.95,
                         "ADDRESS": 0.9, "HANDLE": 0.95, "CREDENTIAL": 0.97}
PII_TAGGER_GATE_ABOVE = 0.97
ENTITY_TRACE_THRESHOLDS = {"high": 0.90, "low": 0.70, "fallback": 0.75}


def _default_stages() -> Tuple[Stage, ...]:
    from .canonical import DEFAULT_VIEWS
    return (
        Stage("pattern_scan"),
        Stage("canonicaliser", options={"views": list(DEFAULT_VIEWS)}),
        Stage("entity_trace", model=SPACY_MODEL, revision=SPACY_REVISION,
              thresholds=dict(ENTITY_TRACE_THRESHOLDS)),
        Stage("context_guard", enabled=False, model=CONTEXT_GUARD_MODEL, revision=CONTEXT_GUARD_REVISION,
              thresholds={"accept": 0.70}),
        Stage("pii_tagger", model=PII_TAGGER_MODEL, revision=PII_TAGGER_REVISION,
              thresholds=dict(PII_TAGGER_THRESHOLDS), options={"gate_above": PII_TAGGER_GATE_ABOVE}),
        Stage("structural"),
    )


_STRUCTURED_SOURCES = ("pattern_scan", "canonicaliser", "structural")
# the structured types the tagger adds to the patterns (it reads e-mails,
# URLs, network addresses, ages and dates of birth no better than they do)
TAGGER_STRUCTURED_TYPES = ("PHONE", "ID", "ADDRESS", "HANDLE", "CREDENTIAL")


def _default_type_sources() -> Dict[str, Tuple[str, ...]]:
    """Names come from the tagger (ContextGuard too, when it is switched on);
    places from spaCy as well; spaCy's names cost more harmless edits than
    they find (bench/tagger/configs/sel1ga9-slm, DETECTOR_PROGRESS)."""
    names = _STRUCTURED_SOURCES + ("pii_tagger", "context_guard")
    out = {t: (_STRUCTURED_SOURCES + ("pii_tagger",) if t in TAGGER_STRUCTURED_TYPES else _STRUCTURED_SOURCES)
           for t in ALL_TYPES}
    out.update(PERSON=names, ORG=names, LOCATION=names + ("entity_trace",), OTHER=("*",))
    return out


DEFAULT_SOURCE_PRIORITY = ("pattern_scan", "canonicaliser", "structural", "pii_tagger",
                           "context_guard", "entity_trace")

# Two candidates of one rank that read a value as different types: the entry
# for the pair (sorted, "A|B") names the type that stands, or "score" (the
# higher-scored candidate). A pair with no entry is "score".
DEFAULT_TYPE_CONFLICTS = {
    "LOCATION|ORG": "score",
    "LOCATION|PERSON": "score",
    "ORG|PERSON": "score",
}


# ─────────────────────────────────────────────────────────────────────────────
# DetectionConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionConfig:
    preset: str = "balanced"
    detectors: Tuple[Stage, ...] = field(default_factory=_default_stages)
    type_sources: Mapping[str, Tuple[str, ...]] = field(default_factory=_default_type_sources)
    type_actions: Mapping[str, str] = field(default_factory=lambda: {"ADDRESS": "auto"})
    gate: bool = True
    gate_bypass: Tuple[str, ...] = ()
    source_priority: Tuple[str, ...] = DEFAULT_SOURCE_PRIORITY
    type_conflicts: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPE_CONFLICTS))
    address_shift_range: int = 1
    service_queries: bool = True

    def __post_init__(self):
        object.__setattr__(self, "detectors", tuple(
            s if isinstance(s, Stage) else Stage.from_dict(s) for s in self.detectors))
        object.__setattr__(self, "type_sources", {k: tuple(v) for k, v in self.type_sources.items()})
        object.__setattr__(self, "type_actions", dict(self.type_actions))
        object.__setattr__(self, "gate_bypass", tuple(self.gate_bypass))
        object.__setattr__(self, "source_priority", tuple(self.source_priority))
        object.__setattr__(self, "type_conflicts", dict(self.type_conflicts))
        self.validate()

    # ── lookups ──────────────────────────────────────────────────────────────

    def stage(self, name: str) -> Stage:
        """The stage named *name*; a disabled one when it is not listed."""
        for s in self.detectors:
            if s.name == name:
                return s
        return Stage(name, enabled=False)

    def enabled(self, name: str) -> bool:
        return self.stage(name).enabled

    def plugin_stages(self) -> Tuple[Stage, ...]:
        """Enabled stages run through the detector registry, in list order."""
        return tuple(s for s in self.detectors
                     if s.enabled and (s.name not in BUILTIN_STAGES or s.name == "pii_tagger"))

    def allows(self, stage: str, typ: str) -> bool:
        """May *stage* report a value of (internal or public) type *typ*?"""
        if not self.enabled(stage):
            return False
        sources = self.type_sources.get(public_type(typ), self.type_sources.get("*", ("*",)))
        return "*" in sources or stage in sources

    def min_score(self, stage: str, typ: str) -> float:
        return float(self.stage(stage).thresholds.get(public_type(typ), 0.0))

    def action_for(self, typ: str) -> str:
        a = self.type_actions.get(typ) or self.type_actions.get(public_type(typ))
        return a or self.type_actions.get("*", "replace")

    @property
    def address_mode(self) -> str:
        """The generator's address mode: the action of ``address``."""
        a = self.action_for("address")
        return a if a in ADDRESS_MODES else "replace"

    def redacted_types(self) -> frozenset:
        """Public types (and internal type names) whose action is redact."""
        return frozenset(k for k, a in self.type_actions.items() if a == "redact")

    def redacts(self, typ: str) -> bool:
        return self.action_for(typ) == "redact"

    def keeps(self, typ: str) -> bool:
        return self.action_for(typ) == "keep"

    def bypasses_gate(self, typ: str) -> bool:
        return not self.gate or "*" in self.gate_bypass or public_type(typ) in self.gate_bypass

    def vouches(self, stage: str, score: float) -> bool:
        """Does *stage*'s ``gate_above`` option vouch, at the relation gate,
        for a candidate of *score*?"""
        above = self.stage(stage).options.get("gate_above")
        return above is not None and score >= above

    def rank(self, stage: str) -> int:
        """Position in ``source_priority`` (lower wins); stages not listed
        share the last rank (an empty list: every stage ties)."""
        if stage in self.source_priority:
            return self.source_priority.index(stage)
        return len(self.source_priority)

    def conflict_winner(self, a: str, b: str) -> str:
        """``type_conflicts`` entry for two public types: a type, or "score"."""
        return self.type_conflicts.get("|".join(sorted((a, b))), "score")

    # ── changes ──────────────────────────────────────────────────────────────

    def replace(self, **changes) -> "DetectionConfig":
        return dataclasses.replace(self, **changes)

    def with_stage(self, name: str, **changes) -> "DetectionConfig":
        """A copy with stage *name* changed (``thresholds`` / ``options``
        merged); a stage not listed yet is appended."""
        out, found = [], False
        for s in self.detectors:
            if s.name == name:
                found = True
                s = _update_stage(s, changes)
            out.append(s)
        if not found:
            out.append(_update_stage(Stage(name), changes))
        return self.replace(detectors=tuple(out))

    def with_actions(self, **actions: str) -> "DetectionConfig":
        return self.replace(type_actions={**self.type_actions, **actions})

    def with_sources(self, **sources) -> "DetectionConfig":
        return self.replace(type_sources={**self.type_sources,
                                          **{k: tuple(v) for k, v in sources.items()}})

    def with_plugin(self, name: str, types=("*",), **stage) -> "DetectionConfig":
        """Add detector *name* and let it report *types* (public types; a
        type whose sources are ``"*"`` already accepts every stage)."""
        cfg = self.with_stage(name, enabled=True, **stage)
        targets = ALL_TYPES if "*" in types else types
        return cfg.with_sources(**{t: tuple(dict.fromkeys(cfg.type_sources.get(t, ()) + (name,)))
                                   for t in targets if "*" not in cfg.type_sources.get(t, ())})

    def merged(self, overrides: Mapping[str, Any]) -> "DetectionConfig":
        """Apply a partial config (JSON form): ``detectors`` merge by stage
        name, the mappings merge by key, anything else replaces."""
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"unknown detection config field(s): {sorted(unknown)}")
        cfg = self
        for key, value in overrides.items():
            if key == "detectors":
                for st in value:
                    st = dict(st)
                    cfg = cfg.with_stage(st.pop("name"), **st)
            elif key in ("type_sources", "type_actions", "type_conflicts"):
                cfg = cfg.replace(**{key: {**getattr(cfg, key), **value}})
            elif key != "preset":
                cfg = cfg.replace(**{key: value})
        return cfg

    # ── JSON and hash ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "detectors": [s.to_dict() for s in self.detectors],
            "type_sources": {k: list(v) for k, v in sorted(self.type_sources.items())},
            "type_actions": dict(sorted(self.type_actions.items())),
            "gate": self.gate,
            "gate_bypass": list(self.gate_bypass),
            "source_priority": list(self.source_priority),
            "type_conflicts": dict(sorted(self.type_conflicts.items())),
            "address_shift_range": self.address_shift_range,
            "service_queries": self.service_queries,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DetectionConfig":
        """A complete config; a partial one is ``preset(name).merged(d)``."""
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown detection config field(s): {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "DetectionConfig":
        return cls.from_dict(json.loads(s))

    def config_hash(self) -> str:
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canon.encode("ascii")).hexdigest()

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self) -> None:
        names = [s.name for s in self.detectors]
        if len(set(names)) != len(names):
            raise ValueError(f"a stage is listed twice in detectors: {names}")
        builtin = [n for n in names if n in BUILTIN_STAGES]
        if builtin != sorted(builtin, key=BUILTIN_STAGES.index):
            raise ValueError(f"built-in stages must keep the order {BUILTIN_STAGES}, got {builtin}")
        for s in self.detectors:
            if not s.name or not isinstance(s.name, str) or "|" in s.name:
                raise ValueError(f"bad stage name {s.name!r}")
            own = STAGE_THRESHOLDS.get(s.name, ())
            for k, v in s.thresholds.items():
                if k not in own and k not in ALL_TYPES:
                    raise ValueError(f"stage {s.name!r}: unknown threshold {k!r} "
                                     f"(its own: {own}, or a public type)")
                if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                    raise ValueError(f"stage {s.name!r}: threshold {k!r} must be in [0, 1], got {v!r}")
            th = s.thresholds
            if s.name == "entity_trace" and th.get("low", 0) > th.get("high", 1):
                raise ValueError("entity_trace: threshold 'low' must not exceed 'high'")
            for k in ("window", "stride"):
                v = getattr(s, k)
                if v is not None and (not isinstance(v, int) or v <= 0):
                    raise ValueError(f"stage {s.name!r}: {k} must be a positive int")
            if s.max_latency_ms is not None and not s.max_latency_ms > 0:
                raise ValueError(f"stage {s.name!r}: max_latency_ms must be > 0")
            if not isinstance(s.device, int) or s.device < -1:
                raise ValueError(f"stage {s.name!r}: device must be -1 (CPU) or a GPU index")
            above = s.options.get("gate_above")
            if above is not None and (isinstance(above, bool) or not isinstance(above, (int, float))
                                      or not 0 <= above <= 1):
                raise ValueError(f"stage {s.name!r}: option 'gate_above' must be a score in [0, 1]")
            labels = s.options.get("labels") if s.name == "context_guard" else None
            if labels is not None:
                if not isinstance(labels, Mapping) or not labels or not all(
                        isinstance(k, str) and k and isinstance(v, str) for k, v in labels.items()):
                    raise ValueError("context_guard: option 'labels' must map model labels to types")
                bad = sorted({v for v in labels.values() if v not in ALL_TYPES and v not in _PUBLIC_OF})
                if bad:
                    raise ValueError(f"context_guard: labels map onto unknown type(s) {bad}")
        known = set(names) | set(BUILTIN_STAGES) | {"*"}
        for t, sources in self.type_sources.items():
            if t not in ALL_TYPES and t != "*":
                raise ValueError(f"type_sources: unknown type {t!r} (public types: {ALL_TYPES})")
            bad = [x for x in sources if x not in known]
            if bad:
                raise ValueError(f"type_sources[{t!r}]: unknown stage(s) {bad}")
        for t, a in self.type_actions.items():
            if a not in ACTIONS:
                raise ValueError(f"type_actions[{t!r}]: {a!r} is not one of {ACTIONS}")
            if t != "*" and t not in ALL_TYPES and t not in _PUBLIC_OF:
                raise ValueError(f"type_actions: unknown type {t!r}")
            if a in ("shift", "auto") and t not in ("ADDRESS", "address"):
                raise ValueError(f"type_actions[{t!r}]: {a!r} applies to ADDRESS only")
        bad = [t for t in self.gate_bypass if t not in ALL_TYPES and t != "*"]
        if bad:
            raise ValueError(f"gate_bypass: unknown type(s) {bad}")
        bad = [x for x in self.source_priority if x not in known - {"*"}]
        if bad:
            raise ValueError(f"source_priority: unknown stage(s) {bad}")
        for pair, winner in self.type_conflicts.items():
            parts = pair.split("|")
            if (len(parts) != 2 or parts != sorted(parts) or parts[0] == parts[1]
                    or any(p not in ALL_TYPES for p in parts)):
                raise ValueError(f"type_conflicts: key {pair!r} must be two sorted public types, 'A|B'")
            if winner != "score" and winner not in parts:
                raise ValueError(f"type_conflicts[{pair!r}]: {winner!r} is neither type nor 'score'")
        if (not isinstance(self.address_shift_range, int) or isinstance(self.address_shift_range, bool)
                or self.address_shift_range < 1):
            raise ValueError("address_shift_range must be an int >= 1")


def _update_stage(s: Stage, changes: Mapping[str, Any]) -> Stage:
    changes = dict(changes)
    for k in ("thresholds", "options"):
        if k in changes:
            changes[k] = {**getattr(s, k), **changes[k]}
    return dataclasses.replace(s, **changes)


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = ("fast", "balanced", "strict", "classic")
# E9 ablations: one part of the balanced config switched off each
ABLATIONS = ("no_tagger", "no_canonicaliser", "no_gate", "no_models", "no_structural")


def preset(name: str = "balanced") -> DetectionConfig:
    """``balanced``: the benchmark configuration (the default): PatternScan,
    the canonicaliser, the tagger, spaCy for places, the structural passes;
    ContextGuard off. ``fast``: no spaCy; the tagger reads places too, from
    0.4 (bench/tagger/configs/sel3ga9-loc4.json). ``strict``: every stage for
    every type it reads, low thresholds, every type past the gate (maximal
    recall; it edits far more harmless text). ``classic``: the detector
    before the tagger (spaCy and ContextGuard read names and places), for a
    machine without the tagger's weights. ``fast``, ``strict`` and
    ``classic`` keep the values they were defined with when ``balanced``
    moved to the curve's op4 (SEL3_*, SPACY_TYPE_GATES). Ablations, each one
    component of ``balanced`` off: ``no_tagger``, ``no_canonicaliser``,
    ``no_gate``, ``no_models``, ``no_structural``."""
    base = DetectionConfig()
    if name == "balanced":
        return base
    if name == "fast":
        return (base.with_stage("entity_trace", enabled=False, thresholds=dict(SPACY_TYPE_GATES))
                    .with_stage("pii_tagger", thresholds={**SEL3_TAGGER_THRESHOLDS, "LOCATION": 0.4},
                                options={"gate_above": SEL3_GATE_ABOVE})
                    .replace(preset="fast"))
    if name == "strict":
        sources = {t: (("*",) if t in FREE_TEXT_TYPES else _STRUCTURED_SOURCES + ("pii_tagger",))
                   for t in ALL_TYPES}
        return (base.with_stage("entity_trace", thresholds={"high": 0.70, "low": 0.40, "fallback": 0.45})
                    .with_stage("context_guard", enabled=True, thresholds={"accept": 0.50})
                    .with_stage("pii_tagger", thresholds={t: 0.3 for t in PII_TAGGER_THRESHOLDS},
                                options={"gate_above": SEL3_GATE_ABOVE})
                    .replace(preset="strict", gate_bypass=("*",), type_sources=sources))
    if name == "classic":
        sources = {t: (("*",) if t in FREE_TEXT_TYPES else _STRUCTURED_SOURCES) for t in ALL_TYPES}
        return (base.with_stage("pii_tagger", enabled=False, thresholds=dict(SEL3_TAGGER_THRESHOLDS),
                                options={"gate_above": SEL3_GATE_ABOVE})
                    .with_stage("entity_trace", thresholds=dict(SPACY_TYPE_GATES))
                    .with_stage("context_guard", enabled=True)
                    .replace(preset="classic", type_sources=sources))
    if name == "no_tagger":
        return base.with_stage("pii_tagger", enabled=False).replace(preset=name)
    if name == "no_canonicaliser":
        return base.with_stage("canonicaliser", enabled=False).replace(preset=name)
    if name == "no_gate":
        return base.replace(preset=name, gate=False)
    if name == "no_models":
        cfg = base
        for st in MODEL_STAGES:
            cfg = cfg.with_stage(st, enabled=False)
        return cfg.replace(preset=name)
    if name == "no_structural":
        return base.with_stage("structural", enabled=False).replace(preset=name)
    raise ValueError(f"unknown preset {name!r}; presets: {PRESETS}, ablations: {ABLATIONS}")


def benchmark() -> DetectionConfig:
    """The real-data benchmark's config: ``balanced`` with every address
    redrawn (deviation D3-4: the product's ``auto`` keeps street and town
    inside a service query, and the scorer cannot tell)."""
    return preset("balanced").with_actions(ADDRESS="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Settings and the environment
# ─────────────────────────────────────────────────────────────────────────────

ENV_PRESET = "SURROGATESHIELD_PRESET"
ENV_FILE = "SURROGATESHIELD_DETECTION_CONFIG"


def env_selected(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Does the environment choose a preset or a config file?"""
    environ = os.environ if environ is None else environ
    return bool(environ.get(ENV_PRESET) or environ.get(ENV_FILE))


def from_partial(overrides: Mapping[str, Any],
                 base: Optional[DetectionConfig] = None) -> DetectionConfig:
    """A config from its JSON form, whole or partial: the preset it names
    (else *base*, else ``balanced``) with the rest merged on top
    (:meth:`DetectionConfig.merged`). A whole config comes back as it was."""
    if not isinstance(overrides, Mapping):
        raise ValueError(f"a detection config must be a JSON object, got {type(overrides).__name__}")
    cfg = base if base is not None else preset("balanced")
    if overrides.get("preset") and overrides["preset"] != cfg.preset:
        cfg = preset(overrides["preset"])
    return cfg.merged(overrides)


def from_env(base: Optional[DetectionConfig] = None,
             environ: Optional[Mapping[str, str]] = None) -> DetectionConfig:
    """*base* (``balanced``), replaced by ``$SURROGATESHIELD_PRESET`` when set,
    with ``$SURROGATESHIELD_DETECTION_CONFIG`` (a JSON file, partial or
    complete) merged on top."""
    environ = os.environ if environ is None else environ
    cfg = base if base is not None else preset("balanced")
    if environ.get(ENV_PRESET):
        cfg = preset(environ[ENV_PRESET])
    path = environ.get(ENV_FILE)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = from_partial(json.load(f), cfg)
    return cfg


def from_settings(base: Optional[DetectionConfig] = None, **settings) -> DetectionConfig:
    """*base* with the flat settings of ``run_cascade`` / the library
    ``Config`` / ``config.py`` applied: ``spacy_model``,
    ``context_guard_enabled``, ``use_context_guard``, ``use_entity_trace``,
    ``use_tagger``, ``use_post_passes``, ``canonical_views``, the four thresholds,
    ``context_guard_model`` / ``_device``, ``address_mode``,
    ``address_shift_range``, ``service``."""
    cfg = base if base is not None else preset("balanced")
    s = dict(settings)
    et, cg = {}, {}
    if "spacy_model" in s:
        m = s.pop("spacy_model")
        et.update(model=m, revision=cfg.stage("entity_trace").revision if m == cfg.stage("entity_trace").model else None)
    if "use_entity_trace" in s:
        et["enabled"] = bool(s.pop("use_entity_trace"))
    th = {k: s.pop(f"entity_trace_{k}_threshold") for k in ("high", "low", "fallback")
          if f"entity_trace_{k}_threshold" in s}
    if th:
        et["thresholds"] = th
    use_cg = s.pop("use_context_guard", None)
    if "context_guard_enabled" in s:
        enabled = s.pop("context_guard_enabled")
        cg["enabled"] = bool(enabled if use_cg is None else use_cg)
    elif use_cg is not None:
        cg["enabled"] = bool(use_cg)
    if "context_guard_threshold" in s:
        cg["thresholds"] = {"accept": s.pop("context_guard_threshold")}
    if "context_guard_model" in s:
        m = s.pop("context_guard_model")
        cg.update(model=m, revision=cfg.stage("context_guard").revision if m == cfg.stage("context_guard").model else None)
    if "context_guard_device" in s:
        cg["device"] = s.pop("context_guard_device")
    if et:
        cfg = cfg.with_stage("entity_trace", **et)
    if cg:
        cfg = cfg.with_stage("context_guard", **cg)
    if "use_tagger" in s:
        cfg = cfg.with_stage("pii_tagger", enabled=bool(s.pop("use_tagger")))
    if "use_post_passes" in s:
        cfg = cfg.with_stage("structural", enabled=bool(s.pop("use_post_passes")))
    if "canonical_views" in s:
        views = s.pop("canonical_views")
        if views is not None:
            views = list(views)
            cfg = cfg.with_stage("canonicaliser", enabled=bool(views), options={"views": views})
    if "address_mode" in s:
        cfg = cfg.with_actions(ADDRESS=s.pop("address_mode"))
    if "address_shift_range" in s:
        cfg = cfg.replace(address_shift_range=s.pop("address_shift_range"))
    if "service" in s:
        cfg = cfg.replace(service_queries=bool(s.pop("service")))
    if s:
        raise TypeError(f"unknown detection setting(s): {sorted(s)}")
    return cfg
