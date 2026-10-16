"""
surrogateshield/core/detection/plugins.py — detectors added from outside
(V3 §3.6).

A detector is any object with ``detect(text, view) -> list[Candidate]``.
``text`` is the message; ``view`` is a :class:`~.canonical.View` of it with
look-alike characters folded and spelled-out digits written as digits (its
``original(s, e)`` maps a span back), or None when nothing needed
rewriting. Offsets in a :class:`Candidate` are offsets in ``text``.

Register a factory under a stage name, then list the stage::

    from surrogateshield.core.detection import plugins, config

    plugins.register_detector("employee_ids", lambda stage: EmployeeIds())
    cfg = config.preset("balanced").with_plugin("employee_ids", types=["ID"])

The factory receives the :class:`~.config.Stage` (its ``model``,
``thresholds``, ``options``). A package can instead declare an entry point
in the ``surrogateshield.detectors`` group (``name = "pkg.module:factory"``);
it is loaded the first time a config lists that stage. ``pii_tagger`` is
built in (:mod:`.pii_tagger`); a factory registered under that name
replaces it.

A plugin's candidates go through everything a model's do: the type routing
(``type_sources``), the stage's per-type thresholds, the URL rule (nothing
inside a URL), the relation gate (unless the type bypasses it) and the
resolver. A type is a public type ("ID") or an internal one ("ssn").
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from ..entities import DetectedEntity
from ..errors import DetectorUnavailable
from .config import GENERATOR_TYPE, Stage

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "surrogateshield.detectors"


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    type: str
    score: float = 1.0


@runtime_checkable
class Detector(Protocol):
    def detect(self, text: str, view) -> List[Candidate]: ...


Factory = Callable[[Stage], Detector]

_factories: Dict[str, Factory] = {}
_instances: Dict[tuple, Detector] = {}
_lock = threading.Lock()


def register_detector(name: str, factory: Factory, *, replace: bool = False) -> None:
    """Make *factory* the detector of stage *name*."""
    from .config import BUILTIN_STAGES
    if name in BUILTIN_STAGES and name != "pii_tagger":
        raise ValueError(f"{name!r} is a built-in stage")
    if not callable(factory):
        raise TypeError("factory must be callable")
    with _lock:
        if name in _factories and not replace and _factories[name] is not factory:
            raise ValueError(f"a detector is already registered as {name!r} (replace=True to swap it)")
        _factories[name] = factory
        for key in [k for k in _instances if k[0] == name]:
            del _instances[key]


def unregister_detector(name: str) -> None:
    with _lock:
        _factories.pop(name, None)
        for key in [k for k in _instances if k[0] == name]:
            del _instances[key]


def registered() -> List[str]:
    return sorted(_factories)


def _entry_point(name: str) -> Optional[Factory]:
    try:
        from importlib.metadata import entry_points
    except ImportError:          # pragma: no cover - Python < 3.8
        return None
    eps = entry_points()
    group = (eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select")
             else eps.get(ENTRY_POINT_GROUP, ()))
    for ep in group:
        if ep.name == name:
            return ep.load()
    return None


def _builtin(name: str) -> Optional[Factory]:
    """The project's own detectors, imported on first use (they load torch)."""
    if name == "pii_tagger":
        from .pii_tagger import factory
        return factory
    return None


def get_detector(stage: Stage) -> Detector:
    """The detector instance for *stage* (one per stage settings). A
    registered factory wins over the built-in one, then an entry point."""
    key = (stage.name, repr(stage.to_dict()))
    with _lock:
        if key in _instances:
            return _instances[key]
        factory = _factories.get(stage.name)
    if factory is None:
        factory = _builtin(stage.name)
    if factory is None:
        factory = _entry_point(stage.name)
        if factory is None:
            raise DetectorUnavailable(
                f"no detector is registered as {stage.name!r} (register_detector or an "
                f"entry point in {ENTRY_POINT_GROUP!r}); switch the stage off to run without it")
        with _lock:
            _factories.setdefault(stage.name, factory)
    det = factory(stage)
    if not callable(getattr(det, "detect", None)):
        raise TypeError(f"detector {stage.name!r} has no detect(text, view) method")
    with _lock:
        return _instances.setdefault(key, det)


def candidates_to_entities(name: str, text: str, found) -> List[DetectedEntity]:
    """Plugin candidates as entities of source *name*; one with offsets
    outside *text*, or empty, is dropped (and logged)."""
    out = []
    for c in found or ():
        start, end, typ, score = c.start, c.end, c.type, float(getattr(c, "score", 1.0))
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            logger.warning(f"[plugins] {name}: candidate with bad offsets ({start}, {end}) dropped")
            continue
        if not text[start:end].strip():
            continue
        out.append(DetectedEntity(text[start:end], start, end, GENERATOR_TYPE.get(typ, typ),
                                  score, name))
    return out
