"""
surrogateshield/session.py — one conversation's masking state (audit I9, I10).

A :class:`Session` owns the surrogate map of one end-user conversation. Every
method takes the session's lock, so a session can be shared between threads
without corrupting it, and two sessions never see each other's values.

The module-level ``mask()`` / ``unmask()`` / ``scan()`` / ``flush()`` use the
*current* session: the one bound with :func:`use_session`, otherwise one
created on first use by the calling thread / asyncio task (never shared with
another thread or task). In a server, create one ``Session`` per conversation
(pass ``session_id`` to resume a persistent one) and use it directly or
through ``use_session``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import threading
import uuid
import weakref
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from . import _display, _response_parser
from ._state import Config, cfg as _default_config
from .core.detection import pipeline as _pipeline
from .core.detection import service_query as _service_query
from .core.entities import plan_substitutions, splice
from .core.generation.mimic import MimicGen
from .core.reconstruction.resolve import ResolvePass
from .core.storage.shadow_map import ShadowMap, validate_id


@dataclass(frozen=True)
class Detection:
    """One detected PII span. ``start``/``end`` index the scanned text."""
    text: str
    type: str
    start: int
    end: int
    score: float
    source: str
    masked: bool = True      # False when the type is switched off by pii_off


@dataclass(frozen=True)
class MaskResult:
    """What :meth:`Session.mask_result` did to one message."""
    text: str                         # the sanitized text to send
    detections: Tuple[Detection, ...]  # every detection, masked or not
    replacements: Dict[str, str]      # original → surrogate used in ``text``
    service_query: bool

    @property
    def unmasked(self) -> Tuple[Detection, ...]:
        """Detections sent verbatim because their type is in ``pii_off``."""
        return tuple(d for d in self.detections if not d.masked)


class Session:
    """Masking state for one conversation.

    Args:
        session_id:  Id of the conversation (``[A-Za-z0-9_-]{1,64}``). A new
                     random id when omitted. With persistent storage, the same
                     id reopens the same surrogate map.
        config:      Settings for this session. Defaults to a snapshot of the
                     module settings made by :func:`surrogateshield.config`.
        storage_dir: Directory for an encrypted persistent map. ``None``
                     (default) keeps the map in memory only; the default also
                     follows ``config.pii_mem``.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        config: Optional[Config] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        self.id = validate_id(session_id) if session_id is not None else uuid.uuid4().hex
        self.config = dataclasses.replace(config if config is not None else _default_config)
        if storage_dir is None and self.config.pii_mem != "temp":
            storage_dir = self.config.pii_mem
        self._lock = threading.RLock()
        self._shadow = ShadowMap(self.id, storage_dir=storage_dir)
        self._mimic = MimicGen()
        self._mimic.used_surrogates.update(self._shadow.get_all())
        self._resolver = ResolvePass()
        self._closed = False

    # ── detection ────────────────────────────────────────────────────────────

    def _detect(self, text: str):
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        c = self.config
        is_svc, address_mode = _service_query.resolve(text, c.address_mode, c.service)
        confirmed, _ = _pipeline.run_cascade(
            text=text,
            skip_values=None,
            skip_location_entities=is_svc,
            pii_off=None,                 # split below so scan() can report them
            spacy_model=c.spacy_model,
            context_guard_enabled=c.context_guard_enabled,
            entity_trace_high_threshold=c.entity_trace_high_threshold,
            entity_trace_low_threshold=c.entity_trace_low_threshold,
            context_guard_threshold=c.context_guard_threshold,
            entity_trace_fallback_threshold=c.entity_trace_fallback_threshold,
            context_guard_model=c.context_guard_model,
            context_guard_device=c.context_guard_device,
        )
        off = _pipeline.resolve_pii_off(c.pii_off)
        seen, detections, masked = set(), [], []
        for ent in sorted(confirmed, key=lambda e: (e.start, -e.end)):
            if (ent.start, ent.end) in seen:
                continue
            seen.add((ent.start, ent.end))
            d = Detection(ent.text, ent.type, ent.start, ent.end, float(ent.score),
                          ent.source, masked=ent.type not in off)
            detections.append(d)
            if d.masked:
                masked.append(ent)
        return (is_svc, address_mode), detections, masked

    def scan(self, text: str) -> List[Detection]:
        """Detect PII without masking it, with the same settings as
        :meth:`mask`. Detections of ``pii_off`` types have ``masked=False``.

        Raises:
            DetectorUnavailable: a detection model is missing or failed.
        """
        self._check_open()
        _, detections, _ = self._detect(text)
        if self.config.detailed_view:
            _display.show_scan_results(detections)
        return detections

    # ── masking ──────────────────────────────────────────────────────────────

    def mask_result(self, text: str) -> MaskResult:
        """Mask *text* and report what was replaced and what was not.

        Raises:
            DetectorUnavailable: a detection model is missing or failed; the
                text is not masked (fail closed, audit I17).
            TypeError: *text* is not a str.
        """
        (is_svc, address_mode), detections, masked = self._detect(text)   # models: outside the lock
        c = self.config
        with self._lock:
            self._check_open()
            if not masked:
                result = MaskResult(text, tuple(detections), {}, is_svc)
            else:
                result = self._substitute(text, is_svc, address_mode, detections, masked)
        if c.detailed_view:
            _display.show_mask_results(result)
        return result

    def _substitute(self, text, is_svc, address_mode, detections, masked) -> MaskResult:
        c = self.config
        unique = _pipeline.deduplicate(masked)
        replacements: Dict[str, str] = {}
        new_entities = []
        for ent in unique:
            key = ent.text.strip()
            existing = self._shadow.lookup_original(key)
            if existing is not None:
                replacements[key] = existing
            else:
                new_entities.append(ent)
        if new_entities:
            new_map = self._mimic.generate_all(
                new_entities,
                address_mode=address_mode,
                address_shift_range=c.address_shift_range,
                forbidden=set(self._shadow.originals()),
            )
            replacements.update(new_map)
            self._shadow.update({v: k for k, v in new_map.items()})
        edits = plan_substitutions(text, masked, replacements)
        return MaskResult(splice(text, edits), tuple(detections), replacements, is_svc)

    def mask(self, text: str) -> str:
        """Return *text* with PII replaced by surrogates (see :meth:`mask_result`)."""
        return self.mask_result(text).text

    # ── unmasking ────────────────────────────────────────────────────────────

    def unmask(self, response) -> str:
        """Restore the originals in an LLM response.

        *response* is a str or an Anthropic / OpenAI / Gemini response (object
        or dict). Raises ``TypeError`` for ``None`` or anything without text.
        """
        text = _response_parser.extract_text(response)
        with self._lock:
            self._check_open()
            mapping = self._shadow.get_all()
            before = len(self._resolver.failures)
            restored = self._resolver.resolve(
                response_text=text, shadow_map=mapping,
                fuzzy_threshold=self.config.fuzzy_threshold,
            )
            del self._resolver.failures[before:]
        if self.config.detailed_view:
            _display.show_unmask_results(sum(1 for s in mapping if s in text))
        return restored

    # ── lifecycle ────────────────────────────────────────────────────────────

    def forget(self, original: str) -> int:
        """Drop every mapping for *original* (per-value erasure). Returns the
        number of mappings removed; persisted immediately in persistent mode."""
        with self._lock:
            self._check_open()
            return self._shadow.forget(original)

    @property
    def mappings(self) -> Dict[str, str]:
        """A copy of the surrogate → original map."""
        with self._lock:
            return self._shadow.get_all()

    def close(self) -> None:
        """Discard every mapping (and the persistent file, if any)."""
        with self._lock:
            if not self._closed:
                self._shadow.flush()
                self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"session {self.id} is closed")

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<surrogateshield.Session id={self.id} mappings={len(self._shadow.get_all())}>"


# ── current session ──────────────────────────────────────────────────────────
#
# The context variable holds (session, owner). ``owner`` is None for a session
# bound explicitly with use_session() — child tasks and asyncio.to_thread()
# calls, which copy the context, use it on purpose. A session the module-level functions
# create implicitly is owned by the thread + asyncio task that created it: a
# context copied into another task or thread does NOT inherit it and gets its
# own, so a warm-up ``mask()`` at server start can never become one map shared
# by every request (audit I9).

_current: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar(
    "surrogateshield_session", default=None
)


def _owner() -> tuple:
    try:
        task = asyncio.current_task()
    except RuntimeError:                  # no running event loop
        task = None
    return threading.get_ident(), (weakref.ref(task) if task is not None else None)


def _same_owner(a: tuple, b: tuple) -> bool:
    if a[0] != b[0] or (a[1] is None) != (b[1] is None):
        return False
    if a[1] is None:
        return True
    task = a[1]()
    return task is not None and task is b[1]()


def current_session(*, create: bool = True) -> Optional[Session]:
    """The session the module-level functions use in this context: the one
    bound with :func:`use_session`, else the one this thread / asyncio task
    created implicitly (created now if *create*)."""
    bound = _current.get()
    if bound is not None:
        session, owner = bound
        if not session._closed and (owner is None or _same_owner(owner, _owner())):
            return session
    if not create:
        return None
    session = Session()
    _current.set((session, _owner()))
    return session


@contextlib.contextmanager
def use_session(session: Session) -> Iterator[Session]:
    """Bind *session* as the current session for the ``with`` block (and for
    asyncio tasks and ``asyncio.to_thread`` calls started inside it).
    ``loop.run_in_executor`` does not copy the context: pass the Session to
    the worker instead."""
    if not isinstance(session, Session):
        raise TypeError("use_session() needs a surrogateshield.Session")
    token = _current.set((session, None))
    try:
        yield session
    finally:
        _current.reset(token)
