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
from ._state import Config, cfg as _default_config, effective_detection
from .core.detection import pipeline as _pipeline
from .core.consistency import assign_surrogates, quoted_back
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
        seed:        Seed for the surrogate generator. The same seed and the
                     same messages give the same surrogates (tests, audits);
                     ``None`` (default) draws a fresh seed from the OS.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        config: Optional[Config] = None,
        storage_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.id = validate_id(session_id) if session_id is not None else uuid.uuid4().hex
        self.config = dataclasses.replace(config if config is not None else _default_config)
        if storage_dir is None and self.config.pii_mem != "temp":
            storage_dir = self.config.pii_mem
        self._lock = threading.RLock()
        self._shadow = ShadowMap(self.id, storage_dir=storage_dir)
        self._mimic = MimicGen(seed)
        self._mimic.used_surrogates.update(self._shadow.get_all())
        for surrogate, original in self._shadow.get_all().items():
            self._mimic.people.learn(surrogate, original)    # a reopened session
        self._resolver = ResolvePass()
        self._last_surrogates: frozenset = frozenset()
        self._last_sent: Optional[str] = None
        self._closed = False

    # ── detection ────────────────────────────────────────────────────────────

    def _detect(self, text: str):
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        c = self.config
        det = self.detection_config
        is_svc, address_mode = _service_query.resolve(text, det.address_mode, det.service_queries)
        with self._lock:
            # Surrogates quoted back from earlier answers are not re-masked
            # (low-entropy ones such as "72" are values of their own, I5).
            known = quoted_back(self._shadow.get_all())
        # "keep" types are detected and reported, like pii_off, not dropped
        kept = {k: "replace" for k, a in det.type_actions.items() if a == "keep"}
        confirmed, _ = _pipeline.run_cascade(
            text=text,
            skip_values=known or None,
            skip_location_entities=is_svc,
            pii_off=None,                 # split below so scan() can report them
            config=det.with_actions(**kept) if kept else det,
        )
        off = _pipeline.resolve_pii_off(c.pii_off)
        seen, detections, masked = set(), [], []
        for ent in sorted(confirmed, key=lambda e: (e.start, -e.end)):
            if (ent.start, ent.end) in seen:
                continue
            seen.add((ent.start, ent.end))
            d = Detection(ent.text, ent.type, ent.start, ent.end, float(ent.score),
                          ent.source, masked=ent.type not in off and not (kept and det.keeps(ent.type)))
            detections.append(d)
            if d.masked:
                masked.append(ent)
        return (is_svc, address_mode, det), detections, masked

    @property
    def detection_config(self):
        """The :class:`DetectionConfig` this session detects with (its
        ``detection`` setting, or the environment's, or ``balanced``, with
        the flat settings that differ from their defaults on top)."""
        return effective_detection(self.config)

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
        (is_svc, address_mode, det), detections, masked = self._detect(text)   # models: outside the lock
        c = self.config
        with self._lock:
            self._check_open()
            if not masked:
                result = MaskResult(text, tuple(detections), {}, is_svc)
            else:
                result = self._substitute(text, is_svc, address_mode, detections, masked, det)
            self._last_surrogates = frozenset(result.replacements.values())
            self._last_sent = result.text
        if c.detailed_view:
            _display.show_mask_results(result)
        return result

    def _substitute(self, text, is_svc, address_mode, detections, masked, det) -> MaskResult:
        unique = _pipeline.deduplicate(masked, det)
        replacements = assign_surrogates(
            unique, text, self._mimic, [self._shadow],
            forbidden=set(self._shadow.originals()),
            address_mode=address_mode,
            address_shift_range=det.address_shift_range,
            redact=det.redacts if det.redacted_types() else None,
        )
        self._shadow.update({v: k for k, v in replacements.items()})
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

        Low-entropy surrogates (a bare age, a gender term, one or two
        characters) are restored only if the most recent :meth:`mask` sent
        them: a "72" in an answer about a later message is left alone. A
        bare first name or surname that the user typed unmasked in that
        message ("Peter the Great") is not taken for a surrogate's.
        """
        text = _response_parser.extract_text(response)
        with self._lock:
            self._check_open()
            mapping = self._shadow.get_all()
            before = len(self._resolver.failures)
            restored = self._resolver.resolve(
                response_text=text, shadow_map=mapping,
                fuzzy_threshold=self.config.fuzzy_threshold,
                current=self._last_surrogates,
                sent=self._last_sent,
            )
            del self._resolver.failures[before:]
        if self.config.detailed_view:
            _display.show_unmask_results(sum(1 for s in mapping if s in text))
        return restored

    # ── asyncio ──────────────────────────────────────────────────────────────
    #
    # Detection is CPU-bound model inference. The a* methods run the method of
    # the same name in a worker thread (asyncio.to_thread), so an event loop
    # keeps serving other requests meanwhile. Results and exceptions are the
    # same as the sync methods'. The worker copies the caller's context, so a
    # session bound with use_session() stays current inside it.

    async def ascan(self, text: str) -> List[Detection]:
        """:meth:`scan` in a worker thread."""
        return await asyncio.to_thread(self.scan, text)

    async def amask_result(self, text: str) -> MaskResult:
        """:meth:`mask_result` in a worker thread."""
        return await asyncio.to_thread(self.mask_result, text)

    async def amask(self, text: str) -> str:
        """:meth:`mask` in a worker thread."""
        return await asyncio.to_thread(self.mask, text)

    async def aunmask(self, response) -> str:
        """:meth:`unmask` in a worker thread."""
        return await asyncio.to_thread(self.unmask, response)

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

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *exc) -> None:
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
