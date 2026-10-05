# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
pipeline.py — SurrogateShield Full Message Pipeline

Orchestrates the end-to-end message flow for every turn:

    User message
        │
        ▼
    [ServiceQueryDetector]
        ├─ service query + street address → fuzz house number ±1, preserve city/state
        ├─ service query, no street addr  → send unchanged (location not PII here)
        └─ not a service query            → fall through to full cascade
        │
        ▼
    SentinelLayer (PatternScan → EntityTrace → ContextGuard)
        • service query mode: skip_location_entities=True
          (city/state names are NOT replaced so LLM can give useful local answers)
        • PatternScan receives existing surrogate keys as skip_values
          (prevents re-detection of surrogates quoted back by the user)
        │
        ▼
    MimicGen → generate surrogates
        │
        ▼
    Apply substitutions → sanitised_message
        │
        ▼
    ShadowMap.update({surrogate: original}) + save
        │
        ▼
    [Optional] RAG query → prepend context
        │
        ▼
    Claude API → raw_response (surrogates)
        │
        ▼
    ResolvePass → restored_response (originals back)
        │
        ▼
    [Optional] Transparency panel
        │
        ▼
    Display to user
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from util import (
    DetectedEntity,
    apply_entity_surrogates,
    get_logger,
    print_detection_table,
    print_needs_confirmation,
)
from detection import logic as sentinel_layer
from detection.service_query import resolve as resolve_service
from detection.quasi_identifier import format_warning as _qi_format_warning
from generation.logic import MimicGen
from storage.logic import RAG_STORE_ID, ShadowMap, surrogates_in_use
from reconstruction.logic import ResolvePass
from surrogateshield.core.consistency import assign_surrogates, quoted_back
from chatbot.chat import ClaudeChat
from config import (
    ADDRESS_MODE,
    ADDRESS_SHIFT_RANGE,
    SERVICE_QUERY_DETECTION_ENABLED,
)

if TYPE_CHECKING:
    from chatbot.rag import RAGStore

logger = get_logger(__name__)
_console = Console()


# ─────────────────────────────────────────────
# Transparency display
# ─────────────────────────────────────────────

def _show_transparency(sanitised: str, raw_response: str, restored: str, provider: str = "LLM") -> None:
    _console.print()
    _console.print(Rule("[dim]API Transparency[/dim]", style="dim blue"))

    def _trim(s: str, n: int = 300) -> str:
        return s if len(s) <= n else s[:n] + f"… [dim]({len(s) - n} chars omitted)[/dim]"

    _console.print(
        Panel(
            f"[dim]Sent to {provider}:[/dim]\n[blue]{_trim(sanitised)}[/blue]\n\n"
            f"[dim]Received from {provider}:[/dim]\n[yellow]{_trim(raw_response)}[/yellow]\n\n"
            f"[dim]Final output (real values restored):[/dim]\n[green]{_trim(restored)}[/green]",
            border_style="dim blue",
            padding=(0, 2),
        )
    )
    _console.print()


# ─────────────────────────────────────────────
# Standalone anonymiser (no API key required)
# ─────────────────────────────────────────────

def anonymise_text(text: str, mimic: Optional[MimicGen] = None, *,
                   shadow: Optional[ShadowMap] = None) -> Tuple[str, Dict[str, str]]:
    """
    Detect and replace PII in *text* without constructing a ClaudeChat.

    Used by add-doc to anonymise documents before indexing. With *shadow*,
    an original it already maps keeps its surrogate, and no new surrogate
    equals one of its originals.

    Raises:
        DetectorUnavailable: detection failed; nothing is returned.
    """
    if mimic is None:
        mimic = MimicGen()
    known = shadow.all_mappings() if shadow is not None else {}
    confirmed, _ = sentinel_layer.run_cascade(text, skip_values=quoted_back(known) or None)
    confirmed = sentinel_layer.deduplicate(confirmed)
    # Documents have no service-query context: "auto" resolves to replace.
    doc_address_mode = ADDRESS_MODE if ADDRESS_MODE != "auto" else "replace"
    surrogate_map = assign_surrogates(
        confirmed, text, mimic, [shadow],
        forbidden=set(shadow.originals()) if shadow is not None else None,
        address_mode=doc_address_mode,
        address_shift_range=ADDRESS_SHIFT_RANGE,
    )
    sanitised = apply_entity_surrogates(text, confirmed, surrogate_map)
    return sanitised, surrogate_map


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """End-to-end SurrogateShield pipeline for a single conversation."""

    def __init__(self, chat: ClaudeChat, rag=None) -> None:
        self.chat   = chat
        self.shadow = ShadowMap(chat.conversation.id)
        self.resolve = ResolvePass()

        self.mimic = MimicGen()
        existing_surrogates = set(self.shadow.all_mappings().keys())
        if existing_surrogates:
            self.mimic.used_surrogates.update(existing_surrogates)
            logger.debug(
                f"[Pipeline] Seeded MimicGen with {len(existing_surrogates)} "
                "existing surrogates from ShadowMap"
            )

        self.rag = rag

    def _rag_shadow(self) -> Optional[ShadowMap]:
        """The document map (re-read every turn: add-doc may run meanwhile)."""
        if self.rag is None:
            return None
        rag_shadow = ShadowMap(RAG_STORE_ID)
        self.mimic.used_surrogates.update(rag_shadow.all_mappings().keys())
        return rag_shadow

    def _invert_map(self, surrogate_map: Dict[str, str]) -> Dict[str, str]:
        return {v: k for k, v in surrogate_map.items()}

    def process_turn(
        self,
        user_message: str,
        interactive: bool = True,
    ) -> Tuple[str, List[DetectedEntity], Dict[str, str]]:
        """
        Process a single conversation turn end-to-end.

        Returns:
            Tuple of (restored_response, confirmed_entities, surrogate_map).
        """
        from config import SHOW_API_TRANSPARENCY
        from settings_manager import load_settings
        _s = load_settings()
        detailed = _s.get("detailed_view", False)
        _PROVIDER_NAMES = {
            "claude":  "Claude",
            "gemini":  "Gemini",
            "chatgpt": "ChatGPT",
            "local":   "Local LLM",
        }
        provider = _PROVIDER_NAMES.get(_s.get("llm_provider", "claude"), "LLM")

        # ── Service query check / address-mode resolution ─────────────────────
        # Service queries (e.g. "restaurants near 1126 E Apache Blvd, Tempe, AZ")
        # get minimal treatment:
        #   • City/state names NOT replaced (preserves answer utility)
        #   • Other PII (names, emails, SSNs) still detected and replaced
        # Addresses flow through the normal detect→generate path in every mode;
        # ADDRESS_MODE decides shift vs replace ("auto" = shift only for
        # non-sensitive service queries — the v1 behaviour).
        is_svc, address_mode = resolve_service(user_message, ADDRESS_MODE,
                                               SERVICE_QUERY_DETECTION_ENABLED)

        # ── Step 1: Detection ─────────────────────────────────────────────────
        logger.info("[Pipeline] Running SentinelLayer cascade")
        # In RAG mode the document map is shared with this conversation
        # (audit I16): a name the documents already contain gets the same
        # surrogate here, so retrieval matches and quoted excerpts restore.
        rag_shadow = self._rag_shadow()
        rag_map = rag_shadow.all_mappings() if rag_shadow is not None else {}
        # Surrogates quoted back from earlier answers are not re-detected;
        # low-entropy ones ("72", "female") are values of their own (I5).
        existing_surrogates = quoted_back(set(self.shadow.all_mappings()) | set(rag_map))
        confirmed, needs_confirmation = sentinel_layer.run_cascade(
            user_message,
            skip_values=existing_surrogates,
            # In service-query mode, suppress GPE/LOC/FAC so city/state names
            # are NOT replaced — they're needed for the LLM to give useful answers.
            skip_location_entities=is_svc,
        )

        # ── Step 2: User confirmation for borderlines ─────────────────────────
        if interactive and needs_confirmation:
            approved = print_needs_confirmation(needs_confirmation)
            confirmed.extend(approved)

        confirmed = sentinel_layer.deduplicate(confirmed)

        # ── Step 3: Generate surrogates ───────────────────────────────────────
        surrogate_map: Dict[str, str] = {}
        if confirmed:
            # An original seen in an earlier turn (or in the documents) keeps
            # its surrogate, case-insensitively (I3); a new surrogate never
            # equals a real value from any turn nor text already in the message.
            surrogate_map = assign_surrogates(
                confirmed, user_message, self.mimic, [self.shadow, rag_shadow],
                forbidden=set(self.shadow.originals()) | set(rag_map.values()),
                address_mode=address_mode,
                address_shift_range=ADDRESS_SHIFT_RANGE,
            )

            if detailed:
                print_detection_table(confirmed, surrogate_map)
                qi_matches = getattr(confirmed, "_qi_matches", [])
                if qi_matches:
                    _console.print(f"[bold yellow]{_qi_format_warning(qi_matches)}[/bold yellow]")
                else:
                    _console.print("[dim green]✓  No quasi-identifier combination risk detected.[/dim green]")
        else:
            logger.info("[Pipeline] No PII detected — message sent as-is")

        # ── Step 4: Sanitise message (span-safe substitution) ─────────────────
        sanitised = apply_entity_surrogates(user_message, confirmed, surrogate_map)

        # ── Step 5: Update ShadowMap ──────────────────────────────────────────
        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()

        # ── Step 6: RAG context retrieval ─────────────────────────────────────
        # The context goes into this request only, never into the history
        # (audit I18).
        context_prefix = ""
        if self.rag is not None:
            chunks = self.rag.query(sanitised)
            if chunks:
                context_prefix = self.rag.build_context_prompt(chunks)
                logger.info(f"[Pipeline] RAG: prepended {len(chunks)} chunks")

        # ── Step 7: Send to LLM API ──────────────────────────────────────────
        logger.info(f"[Pipeline] Sending sanitised message to {provider} API")
        raw_response = self.chat.send(sanitised, display_message=user_message,
                                      context_prefix=context_prefix)

        # ── Step 8: Reconstruct originals ─────────────────────────────────────
        all_mappings = {**rag_map, **self.shadow.all_mappings()}   # conversation pairs win
        restored_response = self.resolve.resolve(raw_response, all_mappings,
                                                  current=set(surrogate_map.values()),
                                                  sent=sanitised)

        self.chat.update_last_assistant_message(restored_response)
        self.chat.save()

        # ── Step 9: Transparency panel ────────────────────────────────────────
        if SHOW_API_TRANSPARENCY and detailed:
            _show_transparency(
                sanitised=sanitised,
                raw_response=raw_response,
                restored=restored_response,
                provider=provider,
            )

        return restored_response, confirmed, surrogate_map


# ─────────────────────────────────────────────
# Convenience: anonymise without any chat handler
# ─────────────────────────────────────────────

DOC_SEGMENT_CHARS = 20_000   # detection unit for documents (spaCy limit is 1,000,000)


def split_segments(text: str, max_chars: int = DOC_SEGMENT_CHARS) -> List[str]:
    """Split *text* into pieces of at most *max_chars* that join back to
    *text* exactly, cutting at a paragraph break, else a line break, else a
    space when one is available."""
    pieces, start = [], 0
    while len(text) - start > max_chars:
        window = text[start:start + max_chars]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        cut = cut + 1 if cut > 0 else max_chars
        pieces.append(text[start:start + cut])
        start += cut
    pieces.append(text[start:])
    return pieces


def anonymise_for_rag(raw_text: str, rag_store, *, source: Optional[str] = None,
                      mimic: Optional[MimicGen] = None) -> Tuple[int, ShadowMap]:
    """Anonymise *raw_text* and index it (no provider key needed).

    The document is detected in segments of at most ``DOC_SEGMENT_CHARS``;
    a detection failure in any segment raises before anything is stored
    (fail closed, audit I16/I17). Pairs go to the shared ``rag_global`` map:
    an original already seen in another document keeps its surrogate, and
    a new surrogate never equals one issued by any conversation.

    Returns:
        (chunks indexed, the rag_global map).
    """
    rag_shadow = ShadowMap(RAG_STORE_ID)
    mimic = mimic or MimicGen()
    mimic.used_surrogates.update(rag_shadow.all_mappings().keys())
    mimic.used_surrogates.update(surrogates_in_use())

    parts, new_pairs = [], {}
    for segment in split_segments(raw_text):
        sanitised, surrogate_map = anonymise_text(segment, mimic=mimic, shadow=rag_shadow)
        parts.append(sanitised)
        pairs = {v: k for k, v in surrogate_map.items()}
        rag_shadow.update(pairs)          # later segments reuse these (in memory)
        new_pairs.update(pairs)

    if new_pairs:
        rag_shadow.save()
    n = rag_store.add_document("".join(parts),
                               metadata={"source": source} if source else None)
    return n, rag_shadow


def forget_rag_document(doc: str, rag_store) -> Tuple[int, int]:
    """Remove *doc* (doc_id or source name) from the index, then erase every
    ``rag_global`` pair whose surrogate no remaining chunk contains.

    Returns:
        (chunks removed, mappings erased).
    """
    removed = rag_store.forget(doc)
    if not removed:
        return 0, 0
    remaining = "\n".join(rag_store.texts())
    rag_shadow = ShadowMap(RAG_STORE_ID)
    by_original: Dict[str, List[str]] = {}
    for surrogate, original in rag_shadow.all_mappings().items():
        by_original.setdefault(original, []).append(surrogate)
    erased = sum(rag_shadow.forget(o) for o, surrogates in by_original.items()
                 if not any(s in remaining for s in surrogates))
    return removed, erased
