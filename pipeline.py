"""
pipeline.py — SurrogateShield Full Message Pipeline

Orchestrates the end-to-end message flow for every turn:

    User message
        │
        ▼
    [ServiceQueryDetector] — if service query, fuzz addresses minimally
        │
        ▼
    SentinelLayer (PatternScan → EntityTrace → ContextGuard)
    (PatternScan receives existing surrogate keys as skip_values to
     prevent re-detection of surrogates quoted back by the user)
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
    [Optional] Transparency panel — what was sent / received / restored
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
    get_logger,
    print_detection_table,
    print_needs_confirmation,
)
from detection import logic as sentinel_layer
from detection.service_query import is_service_query, fuzz_addresses
from generation.logic import MimicGen
from storage.logic import ShadowMap
from reconstruction.logic import ResolvePass
from chatbot.chat import ClaudeChat
from config import SERVICE_QUERY_DETECTION_ENABLED

if TYPE_CHECKING:
    from chatbot.rag import RAGStore

logger = get_logger(__name__)
_console = Console()


# ─────────────────────────────────────────────
# Transparency display
# ─────────────────────────────────────────────

def _show_transparency(
    sanitised: str,
    raw_response: str,
    restored: str,
) -> None:
    """
    Print a three-section panel showing the full API round-trip.

    Sent to Anthropic    — sanitised message (surrogates, no real PII)
    Received from Claude — raw API response  (still contains surrogates)
    Final output         — restored response (real values back)

    Args:
        sanitised:    Message actually sent to the API.
        raw_response: Unmodified API response text.
        restored:     Response after ResolvePass swap-back.
    """
    _console.print()
    _console.print(Rule("[dim]API Transparency[/dim]", style="dim blue"))

    # Trim long messages for display
    def _trim(s: str, n: int = 300) -> str:
        return s if len(s) <= n else s[:n] + f"… [dim]({len(s) - n} chars omitted)[/dim]"

    _console.print(
        Panel(
            f"[dim]Sent to Anthropic:[/dim]\n[blue]{_trim(sanitised)}[/blue]\n\n"
            f"[dim]Received from Claude:[/dim]\n[yellow]{_trim(raw_response)}[/yellow]\n\n"
            f"[dim]Final output (real values restored):[/dim]\n[green]{_trim(restored)}[/green]",
            border_style="dim blue",
            padding=(0, 2),
        )
    )
    _console.print()


# ─────────────────────────────────────────────
# Standalone anonymiser (no API key required)
# ─────────────────────────────────────────────

def anonymise_text(text: str, mimic: Optional[MimicGen] = None) -> Tuple[str, Dict[str, str]]:
    """
    Detect and replace PII in *text* without constructing a ClaudeChat.

    Used by add-doc to anonymise documents before indexing — no API key needed.

    Args:
        text:  Raw text that may contain PII.
        mimic: Existing MimicGen to reuse (for collision avoidance). If None,
               a fresh one is created.

    Returns:
        Tuple of (sanitised_text, {original: surrogate} mapping).
    """
    if mimic is None:
        mimic = MimicGen()
    confirmed, _ = sentinel_layer.run_cascade(text)
    confirmed = sentinel_layer.deduplicate(confirmed)
    surrogate_map = mimic.generate_all(confirmed) if confirmed else {}
    sanitised = text
    for orig in sorted(surrogate_map, key=len, reverse=True):
        sanitised = sanitised.replace(orig, surrogate_map[orig])
    return sanitised, surrogate_map


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-end SurrogateShield pipeline for a single conversation.

    Attributes:
        chat:    Claude API handler.
        shadow:  Encrypted surrogate↔original mapping store.
        mimic:   Surrogate generator (seeded from ShadowMap on resume).
        resolve: Response reconstructor.
        rag:     Optional RAG vector store.
    """

    def __init__(
        self,
        chat: ClaudeChat,
        rag=None,   # Optional[RAGStore] — lazy import avoids chromadb crash
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            chat: ClaudeChat instance (new or loaded conversation).
            rag:  Optional RAGStore. Import is lazy — missing chromadb will
                  never crash a standard non-RAG chat session.
        """
        self.chat   = chat
        self.shadow = ShadowMap(chat.conversation.id)
        self.resolve = ResolvePass()

        # Bug fix: seed MimicGen with surrogates already in the ShadowMap.
        # Without this, a resumed conversation can generate a duplicate surrogate
        # that maps to a different original value — silently corrupting PII.
        self.mimic = MimicGen()
        existing_surrogates = set(self.shadow.all_mappings().keys())
        if existing_surrogates:
            self.mimic.used_surrogates.update(existing_surrogates)
            logger.debug(
                f"[Pipeline] Seeded MimicGen with {len(existing_surrogates)} "
                "existing surrogates from ShadowMap"
            )

        # Lazy RAG: only validate import when rag is actually provided.
        if rag is not None:
            from chatbot.rag import RAGStore  # noqa: F401 — validates type at runtime
        self.rag = rag

    # ── Internal helpers ────────────────────────────────────────

    def _apply_surrogates(self, text: str, surrogate_map: Dict[str, str]) -> str:
        """Replace original PII values with surrogates. Longest-first to avoid substring conflicts."""
        result = text
        for original in sorted(surrogate_map, key=len, reverse=True):
            result = result.replace(original, surrogate_map[original])
        return result

    def _invert_map(self, surrogate_map: Dict[str, str]) -> Dict[str, str]:
        """Invert {original: surrogate} → {surrogate: original} for ShadowMap storage."""
        return {v: k for k, v in surrogate_map.items()}

    # ── Main turn handler ────────────────────────────────────────

    def process_turn(
        self,
        user_message: str,
        interactive: bool = True,
    ) -> Tuple[str, List[DetectedEntity], Dict[str, str]]:
        """
        Process a single conversation turn end-to-end.

        Args:
            user_message: Raw user message (may contain PII).
            interactive:  If True, prompt user to confirm borderline entities.

        Returns:
            Tuple of (restored_response, confirmed_entities, surrogate_map).
        """
        from config import SHOW_API_TRANSPARENCY

        # ── Service query check ────────────────────────────────
        # For messages that are service/knowledge queries (e.g. "restaurants near
        # 1126 E Apache Blvd"), apply minimal address fuzzing instead of full
        # surrogate replacement.  Full replacement would remove the street name
        # and city, making the LLM's answer useless.  Fuzzed addresses are stored
        # in the ShadowMap (fuzzed→original) so ResolvePass can restore them.
        service_addr_map: Dict[str, str] = {}
        if SERVICE_QUERY_DETECTION_ENABLED and is_service_query(user_message):
            from config import SERVICE_QUERY_VERIFY_ADDRESSES
            fuzzed_message, service_addr_map = fuzz_addresses(
                user_message, verify=SERVICE_QUERY_VERIFY_ADDRESSES
            )
            if service_addr_map:
                # Store fuzzed→original mappings so ResolvePass can restore them
                self.shadow.update({v: k for k, v in service_addr_map.items()})
                self.shadow.save()
                logger.info(
                    f"[Pipeline] Service query: fuzzed {len(service_addr_map)} address(es)"
                )
                # Continue pipeline with the fuzzed message
                user_message = fuzzed_message

        # ── Step 1: Detection ──────────────────────────────────
        # Pass existing surrogate keys to PatternScan so it doesn't re-detect
        # surrogate values a user quotes back in a follow-up message.  Without
        # this, a surrogate SSN/credit card would trigger a new detection and
        # generate a second surrogate on top of the first.
        logger.info("[Pipeline] Running SentinelLayer cascade")
        existing_surrogates = set(self.shadow.all_mappings().keys())
        confirmed, needs_confirmation = sentinel_layer.run_cascade(
            user_message, skip_values=existing_surrogates
        )

        # ── Step 2: User confirmation for borderlines ──────────
        if interactive and needs_confirmation:
            approved = print_needs_confirmation(needs_confirmation)
            confirmed.extend(approved)

        confirmed = sentinel_layer.deduplicate(confirmed)

        # ── Step 3: Generate surrogates ─────────────────────────
        surrogate_map: Dict[str, str] = {}
        if confirmed:
            surrogate_map = self.mimic.generate_all(confirmed)
            print_detection_table(confirmed, surrogate_map)
        else:
            logger.info("[Pipeline] No PII detected — message sent as-is")

        # ── Step 4: Sanitise message ────────────────────────────
        sanitised = self._apply_surrogates(user_message, surrogate_map)

        # ── Step 5: Update ShadowMap ────────────────────────────
        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()

        # ── Step 6: RAG context retrieval ──────────────────────
        if self.rag is not None:
            chunks = self.rag.query(sanitised)
            if chunks:
                context_prefix = self.rag.build_context_prompt(chunks)
                sanitised = context_prefix + sanitised
                logger.info(f"[Pipeline] RAG: prepended {len(chunks)} chunks")

        # ── Step 7: Send to Claude API ──────────────────────────
        logger.info("[Pipeline] Sending sanitised message to Claude API")
        raw_response = self.chat.send(sanitised)

        # ── Step 8: Reconstruct originals ──────────────────────
        all_mappings = self.shadow.all_mappings()
        restored_response = self.resolve.resolve(raw_response, all_mappings)

        self.chat.update_last_assistant_message(restored_response)
        self.chat.save()

        # ── Step 9: Transparency panel ──────────────────────────
        if SHOW_API_TRANSPARENCY:
            _show_transparency(
                sanitised=sanitised,
                raw_response=raw_response,
                restored=restored_response,
            )

        return restored_response, confirmed, surrogate_map

    # ── RAG document indexing ────────────────────────────────────

    def add_rag_document(
        self,
        raw_text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Anonymise and index a document into the RAG store.

        Uses the shared MimicGen and ShadowMap so surrogate→original
        mappings for indexed content are persisted across sessions via
        the global RAG ShadowMap.

        Args:
            raw_text: Document text that may contain PII.
            metadata: Optional metadata dict.

        Returns:
            Number of chunks indexed.
        """
        if self.rag is None:
            raise RuntimeError("RAG is not enabled. Use --rag flag.")

        sanitised, surrogate_map = anonymise_text(raw_text, mimic=self.mimic)

        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()

        return self.rag.add_document(sanitised, metadata=metadata)


# ─────────────────────────────────────────────
# Convenience: anonymise without any chat handler
# (used by main.py add-doc — no API key needed)
# ─────────────────────────────────────────────

def anonymise_for_rag(raw_text: str, rag_store) -> Tuple[int, ShadowMap]:
    """
    Anonymise *raw_text* and index it into *rag_store* without a ClaudeChat.

    This function exists so `add-doc` never requires ANTHROPIC_API_KEY —
    indexing is a purely local operation that only runs PatternScan + NER.

    Args:
        raw_text:  Raw document text.
        rag_store: Initialised RAGStore instance.

    Returns:
        Tuple of (n_chunks_indexed, rag_global_shadowmap).
    """
    mimic = MimicGen()
    rag_shadow = ShadowMap("rag_global")
    # Seed MimicGen from existing RAG ShadowMap to avoid collisions
    mimic.used_surrogates.update(rag_shadow.all_mappings().keys())

    sanitised, surrogate_map = anonymise_text(raw_text, mimic=mimic)

    if surrogate_map:
        inverted = {v: k for k, v in surrogate_map.items()}
        rag_shadow.update(inverted)
        rag_shadow.save()

    n = rag_store.add_document(sanitised)
    return n, rag_shadow