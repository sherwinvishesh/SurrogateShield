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
    get_logger,
    print_detection_table,
    print_needs_confirmation,
)
from detection import logic as sentinel_layer
from detection.service_query import is_service_query, fuzz_addresses
from detection.quasi_identifier import format_warning as _qi_format_warning
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

def anonymise_text(text: str, mimic: Optional[MimicGen] = None) -> Tuple[str, Dict[str, str]]:
    """
    Detect and replace PII in *text* without constructing a ClaudeChat.

    Used by add-doc to anonymise documents before indexing.
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

        if rag is not None:
            from chatbot.rag import RAGStore  # noqa: F401
        self.rag = rag

    def _apply_surrogates(self, text: str, surrogate_map: Dict[str, str]) -> str:
        result = text
        for original in sorted(surrogate_map, key=len, reverse=True):
            result = result.replace(original, surrogate_map[original])
        return result

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

        # ── Service query check ───────────────────────────────────────────────
        # Service queries (e.g. "restaurants near 1126 E Apache Blvd, Tempe, AZ")
        # get minimal treatment:
        #   • Street address house number shifted by ±1 only
        #   • City/state names NOT replaced (preserves answer utility)
        #   • Other PII (names, emails, SSNs) still detected and replaced
        is_svc = SERVICE_QUERY_DETECTION_ENABLED and is_service_query(user_message)
        service_addr_map: Dict[str, str] = {}

        if is_svc:
            from config import SERVICE_QUERY_VERIFY_ADDRESSES
            fuzzed_message, service_addr_map = fuzz_addresses(
                user_message, verify=SERVICE_QUERY_VERIFY_ADDRESSES
            )
            if service_addr_map:
                self.shadow.update({v: k for k, v in service_addr_map.items()})
                self.shadow.save()
                logger.info(
                    f"[Pipeline] Service query: fuzzed {len(service_addr_map)} address(es)"
                )
                user_message = fuzzed_message

        # ── Step 1: Detection ─────────────────────────────────────────────────
        logger.info("[Pipeline] Running SentinelLayer cascade")
        existing_surrogates = set(self.shadow.all_mappings().keys())
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
            surrogate_map = self.mimic.generate_all(confirmed)
            if detailed:
                print_detection_table(confirmed, surrogate_map)
                qi_matches = getattr(confirmed, "_qi_matches", [])
                if qi_matches:
                    _console.print(f"[bold yellow]{_qi_format_warning(qi_matches)}[/bold yellow]")
                else:
                    _console.print("[dim green]✓  No quasi-identifier combination risk detected.[/dim green]")
        else:
            logger.info("[Pipeline] No PII detected — message sent as-is")

        # ── Step 4: Sanitise message ──────────────────────────────────────────
        sanitised = self._apply_surrogates(user_message, surrogate_map)

        # ── Step 5: Update ShadowMap ──────────────────────────────────────────
        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()

        # ── Step 6: RAG context retrieval ─────────────────────────────────────
        if self.rag is not None:
            chunks = self.rag.query(sanitised)
            if chunks:
                context_prefix = self.rag.build_context_prompt(chunks)
                sanitised = context_prefix + sanitised
                logger.info(f"[Pipeline] RAG: prepended {len(chunks)} chunks")

        # ── Step 7: Send to LLM API ──────────────────────────────────────────
        logger.info(f"[Pipeline] Sending sanitised message to {provider} API")
        raw_response = self.chat.send(sanitised)

        # ── Step 8: Reconstruct originals ─────────────────────────────────────
        all_mappings = self.shadow.all_mappings()
        restored_response = self.resolve.resolve(raw_response, all_mappings)

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

    def add_rag_document(self, raw_text: str, metadata: Optional[dict] = None) -> int:
        if self.rag is None:
            raise RuntimeError("RAG is not enabled. Use --rag flag.")
        sanitised, surrogate_map = anonymise_text(raw_text, mimic=self.mimic)
        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()
        return self.rag.add_document(sanitised, metadata=metadata)


# ─────────────────────────────────────────────
# Convenience: anonymise without any chat handler
# ─────────────────────────────────────────────

def anonymise_for_rag(raw_text: str, rag_store) -> Tuple[int, ShadowMap]:
    """Anonymise raw_text and index it without requiring ANTHROPIC_API_KEY."""
    mimic = MimicGen()
    rag_shadow = ShadowMap("rag_global")
    mimic.used_surrogates.update(rag_shadow.all_mappings().keys())

    sanitised, surrogate_map = anonymise_text(raw_text, mimic=mimic)

    if surrogate_map:
        inverted = {v: k for k, v in surrogate_map.items()}
        rag_shadow.update(inverted)
        rag_shadow.save()

    n = rag_store.add_document(sanitised)
    return n, rag_shadow