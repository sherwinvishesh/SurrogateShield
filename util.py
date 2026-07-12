# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
util.py — SurrogateShield Shared Utilities

Logging setup, shared dataclasses (DetectedEntity, Conversation),
text utilities, and Rich console helpers used across the project.

Note: logging.basicConfig() is intentionally NOT called here.
It is called once at startup in main.py. Modules that call
get_logger() before main.py initialises will use the root logger
with default settings (which is harmless for tests).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

# ─────────────────────────────────────────────
# Rich console — shared singleton
# ─────────────────────────────────────────────

console = Console()


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Logging configuration (handlers, level, format) is set up once
    in main.py at startup via logging.basicConfig(). This function
    simply returns the named logger — it does NOT call basicConfig()
    itself, avoiding the anti-pattern of re-configuring the root
    logger on every module import.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


# ─────────────────────────────────────────────
# Core dataclasses
# ─────────────────────────────────────────────

@dataclass
class DetectedEntity:
    """
    Represents a single piece of detected PII.

    Attributes:
        text:   The original text snippet detected as PII.
        start:  Character start index in the source string.
        end:    Character end index in the source string.
        type:   PII type label (e.g. 'email', 'PERSON', 'implicit_location').
        score:  Confidence score in [0.0, 1.0]. PatternScan always yields 1.0.
        source: Which detector produced this entity ('pattern', 'ner', 'slm').
        parsed: Structured payload for entities that carry one (addresses
                carry a ParsedAddress from the canonical address parser).
                Optional and unused by all non-address code paths.
    """
    text: str
    start: int
    end: int
    type: str
    score: float = 1.0
    source: str = "pattern"
    parsed: Optional[object] = field(default=None, compare=False)

    def overlaps(self, other: "DetectedEntity") -> bool:
        """Return True if this entity's span overlaps with another entity's span."""
        return not (self.end <= other.start or self.start >= other.end)


@dataclass
class ConversationMessage:
    """A single turn in a conversation."""
    role: str        # 'user' or 'assistant'
    content: str     # Final content (real values restored)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Conversation:
    """
    Full conversation state including history and metadata.

    Two separate message lists are maintained:
        messages     — display history with REAL values restored (shown to user,
                       persisted for human readability)
        api_messages — API history with SURROGATE values only (sent to Claude,
                       never contains real PII)

    This separation is the core privacy guarantee for multi-turn conversations.

    Attributes:
        id:           Unique conversation identifier (UUID).
        messages:     Display history (real values, for the user).
        api_messages: API history (surrogates only, sent to Claude).
        created:      ISO timestamp of conversation creation.
        rag_mode:     Whether RAG mode is active for this conversation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ConversationMessage] = field(default_factory=list)
    api_messages: List[ConversationMessage] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    rag_mode: bool = False

    def to_api_history(self) -> list:
        """
        Return the sanitised API history for sending to Claude.

        Uses api_messages (surrogate values) — never the display messages.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [{"role": m.role, "content": m.content} for m in self.api_messages]


# ─────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────

def mask_spans(text: str, entities: List[DetectedEntity], placeholder: str = "█") -> str:
    """
    Replace all entity spans in *text* with a placeholder character.

    Used to remove already-detected spans from the text before passing
    the remainder to the next detection stage.

    Args:
        text:        Original text string.
        entities:    Entities whose spans should be masked.
        placeholder: Single character to fill masked spans.

    Returns:
        New string with entity spans replaced by placeholder characters.
    """
    if not entities:
        return text
    chars = list(text)
    for ent in entities:
        for i in range(ent.start, min(ent.end, len(chars))):
            chars[i] = placeholder
    return "".join(chars)


def remove_span_overlap(candidate: DetectedEntity, existing: List[DetectedEntity]) -> bool:
    """
    Return True if *candidate* overlaps with any entity in *existing*.

    Used by EntityTrace to skip spans already covered by PatternScan.

    Args:
        candidate: The entity being tested.
        existing:  Already-confirmed entities.

    Returns:
        True if there is an overlap (candidate should be skipped).
    """
    return any(candidate.overlaps(e) for e in existing)


def apply_entity_surrogates(
    text: str,
    entities: List[DetectedEntity],
    mapping: Dict[str, str],
) -> str:
    """
    Replace each detected entity with its surrogate, span-safely.

    Two stages:
      1. Splice surrogates at the exact (start, end) offsets, right-to-left
         so earlier offsets stay valid.  This guarantees a surrogate for one
         entity can never rewrite text inside another entity's span (e.g. a
         standalone "Tempe" surrogate must not touch the "Tempe" inside a
         shift-mode address that is deliberately preserved).
      2. A word-boundary pass replaces any ADDITIONAL occurrences of each
         original outside the already-claimed spans, longest-original first
         (preserves recall for repeated values the cascade only saw once).
    """
    if not mapping:
        return text

    # Stage 1 — span splice (right-to-left, skip overlaps defensively)
    spliceable = sorted(
        (e for e in entities if e.text in mapping and 0 <= e.start < e.end <= len(text)),
        key=lambda e: e.start,
        reverse=True,
    )
    claimed_end = len(text) + 1
    result = text
    for ent in spliceable:
        if ent.end > claimed_end:
            continue  # overlaps an entity already spliced
        if result[ent.start:ent.end] != ent.text:
            continue  # offsets no longer line up with the text — skip
        result = result[:ent.start] + mapping[ent.text] + result[ent.end:]
        claimed_end = ent.start

    # Stage 2 — remaining occurrences of each original, outside surrogates.
    # Longest originals first so substrings never clobber longer values.
    for original in sorted(mapping, key=len, reverse=True):
        if original in result:
            pattern = re.compile(
                r"(?<![\w])" + re.escape(original) + r"(?![\w])"
            )
            surrogate = mapping[original]
            result = pattern.sub(lambda _m: surrogate, result)

    return result


def new_conversation_id() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# Rich display helpers
# ─────────────────────────────────────────────

def print_detection_table(
    confirmed: List[DetectedEntity],
    surrogate_map: dict,
) -> None:
    """
    Print a Rich table showing detected PII and their surrogates.

    Args:
        confirmed:     List of confirmed DetectedEntity objects.
        surrogate_map: Dict mapping original text → surrogate text.
    """
    if not confirmed:
        return

    table = Table(title="[bold cyan]SentinelLayer — PII Detected[/bold cyan]", show_lines=True)
    table.add_column("Original", style="red bold")
    table.add_column("Type", style="yellow")
    table.add_column("Score", style="white")
    table.add_column("Source", style="dim")
    table.add_column("Surrogate", style="green bold")

    for ent in confirmed:
        surrogate = surrogate_map.get(ent.text, "[dim]no surrogate[/dim]")
        table.add_row(
            ent.text,
            ent.type,
            f"{ent.score:.2f}",
            ent.source,
            surrogate,
        )

    console.print(table)


def print_needs_confirmation(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """
    Prompt the user to confirm whether each borderline entity should be replaced.

    Prints each entity and reads 'y'/'n' input. Returns only confirmed entities.

    Args:
        entities: Entities that were below the auto-replace threshold.

    Returns:
        Subset of entities the user approved for replacement.
    """
    approved = []
    if not entities:
        return approved

    console.print("\n[bold yellow]⚠  Some entities need your confirmation:[/bold yellow]")
    for ent in entities:
        console.print(
            f"  • [yellow]{ent.text!r}[/yellow] "
            f"([dim]{ent.type}, score={ent.score:.2f}[/dim])"
        )
        answer = console.input("    Replace this? [y/N] ").strip().lower()
        if answer == "y":
            approved.append(ent)

    return approved