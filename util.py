# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
util.py — SurrogateShield Shared Utilities

Logging setup, the Conversation dataclasses and Rich console helpers.
DetectedEntity and the span helpers are re-exported from the package.

Note: logging.basicConfig() is intentionally NOT called here.
It is called once at startup in main.py. Modules that call
get_logger() before main.py initialises will use the root logger
with default settings (which is harmless for tests).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Entity type and span helpers live in the package (single implementation,
# audit F4); re-exported here for the app.
from surrogateshield.core.entities import (  # noqa: F401
    DetectedEntity,
    apply_entity_surrogates,
    mask_spans,
    plan_substitutions,
    remove_span_overlap,
    splice,
)

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
class ConversationMessage:
    """A single turn in a conversation."""
    role: str        # 'user' or 'assistant'
    content: str     # Final content (real values restored)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rag_mode: bool = False

    def to_api_history(self) -> list:
        """
        Return the sanitised API history for sending to Claude.

        Uses api_messages (surrogate values) — never the display messages.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [{"role": m.role, "content": m.content} for m in self.api_messages]


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