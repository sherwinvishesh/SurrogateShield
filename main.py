"""
main.py — SurrogateShield CLI Entry Point

Typer-based terminal interface for SurrogateShield.

Commands:
    chat              — Start a new conversation
    chat --load ID    — Continue an existing conversation
    chat --delete ID  — Permanently delete a conversation
    chat --rag        — Start a new conversation with RAG mode
    list              — List all saved conversations
    add-doc FILEPATH  — Anonymise and index a document into the RAG store
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    name="surrogateshield",
    help="Privacy-preserving CLI proxy for Claude — PII never leaves your device.",
    add_completion=False,
)
console = Console()

# ─────────────────────────────────────────────
# Shared RAG store (lazy, one per process)
# ─────────────────────────────────────────────

_rag_store = None


def _get_rag() -> "RAGStore":
    """Lazily initialise the shared RAG store."""
    global _rag_store
    if _rag_store is None:
        from chatbot.rag import RAGStore
        _rag_store = RAGStore()
    return _rag_store


# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────

def _print_banner() -> None:
    """Print the SurrogateShield startup banner."""
    console.print(
        Panel.fit(
            "[bold cyan]SurrogateShield[/bold cyan] [dim]v1.0[/dim]\n"
            "[dim]Privacy-preserving Claude proxy · PII stays on your device[/dim]",
            border_style="cyan",
        )
    )


# ─────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────

def _run_chat_loop(pipeline: "Pipeline", rag_mode: bool) -> None:
    """
    Run the interactive chat REPL.

    Reads user input, processes each turn through the pipeline,
    and prints the restored response.

    Args:
        pipeline: Initialised Pipeline instance.
        rag_mode: Whether RAG mode is active (shown in prompt).
    """
    from pipeline import Pipeline  # local import avoids circular at module level

    mode_label = "[bold magenta]RAG[/bold magenta] " if rag_mode else ""
    conv_id = pipeline.chat.conversation.id
    console.print(
        f"\n[dim]Conversation ID:[/dim] [cyan]{conv_id}[/cyan]"
    )
    console.print(
        "[dim]Type[/dim] [bold]exit[/bold] [dim]or[/dim] [bold]quit[/bold] "
        "[dim]to end the session.[/dim]\n"
    )

    while True:
        try:
            user_input = console.input(
                f"[bold green]{mode_label}You:[/bold green] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            response, entities, _ = pipeline.process_turn(
                user_input, interactive=True
            )
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] {exc}")
            console.print("[dim]Check your ANTHROPIC_API_KEY and connection.[/dim]")
            continue

        console.print(
            Panel(
                response,
                title="[bold blue]Claude[/bold blue]",
                border_style="blue",
            )
        )


# ─────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────

@app.command()
def chat(
    load: Optional[str] = typer.Option(
        None,
        "--load",
        help="Conversation ID to continue.",
        metavar="CONV_ID",
    ),
    delete: Optional[str] = typer.Option(
        None,
        "--delete",
        help="Permanently delete a conversation and its ShadowMap.",
        metavar="CONV_ID",
    ),
    rag: bool = typer.Option(
        False,
        "--rag",
        help="Enable RAG mode (requires indexed documents).",
    ),
) -> None:
    """
    Start or continue a conversation with Claude via SurrogateShield.

    PII is detected, replaced with realistic surrogates, and restored
    in Claude's response. Nothing sensitive is ever sent to the API.
    """
    from chatbot.chat import ClaudeChat
    from pipeline import Pipeline

    _print_banner()

    # ── Delete mode ────────────────────────────────────────────
    if delete:
        from storage.logic import ShadowMap
        console.print(f"[yellow]Deleting conversation:[/yellow] {delete}")
        ClaudeChat.delete(delete)
        sm = ShadowMap(delete)
        sm.delete()
        console.print("[green]✓[/green] Conversation and ShadowMap deleted.")
        return

    # ── Load existing or create new ────────────────────────────
    try:
        if load:
            chat_handler = ClaudeChat.load(load)
            console.print(
                f"[green]Resuming conversation[/green] [cyan]{load}[/cyan] "
                f"([dim]{len(chat_handler.conversation.messages)} messages[/dim])"
            )
        else:
            chat_handler = ClaudeChat()
            chat_handler.conversation.rag_mode = rag
            console.print("[green]New conversation started.[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    except EnvironmentError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        raise typer.Exit(1)

    # ── Build pipeline ──────────────────────────────────────────
    rag_store = None
    effective_rag = rag or (load and chat_handler.conversation.rag_mode)
    if effective_rag:
        try:
            rag_store = _get_rag()
            console.print(
                f"[magenta]RAG mode enabled[/magenta] "
                f"([dim]{rag_store.document_count()} chunks indexed[/dim])"
            )
        except Exception as exc:
            console.print(
                f"[yellow]Warning:[/yellow] RAG unavailable: {exc}. "
                "Continuing without RAG."
            )
            rag_store = None

    pipeline = Pipeline(chat=chat_handler, rag=rag_store)

    # ── Start chat loop ─────────────────────────────────────────
    _run_chat_loop(pipeline, rag_mode=bool(effective_rag))


@app.command(name="list")
def list_conversations() -> None:
    """List all saved conversations."""
    from chatbot.chat import ClaudeChat

    conversations = ClaudeChat.list_conversations()
    if not conversations:
        console.print("[dim]No saved conversations found.[/dim]")
        return

    table = Table(
        title="[bold cyan]Saved Conversations[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Created", style="dim")
    table.add_column("Messages", justify="right")
    table.add_column("Mode", style="magenta")

    for conv in conversations:
        mode = "RAG" if conv.get("rag_mode") else "standard"
        table.add_row(
            conv["id"],
            conv["created"][:19].replace("T", " "),
            str(conv["message_count"]),
            mode,
        )

    console.print(table)


@app.command(name="add-doc")
def add_document(
    filepath: str = typer.Argument(
        ..., help="Path to the document to index into the RAG store."
    )
) -> None:
    """
    Anonymise and index a document into the RAG vector store.

    The document is processed through the SurrogateShield pipeline
    before embedding — no raw PII is stored in the vector index.
    """
    from chatbot.chat import ClaudeChat
    from pipeline import Pipeline

    _print_banner()

    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {filepath}")
        raise typer.Exit(1)

    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Could not read file:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[cyan]Indexing document:[/cyan] {path.name} "
        f"([dim]{len(raw_text):,} characters[/dim])"
    )

    try:
        rag_store = _get_rag()
        # Use a temporary pipeline for anonymisation (no API key needed here)
        # We create a disposable chat handler — it won't make API calls
        chat_handler = ClaudeChat()
        pipeline = Pipeline(chat=chat_handler, rag=rag_store)
        n_chunks = pipeline.add_rag_document(
            raw_text,
            metadata={"source": str(path), "filename": path.name},
        )
        console.print(
            f"[green]✓[/green] Indexed [bold]{n_chunks}[/bold] chunks from "
            f"[cyan]{path.name}[/cyan]"
        )
    except EnvironmentError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app()
