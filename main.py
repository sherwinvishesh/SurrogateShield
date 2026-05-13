"""
main.py — SurrogateShield CLI

Run with no arguments for the interactive dashboard:
    python main.py

Or use direct commands:
    python main.py chat                  — new conversation
    python main.py chat --load <id>      — continue conversation
    python main.py chat --rag            — new conversation with RAG
    python main.py list                  — list conversations
    python main.py add-doc <filepath>    — index a document into RAG
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

app = typer.Typer(
    name="surrogateshield",
    help="Privacy-preserving CLI proxy for Claude — PII never leaves your device.",
    add_completion=False,
)

_rag_store = None

VERSION = "v1.0"
TAGLINE = "Privacy-preserving proxy for Claude  ·  PII never leaves your device"

LOGO = """\
  ██████  █    ██  ██████  ██████   ██████   ██████   █████  ████████ ███████
 ██      ██    ██ ██   ██ ██   ██ ██    ██ ██       ██   ██    ██    ██
  █████  ██    ██ ██████  ██████  ██    ██ ██   ███ ███████    ██    █████
      ██ ██    ██ ██   ██ ██   ██ ██    ██ ██    ██ ██   ██    ██    ██
 ██████   ██████  ██   ██ ██   ██  ██████   ██████  ██   ██    ██    ███████
                         ███████ ██   ██ ██ ███████ ██      ██████
                         ██      ██   ██ ██ ██      ██      ██   ██
                         ███████ ███████ ██ █████   ██      ██   ██
                              ██ ██   ██ ██ ██      ██      ██   ██
                         ███████ ██   ██ ██ ███████ ███████ ██████"""


# ─────────────────────────────────────────────────────────────────────────────
# Banners
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Full dashboard banner with logo."""
    console.print()
    console.print(Align.center(Text(LOGO, style="bold blue")))
    console.print()
    console.print(Align.center(Text(f"{TAGLINE}  [{VERSION}]", style="dim cyan")))
    console.print()


def _print_compact_banner() -> None:
    """Single-line banner for chat/command mode."""
    console.print(
        Panel.fit(
            f"[bold blue]SurrogateShield[/bold blue] [dim]{VERSION}[/dim]"
            "  [dim]·[/dim]  [dim cyan]{tagline}[/dim cyan]".format(
                tagline="PII never leaves your device"
            ),
            border_style="blue",
            padding=(0, 2),
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# How it works section
# ─────────────────────────────────────────────────────────────────────────────

def _print_how_it_works() -> None:
    """Six-panel pipeline overview."""
    steps = [
        ("1  PatternScan",   "Regex — SSNs, emails,\nphones, cards, API keys"),
        ("2  EntityTrace",   "spaCy NER — names,\nplaces, organisations"),
        ("3  MimicGen",      "Realistic fake values\nper PII type (Faker)"),
        ("4  ShadowMap",     "AES-256-GCM encrypted\nmap — stays on device"),
        ("5  Claude API",    "Receives surrogates\nnever real values"),
        ("6  ResolvePass",   "Swaps fakes back to\nreal values in response"),
    ]
    panels = []
    for title, body in steps:
        panels.append(
            Panel(
                f"[bold blue]{title}[/bold blue]\n[dim]{body}[/dim]",
                border_style="blue",
                padding=(0, 1),
                expand=True,
            )
        )
    console.print(Rule("[blue]How It Works[/blue]", style="blue"))
    console.print()
    console.print(Columns(panels, equal=True, expand=True))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Conversations table
# ─────────────────────────────────────────────────────────────────────────────

def _relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to human-readable relative time."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = datetime.utcnow() - dt.replace(tzinfo=None)
        s = int(diff.total_seconds())
        if s < 60:   return "just now"
        if s < 3600: return f"{s // 60}m ago"
        if s < 86400: return f"{s // 3600}h ago"
        if s < 604800: return f"{diff.days}d ago"
        return dt.strftime("%b %d %Y")
    except Exception:
        return iso_str[:10] if len(iso_str) >= 10 else "—"


def _print_conversations_table(conversations: list) -> None:
    """Render conversations as a numbered Rich table."""
    if not conversations:
        console.print(
            Panel(
                "[dim]No saved conversations yet.\n\n"
                "Press [bold white]N[/bold white] to start one.[/dim]",
                border_style="blue",
                title="[blue]Conversations[/blue]",
                padding=(1, 4),
            )
        )
        return

    table = Table(
        title="[bold blue]Conversations[/bold blue]",
        box=box.ROUNDED,
        border_style="blue",
        show_lines=True,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("#",          style="bold blue",  justify="right",  width=4)
    table.add_column("ID",         style="cyan",       no_wrap=True)
    table.add_column("Created",    style="dim white",  width=12)
    table.add_column("Turns",      justify="right",    style="white",    width=6)
    table.add_column("Mode",       style="blue",       width=9)

    for i, conv in enumerate(conversations, 1):
        uid      = conv["id"]
        turns    = conv["message_count"] // 2
        mode_str = "[magenta]RAG[/magenta]" if conv.get("rag_mode") else "standard"
        table.add_row(
            str(i),
            uid,
            _relative_time(conv.get("created", "")),
            str(turns),
            mode_str,
        )
    console.print(table)


def _print_menu(has_convs: bool) -> None:
    """Action menu below the conversations table."""
    console.print()
    console.print(Rule("[blue]Actions[/blue]", style="blue"))
    console.print()

    rows = [
        ("[bold blue]N[/bold blue]",       "New conversation"),
        ("[bold blue]R[/bold blue]",       "New conversation + RAG mode"),
    ]
    if has_convs:
        rows += [
            ("[bold blue]1 – 9[/bold blue]",   "Open conversation by number"),
            ("[bold blue]D1 – D9[/bold blue]", "Delete conversation by number"),
        ]
    rows.append(("[bold blue]Q[/bold blue]", "Quit"))

    for key, desc in rows:
        console.print(f"  {key}    [dim]{desc}[/dim]")

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_dashboard() -> None:
    """Full interactive dashboard — entry point when no subcommand is given."""
    from chatbot.chat import ClaudeChat

    while True:
        console.clear()
        _print_banner()
        _print_how_it_works()

        conversations = ClaudeChat.list_conversations()
        _print_conversations_table(conversations)
        _print_menu(has_convs=bool(conversations))

        try:
            raw = console.input("[bold blue]›[/bold blue]  ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            sys.exit(0)

        if not raw:
            continue

        upper = raw.upper()

        # Quit
        if upper == "Q":
            console.print("\n[dim]Goodbye.[/dim]")
            sys.exit(0)

        # New chat
        if upper == "N":
            console.clear()
            _print_compact_banner()
            _start_chat(rag=False)
            continue

        # New RAG chat
        if upper == "R":
            console.clear()
            _print_compact_banner()
            _start_chat(rag=True)
            continue

        # Delete  D<n>
        if upper.startswith("D") and upper[1:].isdigit():
            idx = int(upper[1:]) - 1
            if 0 <= idx < len(conversations):
                uid = conversations[idx]["id"]
                short = f"{uid[:8]}…{uid[-4:]}"
                console.print(
                    f"\n[yellow]Delete [bold]{short}[/bold]?[/yellow] "
                    "[dim](y / N)[/dim] ",
                    end="",
                )
                try:
                    confirm = console.input("").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    continue
                if confirm == "y":
                    _delete_conversation(uid)
                    console.print("[green]✓[/green]  Deleted.")
                    time.sleep(0.7)
                else:
                    console.print("[dim]Cancelled.[/dim]")
                    time.sleep(0.4)
            else:
                console.print(f"[red]No conversation #{idx + 1}[/red]")
                time.sleep(0.6)
            continue

        # Open  <n>
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(conversations):
                uid = conversations[idx]["id"]
                console.clear()
                _print_compact_banner()
                _start_chat(load=uid)
            else:
                console.print(f"[red]No conversation #{idx + 1}[/red]")
                time.sleep(0.6)
            continue

        console.print(f"[red]Unknown:[/red] {raw!r}")
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_rag():
    global _rag_store
    if _rag_store is None:
        from chatbot.rag import RAGStore
        _rag_store = RAGStore()
    return _rag_store


def _delete_conversation(conv_id: str) -> None:
    from chatbot.chat import ClaudeChat
    from storage.logic import ShadowMap
    ClaudeChat.delete(conv_id)
    ShadowMap(conv_id).delete()


def _start_chat(load: Optional[str] = None, rag: bool = False) -> None:
    """Build the pipeline and enter the chat REPL."""
    from chatbot.chat import ClaudeChat
    from pipeline import Pipeline

    try:
        if load:
            chat_handler = ClaudeChat.load(load)
            turns = len(chat_handler.conversation.messages) // 2
            console.print(
                f"[green]Resumed[/green]  "
                f"[cyan]{load[:8]}…{load[-4:]}[/cyan]  "
                f"[dim]{turns} turn{'s' if turns != 1 else ''}[/dim]"
            )
        else:
            chat_handler = ClaudeChat()
            chat_handler.conversation.rag_mode = rag
            console.print("[green]New conversation started.[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Not found:[/red] {exc}")
        return
    except EnvironmentError as exc:
        console.print(f"[red bold]API key missing.[/red bold]")
        console.print("[dim]Run:  export ANTHROPIC_API_KEY=your_key_here[/dim]")
        return

    rag_store = None
    effective_rag = rag or bool(load and chat_handler.conversation.rag_mode)
    if effective_rag:
        try:
            rag_store = _get_rag()
            console.print(
                f"[blue]RAG enabled[/blue]  "
                f"[dim]{rag_store.document_count()} chunks indexed[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]RAG unavailable:[/yellow] {exc}")
            rag_store = None

    pipeline = Pipeline(chat=chat_handler, rag=rag_store)
    _run_chat_loop(pipeline, rag_mode=bool(effective_rag))


def _run_chat_loop(pipeline, rag_mode: bool) -> None:
    """Interactive chat REPL."""
    conv_id  = pipeline.chat.conversation.id
    mode_tag = "  [dim blue]· RAG[/dim blue]" if rag_mode else ""

    console.print()
    console.print(
        f"[dim]ID [/dim][blue]{conv_id}[/blue]{mode_tag}"
        "[dim]  ·  type [bold]exit[/bold] to return to dashboard[/dim]"
    )
    console.print(Rule(style="blue"))
    console.print()

    while True:
        try:
            user_input = console.input("[bold blue]You[/bold blue]  ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "back"}:
            console.print("[dim]Returning to dashboard…[/dim]")
            time.sleep(0.4)
            break

        try:
            response, _, _ = pipeline.process_turn(user_input, interactive=True)
        except EnvironmentError:
            console.print("[red]API key error.[/red]  [dim]Check ANTHROPIC_API_KEY.[/dim]")
            break
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] {exc}")
            continue

        console.print()
        console.print(
            Panel(
                response,
                title="[bold blue]Claude[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Typer commands
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Open the interactive dashboard when called with no subcommand."""
    if ctx.invoked_subcommand is None:
        _run_dashboard()


@app.command()
def chat(
    load:   Optional[str] = typer.Option(None,  "--load",   help="Resume by ID.",  metavar="ID"),
    delete: Optional[str] = typer.Option(None,  "--delete", help="Delete by ID.",  metavar="ID"),
    rag:    bool           = typer.Option(False, "--rag",    help="Enable RAG mode."),
) -> None:
    """Start, resume, or delete a conversation."""
    _print_compact_banner()
    if delete:
        console.print(f"[yellow]Deleting:[/yellow] {delete}")
        _delete_conversation(delete)
        console.print("[green]✓[/green]  Deleted.")
        return
    _start_chat(load=load, rag=rag)


@app.command(name="list")
def list_conversations() -> None:
    """List all saved conversations."""
    from chatbot.chat import ClaudeChat
    _print_compact_banner()
    _print_conversations_table(ClaudeChat.list_conversations())


@app.command(name="add-doc")
def add_document(
    filepath: str = typer.Argument(..., help="Path to document to index."),
) -> None:
    """Anonymise and index a document into the RAG vector store."""
    from chatbot.chat import ClaudeChat
    from pipeline import Pipeline

    _print_compact_banner()
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {filepath}")
        raise typer.Exit(1)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Could not read file:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[blue]Indexing[/blue]  [white]{path.name}[/white]  [dim]{len(raw_text):,} chars[/dim]")
    try:
        rag_store   = _get_rag()
        chat_handler = ClaudeChat()
        pipeline    = Pipeline(chat=chat_handler, rag=rag_store)
        n           = pipeline.add_rag_document(raw_text, metadata={"filename": path.name})
        console.print(f"[green]✓[/green]  {n} chunks indexed  [dim](total: {rag_store.document_count()})[/dim]")
    except EnvironmentError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()