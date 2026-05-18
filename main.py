"""
main.py — SurrogateShield CLI

Run with no arguments for the interactive dashboard:
    python main.py

Or use direct commands:
    python main.py chat                  — new conversation
    python main.py chat --load <id>      — continue conversation
    python main.py chat --rag            — new conversation with RAG
    python main.py list                  — list conversations
    python main.py pii-finder            — test PII detection (no API call)
    python main.py add-doc <filepath>    — index a document into RAG
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
# Called ONCE here, before any module imports get_logger().
# util.get_logger() does NOT call basicConfig() itself.
from rich.logging import RichHandler
from rich.console import Console as _LogConsole

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=_LogConsole(), rich_tracebacks=True, markup=True)],
)
# ─────────────────────────────────────────────────────────────────────────────

import typer
from rich.align import Align
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


# ─── Banners ──────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    content = Align.center(
        Text.assemble(
            "\n",
            ("◆  ", "bold blue"),
            ("Surrogate", "bold white"),
            ("Shield", "bold blue"),
            (f"  {VERSION}\n\n", "dim blue"),
            (TAGLINE + "\n", "dim"),
        )
    )
    console.print(Panel(content, border_style="blue", padding=(1, 8), expand=False))
    console.print()


def _print_compact_banner() -> None:
    line = Text.assemble(
        ("◆  ", "bold blue"),
        ("Surrogate", "bold white"),
        ("Shield", "bold blue"),
        ("  ·  ", "dim"),
        ("PII never leaves your device", "dim"),
        (f"  {VERSION}", "dim blue"),
    )
    console.print(Rule(style="blue"))
    console.print(Align.center(line))
    console.print(Rule(style="blue"))
    console.print()


# ─── Pipeline overview ────────────────────────────────────────────────────────

def _print_how_it_works() -> None:
    steps = [
        ("PatternScan",  "Regex — SSNs, emails, phones, cards, API keys"),
        ("EntityTrace",  "spaCy NER — names, places, organisations"),
        ("MimicGen",     "Realistic fake values per PII type (Faker)"),
        ("ShadowMap",    "AES-256-GCM encrypted map — stays on device"),
        ("Claude API",   "Receives surrogates — never real values"),
        ("ResolvePass",  "Swaps fakes back to real values in response"),
    ]
    console.print(Rule("[bold blue]Pipeline[/bold blue]", style="blue"))
    console.print()
    for i, (name, desc) in enumerate(steps, 1):
        console.print(
            f"  [bold blue]{i}[/bold blue]"
            f"  [bold white]{name:<14}[/bold white]"
            f"  [dim]{desc}[/dim]"
        )
        if i < len(steps):
            console.print("   [blue]│[/blue]")
    console.print()
    console.print(Rule(style="blue"))
    console.print()


# ─── Conversations table ───────────────────────────────────────────────────────

def _relative_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = datetime.utcnow() - dt.replace(tzinfo=None)
        s = int(diff.total_seconds())
        if s < 60:    return "just now"
        if s < 3600:  return f"{s // 60}m ago"
        if s < 86400: return f"{s // 3600}h ago"
        if s < 604800: return f"{diff.days}d ago"
        return dt.strftime("%b %d %Y")
    except Exception:
        return iso_str[:10] if len(iso_str) >= 10 else "—"


def _print_conversations_table(conversations: list) -> None:
    if not conversations:
        console.print(Panel(
            "[dim]No saved conversations yet.\n\nPress [bold white]N[/bold white] to start one.[/dim]",
            border_style="blue", title="[blue]Conversations[/blue]", padding=(1, 4),
        ))
        return
    table = Table(
        title="[bold blue]Conversations[/bold blue]",
        box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1), expand=True,
    )
    table.add_column("#",       style="bold blue", justify="right", width=4)
    table.add_column("ID",      style="cyan",      no_wrap=True)
    table.add_column("Created", style="dim white", width=12)
    table.add_column("Turns",   justify="right",   style="white",   width=6)
    table.add_column("Mode",    style="blue",      width=9)
    for i, conv in enumerate(conversations, 1):
        uid   = conv["id"]
        turns = conv["message_count"] // 2
        mode  = "[magenta]RAG[/magenta]" if conv.get("rag_mode") else "standard"
        table.add_row(str(i), uid, _relative_time(conv.get("created", "")), str(turns), mode)
    console.print(table)


def _print_menu(has_convs: bool) -> None:
    console.print()
    console.print(Rule("[blue]Actions[/blue]", style="blue"))
    console.print()
    rows = [
        ("[bold blue]N[/bold blue]",         "New conversation"),
        ("[bold blue]R[/bold blue]",         "New conversation + RAG mode"),
        ("[bold blue]P[/bold blue]",         "PII Finder  — test detection, zero API calls"),
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


# ─── PII Finder ───────────────────────────────────────────────────────────────

def _run_pii_finder() -> None:
    """
    Interactive PII detection sandbox — no API calls, no credits spent.

    Shows the SAME logic that process_turn() would apply, including the
    service-query path (address fuzzing + location suppression).
    """
    from detection.logic import run_cascade, deduplicate
    from detection.service_query import is_service_query, fuzz_addresses
    from generation.logic import MimicGen
    from config import SERVICE_QUERY_DETECTION_ENABLED

    mimic = MimicGen()

    console.print(Panel(
        "[bold blue]PII Finder[/bold blue]  [dim]· No API calls · No credits spent[/dim]\n\n"
        "[dim]Type any message to see what SurrogateShield would detect.\n"
        "Service queries (restaurants near X, directions to Y) trigger minimal\n"
        "address fuzzing instead of full replacement — just like the real pipeline.\n\n"
        "Type [bold white]reset[/bold white] to clear surrogate memory.\n"
        "Type [bold white]exit[/bold white] to return to the dashboard.[/dim]",
        border_style="blue", padding=(1, 2),
    ))
    console.print()

    while True:
        try:
            user_input = console.input("[bold blue]Test[/bold blue]  ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Returning to dashboard...[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "back"}:
            console.print("[dim]Returning to dashboard...[/dim]")
            time.sleep(0.3)
            break
        if user_input.lower() == "reset":
            mimic = MimicGen()
            console.print("[green]Surrogate session reset.[/green]\n")
            continue

        # ── Service query path ────────────────────────────────────────────────
        if SERVICE_QUERY_DETECTION_ENABLED and is_service_query(user_input):
            fuzzed, addr_map = fuzz_addresses(user_input, verify=False)

            if addr_map:
                addr_lines = "\n".join(
                    f"  [red]{orig}[/red]  →  [green]{fuzz}[/green]"
                    for orig, fuzz in addr_map.items()
                )
                console.print(Panel(
                    "[bold blue]Service query[/bold blue]  "
                    "[dim]· House number ±1, city/state unchanged[/dim]\n\n"
                    f"{addr_lines}\n\n"
                    f"[dim]Would send to Claude:[/dim]\n[blue]{fuzzed}[/blue]",
                    border_style="blue", padding=(1, 2),
                ))
            else:
                console.print(Panel(
                    "[bold blue]Service query[/bold blue]  "
                    "[dim]· No specific street address found[/dim]\n\n"
                    "[dim]Location names are not PII in service queries — "
                    "message would be sent unchanged.[/dim]\n\n"
                    f"[dim]Would send to Claude:[/dim]\n[blue]{user_input}[/blue]",
                    border_style="blue", padding=(1, 2),
                ))
            console.print()
            continue

        # ── Standard PII detection path ───────────────────────────────────────
        confirmed, needs_confirmation = run_cascade(user_input)
        confirmed = deduplicate(confirmed)

        if not confirmed and not needs_confirmation:
            console.print(Panel(
                "[green]No PII detected.[/green]\n"
                "[dim]This message would be sent to Claude unchanged.[/dim]",
                border_style="green", padding=(0, 2),
            ))
            console.print()
            continue

        surrogate_map = mimic.generate_all(confirmed) if confirmed else {}

        tbl = Table(
            title="[bold blue]SentinelLayer — Detected PII[/bold blue]",
            box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1),
        )
        tbl.add_column("Original",  style="red bold",  no_wrap=True)
        tbl.add_column("Type",      style="yellow",    width=14)
        tbl.add_column("Score",     style="white",     width=6,  justify="right")
        tbl.add_column("Source",    style="dim",       width=8)
        tbl.add_column("Surrogate", style="green bold")

        for ent in confirmed:
            tbl.add_row(
                ent.text, ent.type, f"{ent.score:.2f}", ent.source,
                surrogate_map.get(ent.text, "[dim]—[/dim]"),
            )
        for ent in needs_confirmation:
            tbl.add_row(
                ent.text, ent.type, f"{ent.score:.2f}", ent.source,
                "[dim yellow]needs confirmation[/dim yellow]",
            )

        console.print(tbl)

        sanitised = user_input
        for orig in sorted(surrogate_map, key=len, reverse=True):
            sanitised = sanitised.replace(orig, surrogate_map[orig])

        console.print(Panel(
            "[dim]Would send to Claude:[/dim]\n[blue]" + sanitised + "[/blue]",
            border_style="dim blue", padding=(0, 2),
        ))
        console.print()


# ─── Dashboard ────────────────────────────────────────────────────────────────

def _run_dashboard() -> None:
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

        if upper == "Q":
            console.print("\n[dim]Goodbye.[/dim]")
            sys.exit(0)
        if upper == "N":
            console.clear(); _print_compact_banner(); _start_chat(rag=False); continue
        if upper == "R":
            console.clear(); _print_compact_banner(); _start_chat(rag=True); continue
        if upper == "P":
            console.clear(); _print_compact_banner(); _run_pii_finder(); continue

        if upper.startswith("D") and upper[1:].isdigit():
            idx = int(upper[1:]) - 1
            if 0 <= idx < len(conversations):
                uid = conversations[idx]["id"]
                console.print(
                    f"\n[yellow]Delete [bold]{uid[:8]}...{uid[-4:]}[/bold]?[/yellow] "
                    "[dim](y / N)[/dim] ", end="",
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

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(conversations):
                console.clear(); _print_compact_banner()
                _start_chat(load=conversations[idx]["id"])
            else:
                console.print(f"[red]No conversation #{idx + 1}[/red]")
                time.sleep(0.6)
            continue

        console.print(f"[red]Unknown:[/red] {raw!r}")
        time.sleep(0.5)


# ─── Shared helpers ───────────────────────────────────────────────────────────

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
    from chatbot.chat import ClaudeChat
    from pipeline import Pipeline

    try:
        if load:
            chat_handler = ClaudeChat.load(load)
            turns = len(chat_handler.conversation.messages) // 2
            console.print(
                f"[green]Resumed[/green]  [cyan]{load[:8]}...{load[-4:]}[/cyan]  "
                f"[dim]{turns} turn{'s' if turns != 1 else ''}[/dim]"
            )
        else:
            chat_handler = ClaudeChat()
            chat_handler.conversation.rag_mode = rag
            console.print("[green]New conversation started.[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Not found:[/red] {exc}"); return
    except EnvironmentError:
        console.print("[red bold]API key missing.[/red bold]")
        console.print("[dim]Run:  export ANTHROPIC_API_KEY=your_key_here[/dim]"); return

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

    _run_chat_loop(Pipeline(chat=chat_handler, rag=rag_store), rag_mode=bool(effective_rag))


def _run_chat_loop(pipeline, rag_mode: bool) -> None:
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
            console.print("\n[dim]Session ended.[/dim]"); break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "back"}:
            console.print("[dim]Returning to dashboard...[/dim]")
            time.sleep(0.4); break

        try:
            response, _, _ = pipeline.process_turn(user_input, interactive=True)
        except EnvironmentError:
            console.print(
                "[red]API key error.[/red]  [dim]Check ANTHROPIC_API_KEY.[/dim]"
            ); break
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] {exc}"); continue

        console.print()
        console.print(Panel(
            response,
            title="[bold blue]Claude[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        ))
        console.print()


# ─── Typer commands ───────────────────────────────────────────────────────────

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


@app.command(name="pii-finder")
def pii_finder_cmd() -> None:
    """Test PII detection on any text — no API call, no credits spent."""
    _print_compact_banner()
    _run_pii_finder()


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
    _print_compact_banner()
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {filepath}"); raise typer.Exit(1)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Could not read file:[/red] {exc}"); raise typer.Exit(1)

    console.print(
        f"[blue]Indexing[/blue]  [white]{path.name}[/white]  [dim]{len(raw_text):,} chars[/dim]"
    )
    try:
        from pipeline import anonymise_for_rag
        rag_store = _get_rag()
        n, _ = anonymise_for_rag(raw_text, rag_store)
        console.print(
            f"[green]✓[/green]  {n} chunks indexed  "
            f"[dim](total: {rag_store.document_count()})[/dim]"
        )
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}"); raise typer.Exit(1)


if __name__ == "__main__":
    app()