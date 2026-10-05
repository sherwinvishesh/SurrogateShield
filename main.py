# Paper available on arXiv: https://arxiv.org/abs/2606.29567

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
    python main.py rag [--forget DOC]    — list / remove indexed documents
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import config as _cfg

# ── Logging setup ─────────────────────────────────────────────────────────────
# Called ONCE here, before any module imports get_logger().
# util.get_logger() does NOT call basicConfig() itself.
from rich.logging import RichHandler
from rich.console import Console as _LogConsole

logging.basicConfig(
    # WARNING by default so stage chatter does not interleave with the UI
    # (audit I23); SURROGATESHIELD_LOG_LEVEL=INFO or DEBUG to see it.
    level=getattr(logging, os.getenv("SURROGATESHIELD_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=_LogConsole(), rich_tracebacks=True, markup=True)],
)

# Our own loggers. Detailed View raises only these to INFO; the root logger
# (and with it httpx, transformers, …) never drops below WARNING (audit I17).
_APP_LOGGERS = ("surrogateshield", "pipeline", "chatbot", "__main__")


def _set_detailed_logging(detailed: bool) -> None:
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO if detailed else logging.NOTSET)
# ─────────────────────────────────────────────────────────────────────────────

import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from surrogateshield.core.errors import DetectorUnavailable
from surrogateshield.core.storage.shadow_map import StorageError


def _print_detector_unavailable(exc: DetectorUnavailable) -> None:
    """Detection failed closed (audit I17): say so loudly; nothing was sent."""
    console.print(Panel(
        f"{exc}\n\n[dim]Nothing was masked and nothing was sent.[/dim]",
        title="[bold red]Detector unavailable[/bold red]",
        border_style="red",
    ))

console = Console()

app = typer.Typer(
    name="surrogateshield",
    help="Privacy-preserving CLI proxy for LLMs — masks PII before it leaves your device.",
    add_completion=False,
)

_rag_store = None

# (slug, display name, short description)
# Model ids come from config.py — the ones chatbot/chat.py actually calls.
_PROVIDERS = [
    ("claude",  "Claude",    f"Anthropic  ·  {_cfg.CLAUDE_MODEL}"),
    ("gemini",  "Gemini",    f"Google     ·  {_cfg.GEMINI_MODEL}"),
    ("chatgpt", "ChatGPT",   f"OpenAI     ·  {_cfg.OPENAI_MODEL}"),
    ("local",   "Local LLM", "Ollama     ·  runs fully offline"),
]


def _current_provider_name() -> str:
    """Return the display name of the currently configured LLM provider."""
    from settings_manager import load_settings
    slug = load_settings()["llm_provider"]
    return next((n for s, n, _ in _PROVIDERS if s == slug), "LLM")


# ─── Banners ──────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    _TOP = (
        "███████╗██╗  ██╗██╗███████╗██╗     ██████╗ \n"
        "██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗\n"
        "███████╗███████║██║█████╗  ██║     ██║  ██║\n"
    )
    _BOT = (
        "╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║\n"
        "███████║██║  ██║██║███████╗███████╗██████╔╝\n"
        "╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ "
    )
    content = Align.center(
        Text.assemble(
            "\n",
            ("◆  ────────────────────────────────────  ◆\n\n", "dim blue"),
            ("S   U   R   R   O   G   A   T   E\n\n", "bold white"),
            (_TOP, "bold blue"),
            (_BOT, "bold cyan"),
            ("\n\n", ""),
            ("◆  ────────────────────────────────────  ◆\n\n", "dim blue"),
            ("Privacy-preserving proxy for LLMs\n", "dim"),
            ("Masks PII before it leaves your device\n", "dim"),
        )
    )
    console.print(Panel(content, border_style="blue", padding=(1, 6), expand=False))
    console.print()


def _print_compact_banner() -> None:
    line = Text.assemble(
        ("◆  ", "bold blue"),
        ("Surrogate", "bold white"),
        ("Shield", "bold cyan"),
        ("  ·  ", "dim"),
        ("Masks PII before it leaves your device", "dim"),
    )
    console.print(Rule(style="blue"))
    console.print(Align.center(line))
    console.print(Rule(style="blue"))
    console.print()


# ─── Pipeline overview ────────────────────────────────────────────────────────

def _print_how_it_works() -> None:
    from settings_manager import load_settings
    provider_slug = load_settings()["llm_provider"]
    provider_name = next((n for s, n, _ in _PROVIDERS if s == provider_slug), "LLM")
    steps = [
        ("PatternScan",       "Regex — SSNs, emails, phones, cards, API keys"),
        ("EntityTrace",       "spaCy NER — names, places, organisations"),
        ("ContextGuard",      "distilbert-NER — borderline entity resolution"),
        ("MimicGen",          "Realistic fake values per PII type (Faker)"),
        ("ShadowMap",         "AES-256-GCM encrypted map — stays on device"),
        (f"{provider_name} API", "Receives surrogates — never real values"),
        ("ResolvePass",       "Swaps fakes back to real values in response"),
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
        diff = datetime.now(timezone.utc).replace(tzinfo=None) - dt.replace(tzinfo=None)
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
    from help_screen import MENU_ITEMS
    rows = [(key, name if key in {"N", "R", "S", "H", "Q"} else f"{name}  — {desc}")
            for key, name, desc, convs_only in MENU_ITEMS if has_convs or not convs_only]
    for key, desc in rows:
        console.print(f"  [bold blue]{key}[/bold blue]    [dim]{desc}[/dim]")
    console.print()


# ─── PII Finder ───────────────────────────────────────────────────────────────

def _run_pii_finder() -> None:
    """
    Interactive PII detection sandbox — no API calls, no credits spent.

    Shows the SAME logic that process_turn() would apply, including the
    service-query path (address fuzzing + location suppression).
    """
    from settings_manager import load_settings as _ls
    _settings  = _ls()
    _detailed  = _settings["detailed_view"]
    _show_pres = _settings["presidio_comparison"]
    _set_detailed_logging(_detailed)
    from detection.logic import run_cascade, deduplicate
    from detection.service_query import resolve as resolve_service
    from generation.logic import MimicGen
    from detection.service_query import resolve_address_mode
    from util import apply_entity_surrogates
    from config import (
        ADDRESS_MODE,
        ADDRESS_SHIFT_RANGE,
        SERVICE_QUERY_DETECTION_ENABLED,
    )

    mimic = MimicGen()

    # ── Initialize Presidio once upfront ─────────────────────────────
    # Always import the names so _show_presidio_panel's closure is valid
    # regardless of whether _show_pres is True or False.
    # These imports are instant — no presidio_analyzer load happens here.
    from presidio.engine import is_available, unavailability_reason
    from presidio.detect import detect as presidio_detect
    from presidio.redact import redact as presidio_redact

    if _show_pres:
        console.print("[dim]Initializing Presidio comparison engine...[/dim]", end="\r")
        _presidio_ready = is_available()   # triggers the lazy load (3-5s first time)
        if _presidio_ready:
            console.print("[dim green]✓  Presidio ready[/dim green]                              ")
        else:
            console.print(
                f"[dim yellow]⚠  Presidio unavailable: {unavailability_reason()}[/dim yellow]"
                "                    "
            )
        console.print()
    else:
        _presidio_ready = False

    console.print(Panel(
        "[bold blue]PII Finder[/bold blue]  [dim]· No API calls · No credits spent[/dim]\n\n"
        "[dim]Type any message to see what SurrogateShield would detect.\n"
        "Service queries (restaurants near X, directions to Y) trigger minimal\n"
        "address fuzzing instead of full replacement — just like the real pipeline.\n\n"
        "Type [bold white]reset[/bold white] to clear surrogate memory.\n"
        "Type [bold white]exit[/bold white] to return to the dashboard.[/dim]"
        + ("\n[dim]Presidio comparison shown below each result.[/dim]" if _presidio_ready else ""),
        border_style="blue", padding=(1, 2),
    ))
    console.print()

    def _show_presidio_panel(original_text: str) -> None:
        """Show Presidio detection table and redacted text."""
        if not _show_pres:
            return
        if not _presidio_ready:
            console.print(Panel(
                f"[dim]Presidio unavailable: {unavailability_reason()}[/dim]",
                title="[dim]Presidio Comparison[/dim]",
                border_style="dim",
                padding=(0, 2),
            ))
            console.print()
            return

        entities = presidio_detect(original_text)

        if entities is None:
            console.print(Panel(
                "[dim]Presidio detection failed.[/dim]",
                title="[dim]Presidio Comparison[/dim]",
                border_style="dim",
                padding=(0, 2),
            ))
            console.print()
            return

        if not entities:
            console.print(Panel(
                "[dim green]Presidio detected no PII.[/dim green]\n"
                f"[dim]Would send to LLM unchanged:[/dim]\n[blue]{original_text}[/blue]",
                title="[bold]Presidio Comparison[/bold]",
                border_style="dim blue",
                padding=(0, 2),
            ))
            console.print()
            return

        # Build detection table
        tbl = Table(
            title="[bold]Presidio — Detected PII[/bold]",
            box=box.ROUNDED,
            border_style="dim blue",
            show_lines=True,
            padding=(0, 1),
        )
        tbl.add_column("Detected Value", style="red bold",  no_wrap=True)
        tbl.add_column("Type",           style="yellow",    width=22)
        tbl.add_column("Score",          style="dim white", width=6, justify="right")

        for ent in entities:
            tbl.add_row(ent.text, ent.entity_type, f"{ent.score:.2f}")

        console.print(tbl)

        # Redacted text panel
        redacted = presidio_redact(original_text, entities)
        console.print(Panel(
            f"[dim]Would send to LLM (Presidio — placeholder redaction):[/dim]\n"
            f"[blue]{redacted}[/blue]",
            border_style="dim blue",
            padding=(0, 2),
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
        is_svc, sq_mode = resolve_service(user_input, ADDRESS_MODE,
                                          SERVICE_QUERY_DETECTION_ENABLED)
        if is_svc:
            # v2: addresses flow through the unified detect→generate path.
            # "auto" resolves to shift (±N house number), or to coarse ("my
            # area", city/state kept) for a sensitive service query (I6).

            try:
                sq_confirmed, _ = run_cascade(user_input, skip_location_entities=True)
            except DetectorUnavailable as exc:
                _print_detector_unavailable(exc); continue
            sq_confirmed = deduplicate(sq_confirmed)
            sq_skipped   = getattr(sq_confirmed, '_skipped_entities', [])
            sq_surrogate_map = (
                mimic.generate_all(
                    sq_confirmed,
                    address_mode=sq_mode,
                    address_shift_range=ADDRESS_SHIFT_RANGE,
                    text=user_input,
                )
                if sq_confirmed
                else {}
            )
            sq_sanitised = apply_entity_surrogates(
                user_input, sq_confirmed, sq_surrogate_map
            )

            addr_map = {
                e.text: sq_surrogate_map[e.text]
                for e in sq_confirmed
                if e.type == "address" and e.text in sq_surrogate_map
            }
            if addr_map:
                addr_lines = "\n".join(
                    f"  [red]{orig}[/red]  →  [green]{surr}[/green]"
                    for orig, surr in addr_map.items()
                )
                mode_note = (
                    f"House number ±{ADDRESS_SHIFT_RANGE}, city/state unchanged"
                    if sq_mode == "shift"
                    else "Street line coarsened to 'my area', city/state unchanged"
                    if sq_mode == "coarse"
                    else "Structure-preserving fake address"
                )
                console.print(Panel(
                    "[bold blue]Service query[/bold blue]  "
                    f"[dim]· {mode_note}[/dim]\n\n"
                    f"{addr_lines}\n\n"
                    f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{sq_sanitised}[/blue]",
                    border_style="blue", padding=(1, 2),
                ))
            else:
                console.print(Panel(
                    "[bold blue]Service query[/bold blue]  "
                    "[dim]· No specific street address found[/dim]\n\n"
                    "[dim]Location names are not PII in service queries — "
                    "message would be sent unchanged.[/dim]\n\n"
                    f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{sq_sanitised}[/blue]",
                    border_style="blue", padding=(1, 2),
                ))
            console.print()

            if sq_confirmed or sq_skipped:
                sq_tbl = Table(
                    title="[bold blue]SentinelLayer — Detected PII[/bold blue]",
                    box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1),
                )
                sq_tbl.add_column("Original",  style="red bold",  no_wrap=True)
                sq_tbl.add_column("Type",      style="yellow",    width=14)
                sq_tbl.add_column("Score",     style="white",     width=6,  justify="right")
                sq_tbl.add_column("Source",    style="dim",       width=8)
                sq_tbl.add_column("Surrogate", style="green bold")
                for ent in sq_confirmed:
                    sq_tbl.add_row(
                        ent.text, ent.type, f"{ent.score:.2f}", ent.source,
                        sq_surrogate_map.get(ent.text, "[dim]—[/dim]"),
                    )
                for ent in sq_skipped:
                    sq_tbl.add_row(
                        ent.text, ent.type, f"{ent.score:.2f}", ent.source,
                        "[dim yellow]skipped — service query[/dim yellow]",
                    )
                console.print(sq_tbl)
                console.print()

            _show_presidio_panel(user_input)
            continue

        # ── Standard PII detection path ───────────────────────────────────────
        try:
            confirmed, needs_confirmation = run_cascade(user_input)
        except DetectorUnavailable as exc:
            _print_detector_unavailable(exc); continue
        confirmed = deduplicate(confirmed)
        skipped   = getattr(confirmed, '_skipped_entities', [])

        if not confirmed and not needs_confirmation and not skipped:
            console.print(Panel(
                "[green]No PII detected.[/green]\n"
                f"[dim]This message would be sent to {_current_provider_name()} unchanged.[/dim]",
                border_style="green", padding=(0, 2),
            ))
            console.print()
            _show_presidio_panel(user_input)
            continue

        std_mode = resolve_address_mode(ADDRESS_MODE, False)
        surrogate_map = (
            mimic.generate_all(
                confirmed,
                address_mode=std_mode,
                address_shift_range=ADDRESS_SHIFT_RANGE,
                text=user_input,
            )
            if confirmed
            else {}
        )

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
        for ent in skipped:
            tbl.add_row(
                ent.text, ent.type, f"{ent.score:.2f}", ent.source,
                "[dim yellow]skipped — topical query[/dim yellow]",
            )

        console.print(tbl)

        if _detailed:
            from detection.quasi_identifier import format_warning as _qi_fmt
            qi_matches = getattr(confirmed, "_qi_matches", [])
            if qi_matches:
                console.print(f"[bold yellow]{_qi_fmt(qi_matches)}[/bold yellow]")
            else:
                console.print("[dim green]✓  No quasi-identifier combination risk detected.[/dim green]")
            console.print()

        # Same span-based substitution as the real send path (E1).
        sanitised = apply_entity_surrogates(user_input, confirmed, surrogate_map)

        console.print(Panel(
            f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{sanitised}[/blue]",
            border_style="dim blue", padding=(0, 2),
        ))
        console.print()
        _show_presidio_panel(user_input)


# ─── JSON Test ────────────────────────────────────────────────────────────────

def _run_json_test() -> None:
    """Three-screen JSON batch testing flow."""
    import json as _json
    from json_tester import EXPERIMENT_DIR, OUTPUT_FIELDS, DEFAULT_FIELDS, run_batch

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Screen 1: Enter filename ──────────────────────────────────────────────
    console.print(Panel(
        "[bold blue]JSON Test[/bold blue]  "
        "[dim]· Batch-process questions through the full pipeline[/dim]\n\n"
        "  [dim]Place your input file in:[/dim]  [cyan]experiment/<name>.json[/cyan]\n"
        "  [dim]Output will be saved to:[/dim]  [cyan]experiment/<name>_answers.json[/cyan]\n\n"
        "[dim]Input format:[/dim]\n"
        "  [cyan][ {\"input\": \"question 1\"}, {\"input\": \"question 2\"}, … ][/cyan]\n\n"
        "[dim]Results are saved every 25 questions — safe to interrupt and resume.[/dim]",
        border_style="blue", padding=(1, 2),
    ))
    console.print()

    try:
        filename = console.input(
            "  [dim]experiment/[/dim][bold blue]filename › [/bold blue]"
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not filename or filename.upper() == "B":
        return
    if not filename.endswith(".json"):
        filename += ".json"

    in_path = EXPERIMENT_DIR / filename
    if not in_path.exists():
        console.print(f"\n  [red]File not found:[/red] experiment/{filename}")
        time.sleep(1.5)
        return

    try:
        questions = _json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"\n  [red]Invalid JSON:[/red] {exc}")
        time.sleep(1.5)
        return

    total = len(questions)
    stem  = in_path.stem
    out_path = EXPERIMENT_DIR / f"{stem}_answers.json"

    existing = 0
    if out_path.exists():
        try:
            existing = len(_json.loads(out_path.read_text(encoding="utf-8")))
        except Exception:
            existing = 0

    # ── Screen 2: Field selection ─────────────────────────────────────────────
    fields = DEFAULT_FIELDS.copy()

    while True:
        console.clear()
        _print_compact_banner()

        resume_note = (
            f"  [green]Resuming:[/green] {existing}/{total} already answered"
            f" — will start from question {existing + 1}\n\n"
            if existing > 0 else ""
        )

        console.print(Panel(
            f"[bold blue]Field Selection[/bold blue]  "
            f"[dim]· {filename}  ({total} question{'s' if total != 1 else ''})[/dim]\n\n"
            f"{resume_note}"
            "[dim]Press a number to toggle. Press [bold white]Enter[/bold white] to run.[/dim]",
            border_style="blue", padding=(1, 2),
        ))
        console.print()

        for i, (key, label) in enumerate(OUTPUT_FIELDS, 1):
            mark = "[green]✓[/green]" if fields[key] else "[dim]□[/dim]"
            console.print(f"  [bold blue]{i}[/bold blue]  {mark}  [white]{label}[/white]")

        console.print()
        console.print(Rule(style="dim blue"))
        console.print(f"\n  [dim]Enter[/dim] → Run  ·  [bold blue]B[/bold blue] → Back\n")

        try:
            raw = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return

        if raw == "B":
            return
        if raw == "":
            break
        if raw.isdigit() and 1 <= int(raw) <= len(OUTPUT_FIELDS):
            key = OUTPUT_FIELDS[int(raw) - 1][0]
            fields[key] = not fields[key]

    # ── Screen 3: Run ─────────────────────────────────────────────────────────
    console.clear()
    _print_compact_banner()
    console.print(Panel(
        f"[bold blue]JSON Test Running[/bold blue]  [dim]· {filename}[/dim]\n\n"
        f"  [dim]Output →[/dim] [cyan]{out_path}[/cyan]\n"
        f"  [dim]Total  :[/dim] {total} questions"
        + (f"  [dim](resuming from {existing + 1})[/dim]" if existing else ""),
        border_style="blue", padding=(0, 2),
    ))
    console.print()
    console.print(Rule(style="blue"))
    console.print()

    logging.getLogger().setLevel(logging.WARNING)

    errors = 0

    def _on_progress(i: int, total: int, question: str, status: str, elapsed: float) -> None:
        nonlocal errors

        if i < 0:
            # Sentinel status — handle below, do not format as a question row
            if status == "bertscore_start":
                console.print()
                console.print(
                    "  [dim blue]⟳[/dim blue]  "
                    "[dim]Computing BERTScore (roberta-large)  "
                    "— may take 15–30 min on CPU...[/dim]"
                )
            elif status == "bertscore_done":
                console.print(
                    f"  [green]✓[/green]  "
                    f"[dim]BERTScore complete  ({elapsed:.1f}s)[/dim]"
                )
                console.print()
            elif status == "bertscore_skipped":
                console.print()
                console.print(
                    "  [yellow]⚠[/yellow]  "
                    "[yellow]bert-score not installed — BERTScore skipped.[/yellow]\n"
                    "  [dim]Run: pip install bert-score  then re-run this batch.[/dim]"
                )
                console.print()
            elif status == "bertscore_warn":
                missing = int(elapsed)   # elapsed repurposed to carry null_count
                console.print(
                    f"  [yellow]⚠[/yellow]  "
                    f"[yellow]Presidio BERTScore: {missing} question(s) had no "
                    f"presidio_sanitized_input — skipped.[/yellow]\n"
                    "  [dim]Enable 'Presidio sanitized' field in JSON Test "
                    "and re-run for full coverage.[/dim]"
                )
            return

        short_q = (question[:68] + "…") if len(question) > 68 else question
        idx     = f"[dim]{i + 1:>{len(str(total))}}/{total}[/dim]"

        if status == "running":
            console.print(f"  [dim blue]⟳[/dim blue]  {idx}  [dim]{short_q}[/dim]")
        elif status == "ok":
            save_note = "  [dim blue]💾 saved[/dim blue]" if (i + 1 - existing) % 25 == 0 or (i + 1) == total else ""
            console.print(f"  [green]✓[/green]  {idx}  [dim]{elapsed:.1f}s[/dim]{save_note}")
        elif status == "error":
            errors += 1
            console.print(f"  [red]✗[/red]  {idx}  [red]error — see output file[/red]")

    try:
        out = run_batch(filename, fields, progress_cb=_on_progress)
        new_count = total - existing

        console.print()
        console.print(Rule(style="green"))
        console.print()
        console.print(
            f"  [green bold]Done![/green bold]  "
            f"{new_count} new question{'s' if new_count != 1 else ''} processed."
        )
        if errors:
            console.print(f"  [yellow]{errors} error{'s' if errors != 1 else ''}[/yellow] — details in the output file.")
        console.print(f"  [dim]Saved to:[/dim] [cyan]{out}[/cyan]")

    except KeyboardInterrupt:
        # run_batch flushed every finished row before re-raising (audit I26)
        console.print("\n  [yellow]Interrupted.[/yellow]  [dim]Progress saved — run the "
                      "same file again to resume.[/dim]")
    except EnvironmentError as exc:
        console.print(f"\n  [red bold]Configuration error:[/red bold] {exc}")
        console.print("  [dim]Go to Settings (S) to configure your LLM provider.[/dim]")
    except Exception as exc:
        console.print(f"\n  [red bold]Error:[/red bold] {exc}")

    console.print()
    try:
        console.input("  [dim]Press Enter to return to dashboard…[/dim]")
    except (EOFError, KeyboardInterrupt):
        pass


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _run_evaluation() -> None:
    """Four-screen pipeline evaluation flow."""
    import json as _json
    from evaluator import EXPERIMENT_DIR, EVAL_FIELDS, run_evaluation

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Screen 1: File inputs ─────────────────────────────────────────────────
    console.print(Panel(
        "[bold blue]Evaluation[/bold blue]  "
        "[dim]· Score pipeline quality from JSON files[/dim]\n\n"
        "Compare pipeline output against ground-truth keys to measure\n"
        "detection quality, sanitization accuracy, and ResolvePass effectiveness.\n\n"
        "[dim]All files are read from:[/dim]  [cyan]experiment/[/cyan]",
        border_style="blue", padding=(1, 2),
    ))
    console.print()

    def _ask_file(prompt: str):
        try:
            fn = console.input(f"  [dim]experiment/[/dim][bold blue]{prompt}[/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not fn or fn.upper() == "B":
            return None
        if not fn.endswith(".json"):
            fn += ".json"
        return fn

    questions_file = _ask_file("questions file  › ")
    if questions_file is None:
        return
    answers_file = _ask_file("answers file    › ")
    if answers_file is None:
        return
    key_file = _ask_file("key file        › ")
    if key_file is None:
        return

    missing = [fn for fn in (questions_file, answers_file, key_file)
               if not (EXPERIMENT_DIR / fn).exists()]
    if missing:
        for fn in missing:
            console.print(f"\n  [red]File not found:[/red] experiment/{fn}")
        time.sleep(1.5)
        return

    try:
        _q = _json.loads((EXPERIMENT_DIR / questions_file).read_text(encoding="utf-8"))
        _a = _json.loads((EXPERIMENT_DIR / answers_file).read_text(encoding="utf-8"))
        _k = _json.loads((EXPERIMENT_DIR / key_file).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        console.print(f"\n  [red]Invalid JSON:[/red] {exc}")
        time.sleep(1.5)
        return

    if not (len(_q) == len(_a) == len(_k)):
        console.print(
            f"\n  [red]Length mismatch:[/red] "
            f"questions={len(_q)}, answers={len(_a)}, keys={len(_k)}\n"
            "  [dim]All three files must have the same number of entries.[/dim]"
        )
        time.sleep(2.0)
        return

    total = len(_q)

    # ── Screen 2: Field selection ─────────────────────────────────────────────
    fields = {key: True for key, _ in EVAL_FIELDS}

    while True:
        console.clear()
        _print_compact_banner()

        console.print(Panel(
            f"[bold blue]Field Selection[/bold blue]  "
            f"[dim]· {questions_file}  ({total} question{'s' if total != 1 else ''})[/dim]\n\n"
            "[dim]Press a number to toggle. Press [bold white]Enter[/bold white] to run.[/dim]",
            border_style="blue", padding=(1, 2),
        ))
        console.print()

        for i, (key, label) in enumerate(EVAL_FIELDS, 1):
            mark = "[green]✓[/green]" if fields[key] else "[dim]□[/dim]"
            console.print(f"  [bold blue]{i}[/bold blue]  {mark}  [white]{label}[/white]")

        console.print()
        console.print(Rule(style="dim blue"))
        console.print(f"\n  [dim]Enter[/dim] → Run  ·  [bold blue]B[/bold blue] → Back\n")

        try:
            raw = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return

        if raw == "B":
            return
        if raw == "":
            break
        if raw.isdigit() and 1 <= int(raw) <= len(EVAL_FIELDS):
            key = EVAL_FIELDS[int(raw) - 1][0]
            fields[key] = not fields[key]

    # ── Screen 3: Run ─────────────────────────────────────────────────────────
    console.clear()
    _print_compact_banner()
    console.print(Panel(
        f"[bold blue]Evaluation Running[/bold blue]  [dim]· {questions_file}[/dim]",
        border_style="blue", padding=(0, 2),
    ))
    console.print()
    console.print(Rule(style="blue"))
    console.print()

    def _on_progress(i: int, total: int, status: str) -> None:
        idx = f"[dim]{i + 1:>{len(str(total))}}/{total}[/dim]"
        if status == "error":
            console.print(f"  [red]✗[/red]  {idx}  [red]runner error row — excluded[/red]")
        elif (i + 1) % 100 == 0 or i + 1 == total:
            console.print(f"  [green]✓[/green]  {idx}  [dim]scored[/dim]")

    try:
        metrics = run_evaluation(
            questions_file, answers_file, key_file,
            fields, progress_cb=_on_progress,
        )
    except (ValueError, OSError) as exc:
        console.print(f"\n  [red bold]Error:[/red bold] {exc}")
        console.print()
        try:
            console.input("  [dim]Press Enter to return to dashboard…[/dim]")
        except (EOFError, KeyboardInterrupt):
            pass
        return

    # ── Screen 4: Results ─────────────────────────────────────────────────────
    from eval_report import render as _render_eval

    console.clear()
    _print_compact_banner()
    _render_eval(metrics, console)

    # ── Actions ───────────────────────────────────────────────────────────────
    console.print(f"  [bold blue]S[/bold blue]    [dim]Save results as JSON[/dim]")
    console.print(f"  [bold blue]B[/bold blue]    [dim]Back to dashboard[/dim]")
    console.print()

    while True:
        try:
            choice = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return

        if choice in ("B", ""):
            return
        if choice == "S":
            stem     = Path(answers_file).stem
            out_path = EXPERIMENT_DIR / f"{stem}_eval_results.json"
            out_path.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
            console.print(f"\n  [green]✓[/green]  Saved to [cyan]{out_path}[/cyan]\n")


# ─── Attacker Experiment ───────────────────────────────────────────────────────

def _run_attacker_experiment() -> None:
    """Attacker experiment: a separate model tries to recover originals from the sent text."""
    import json as _json
    import attacker as _atk
    from eval_report import render_attacker

    _atk.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold blue]Attacker Experiment[/bold blue]  "
        "[dim]· can a second model recover the originals from what was sent?[/dim]\n\n"
        "For each sampled question, an attacker model reads the text each system sent\n"
        "(SurrogateShield surrogates; Presidio <TYPE> placeholders) and estimates the original\n"
        "values. Scored per gold value: visible verbatim (leak-through), exact and partial\n"
        "recovery; redactions outside the key are reported separately.\n\n"
        f"[dim]Attacker model: {_atk.ATTACKER_MODEL or 'not set (SURROGATESHIELD_ATTACKER_MODEL)'} · "
        f"responder: {_atk.CLAUDE_MODEL} · 1 call per arm per question[/dim]",
        border_style="blue", padding=(1, 2),
    ))
    console.print()

    def _ask(prompt: str, default: str = "") -> Optional[str]:
        try:
            v = console.input(f"  [bold blue]{prompt}[/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if v.upper() == "B":
            return None
        return v or default

    answers_file = _ask("experiment/ answers file › ")
    if not answers_file:
        return
    key_file = _ask("experiment/ key file     › ")
    if not key_file:
        return
    answers_file, key_file = (f if f.endswith(".json") else f + ".json" for f in (answers_file, key_file))
    sample_s = _ask("sample size (blank = all) › ")
    if sample_s is None:
        return
    seed_s = _ask("seed (blank = 0)          › ", "0")
    if seed_s is None:
        return
    try:
        sample = int(sample_s) if sample_s else None
        seed = int(seed_s)
        model = _atk.resolve_model(None)
        answers, _keys = _atk._load_inputs(_atk.EXPERIMENT_DIR, answers_file, key_file)
    except (ValueError, OSError) as exc:
        console.print(f"\n  [red]{exc}[/red]\n")
        time.sleep(2.0)
        return
    indices, calls = _atk.plan(answers, sample, seed)

    console.print()
    console.print(f"  questions: {len(indices)}   provider calls: {calls}   model: {model}")
    console.print("  Press [bold white]Enter[/bold white] to run  ·  [bold blue]B[/bold blue] to go back\n")
    try:
        if console.input("[bold blue]›[/bold blue]  ").strip().upper() == "B":
            return
    except (EOFError, KeyboardInterrupt):
        return

    def _on_progress(n: int, total_n: int, status: str) -> None:
        mark = "[green]✓[/green]" if status == "ok" else "[red]✗ unavailable[/red]"
        console.print(f"  {mark}  [dim]{n + 1}/{total_n}[/dim]")

    try:
        path = _atk.run_experiment(answers_file, key_file, sample=sample, seed=seed,
                                   model=model, progress_cb=_on_progress)
    except (ValueError, OSError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            console.print("\n  [yellow]Interrupted.[/yellow]  [dim]Finished rows saved — "
                          "run again to resume.[/dim]\n")
        else:
            console.print(f"\n  [red bold]Error:[/red bold] {exc}\n")
        try:
            console.input("  [dim]Press Enter to return to dashboard…[/dim]")
        except (EOFError, KeyboardInterrupt):
            pass
        return

    out = _json.loads(path.read_text(encoding="utf-8"))
    console.clear()
    _print_compact_banner()
    render_attacker(out["analysis"], out["meta"], console)
    console.print(f"\n  [dim]results: {path}[/dim]\n")
    try:
        console.input("  [dim]Press Enter to return to dashboard…[/dim]")
    except (EOFError, KeyboardInterrupt):
        pass


# ─── Settings ─────────────────────────────────────────────────────────────────

_PROVIDER_INSTRUCTIONS: dict = {
    "claude": [
        "1. Visit [blue]console.anthropic.com[/blue] and sign in.",
        "2. Go to [bold white]API Keys[/bold white] and create a new key.",
        "3. Add to your [bold white].env[/bold white] file (then [cyan]chmod 600 .env[/cyan]):\n\n"
        "       [cyan]ANTHROPIC_API_KEY=sk-ant-...[/cyan]",
        "4. Press [bold white]T[/bold white] to test your current key.",
    ],
    "gemini": [
        "1. Visit [blue]aistudio.google.com[/blue] and sign in.",
        "2. Click [bold white]Get API Key[/bold white] to generate a key.",
        "3. Add to your [bold white].env[/bold white] file (then [cyan]chmod 600 .env[/cyan]):\n\n"
        "       [cyan]GEMINI_API_KEY=AIza...[/cyan]",
        "4. Install the SDK:\n\n"
        "       [cyan]pip install google-genai[/cyan]",
        "5. Press [bold white]T[/bold white] to test your current key.",
    ],
    "chatgpt": [
        "1. Visit [blue]platform.openai.com[/blue] and sign in.",
        "2. Go to [bold white]API Keys[/bold white] and create a new secret key.",
        "3. Add to your [bold white].env[/bold white] file (then [cyan]chmod 600 .env[/cyan]):\n\n"
        "       [cyan]OPENAI_API_KEY=sk-...[/cyan]",
        "4. Install the SDK:\n\n"
        "       [cyan]pip install openai[/cyan]",
        "5. Press [bold white]T[/bold white] to test your current key.",
    ],
    "local": [
        "1. Download and install Ollama from [blue]ollama.ai[/blue].",
        "2. Pull a model, e.g.:\n\n"
        "       [cyan]ollama pull llama3.2[/cyan]",
        "3. Start the Ollama server:\n\n"
        "       [cyan]ollama serve[/cyan]",
        "4. (Optional) Add to your [bold white].env[/bold white] file:\n\n"
        "       [cyan]LOCAL_LLM_HOST=http://localhost:11434[/cyan]\n"
        "       [cyan]LOCAL_LLM_MODEL=llama3.2[/cyan]",
        "5. Press [bold white]T[/bold white] to test the connection.",
    ],
}


def _test_provider(slug: str, name: str) -> None:
    """Make a minimal API call to verify the provider is reachable."""
    # Pick up keys just added to .env; a variable already exported in the
    # shell keeps precedence, as at start-up (audit I27).
    load_dotenv(override=False)
    console.print(f"\n  [dim]Testing {name} connection…[/dim]")
    # Same adapter, model id and retry policy as the chat itself (F2, I20,
    # I21): "successful" means the configured production model answered.
    from chatbot import providers
    try:
        adapter = providers.build(slug)
        providers.complete(adapter, [{"role": "user", "content": "Hi"}],
                           "Reply with one word.", provider=slug)
        console.print(f"  [green]✓[/green]  [green]{name} connection successful![/green]")
    except (EnvironmentError, providers.ProviderError) as exc:
        console.print(f"  [red]✗[/red]  {exc}")
    time.sleep(1.8)


def _run_provider_setup(slug: str, name: str) -> None:
    """Show setup instructions for a provider and allow testing / activation."""
    from settings_manager import load_settings, save_settings

    steps = _PROVIDER_INSTRUCTIONS.get(slug, [])

    while True:
        console.clear()
        _print_compact_banner()
        settings = load_settings()
        is_active = settings["llm_provider"] == slug
        status = "[green]Active[/green]" if is_active else "[dim]Inactive[/dim]"

        console.print(Panel(
            f"[bold blue]{name}[/bold blue]  ·  {status}",
            border_style="blue", padding=(0, 2),
        ))
        console.print()
        console.print(Rule("[blue]Setup Instructions[/blue]", style="blue"))
        console.print()
        for step in steps:
            console.print(f"  {step}")
            console.print()
        console.print(Rule(style="dim blue"))
        console.print()
        console.print(f"  [bold blue]T[/bold blue]    Test connection")
        if not is_active:
            console.print(f"  [bold blue]A[/bold blue]    Set as active provider")
        console.print(f"  [bold blue]B[/bold blue]    Back")
        console.print()

        try:
            choice = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "B":
            break
        elif choice == "T":
            _test_provider(slug, name)
        elif choice == "A" and not is_active:
            settings["llm_provider"] = slug
            save_settings(settings)
            console.print(f"\n  [green]✓[/green]  Provider set to [bold white]{name}[/bold white].")
            time.sleep(0.8)
            break  # go back to provider list so checkmark updates


def _run_llm_provider_settings() -> None:
    """Provider selection screen."""
    from settings_manager import load_settings

    while True:
        console.clear()
        _print_compact_banner()
        current = load_settings()["llm_provider"]

        console.print(Panel(
            "[bold blue]LLM Provider[/bold blue]  "
            "[dim]· Choose which model handles your conversations[/dim]",
            border_style="blue", padding=(0, 2),
        ))
        console.print()
        for i, (slug, name, desc) in enumerate(_PROVIDERS, 1):
            marker = "[green]✓[/green]" if slug == current else " "
            tag    = "  [dim](default)[/dim]" if slug == "claude" else ""
            console.print(
                f"  [bold blue]{i}[/bold blue]  {marker}  [white]{name:<12}[/white]"
                f"  [dim]{desc}[/dim]{tag}"
            )
        console.print()
        console.print(Rule(style="dim blue"))
        console.print(f"\n  [bold blue]B[/bold blue]    Back\n")

        try:
            choice = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "B":
            break
        elif choice in ("1", "2", "3", "4"):
            slug, name, _ = _PROVIDERS[int(choice) - 1]
            _run_provider_setup(slug, name)


def _run_settings() -> None:
    """Top-level settings screen."""
    from settings_manager import load_settings, save_settings

    _provider_label = {s: n for s, n, _ in _PROVIDERS}

    while True:
        console.clear()
        _print_compact_banner()
        settings   = load_settings()
        cur_label  = _provider_label.get(settings["llm_provider"], settings["llm_provider"].title())
        dv_on      = settings["detailed_view"]
        dv_label   = "[green]On[/green]" if dv_on else "[dim]Off[/dim]"
        pc_on      = settings["presidio_comparison"]
        pc_label   = "[green]On[/green]" if pc_on else "[dim]Off[/dim]"

        console.print(Panel(
            "[bold blue]Settings[/bold blue]",
            border_style="blue", padding=(0, 2),
        ))
        console.print()
        console.print(
            f"  [bold blue]L[/bold blue]    [white]LLM Provider[/white]"
            f"    [dim]Current: {cur_label}[/dim]"
        )
        console.print()
        console.print(
            f"  [bold blue]D[/bold blue]    [white]Detailed View[/white]"
            f"    {dv_label}  [dim]— show pipeline logs, PII table & transparency panel[/dim]"
        )
        console.print()
        console.print(
            f"  [bold blue]C[/bold blue]    [white]Presidio Comparison[/white]"
            f"    {pc_label}  [dim]— show Presidio side-by-side panel in PII Finder[/dim]"
        )
        console.print()
        console.print(Rule(style="dim blue"))
        console.print(f"\n  [bold blue]B[/bold blue]    Back\n")

        try:
            choice = console.input("[bold blue]›[/bold blue]  ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "B":
            break
        elif choice == "L":
            _run_llm_provider_settings()
        elif choice == "D":
            settings["detailed_view"] = not dv_on
            save_settings(settings)
            new_label = "[green]On[/green]" if settings["detailed_view"] else "[dim]Off[/dim]"
            console.print(f"\n  [green]✓[/green]  Detailed View set to {new_label}.")
            time.sleep(0.6)
        elif choice == "C":
            settings["presidio_comparison"] = not pc_on
            save_settings(settings)
            new_label = "[green]On[/green]" if settings["presidio_comparison"] else "[dim]Off[/dim]"
            console.print(f"\n  [green]✓[/green]  Presidio Comparison set to {new_label}.")
            time.sleep(0.6)


# ─── Help ─────────────────────────────────────────────────────────────────────

def _run_help() -> None:
    from help_screen import print_help

    console.clear()
    _print_compact_banner()
    print_help(console)
    console.print(Rule(style="blue"))
    console.print()
    try:
        console.input("  [dim]Press Enter to return to dashboard…[/dim]")
    except (EOFError, KeyboardInterrupt):
        pass


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
        if upper == "J":
            console.clear(); _print_compact_banner(); _run_json_test(); continue
        if upper == "E":
            console.clear(); _print_compact_banner(); _run_evaluation(); continue
        if upper == "A":
            console.clear(); _print_compact_banner(); _run_attacker_experiment(); continue
        if upper == "S":
            _run_settings(); continue
        if upper == "H":
            _run_help(); continue

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
                    if _delete_conversation(uid):
                        console.print("[green]✓[/green]  Deleted.")
                    else:
                        console.print("[yellow]Already gone.[/yellow]")
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


def _delete_conversation(conv_id: str) -> bool:
    """Delete transcript and shadow map without decrypting either (so an
    unreadable conversation can still be removed). Returns False when neither
    existed; raises ValueError on an invalid id."""
    from chatbot.chat import ClaudeChat
    from storage.logic import erase, validate_id
    validate_id(conv_id)
    removed = ClaudeChat.delete(conv_id)
    return erase(conv_id) or removed


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
    except (ValueError, StorageError) as exc:
        # invalid id, undecryptable transcript, unusable device key (I11, I19)
        console.print(f"[red]Cannot open conversation:[/red] {exc}"); return
    except EnvironmentError as exc:
        console.print(f"[red bold]Configuration error:[/red bold] {exc}")
        console.print("[dim]Press S from the dashboard to configure your LLM provider.[/dim]"); return

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

    try:
        pipeline = Pipeline(chat=chat_handler, rag=rag_store)
    except StorageError as exc:
        # corrupt shadow map (moved aside) or unusable device key
        console.print(f"[red]Cannot open the surrogate map:[/red] {exc}"); return
    _run_chat_loop(pipeline, rag_mode=bool(effective_rag))


MULTILINE_FENCE = '"""'


def read_message(read_line) -> str:
    """Read one chat message (audit I26). A single line is a message; a line
    ending in a backslash continues on the next; a line starting with
    MULTILINE_FENCE (three double quotes) opens a block that ends at a line
    ending with the fence — for pasting multi-line text as one message."""
    first = read_line("[bold blue]You[/bold blue]  ")
    if first.lstrip().startswith(MULTILINE_FENCE):
        body = first.lstrip()[len(MULTILINE_FENCE):]
        lines = []
        while not body.rstrip().endswith(MULTILINE_FENCE):
            lines.append(body)
            body = read_line("[dim]  …[/dim]  ")
        lines.append(body.rstrip()[:-len(MULTILINE_FENCE)])
        return "\n".join(lines).strip()
    lines = [first]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(read_line("[dim]  …[/dim]  "))
    return "\n".join(lines).strip()


def _run_chat_loop(pipeline, rag_mode: bool) -> None:
    from chatbot.providers import ProviderAuthError
    from settings_manager import load_settings as _ls
    _settings = _ls()
    _detailed = _settings["detailed_view"]
    _set_detailed_logging(_detailed)

    provider_slug = getattr(pipeline.chat, "_provider", "claude")
    provider_name = next((n for s, n, _ in _PROVIDERS if s == provider_slug), "LLM")

    conv_id  = pipeline.chat.conversation.id
    mode_tag = "  [dim blue]· RAG[/dim blue]" if rag_mode else ""
    console.print()
    console.print(
        f"[dim]ID [/dim][blue]{conv_id}[/blue]{mode_tag}"
        "[dim]  ·  type [bold]exit[/bold] to return to dashboard  ·  "
        f"paste several lines between [bold]{MULTILINE_FENCE}[/bold] and [bold]{MULTILINE_FENCE}[/bold][/dim]"
    )
    console.print(Rule(style="blue"))
    console.print()

    while True:
        try:
            user_input = read_message(console.input)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]"); break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "back"}:
            console.print("[dim]Returning to dashboard...[/dim]")
            time.sleep(0.4); break

        try:
            response, _, _ = pipeline.process_turn(user_input, interactive=True)
        except (KeyboardInterrupt, EOFError):
            # history is only written after a reply (I18): nothing to undo
            console.print("\n[yellow]Turn cancelled.[/yellow]  [dim]Type exit to leave.[/dim]")
            continue
        except DetectorUnavailable as exc:
            _print_detector_unavailable(exc)
            break
        except (EnvironmentError, ProviderAuthError) as exc:
            console.print(f"[red]Configuration error.[/red]  [dim]{exc}[/dim]")
            break
        except Exception as exc:
            # the turn was not recorded (audit I18): the user can retype it
            console.print(f"[red bold]Error:[/red bold] {exc}  [dim](not sent — try again)[/dim]")
            continue

        console.print()
        console.print(Panel(
            response,
            title=f"[bold blue]{provider_name}[/bold blue]",
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
    yes:    bool           = typer.Option(False, "--yes", "-y", help="Delete without asking."),
    rag:    bool           = typer.Option(False, "--rag",    help="Enable RAG mode."),
) -> None:
    """Start, resume, or delete a conversation."""
    _print_compact_banner()
    if delete:
        if not yes and not typer.confirm(f"Delete conversation {delete}?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(1)
        try:
            found = _delete_conversation(delete)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        if not found:
            console.print(f"[red]No conversation {delete}[/red]")
            raise typer.Exit(1)
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
        n, _ = anonymise_for_rag(raw_text, rag_store, source=path.name)
        console.print(
            f"[green]✓[/green]  {n} chunks indexed  "
            f"[dim](total: {rag_store.document_count()})[/dim]"
        )
    except DetectorUnavailable as exc:
        _print_detector_unavailable(exc); raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}"); raise typer.Exit(1)


@app.command(name="rag")
def rag_command(
    forget: Optional[str] = typer.Option(None, "--forget", metavar="DOC",
                                         help="Remove a document (file name or id) and its mappings."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
) -> None:
    """List indexed RAG documents, or remove one with --forget."""
    from pipeline import forget_rag_document
    rag_store = _get_rag()
    docs = rag_store.documents()
    if forget is None:
        if not docs:
            console.print("[dim]No documents indexed.[/dim]"); return
        for doc_id, row in sorted(docs.items(), key=lambda kv: kv[1]["source"]):
            console.print(f"  [white]{row['source']}[/white]  [dim]{doc_id} · {row['chunks']} chunks[/dim]")
        return
    if not any(forget in (d, row["source"]) for d, row in docs.items()):
        console.print(f"[red]No indexed document named {forget!r}.[/red]"); raise typer.Exit(1)
    if not yes and not typer.confirm(f"Remove {forget!r} from the RAG index?"):
        raise typer.Exit(0)
    removed, erased = forget_rag_document(forget, rag_store)
    console.print(f"[green]✓[/green]  removed {removed} chunks, erased {erased} mappings")


if __name__ == "__main__":
    app()