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
import os
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
    help="Privacy-preserving CLI proxy for LLMs — PII never leaves your device.",
    add_completion=False,
)

_rag_store = None
VERSION = "v1.0"
TAGLINE = "Privacy-preserving proxy for LLMs  ·  PII never leaves your device"

# (slug, display name, short description)
_PROVIDERS = [
    ("claude",  "Claude",    "Anthropic  ·  claude-sonnet-4-6"),
    ("gemini",  "Gemini",    "Google     ·  gemini-1.5-flash"),
    ("chatgpt", "ChatGPT",   "OpenAI     ·  gpt-4o-mini"),
    ("local",   "Local LLM", "Ollama     ·  runs fully offline"),
]


def _current_provider_name() -> str:
    """Return the display name of the currently configured LLM provider."""
    from settings_manager import load_settings
    slug = load_settings().get("llm_provider", "claude")
    return next((n for s, n, _ in _PROVIDERS if s == slug), "LLM")


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
    from settings_manager import load_settings
    provider_slug = load_settings().get("llm_provider", "claude")
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
    rows.append(("[bold blue]J[/bold blue]", "JSON Test  — batch evaluation from a JSON file"))
    rows.append(("[bold blue]E[/bold blue]", "Evaluation  — score pipeline quality from JSON files"))
    rows.append(("[bold blue]S[/bold blue]", "Settings"))
    rows.append(("[bold blue]Q[/bold blue]", "Quit"))
    for key, desc in rows:
        console.print(f"  {key}    [dim]{desc}[/dim]")
    console.print()


# ─── PII Finder ───────────────────────────────────────────────────────────────

def _run_pii_finder() -> None:
    from settings_manager import load_settings as _ls
    _detailed = _ls().get("detailed_view", False)
    logging.getLogger().setLevel(logging.INFO if _detailed else logging.ERROR)
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
                    f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{fuzzed}[/blue]",
                    border_style="blue", padding=(1, 2),
                ))
            else:
                console.print(Panel(
                    "[bold blue]Service query[/bold blue]  "
                    "[dim]· No specific street address found[/dim]\n\n"
                    "[dim]Location names are not PII in service queries — "
                    "message would be sent unchanged.[/dim]\n\n"
                    f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{user_input}[/blue]",
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
                f"[dim]This message would be sent to {_current_provider_name()} unchanged.[/dim]",
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

        if _detailed:
            from detection.quasi_identifier import format_warning as _qi_fmt
            qi_matches = getattr(confirmed, "_qi_matches", [])
            if qi_matches:
                console.print(f"[bold yellow]{_qi_fmt(qi_matches)}[/bold yellow]")
            else:
                console.print("[dim green]✓  No quasi-identifier combination risk detected.[/dim green]")
            console.print()

        sanitised = user_input
        for orig in sorted(surrogate_map, key=len, reverse=True):
            sanitised = sanitised.replace(orig, surrogate_map[orig])

        console.print(Panel(
            f"[dim]Would send to {_current_provider_name()}:[/dim]\n[blue]{sanitised}[/blue]",
            border_style="dim blue", padding=(0, 2),
        ))
        console.print()


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
    except Exception as exc:
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
        if status == "ok":
            console.print(f"  [green]✓[/green]  {idx}  [dim]0.0ms[/dim]")
        elif status == "error":
            console.print(f"  [red]✗[/red]  {idx}  [red]error[/red]")

    try:
        metrics = run_evaluation(
            questions_file, answers_file, key_file,
            fields, progress_cb=_on_progress,
        )
    except Exception as exc:
        console.print(f"\n  [red bold]Error:[/red bold] {exc}")
        console.print()
        try:
            console.input("  [dim]Press Enter to return to dashboard…[/dim]")
        except (EOFError, KeyboardInterrupt):
            pass
        return

    # ── Screen 4: Results ─────────────────────────────────────────────────────
    console.clear()
    _print_compact_banner()
    from rich.columns import Columns as _Columns

    # ── Formatting helpers ────────────────────────────────────────────────────
    def _fmt(key: str, val) -> str:
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            if key in {
                "answer_rate", "precision_surrogates", "recall_surrogates",
                "f1_surrogates", "accuracy_surrogates", "error_surrogates",
                "resolve_leak_rate", "precision_resolve", "error_resolve",
                "pii_leak_rate", "accuracy_sanitization", "error_sanitization",
            }:
                return f"{val * 100:.2f}%"
            if key in {
                "avg_pattern_scan_ms", "avg_entity_trace_ms",
                "avg_context_guard_ms", "avg_surrogate_gen_ms",
            }:
                return f"{val:.2f} ms"
            return f"{val:.2f}"
        return str(val)

    def _color_surr(key: str, val: float) -> str:
        """3-tier color for Surrogate Detection quality metrics."""
        pct = f"{val * 100:.2f}%"
        if key == "error_surrogates":
            if val == 0.0:   return f"[bold green]{pct}[/bold green]"
            if val <= 0.10:  return f"[yellow]{pct}[/yellow]"
            return f"[red]{pct}[/red]"
        if val >= 0.90:      return f"[bold green]{pct}[/bold green]"
        if val >= 0.70:      return f"[yellow]{pct}[/yellow]"
        return f"[red]{pct}[/red]"

    def _color_pipe(key: str, val: float) -> str:
        """Binary color for ResolvePass / Sanitization metrics."""
        pct = f"{val * 100:.2f}%"
        if key in {"precision_resolve", "accuracy_sanitization"}:
            return f"[bold green]{pct}[/bold green]" if val >= 0.90 else f"[red]{pct}[/red]"
        return f"[bold green]{pct}[/bold green]" if val <= 0.10 else f"[red]{pct}[/red]"

    console.print(Rule(
        f"[bold blue]Evaluation Results[/bold blue]  [dim]· {questions_file}[/dim]",
        style="blue",
    ))
    console.print()

    # ── Section 1: Overview (full-width 4-column table) ───────────────────────
    ov = Table(
        title="[bold blue]Overview[/bold blue]",
        box=box.ROUNDED, border_style="blue", padding=(0, 2), expand=True,
    )
    ov.add_column("Questions",   style="white",     justify="center")
    ov.add_column("Answered",    style="white",     justify="center")
    ov.add_column("Empty",       style="white",     justify="center")
    ov.add_column("Answer Rate", style="bold blue", justify="center")
    ov.add_row(
        str(metrics.get("no_of_questions", total)),
        str(metrics.get("no_of_answers",       "—")),
        str(metrics.get("no_of_answers_empty", "—")),
        _fmt("answer_rate", metrics["answer_rate"]) if "answer_rate" in metrics else "—",
    )
    console.print(ov)
    console.print()

    # ── Section 2: Surrogate Detection | Stage Timings ────────────────────────
    sd = Table(box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1), expand=True)
    sd.add_column("Metric",  style="white", no_wrap=True)
    sd.add_column("Value",   justify="right")
    sd_rows = 0
    for k, lbl in [
        ("no_surrogates_found",               "Found (total)"),
        ("no_surrogates_in_key",              "In key (total)"),
        ("avg_surrogates_per_question_found",  "Avg found / question"),
        ("avg_surrogates_per_question_in_key", "Avg in key / question"),
    ]:
        if k in metrics:
            sd.add_row(lbl, _fmt(k, metrics[k]))
            sd_rows += 1
    quality_rows = [(k, lbl) for k, lbl in [
        ("precision_surrogates", "Precision"),
        ("recall_surrogates",    "Recall"),
        ("f1_surrogates",        "F1 Score"),
        ("accuracy_surrogates",  "Accuracy (Jaccard)"),
        ("error_surrogates",     "Error (miss rate)"),
    ] if k in metrics]
    if quality_rows:
        if sd_rows:
            sd.add_section()
        for k, lbl in quality_rows:
            sd.add_row(lbl, _color_surr(k, metrics[k]))
        sd_rows += len(quality_rows)

    st = Table(box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1), expand=True)
    st.add_column("Stage",    style="white", no_wrap=True)
    st.add_column("Avg Time", justify="right")
    st_rows = 0
    for k, lbl in [
        ("avg_pattern_scan_ms",  "PatternScan"),
        ("avg_entity_trace_ms",  "EntityTrace"),
        ("avg_context_guard_ms", "ContextGuard"),
        ("avg_surrogate_gen_ms", "SurrogateGen"),
    ]:
        if k in metrics:
            val = metrics[k]
            cell = f"[yellow]{val:.2f} ms[/yellow]" if k == "avg_entity_trace_ms" and val > 500 else f"{val:.2f} ms"
            st.add_row(lbl, cell)
            st_rows += 1

    left2  = Panel(sd, title="[bold blue]Surrogate Detection[/bold blue]", border_style="blue", padding=(1, 2)) if sd_rows else None
    right2 = Panel(st, title="[bold blue]Stage Timings[/bold blue]",       border_style="blue", padding=(1, 2)) if st_rows else None
    if left2 and right2:
        console.print(_Columns([left2, right2], equal=True, expand=True))
    elif left2:
        console.print(left2)
    elif right2:
        console.print(right2)
    if left2 or right2:
        console.print()

    # ── Section 3: ResolvePass Quality | Sanitization Quality ─────────────────
    rp = Table(box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1), expand=True)
    rp.add_column("Metric", style="white", no_wrap=True)
    rp.add_column("Value",  justify="right")
    rp_rows = 0
    for k, lbl in [
        ("total_resolve_leaks",            "Questions with leaks"),
        ("total_individual_resolve_leaks", "Individual leaked"),
    ]:
        if k in metrics:
            rp.add_row(lbl, _fmt(k, metrics[k]))
            rp_rows += 1
    for k, lbl in [
        ("resolve_leak_rate", "Leak rate"),
        ("precision_resolve", "Accuracy"),
        ("error_resolve",     "Error rate"),
    ]:
        if k in metrics:
            rp.add_row(lbl, _color_pipe(k, metrics[k]))
            rp_rows += 1

    sq = Table(box=box.ROUNDED, border_style="blue", show_lines=True, padding=(0, 1), expand=True)
    sq.add_column("Metric", style="white", no_wrap=True)
    sq.add_column("Value",  justify="right")
    sq_rows = 0
    for k, lbl in [
        ("total_pii_leaks_to_llm",     "Questions with PII leaked"),
        ("total_individual_pii_leaks", "Individual PII leaked"),
    ]:
        if k in metrics:
            sq.add_row(lbl, _fmt(k, metrics[k]))
            sq_rows += 1
    for k, lbl in [
        ("pii_leak_rate",         "PII leak rate"),
        ("accuracy_sanitization", "Accuracy"),
        ("error_sanitization",    "Error rate"),
    ]:
        if k in metrics:
            sq.add_row(lbl, _color_pipe(k, metrics[k]))
            sq_rows += 1

    left3  = Panel(rp, title="[bold blue]ResolvePass Quality[/bold blue]",  border_style="blue", padding=(1, 2)) if rp_rows else None
    right3 = Panel(sq, title="[bold blue]Sanitization Quality[/bold blue]", border_style="blue", padding=(1, 2)) if sq_rows else None
    if left3 and right3:
        console.print(_Columns([left3, right3], equal=True, expand=True))
    elif left3:
        console.print(left3)
    elif right3:
        console.print(right3)
    if left3 or right3:
        console.print()

    # ── Section 4: Summary Banner ─────────────────────────────────────────────
    san_acc = metrics.get("accuracy_sanitization", 1.0)
    res_acc = metrics.get("precision_resolve",      1.0)
    if san_acc == 1.0 and res_acc == 1.0:
        summary_style = "green"
        summary_msg   = "✓  Pipeline fully clean — no PII leaked to LLM, all surrogates restored"
    elif san_acc > 0.80 and res_acc > 0.80:
        summary_style = "yellow"
        summary_msg   = "⚠  Minor issues detected — review leak details above"
    else:
        summary_style = "red"
        summary_msg   = "✗  Significant leaks detected — pipeline needs attention"
    console.print(Panel(summary_msg, border_style=summary_style, padding=(0, 2)))
    console.print()

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
            stem     = Path(questions_file).stem
            out_path = EXPERIMENT_DIR / f"{stem}_eval_results.json"
            out_path.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
            console.print(f"\n  [green]✓[/green]  Saved to [cyan]{out_path}[/cyan]\n")


# ─── Settings ─────────────────────────────────────────────────────────────────

_PROVIDER_INSTRUCTIONS: dict = {
    "claude": [
        "1. Visit [blue]console.anthropic.com[/blue] and sign in.",
        "2. Go to [bold white]API Keys[/bold white] and create a new key.",
        "3. Add to your [bold white].env[/bold white] file:\n\n"
        "       [cyan]ANTHROPIC_API_KEY=sk-ant-...[/cyan]",
        "4. Press [bold white]T[/bold white] to test your current key.",
    ],
    "gemini": [
        "1. Visit [blue]aistudio.google.com[/blue] and sign in.",
        "2. Click [bold white]Get API Key[/bold white] to generate a key.",
        "3. Add to your [bold white].env[/bold white] file:\n\n"
        "       [cyan]GEMINI_API_KEY=AIza...[/cyan]",
        "4. Install the SDK:\n\n"
        "       [cyan]pip install google-generativeai[/cyan]",
        "5. Press [bold white]T[/bold white] to test your current key.",
    ],
    "chatgpt": [
        "1. Visit [blue]platform.openai.com[/blue] and sign in.",
        "2. Go to [bold white]API Keys[/bold white] and create a new secret key.",
        "3. Add to your [bold white].env[/bold white] file:\n\n"
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
    load_dotenv(override=True)  # pick up any keys just added to .env
    console.print(f"\n  [dim]Testing {name} connection…[/dim]")
    try:
        if slug == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                console.print("  [red]✗[/red]  ANTHROPIC_API_KEY not set in .env")
                time.sleep(1.5); return
            import anthropic as _ant
            r = _ant.Anthropic(api_key=api_key).messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            _ = r.content[0].text

        elif slug == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                console.print("  [red]✗[/red]  GEMINI_API_KEY not set in .env")
                time.sleep(1.5); return
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            _ = genai.GenerativeModel("gemini-1.5-flash").generate_content("Hi").text

        elif slug == "chatgpt":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                console.print("  [red]✗[/red]  OPENAI_API_KEY not set in .env")
                time.sleep(1.5); return
            import openai as _oai
            r = _oai.OpenAI(api_key=api_key).chat.completions.create(
                model="gpt-4o-mini", max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            _ = r.choices[0].message.content

        elif slug == "local":
            import ollama as _ol
            host  = os.environ.get("LOCAL_LLM_HOST", "http://localhost:11434")
            model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
            r = _ol.Client(host=host).chat(
                model=model, messages=[{"role": "user", "content": "Hi"}]
            )
            _ = r.message.content

        console.print(f"  [green]✓[/green]  [green]{name} connection successful![/green]")
    except ImportError as exc:
        console.print(f"  [red]✗[/red]  Package not installed: {exc}")
    except Exception as exc:
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
        dv_on      = settings.get("detailed_view", False)
        dv_label   = "[green]On[/green]" if dv_on else "[dim]Off[/dim]"

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
        if upper == "S":
            _run_settings(); continue

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

    _run_chat_loop(Pipeline(chat=chat_handler, rag=rag_store), rag_mode=bool(effective_rag))


def _run_chat_loop(pipeline, rag_mode: bool) -> None:
    from settings_manager import load_settings as _ls
    _settings = _ls()
    _detailed = _settings.get("detailed_view", False)
    logging.getLogger().setLevel(logging.INFO if _detailed else logging.ERROR)

    provider_slug = getattr(pipeline.chat, "_provider", "claude")
    provider_name = next((n for s, n, _ in _PROVIDERS if s == provider_slug), "LLM")

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
        except EnvironmentError as exc:
            console.print(f"[red]Configuration error.[/red]  [dim]{exc}[/dim]")
            break
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] {exc}"); continue

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