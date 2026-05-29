"""
surrogateshield/_display.py — Output display helpers

Uses Rich tables when available; falls back to plain print otherwise.
"""

from __future__ import annotations

try:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    _console = _Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def show_scan_results(entities: list, pii_off: list) -> None:
    """Print a table of detected PII entities from scan()."""
    if not entities:
        print("[SurrogateShield] No PII detected.")
        return

    pii_off_lower = {t.lower() for t in (pii_off or [])}

    if HAS_RICH:
        table = _Table(title="[bold cyan]SurrogateShield — Scan Results[/bold cyan]", show_lines=True)
        table.add_column("Detected Value", style="red bold")
        table.add_column("Type", style="yellow")
        table.add_column("Score", style="white")
        table.add_column("Source", style="dim")
        for ent in entities:
            skipped = ent.type.lower() in pii_off_lower
            note = "[dim](skipped — pii_off)[/dim]" if skipped else ""
            table.add_row(
                ent.text,
                ent.type,
                f"{ent.score:.2f}",
                ent.source,
                note,
            )
        _console.print(table)
    else:
        print("\n[SurrogateShield] Scan Results")
        print(f"{'Detected Value':<30} {'Type':<20} {'Score':<8} {'Source':<10}")
        print("-" * 70)
        for ent in entities:
            skipped = ent.type.lower() in pii_off_lower
            note = " (skipped — pii_off)" if skipped else ""
            print(f"{ent.text:<30} {ent.type:<20} {ent.score:<8.2f} {ent.source:<10}{note}")
        print()


def show_mask_results(entities: list, surrogate_map: dict) -> None:
    """Print a table showing original PII and their surrogates."""
    if not entities:
        return

    if HAS_RICH:
        table = _Table(title="[bold cyan]SurrogateShield — Masked[/bold cyan]", show_lines=True)
        table.add_column("Original", style="red bold")
        table.add_column("Type", style="yellow")
        table.add_column("Score", style="white")
        table.add_column("Source", style="dim")
        table.add_column("Surrogate", style="green bold")
        for ent in entities:
            surrogate = surrogate_map.get(ent.text, "[dim]—[/dim]")
            table.add_row(
                ent.text,
                ent.type,
                f"{ent.score:.2f}",
                ent.source,
                surrogate,
            )
        _console.print(table)
    else:
        print("\n[SurrogateShield] Mask Results")
        print(f"{'Original':<30} {'Type':<20} {'Score':<8} {'Source':<10} {'Surrogate':<30}")
        print("-" * 100)
        for ent in entities:
            surrogate = surrogate_map.get(ent.text, "—")
            print(f"{ent.text:<30} {ent.type:<20} {ent.score:<8.2f} {ent.source:<10} {surrogate:<30}")
        print()


def show_unmask_results(restored_count: int) -> None:
    """Print a one-liner confirming how many surrogates were restored."""
    msg = f"[SurrogateShield] Restored {restored_count} surrogate(s)"
    if HAS_RICH:
        _console.print(f"[green]{msg}[/green]")
    else:
        print(msg)


def show_flush() -> None:
    """Print a one-liner confirming the session was cleared."""
    msg = "[SurrogateShield] Session memory cleared"
    if HAS_RICH:
        _console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(msg)
