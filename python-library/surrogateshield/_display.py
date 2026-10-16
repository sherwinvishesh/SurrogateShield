"""
surrogateshield/_display.py — the ``detailed_view`` tables.

Only called when ``detailed_view=True``. They print ORIGINAL values to stdout,
which is why that setting is off by default (audit I10). Uses Rich when the
``display`` extra is installed, plain print otherwise.
"""

from __future__ import annotations

try:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    _console = _Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _table(title, columns, rows) -> None:
    if HAS_RICH:
        table = _Table(title=f"[bold cyan]{title}[/bold cyan]", show_lines=True)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        _console.print(table)
    else:
        print(f"\n[{title}]")
        print("  ".join(f"{c:<24}" for c in columns))
        for row in rows:
            print("  ".join(f"{c:<24}" for c in row))
        print()


def show_scan_results(detections) -> None:
    """Print the detections of ``scan()``."""
    if not detections:
        print("[SurrogateShield] No PII detected.")
        return
    _table("SurrogateShield — Scan Results",
           ("Detected Value", "Type", "Span", "Score", "Source", "Masked"),
           [(d.text, d.type, f"{d.start}-{d.end}", f"{d.score:.2f}", d.source,
             "yes" if d.masked else "no (pii_off)") for d in detections])


def show_mask_results(result) -> None:
    """Print what ``mask()`` replaced (a :class:`MaskResult`)."""
    if not result.detections:
        return
    _table("SurrogateShield — Masked",
           ("Original", "Type", "Score", "Source", "Surrogate"),
           [(d.text, d.type, f"{d.score:.2f}", d.source,
             result.replacements.get(d.text.strip(), "— (pii_off)" if not d.masked else "—"))
            for d in result.detections])


def show_unmask_results(restored_count: int) -> None:
    """Print how many surrogates were found in the response."""
    msg = f"[SurrogateShield] Restored {restored_count} surrogate(s)"
    if HAS_RICH:
        _console.print(f"[green]{msg}[/green]")
    else:
        print(msg)
