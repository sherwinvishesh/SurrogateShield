"""Audit I28 — one menu list for the dashboard and the help screen (the
help screen had no Attacker entry), a current PII type list, and no dead
constants in main.py.
"""

import io

import pytest
from rich.console import Console

pytest.importorskip("typer")

import help_screen
import main


def test_I28_help_and_menu_share_every_action(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(main, "console", Console(file=buf, width=200, color_system=None))
    main._print_menu(has_convs=True)
    menu = buf.getvalue()
    out = io.StringIO()
    help_screen.print_help(Console(file=out, width=200, color_system=None))
    help_text = out.getvalue()
    for key, name, _desc, _convs in help_screen.MENU_ITEMS:
        assert key in menu and name in menu, key
        assert key in help_text and name in help_text, key
    assert "Attacker Experiment" in help_text


def test_I28_conversation_actions_only_with_conversations(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(main, "console", Console(file=buf, width=200, color_system=None))
    main._print_menu(has_convs=False)
    assert "Open conversation" not in buf.getvalue()


@pytest.mark.parametrize("word", ["IBAN", "passport", "handle", "personal URL", "age", "VIN"])
def test_I28_help_lists_current_pattern_types(word):
    assert word in help_screen.PATTERN_TYPES


def test_I28_dead_main_constants_removed():
    assert not hasattr(main, "VERSION") and not hasattr(main, "TAGLINE")


def test_I28_pii_finder_docstring_is_its_docstring():
    assert main._run_pii_finder.__doc__ and "sandbox" in main._run_pii_finder.__doc__


def test_readme_shortcut_table_lists_exactly_the_menu_keys():
    """The README's "Dashboard Keyboard Shortcuts" table is MENU_ITEMS, in order."""
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    section = text[text.index("### Dashboard Keyboard Shortcuts"):]
    section = section[:re.search(r"\n#{2,3} ", section[10:]).start() + 10]
    readme = [k.replace(" ", "") for k in re.findall(r"^\| `([^`]+)` \|", section, re.M)]
    menu = [key.replace(" ", "") for key, *_ in help_screen.MENU_ITEMS]
    assert readme == menu

