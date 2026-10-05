# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
help_screen.py — SurrogateShield help content (displayed via main menu H key).
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Dashboard actions: (key, name, description, shown only with saved conversations).
# The main menu and the help screen both render this list (audit I28).
MENU_ITEMS = [
    ("N",       "New conversation",       "Start a fresh chat with PII protection active", False),
    ("R",       "New conversation + RAG", "Chat grounded in your indexed documents", False),
    ("P",       "PII Finder",             "Test detection on any text — zero API calls", False),
    ("1 – 9",   "Open conversation",      "Resume a saved conversation by number", True),
    ("D1 – D9", "Delete conversation",    "Permanently remove a saved conversation", True),
    ("J",       "JSON Test",              "Batch-process a JSON file of questions through the pipeline", False),
    ("E",       "Evaluation",             "Score pipeline quality against a ground-truth key file", False),
    ("A",       "Attacker Experiment",    "Ask a second model to recover the originals from sent text", False),
    ("S",       "Settings",               "Configure LLM provider, detailed view, Presidio comparison", False),
    ("H",       "Help",                   "Show this screen", False),
    ("Q",       "Quit",                   "Exit SurrogateShield", False),
]

NER_TYPES = "PERSON · ORG · GPE · LOC · FAC"
PATTERN_TYPES = (
    "email · phone (US, UK, international) · SSN · address · ZIP / UK postcode · "
    "date of birth · age · credit card · IBAN · bank routing number · passport · "
    "driver licence · ID and account numbers · licence plate · VIN · IP / MAC address · "
    "crypto wallet · API key · password / PIN / code · handle · personal URL · hostname"
)


def print_help(console: Console) -> None:
    console.print(Panel(
        Text.assemble(
            ("SurrogateShield", "bold white"),
            ("  — Privacy-preserving proxy for LLMs\n\n", "dim"),
            (
                "Your messages are intercepted locally, PII is replaced with realistic\n"
                "fake values (surrogates), the sanitized text is sent to the LLM, and\n"
                "real values are restored in the response. ",
                "dim",
            ),
            ("Your data never leaves your device.", "bold white"),
        ),
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    console.print(Rule("[bold blue]Detection Pipeline[/bold blue]", style="blue"))
    console.print()
    stages = [
        ("PatternScan",    "Regex — SSNs, emails, phones, credit cards, API keys"),
        ("EntityTrace",    "spaCy NER — names, places, organisations"),
        ("ContextGuard",   "DistilBERT NER — borderline / ambiguous entities"),
        ("MimicGen",       "Generates realistic fake values via Faker"),
        ("ShadowMap",      "AES-256-GCM encrypted surrogate map — stays on device"),
        ("LLM API",        "Receives surrogates only — real values never transmitted"),
        ("ResolvePass",    "Swaps surrogate values back to real values in the response"),
    ]
    for i, (name, desc) in enumerate(stages, 1):
        console.print(
            f"  [bold blue]{i}[/bold blue]"
            f"  [bold white]{name:<14}[/bold white]"
            f"  [dim]{desc}[/dim]"
        )
        if i < len(stages):
            console.print("   [blue]│[/blue]")
    console.print()

    # ── Menu options ──────────────────────────────────────────────────────────
    console.print(Rule("[bold blue]Menu Options[/bold blue]", style="blue"))
    console.print()
    for key, name, desc, _ in MENU_ITEMS:
        console.print(
            f"  [bold blue]{key:<8}[/bold blue]"
            f"  [white]{name:<26}[/white]"
            f"  [dim]{desc}[/dim]"
        )
    console.print()

    # ── PII types ─────────────────────────────────────────────────────────────
    console.print(Rule("[bold blue]PII Types Detected[/bold blue]", style="blue"))
    console.print()
    console.print(f"  [white]NER (spaCy / DistilBERT)[/white]   [dim]{NER_TYPES}[/dim]")
    console.print(f"  [white]Pattern and structural   [/white]   [dim]{PATTERN_TYPES}[/dim]")
    console.print()

    # ── RAG & JSON testing ────────────────────────────────────────────────────
    console.print(Rule("[bold blue]RAG & Batch Testing[/bold blue]", style="blue"))
    console.print()
    console.print("  [white]RAG[/white]         [dim]Index documents:[/dim]  [cyan]python main.py add-doc <filepath>[/cyan]")
    console.print("              [dim]Answers are grounded in indexed content; PII is anonymised before indexing.[/dim]")
    console.print()
    console.print("  [white]JSON Test[/white]   [dim]Input:[/dim]   [cyan]experiment/<name>.json[/cyan]  →  [cyan][ {\"input\": \"...\"}, … ][/cyan]")
    console.print("              [dim]Output:[/dim]  [cyan]experiment/<name>_answers.json[/cyan]  [dim](auto-saved every 25 questions)[/dim]")
    console.print()
    console.print("  [white]Evaluation[/white]  [dim]Requires a questions file, an answers file, and a ground-truth key file.[/dim]")
    console.print("              [dim]Reports precision, recall, F1, leak rates, and per-entity-type breakdown.[/dim]")
    console.print()

    # ── Settings quick-ref ────────────────────────────────────────────────────
    console.print(Rule("[bold blue]Settings Quick Reference[/bold blue]", style="blue"))
    console.print()
    settings_items = [
        ("LLM Provider",         "Claude · Gemini · ChatGPT · Local (Ollama)"),
        ("Detailed View",         "Show full pipeline logs and PII transparency panel in chat"),
        ("Presidio Comparison",   "Side-by-side Presidio panel in PII Finder (for research)"),
    ]
    for name, desc in settings_items:
        console.print(f"  [white]{name:<24}[/white]  [dim]{desc}[/dim]")
    console.print()
