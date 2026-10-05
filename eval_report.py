# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
eval_report.py — render an ``evaluator.run_evaluation`` result with Rich.

Every number and sentence comes from the result dict (audit A11/I23): a
missing value prints ``n/a``, no conclusion is written that the data did not
compute, and deltas keep their sign.
"""

from __future__ import annotations

from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

NA = "[dim]n/a[/dim]"


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return NA
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v: Optional[float]) -> str:
    return NA if v is None else f"{v * 100:.2f}%"


def _signed(v: Optional[float]) -> str:
    if v is None:
        return NA
    color = "green" if v > 0 else "red" if v < 0 else "dim"
    return f"[{color}]{v:+.4f}[/{color}]"


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right")
    for k, v in rows:
        t.add_row(k, v)
    return t


def _prf_table(title: str, blocks: list[tuple[str, dict]]) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAVY, title_justify="left")
    for col in ("", "P", "R", "F1", "gold", "tp", "fn", "pred", "fp", "exact"):
        t.add_column(col, justify="left" if not col else "right")
    for label, m in blocks:
        t.add_row(label, *(_num(m.get(k)) for k in ("precision", "recall", "f1")),
                  *(_num(m.get(k)) for k in ("gold", "tp", "fn", "pred", "fp",
                                              "exact_boundary_matches")))
    return t


def _macro_line(label: str, ma: dict) -> str:
    return (f"{label} macro  P {_num(ma.get('precision'))}  R {_num(ma.get('recall'))}  "
            f"F1 {_num(ma.get('f1'))}  over {_num(ma.get('n_types'))} types")


def render_overview(r: dict, console: Console) -> None:
    rows = [("questions", _num(r.get("questions"))),
            ("rows scored", _num(r.get("rows_scored"))),
            ("error rows (excluded)", _num(r.get("error_rows")))]
    if r.get("error_types"):
        rows.append(("error types", ", ".join(f"{k}: {v}" for k, v in r["error_types"].items())))
    if r.get("span_source"):
        rows.append(("prediction spans from", ", ".join(f"{k}: {v}" for k, v in r["span_source"].items())))
    a = r.get("answers")
    if a:
        rows += [("answered", _num(a.get("answered"))), ("empty answers", _num(a.get("empty"))),
                 ("answer rate (all rows)", _pct(a.get("answer_rate"))),
                 ("answer rate (excluding errors)", _pct(a.get("answer_rate_excluding_errors")))]
    sc = r.get("surrogate_counts")
    if sc:
        rows += [("replaced values", _num(sc.get("replaced_values"))),
                 ("gold values in text", _num(sc.get("gold_values_in_text"))),
                 ("gold values not in their question", _num(sc.get("gold_values_not_in_text")))]
    meta = r.get("run_meta") or {}
    gen = meta.get("generator") or {}
    if meta:
        rows += [("generator commit", _num(gen.get("commit"))),
                 ("generator dirty", _num(gen.get("dirty"))),
                 ("base seed", _num(meta.get("base_seed")))]
    else:
        rows.append(("run metadata", "[yellow]none — answers file has no .meta.json[/yellow]"))
    console.print(Panel(_kv_table(rows), title="[bold blue]Overview[/bold blue]",
                        border_style="blue", padding=(1, 2)))


def render_detection(d: dict, console: Console) -> None:
    prot, recog = d["protection"], d["recognition"]
    console.print(_prf_table(
        "Detection — micro, span overlap (protection is the headline)",
        [("protection", prot["micro"]), ("recognition", recog["micro"])]))
    console.print("  " + _macro_line("protection ", prot["macro"]))
    console.print("  " + _macro_line("recognition", recog["macro"]))
    neg = prot["negatives"]
    console.print(f"  negatives: {_num(neg.get('negative_questions_with_any_prediction'))} of "
                  f"{_num(neg.get('negative_questions'))} negative questions had a replaced span "
                  f"({_num(neg.get('fp_spans_on_negatives'))} spans)")
    console.print()
    pt = prot.get("per_type") or {}
    if pt:
        console.print(_prf_table("Protection per type", sorted(pt.items())))


def render_sanitization(s: dict, console: Console) -> None:
    rows = [
        ("gold values in input", _num(s.get("gold_values_in_input"))),
        ("verbatim in sent text", f"{_num(s.get('leaked_values'))}  ({_pct(s.get('leak_rate'))})"),
        ("  deliberate (documented policy)", _num(s.get("deliberate_leaks"))),
        ("  unintended", f"{_num(s.get('unintended_leaks'))}  ({_pct(s.get('unintended_leak_rate'))})"),
        ("rows with an unintended leak", _num(s.get("rows_with_unintended_leak"))),
        ("shift mismatches", _num(s.get("shift_mismatches"))),
    ]
    for reason, n in sorted((s.get("deliberate_by_reason") or {}).items()):
        rows.append((f"  policy: {reason}", _num(n)))
    console.print(Panel(_kv_table(rows), title="[bold blue]Sent to the LLM[/bold blue]",
                        border_style="blue", padding=(1, 2)))
    ex = s.get("unintended_examples") or []
    if ex:
        console.print(f"  [dim]first unintended leaks ({len(ex)} shown):[/dim]")
        for e in ex[:10]:
            console.print(f"    row {e['row']}  {e['type']}: {e['value']!r}  [dim]{', '.join(e['reasons']) or 'no reason'}[/dim]")
    console.print()


def render_restoration(r: dict, console: Console) -> None:
    rows = [
        ("source", r.get("source", "n/a")),
        ("rows recomputed with ResolvePass", _num(r.get("rows_recomputed_with_resolvepass"))),
        ("answers scored", _num(r.get("rows"))),
        ("answers with replacements", _num(r.get("rows_with_replacements"))),
        ("surrogates left after restore", f"{_num(r.get('surrogates_left'))} in {_num(r.get('rows_with_surrogate_left'))} rows"),
        ("over-restored originals", _num(r.get("over_restored_values"))),
        ("collateral edits", f"{_num(r.get('collateral_edits'))} in {_num(r.get('rows_with_collateral_edit'))} rows"),
    ]
    console.print(Panel(_kv_table(rows), title="[bold blue]Restoration (final_output)[/bold blue]",
                        border_style="blue", padding=(1, 2)))


def render_presidio(pc: dict, console: Console) -> None:
    if not pc.get("available"):
        console.print(Panel(f"Presidio comparison not available: {pc.get('reason', 'n/a')}",
                            border_style="yellow", padding=(0, 2)))
        return
    console.print(f"  [dim]rows: {_num(pc['rows'])}  ({pc['data_status']} data; "
                  f"{_num(pc['rows_without_presidio_data'])} rows without Presidio output)[/dim]")
    console.print(f"  [dim]universe: {', '.join(pc['universe'])}[/dim]")
    console.print(f"  [dim]{pc['matching']}[/dim]")
    cfg = pc.get("presidio_config")
    console.print(f"  [dim]Presidio config: {cfg if cfg else 'not recorded in run metadata'}[/dim]")
    for mode in ("untyped", "typed"):
        console.print(_prf_table(
            f"SurrogateShield vs Presidio — {mode} matching (micro)",
            [("SurrogateShield", pc[mode]["ss"]["micro"]), ("Presidio", pc[mode]["presidio"]["micro"])]))
        for arm, label in (("ss", "SurrogateShield"), ("presidio", "Presidio")):
            console.print("  " + _macro_line(f"{label:<15}", pc[mode][arm]["macro"]))
        console.print()
    bs = pc.get("bootstrap_untyped_micro_f1_ss_minus_presidio") or {}
    if bs.get("available"):
        console.print(f"  untyped micro-F1 difference (SS − Presidio): {_signed(bs['diff'])}  "
                      f"{int((1 - bs['alpha']) * 100)}% bootstrap CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}] "
                      f"over {bs['n_questions']} questions")
    sent = pc.get("gold_values_verbatim_in_sent_text") or {}
    if sent.get("rows_with_presidio_text"):
        console.print(f"  gold values verbatim in the sent text ({_num(sent['gold_values'])} gold values): "
                      f"SS {_num(sent['ss'])} ({_num(sent['ss_unintended'])} unintended) · "
                      f"Presidio {_num(sent['presidio'])}")
    console.print()

    out = pc.get("outside_universe") or {}
    if out:
        t = Table(title="Gold types outside the shared universe (not in the head-to-head)",
                  box=box.SIMPLE_HEAVY, title_justify="left")
        for col in ("type", "gold", "SS found", "Presidio overlaps", "why excluded"):
            t.add_column(col, justify="right" if col in ("gold", "SS found", "Presidio overlaps") else "left")
        for typ, c in out.items():
            t.add_row(typ, _num(c["gold"]), _num(c["ss_found"]), _num(c["presidio_found"]), c["reason"])
        console.print(t)
    po = pc.get("presidio_only_counts") or {}
    if po:
        console.print("  [dim]Presidio entities outside the universe: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(po.items())) + "[/dim]")
    console.print()


def render_bertscore(b: dict, console: Console) -> None:
    resc = b.get("rescaled")
    console.print(f"  [dim]rescaled with baseline: {resc if resc is not None else 'n/a'} · "
                  f"rows with a scoring error: {_num(b.get('rows_with_error'))}[/dim]")
    if b.get("warning"):
        console.print(f"  [yellow]{b['warning']}[/yellow]")
    for kind in ("input_fidelity", "output_utility"):
        k = b.get(kind) or {}
        t = Table(title=f"BERTScore — {kind.replace('_', ' ')}", box=box.SIMPLE_HEAVY,
                  title_justify="left", caption=k.get("note"))
        for col in ("arm", "n", "P", "R", "F1"):
            t.add_column(col, justify="left" if col == "arm" else "right")
        for arm, label in (("ss", "SurrogateShield"), ("presidio", "Presidio")):
            m = k.get(arm) or {}
            t.add_row(label, _num(m.get("n")), _num(m.get("precision")), _num(m.get("recall")), _num(m.get("f1")))
        console.print(t)
        p = k.get("paired") or {}
        if p.get("available"):
            console.print(f"  paired F1 difference (SS − Presidio) over {p['n_paired']} rows "
                          f"(excluded {_num(k.get('excluded_identical_text'))} identical-text rows): "
                          f"{_signed(p['mean_diff'])}  CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  "
                          f"d_z {_num(p.get('effect_size_dz'))}  p {p.get('p_display', 'n/a')}")
        else:
            console.print(f"  [dim]paired statistics not available: {p.get('reason', 'n/a')}[/dim]")
        console.print()


def render_ablation(a: dict, console: Console) -> None:
    if not a.get("available"):
        console.print(Panel(f"{a.get('reason', 'n/a')}\n\nRun: [cyan]{a.get('command', 'n/a')}[/cyan]",
                            title="[bold blue]Ablation[/bold blue]", border_style="yellow", padding=(1, 2)))


def render_timing(tm: dict, console: Console) -> None:
    t = Table(title="Stage timings (ms)", box=box.SIMPLE_HEAVY, title_justify="left")
    for col in ("stage", "n", "mean", "p50", "p95", "max"):
        t.add_column(col, justify="left" if col == "stage" else "right")
    for stage, m in tm.items():
        t.add_row(stage, _num(m["n"]), *(_num(m[k], 3) for k in ("mean", "p50", "p95", "max")))
    console.print(t)


_SECTIONS = [
    ("detection", render_detection),
    ("sanitization", render_sanitization),
    ("restoration", render_restoration),
    ("presidio_comparison", render_presidio),
    ("bertscore", render_bertscore),
    ("ablation", render_ablation),
    ("timing", render_timing),
]


def render(result: dict, console: Console) -> None:
    """Render every section present in *result*."""
    render_overview(result, console)
    console.print()
    for key, fn in _SECTIONS:
        if result.get(key) is not None:
            fn(result[key], console)
            console.print()
