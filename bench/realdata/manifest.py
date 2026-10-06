"""``bench/realdata/manifest.json`` and its rendering, ``MANIFEST.md``.

Every script that writes a frozen file records its SHA-256 here; MANIFEST.md is
regenerated from the JSON, so the counts in it come from the scripts that ran.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.realdata.common import RAW, RD, ROOT, SEED, SOURCES, file_sha256  # noqa: F401 (ROOT re-exported)

PATH = RD / "manifest.json"
MD = RD / "MANIFEST.md"


def load(path: Path = PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"seed": SEED, "datasets": {}, "frozen": {}}


def save(m: dict, path: Path = PATH, md: Path | None = MD) -> None:
    path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    if md is not None:
        md.write_text(render(m))


def section(m: dict, coll) -> dict:
    """Per-dataset records of a collection: ``m["datasets"]`` for test1,
    ``m["<name>"]["datasets"]`` for a later one."""
    if not coll.prefix:
        return m["datasets"]
    return m.setdefault(coll.name, {}).setdefault("datasets", {})


def file_hash(path: Path) -> str:
    return file_sha256(path)


def raw_hashes(dataset: str, raw: Path = RAW) -> dict:
    src = SOURCES[dataset]
    return {f: file_sha256(raw / src["local"] / f) for f in src["files"]}


_REDIST = {
    "oasst1": "Apache-2.0 permits redistribution with attribution.",
    "sharegpt": "CC0-1.0 as declared by the uploader; the underlying conversations were shared "
                "by ShareGPT users, so the provenance of individual texts is not verifiable.",
    "wildchat": "ODC-BY-1.0 permits redistribution with attribution; AI2's terms ask users not "
                "to attempt re-identification.",
}

_STEP_ORDER = [
    "source_roots", "source_conversations", "drop_duplicate_hash", "drop_root_not_prompter",
    "drop_deleted", "drop_not_english", "drop_toxic", "drop_not_starting_with_user",
    "drop_first_turn_empty", "drop_first_turn_not_english", "drop_first_turn_too_short",
    "drop_first_turn_too_long", "drop_first_turn_only_url", "drop_first_turn_only_code",
    "drop_first_turn_jailbreak", "first_turn_kept", "single_exact_duplicates",
    "single_near_duplicates_skipped", "drawn_single", "multi_candidates",
    "drop_multi_later_turn_empty", "drop_multi_later_turn_too_long",
    "drop_multi_later_turn_not_english", "drop_multi_later_turn_jailbreak", "multi_kept",
    "multi_exact_duplicates", "multi_near_duplicates_skipped", "drawn_multi",
]


def _counts_table(counts: dict) -> str:
    keys = [k for k in _STEP_ORDER if k in counts] + sorted(k for k in counts if k not in _STEP_ORDER)
    return "\n".join(f"| `{k}` | {counts[k]:,} |" for k in keys)


def render(m: dict) -> str:
    from bench.realdata.common import _JAILBREAK
    out = [
        "# Real-data benchmark — manifest",
        "",
        "Generated from `bench/realdata/manifest.json` by `bench/realdata/manifest.py`; do not edit by hand.",
        "",
        "## Sources",
        "",
        "| dataset | Hugging Face repo | revision | files | licence |",
        "|---|---|---|---|---|",
    ]
    for ds, src in SOURCES.items():
        out.append(f"| {ds} | `{src['repo']}` | `{src['revision']}` | {', '.join(f'`{f}`' for f in src['files'])} | {src['licence']} |")
    out += [
        "",
        "Files were fetched with `huggingface_hub.hf_hub_download(repo, file, revision=<sha>)` into the",
        "git-ignored `bench/realdata/raw/`. WildChat-1M has 14 shards; two were used, drawn with",
        f"`sorted(random.Random({SEED}).sample(range(14), 2))` = [4, 8]. WildChat-1M is the non-toxic release,",
        "so `toxic == True` excludes nothing in it (counted below); conversations the publishers marked",
        "`redacted` (PII they found was scrubbed) are kept and flagged in `meta.redacted`.",
        "",
        "## Redistribution and what is committed",
        "",
    ]
    for ds in SOURCES:
        out.append(f"- **{ds}**: {_REDIST[ds]}")
    out += [
        "",
        "Real prompts can carry real personal data even after the publishers' scrubbing, so no natural-slice",
        "text is committed, whatever the licence allows. `<dataset>/pool.jsonl` holds the source id, the",
        "references that locate each user turn in the pinned files, the SHA-256 and word count of each turn,",
        "and the split. `python -m bench.realdata.build` rebuilds the text into the git-ignored",
        "`bench/realdata/build/` and checks every hash. Injected rows (a PII-free base plus our fake values)",
        "are committed in full.",
        "",
        "## Filters and sampling",
        "",
        "Fixed in `bench/realdata/pull.py` before any system ran. Single-turn prompt = the first user turn;",
        "multi-turn = the first ≤ 3 user turns of a conversation with ≥ 2 (OASST1: along the best-ranked",
        "reply chain from the root). First turns: 15–400 words, not only code (< 8 alphabetic words once code",
        "fences, URLs and code-like lines are removed: `common._CODE_LINE`), not only a URL (< 5 words besides",
        "URLs), no jailbreak template. Later turns: ≤ 400 words, English when ≥ 5 words,",
        "no jailbreak template. ShareGPT has no language field: English = ≥ 90 % ASCII letters and ≥ 12 % English",
        "function words (`common.is_english`). Exact duplicates (lower-cased, punctuation-stripped,",
        "whitespace-collapsed) collapse to the smallest source id; near duplicates (rapidfuzz ratio ≥ 95 on the",
        "normalised first turn) are rejected during the seeded draw. Multi-turn conversations are drawn first;",
        "single-turn prompts come from the remaining conversations. Split 20 % dev / 80 % test by a seeded draw.",
        "",
        f"Jailbreak / prompt-injection keyword pattern (case-insensitive): `{_JAILBREAK.pattern}`",
        "",
        "Check of the English heuristic against WildChat's own language label (`python -m",
        "bench.realdata.pull --english-check`; first user turns of 15–400 words in a seeded sample of 20,000",
        "WildChat conversations) is under sharegpt below.",
        "",
    ]
    for ds, d in m.get("datasets", {}).items():
        out += [f"### {ds}", ""]
        if "pull_counts" in d:
            out += ["| step | count |", "|---|---|", _counts_table(d["pull_counts"]), ""]
        if "drawn" in d:
            out += ["Drawn: " + ", ".join(f"{k} = {v}" for k, v in d["drawn"].items()), ""]
        if "seeds" in d:
            out += ["Seeds (`common.derive_seed(dataset, *key)`): " + ", ".join(f"{k} = {v}" for k, v in d["seeds"].items()), ""]
        if "raw_files" in d:
            out += ["Raw file SHA-256:", ""] + [f"- `{f}` `{h}`" for f, h in d["raw_files"].items()] + [""]
        for k in sorted(d):
            if k not in ("pull_counts", "drawn", "seeds", "raw_files"):
                out += [f"**{k}**", "", "```", json.dumps(d[k], indent=1, sort_keys=True), "```", ""]
    for name in ("test2",):
        if not m.get(name):
            continue
        out += [f"## Collection `{name}`", "",
                "The sealed second test (PROMPT_FOR_OPUS_V3 §5.2): drawn by `python -m bench.realdata.pull",
                f"--collection {name}` from sources the first draw never touched (its ids skipped, its first turns",
                "seeding the near-duplicate check), one split, files under",
                f"`bench/realdata/{name}/<dataset>/`.", ""]
        for ds, d in m[name].get("datasets", {}).items():
            out += [f"### {name} / {ds}", ""]
            if "pull_counts" in d:
                out += ["| step | count |", "|---|---|", _counts_table(d["pull_counts"]), ""]
            if "drawn" in d:
                out += ["Drawn: " + ", ".join(f"{k} = {v}" for k, v in d["drawn"].items()), ""]
            if "seeds" in d:
                out += ["Seeds (`common.derive_seed(dataset, kind, \"test2\")`): "
                        + ", ".join(f"{k} = {v}" for k, v in d["seeds"].items()), ""]
            for k in sorted(d):
                if k not in ("pull_counts", "drawn", "seeds", "raw_files"):
                    out += [f"**{k}**", "", "```", json.dumps(d[k], indent=1, sort_keys=True), "```", ""]
        for k in sorted(m[name]):
            if k != "datasets":
                out += [f"**{name} {k}**", "", "```", json.dumps(m[name][k], indent=1, sort_keys=True), "```", ""]
    if m.get("arms"):
        out += ["## Systems under test", "",
                "Recorded by `python -m bench.arms.run` from each arm's meta sidecar",
                "(`bench/results/spans/<arm>/*.meta.json`); every arm runs offline.", ""]
        for arm, rec in sorted(m["arms"].items()):
            out += [f"### {arm}", "", f"Interpreter `{rec['interpreter']}/bin/python`, seed {rec['seed']}.", "",
                    "```", json.dumps(rec["config"], indent=1, sort_keys=True), "```", ""]
    if m.get("labels"):
        out += ["## Silver labels", "", "```", json.dumps(m["labels"], indent=1, sort_keys=True), "```", ""]
    out += ["## Frozen files (SHA-256)", "", "| file | SHA-256 |", "|---|---|"]
    for f, h in sorted(m.get("frozen", {}).items()):
        out.append(f"| `bench/realdata/{f}` | `{h}` |")
    out.append("")
    return "\n".join(out)
