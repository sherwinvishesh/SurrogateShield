# Real-data benchmark: how to rebuild it

Real user prompts from three public datasets (OASST1, ShareGPT, WildChat) in two slices:

- **natural**: the prompts as users wrote them, with silver labels by Claude Sonnet 4.6;
- **injected**: real prompts that Sonnet adjudicated PII-free, with our own fake values placed in them, so the gold is exact.

[MANIFEST.md](MANIFEST.md) lists the pinned sources, filters, counts, seeds and hashes; it is generated from `manifest.json`. [HUMAN_CHECK.md](HUMAN_CHECK.md) is the 100-message check of the silver labels. Every result file is listed with its command in [`bench/results/README.md`](../results/README.md).

## What is committed and what is not

Real prompts can carry real personal data even after the publishers' scrubbing, so **no natural-slice text is committed**, whatever the licence allows.

| path | committed | holds |
|---|---|---|
| `<dataset>/pool.jsonl` | yes | source ids, references into the pinned files, SHA-256 and word count of every user turn, split |
| `<dataset>/labels.jsonl` | yes | silver labels as character offsets and types, no values |
| `<dataset>/pii_free.json` | yes | ids of the turns Sonnet adjudicated PII-free (the injection bases) |
| `<dataset>/dev.jsonl`, `<dataset>/test.jsonl` | yes | the injected slice in full: real carrier text plus our fake values, with the gold |
| `manifest.json`, `MANIFEST.md` | yes | sources, filter counts, seeds, hashes of every frozen file |
| `raw/` | no (git-ignored) | the downloads |
| `build/` | no (git-ignored, files 0600) | rebuilt natural text, label values, arm inputs, private span files, batch state |
| `experiment/realdata/` (repository root) | no (git-ignored, 0600 in a 0700 directory) | the call ledger and every Phase 6 provider reply |

Rules for anyone working with it:

- Never paste a natural-slice message anywhere (report, issue, commit, test, chat).
- Never try to identify anyone in it.
- Look at the text only through counts, hashes and masked shapes.
- WildChat's terms ask users not to attempt re-identification.

## 1. Environments

```
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt -e python-library    # SurrogateShield, Presidio arms, BERTScore
python3.12 -m venv .venv-baselines
.venv-baselines/bin/pip install -r bench/arms/requirements-baselines.txt                 # LLM Guard, GLiNER-PII, Presidio+transformers
```

The model revisions are in the header of `bench/arms/requirements-baselines.txt` and in each span file's `.meta.json`. Phase 6 also needs `roberta-large` for BERTScore. Fetch the models once. Everything after that runs with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## 2. Download the pinned files (the only network step besides the provider phases)

WildChat-1M is gated on Hugging Face: accept its terms and `hf auth login` first.

```
.venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
from bench.realdata.common import RAW, SOURCES
for src in SOURCES.values():
    for f in src["files"]:
        hf_hub_download(src["repo"], f, repo_type="dataset", revision=src["revision"], local_dir=RAW / src["local"])
EOF
```

The SHA-256 of every raw file is in MANIFEST.md.

## 3. Rebuild the natural text and check it

```
.venv/bin/python -m bench.realdata.build --check   # every turn's hash against pool.jsonl; writes nothing
.venv/bin/python -m bench.realdata.build           # writes build/<dataset>/pool.jsonl (0600)
```

Any hash mismatch stops the build. `bench.realdata.pull` produced `pool.jsonl` (Phase 1). Rerunning it on the same raw files gives the same pools. It is not needed for a rebuild.

## 4. What each phase ran

| phase | command | network | output |
|---|---|---|---|
| 1 sample and split | `.venv/bin/python -m bench.realdata.pull` | none (reads `raw/`) | `<dataset>/pool.jsonl`, manifest |
| 4a arms on the natural pools | `.venv/bin/python -m bench.arms.run --natural` | none | `bench/results/spans/<arm>/natural-<dataset>.jsonl` (the candidate pool for labelling) |
| 2 silver labels | `.venv/bin/python -m bench.realdata.label --estimate`, `--pilot`, `--run --write` | Anthropic (Sonnet 4.6, Message Batches) | `<dataset>/labels.jsonl`, `pii_free.json`, `bench/results/realdata_prevalence.json` |
| 3 injection | `.venv/bin/python -m bench.realdata.inject --plan`, `--estimate`, `--pilot`, `--run --write` | Anthropic (Sonnet 4.6, Message Batches) | `<dataset>/dev.jsonl`, `test.jsonl`, `bench/results/realdata_injection.json` |
| 5 E5 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split {dev,test} --out bench/results/realdata_{dev,test}.json` | none | the six arms on one split, one scorer |
| 5 E6 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.regex_ablation --split {dev,test} --out bench/results/realdata_regex_ablation_{dev,test}.json` | none | PatternScan ablation (60 conditions) |
| 5 E7 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms.json` | none | latency, cold start, peak RSS on 200 dev messages |
| 6 E5b, E7b, E8 | `.venv/bin/python -m bench.realdata.live {estimate,utility,multiturn,attacker}` | Anthropic (Sonnet 4.6 responder; Opus 5.5 judge and attacker) | `bench/results/{utility,multiturn,attacker}_realdata.json` |

You do not need to repeat Phases 2 and 3 to reproduce the scores: their outputs are committed and hash-checked. Re-labelling or re-injecting would draw new model outputs and therefore a different benchmark.

The scorer checks every frozen file against `manifest.json`, and every rebuilt natural text against its committed hash, before it scores anything. Each arm runs as a subprocess on one private input per dataset and split (`build/score/<split>/<dataset>.jsonl`). Its spans are used only if their meta records that input's SHA-256. The committed copies under `bench/results/spans/` hold offsets, type, replacement length and a copied flag only.

## 5. Provider calls

`provider.py` takes every request from a persistent ledger (`experiment/realdata/ledger/`, cap 4,000 calls) *before* it is sent; a request that would pass the cap is not sent. A batch's id is saved as soon as it is created (`build/batches/`), so a rerun after a crash fetches the same batch instead of paying again. `live.py` caches each reply by the hash of its slot and request, so a rerun sends only what is missing, and its pilots (`--pilot N`) are a subset of the main run. The key comes from the environment or `.env` and is never printed or stored.

Splits are 20 % dev / 80 % test, drawn over source prompts before any system ran; injected rows inherit their base prompt's split. Dev is for diagnosis; test is scored once, at a recorded commit.
