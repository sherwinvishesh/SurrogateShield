# bench/results

Summaries written by the bench scripts (counts only — no message text).

| File | Command | Code |
|---|---|---|
| `j2_baseline_dev.json` | `python bench/realworld.py --split dev --json bench/results/j2_baseline_dev.json` | bf142ed (before any J2 tuning) |
| `j2_baseline_test.json` | `python bench/realworld.py --split test --json bench/results/j2_baseline_test.json` | bf142ed (before any J2 tuning) |
| `j2_phase2_dev.json` | `python bench/realworld.py --split dev --json bench/results/j2_phase2_dev.json` | 21f1574 (end of Phase 2; tuned on dev) |
| `j2_phase2_test.json` | `python bench/realworld.py --split test --json bench/results/j2_phase2_test.json` | 21f1574 (end of Phase 2; first test run after tuning) |
| `j4_baseline.json` | `python bench/echo.py --json bench/results/j4_baseline.json` | 8d1d518 (before Phase 3; run with bench/echo.py and core/consistency.py copied in) |
| `j4_phase3.json` | `python bench/echo.py --json bench/results/j4_phase3.json` | end of Phase 3 |
| `j2_phase4_dev.json` | `python bench/realworld.py --split dev --json bench/results/j2_phase4_dev.json` | 6b66851 (end of Phase 4; each record seeded) |
| `j4_phase4.json` | `python bench/echo.py --json bench/results/j4_phase4.json` | 6b66851 (end of Phase 4; `Pipeline(seed=0)`) |
| `compare_phase5_baseline_dev.json` | `python bench/compare.py --json bench/results/compare_phase5_baseline_dev.json` | 5394bd0 (start of Phase 5; synth dev + realworld dev) |
| `compare_phase5_baseline_dev2.json` | `python bench/compare.py --synth none --realworld dev2 --json bench/results/compare_phase5_baseline_dev2.json` | 5394bd0 (start of Phase 5; dev2 never tuned on before) |
| `j2_phase5_test.json` | `python bench/realworld.py --split test --json bench/results/j2_phase5_test.json` | d6a2cdb (end of Phase 5 tuning on dev + dev2; second test run) |
| `compare_phase5_final.json` | `python bench/compare.py --final --json bench/results/compare_phase5_final.json` | d6a2cdb (synth test, first and only run; realworld test, same run as above) |
| `j15_perf.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python bench/perf.py --json bench/results/j15_perf.json` | 8fb7f1d (J15 latency, cold start, peak RSS) |
| `compare_phase5_baseline_dev3.json` | `python bench/compare.py --synth none --realworld dev3 --json bench/results/compare_phase5_baseline_dev3.json` | 44a9979 (dev3 never seen by any rule; before tuning on it) |
| `compare_phase5_baseline_dev4.json` | `python bench/compare.py --synth none --realworld dev4 --json bench/results/compare_phase5_baseline_dev4.json` | 44a9979 (dev4 is scored only, never inspected) |
| `compare_final_dev3.json` | `python bench/compare.py --synth none --realworld dev3 --json bench/results/compare_final_dev3.json` | b653f50 (after diagnosing on dev3; same counts at 31fcabc) |
| `compare_final_dev4.json` | `python bench/compare.py --synth none --realworld dev4 --json bench/results/compare_final_dev4.json` | b653f50 (dev4 still scored only; same counts at 31fcabc) |
| `compare_final_test_seen.json` | `python bench/compare.py --final --json bench/results/compare_final_test_seen.json` | b653f50 (third run on test; test is *seen*, not held out) |
| `j15_perf_final.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python bench/perf.py --json bench/results/j15_perf_final.json` | 31fcabc (J15 re-run) |
| `ablation_synth_dev.json` | `python offline_eval.py --key experiment/synth_dev_key.json --ablation --json bench/results/ablation_synth_dev.json` | 7a81253 (A9 ablation, cascade re-run per configuration; synth dev, tuned on) |
| `phase8.json` | `python bench/phase8.py --sample 50 --seed 0 --attacker claude-opus-5-5 --run main --max-calls 292 --json bench/results/phase8.json` | c9fa81d (Phase 8 live run, 251 provider calls; synth test, seen) |
| `spans/<arm>/natural-<dataset>.jsonl` (+ `.meta.json`) | `.venv/bin/python -m bench.arms.run --natural` (after `python -m bench.realdata.build`) | dcec6ac (E4: six span producers on the 2,251 natural user turns; offsets, type, replacement length and a copied flag only, no text; arm configs in `bench/realdata/manifest.json`) |
| `spans/<arm>/test2-natural-<dataset>.jsonl` (+ `.meta.json`) | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.arms.run --natural --collection test2` (after `python -m bench.arms.inputs --collection test2`) | E4 on test-2's natural pools (2,712 user turns: 852 / 938 / 922), the silver-label candidates of the sealed collection; produced by the six arms as of 3280d70 (SS unchanged since c460b7d, before any detector change), every arm on the CPU; offsets, type, replacement length and a copied flag only, no text. LLM Guard's test-1 `natural-*` files predate the CPU pin and are listed apart in `bench/realdata/manifest.json` (`historical`) |
| `realdata_prevalence.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.label --run --write --out bench/results/realdata_prevalence.json` | E1: silver labels (Sonnet 4.6, prompt `bebc6e281801fa74`, Message Batches; a rerun reads the saved batch results and makes no new call) for the 2,251 natural user turns; counts only: messages with personal data by dataset, type, task and kind, and PII-free sources under the adjudicated and the literal rule. Per-message labels (offsets and types, no text) are `bench/realdata/<dataset>/labels.jsonl` |
| `realdata_prevalence_test2.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.label --collection test2 --run --write --out bench/results/realdata_prevalence_test2.json` | E1 on the sealed test-2 natural pools: same annotator, prompt (`bebc6e281801fa74`) and pooling as test-1, candidates from the six arms' `test2-natural-*` spans; 2,641 of 2,712 messages labelled (71 failed validation twice); offsets-only labels in `bench/realdata/test2/<dataset>/labels.jsonl`; 334 calls (282 + 52 re-asks) |
| `realdata_injection.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.inject --run --write --out bench/results/realdata_injection.json` | E2/E3: injected slice, Sonnet 4.6 places local seeded fake values as span edits into PII-free real prompts; 878 of 900 accepted (22 dropped after one re-ask), acceptance by round, drop-reason kinds (no text), types and layouts per dataset; writes `bench/realdata/<dataset>/{dev,test}.jsonl` and their hashes into the manifest |
| `realdata_dev.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split dev --out bench/results/realdata_dev.json` | E5 on dev (diagnosis split; not the paper's numbers): six arms on the natural and injected dev slices of each dataset, leak and spurious rates with Wilson and paired bootstrap CIs, H1/H2 checks; produced at af3607d on a clean tree (the file records the commit); per-arm offsets are `spans/<arm>/dev-<dataset>.jsonl` |
| `spans/<arm>/dev-<dataset>.jsonl` (+ `.meta.json`) | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split dev --out bench/results/realdata_dev.json` | af3607d (span producers on the dev split, written by the scorer; offsets, type, replacement length and a copied flag only, no text; byte-identical across reruns apart from the `ms` and `load_seconds` timings) |
| `realdata_regex_ablation_dev.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.regex_ablation --split dev --out bench/results/realdata_regex_ablation_dev.json` | E6 on dev (diagnosis split): SurrogateShield's PatternScan with each regex family dropped, each made to never match its own format, and random 25 % / 50 % drops (five draws each), 60 conditions on the frozen dev split; leak per slice against Presidio-default unablated, paired bootstrap against `none`, and per value whether EntityTrace or ContextGuard recovered what the pattern stage lost. `none` equals the `ss` arm on every message. H6 does not hold on dev for any dataset (the all-wrong and 50 % conditions pass Presidio-default's leak). Produced at 5413581 on a clean tree; the `.md` beside it is the table |
| `realdata_test.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split test --out bench/results/realdata_test.json` | E5 on test, run once at the end of Phase 5: the paper's detection numbers. Six arms on the natural and injected test slices of each dataset, leak and spurious rates with Wilson and paired bootstrap CIs, H1/H2 checks. H1 does not hold on any dataset: GLiNER-PII leaks less on oasst1 (CI excludes 0) and is not separable from SS on sharegpt and wildchat; SS's deficit is the format-shift subset (SS − GLiNER +0.18 / +0.23 / +0.20). H2 holds on sharegpt and wildchat. Produced at 230a470 on a clean tree (the file records the commit); detection code is frozen from here, and any later change starts `realdata_test_v2.json`; the `.md` beside it is the table |
| `spans/<arm>/test-<dataset>.jsonl` (+ `.meta.json`) | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split test --out bench/results/realdata_test.json` | 230a470 (span producers on the test split, written by the scorer; offsets, type, replacement length and a copied flag only, no text; byte-identical across reruns apart from the `ms` and `load_seconds` timings) |
| `realdata_regex_ablation_test.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.regex_ablation --split test --out bench/results/realdata_regex_ablation_test.json` | E6 on test (A1, H6), run once: the 60 PatternScan conditions of the dev run on the frozen test split. `none` equals the `ss` arm on every message. H6 does not hold on any dataset (`wrong-all`, every 50 % draw and one or two 25 % draws pass Presidio-default's unablated leak). Produced at 230a470 on a clean tree; the `.md` beside it is the table |
| `perf_arms.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms.json` | E7 (B4): every arm on the same seeded 200 real-data dev messages (67/67/66), offline, one arm at a time on one machine: warm p50/p95/mean/max of the arm's own per-message time, model load, cold start (median of 3) and peak RSS. Produced at 230a470 on a clean tree; the `.md` beside it is the table |
| `attribution_dev.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split dev --out bench/results/attribution_dev.json` | (+ `.md`) §3.1 baseline at 3e3ae7c (detector as of c460b7d): one cause per injected protect value on dev (protected / policy / refused / surrogate / partial + parts left / unplanned / gated + pass, rule, source / missed + in_url), by type × layout × format × slice, and natural spurious edits by source; counts only; leak and policy counts agree with the J2 scorer per message; traced edits equal the committed `ss` dev spans (662/662) |
| `attribution_devlarge.json` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split devlarge --out bench/results/attribution_devlarge.json` | (+ `.md`) §3.1 baseline on devlarge (test-1's test split, development only) at 3e3ae7c: 2,368 values, leak 0.118; traced edits equal the committed `ss` test spans (2,608/2,608) |

The Phase 2 numbers show overfitting. On dev, leaked went from 25.6 % to 0 % and spurious from 28.1 % to 0.75 %. On test, leaked went from 32.3 % to 19.3 % and spurious from 37.5 % to 21.7 %. J2 fails on test. After this run the test split counts as *seen*: later test numbers are reported next to these and are not tuned on.

J4 (`bench/echo.py`) sends 250 turns from the dev corpus and experiment questions through `Pipeline.process_turn` with a provider that echoes the masked message. A turn is corrupted when the restored echo differs from what the user typed. Before Phase 3: 24 of 250 turns corrupted, the first at turn 64. After Phase 3: 0. Surrogates are not yet seeded per session (Phase 4), so each run draws different ones. In 80 runs after the last fix, one run corrupted one turn. Its cause was a random zip surrogate (30303) that equalled a zip the user typed later (see AUDIT_PROGRESS, open item on quote-backs).

After Phase 4, surrogates are seeded: `bench/echo.py` uses `Pipeline(seed=0)` and `bench/realworld.py` seeds record *i* with *i*, so runs repeat exactly. J4 on seeds 0–7 (`seed=` edited in bench/echo.py): 0 corrupted turns each. Seed 0 issued "Ann Arbor" as the surrogate for Tempe, and a later user typed a real Ann Arbor. That later value is masked as a new original and restored correctly. It is listed under `later_original_equals_earlier_surrogate` and not gated. A surrogate issued equal to a value already seen still fails J4.

Phase 5 adds `bench/compare.py`, which runs SurrogateShield and default-config Presidio (threshold 0.4) on the same messages. Two data sets feed it. The first is the seeded synthetic set (`experiment/make_dataset.py`), split 1200 dev / 300 test, with gold offsets. The second is the real-world corpus: dev, plus `dev2`, a second set of 240 messages written for tuning. Before any Phase 5 tuning, the dev2 corpus shows the same picture as test did in Phase 2. SurrogateShield leaks 19.8 % of must-protect values (86 of 435) with 15.1 % spurious edits; Presidio leaks 39.3 % with 34.4 % spurious. On synth dev, SurrogateShield's any-type micro F1 is 0.946 against Presidio's 0.772. One type is worse than Presidio, personal URLs: recall 0.702 vs 1.000. The 0 % dev leak rate is a tuned number and does not describe unseen text.

At the end of Phase 5 tuning the gap is wide, and J2 still fails. Tuning on dev and dev2 took dev2 from 19.8 % leaked and 15.1 % spurious to 0.9 % and 1.3 %. On the held-out test split, leaked went from 19.3 % (Phase 2) to 16.4 % and spurious from 21.7 % to 19.1 %; Presidio scores 39.0 % and 35.8 % on the same messages. Nearly all of the dev2 gain came from rules fitted to dev2's phrasing, and it does not carry over. On the synthetic test split (300 messages, first run), any-type micro F1 is 0.986 (P 0.990, R 0.982) against Presidio's 0.746. 6 of 75 negatives get an edit, against Presidio's 61. J1 on synth test fails: 8 unintended leaks (1.2 % of gold values), where synth dev has 0. Neither test split has been tuned on. Any later test number will be reported next to these.

dev3 and dev4 (436335e) are two new 240-message splits, written blind to the rules. dev3 is for diagnosis; dev4 is only scored (`--show` refuses it), as test is. Their first runs at 44a9979 give the generalisation estimate for the rules tuned on dev and dev2. On dev3, SurrogateShield leaks 19.1 % of must-protect values (109 of 572) with 18.3 % spurious edits, and Presidio 40.2 % and 35.8 %. On dev4, SurrogateShield leaks 25.7 % (182 of 707) with 18.8 % spurious, and Presidio 47.4 % and 39.8 %. These match the test split (16.4 % / 19.1 %), not dev2 (0.9 % / 1.3 %). SurrogateShield leaks about half as much as Presidio and makes about half as many spurious edits, but J2 (≤ 2 % / ≤ 3 %) is far off on unseen text. Running these splits first also found five messages that crashed generation (fixed in 44a9979).

The diagnosis on dev3 used rules by category (2ed18ea … b653f50). It moved dev3 from 19.1 % leaked and 18.3 % spurious to 0.9 % (5 of 572) and 11.9 % (84 of 704). dev3 was looked at, so these are tuned numbers. dev4 was never inspected. On it, leaked went from 25.7 % to 21.6 % (153 of 708) and spurious from 18.8 % to 16.5 % (127 of 770). On the seen test split (third run, not tuned on), leaked is 13.2 % (65 of 494) and spurious 16.9 % (93 of 551), against 16.4 % and 19.1 % at Phase 5. Presidio scores the same as before on every split: 40.2 / 35.8 % (dev3), 47.4 / 39.8 % (dev4), 39.0 / 35.8 % (test). Negatives left untouched: SurrogateShield 57/83, 54/77, 46/65; Presidio 26/83, 18/77, 14/65. J2 (≤ 2 % leaked, ≤ 3 % spurious) fails on dev4 and on test. The fix of this phase shows up on unseen text, but it is small: about 4 points of leak rate, against the 18 points it gained on dev3.

J15 re-run at 31fcabc: warm p50 is 47.5 ms without ContextGuard (gate 50) and 74.8 ms with it (gate 150). Both pass, the first by 2.5 ms. Cold first mask takes 3.4 s and 6.0 s, with peak RSS 823 MB and 1090 MB, on a 10-core arm64 Mac with Python 3.13.2 and torch 2.12.0.

The ablation (A9) re-runs the cascade with stages switched off on the
1,200 synthetic dev messages. It scores protection only: values that were
replaced, matched by span overlap. The 95 % intervals are a paired
bootstrap over messages (2,000 resamples).

| configuration | micro F1 | change against the full cascade |
|---|---|---|
| PatternScan only | 0.842 (P 0.986, R 0.735) | −0.138 [−0.149, −0.127] |
| + EntityTrace, ContextGuard off | 0.980 | −0.001 [−0.003, +0.001] |
| all three stages, post-passes off | 0.968 | −0.013 [−0.016, −0.009] |
| full cascade | 0.981 (P 0.966, R 0.996) | — |

ContextGuard adds no measurable F1 on this split: the interval includes
0. It costs about 27 ms per message (J15). The split was used for tuning,
so these numbers describe fit, not unseen text.

Phase 8 (J10, A8, A10) is the only live run. 50 questions with gold
personal data were sampled with seed 0 from the synthetic test split
(seen). The responder is claude-sonnet-4-6 and the attacker is
claude-opus-5-5, a different and stronger model, with prompt version 2.
The run made 251 provider calls (151 responder, one a timeout retry; 100
attacker); no answer row failed.

Utility is BERTScore F1 (roberta-large, rescaled) of each arm's answer
against the answer to the original message. Mean 0.563 for
SurrogateShield and 0.384 for Presidio; the paired difference is +0.178
[+0.116, +0.246] (bootstrap, 2,000 resamples). Input fidelity (sanitised
message against the original) is 0.637 against 0.317, but it favours
realistic surrogates by construction and is not a utility measure.

The attacker recovered no value exactly from either arm (0 of 149
inference targets for SurrogateShield, Wilson 95 % upper bound 2.5 %; 0 of
108 for Presidio, upper bound 3.4 %). Partial recovery (same e-mail
domain, phone area code, a shared name token, street or city, or birth
year) was 26 of 149 (17.4 %) for SurrogateShield and 5 of 108 (4.6 %) for
Presidio. Most of SurrogateShield's partials come from what its
surrogates keep on purpose: the category word of an organisation name (13
of 16, e.g. "Pharmacy", "LLP"), the e-mail domain (6 of 12) and the birth
year (1 of 5). Four are name tokens: two kept first names and two
shared "der" particles. Two are addresses.
Presidio's exposure is mostly verbatim: 37 of 145 gold values (25.5 %)
were left in plain text, all 16 organisation names among them, against 2
of 151 (1.3 %) for SurrogateShield. Counting a value as exposed when it
was sent verbatim or partly recovered, SurrogateShield exposed 28 of 151
(18.5 %) and Presidio 42 of 145 (29.0 %). One SurrogateShield and two
Presidio attacker replies were not valid JSON; those rows are excluded,
which is why the denominators differ. 50 questions on generated data is
a small sample.

