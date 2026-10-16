# Test-3 hypotheses and development targets (pre-registered)

Written on 2026-10-08, at `5fc2d53`, **before test-3 is drawn and before any
change to detection; the detector is the one frozen for test-2 at `e2c8fa5`**
(`bench/realdata/FREEZE.json`, tree `92806e9`). The SHA-256 of this file is
recorded in `bench/realdata/test3/FREEZE.json` when test-3 is frozen; neither
this file nor the hypotheses move after that.

Statistics as in `bench/realdata/score.py`: paired bootstrap over
conversations (2,000 resamples), 95 % percentile intervals, Wilson intervals on
rates. "Each dataset" means OASST1, ShareGPT and WildChat; "the six baselines"
are presidio_default, presidio_faker, presidio_transformers, llm_guard,
gliner_pii and gliner_pii_tuned (GLiNER-PII at the configuration with the
lowest dev leak in `bench/results/gliner_sweep_dev.json`, frozen before test-2
and unchanged). Readings the hypotheses leave open are those of
`bench/realdata/verdict.py` (committed before test-2's freeze; `decide` is not
changed for test-3).

## 1. Test-2's outcome (stays reported as scored)

Test-2 was scored once at `e2c8fa5` (2026-10-06; results committed at
`5fc2d53`). Under its pre-registered rule the verdict is **NO-GO, on H5''
only**: SurrogateShield (SS) leaked 7 of 202 injected ages where GLiNER-PII
leaked 0 of 202, a difference of +3.5 pp [+1.1, +6.3], above the 2 pp limit
with the interval above 0. H1''–H4'' and H6'' held (pooled injected leak SS
28/3,657, GLiNER-PII 371, Presidio-default 1,159); H7''–H10'' were not run
(the authors stopped the run on 2026-10-06; `verdict_test2.json` prints
"incomplete" because H8'' is in the rule). That outcome is the pre-registered
test-2 result and stays reported as such; it is never re-scored and never
relabelled. Test-3 is a fresh draw that tests the changes below once.

## 2. The planned changes between the two freezes

Every change is accepted or rejected on development data only (`dev` and
`devlarge`, i.e. test-1) by the targets in §3, then frozen.

- **A. AGE becomes a tagger type as well as a pattern type.** Today AGE is
  routed to `pattern_scan`, `canonicaliser` and `structural` only, so a layout
  the patterns do not know is a leak. AGE joins `TAGGER_STRUCTURED_TYPES`
  (union with the patterns), with an AGE threshold chosen on dev and
  devlarge-calib. Motivation: the dev acceptance of 2026-10-06 already flagged
  AGE (`ok: false`, 1 of 33 on dev) before the test-2 freeze, and the tagger
  alone finds 33/33 dev ages and 65/66 devlarge-val ages. Motivated by
  development data; confirmed by test-2's failure.
- **B1. `_NAMED_AGE` accepts a line or text end after the age** ("Name Name,
  NN" closing a line). **Motivated by test-2's failure shapes** (4 of the 7
  leaked ages, read as masked shapes only).
- **B2. Name classes in the age rules accept Latin letters beyond ASCII**
  (Latin-1 Supplement, Latin Extended-A and -B), through one shared fragment
  used by every name-bearing age rule. **Motivated by test-2's failure shapes**
  (2 of the 7: Turkish and Romanian letters).
- **C. A worded age surrogate leaves the original's decade** (a generator bug:
  "thirty" → "thirty-one" keeps the original word). **Motivated by test-2's
  failure shapes** (1 of the 7).
- **D. Conditional: retraining the tagger** with third-person, sign-off,
  contact-block, bracket and forum-style AGE templates, only if a synthetic
  probe built from the training identity pools shows tagger recall below 0.95
  on those layouts after A–C; at most two runs; full acceptance for every
  type; template-only training as before. If D runs, the templates were
  written knowing test-2's masked failure shapes, and test-3 is their only
  out-of-sample measurement.
- **E. Conditional, outside the default: an "off stage contributes nothing"
  guard** for configurations with `pattern_scan` disabled (today its URL
  opacity check still runs and drops tagger candidates inside URLs, which
  inflates the H9(a) ablation's URL leak). Made only if it is a one-guard
  change that leaves the default configuration's output byte-identical on dev.
- **I. AGE oversampled in the injected slice** (§5, "AGE sample"): a design
  choice, so that the type that failed is measured on about 300 values.

No change is made to the scorer, the leak rule, the bootstrap, the GO rule,
the baselines or `gliner_pii_tuned`.

## 3. Development acceptance targets (dev and dev-large validation half; not hypotheses)

Measured with `bench.tagger.evaluate --ss` on `dev` and `devlarge --half val`,
pooled by `bench.tagger.accept`, plus `score --split dev` and
`--split devlarge` regenerated at the new default, `components --split dev`,
`operating_curve --split dev`, `attribute --split dev|devlarge`.

- **G1 (AGE).** AGE leak ≤ max(0.02, GLiNER-PII's) on the same rows: **0/33 on dev, ≤ 1/66 on devlarge-val**; and 0 `surrogate`-cause AGE leaks anywhere.
- **G2 (no new hole).** Every other type passes the same per-type rule, pooled leak ≤ 0.06, and **no type's leaked count rises** against `balanced` at `a1abaa92142b0db9` on the same rows (dev 9/627 pooled; devlarge-val 7/1,249).
- **G3 (over-redaction).** Natural spurious edits ≤ `balanced`'s on the same messages: **≤ 133 on dev, ≤ 417 on devlarge-val**; negatives untouched **≥ 377/431 and ≥ 791/915**; and per dataset, OASST1's untouched count does not fall. If A alone breaks G3, first raise the tagger's AGE threshold, then add a shape guard for tagger AGE candidates (the span must be a number 1–120 or a number word, optionally with an age cue), before anything else moves.
- **G4 (speed).** `bench.perf_arms` p50 and peak RSS within 10 % of `perf_arms_v3.json`.
- **G5.** Fast suite green; `FORBIDDEN` clean; the default config hash changes and every place that pins it is updated knowingly; the committed `h9/*.json` files do not change; `bench.arms.run --record` passes with the test-2 SS configuration listed as historical.

If G1–G3 cannot all hold, D is applied once (if its trigger fired). If they
still cannot hold, the configuration that satisfies G2 + G3 with the lowest
AGE leak is frozen, and the report says so.

## 4. Hypotheses on test-3

- **H1'''** Injected-slice leak of SS is lower than **each** of the six baselines, including GLiNER-PII at its default **and** `gliner_pii_tuned`, on **each** dataset, CI excluding 0.
- **H2'''** Injected-slice spurious-edit rate of SS is lower than GLiNER-PII, Presidio-default and Presidio+Faker on each dataset, CI excluding 0.
- **H3'''** Natural-slice `negatives_untouched` of SS is higher than every baseline on each dataset.
- **H4'''** Format-shift leak of SS is lower than GLiNER-PII's, pooled, CI excluding 0.
- **H5'''** Per-type: no type where SS's pooled leak exceeds GLiNER-PII's by more than 0.02 with a CI excluding 0 (no new hole opened). AGE is the type that failed on test-2; it is held to the same 0.02 bound, with n ≈ 300 by design (§5, "AGE sample").
- **H6'''** Performance: SS p50 latency ≤ GLiNER-PII's and peak RSS ≤ GLiNER-PII's (`perf_arms`), with the full default pipeline.
- **H7'''** Utility (E5b, BERTScore + judge) not below Presidio+Faker's and not below GLiNER-PII's by more than 0.02; restored correctness and cross-turn consistency (E7b) above Presidio+Faker and GLiNER-PII.
- **H8'''** Attacker (E8): 0 exact recoveries of SS-protected values; recovery not above GLiNER-PII's.
- **H9'''** Regex ablation: SS with all PatternScan patterns removed still leaks less than Presidio-default (the tagger carries structured types as free text when regex is gone); SS with the tagger removed leaks less than Presidio-default (regex alone is still a system).
- **H10'''** External benchmark: SS's span F1 or leak-equivalent is not below GLiNER-PII's on data neither system was tuned on.

H7''' and H8''' are operationalised exactly as test-2's (the "Criteria for
test-2" paragraph of `bench/realdata/live.py`, committed at `7cacf05`), on
test-3's rows. H9''' uses the committed configs `h9/no_patterns.json` (the
deciding one) and `no_tagger.json`, with `h9/no_patterns_unrouted.json`
reported beside, as on test-2.

## 5. Rule, sizes and sealing

- **Rule.** **GO** = H1''' and H2''' on all three datasets, H3''', H5''', H6''', H8'''. **GO-with-caveat** (the report says exactly which) = H1''' on two datasets with the third not worse than GLiNER-PII (CI includes 0), everything else held. Anything less is **NO-GO**.
- **One shot.** Test-3 is scored once at the frozen commit. A detection change after `bench/realdata/test3/FREEZE.json` makes the next measurement a test-4, with its own budget. The steps never run on test-2 (live utility, multi-turn and attacker; component ablation; attribution; external benchmark; operating curve; beyond-English run) run on test-3 only; test-2 is not re-scored.
- **OASST1 pool rule.** Test-3 draws 600 single-turn and 120 multi-turn prompts per dataset (test-2's size), from carriers neither test-1 nor test-2 used, if the filters leave enough. If OASST1 cannot supply 600 single-turn carriers after exact and near-duplicate removal, test-3 takes what it has down to a floor of 400 and records the count; the other two datasets are not reduced to match. This floor applies only to running out of carriers, never to saving budget.
- **AGE sample.** The injected slice keeps test-2's prompt counts (300 single-turn prompts per dataset, 48 of them format-shift, and 60 conversations), and the per-type target of single-turn prompts is **100 for AGE** and 60 for every other type, as on test-2. This changes the type mix against test-2; H1'''–H4''' compare arms on the same rows and are unaffected.
- **Live sizes** (H7''' and H8'''): E5b 100 injected single-turn prompts per dataset; natural slice 50 first turns per dataset; E7b **20** conversations per dataset (test-2's plan said 30; reduced so the live phase fits the unchanged 6,500-call ledger cap with a reserve); E8 60 prompts per dataset. GLiNER-PII is live in all three, as in test-2's plan.

**Sealing.** `score.check_freeze` refuses `--split test3` until
`bench/realdata/test3/FREEZE.json` exists and matches this file's hash, the
code, the default config and the config files; the live runner refuses
`--split test3` before the sealed E5''' result exists. Seeds, batch names,
span files, ledger phases and the run directory are test-3's own; nothing of
test-1 or test-2 is reused. Test-3's natural-slice pooling (the arms whose
spans the annotator adjudicates) and the injection's PII-free check run with
the arms as they are at the test-2 freeze, before any detection change, as
test-2's did with the detector of its time. Test-3's identities come from the
evaluation half of the identity pools, disjoint from the tagger's training
half.
