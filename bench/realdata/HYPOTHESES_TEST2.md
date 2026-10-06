# Test-2 hypotheses and development targets (pre-registered)

Written on 2026-10-05, at `c460b7d` (detector of the test-1 era, unchanged),
before test-2 is drawn and before any change to detection. Copied verbatim
from the run instructions (PROMPT_FOR_OPUS_V3 §5.2 and §3.3). The SHA-256 of
this file is recorded in `bench/realdata/FREEZE.json` when test-2 is frozen;
neither this file nor the hypotheses move after test-2 is scored.

Statistics as in `bench/realdata/score.py`: paired bootstrap over
conversations (2,000 resamples), 95 % percentile intervals, Wilson intervals on
rates. "Each dataset" means OASST1, ShareGPT and WildChat; "the six baselines"
are presidio_default, presidio_faker, presidio_transformers, llm_guard,
gliner_pii and gliner_pii_tuned (GLiNER-PII at the configuration with the
lowest dev leak in `bench/results/gliner_sweep_dev.json`, frozen before test-2).

## Hypotheses on test-2

- **H1''** Injected-slice leak of SS is lower than **each** of the six baselines, including GLiNER-PII at its default **and** `gliner_pii_tuned` (§3.8), on **each** dataset, CI excluding 0.
- **H2''** Injected-slice spurious-edit rate of SS is lower than GLiNER-PII, Presidio-default and Presidio+Faker on each dataset, CI excluding 0.
- **H3''** Natural-slice `negatives_untouched` of SS is higher than every baseline on each dataset.
- **H4''** Format-shift leak of SS is lower than GLiNER-PII's, pooled, CI excluding 0 (the gap this plan attacks).
- **H5''** Per-type: no type where SS's pooled leak exceeds GLiNER-PII's by more than 0.02 with a CI excluding 0 (no new hole opened).
- **H6''** Performance: SS p50 latency ≤ GLiNER-PII's and peak RSS ≤ GLiNER-PII's (`perf_arms`), with the full default pipeline.
- **H7''** Utility (E5b, BERTScore + judge) not below Presidio+Faker's and not below GLiNER-PII's by more than 0.02; restored correctness and cross-turn consistency (E7b) above Presidio+Faker and GLiNER-PII.
- **H8''** Attacker (E8): 0 exact recoveries of SS-protected values; recovery not above GLiNER-PII's.
- **H9''** Regex ablation: SS with all PatternScan patterns removed still leaks less than Presidio-default (the tagger carries structured types as free text when regex is gone); SS with the tagger removed leaks less than Presidio-default (regex alone is still a system).
- **H10''** External benchmark (§5.4): SS's span F1 or leak-equivalent is not below GLiNER-PII's on data neither system was tuned on.

**GO** = H1'' and H2'' on all three datasets, H3'', H5'', H6'', H8''. **GO-with-caveat** (report says exactly which) = H1'' on two datasets with the third not worse than GLiNER-PII (CI includes 0), everything else held. Anything less is **NO-GO**, in which case the report names the failing types and the next move, and we decide together whether a test-3 is worth its budget.

## Development acceptance targets (dev and dev-large validation half; not hypotheses)

**Acceptance on dev (dev + dev-large validation half) before anything else moves:** per-type leak ≤ GLiNER's on the same rows or ≤ 0.02 for every type; pooled leak ≤ 0.06; natural-slice spurious edits ≤ the current SS's on the same messages (do not trade the moat for recall); perf budget above. If after two training iterations the tagger cannot meet the spurious bound, the problem is in the gate (§3.4), not in the model: tighten the gate for tagger-sourced candidates before loosening the bound.
