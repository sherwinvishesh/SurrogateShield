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

The Phase 2 numbers show overfitting. On dev, leaked went from 25.6 % to 0 % and spurious from 28.1 % to 0.75 %. On test, leaked went from 32.3 % to 19.3 % and spurious from 37.5 % to 21.7 %. J2 fails on test. After this run the test split counts as *seen*: later test numbers are reported next to these and are not tuned on.

J4 (`bench/echo.py`) sends 250 turns from the dev corpus and experiment questions through `Pipeline.process_turn` with a provider that echoes the masked message. A turn is corrupted when the restored echo differs from what the user typed. Before Phase 3: 24 of 250 turns corrupted, the first at turn 64. After Phase 3: 0. Surrogates are not yet seeded per session (Phase 4), so each run draws different ones. In 80 runs after the last fix, one run corrupted one turn. Its cause was a random zip surrogate (30303) that equalled a zip the user typed later (see AUDIT_PROGRESS, open item on quote-backs).

After Phase 4, surrogates are seeded: `bench/echo.py` uses `Pipeline(seed=0)` and `bench/realworld.py` seeds record *i* with *i*, so runs repeat exactly. J4 on seeds 0–7 (`seed=` edited in bench/echo.py): 0 corrupted turns each. Seed 0 issued "Ann Arbor" as the surrogate for Tempe, and a later user typed a real Ann Arbor. That later value is masked as a new original and restored correctly. It is listed under `later_original_equals_earlier_surrogate` and not gated. A surrogate issued equal to a value already seen still fails J4.

Phase 5 adds `bench/compare.py`, which runs SurrogateShield and default-config Presidio (threshold 0.4) on the same messages. Two data sets feed it. The first is the seeded synthetic set (`experiment/make_dataset.py`), split 1200 dev / 300 test, with gold offsets. The second is the real-world corpus: dev, plus `dev2`, a second set of 240 messages written for tuning. Before any Phase 5 tuning, the dev2 corpus shows the same picture as test did in Phase 2. SurrogateShield leaks 19.8 % of must-protect values (86 of 435) with 15.1 % spurious edits; Presidio leaks 39.3 % with 34.4 % spurious. On synth dev, SurrogateShield's any-type micro F1 is 0.946 against Presidio's 0.772. One type is worse than Presidio, personal URLs: recall 0.702 vs 1.000. The 0 % dev leak rate is a tuned number and does not describe unseen text.
