# bench/results

Summaries written by the bench scripts (counts only — no message text).

| File | Command | Code |
|---|---|---|
| `j2_baseline_dev.json` | `python bench/realworld.py --split dev --json bench/results/j2_baseline_dev.json` | bf142ed (before any J2 tuning) |
| `j2_baseline_test.json` | `python bench/realworld.py --split test --json bench/results/j2_baseline_test.json` | bf142ed (before any J2 tuning) |
| `j2_phase2_dev.json` | `python bench/realworld.py --split dev --json bench/results/j2_phase2_dev.json` | 21f1574 (end of Phase 2; tuned on dev) |
| `j2_phase2_test.json` | `python bench/realworld.py --split test --json bench/results/j2_phase2_test.json` | 21f1574 (end of Phase 2; first test run after tuning) |

The Phase 2 numbers show overfitting. On dev, leaked went from 25.6 % to 0 % and spurious from 28.1 % to 0.75 %. On test, leaked went from 32.3 % to 19.3 % and spurious from 37.5 % to 21.7 %. J2 fails on test. After this run the test split counts as *seen*: later test numbers are reported next to these and are not tuned on.
