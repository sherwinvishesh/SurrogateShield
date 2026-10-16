# The tagger's AGE threshold (V4 §3.1 A)

`... -m bench.tagger.age_sweep --model bench/tagger/build/models/dv3s-40k --out bench/results/age_sweep_dev_guard.json`, model weights `f9e5821151ac`, code `544df16ecb0d`. Reference: `balanced` at `a1abaa92142b0db9`. Choice data only: dev and dev-large's calib half.

| variant | part | AGE leaked | G1 | G2 (types risen) | natural spurious (ref → got) | G3 | natural AGE edits |
|---|---|---|---|---|---|---|---|
| `age-off` | dev | 1/33 (bound 0.02) | no | yes | 133 → 133 | yes | {'age': 17} |
| `age-off` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 315 | yes | {'age': 17, 'age:worded': 1} |
| `age50` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 134 | no | {'age': 18} |
| `age50` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 317 | no | {'age': 19, 'age:worded': 1} |
| `age60` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 134 | no | {'age': 18} |
| `age60` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |
| `age70` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 134 | no | {'age': 18} |
| `age70` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |
| `age80` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 133 | yes | {'age': 17} |
| `age80` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |
| `age85` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 133 | yes | {'age': 17} |
| `age85` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |
| `age90` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 133 | yes | {'age': 17} |
| `age90` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |
| `age95` | dev | 0/33 (bound 0.02) | yes | yes | 133 → 133 | yes | {'age': 17} |
| `age95` | devlarge-calib | 0/68 (bound 0.02) | yes | yes | 315 → 316 | no | {'age': 18, 'age:worded': 1} |

Chosen by G1: **0.95** (meeting G1: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95; meeting G1–G3 on the choice data: none).
