# The tagger's AGE threshold (V4 §3.1 A)

`... -m bench.tagger.age_sweep --model bench/tagger/build/models/dv3s-40k-v4 --out bench/results/age_sweep_dev.json`, model weights `dc75ed07f2a0`, code `370c18c0f9fc`. Reference: `balanced` at `a1abaa92142b0db9`. Choice data only: dev and dev-large's calib half.

| variant | part | AGE leaked | G1 | G2 (types risen) | natural spurious (ref → got) | G3 | natural AGE edits |
|---|---|---|---|---|---|---|---|
| `age-off` | dev | 1/33 (bound 0.02) | no | no {"ORG": [3, 4]} | 133 → 158 | no | {'age': 17} |
| `age-off` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 366 | no | {'age': 17, 'age:worded': 1} |
| `age50` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age50` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 387 | no | {'age': 36, 'age:worded': 1} |
| `age60` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age60` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 384 | no | {'age': 33, 'age:worded': 1} |
| `age70` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age70` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 383 | no | {'age': 32, 'age:worded': 1} |
| `age80` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age80` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 378 | no | {'age': 27, 'age:worded': 1} |
| `age85` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age85` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 378 | no | {'age': 27, 'age:worded': 1} |
| `age90` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age90` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 375 | no | {'age': 24, 'age:worded': 1} |
| `age95` | dev | 0/33 (bound 0.02) | yes | no {"ORG": [3, 4]} | 133 → 162 | no | {'age': 22} |
| `age95` | devlarge-calib | 0/68 (bound 0.02) | yes | no {"PERSON": [0, 1]} | 315 → 371 | no | {'age': 22, 'age:worded': 1} |

No threshold passes all three checks on the choice data.
