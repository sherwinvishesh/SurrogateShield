# AGE probe (synthetic, diagnostic)

Command: `PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.tagger.age_probe --ss bc --out bench/results/age_probe_dev.json` at commit `1c4fd7951697`

602 synthetic messages (seed 2688461741158155648, messages d0433d9e839a); names, cities and ages from the training half of the identity pools. Recall = ages caught / ages (the scorer's leak rule at the known span). Tagger alone `pii-tagger-dv3s-40k`: `as_age` its AGE candidates only, `any` every candidate.

| layout | trained | n | non-ASCII names | tagger as_age @0.5 | tagger as_age @0.7 | tagger as_age @0.9 | tagger any @0.5 | tagger any @0.7 | tagger any @0.9 | SS `frozen` | SS `bc` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence | yes | 75 | 0 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 63/75 | 63/75 |
| form | yes | 75 | 0 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 43/75 | 43/75 |
| json | yes | 75 | 0 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 |
| table | yes | 75 | 0 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 | 75/75 |
| eol | no | 38 | 16 | 26/38 | 23/38 | 20/38 | 38/38 | 35/38 | 28/38 | 0/38 | 35/38 |
| signoff | no | 38 | 18 | 37/38 | 35/38 | 34/38 | 38/38 | 36/38 | 35/38 | 0/38 | 37/38 |
| mid | no | 38 | 17 | 38/38 | 38/38 | 38/38 | 38/38 | 38/38 | 38/38 | 21/38 | 38/38 |
| bracket | no | 38 | 20 | 35/38 | 35/38 | 33/38 | 37/38 | 37/38 | 34/38 | 33/38 | 37/38 |
| from_city | no | 38 | 16 | 34/38 | 33/38 | 32/38 | 37/38 | 36/38 | 34/38 | 22/38 | 38/38 |
| contact | no | 38 | 16 | 38/38 | 38/38 | 38/38 | 38/38 | 38/38 | 38/38 | 28/38 | 28/38 |
| reddit | no | 37 | 0 | 36/37 | 36/37 | 30/37 | 37/37 | 37/37 | 30/37 | 26/37 | 26/37 |
| worded | no | 37 | 16 | 37/37 | 37/37 | 37/37 | 37/37 | 37/37 | 37/37 | 34/37 | 37/37 |
| **trained** |  | 300 |  | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 256/300 | 256/300 |
| **new** |  | 302 |  | 281/302 | 275/302 | 262/302 | 300/302 | 294/302 | 274/302 | 164/302 | 276/302 |
| **sign-off (eol + signoff)** |  |  |  | 63/76 | 58/76 | 54/76 | 76/76 | 71/76 | 63/76 | 0/76 | 72/76 |
| non-ASCII names |  |  |  | 109/119 | 107/119 | 104/119 | 117/119 | 115/119 | 111/119 | 41/119 | 110/119 |
| ASCII names (new layouts) |  |  |  | 136/146 | 132/146 | 128/146 | 146/146 | 142/146 | 133/146 | 97/146 | 140/146 |

Tagger `as_age` recall on the sign-off layouts by threshold (D's trigger: < 0.95 after A–C): 0.3: 0.8289, 0.4: 0.8289, 0.5: 0.8289, 0.6: 0.7895, 0.7: 0.7632, 0.8: 0.7632, 0.9: 0.7105, 0.95: 0.6316.

SS variants: `frozen` config `a1abaa92142b0db9`, code `0e3312e39f48`, 2026-10-08; `bc` config `a1abaa92142b0db9`, code `370c18c0f9fc`, 2026-10-08.
