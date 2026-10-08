# SurrogateShield on WildChat's non-English prompts

Command: `python -m bench.realdata.foreign --out bench/results/foreign_wildchat.json` at commit `45754af15364`

No gold: edits bound over-redaction from above and leak is not measured. English is the control from the same shards. Rates with Wilson 95 % intervals.

| language | gate reads | messages | gate guesses it | edited messages | edits | edits / message | foreign-fragment drops | other model drops |
|---|---|---|---|---|---|---|---|---|
| English | – | 100 | 99.0 % [94.5, 99.8] | 23.0 % [15.8, 32.1] | 147 | 1.47 | 0 | 40 |
| German | de | 100 | 79.0 % [70.0, 85.8] | 26.0 % [18.4, 35.4] | 76 | 0.76 | 37 | 214 |
| French | fr | 100 | 94.0 % [87.5, 97.2] | 13.0 % [7.8, 21.0] | 34 | 0.34 | 4 | 18 |
| Spanish | es | 100 | 91.0 % [83.8, 95.2] | 17.0 % [10.9, 25.6] | 34 | 0.34 | 8 | 25 |
| Italian | it | 100 | 94.0 % [87.5, 97.2] | 7.0 % [3.4, 13.8] | 18 | 0.18 | 17 | 30 |
| Portuguese | pt | 100 | 90.0 % [82.6, 94.5] | 20.0 % [13.3, 28.9] | 44 | 0.44 | 4 | 22 |
| Dutch | nl | 92 | 88.0 % [79.8, 93.2] | 16.3 % [10.1, 25.2] | 31 | 0.337 | 3 | 30 |
| Russian | – | 100 | 100.0 % [96.3, 100.0] | 18.0 % [11.7, 26.7] | 75 | 0.75 | 0 | 60 |

For English and Russian the gate's guess is right when it reads none of its six languages.
