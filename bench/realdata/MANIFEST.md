# Real-data benchmark — manifest

Generated from `bench/realdata/manifest.json` by `bench/realdata/manifest.py`; do not edit by hand.

## Sources

| dataset | Hugging Face repo | revision | files | licence |
|---|---|---|---|---|
| oasst1 | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `data/train-00000-of-00001-b42a775f407cee45.parquet`, `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` | Apache-2.0 |
| sharegpt | `RyokoAI/ShareGPT52K` | `6f9b78cc1dd15dbb51d3c51ccc219c558962fd77` | `sg_90k_part1.json` | CC0-1.0 |
| wildchat | `allenai/WildChat-1M` | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | `data/train-00004-of-00014.parquet`, `data/train-00008-of-00014.parquet` | ODC-BY-1.0 |

Files were fetched with `huggingface_hub.hf_hub_download(repo, file, revision=<sha>)` into the
git-ignored `bench/realdata/raw/`. WildChat-1M has 14 shards; two were used, drawn with
`sorted(random.Random(20261005).sample(range(14), 2))` = [4, 8]. WildChat-1M is the non-toxic release,
so `toxic == True` excludes nothing in it (counted below); conversations the publishers marked
`redacted` (PII they found was scrubbed) are kept and flagged in `meta.redacted`.

## Redistribution and what is committed

- **oasst1**: Apache-2.0 permits redistribution with attribution.
- **sharegpt**: CC0-1.0 as declared by the uploader; the underlying conversations were shared by ShareGPT users, so the provenance of individual texts is not verifiable.
- **wildchat**: ODC-BY-1.0 permits redistribution with attribution; AI2's terms ask users not to attempt re-identification.

Real prompts can carry real personal data even after the publishers' scrubbing, so no natural-slice
text is committed, whatever the licence allows. `<dataset>/pool.jsonl` holds the source id, the
references that locate each user turn in the pinned files, the SHA-256 and word count of each turn,
and the split. `python -m bench.realdata.build` rebuilds the text into the git-ignored
`bench/realdata/build/` and checks every hash. Injected rows (a PII-free base plus our fake values)
are committed in full.

## Filters and sampling

Fixed in `bench/realdata/pull.py` before any system ran. Single-turn prompt = the first user turn;
multi-turn = the first ≤ 3 user turns of a conversation with ≥ 2 (OASST1: along the best-ranked
reply chain from the root). First turns: 15–400 words, not only code (< 8 alphabetic words once code
fences, URLs and code-like lines are removed: `common._CODE_LINE`), not only a URL (< 5 words besides
URLs), no jailbreak template. Later turns: ≤ 400 words, English when ≥ 5 words,
no jailbreak template. ShareGPT has no language field: English = ≥ 90 % ASCII letters and ≥ 12 % English
function words (`common.is_english`). Exact duplicates (lower-cased, punctuation-stripped,
whitespace-collapsed) collapse to the smallest source id; near duplicates (rapidfuzz ratio ≥ 95 on the
normalised first turn) are rejected during the seeded draw. Multi-turn conversations are drawn first;
single-turn prompts come from the remaining conversations. Split 20 % dev / 80 % test by a seeded draw.

Jailbreak / prompt-injection keyword pattern (case-insensitive): `\b(do anything now|jailbr(?:eak|oken)|ignore (?:all )?(?:the )?(?:previous|prior|above) (?:instructions|prompts?)|developer mode|DAN mode|act as DAN|you are DAN|unfiltered and amoral|without any (?:moral|ethical) (?:restrictions|guidelines)|no (?:ethical|moral) (?:restrictions|guidelines)|uncensored (?:ai|model|assistant)|opposite mode|evil confidant)\b`

Check of the English heuristic against WildChat's own language label (`python -m
bench.realdata.pull --english-check`; first user turns of 15–400 words in a seeded sample of 20,000
WildChat conversations) is under sharegpt below.

### oasst1

| step | count |
|---|---|
| `source_roots` | 10,364 |
| `drop_deleted` | 1 |
| `drop_not_english` | 6,693 |
| `drop_first_turn_too_short` | 1,775 |
| `drop_first_turn_too_long` | 6 |
| `drop_first_turn_only_code` | 2 |
| `drop_first_turn_jailbreak` | 3 |
| `first_turn_kept` | 1,884 |
| `single_exact_duplicates` | 1 |
| `drawn_single` | 500 |
| `multi_candidates` | 1,399 |
| `drop_multi_later_turn_too_long` | 1 |
| `multi_kept` | 1,398 |
| `multi_exact_duplicates` | 0 |
| `drawn_multi` | 100 |

Drawn: multi_dev = 20, multi_test = 80, single_dev = 100, single_test = 400

Seeds (`common.derive_seed(dataset, *key)`): multi = 8297385324506450750, multi/split = 51635376429254779, single = 4200958982437220154, single/split = 149393221693701657

Raw file SHA-256:

- `data/train-00000-of-00001-b42a775f407cee45.parquet` `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e`
- `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` `24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416`

### sharegpt

| step | count |
|---|---|
| `source_conversations` | 45,332 |
| `drop_not_starting_with_user` | 1,673 |
| `drop_first_turn_empty` | 5 |
| `drop_first_turn_not_english` | 14,651 |
| `drop_first_turn_too_short` | 9,369 |
| `drop_first_turn_too_long` | 1,929 |
| `drop_first_turn_only_code` | 106 |
| `drop_first_turn_jailbreak` | 286 |
| `first_turn_kept` | 17,313 |
| `single_exact_duplicates` | 2,158 |
| `single_near_duplicates_skipped` | 2 |
| `drawn_single` | 500 |
| `multi_candidates` | 12,464 |
| `drop_multi_later_turn_empty` | 1 |
| `drop_multi_later_turn_too_long` | 421 |
| `drop_multi_later_turn_not_english` | 859 |
| `drop_multi_later_turn_jailbreak` | 4 |
| `multi_kept` | 11,179 |
| `multi_exact_duplicates` | 1,445 |
| `drawn_multi` | 100 |

Drawn: multi_dev = 20, multi_test = 80, single_dev = 100, single_test = 400

Seeds (`common.derive_seed(dataset, *key)`): multi = 7337422658528046808, multi/split = 2737957914994815187, single = 1727909325923492305, single/split = 6632606893631942324

Raw file SHA-256:

- `sg_90k_part1.json` `ed0e3af68ed968aed86aafd3796dacdb6fc2bb91f82edec880086ffcb7280f56`

**english_heuristic_vs_wildchat_label**

```
{
 "false_neg": 291,
 "false_pos": 117,
 "n": 8518,
 "precision": 0.9808,
 "recall": 0.9535,
 "true_neg": 2140,
 "true_pos": 5970
}
```

### wildchat

| step | count |
|---|---|
| `source_conversations` | 119,713 |
| `drop_duplicate_hash` | 1,428 |
| `drop_not_english` | 51,104 |
| `drop_first_turn_empty` | 204 |
| `drop_first_turn_too_short` | 16,212 |
| `drop_first_turn_too_long` | 13,276 |
| `drop_first_turn_only_code` | 601 |
| `drop_first_turn_jailbreak` | 148 |
| `first_turn_kept` | 36,740 |
| `single_exact_duplicates` | 6,271 |
| `single_near_duplicates_skipped` | 11 |
| `drawn_single` | 500 |
| `multi_candidates` | 10,678 |
| `drop_multi_later_turn_empty` | 301 |
| `drop_multi_later_turn_too_long` | 299 |
| `drop_multi_later_turn_jailbreak` | 3 |
| `multi_kept` | 10,075 |
| `multi_exact_duplicates` | 706 |
| `drawn_multi` | 100 |

Drawn: multi_dev = 20, multi_test = 80, single_dev = 100, single_test = 400

Seeds (`common.derive_seed(dataset, *key)`): multi = 8004489117543738831, multi/split = 8872168885709066928, single = 8280196155434402060, single/split = 6786772035386712793

Raw file SHA-256:

- `data/train-00004-of-00014.parquet` `08c7e0c31edc120606d67cfa82486471d82945a7a414761501ea44b1ab0cd804`
- `data/train-00008-of-00014.parquet` `a69578183dde5de1aa94134da2d7c2d13f00d88a577ef91140a492fbbbf10792`

## Systems under test

Recorded by `python -m bench.arms.run` from each arm's meta sidecar
(`bench/results/spans/<arm>/*.meta.json`); every arm runs offline.

### gliner_pii

Interpreter `.venv-baselines/bin/python`, seed 20261005.

```
{
 "backbone_revision": {
  "microsoft/mdeberta-v3-base": "a0484667b22365f84929a935b5e50a51f71f159d"
 },
 "flat_ner": true,
 "labels": [
  "person",
  "organization",
  "email",
  "phone number",
  "address",
  "location",
  "date of birth",
  "age",
  "passport number",
  "social security number",
  "credit card number",
  "bank account number",
  "iban",
  "driver's license number",
  "national id number",
  "tax identification number",
  "health insurance id number",
  "ip address",
  "url",
  "username",
  "password",
  "api key"
 ],
 "model": "urchade/gliner_multi_pii-v1",
 "model_revision": "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
 "overlap_words": 30,
 "replacement": "[LABEL] placeholder",
 "threshold": 0.5,
 "versions": {
  "gliner": "0.2.29",
  "python": "3.12.9",
  "torch": "2.14.1",
  "transformers": "4.51.3"
 },
 "window_words": 200
}
```

### llm_guard

Interpreter `.venv-baselines/bin/python`, seed 20261005.

```
{
 "entity_types": [
  "CREDIT_CARD",
  "CRYPTO",
  "EMAIL_ADDRESS",
  "IBAN_CODE",
  "IP_ADDRESS",
  "PERSON",
  "PHONE_NUMBER",
  "US_SSN",
  "US_BANK_NUMBER",
  "CREDIT_CARD_RE",
  "UUID",
  "EMAIL_ADDRESS_RE",
  "US_SSN_RE"
 ],
 "faker": "module Faker re-seeded per message (seed_instance)",
 "faker_arity_shim": "one-argument lambdas in _entity_faker_map called with a dummy argument (0.3.16 bug)",
 "model": "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
 "model_revision": "9ea992753ab2686be4a8f64605ccc7be197ad794",
 "recognizer_conf": "DEBERTA_AI4PRIVACY_v2_CONF",
 "scanner": "llm_guard.input_scanners.Anonymize(Vault(), use_faker=True)",
 "threshold": 0.5,
 "versions": {
  "faker": "37.12.0",
  "llm-guard": "0.3.16",
  "presidio-analyzer": "2.2.358",
  "python": "3.12.9",
  "torch": "2.14.1",
  "transformers": "4.51.3"
 }
}
```

### presidio_default

Interpreter `.venv/bin/python`, seed 20261005.

```
{
 "config": "AnalyzerEngineProvider() shipped defaults (default_analyzer.yaml, spacy.yaml, default_recognizers.yaml)",
 "overlap_resolution": "longest span, then highest score",
 "presidio_analyzer_version": "2.2.362",
 "replacement": "[ENTITY_TYPE] placeholder (presidio/redact.py)",
 "score_threshold": 0.4
}
```

### presidio_faker

Interpreter `.venv/bin/python`, seed 20261005.

```
{
 "config": "AnalyzerEngineProvider() shipped defaults (default_analyzer.yaml, spacy.yaml, default_recognizers.yaml)",
 "faker_types": [
  "CREDIT_CARD",
  "DATE_TIME",
  "EMAIL_ADDRESS",
  "IBAN_CODE",
  "IP_ADDRESS",
  "LOCATION",
  "NRP",
  "ORGANIZATION",
  "PERSON",
  "PHONE_NUMBER",
  "URL",
  "US_BANK_NUMBER",
  "US_SSN"
 ],
 "merge": "AnonymizerEngine default: same-type entities separated only by spaces become one",
 "overlap_resolution": "longest span, then highest score",
 "presidio_analyzer_version": "2.2.362",
 "replacement": "presidio_anonymizer custom operator: Faker('en_US') seeded per message, per-message (type, value) memo; types without a Faker provider keep their shape (digit\u2192digit, letter\u2192letter)",
 "score_threshold": 0.4,
 "versions": {
  "faker": "40.15.0",
  "presidio-anonymizer": "2.2.362",
  "python": "3.13.2"
 }
}
```

### presidio_transformers

Interpreter `.venv-baselines/bin/python`, seed 20261005.

```
{
 "model_revision": "78f2152eb93ddd817290ce8dbe46f1a6685e09fc",
 "nlp_configuration": {
  "models": [
   {
    "lang_code": "en",
    "model_name": {
     "spacy": "en_core_web_sm",
     "transformers": "obi/deid_roberta_i2b2"
    }
   }
  ],
  "ner_model_configuration": {
   "aggregation_strategy": "max",
   "alignment_mode": "expand",
   "labels_to_ignore": [
    "O"
   ],
   "low_confidence_score_multiplier": 0.4,
   "low_score_entity_names": [
    "ID"
   ],
   "model_to_presidio_entity_mapping": {
    "AGE": "AGE",
    "DATE": "DATE_TIME",
    "EMAIL": "EMAIL",
    "FACILITY": "LOCATION",
    "GPE": "LOCATION",
    "HCW": "PERSON",
    "HOSP": "ORGANIZATION",
    "HOSPITAL": "LOCATION",
    "ID": "ID",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "NORP": "NRP",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "PATIENT": "PERSON",
    "PATORG": "ORGANIZATION",
    "PER": "PERSON",
    "PERSON": "PERSON",
    "PHONE": "PHONE_NUMBER",
    "STAFF": "PERSON",
    "TIME": "DATE_TIME",
    "VENDOR": "ORGANIZATION"
   },
   "stride": 16
  },
  "nlp_engine_name": "transformers"
 },
 "overlap_resolution": "longest span, then highest score",
 "replacement": "[ENTITY_TYPE] placeholder",
 "score_threshold": 0.4,
 "versions": {
  "presidio-analyzer": "2.2.358",
  "python": "3.12.9",
  "spacy": "3.8.16",
  "spacy-huggingface-pipelines": "0.0.4",
  "torch": "2.14.1",
  "transformers": "4.51.3"
 }
}
```

### ss

Interpreter `.venv/bin/python`, seed 20261005.

```
{
 "ADDRESS_MODE": "auto",
 "SERVICE_QUERY_DETECTION_ENABLED": true,
 "refusal": "RuntimeError('could not generate a surrogate ...') from MimicGen.generate_all -> refused row",
 "send_path": "json_tester.prepare_send(text, MimicGen(seed=msg_seed))",
 "versions": {
  "faker": "40.15.0",
  "python": "3.13.2",
  "spacy": "3.8.14",
  "surrogateshield": "2.1.0",
  "torch": "2.12.0",
  "transformers": "5.8.1"
 }
}
```

## Silver labels

```
{
 "annotator": "claude-sonnet-4-6",
 "literal_rule_arms": [
  "ss",
  "presidio_default",
  "gliner_pii"
 ],
 "pooled_arms": [
  "ss",
  "presidio_default",
  "presidio_faker",
  "presidio_transformers",
  "llm_guard",
  "gliner_pii"
 ],
 "prompt_version": "bebc6e281801fa74"
}
```

## Frozen files (SHA-256)

| file | SHA-256 |
|---|---|
| `bench/realdata/oasst1/labels.jsonl` | `cd9fa676fc0aa67b3d1f080be0b675d73f55d55bed7f506b435650dba4c66216` |
| `bench/realdata/oasst1/pii_free.json` | `67ba6132db47b73ee2a04c9b99148f368f3a8a1c4967eddd2ae4de55c9bc7456` |
| `bench/realdata/oasst1/pool.jsonl` | `0573dfdfdc2ace6524d3979b5cbe16f0f528892472cf505cbcbb01b127482dc0` |
| `bench/realdata/sharegpt/labels.jsonl` | `4490f4d3a7cc39a8ccbb4d638fad2cdb06fe0092759cde1b61bea3361357e060` |
| `bench/realdata/sharegpt/pii_free.json` | `2207dd6db4f83e3e2d705bb7d5c83aee61ea2ecbe3c2122e09fff851ea37691b` |
| `bench/realdata/sharegpt/pool.jsonl` | `f9d7636ffeb99beba8487fdbf3bf39d94f877823f86c42aa945723798dc2564d` |
| `bench/realdata/wildchat/labels.jsonl` | `8e2669d041deb6e91de36e49a1b44cb112cf9853c0b5b5682320bfa4974b628f` |
| `bench/realdata/wildchat/pii_free.json` | `fd096fe8325d5dd1a54dab3c14dc512069e7772790f2914d56efb1ea190b124d` |
| `bench/realdata/wildchat/pool.jsonl` | `912b55e509227bdf0a2c6442352ac668d44153febd369d3930489c0154843c9f` |
