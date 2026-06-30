# Experiment Folder

> Paper available on arXiv: https://arxiv.org/abs/2606.29567

This folder holds all input question sets and their corresponding outputs used for batch testing SurrogateShield.



## File Types

### `*.json`: Questions Only

These are the **input files** fed into the batch tester. Each file is a JSON array where every entry has a single `"input"` key containing the question string.

```json
[
  {
    "input": "hi my name is Revanth and my ssn is 544-87-2944 and give me the tax benefits of the state of wyoming, please"
  },
  {
    "input": "my email is revanth@gmail.com and phone is 480-555-1234, can you help me draft a resignation letter?"
  }
]
```

**Files of this type:** `example.json`, `test.json`



### `*_key.json`: Questions with Answer Keys

These are the **reference/ground-truth files**. Each entry pairs a question with an `"Answer-Key"` object that lists the PII entities expected to be detected. If no PII is present, `"Answer-Key"` is an empty object `{}`.

```json
[
  {
    "Question": "hi my name is Revanth and my ssn is 544-87-2944 and give me the tax benefits of the state of wyoming, please",
    "Answer-Key": {
      "name": "Revanth",
      "ssn": "544-87-2944",
      "GPE": "wyoming"
    }
  },
  {
    "Question": "How do deep-sea hydrothermal vents support marine life ecosystems without sunlight?",
    "Answer-Key": {}
  }
]
```

**Files of this type:** `example_key.json`, `test_key.json`



### `*_answers.json`: Pipeline Output

These are **auto-generated** by the batch tester after running a `*.json` input file through SurrogateShield. The stem name is inherited from the input file (e.g. `test.json` → `test_answers.json`).

Each entry contains the full pipeline output for one question, including all detection stages, surrogate mappings, the sanitized text sent to the LLM, the LLM response, and optional evaluation scores. See the [Output Fields](#output-fields) section below for details.

**Files of this type:** `example_answers.json`, `test_answers.json`



## Extracting Questions from a `*_key.json` File

To generate a `*.json` input file from a `*_key.json` reference file, run from the `experiment/` folder:

```bash
python3 -c "
import json

with open('test_key.json', 'r') as f:
    key_data = json.load(f)

questions = [{'input': entry['Question']} for entry in key_data]

with open('test.json', 'w') as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f'Extracted {len(questions)} questions → test.json')
"
```



## Running the Batch Tester → `*_answers.json`

The batch tester is `json_tester.py` at the project root. It reads a `*.json` input file, runs every question through the full SurrogateShield pipeline, and writes results to `<stem>_answers.json` in this folder.

**From the main menu (`python main.py`):** select **JSON Batch Test**, choose your input file (e.g. `test.json`), and pick which output fields you want included.

**Programmatically**, run from the project root:

```bash
python3 -c "
from json_tester import run_batch, DEFAULT_FIELDS

fields = {**DEFAULT_FIELDS}
fields['llm_response']       = True   # set to False for detection-only (no API key needed)
fields['bertscore_ss']       = True   # BERTScore utility evaluation
fields['bertscore_presidio'] = True   # BERTScore Presidio baseline comparison

output_path = run_batch('test.json', fields)
print(f'Results written to: {output_path}')
"
```

Progress is flushed to disk every 25 questions, so a run can be safely interrupted and resumed; already-processed questions are skipped automatically on restart.



## Output Fields

Each entry in `*_answers.json` can contain the following fields (depending on which were enabled):

| Field | Description |
|---|---|
| `question` | The original input question |
| `pattern_scan_pii` | PII found by regex pattern matching |
| `entity_trace_pii` | PII found by spaCy NER |
| `context_guard_pii` | PII found by DistilBERT context classifier |
| `confirmed_pii` | Final combined list of all confirmed PII |
| `pii_detail` | Per-entity type, confidence score, and detection source |
| `quasi_id_risks` | Combination re-identification risks (e.g. Name + SSN) |
| `surrogate_map` | Mapping of original PII → surrogate replacement |
| `sanitized_input` | The text sent to the LLM after surrogate substitution |
| `recognized_not_replaced` | Entities detected but intentionally not replaced (e.g. topical locations) |
| `llm_response` | Raw response from the LLM |
| `stage_timings_ms` | Per-stage timing breakdown in milliseconds |
| `presidio_sanitized_input` | Presidio `[TYPE]`-redacted text (baseline) |
| `presidio_found_piis` | Raw Presidio detections with type and score |
| `bertscore_ss` | BERTScore comparing original vs SurrogateShield sanitized input |
| `bertscore_presidio` | BERTScore comparing original vs Presidio sanitized input |

---

Made with ❤️ by Sherwin Vishesh Jathanna
