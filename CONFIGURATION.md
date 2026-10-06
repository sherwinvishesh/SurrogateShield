# Configuring detection

Everything the detector does is set by one `DetectionConfig`
(`python-library/surrogateshield/core/detection/config.py`). That covers
which stages run, the model and revision each stage loads, thresholds, which
stage may report which type, what happens to each type, and the relation
gate. A config is a frozen dataclass. It serialises to JSON and has a stable
hash (`config_hash()`, SHA-256 of the canonical JSON), so a result can name
the exact config that produced it.

## Choosing a config

| where | how |
|---|---|
| library | `ss.config(detection="strict")`, or `Session(config=Config(detection=...))` with a preset name, a `DetectionConfig` or a dict |
| environment | `SURROGATESHIELD_PRESET=fast`; `SURROGATESHIELD_DETECTION_CONFIG=/path/det.json` (merged onto the preset) |
| command line | `surrogateshield scan --preset strict …`, `--detection-config det.json` (scan, mask, unmask, bench, doctor) |
| code | `preset("balanced").with_stage("context_guard", thresholds={"accept": 0.6}).with_actions(PHONE="redact")` |

A dict or JSON file may be **partial**. It is merged onto the preset it names
(`"preset": "strict"`), or else onto `balanced`:

- `detectors` merge by stage name (a stage's `thresholds` and `options` merge by key);
- `type_sources`, `type_actions` and `type_conflicts` merge by key;
- any other field is replaced.

A whole config (the output of `to_dict()`) comes back unchanged.
`from_partial(d)` does this in code.

```json
{"preset": "balanced",
 "detectors": [{"name": "context_guard", "thresholds": {"accept": 0.6}, "device": 0}],
 "type_actions": {"PHONE": "redact", "GENDER": "keep"},
 "gate_bypass": ["EMAIL"]}
```

### Precedence

1. **Library sessions.**
   - `Config.detection` wins when set.
   - If it is not set, the environment's preset or file applies.
   - Otherwise `balanced`.
   - On top of that, every flat setting in `Config` that differs from its default is applied: `spacy_model`, `context_guard_enabled`, the four thresholds, `context_guard_model` / `_device`, `address_mode`, `address_shift_range`, `service`. So existing code that sets `context_guard_enabled=False` keeps working.
   - `Session(...).detection_config` shows the result.
2. **`run_cascade(text, config=...)`.**
   - The `config` argument wins.
   - If no `config` is passed, the environment applies, else `balanced`.
   - Flat keyword arguments that are passed explicitly (`use_context_guard=False`, `canonical_views=()`, …) are applied on top.
3. **The application (`main.py`, `pipeline.py`, `json_tester.py`).**
   - These read the root `config.py` unless `SURROGATESHIELD_PRESET` or `SURROGATESHIELD_DETECTION_CONFIG` is set.
   - When either is set, the environment config is used as is.
   - The defaults in `config.py` are the `balanced` preset (a test checks this).
4. **CLI.**
   - `--preset` replaces the session's config.
   - `--detection-config` is merged onto it.
   - A file that names a different preset from `--preset` is an error.

## Presets

| preset | what changes from `balanced` | hash (first 16 hex) |
|---|---|---|
| `balanced` | the default: every stage at the benchmark settings | `68b6f9e8aba3a23f` |
| `fast` | EntityTrace (spaCy) off | `733e3211082fb461` |
| `strict` | EntityTrace thresholds high 0.70 / low 0.40 / fallback 0.45; ContextGuard accept 0.50; every type bypasses the relation gate | `28f7bff49e4f4041` |

There are also four ablations, used by the benchmark only:

| ablation | hash (first 16 hex) |
|---|---|
| `no_canonicaliser` | `9a6999141b7394b8` |
| `no_gate` | `f5f172775702cf7b` |
| `no_models` | `0db23db11b36e200` |
| `no_structural` | `07b35e9f70a4a1c7` |

`benchmark()` is the config of the real-data benchmark: `balanced` with
`type_actions["ADDRESS"] = "replace"` (hash `8e463c3c6b7562fd`). The product
default `auto` keeps street and town inside a service query ("is there a
pharmacy near …"), and a leak scorer cannot tell that apart from a miss, so
the benchmark redraws every address. The SS arm writes `detection_config` and
`detection_config_hash` into every `.meta.json`. `bench/realdata/score.py`
records them per arm under `config_hashes`, and a run whose arm mixes two
hashes fails. `tests/test_detection_config.py` pins both hashes. Once test-2
is scored, it also checks that the hash in `realdata_test2.json` is
`benchmark()`'s.

## Stages (`detectors`)

The built-in stages always run in this order, and each one reads the text
with what the earlier ones found already masked:

| stage | does | model (revision, licence) | own thresholds |
|---|---|---|---|
| `pattern_scan` | regexes with checksums and context cues (email, phone, ID, card, IBAN, IP, credential, DOB, age, …) | — | — |
| `canonicaliser` | the same patterns on rewritten views of the text: `worded` (number words to digits), `spelled` (`dot`/`at` in e-mail shapes), `dates`, `folded` (full-width forms, look-alikes, zero-width), `joined` (spaced digit groups); `options.views` picks them | — | — |
| `entity_trace` | spaCy NER | `en_core_web_lg` (3.8.0, MIT) | `high` 0.85, `low` 0.60, `fallback` 0.65 |
| `context_guard` | transformer NER over the masked text | `dslim/distilbert-NER` (`dfa2838a127384aabb82ed7719e16dab84c42a2a`, Apache-2.0) | `accept` 0.70 |
| `pii_tagger` | the PII tagger (off; resolved through the detector registry like a plugin) | — | — |
| `structural` | the layout passes: addresses read part by part, organisation names read whole, names after titles and greetings, host names, Pass S | — | — |

Every `Stage` has these fields:

- `enabled`
- `model` and `revision` (a Hugging Face model loads at its pinned
  revision; for spaCy the revision is the package version, and `doctor`
  warns if the installed one differs)
- `device` (-1 = CPU, else a GPU index)
- `thresholds`: the stage's own keys above, and/or a **public type** mapped to the lowest score kept for that type, e.g. `{"PERSON": 0.8}`
- `window`, `stride`
- `max_latency_ms`
- `options`

A stage that runs past `max_latency_ms` is logged and reported in the timings
as `<stage>_over_budget_ms`. Its candidates are kept, because dropping them
after the time is already spent would only lose detections.

A stage whose name is not built in is a plugin (see below). Plugins run
after `context_guard`, in list order.

## Types

Detections carry internal names (`phone_uk`, `ssn`, `GPE`, …). Every setting
keyed by type accepts the **public type**. `type_actions` also accepts an
internal name, and the internal name wins.

| public | internal |
|---|---|
| PERSON | PERSON, person |
| ORG | ORG |
| LOCATION | GPE, LOC, FAC, implicit_location |
| ADDRESS | address, zip_us, postcode_uk |
| EMAIL | email |
| PHONE | phone_us, phone_uk, phone_intl |
| URL | url, hostname |
| HANDLE | handle |
| ID | id_number, ssn, passport, us_driver_license, license_plate, vin, credit_card, iban, us_bank_number, crypto |
| CREDENTIAL | credential, api_key |
| NETWORK | ip_address, mac_address |
| AGE | age |
| DATE_OF_BIRTH | dob |
| GENDER | gender_indicator |
| OTHER | any other label (spaCy's MISC, a plugin's own) |

## What may report a type (`type_sources`)

`type_sources` maps each public type to the stages allowed to report it:

- Structured types (EMAIL, PHONE, ID, …) default to `pattern_scan`,
  `canonicaliser` and `structural`. A model's guess at a phone number is
  therefore ignored unless you allow it.
- PERSON, ORG, LOCATION and OTHER default to `"*"`, meaning every stage.

```python
cfg = preset("balanced").with_sources(ID=("pattern_scan", "canonicaliser", "pii_tagger"))
```

## What happens to a type (`type_actions`)

| action | effect |
|---|---|
| `replace` (default) | a realistic surrogate of the same kind, restored in the answer |
| `shift` | ADDRESS only: a nearby real-looking address (`address_shift_range` blocks away) |
| `auto` | ADDRESS only, the default: `shift` inside a service query, `replace` otherwise |
| `redact` | a placeholder such as `[PHONE_1]`, unique in the message and the session, restored in the answer |
| `keep` | detected and reported (`masked=False` in `scan`), sent as typed |

`"*"` sets the action for every type not listed. `service_queries` turns
service-query detection (the `auto` rule) on or off.

## The relation gate (`gate`, `gate_bypass`)

The gate keeps a name as typed when the message does not tie it to a person:

- an organisation or place that is only the topic of a question ("How far
  is Tempe from Phoenix?");
- a public figure;
- an acronym or code token.

A place tied to someone ("we live near Reno") is masked. Pattern hits other
than PERSON never pass through the gate.

- `gate=False` turns it off.
- `gate_bypass=("PERSON",)` masks every PERSON whatever the gate says.
- `("*",)` bypasses it for every type, as in `strict`.

## Conflicts (`source_priority`, `type_conflicts`)

These settle which candidate stands when two stages report the same value
differently. They are part of the config and its hash.

- `source_priority` ranks the stages (default `pattern_scan`, `canonicaliser`, `structural`, `pii_tagger`, `context_guard`, `entity_trace`). Plugins not listed rank after it, in `detectors` order.
- `type_conflicts` is keyed by a sorted pair of public types (`"ORG|PERSON"`). Its value names the type that stands, or `"score"`.

They are read by the resolver (V3 §3.5). Until the resolver lands, a value
reported twice keeps its highest-scored candidate, which is the rule every
committed benchmark number used.

## Plugins

A detector is any object with `detect(text, view) -> list[Candidate]`:

- `text` is the message.
- `view` is the canonicalised view of it (look-alikes folded, spelled
  digits written as digits), with `view.original(s, e)` mapping a span back.
  It is None when nothing needed rewriting.
- `Candidate(start, end, type, score)` uses offsets into `text`. The type is
  a public type or an internal name.

```python
import re
import surrogateshield as ss
from surrogateshield import Candidate, register_detector, preset

PATTERN = re.compile(r"\bEMP-\d{6}\b")

class EmployeeIds:
    def detect(self, text, view):
        return [Candidate(m.start(), m.end(), "ID", 0.99) for m in PATTERN.finditer(text)]

register_detector("employee_ids", lambda stage: EmployeeIds())
ss.config(detection=preset("balanced").with_plugin("employee_ids", types=["ID"],
                                                   thresholds={"ID": 0.9}))
```

The factory receives the plugin's `Stage` (`model`, `thresholds`, `options`).
A plugin's candidates get everything a model's do:

- type routing;
- per-type thresholds;
- nothing inside a URL;
- the relation gate;
- the type's action.

If a config lists a plugin that is not registered, detection fails closed
with `DetectorUnavailable`.

A package can register its detector through an entry point instead, which
is loaded the first time a config lists the stage:

```toml
[project.entry-points."surrogateshield.detectors"]
employee_ids = "mypkg.detectors:factory"
```

`examples/employee_id_detector.py` is a complete plugin, tested in
`tests/test_detection_config.py`.

## `surrogateshield doctor`

`doctor` prints:

- the effective config: preset, hash and source, each stage with its model,
  revision, thresholds and budget, the actions and the gate;
- one row per model that an enabled stage loads, showing whether it is
  installed or cached at its revision, and its licence.

Flags:

- `--cold-start` loads each model and times it;
- `--show-config` prints the whole config as JSON;
- `--preset` / `--detection-config` check another config;
- `--smoke` masks one sample message.

```
$ surrogateshield doctor --cold-start
detection config: preset balanced, hash 68b6f9e8aba3a23f (default)
  stage pattern_scan   on
  stage canonicaliser  on
  stage entity_trace   on, en_core_web_lg@3.8.0, fallback=0.65 high=0.85 low=0.6
  stage context_guard  on, dslim/distilbert-NER@dfa2838a1273, accept=0.7
  stage pii_tagger     off
  stage structural     on
  actions        ADDRESS auto (others replace)
  relation gate  on
[  ok] spaCy model en_core_web_lg — version 3.8.0, licence MIT
[  ok] ContextGuard model dslim/distilbert-NER@dfa2838a1273 — licence Apache-2.0
[  ok] cold start entity_trace (en_core_web_lg) — 739 ms
[  ok] cold start context_guard (dslim/distilbert-NER) — 2230 ms
…
```

(Cold-start times are from one laptop run, Apple-silicon CPU.)
