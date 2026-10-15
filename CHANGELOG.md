# Changelog

## 1.9.0 — 2026-10-15 (`v2` branch)

The first release after 1.0.0. Audit IDs refer to the
technical audit. Measured results, with their commands, are reported
separately and are not part of this file.

### Migration notes (from 1.0.0)

**Python 3.10 or later**
- The package requires Python 3.10 (`requires-python = ">=3.10"`) and
  `transformers>=5.0.0`: the pinned tagger's tokenizer config loads only with
  transformers 5, which has no 3.9 build, and spaCy 3.8 stopped shipping 3.9
  wheels. On 3.9, `pip install surrogateshield` is refused instead of
  installing a package that cannot load the default preset.

**Sessions instead of module-global state (I9, I10)**
- PII state lives in `surrogateshield.Session` objects, one per end-user
  conversation: `with Session() as s: s.mask(...); s.unmask(...)`. Async
  code uses `amask` and `aunmask`.
- The module functions (`mask`, `unmask`, `scan`, `flush`, `forget`) still
  work. They use the session that `use_session()` bound, or else the one owned
  by the calling thread and asyncio task. A thread or task you start does not
  inherit the caller's implicit session; pass it with `use_session(s)`.
- `config()` now changes only the arguments you pass, and returns the
  `Config`. Before, every call reset all the other settings to their defaults.

**Return types and errors**
- `scan()` returns a list of `Detection` objects (text, type, start, end,
  score, source, masked). `scan(text, as_dict=True)` returns the old
  `{text: type}` mapping.
- New `mask_result()` returns the detections and the replacements made.
- `unmask(None)`, and a response with no text, raise `TypeError` instead of
  returning an empty string.
- Detection fails closed (I17). If spaCy, the spaCy model or the
  ContextGuard model is missing or fails, `mask()` raises
  `DetectorUnavailable` and the text is not masked. Before, a stage was
  skipped silently. Either install what the message names, or switch the
  stage off explicitly.
- Storage errors raise `StorageError`. A corrupt store file raises
  `CorruptStoreError` after it is moved aside. A shadow map that cannot be
  written or read raises `ShadowMapStorageError` (a `StorageError`), not a
  bare `OSError`.
- A detector that raises on an input (the tagger, ContextGuard, spaCy, or a
  registered plugin) also fails closed: `mask()` raises
  `DetectorUnavailable` naming the stage and the exception type, and no
  surrogate is issued. The message never carries the model's own error
  text, which can quote the input; the cause is chained (`__cause__`).
- Invalid `config()` values raise `ValueError` with the allowed values.
- `pii_off` entries are case-insensitive: `"person"` and `"PERSON"` (and
  `"gpe"`, `"email"`, …) switch off the same type. Before, a lower-case NER
  type name was accepted and ignored. `"hostname"`, and the aliases
  `"username"` and `"password"`, are accepted.
- `unmask()` reads a Gemini response given as a dict (the text parts of its
  first candidate), as it already did for the SDK object.

**Defaults that changed**
- `detailed_view` is now `False`. It printed original values to stdout.
  The library never prints unless you turn it on.
- `address_mode="auto"` is new. It shifts the house number for service
  queries ("nearest pharmacy to …") and replaces the address otherwise.
  `address_shift_range` is the house-number delta.
- `verify_addresses` has no effect and emits a `DeprecationWarning` (F3).
  The Nominatim check sent every street to a third party and never changed
  the output. The `requests` dependency is gone.
- `context_guard_enabled` is now `False`: in `balanced` the PII tagger reads
  names and ContextGuard is opt-in (`strict` and `classic` turn it on).
  Code that sets it keeps working.
- The EntityTrace thresholds of the default preset are `high` 0.90, `low`
  0.70 and `fallback` 0.75 (1.0.0: 0.85 / 0.60 / 0.65; `classic` keeps
  those). The flat `entity_trace_*_threshold` settings still apply on top.
- `config()` gained `address_mode`, `address_shift_range`, `verify_addresses`,
  `context_guard_model`, `context_guard_device`, `tagger_model`,
  `tagger_revision`, `tagger_device` and `detection`. No setting was
  removed or renamed, and no public name was: `config`, `scan`,
  `pii_finder`, `mask`, `unmask` and `flush` keep their names, `mask` and
  `unmask` their signatures; `scan` gained the keyword `as_dict`.

**Storage (I11, F1, I19)**
- Everything persistent lives under `$SURROGATESHIELD_HOME`, or
  `~/.surrogateshield` by default. Directories are 0700, files 0600, and
  writes are atomic.
- Store ids must match `^[A-Za-z0-9_-]{1,64}$`.
- The format is SSv1, with the kind and id bound as AAD. Pre-v1 files are
  read and upgraded on the next write. `forget()` erases one value, and
  `ShadowMap.erase()` removes a store without decrypting it.
  A 1.0.0 library `<id>.shadowmap` (`nonce || ciphertext`, keyed from the
  `<id>.key` file beside it) opens as before; the next save rewrites it as
  SSv1 under the device key and removes the `.key` file. The app's earlier
  plain-JSON conversation files load too. Both are tested
  (`tests/test_shadow_map.py`).

**Surrogates are seeded and linked (D1–D4, I7)**
- `Session(seed=…)` makes masking deterministic: one RNG per session, and no
  global Faker state.
- A person keeps one surrogate per conversation, in matching case. Their
  e-mail address and handle are built from the surrogate name.
- First names keep the gender of the original. Phones keep their country
  code and format. Dates of birth stay within two years. Places are real
  places. Organisations contain no comma.
- If you compared surrogates against fixed strings in your own tests, pass
  a seed.

**CLI (J13, F5)**
- New `surrogateshield` command with the subcommands `scan`, `mask`,
  `unmask`, `bench`, `doctor` and `download`. `doctor` checks the models and
  the storage permissions; `download` fetches the models ahead of time.

**PII tagger**
- The default `balanced` preset adds a DeBERTa-v3-small token classifier,
  `sherwinvishesh/surrogateshield-pii-tagger` on Hugging Face (MIT,
  ~565 MB). It downloads once on first use, at a fixed commit, and its
  weights are checked against the SHA-256 pin in the detection config. Later
  runs read the local cache with no network; with `HF_HUB_OFFLINE` set and no
  cached copy, detection fails closed. A folder of the same name under
  `$SURROGATESHIELD_MODELS` is used instead when there is one.
- `config()` takes `tagger_model`, `tagger_revision` and `tagger_device` (a
  local folder or another hub id; a commit or a `sha256:` pin; CPU, a GPU id
  or a torch device string). The `classic` preset runs without the tagger.

**App (outside the library)**
- `run.sh` never parses `.env` and works for every provider (I24).
- Settings are validated. An invalid setting falls back to its default, and a
  corrupt settings file is moved aside (I27).
- Chat history is recorded only after a successful call (I18).
- RAG answers restore document surrogates (I16, E3).
- `.gitignore` covers experiment outputs and conversations (I25).

### Detection

Every change below lists the forms it covers. Each has a regression test
named after its audit ID.

**Substitution (E1, E2, C3)**
- Substitution is span-based. No `str.replace` is left on any send or
  preview path, and reconstruction rewrites whole values only.
- ContextGuard spans always match the text at their offsets, including a
  span that runs across a line break.

**Relation gate (I12)**
- A name not tied to a person is kept verbatim: public figures, brands,
  authors of public works, titles, column names, defined terms.
- Widely known people come from a Wikidata gazetteer
  (`core/detection/public_names.py`, `public_people.txt`).
- A name in impersonal text (sixty words or more with no first or second
  person and no identifier within reach) and a lone surname of a public
  figure are kept verbatim. An anchor ties a name only within reach: an
  identifier within 200 characters, another name within the same paragraph.
- A place or an organisation in impersonal text with no identifier or
  surviving name near it is kept verbatim too; a legal form or a vouching
  detector does not rescue it there, a heading line is left alone.
- Trademark marks, lone all-caps or lower-case tagger names without a cue,
  citation authors, product versions read as places, and names whose age
  sentence is about someone else are not PII.
- A name that is also an ordinary word ("Hunter", "Joy") stays a name after
  "my name is", "her name's".

**Ensemble fallback (P2-C1)**
- `entity_trace` gains `options.ensemble` (`{"PERSON": 0.88}` by default):
  in a message where no stage named anyone, a spaCy name at or above that
  score stands in when it is tied to a person (a personal cue before or
  after it, ", who" or "and his" after it, a possessive, a title; never an
  addressee, never in a fiction-shaped exchange, at most three such names).
  A dash that opens the line ties a name only when the name ends the line
  (a sign-off), and a single word right after "my", "our", ... ("our
  Mission") does not stand in. `{}` turns it off.
  spaCy is still not a PERSON source of its own. The `balanced` hash moves
  from `9470f9336d853f67` to `1277eddc6c55a0f7` and `benchmark()`'s from
  `a1abaa92142b0db9` to `19baf5415a894c80`; the test-2 and test-3 results
  keep the hash they were measured at.

**Service queries (I6)**
- A service query needs a request plus a located proximity. A sensitive
  query is made coarse.

**Pattern families (I1, I14)**
- Ages (also with 歲); birth years; birthdays without a year.
- Credentials, in their general forms.
- Labelled IDs: EIN, ITIN, CPF, NI, customer and student numbers, serials,
  IMEI and MEID, plates, health-plan group numbers.
- National-format phones with a trunk 0 (UK, AU, FR, DE, NL, ES, IN, CN),
  with their extensions. Phone labels in Cyrillic, Greek, Arabic, Persian,
  Hebrew, Thai, Vietnamese and Indonesian, and 電話 / 手機.
- Handles: platform frames, stream and seller names, web-log users. An
  identifier in code (a call, an index or an assignment after the value),
  a Vue event, a platform word running on into a longer identifier
  (`snapshot_id`), or a doc-comment tag is not a handle.
- Personal URLs: seller pages, share links, family and vanity domains. A
  property path in a template expression or a call argument
  (`${user.name}`, `$t('menu.home.link')`) is not a domain.
- Connection URLs (`postgres://user:pass@host/db`): the password and the
  deployment host; a host surrogate keeps the last two labels.
- A URL on a contact strip (` | ` or ` · ` fields beside an e-mail, a phone
  or a labelled number) is personal.

**Structural pass (I8, I13)**
- CSV columns, chat speakers, and the parts of names that appear in e-mail
  addresses and handles.
- Fiction-shaped exchanges (labelled turns, two-word or `???` labels, stage
  directions): the labels are not names, and a name with no identifier near
  it is left as written; a narrated story (quoted speech beside speech
  verbs, no first person outside the quotes) follows the same rule, and a
  place a frame rule finds in impersonal text is no one's unless an
  identifier stands near it. Code-shaped speaker labels, form labels inside
  code, generic addressees ("dear customer") and hex dumps are not PII
  either; quoted title lists and line breaks no longer make an organisation.
- Greeting and signature lines.
- Kin frames in English, Hindi, Chinese and Japanese.
- Titles in several languages.
- Names the model misses or clips.

**Addresses**
- Streets, units and postcodes outside the US and UK: Romance, Dutch,
  Nordic and Finnish, German floors, Indian house numbers and PINs, and
  Chinese road addresses.
- Each surrogate keeps the shape of the original.

**Organisations**
- Labelled employers, legal suffixes, payroll lines, "a shop called X" and
  "Title | Org" signatures.
- Romance and lowercase institutions, English parishes and practices, and
  Japanese and Chinese schools and companies.
- An institution's name runs across spaces or tabs only, never a line
  break, and a tagger name or organisation ends at full-width clause
  punctuation (，。；：).
- Surrogates keep the kind word ("family practice", 小学校) and the script.

**Devices**
- Device hostnames named after their owner ("Galaxy-S23-<name>").

**Frames, read after every stage**
- Names in an image-generation prompt (two marks or more: `::2` weights,
  `--ar 16:9`, `/imagine`), in a text with two footnote marks or more, and
  in a citation ("Okafor & Lind, 2019", "Brandt et al.") are no one the
  writer knows; nor is a name the prompt gives the assistant ("you are X,
  a …").
- A place only spaCy read, in a sentence with no "I" or "we", outside a
  form, a table, a sign-off or a contact block.
- A lower-case phrase of two words or more the tagger read as a name, in a
  text that capitalises, neither alone on its line nor in a form.
- A place or an organisation in an English text of sixty words or more
  whose sentence and the ones either side say no "I" or "you", outside a
  form line or a résumé.
- Each of these steps aside for a personal cue; all but the spaCy place
  also for an identifier near the name.

**Misreads**
- Spans whose own words say they are no name, age or secret: a name before
  a version ("Engine 5.x"); a subject after "studying", "a degree in"; a
  placeholder company (ABC, XYZ, Acme, Contoso, Fabrikam, Initech, Foo); a
  kind of company ("a Danish shipping company"); the name of a thing ("the
  name of the hotel is …"; a person's noun keeps it).
- A lower-case kebab slug read as a credential or an ID, and a "credential"
  of six words or more that is a sentence, when no secret's cue stands on
  its line (a lower-case word list, a seed phrase, stays).
- A single lower-case word the structural pass read as a name, in a text
  that capitalises, outside a form, a label, an introduction or a
  delimited cell; a place only spaCy read in a list of four capitalised
  items or more.
- Ages that are no one's: a difference ("6 years older"), a range ("ages
  11 to 14"), an audience ("explain it to a five year old"), and a "how old
  am I" puzzle with no identifier near.
- A country, a US state or a larger region ("Canada", "Ohio", "Southeast
  Asia"); a city stays protected, and a place that is also a given name
  ("Georgia", "Jordan") counts only after "in", "from", "to", ....

**Fixed**
- Devanagari names are no longer split at vowel signs. Before this fix they
  came back as unreadable fragments.
- Town surrogates no longer carry trailing spaces (D3).
- Generation no longer aborts a message on a role handle or a letterless
  name (J4).

### Security hardening (pre-publish audit)
- Logs are value-free at every level. Thirty-one `DEBUG` lines that printed
  a detected or restored value (names, organisations, addresses, surrogates
  under reconstruction) now print offsets or counts. `INFO` and above
  already were. A test runs the whole mask/unmask/scan/forget path, in
  memory and on disk, with the logger at `DEBUG` and asserts that no value
  of any type appears.
- Patterns with nested optional whitespace and punctuation (address,
  organisation, structural and greeting frames, the canonical spelled
  numbers, the service-query fragment) were rewritten to equivalent forms
  that cannot backtrack quadratically. Thirty-two sites; every pattern was
  checked against the old one on a differential corpus with no change in
  output. Before: `"123 Main St"` followed by 10,000 spaces took over 25 s
  to mask, 10,000 newlines 7 s. After: 0.05–0.13 s, linear in the length.
- The tagger's output is validated before it is decoded (a missing row, a
  probability row that does not match the labels, an offset outside the
  text); a malformed output is `DetectorUnavailable`, not a partial mask.
- The tagger loader reports the exception type and the folder, not the
  model's message.
- `run.sh` checks the mode of `.env` with the GNU `stat` form first, so the
  world-readable warning works on Linux as well as macOS.

### Evaluation and CI
- The evaluator has been rebuilt on shared metrics (A1–A12):
  - protection and recognition are reported separately;
  - micro and macro metrics;
  - span overlap;
  - stored final output is scored;
  - Presidio runs from its shipped default config at threshold 0.4.
- The synthetic set is seeded, with gold offsets, a dev/test split and a key
  lint (B1–B6).
- The real-world corpus has dev, dev2, dev3, dev4 and test splits; dev4 is
  scored blind (`bench/realworld.py`).
- `bench/perf.py` measures warm latency, cold start and peak RSS (J15).
- `.github/workflows/tests.yml` runs the offline suite with an 85 % coverage
  gate on `python-library/surrogateshield/core` (F8). The deploy and publish
  workflows are unchanged.
- Detection contract tests (`python-library/tests/test_detection_contract.py`,
  pre-publish audit): the README's PatternScan table, public-type map,
  evaluator label map, quasi-identifier table and module counts are derived
  from the code; each preset runs exactly the documented stages (`classic`
  never constructs the tagger); the address modes, PO boxes, highway forms,
  service queries and the sensitive-topic override behave as documented; the
  quasi-identifier scorer triggers each combination at its level and never
  on a ZIP alone.
- `offline_eval.py --ablation` rows switch off what their labels say. Since
  the tagger joined `balanced`, `pattern_only` and `PatternScan +
  EntityTrace` had still run it.
- `requirements.txt` pins `faker==40.15.0`, the version the real-data
  identity pool hash was recorded at; Faker 40.43 on CI produced a different
  pool.
- README: the PatternScan table lists every type the scanner emits (25 regex
  types plus the address parser) with examples the scanner finds on its own;
  the quasi-identifier table lists all ten combinations under the code's
  names; the sensitive-topic override is described as the code has it
  (medical, mental-health, addiction, shelter and immigration keywords; it
  coarsens the street line to "my area" and keeps the city and state, and
  no legal keyword exists); the stale "(new)" markers are gone; the
  deprecated wrapper is `fuzz_addresses`, not `verify_addresses`.
- `bench/arms/ss_ablate.py` imports the regex compiler under its Python 3.11
  names (`re._compiler`, `re._parser`, `re._constants`) with the 3.10 modules
  as the fallback; `tests/test_regex_ablation.py` could not be collected on
  3.10, so the CI leg for the documented floor had never passed. The two
  constants that exist only from 3.11 (`POSSESSIVE_REPEAT`, `ATOMIC_GROUP`)
  are read only where they exist; a test runs the arm under a simulated
  3.10 `sre_constants`.
- `tests/test_results_readme.py` (pre-publish audit): every benchmark number
  the README quotes (the leak, clean-prompt, odd-format and harmless-edit
  columns, the 11× / 39× ratios and the confidence interval, the natural
  slice, the Nemotron-PII rates and F1, answer quality, multi-turn, attacker,
  speed and memory, the verdict and the Limitations line) is recomputed from
  the result file in `bench/results/` it comes from, and the commit each
  file records is the one `bench/results/README.md` names.
- `run.sh` checks that the chosen interpreter has the `surrogateshield`
  package before the spaCy check and the launch; without a `.venv` the
  launcher fell back to the system `python3` and `main.py` died with a
  `ModuleNotFoundError` traceback.
- The README's JSON Test field table (and the copy in
  `experiment/README.md`) lists all twenty output fields in the code's
  order (`pii_spans`, `final_output`, `presidio_llm_response` and
  `clean_llm_response` were missing), names the six fields that are off
  by default, describes the BERTScore fields as the code computes them,
  and documents the always-present `address_mode` key and the shape of an
  error row. The attacker's "compatible answers files" table names the
  four fields `attacker.py` reads (`pii_detail` was listed; it is never
  read).
- `tests/test_app_library_parity.py`, `tests/test_json_test_fields.py` and
  new cases in `tests/test_runner.py`, `tests/test_rag_roundtrip.py`,
  `tests/test_run_sh.py` and `tests/test_I28_dead_config.py` (pre-publish
  audit): the app's mask and the library's mask give the same output on
  the contract messages, the field tables are derived from the code, an
  answers row holds exactly the enabled fields, the on-disk RAG store
  holds no original value, the launcher stops cleanly without the
  package, and the README's shortcut table is the dashboard's menu list.
- README (pre-publish audit, docs pass): the offline evaluator's section
  matches `offline_eval.py`. The documented `--strict` flag did not exist
  (the parser rejected it); exact-boundary matches are counted alongside the
  span-overlap scores instead. The ablation subsection described a post-hoc
  split of answers files into four "simulated" configurations, with panels
  the evaluation screen never had; it now describes `--ablation`, which
  re-runs the cascade per configuration (PatternScan only, PatternScan +
  EntityTrace, every stage without the post-passes, the shipped cascade)
  and reports each with its F1 difference and bootstrap interval. Tests
  derive the table from `ABLATION_CONFIGS` and check every documented flag
  against the parser.
- The library README, which PyPI renders on its own, opens with the install
  (and the version it describes), the spaCy model download, the pinned
  tagger download and the fail-closed behaviour; its links to
  `CONFIGURATION.md` and `bench/results/README.md` are full URLs. A test
  rejects relative links there.
- Website: the Python floor reads 3.10 (three pages said 3.9).
- Migration notes: the `context_guard_enabled` default, the EntityTrace
  thresholds of `balanced`, the settings `config()` gained, and the 1.0.0
  `.shadowmap` + `.key` upgrade, which `tests/test_shadow_map.py` now checks
  by building the 1.0.0 file from the primitives themselves.
