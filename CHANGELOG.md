# Changelog

## Unreleased — `v2` branch (library 2.1.0)

Not released. Nothing here is on PyPI or tagged. Audit IDs refer to the
technical audit. Measured results, with their commands, are reported
separately and are not part of this file.

### Migration notes (from 1.0.0)

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
  `CorruptStoreError` after it is moved aside.
- Invalid `config()` values raise `ValueError` with the allowed values.

**Defaults that changed**
- `detailed_view` is now `False`. It printed original values to stdout.
  The library never prints unless you turn it on.
- `address_mode="auto"` is new. It shifts the house number for service
  queries ("nearest pharmacy to …") and replaces the address otherwise.
  `address_shift_range` is the house-number delta.
- `verify_addresses` has no effect and emits a `DeprecationWarning` (F3).
  The Nominatim check sent every street to a third party and never changed
  the output. The `requests` dependency is gone.

**Storage (I11, F1, I19)**
- Everything persistent lives under `$SURROGATESHIELD_HOME`, or
  `~/.surrogateshield` by default. Directories are 0700, files 0600, and
  writes are atomic.
- Store ids must match `^[A-Za-z0-9_-]{1,64}$`.
- The format is SSv1, with the kind and id bound as AAD. Pre-v1 files are
  read and upgraded on the next write. `forget()` erases one value, and
  `ShadowMap.erase()` removes a store without decrypting it.

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
  `unmask`, `bench` and `doctor`. `doctor` checks the models and the storage
  permissions.

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

**Service queries (I6)**
- A service query needs a request plus a located proximity. A sensitive
  query is made coarse.

**Pattern families (I1, I14)**
- Ages; birth years; birthdays without a year.
- Credentials, in their general forms.
- Labelled IDs: EIN, ITIN, CPF, NI, customer and student numbers, serials,
  IMEI and MEID, plates, health-plan group numbers.
- National-format phones with a trunk 0 (UK, AU, FR, DE, NL, ES, IN, CN),
  with their extensions.
- Handles: platform frames, stream and seller names, web-log users.
- Personal URLs: seller pages, share links, family and vanity domains.

**Structural pass (I8, I13)**
- CSV columns, chat speakers, and the parts of names that appear in e-mail
  addresses and handles.
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
- Surrogates keep the kind word ("family practice", 小学校) and the script.

**Devices**
- Device hostnames named after their owner ("Galaxy-S23-<name>").

**Fixed**
- Devanagari names are no longer split at vowel signs. Before this fix they
  came back as unreadable fragments.
- Town surrogates no longer carry trailing spaces (D3).
- Generation no longer aborts a message on a role handle or a letterless
  name (J4).

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
