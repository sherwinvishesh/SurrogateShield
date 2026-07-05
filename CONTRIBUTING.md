# Contributing to SurrogateShield

Thanks for your interest in improving SurrogateShield. This project is a
privacy-preserving proxy for LLMs — PII is detected locally, replaced with
realistic surrogates, and restored in the response, so **nothing sensitive ever
leaves the device**. That privacy guarantee shapes everything below: the golden
rule is *never let real PII cross an API boundary, get committed to the repo, or
land in a log or test fixture*.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Ways to contribute

- **Report a bug** — open a [Bug report](https://github.com/sherwinvishesh/SurrogateShield/issues/new?template=bug_report.yml)
- **Request a feature** — open a [Feature request](https://github.com/sherwinvishesh/SurrogateShield/issues/new?template=feature_request.yml)
- **Improve detection** — new PII patterns, NER post-processing, or surrogate generators
- **Improve docs** — the README is the primary reference; corrections and clarifications are always welcome
- **Report a security/privacy issue** — please do **not** open a public issue; follow the [Security Policy](SECURITY.md)

> ⚠️ **Never include real PII in an issue, pull request, test fixture, or commit.**
> When you need example data, use obviously fake values (see *Example data* below).

---

## Development setup

Requires Python 3.9+.

```bash
# Fork and clone
git clone https://github.com/<your-username>/SurrogateShield.git
cd SurrogateShield

# Create and activate a virtual environment  ← required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy model used by EntityTrace (and the optional Presidio panel)
python -m spacy download en_core_web_lg
```

> Installing into the system or base conda Python is the most common cause of
> "package not found" errors at runtime. Always work inside the venv.

The ContextGuard model (`dslim/distilbert-NER`, ~250 MB) downloads automatically
from HuggingFace Hub on first run — no manual step needed.

You do **not** need an API key to run the tests or the PII Finder; only live
chat sessions call an LLM.

---

## Running the tests

All tests run without an API key or network access (LLM calls are mocked).

```bash
source .venv/bin/activate

# Core application: detection, generation, storage, reconstruction, round-trip
python tests/test1.py
python tests/test2.py
python tests/test3.py

# Attacker Experiment module (mocked API)
python tests/test6.py

# Standalone python-library package (134 checks)
python tests/test7.py
```

**Every pull request must keep all of these green.** If you touch the
`python-library/` package, run `tests/test7.py`; if you touch `attacker.py`, run
`tests/test6.py`.

You can also exercise detection interactively with zero API calls:

```bash
python main.py pii-finder
```

---

## Project layout

A quick map of where things live (see the README's *Project Structure* section
for the full tree):

| Area | Path | Notes |
|---|---|---|
| Detection cascade | `detection/` | PatternScan → EntityTrace → ContextGuard, plus post-processing passes A–D |
| Surrogate generation | `generation/logic.py` | MimicGen (Faker-based) |
| Encrypted storage | `storage/logic.py` | ShadowMap (AES-256-GCM) |
| Reconstruction | `reconstruction/logic.py` | ResolvePass (exact / component / fuzzy) |
| Constants & thresholds | `config.py` | **Single source of truth** — change values here, not inline |
| Evaluation & experiments | `evaluator.py`, `json_tester.py`, `attacker.py` | |
| Standalone package | `python-library/surrogateshield/` | Self-contained; **imports nothing from the main app** |

### The `python-library/` boundary

The pip package is deliberately self-contained: it carries its own copies of the
detection, generation, storage, and reconstruction modules and shares no runtime
code with the main application. **Do not add imports from the root modules into
`python-library/surrogateshield/`.** If you fix a bug that exists in both places,
patch both copies and note it in the PR.

---

## Adding a new PII type

A common contribution. The full round-trip generally means touching several
places — use this as a checklist:

1. **Detection** — add the pattern to `detection/pattern_scan.py` (regex/structural)
   or extend the NER post-processing in `detection/logic.py`. Add a validator
   (e.g. checksum) where one exists to suppress false positives, following the
   Luhn / ABA examples.
2. **Generation** — add a type-consistent surrogate generator in
   `generation/logic.py` so the fake value is realistic for its type.
3. **Evaluation labels** — if the type should be scorable, map its answer-key
   label aliases in `evaluator.py`.
4. **Library parity** — mirror the change in `python-library/surrogateshield/core/`.
5. **Tests** — add cases to the relevant `tests/testN.py` file with **fake** values.
6. **Docs** — update the PII-type tables in `README.md` (and
   `python-library/README.md` if it affects the library).

---

## Coding guidelines

- **Match the surrounding style.** Follow existing naming, structure, and comment
  density in the file you're editing. No project-wide reformatting in a feature PR.
- **Keep constants in `config.py`.** Don't hard-code thresholds inline.
- **Preserve the privacy invariant.** Any change to detection, masking, storage,
  or the API call path must not allow a real PII value to reach an LLM,
  persist in plaintext, or appear in a log. Call this out explicitly in your PR.
- **Type-consistency matters.** Surrogates must look like real values of their
  type (a fake SSN must pass format checks, a fake wallet must match Base58, etc.).
- **No new heavyweight dependencies** without discussing it in an issue first.

### Example data

Use clearly synthetic values in tests, docs, and issues:

```
Name:   Sarah Mitchell
Email:  jdoe@example.net
SSN:    544-87-2944          (format-valid, not a real number)
Phone:  +1-480-555-1234      (555 exchange = reserved for fiction)
```

---

## Pull request process

1. Create a branch off `main` (`git checkout -b feat/short-description`).
2. Make your change; keep it focused — one logical change per PR.
3. Run the relevant test files and confirm they pass.
4. Update the README/library docs if behavior or the public surface changed.
5. Push and open a PR against `main`, filling out the
   [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
6. Confirm the CI / PyPI-publish workflows are not broken by your change.

Please describe **what** changed and **why**, and — for anything touching the
detection or masking path — explicitly confirm no real PII can leak.

---

## Questions

Open a [Feature request](https://github.com/sherwinvishesh/SurrogateShield/issues/new?template=feature_request.yml)
or start a discussion in an issue. For anything security- or privacy-sensitive,
use the [Security Policy](SECURITY.md) instead of a public issue.

Thanks for helping keep sensitive data on-device. 🛡️
