<!--
Thanks for contributing to SurrogateShield!
Please fill out the sections below. See CONTRIBUTING.md for the full process.
⚠️ Never include real PII in this PR, its diffs, tests, or description — use fake values.
-->

## Summary

<!-- What does this PR change, and why? -->

## Related issue

<!-- e.g. Closes #123 -->

## Type of change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 🔍 New PII type / detector
- [ ] 🎨 Surrogate generation change
- [ ] 🔐 Storage / crypto change
- [ ] 📝 Documentation
- [ ] ♻️ Refactor / cleanup
- [ ] 🧪 Tests only

## How was this tested?

<!-- Commands you ran and their result. All tests run without an API key. -->

- [ ] `python tests/test1.py`
- [ ] `python tests/test2.py`
- [ ] `python tests/test3.py`
- [ ] `python tests/test6.py` (if `attacker.py` changed)
- [ ] `python tests/test7.py` (if the `python-library/` package changed)
- [ ] Manually verified with `python main.py pii-finder`

## Privacy checklist

<!-- SurrogateShield's core guarantee: no real PII ever leaves the device. -->

- [ ] This change does **not** allow a real PII value to reach an LLM API, a network request, a log, or an unencrypted file.
- [ ] Surrogates remain **type-consistent** (fake values still look valid for their type).
- [ ] No real PII appears anywhere in this PR (code, tests, fixtures, description).
- [ ] Thresholds/constants were added or changed in `config.py`, not hard-coded inline.

## Library parity

- [ ] If detection/generation/storage/reconstruction logic changed in the main app, the same change was mirrored in `python-library/surrogateshield/core/` (or N/A).

## Documentation

- [ ] Updated `README.md` and/or `python-library/README.md` if behavior or the public surface changed (or N/A).

## Additional notes

<!-- Anything reviewers should know. -->
