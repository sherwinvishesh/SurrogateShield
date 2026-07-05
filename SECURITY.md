# Security Policy

SurrogateShield is a privacy tool: its entire purpose is to keep personally
identifiable information (PII) on-device and out of third-party LLM APIs. A
vulnerability here can directly expose sensitive data, so we take reports
seriously and ask that you disclose them responsibly.

## Supported versions

Security fixes are released for the latest published version. We recommend always
running the newest release from PyPI.

| Version | Supported |
|---|---|
| Latest release (`pip install -U surrogateshield`) | ✅ |
| Older `0.x` releases | ❌ — please upgrade |

## Reporting a vulnerability

**Please do not report security or privacy vulnerabilities through public GitHub
issues, pull requests, or discussions.**

Instead, use one of the following private channels:

1. **GitHub private advisory (preferred)** — open a report via
   [Security → Report a vulnerability](https://github.com/sherwinvishesh/SurrogateShield/security/advisories/new).
2. **Email** — send details to **sjathann@asu.edu** with the subject line
   `SECURITY: SurrogateShield`.

To help us triage quickly, please include:

- A description of the issue and the impact you believe it has
- The version / commit you tested against
- Step-by-step reproduction instructions
- A proof of concept if you have one
- Any suggested remediation

> ⚠️ **Do not include real PII in your report.** Use clearly synthetic values
> (e.g. `544-87-2944`, `jdoe@example.net`) when demonstrating the issue.

### What to expect

- **Acknowledgement** within 3 business days.
- **An initial assessment** (severity, whether it's in scope) within 7 business days.
- **Progress updates** as we work on a fix.
- **Coordinated disclosure**: we will agree on a public disclosure date with you,
  and credit you in the advisory unless you prefer to remain anonymous.

Please give us a reasonable opportunity to release a fix before any public
disclosure.

## Scope

Issues that are **in scope** include, but are not limited to:

- **PII leakage** — any path where a real PII value can reach an LLM API,
  a network request, a log, or an unencrypted file.
- **ShadowMap / cryptography** — weaknesses in the AES-256-GCM encryption,
  HKDF key derivation, nonce handling, or `~/.surrogateshield/device.key`
  storage (`0o600` expected).
- **Detection bypass** — inputs that cause SentinelLayer to miss PII it should
  catch, in a way that undermines the privacy guarantee.
- **Reconstruction corruption** — ResolvePass restoring the wrong original into
  a response, or exposing one session's mapping to another.
- **Dependency vulnerabilities** with a demonstrable impact on the above.

The following are generally **out of scope**:

- Missing detection of PII types the tool never claimed to support (file a
  [feature request](https://github.com/sherwinvishesh/SurrogateShield/issues/new?template=feature_request.yml) instead).
- Issues that require an attacker to already have local read access to the
  victim's machine / home directory (that access defeats any local tool).
- Third-party LLM providers' own data handling once surrogate-only text has
  been sent (by design, only surrogates leave the device).
- Reports from automated scanners without a demonstrated, exploitable impact.

## Security design reference

For background on the tool's threat model and cryptographic design, see the
**Security Design** and **Privacy Guarantees** sections of the
[README](README.md#security-design). Key properties:

- Only surrogates are transmitted; real values never leave the device.
- Surrogate → original mappings are stored **AES-256-GCM encrypted**, never in plaintext.
- Per-conversation keys are derived via HKDF-SHA256 from a device-local secret.
- `.gitignore` excludes `*.shadowmap`, `conversations/*.json`, `device.key`, and `.env`.

Thank you for helping keep sensitive data where it belongs — on-device.
