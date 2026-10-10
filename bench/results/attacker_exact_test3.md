# E8 on test-3, read letter for letter (made 2026-10-09, after the fact)

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json` at commit `ed85b0b824a6`. Inputs: the committed `bench/results/attacker_realdata_test3.json` (e9519ee3d5ae, run at `d777625`) and the per-value outcomes its run cached (3,359 rows, no text), which reproduce all 24 committed arm summaries. No provider call was made.

**The pre-registered verdict stands: test-3 is NO-GO (`bench/results/verdict_test3.json`), on H8'''.** H8''' counts exact and partial recoveries (SS 102/620 vs GLiNER-PII 11/556). This file reads the same replies with exact recovery only, as the authors asked on 2026-10-09, after that result; it is reported beside the pre-registered reading and does not change the verdict.

**The authors' decision (2026-10-09): go for the paper**, under the revised attacker clause H8-rev in place of H8'''. H8-rev: SS's exact recovery (letter for letter, over the values it replaced) is not above GLiNER-PII's by point estimate, pooled; each dataset and every other baseline reported beside. Adopted by the authors on 2026-10-09, after test-3 was scored, in place of H8''' for the paper. It **holds**: SS 4/620 (0.7 %, 0.2–1.7) vs GLiNER-PII 6/556 (1.1 %, 0.5–2.3), ss − gliner_pii -0.004 [-0.015, +0.006] over 178 messages; on each dataset: oasst1 yes, sharegpt yes, wildchat yes; SS's exact recovery is the lowest of all arms (presidio_default 14/428 (3.3 %, 2.0–5.4), presidio_faker 8/428 (1.9 %, 0.9–3.6), llm_guard 19/264 (7.2 %, 4.7–11.0), gliner_pii 6/556 (1.1 %, 0.5–2.3)). Every other GO-rule part as scored: H1''' yes, H2''' yes, H3''' yes, H5''' yes, H6''' yes. Reason for the revision: a zero-recovery clause is falsified by one hit and measures nothing against the alternatives; the comparative clause measures SS against the strongest baseline on the same prompts. Partial recoveries are parts the surrogates keep by design and are reported, not counted. This is a deviation from the pre-registration, made after the result. The paper states the pre-registered clause, that it was missed (exact recoveries in both conditions, all URLs whose surrogate kept the path), and the clause it reports under. Nothing was re-run; no provider call was made.

| group | arm | values | left in | unavailable | exact (of the values the arm replaced) | known letter for letter (exact + left in) | partial, set aside |
|---|---|---|---|---|---|---|---|
| all | ss | 636 | 7 | 9 | 4/620 (0.7 %, 0.2–1.7) | 11/627 (1.8 %, 1.0–3.1) | 98 |
| all | presidio_default | 636 | 201 | 7 | 14/428 (3.3 %, 2.0–5.4) | 215/629 (34.2 %, 30.6–38.0) | 15 |
| all | presidio_faker | 636 | 201 | 7 | 8/428 (1.9 %, 0.9–3.6) | 209/629 (33.2 %, 29.7–37.0) | 4 |
| all | llm_guard | 636 | 364 | 8 | 19/264 (7.2 %, 4.7–11.0) | 383/628 (61.0 %, 57.1–64.7) | 6 |
| all | gliner_pii | 636 | 73 | 7 | 6/556 (1.1 %, 0.5–2.3) | 79/629 (12.6 %, 10.2–15.4) | 5 |
| all | ss-conversation | 179 | 2 | 0 | 1/177 (0.6 %, 0.1–3.1) | 3/179 (1.7 %, 0.6–4.8) | 27 |
| oasst1 | ss | 192 | 1 | 3 | 1/188 (0.5 %, 0.1–2.9) | 2/189 (1.1 %, 0.3–3.8) | 33 |
| oasst1 | presidio_default | 192 | 59 | 3 | 5/130 (3.9 %, 1.7–8.7) | 64/189 (33.9 %, 27.5–40.9) | 4 |
| oasst1 | presidio_faker | 192 | 59 | 3 | 4/130 (3.1 %, 1.2–7.6) | 63/189 (33.3 %, 27.0–40.3) | 1 |
| oasst1 | llm_guard | 192 | 111 | 5 | 6/76 (7.9 %, 3.7–16.2) | 117/187 (62.6 %, 55.4–69.2) | 4 |
| oasst1 | gliner_pii | 192 | 14 | 2 | 1/176 (0.6 %, 0.1–3.1) | 15/190 (7.9 %, 4.8–12.6) | 0 |
| oasst1 | ss-conversation | 54 | 0 | 0 | 0/54 (0.0 %, 0.0–6.6) | 0/54 (0.0 %, 0.0–6.6) | 8 |
| sharegpt | ss | 221 | 4 | 0 | 1/217 (0.5 %, 0.1–2.6) | 5/221 (2.3 %, 1.0–5.2) | 30 |
| sharegpt | presidio_default | 221 | 67 | 0 | 2/154 (1.3 %, 0.4–4.6) | 69/221 (31.2 %, 25.5–37.6) | 4 |
| sharegpt | presidio_faker | 221 | 67 | 0 | 2/154 (1.3 %, 0.4–4.6) | 69/221 (31.2 %, 25.5–37.6) | 1 |
| sharegpt | llm_guard | 221 | 124 | 0 | 5/97 (5.1 %, 2.2–11.5) | 129/221 (58.4 %, 51.8–64.7) | 1 |
| sharegpt | gliner_pii | 221 | 26 | 0 | 1/195 (0.5 %, 0.1–2.9) | 27/221 (12.2 %, 8.5–17.2) | 3 |
| sharegpt | ss-conversation | 61 | 1 | 0 | 0/60 (0.0 %, 0.0–6.0) | 1/61 (1.6 %, 0.3–8.7) | 11 |
| wildchat | ss | 223 | 2 | 6 | 2/215 (0.9 %, 0.3–3.3) | 4/217 (1.8 %, 0.7–4.6) | 35 |
| wildchat | presidio_default | 223 | 75 | 4 | 7/144 (4.9 %, 2.4–9.7) | 82/219 (37.4 %, 31.3–44.0) | 7 |
| wildchat | presidio_faker | 223 | 75 | 4 | 2/144 (1.4 %, 0.4–4.9) | 77/219 (35.2 %, 29.1–41.7) | 2 |
| wildchat | llm_guard | 223 | 129 | 3 | 8/91 (8.8 %, 4.5–16.4) | 137/220 (62.3 %, 55.7–68.4) | 1 |
| wildchat | gliner_pii | 223 | 33 | 5 | 4/185 (2.2 %, 0.8–5.4) | 37/218 (17.0 %, 12.6–22.5) | 2 |
| wildchat | ss-conversation | 64 | 1 | 0 | 1/63 (1.6 %, 0.3–8.5) | 2/64 (3.1 %, 0.9–10.7) | 8 |

H8''' with exact recovery only (the pre-registered reading counts partials too):

- all: **does not hold** ((a) no exact, single-message: **no**, 4; (b) no exact, conversation: **no**, 1; (c) SS exact not above GLiNER-PII: yes, 4/620 (0.7 %, 0.2–1.7) vs 6/556 (1.1 %, 0.5–2.3); ss − gliner_pii -0.004 [-0.015, +0.006] over 178 messages)
- oasst1: **does not hold** ((a) no exact, single-message: **no**, 1; (b) no exact, conversation: yes, 0; (c) SS exact not above GLiNER-PII: yes, 1/188 (0.5 %, 0.1–2.9) vs 1/176 (0.6 %, 0.1–3.1); ss − gliner_pii -0.000 [-0.017, +0.016] over 59 messages)
- sharegpt: **does not hold** ((a) no exact, single-message: **no**, 1; (b) no exact, conversation: yes, 0; (c) SS exact not above GLiNER-PII: yes, 1/217 (0.5 %, 0.1–2.6) vs 1/195 (0.5 %, 0.1–2.9); ss − gliner_pii -0.001 [-0.015, +0.013] over 60 messages)
- wildchat: **does not hold** ((a) no exact, single-message: **no**, 2; (b) no exact, conversation: **no**, 1; (c) SS exact not above GLiNER-PII: yes, 2/215 (0.9 %, 0.3–3.3) vs 4/185 (2.2 %, 0.8–5.4); ss − gliner_pii -0.012 [-0.038, +0.012] over 59 messages)

**SS's 5 exact recoveries, opened (flags only).** 5 are URLs. In 5 of them the surrogate changed the host and kept the path as it was (5 with one host label changed), and in 5 the kept path carries an opaque identifier of 20 or more characters; the attacker's guess equals the original letter for letter in 5 of 5.

| condition | dataset | id | type | surrogate changed host | kept path | host labels changed | path has opaque id | guess = original, letter for letter |
|---|---|---|---|---|---|---|---|---|
| single | oasst1 | rd-oasst1-test3-0010 | URL | True | True | 1 | True | True |
| single | sharegpt | rd-sharegpt-test3-0042 | URL | True | True | 1 | True | True |
| single | wildchat | rd-wildchat-test3-0016 | URL | True | True | 1 | True | True |
| single | wildchat | rd-wildchat-test3-0021 | URL | True | True | 1 | True | True |
| conversation | wildchat | rd-wildchat-test3-0323 | URL | True | True | 1 | True | True |

**Partials set aside, SS pooled (98):**

| class | n | visible in the text the attacker read | kept by a surrogate rule |
|---|---|---|---|
| address_part | 14 | 14 | mimic.py: a town becomes a real town of the same country |
| birth_year | 8 | 8 | mimic.py: a date of birth moves by at most two years |
| email_domain | 38 | 32 | identity.py: free-mail domains are kept |
| org_word | 19 | 19 | mimic.py: an organisation keeps its legal and industry words |
| phone_area | 1 | 0 | mimic.py: a phone keeps its country code and trunk digit |
| url_host | 18 | 18 | mimic.py: a URL keeps its host and path layout |
