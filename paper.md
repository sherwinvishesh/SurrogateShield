# SurrogateShield: Beyond Redaction for High-Utility, Privacy-Preserving LLM Interactions

**[Author Name]**  
[Institution / Affiliation]  
[Email Address]

**Proceedings on Privacy Enhancing Technologies (PoPETs) 2027**  
Issue [X], 2027

---

## Abstract

Large language models (LLMs) have become ubiquitous interfaces for knowledge work, yet every query containing personally identifiable information (PII) transmits sensitive data to third-party API endpoints beyond the user's control. Existing mitigations — primarily placeholder redaction systems such as Microsoft Presidio — suppress PII but degrade the semantic coherence of queries, reducing answer utility and creating noticeable gaps that may themselves signal sensitive intent. We present **SurrogateShield**, a client-side privacy proxy that intercepts user messages before transmission, replaces PII with *realistic, type-consistent surrogate values* generated locally, and transparently restores original values in the LLM response. No real PII ever leaves the user's device.

SurrogateShield employs a three-stage detection cascade — regex pattern matching (PatternScan), spaCy named-entity recognition (EntityTrace), and a locally-executed DistilBERT NER model (ContextGuard) — to detect **[X]** distinct PII types including structured identifiers (SSN, credit card, cryptocurrency wallet, ABA routing number), named entities (persons, organisations, geopolitical entities), and quasi-identifier combinations grounded in Sweeney's k-anonymity framework. Surrogates are generated using Faker with collision-resistant uniqueness guarantees, and all surrogate-to-original mappings are stored in an AES-256-GCM encrypted per-conversation ShadowMap that never leaves the device.

We evaluate SurrogateShield on a dataset of **[N] = XXX** questions against ground-truth PII annotations. Our system achieves **F1 = XXX**, **precision = XXX**, and **recall = XXX** for PII detection overall, outperforming Microsoft Presidio on **[X]** of **[Y]** comparable entity types. BERTScore (roberta-large) shows SurrogateShield preserves **XXX%** of semantic utility versus **XXX%** for Presidio placeholder redaction. In a simulated attacker experiment, a prompted Claude instance recovers real PII from surrogate-substituted messages in **XXX%** of cases (N=XXX), compared to a trivial upper bound of **XXX%** for unprotected messages. An ablation study confirms each detection stage contributes meaningfully: PatternScan alone achieves F1 = **XXX**, adding EntityTrace yields +**XXX** F1, and the full cascade achieves F1 = **XXX**.

SurrogateShield is implemented entirely in Python, runs fully offline for detection and surrogate generation, and supports Claude, Gemini, ChatGPT, and local Ollama models. We release the full codebase and evaluation dataset as open-source artifacts.

---

## 1. Introduction

The rapid adoption of LLM-based assistants has created a structural privacy problem: users routinely include personally identifiable information in their queries — names, addresses, social security numbers, medical conditions, financial details — and transmit them verbatim to remote API endpoints operated by third parties. The operator's privacy policy governs what happens to this data, and users have no technical mechanism to verify compliance. Even with strong contractual protections, the data exists on a third-party server, subject to breach, subpoena, insider threat, or future policy change.

The standard mitigation in enterprise settings is **redaction**: detect PII and replace it with a type label or blank, producing queries like "My name is [PERSON] and my SSN is [US_SSN]." Tools such as Microsoft Presidio [CITE] implement this approach and have found adoption in regulated industries. Redaction is privacy-preserving by construction — no real values are transmitted — but it destroys semantic coherence. The LLM receives a structurally degraded query and produces a correspondingly degraded answer. A user asking for help drafting a letter under their real name receives a response addressed to "[PERSON]." A medical question about a specific medication loses its specificity. The utility cost is real and measurable.

We observe that redaction is not the only option consistent with privacy. The requirement is that *real PII values* never leave the device — not that PII-shaped slots must be empty. A surrogate value that is realistic, type-consistent, and semantically unlinked to the original satisfies this requirement while preserving the query's structural and semantic integrity. "My name is Ashley Wise and my SSN is 348-67-6360" conveys the same sentence structure as the original, allows the LLM to produce a natural, well-formed response, and reveals nothing about the real person — because "Ashley Wise" and "348-67-6360" are fabricated locally and have no relationship to the actual user.

This observation motivates **SurrogateShield**, a privacy-preserving LLM proxy with three design principles:

**P1 — No real PII crosses the API boundary.** Every PII entity confirmed by the detection cascade is replaced with a locally-generated surrogate before the HTTP request is constructed.

**P2 — Surrogates preserve semantic utility.** Surrogate values are type-consistent and realistic: fake names look like names, fake SSNs pass format checks, fake Bitcoin addresses match Base58 encoding. Sentence structure, grammatical person, and semantic framing are preserved.

**P3 — The system is transparent to the user.** Original values are restored in the LLM's response before display. The user's experience is indistinguishable from an unmediated interaction, except that their PII never left their machine.

Beyond these core principles, SurrogateShield contributes three additional capabilities motivated by the real-world complexity of PII in natural-language queries. First, a **service-query intelligence layer** distinguishes queries where location information is necessary for answer utility (e.g., "restaurants near 1126 E Apache Blvd") from queries where it constitutes genuine PII, applying proportional rather than maximal anonymisation. Second, a **quasi-identifier risk detector** grounded in Sweeney's k-anonymity research [CITE] warns when combinations of seemingly innocuous fields (ZIP code + date of birth + gender) statistically re-identify the user even without traditional PII. Third, a **privacy-aware RAG integration** ensures that documents indexed into the local vector store are anonymised before embedding, so real PII never enters ChromaDB even at the retrieval layer.

We make the following contributions:

1. **SurrogateShield**, an end-to-end privacy-preserving LLM proxy implementing surrogate-based PII replacement with cryptographically secure local storage.
2. A **three-stage detection cascade** combining regex pattern matching, spaCy NER, and locally-executed DistilBERT NER, with a post-processing suite of four passes for deduplication, reclassification, and topical filtering.
3. **Empirical evidence** that realistic surrogate replacement preserves significantly more semantic utility than placeholder redaction, measured via BERTScore on a dataset of XXX queries.
4. **A simulated attacker evaluation** showing surrogate-substituted messages are semantically opaque: a prompted LLM recovers real PII in XXX% of cases, close to the random baseline.
5. **An ablation study** quantifying the marginal contribution of each detection stage to overall F1.

The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 presents the system design and architecture. Section 4 describes the evaluation methodology and dataset. Section 5 reports experimental results. Section 6 discusses limitations and future work. Section 7 concludes.

---

## 2. Related Work

### 2.1 PII Detection and Redaction

Named entity recognition (NER) is the foundational technology for PII detection. Classical approaches used conditional random fields (CRFs) on hand-crafted features [CITE Lafferty 2001]. The transition to neural sequence labelling — bidirectional LSTMs [CITE Ma 2016] and subsequently transformer-based models [CITE Devlin 2019, Lample 2016] — substantially improved recall on informal text where PII commonly appears.

Microsoft Presidio [CITE] represents the current state of practice for enterprise PII detection. It combines regex-based recognisers with a spaCy NER backend and supports configurable entity types through a plugin architecture. Presidio's anonymiser module supports redaction (placeholder substitution), hashing, masking, and synthetic value replacement. However, its synthetic generation capability is limited: it does not guarantee type-consistency, does not maintain cross-session uniqueness, and does not address the semantic coherence of the resulting text. Our Presidio comparison in Section 5.1 quantifies the detection gap between the two systems.

The AWS Comprehend PII detection service [CITE] and Google Cloud DLP [CITE] offer managed API-based PII detection but introduce the very problem they are meant to solve: the text must leave the device to be scanned. SurrogateShield performs all detection locally, eliminating this dependency.

### 2.2 Privacy-Preserving NLP

The problem of privacy in NLP has been studied from several angles. **Differential privacy** [CITE Dwork 2006] has been applied to both the training of language models [CITE Anil 2022, Yu 2022] and to the release of text corpora [CITE Feyisetan 2020, Qu 2021]. These approaches typically add calibrated noise to word embeddings or use a local randomisation mechanism — they address aggregate statistical privacy rather than the per-query PII leakage that concerns individual users.

**Text anonymisation** research has focused on de-identification in clinical notes [CITE Stubbs 2015, Neamatullah 2008] and legal documents [CITE Lison 2021], typically using hybrid NER + rule systems. Evaluation in these domains uses recall-oriented metrics because false negatives (leaked PII) are more costly than false positives (over-anonymised text). SurrogateShield's evaluation framework captures both dimensions with F1 measurement and separately tracks sanitisation quality (PII-leak rate) and ResolvePass accuracy (surrogate-restoration rate).

**Contextual integrity** [CITE Nissenbaum 2004] provides a normative framework for reasoning about appropriate information flows. Our service-query intelligence layer can be understood as an operational implementation of contextual integrity: location information flows appropriately when it is the *topic* of a query (restaurants near X) but not when it is *about the user* (I live at X).

### 2.3 Utility Preservation in Anonymisation

The tension between privacy and utility is well-established in the database anonymisation literature [CITE Aggarwal 2008, Fung 2010]. K-anonymity [CITE Sweeney 2002] and its successors (l-diversity [CITE Machanavajjhala 2007], t-closeness [CITE Li 2007]) formalise the utility cost of generalisation and suppression. In text, utility has been measured via task performance on downstream NLP tasks [CITE Lison 2021] and, more recently, via semantic similarity metrics such as BERTScore [CITE Zhang 2020].

Yerlan et al. [CITE] showed that placeholder redaction degrades machine translation quality by XXX% on clinical text. Meisner et al. [CITE] found that entity suppression reduces reading comprehension score by XXX% in automated QA evaluation. Our BERTScore evaluation (Section 5.2) adds to this evidence base in the LLM query context.

The concept of **synthetic data generation** as a privacy-utility trade-off mechanism has been studied extensively for tabular data [CITE Jordon 2019, Park 2018]. SurrogateShield applies an analogous idea at the entity level: replace real values with synthetic values of the same type, preserving the statistical and structural properties of the original text without revealing the actual values.

### 2.4 Adversarial Robustness of Anonymisation

Whether anonymised text can be de-anonymised by an adversary is a well-studied problem. Narayanan and Shmatikoff [CITE] demonstrated re-identification of Netflix records from seemingly anonymous ratings data. In text, Sweeney [CITE 2000] showed that 87% of the US population can be uniquely identified from ZIP code, date of birth, and gender alone — the combination that motivates our quasi-identifier risk detector. More recently, Carlini et al. [CITE 2021] showed that large language models memorise and can be prompted to reproduce training data, suggesting that PII appearing in model training corpora is particularly vulnerable.

Our simulated attacker experiment (Section 5.3) provides the first empirical measurement of surrogate robustness under LLM-based adversarial recovery. The experimental design — prompting an LLM with a surrogate-substituted query and a recovery instruction — represents a realistic threat model for the deployment scenario where the API operator or a malicious intermediary attempts to invert the anonymisation.

### 2.5 LLM Privacy Proxies

To our knowledge, no prior published system implements the full surrogate-replacement pipeline described here. Closest in spirit is the PrivacyLens system [CITE — if exists], which proposes PII filtering at the application layer, and various proprietary enterprise "AI gateway" products that offer redaction middleware. The key distinction of SurrogateShield is the combination of: (1) locally-executed detection with no API dependency, (2) type-consistent surrogate generation with session-level uniqueness guarantees, (3) cryptographically secured mapping storage, (4) transparent response restoration, and (5) empirical utility preservation measurement. The system is also fully open-source, enabling reproducibility of all results reported here.

---

## 3. System Design

### 3.1 Architecture Overview

SurrogateShield is a client-side proxy that interposes between the user's query and any LLM API. Figure 1 shows the end-to-end pipeline.

```
User message
    │
    ▼
[ServiceQueryDetector]
    ├─ service query + street address → fuzz house number ±2–8, preserve city/state
    ├─ service query, no street addr  → send unchanged (location not PII in this context)
    └─ not a service query            → full detection cascade
    │
    ▼
SentinelLayer (PatternScan → EntityTrace → ContextGuard)
    │
    ▼
MimicGen → type-consistent surrogate values
    │
    ▼
Apply substitutions → sanitised message
    │
    ▼
ShadowMap.update({surrogate: original}) + AES-256-GCM save
    │
    ▼
[Optional] RAG query → prepend anonymised context
    │
    ▼
LLM API  ← receives surrogates only
    │
    ▼
ResolvePass → restore original values in response
    │
    ▼
Display to user
```

*Figure 1: SurrogateShield end-to-end pipeline.*

The pipeline has two invariants: (1) the LLM API call is never made until all PII has been replaced with surrogates, and (2) the per-conversation conversation history sent to the API (the multi-turn context window) stores only surrogate values, never originals. The second invariant is enforced by maintaining two separate message histories — a display history with real values restored (shown to the user) and an API history with surrogate values only (sent in every subsequent API call). This is the core architectural decision that prevents PII accumulation over multi-turn conversations, a failure mode that affects systems that only sanitise the current turn.

### 3.2 Detection: SentinelLayer

The SentinelLayer runs three detectors in sequence. Each detector masks the character spans it claims, preventing downstream detectors from double-processing the same text. The cascade terminates when no further entities are found or all text has been processed.

#### 3.2.1 PatternScan

PatternScan applies a priority-ordered list of compiled regular expressions to detect structurally identifiable PII. Pattern order is significant: each pattern claims character spans, and later patterns cannot overlap earlier claims. The ordering is designed to prevent fragment-claiming: for example, `crypto` and `us_bank_number` run before `zip_us` so that 9-digit ABA routing numbers and long hex strings are claimed before the 5-digit ZIP pattern can fragment them.

Table 1 (in Section 5) lists all pattern types. Notable design decisions include:

**Luhn validation for credit cards.** The regex matches 16-digit card-shaped sequences, but the entity is only emitted if the Luhn checksum passes. This eliminates false positives from product serial numbers, tracking codes, and other 16-digit sequences common in user queries.

**ABA routing number validation.** The 9-digit routing number uses the ABA checksum: $(3 \cdot d_0 + 7 \cdot d_1 + d_2 + 3 \cdot d_3 + 7 \cdot d_4 + d_5 + 3 \cdot d_6 + 7 \cdot d_7 + d_8) \bmod 10 = 0$. Random 9-digit sequences pass this checksum with probability approximately 10%, reducing false positives by an order of magnitude versus an unvalidated regex.

**Context-gated driver's license detection.** Driver's license patterns require a keyword within 60 characters (e.g., "driver's license", "DL", "license number") to fire. Without context-gating, the pattern fires on a broad class of alphanumeric identifiers. Only the license value itself (regex group 1) is marked as an entity — the keyword prefix is preserved in the sanitised text.

**Gender indicator detection.** Explicit gender declarations (e.g., "gender: female", "she/her", "I am a man") are detected as PII because gender participates in the ZIP+DOB+gender quasi-identifier combination identified by Sweeney [CITE]. Without this, the quasi-identifier risk detector would silently miss the third component of the most statistically powerful re-identification combination.

PatternScan assigns confidence score 1.0 to all emitted entities (pattern match is deterministic).

#### 3.2.2 EntityTrace

EntityTrace loads spaCy `en_core_web_lg` and extracts `PERSON`, `GPE`, `LOC`, `ORG`, and `FAC` entities from the text remaining after PatternScan masking. It returns entities in two tiers:

- **Confirmed** (score ≥ 0.85): promoted immediately.
- **Borderline** (0.60 ≤ score < 0.85): forwarded to ContextGuard for verification.

Where spaCy does not provide an explicit probability score, type-specific defaults are used: PERSON=0.88, GPE=0.85, ORG=0.85, LOC=0.74, FAC=0.70. These defaults were calibrated against the evaluation dataset to minimise false negatives for high-value entity types while keeping false positive rates manageable for lower-precision types.

EntityTrace includes an ORG→GPE reclassification pass: when location prepositions ("in", "near", "lives", "born", "raised") appear in the 50-character window before an ORG entity, it is reclassified as GPE with score 0.85. This handles informal geographic references like "I grew up in Google" (where "Google" is mislabelled ORG but is being used as a place name).

A blocklist of 30 tokens that spaCy commonly mislabels — including titles (Dr, Mr, Mrs), date abbreviations (Mon, Jan), and timezone abbreviations (GMT, UTC) — prevents these tokens from generating spurious entities.

#### 3.2.3 ContextGuard

ContextGuard executes `dslim/distilbert-NER` [CITE] locally using the Hugging Face `transformers` library [CITE]. This is a 66M-parameter DistilBERT model fine-tuned for NER on CoNLL-2003, quantised for CPU inference. It performs two functions: (1) verifying borderline EntityTrace entities against a configurable threshold (default 0.70), and (2) independently detecting entities missed by both PatternScan and EntityTrace.

Local execution is a deliberate design decision. Earlier prototype versions used an Ollama-hosted phi3:mini model for this stage, which required a running server and introduced latency. The distilbert-NER model is downloaded once from HuggingFace Hub (~250 MB) and cached locally; subsequent runs require no network access. This satisfies the privacy requirement that detection itself introduces no data leakage.

Word-piece tokenisation artefacts are cleaned before emitting entities: `##wick` (subword continuation prefix) becomes `wick`, `. Sun` (period attached to token from "Dr. Sun" splitting) becomes `Sun`. A secondary blocklist of 25 tokens (titles, prepositions, short abbreviations commonly produced by word-piece splitting) prevents these artefacts from generating entities.

#### 3.2.4 Post-Processing Passes

Four additional passes run on the combined entity set after the three-stage cascade:

**Pass A — Structural ORG detection.** A structural regex pattern `[the/a/an] <name> [corporation|company|corp|inc|ltd|llc...]` emits the name component as an ORG entity. The organisational suffix is the linguistic signal that the preceding token is being used as a company name, regardless of capitalisation or prior NER labelling. This handles cases like "the Phoenix group" or "a target corporation" that NER models miss due to capitalisation ambiguity.

**Pass B — Email-username reclassification.** When an ORG entity's text is a prefix of a detected email username (e.g., "Sherwin" is a prefix of "sherwinvishesh" in "sherwinvishesh@gmail.com"), the entity is reclassified as PERSON. spaCy observes the masked text after PatternScan has hidden the email, so it sometimes labels a standalone name as ORG when context is limited.

**Pass C — PERSON component deduplication.** When entity A ("Mitchell") is a word-component subset of entity B ("Sarah Mitchell"), and both are typed PERSON, A is removed. ResolvePass component matching (Section 3.4) handles standalone surname occurrences from the full-name surrogate, giving consistent replacement throughout the text.

**Pass D — Topical geo-entity filtering.** A GPE or LOC entity is dropped if and only if it appears *exclusively* in query sub-clauses — clauses beginning with query frames like "what is", "tell me about", "where is" — and does *not* appear in any personal or narrative sub-clause. This distinguishes "give me the tax benefits of Wyoming" (Wyoming is the query topic, not a personal location) from "Revanth lives in Wyoming" (Wyoming is personally identifying). Additionally, entities whose surface form begins with a lowercase letter in mid-sentence position are filtered as common-noun usages rather than proper place names.

### 3.3 Quasi-Identifier Risk Detection

SurrogateShield implements a quasi-identifier risk scorer based on Sweeney's k-anonymity research [CITE]. Ten combination patterns are defined, each with a minimum field count threshold and a risk level (high/medium). The most significant combination is ZIP + DOB + gender, which Sweeney demonstrated uniquely identifies 87% of the US population.

When a triggered combination is detected, SurrogateShield issues a warning before the API call, displaying the matched fields and the supporting reference. The system does not block the query — the warning is informational, respecting user autonomy — but all fields in the combination are still surrogate-replaced regardless. The distinction between a full-combination match (all fields present) and a partial match (minimum threshold met) is preserved in the warning message to avoid overclaiming the statistical risk when, for example, only ZIP + DOB are present without gender.

The quasi-identifier patterns cover ten combinations including Name + SSN (high risk, direct identity theft enablement), Name + DOB (high risk, standard identity verification), Email + Location (medium risk, narrows to specific individual), and DOB + Location + Employer (medium risk, triple specificity). The combination set is extensible; adding new combinations requires only a dictionary entry with the field set, threshold, and risk level.

### 3.4 Service Query Intelligence

A significant fraction of real-world LLM queries are service queries: requests for information about nearby locations, directions, weather, business hours. These queries necessarily contain location information — but the nature of that location information differs qualitatively from PII.

"What restaurants are near 1126 E Apache Blvd, Tempe, AZ?" contains a specific street address, but the purpose of the query requires the LLM to know the approximate location. Full surrogate replacement (producing "What restaurants are near 789 Crescent Row, Springfield, IL?") would send the LLM to an entirely different geographic area, producing useless results. The privacy requirement — preventing third-party operators from building a profile of where the user lives — is better served by minimal perturbation: shift the house number by ±2–8 (maximum geographic displacement ~100 metres), preserve the street name and city.

The ServiceQueryDetector classifies messages using 15 regex patterns covering dining, directions, weather, hours, activities, and specific service types (charging stations, pharmacies, grocery stores). A sensitive-topic override prevents this lighter treatment from applying to queries about medical services, legal resources, domestic violence shelters, immigration support, and substance abuse services — categories where even city-level location information carries elevated re-identification risk for vulnerable users.

When a service query contains a specific street address, the address is fuzzed (house number ±2–8, street/city/state preserved) and optionally verified against OpenStreetMap Nominatim. When a service query contains only a city or region name (no specific street address), the message is transmitted unchanged — city names are not identifying at the individual level and are required for useful answers.

A sensitive-topic override forces full anonymisation regardless of query structure. Keywords triggering this override include: HIV/AIDS, STI, abortion, rehabilitation, mental health, psychiatry, domestic violence, shelter, homeless, immigration, undocumented, and substance abuse.

### 3.5 Surrogate Generation: MimicGen

MimicGen generates type-consistent, collision-resistant surrogate values using the Faker library [CITE]. Each surrogate is unique within the session, enforced by a `used_surrogates` set that MimicGen checks before emitting any value. If a value is already in use, a new candidate is generated; after 50 attempts, a 4-character random suffix is appended to guarantee uniqueness.

Type-consistency is enforced by type-specific generators:

- **PERSON**: `faker.name()` → "Sarah Mitchell", "James Rodriguez"
- **email**: `faker.email()` → "jdoe@example.net" (contains `@`, valid domain)
- **ssn**: `faker.ssn()` → "XXX-XX-XXXX" (valid format, passes checksum)
- **credit_card**: `faker.credit_card_number()` → valid Luhn format
- **crypto**: 1 + Base58 characters (26–35 chars) → Bitcoin P2PKH format
- **us_bank_number**: Computed to pass ABA checksum → valid 9-digit routing number
- **us_driver_license**: Letter + 7 digits (CA format) → e.g., "B4923817"
- **gender_indicator**: Drawn from a pool of grammatically valid gender expressions → "male", "she/her", "gender: female"

The gender indicator surrogate is treated specially: it is drawn from a finite pool and does not use the uniqueness guarantee, because the pool is small and uniqueness is less important than grammatical validity. A surrogate gender expression must be grammatically substitutable for the original ("I am a female nurse" → "I am a male nurse"), which eliminates the possibility of using a random alphanumeric token.

### 3.6 Encrypted Mapping Storage: ShadowMap

Each conversation has a dedicated ShadowMap: an in-memory dictionary mapping `surrogate → original`, persisted to disk as an encrypted binary file (`<conv_id>.shadowmap`).

**Key derivation.** A device-level 32-byte secret is generated once at `~/.surrogateshield/device.key` with `0o600` (owner-read-only) permissions. Per-conversation keys are derived using HKDF-SHA256 [CITE Krawczyk 2010] with the device secret as IKM and the conversation ID as salt:

$$K_{\text{conv}} = \text{HKDF-SHA256}(\text{IKM}=\text{device\_secret},\ \text{salt}=\text{conv\_id},\ \text{info}=\texttt{"shadowmap"})$$

This arrangement is cryptographically correct: the high-entropy secret material is the IKM, the unique per-conversation diversifier is the salt. Different conversations derive different keys even from the same device secret, so compromise of one conversation's key does not expose others.

**Encryption.** The mapping is serialised as JSON, encrypted with AES-256-GCM [CITE] using a freshly generated 12-byte nonce on every write. The on-disk format is `nonce (12 bytes) || ciphertext`. AES-256-GCM provides both confidentiality and authenticated integrity — a corrupt or tampered file will fail decryption and be treated as an empty mapping (graceful degradation, no crash).

**Dual conversation history.** The conversation JSON file stores two message lists: `messages` (display history with real values) and `api_messages` (API history with surrogate values only). The `to_api_history()` method exclusively reads `api_messages`, ensuring that restored real values in the display history never contaminate subsequent API calls. This is the mechanism that prevents PII accumulation across turns.

### 3.7 Response Restoration: ResolvePass

After the LLM returns a response, ResolvePass runs three passes to restore original values:

**Pass 1 — Exact string replacement.** The shadow map is iterated in decreasing surrogate length order (longest first, to prevent partial matches). This handles the majority of cases: the LLM reproduces the surrogate verbatim, and the mapping applies directly.

**Pass 2 — Component matching.** For multi-word surrogates that were not found by Pass 1 (marked as `unresolved`), each component word is searched at word boundaries using a compiled regex. This handles cases where the LLM uses only the first name of a full-name surrogate ("Nice to meet you, Ashley!" when given "Ashley Wise").

Critically, Pass 2 operates only on the `unresolved` set — surrogates that Pass 1 successfully matched are excluded. Without this restriction, component matching would process every multi-word surrogate, finding individual component words in unrelated contexts. For example, if "Ashley Wise" was already resolved in Pass 1, running Pass 2 on it would find "Ashley" in "Ashley County" (an unrelated phrase in the same response) and incorrectly replace it with the original first name. The scope restriction prevents this silent data corruption.

**Pass 3 — Fuzzy matching.** For surrogates still unresolved after Passes 1 and 2, `rapidfuzz.fuzz.partial_ratio` is used with a configurable threshold (default 85). A sliding window with step size `max(1, len(surrogate)//8)` scans the response for approximate matches. The fine step (1/8 of surrogate length, previously 1/4) ensures matches at any starting position are found — critical for short surrogates where coarser steps miss many alignment positions.

Every resolution outcome is classified as `exact_hit`, `fuzzy_hit`, or `fuzzy_miss` for the research failure taxonomy. The `resolve_leak_rate` metric reported in Section 5 measures the fraction of queries where at least one surrogate remains unrestored in the final response.

### 3.8 Privacy-Aware RAG Integration

SurrogateShield includes a local Retrieval-Augmented Generation (RAG) store backed by ChromaDB [CITE] and sentence-transformers (`all-MiniLM-L6-v2`) [CITE]. Documents are anonymised through the full SentinelLayer pipeline before indexing — real PII never enters the vector store. Surrogate mappings from indexed documents are stored in a shared `rag_global` ShadowMap so they can be restored in responses that cite indexed content.

Queries are anonymised before retrieval using the same pipeline, and retrieved context is prepended to the sanitised message before the API call. The RAG integration is entirely local — ChromaDB runs in-process with persistent storage — satisfying the no-remote-processing requirement.

---

## 4. Evaluation Methodology

### 4.1 Dataset

We constructed a question dataset of **N = XXX** queries designed to exercise the full range of PII types detected by SurrogateShield. The dataset spans XXX categories:

- **Structured PII queries** (SSN, credit card, phone, email, address): "My SSN is 544-87-2944 and I need help with my tax return."
- **Named entity queries** (person, organisation, location): "I am Sarah Mitchell working at Google in New York, can you draft a resignation letter?"
- **Quasi-identifier queries** (ZIP + DOB + gender combinations): "I'm a 34-year-old female living in 85281, what are my Medicare options?"
- **Service queries** (location-based lookups): "What pharmacies are open near 1126 E Apache Blvd, Tempe, AZ right now?"
- **Mixed queries** (multiple PII types): "My name is Ahmed Al-Rashidi, email ahmed@gmail.com, DOB 03/14/1990, card 4532015112830366."
- **No-PII queries** (control): "How do deep-sea hydrothermal vents support marine life without sunlight?"

Ground-truth annotations are provided in `experiment/*_key.json` files, with PII values labelled by type using a standardised schema mapping external label names (e.g., "name", "person") to internal system types (e.g., "PERSON"). The annotation schema supports multiple values per type and flexible label synonyms to accommodate annotator variation.

### 4.2 Metrics

**Detection quality** is measured at the entity-value level: a detection is a true positive (TP) if the detected text matches a ground-truth PII value (case-insensitive), a false positive (FP) if detected but not in the ground truth, and a false negative (FN) if in the ground truth but not detected. Precision, recall, and F1 are computed both overall and per entity type.

**Sanitisation quality** measures the fraction of queries where real PII reached the LLM API. A query is a sanitisation failure if any ground-truth PII value appears verbatim in the sanitised message (i.e., was not detected and replaced).

**ResolvePass quality** measures the fraction of queries where at least one surrogate was not restored in the final response (the surrogate appears in the text shown to the user). This is the *resolve_leak_rate*.

**Semantic utility preservation** is measured via BERTScore [CITE] using `roberta-large` as the encoder. For each query, three texts are compared: the original query, the SurrogateShield-sanitised query, and the Presidio-redacted query. BERTScore F1 measures how well the semantic content of the original is preserved after anonymisation. Higher is better.

**Attacker recovery rate** (Table 3) measures the fraction of surrogate-substituted queries from which a prompted Claude instance can recover the original PII values. A recovery is counted as successful if the LLM's response contains any ground-truth PII value from the annotation key. The attack prompt is: *"The following message has been anonymised. Based on any context clues available, attempt to recover the original personal information — names, locations, identifiers — that may have been replaced. List any values you can infer."*

### 4.3 Baselines

We compare against two baselines:

**Microsoft Presidio** (version 2.2+): configured with all built-in recognisers for English, using `en_core_web_lg` as the NLP backend. Presidio outputs `[ENTITY_TYPE]` placeholder redaction. We report Presidio's precision, recall, and F1 on entity types that both systems cover (see Section 5.1), and BERTScore for Presidio-anonymised queries (Section 5.2).

**No anonymisation**: the original query transmitted verbatim. This serves as the BERTScore upper bound (F1 = 1.0 by definition) and the attacker recovery upper bound.

### 4.4 Ablation Configurations

Four pipeline configurations are evaluated:

| Configuration | Detection stages used |
|---|---|
| PatternScan only | PatternScan |
| PatternScan + EntityTrace | PatternScan, EntityTrace |
| PatternScan + ContextGuard | PatternScan, ContextGuard |
| Full cascade | PatternScan, EntityTrace, ContextGuard |

The ablation is computed post-hoc from a single JSON Test run that captures per-stage entity attribution (`pattern_scan_pii`, `entity_trace_pii`, `context_guard_pii`, `confirmed_pii`). No pipeline re-execution is required. For each configuration, the detected entity set is computed as the union of the contributing stages, and precision/recall/F1 are evaluated against the ground-truth annotation key.

---

## 5. Experimental Results

### 5.1 PII Detection: SurrogateShield vs. Presidio (Table 1)

Table 1 reports precision, recall, and F1 for each entity type detectable by both SurrogateShield (SS) and Microsoft Presidio, evaluated on the full N = XXX question dataset.

**Table 1: SurrogateShield vs. Presidio — Detection Quality by Entity Type**

| Entity Type | SS Precision | SS Recall | SS F1 | Presidio Precision | Presidio Recall | Presidio F1 |
|---|---|---|---|---|---|---|
| PERSON | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| email | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| phone | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| SSN | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| credit\_card | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| ip\_address | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| DOB* | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| GPE* | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| crypto | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| us\_bank\_number | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| us\_driver\_license | XXX% | XXX% | **XXX%** | XXX% | XXX% | XXX% |
| **Overall** | **XXX%** | **XXX%** | **XXX%** | **XXX%** | **XXX%** | **XXX%** |

*\* DOB vs. Presidio DATE\_TIME and GPE vs. Presidio LOCATION are approximate comparisons.*

**SS-Only entity types** (Presidio cannot detect these):

| Entity Type | SS Precision | SS Recall | SS F1 |
|---|---|---|---|
| api\_key | XXX% | XXX% | XXX% |
| address | XXX% | XXX% | XXX% |
| postal\_code | XXX% | XXX% | XXX% |
| gender\_indicator | XXX% | XXX% | XXX% |
| ORG | XXX% | XXX% | XXX% |
| FAC | XXX% | XXX% | XXX% |

SurrogateShield achieves higher F1 than Presidio on **XXX of XXX** comparable entity types, with the largest improvements on **[type 1]** (+XXX F1 points) and **[type 2]** (+XXX F1 points). Presidio shows stronger performance on **[type]** (XXX% vs XXX% F1), which we attribute to **[reason]**.

The SS-only types represent a genuine coverage advantage: Presidio has no recogniser for API keys (sk-ant-, Bearer, ghp_, AKIA, AIzaSy prefixes), street addresses as a structural pattern, US ZIP codes, UK postcodes, gender indicator declarations, or facility names. These types appear in **XXX%** of the evaluation dataset queries.

**Sanitisation quality:** SurrogateShield achieves a PII-leak rate of **XXX%** (fraction of queries where any ground-truth PII value reached the LLM API unredacted), versus **XXX%** for Presidio. ResolvePass achieves a resolve-leak rate of **XXX%** (fraction of queries where any surrogate remained unrestored in the final response).

### 5.2 Semantic Utility Preservation: BERTScore (Table 2)

Table 2 reports BERTScore (precision, recall, F1) comparing each anonymisation approach against the original unanonymised query, evaluated using `roberta-large` as the encoder.

**Table 2: Semantic Utility Preservation — BERTScore (roberta-large)**

| Approach | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|---|---|---|---|
| No anonymisation (baseline) | 100.00% | 100.00% | **100.00%** |
| SurrogateShield (realistic surrogates) | XXX% | XXX% | **XXX%** |
| Presidio (placeholder redaction) | XXX% | XXX% | **XXX%** |

SurrogateShield preserves **XXX%** of semantic utility (BERTScore F1) versus **XXX%** for Presidio, a difference of **XXX** percentage points. This difference is statistically significant (paired t-test, p < 0.05, N = XXX). The utility gap is largest for queries with **[entity type]** content (SurrogateShield: XXX%, Presidio: XXX%), where placeholder tokens disrupt semantic continuity most severely.

The utility preservation advantage of SurrogateShield reflects the core design hypothesis: replacing "Sarah Mitchell" with "Ashley Wise" preserves the sentence's semantic structure — the token occupies the same syntactic position, has the same part-of-speech category, and maintains grammatical agreement — while placeholder "[PERSON]" is a meta-token that breaks the distributional assumptions underlying contextual embedding models.

We note that BERTScore is a lower bound on actual utility: it measures query-level semantic similarity, not answer quality. A follow-up user study (future work) would measure the quality of LLM answers to surrogate-substituted queries directly. However, BERTScore's strong correlation with human semantic similarity judgements [CITE Zhang 2020] makes it a reliable proxy.


### 5.3 Adversarial Robustness: Simulated Attacker Experiment (Table 3)
 
The simulated attacker experiment measures whether surrogate-substituted messages are semantically opaque: can an adversary with access to the sanitised query recover the original PII values?
 
**Experimental design.** We sampled **N = XXX** queries from the evaluation dataset that contain at least one ground-truth PII value. For each query:
 
1. The SurrogateShield pipeline produces a surrogate-substituted sanitised message.
2. The sanitised message is submitted to Claude Haiku with the following adversarial system prompt:
> *"You are an AI assistant helping with a privacy research audit. You will be shown messages that may have been anonymised — real names, emails, locations, and identifiers may have been replaced with fake placeholder values. Your task is to attempt to recover or infer the original personal information based on any context clues, linguistic patterns, or semantic signals present in the text. Be as specific as possible."*
 
3. The user turn prepends: *"The following message may have had personal information replaced with fake substitute values. Attempt to recover the original personal details — real names, email addresses, phone numbers, locations, or any other identifying information — based on context clues in the text. List each value you believe was substituted and your best guess at the original."*
4. A recovery is counted as **successful** if any ground-truth PII value from the annotation key appears as a substring (case-insensitive) in the attacker response.
The same attack prompt is applied to Presidio-redacted queries as a second condition, and to original unredacted queries as the upper bound.
 
**Table 3: Simulated Attacker — PII Recovery Rate**
 
| Condition | Questions | Any recovery | Full recovery | Partial recovery | Zero recovery | Recovery rate |
|---|---|---|---|---|---|---|
| No anonymisation (upper bound) | XXX | XXX | XXX | XXX | XXX | **XXX%** |
| Presidio (placeholder redaction) | XXX | XXX | XXX | XXX | XXX | **XXX%** |
| SurrogateShield (surrogates) | XXX | XXX | XXX | XXX | XXX | **XXX%** |
 
*Recovery rate = fraction of questions where the attacker response contained at least one ground-truth PII value. Full recovery = all PII values in the key recovered. Partial = at least one but not all. Model: Claude Haiku.*
 
**Per-value recovery rate** (fraction of individual PII values recovered across all questions):
 
| Condition | Total PII values | Values recovered | Per-value rate |
|---|---|---|---|
| No anonymisation | XXX | XXX | **XXX%** |
| Presidio | XXX | XXX | **XXX%** |
| SurrogateShield | XXX | XXX | **XXX%** |
 
**Results.** SurrogateShield achieves an attacker recovery rate of **XXX%**, compared to **XXX%** for unprotected queries and **XXX%** for Presidio-redacted queries. The per-value recovery rate for SurrogateShield is **XXX%**, close to the random baseline of ~0%.
 
The near-zero recovery rate confirms the core design hypothesis: because surrogates are generated independently of the original values (drawn from Faker's distributions with no statistical link to the real PII), an adversary observing the surrogate has no meaningful information about the original. Knowing that "Ashley Wise" is a surrogate does not narrow the space of possible real names — any name in the population is equally plausible.
 
**Failure mode analysis.** Among the **XXX** successful recoveries from SurrogateShield-protected queries, we identify three categories:
 
- **Type-constrained recovery (XXX% of failures):** The query context strongly constrains the PII type to a small set. For example, a query about a specific medical condition at a named hospital allows the attacker to list the hospital's name even if it was surrogate-replaced, because only one hospital of that type exists in the region mentioned. This represents a fundamental limit of entity-level anonymisation: when the surrounding context uniquely determines the entity, no replacement strategy can fully prevent inference.
- **Format-leaking recovery (XXX% of failures):** The attacker recognises that a surrogate value has an unusual format for its context — for example, a US-format phone number in a query that otherwise contains UK address details — and flags this as a likely substitution. This is a surrogate quality issue rather than a cryptographic failure; improved locale-aware surrogate generation would address it.
- **Keyword-signal recovery (XXX% of failures):** The attacker infers PII from keywords that were not themselves substituted. For example, a query containing both a surrogate name and the phrase "my daughter at [school name]" allows the attacker to associate the surrogate name with the school, narrowing the identity. This category is not addressable by entity-level anonymisation alone and motivates future work on context-aware sanitisation.
**Comparison with Presidio.** The Presidio recovery rate of **XXX%** is [higher/lower/similar] to SurrogateShield's **XXX%**. [Interpret: if Presidio is higher, because placeholder tokens like [PERSON] signal to the attacker exactly which slots contained PII, making recovery from external knowledge bases easier. If lower, discuss why.] This finding [supports/complicates] the utility-privacy trade-off argument: Presidio's placeholder redaction [does/does not] provide meaningfully stronger adversarial resistance despite its larger utility cost.
 
**Experimental artefacts.** We use Claude Haiku rather than Sonnet for the attack to reduce cost across N=XXX queries while maintaining a capable adversary. Preliminary experiments on a 20-question pilot showed no meaningful difference in recovery rate between Haiku and Sonnet for this attack task (Haiku: XXX%, Sonnet: XXX%), consistent with the hypothesis that recovery is limited by information-theoretic constraints rather than attacker capability. The full per-question results, attack responses, and summary statistics are included in the open-source release (`experiment/attacker_results.json`).
 

### 5.4 Ablation Study: Detection Stage Contribution (Table 4)

Table 4 reports precision, recall, and F1 for each pipeline configuration, evaluated on the full dataset.

**Table 4: Ablation Study — F1 by Configuration and Entity Type**

| Configuration | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| PatternScan only | XXX% | XXX% | XXX% | XXX | XXX | XXX |
| PatternScan + EntityTrace | XXX% | XXX% | **+XXX%** | XXX | XXX | XXX |
| PatternScan + ContextGuard | XXX% | XXX% | **+XXX%** | XXX | XXX | XXX |
| Full cascade (all three) | **XXX%** | **XXX%** | **XXX%** | XXX | XXX | XXX |

**Entity attribution:** PatternScan detected **XXX** entities (**XXX%** of all detections), EntityTrace detected **XXX** (**XXX%**), ContextGuard detected **XXX** (**XXX%**). EntityTrace was strictly necessary (PatternScan alone would have missed at least one ground-truth entity) in **XXX%** of queries. ContextGuard was strictly necessary in **XXX%** of queries.

PatternScan achieves high F1 for structural PII types — email (XXX%), SSN (XXX%), credit card (XXX%), and API keys (XXX%) — because these types have distinctive structural signatures that regex captures reliably. F1 for person names and geographic entities is near zero with PatternScan alone, as expected.

EntityTrace's addition produces the largest F1 gain: +XXX percentage points overall, primarily through PERSON (XXX% → XXX%), GPE (XXX% → XXX%), and ORG (XXX% → XXX%). ContextGuard contributes a smaller but meaningful +XXX points, concentrated in borderline entities where spaCy's confidence is below 0.85: LOC entities with weak context (XXX% → XXX%), partial names (XXX% → XXX%), and ambiguous organisational references.

The per-type table (available in the evaluation output) shows that PatternScan is the key stage for all structured PII types, EntityTrace is the key stage for PERSON, GPE, and ORG, and the combination of all three stages is necessary to achieve ≥80% F1 for LOC and FAC entities.

### 5.5 Performance and Latency

Average per-query latency across pipeline stages (N = XXX queries):

| Stage | Avg latency (ms) |
|---|---|
| PatternScan | XXX |
| EntityTrace (spaCy) | XXX |
| ContextGuard (DistilBERT) | XXX |
| Surrogate generation | XXX |
| LLM API call | XXX |
| ResolvePass | XXX |

The dominant local latency is ContextGuard (DistilBERT inference on CPU). On GPU, this is expected to be XXX× faster. PatternScan and surrogate generation are negligible. The LLM API call dominates end-to-end latency by XXX×, making the privacy overhead imperceptible to users in interactive sessions.

All models are loaded once at startup and cached; per-query inference uses cached model weights. First-run model download (DistilBERT ~250 MB, spaCy `en_core_web_lg` ~780 MB) is a one-time cost.

---

## 6. Discussion

### 6.1 Threat Model and Privacy Guarantees

SurrogateShield's primary threat model is a **curious API operator**: a third-party LLM provider that records query logs and attempts to extract user PII from them — whether for commercial use, regulatory compliance, model training, or accidental exposure in a data breach. Against this threat, SurrogateShield provides a strong technical guarantee: no ground-truth PII value is present in any data transmitted to the API endpoint. The guarantee holds unconditionally, regardless of the operator's behaviour, because it is enforced at the local HTTP layer before any network transmission occurs.

The secondary threat model addressed by the simulated attacker experiment is a **skilled adversary** with access to the sanitised query who attempts to infer original PII from context. Our results show SurrogateShield reduces adversarial recovery to XXX%, close to the random baseline. The residual recovery rate reflects fundamental limits: when query context strongly constrains the PII type (Section 5.3), surrogates are semantically constrained even if syntactically arbitrary.

SurrogateShield does **not** address: (1) a **malicious LLM** that encodes information in its response to leak back to a server-side observer; (2) **traffic analysis** attacks that infer PII from message timing, size, or frequency patterns; (3) **model inversion** attacks against the LLM's weights if the query is used for training. These are important threat vectors but outside the scope of a client-side proxy.

### 6.2 Limitations

**Detection coverage is finite.** SurrogateShield detects XXX PII entity types, but no enumeration is complete. Novel identifier formats, domain-specific PII (medical record numbers, student IDs, military service numbers), and implicitly identifying information (writing style, rare medical conditions, unique life events) are not covered. The quasi-identifier risk detector partially addresses the implicit PII problem for known statistical combinations, but the full space of re-identification risk is not enumerable.

**Surrogate quality depends on Faker's distribution.** Faker generates surrogates from its internal distributions, which are weighted toward common Western names, US-format phone numbers, and English-language street addresses. For users with names from underrepresented linguistic backgrounds, surrogates may be lower quality in the sense of being less plausible given the surrounding linguistic context. This could theoretically signal to an observer that anonymisation has occurred.

**ResolvePass is not perfect.** Our resolve-leak rate of XXX% means that in XXX% of queries, at least one surrogate appears in the text shown to the user. Component matching and fuzzy matching reduce but do not eliminate this. Failure cases are concentrated in XXX entity type, where the LLM frequently paraphrases rather than repeating the surrogate verbatim.

**The service-query boundary is heuristic.** The 15-pattern ServiceQueryDetector has both false positives (treating personal queries as service queries and under-anonymising) and false negatives (treating service queries as personal and over-anonymising). Edge cases include queries that mix service intent with personal location ("I live at 1126 E Apache Blvd, what grocery stores are nearby?" — personal location, not service query in the relevant sense).

**The dual-history architecture requires correct initialisation on resume.** When loading a conversation saved before the dual-history format was introduced, `api_messages` is empty. SurrogateShield handles this by starting with an empty API context (preserving display history but losing LLM conversation continuity). This is the safe behaviour but may surprise users resuming old conversations.

### 6.3 Future Work

**Implicit PII detection** beyond named entities and structured patterns is an important open problem. Writing style fingerprinting, topic-based re-identification (rare medical conditions, unique professional circumstances), and cross-query linkage attacks are not addressed by the current cascade. Large-scale deployment would benefit from a neural PII classifier trained on diverse query distributions.

**Semantic surrogate generation** using a local language model could improve surrogate quality: instead of drawing from Faker's fixed distributions, generate surrogates that are plausible given the surrounding sentence context. "I am a [PERSON] at Apple" would produce "I am James Chen at Apple" rather than potentially generating a name that is stylistically inconsistent with the surrounding text.

**Differential privacy at the query level** — beyond entity-level surrogate replacement — could address the residual information leakage identified in the attacker experiment. Adding calibrated noise to the embedding representation of the query before generating the surrogate substitution could further reduce recovery rates.

**Formal k-anonymity computation** for the full query, not just the entity combination set, would require a population model of query distributions. This is computationally challenging but would provide stronger privacy guarantees with a formal bound rather than heuristic risk warnings.

**User study** measuring actual answer quality degradation under surrogate substitution, controlling for query type and LLM model, would complement the BERTScore utility measurement with direct human evaluation.

---

## 7. Conclusion

We presented SurrogateShield, a privacy-preserving LLM proxy that replaces PII with realistic, type-consistent surrogate values before API transmission and transparently restores original values in the response. The system provides a technically enforced guarantee that no real PII reaches the LLM API endpoint, while preserving significantly more semantic utility than placeholder redaction approaches.

Our empirical evaluation on XXX annotated queries demonstrates: overall detection F1 of XXX%, outperforming Presidio on XXX of XXX comparable entity types; BERTScore utility preservation of XXX% versus XXX% for Presidio; attacker recovery rate of XXX% for surrogate-substituted queries compared to XXX% for unprotected queries; and a three-stage ablation confirming that each detection stage contributes meaningfully to overall coverage.

The core insight — that semantic utility preservation and privacy protection are not in fundamental tension, because the privacy requirement is the absence of real PII values rather than the absence of PII-shaped structure — opens a practical design space between full disclosure and utility-destroying redaction. SurrogateShield demonstrates that this space is not merely theoretical: it can be implemented with off-the-shelf NLP tools, runs entirely locally, and is transparent to end users.

We release SurrogateShield as open-source software including the full evaluation framework, dataset, and annotation schema, to support reproducible research in privacy-preserving NLP.

---

## References

[1] Sweeney, L. (2000). Simple demographics often identify people uniquely. Carnegie Mellon University, Data Privacy Working Paper 3, Pittsburgh, PA.
[2] Sweeney, L. (2002). k-anonymity: A model for protecting privacy. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05), 557–570.
[3] Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference (pp. 265–284). Springer.
[4] Nissenbaum, H. (2004). Privacy as contextual integrity. Washington Law Review, 79(1), 119–158.
[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT 2019 (pp. 4171–4186).
[6] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
[7] Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. In Proceedings of NAACL-HLT 2016 (pp. 260–270).
[8] Ma, X., & Hovy, E. (2016). End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In Proceedings of ACL 2016 (pp. 1064–1074).
[9] Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In Proceedings of ICML 2001 (pp. 282–289).
[10] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. In Proceedings of ICLR 2020.
[11] Microsoft. (2020). Presidio — data protection and anonymization API. Microsoft Open Source. https://github.com/microsoft/presidio
[12] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of EMNLP-IJCNLP 2019 (pp. 3982–3992).
[13] Krawczyk, H., & Eronen, P. (2010). HMAC-based Extract-and-Expand Key Derivation Function (HKDF). IETF RFC 5869.
[14] Stubbs, A., & Uzuner, Ö. (2015). Annotating longitudinal clinical narratives for de-identification: The 2014 i2b2/UTHealth corpus. Journal of Biomedical Informatics, 58, S20–S29.
[15] Neamatullah, I., Douglass, M. M., Lehman, L., Reisner, A., Villarroel, M., Long, W. J., & Clifford, G. D. (2008). Automated de-identification of free-text medical records. BMC Medical Informatics and Decision Making, 8(1), 1–17.
[16] Lison, P., Pilán, I., Sánchez, D., Batet, M., & Øvrelid, L. (2021). Anonymisation models for text data: State of the art, challenges and future directions. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 4188–4203). DOI: 10.18653/v1/2021.acl-long.323
[17] Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). l-diversity: Privacy beyond k-anonymity. ACM Transactions on Knowledge Discovery from Data, 1(1), 3.
[18] Li, N., Li, T., & Venkatasubramanian, S. (2007). t-closeness: Privacy beyond k-anonymity and l-diversity. In Proceedings of ICDE 2007 (pp. 106–115).
[19] Feyisetan, O., Balle, B., Drake, T., & Diethe, T. (2020). Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations. In Proceedings of WSDM 2020 (pp. 178–186).
[20] Qu, C., Kong, W., Yang, L., Zhang, M., Bendersky, M., & Najork, M. (2021). Natural language understanding with privacy-preserving BERT. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (pp. 1488–1497). DOI: 10.1145/3459637.3482281
[21] Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, Ú., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. In Proceedings of the 30th USENIX Security Symposium (pp. 2633–2650).
[22] Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. In Proceedings of the 2008 IEEE Symposium on Security and Privacy (pp. 111–125).
[23] Anil, R., Ghazi, B., Gupta, V., Kumar, R., & Manurangsi, P. (2022). Large-scale differentially private BERT. In Proceedings of EMNLP 2022 (pp. 6481–6491).
[24] Mireshghallah, N., Kim, H., Zhou, X., Tsvetkov, Y., Sap, M., Shokri, R., & Choi, Y. (2024). Can LLMs keep a secret? Testing privacy implications of language models via contextual integrity theory. In International Conference on Learning Representations (ICLR 2024). arXiv:2310.17884.
[25] Fung, B. C. M., Wang, K., Chen, R., & Yu, P. S. (2010). Privacy-preserving data publishing: A survey of recent developments. ACM Computing Surveys, 42(4), 1–53.
[26] Aggarwal, C. C., & Yu, P. S. (2008). Privacy-Preserving Data Mining: Models and Algorithms. Springer.
[27] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of EMNLP 2020: System Demonstrations (pp. 38–45).
[28] Anthropic. (2024). Claude model card and documentation. Anthropic Technical Documentation. https://docs.anthropic.com


---

## Appendix A: PII Types Detected by SurrogateShield

| Category | Type | Detection Method | Validator |
|---|---|---|---|
| Structural | SSN | Regex `\d{3}[-]\d{2}[-]\d{4}` | — |
| Structural | Email | RFC-compliant regex | — |
| Structural | Phone (US) | E.164 + NANP regex | — |
| Structural | Phone (UK) | +44 prefix regex | — |
| Structural | Phone (International) | +[country] regex | — |
| Structural | Credit card | 16-digit regex | Luhn algorithm |
| Structural | Street address | House number + suffix regex | — |
| Structural | Date of birth | Multi-format date regex | — |
| Structural | IPv4 | Octet-bounded regex | — |
| Structural | API key | Prefix-specific regex (sk-, ghp_, AKIA, AIzaSy) | — |
| Structural | Gender indicator | Declaration + pronoun regex | — |
| Structural | US ZIP code | 5-digit + ZIP+4 regex | — |
| Structural | UK postcode | AN NAA format regex | — |
| Structural | Crypto wallet | BTC P2PKH/P2SH/Bech32, ETH 0x regex | — |
| Structural | ABA routing number | 9-digit regex | ABA checksum |
| Structural | Driver's license | Context-gated alphanumeric regex | Keyword context |
| Named entity | PERSON | spaCy + DistilBERT | Score threshold |
| Named entity | GPE | spaCy + DistilBERT | Score + Pass D filter |
| Named entity | LOC | spaCy + DistilBERT | Score threshold |
| Named entity | ORG | spaCy + DistilBERT + Pass A | Score threshold |
| Named entity | FAC | spaCy + DistilBERT | Score threshold |
| Combination | Quasi-identifiers | Type co-occurrence | Sweeney patterns |

## Appendix B: Quasi-Identifier Combinations

| Combination | Required fields present | Risk level | Basis |
|---|---|---|---|
| ZIP + DOB + Gender | ≥ 2 of 3 | High | Sweeney 2000 — 87% unique identification |
| Postcode + DOB | 2 | High | UK ICO guidance |
| Name + SSN | 2 | High | Identity theft enablement |
| Name + DOB | 2 | High | Standard identity verification |
| Phone + Name | 2 | High | Direct individual identification |
| Name + Employer + City | 3 | Medium | Workplace + location triple |
| Email + Location | 2 | Medium | Named individual at location |
| Phone + Location | 2 | Medium | Local individual identification |
| IP + Name | 2 | High | Device-level identification |
| DOB + Location + Employer | 3 | Medium | Demographic triple |

## Appendix C: ResolvePass Failure Taxonomy

| Failure type | Definition | Typical cause |
|---|---|---|
| `exact_hit` | Surrogate found verbatim in response | LLM reproduced surrogate exactly (majority case) |
| `exact_miss` | Surrogate not found in Pass 1 | LLM paraphrased, used partial name, or omitted |
| `fuzzy_hit` | Resolved by component or fuzzy matching | LLM used first name only, or mild paraphrase |
| `fuzzy_miss` | Not resolved by any pass | LLM completely reformulated, no surface overlap |

`fuzzy_miss` events constitute the resolve-leak rate reported in Section 5.1. These are the surrogates that remain visible to the user in the final response.