# Nemotron-PII: why SS leaks more than GLiNER-PII on 5 types (diagnosis of `external_nemotron_pii.json`, made 2026-10-09)

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.external_diagnosis --out bench/results/external_nemotron_pii_diagnosis.json` at `88d6ccd86f42`. Gold rebuilt from the raw parquet; each arm's recorded spans re-scored with the scorer's code path and anchored on the committed per-type leaked counts (all 7 arms reproduce; span files: private). Nothing re-run, detection as frozen, no rule or result changed. An occurrence leaks when a letter or digit of it lies outside every edit, or when an overlapping replacement contains the value (4+ characters). Reasons are per value: partly_covered > copied > untouched (no edit overlapped any occurrence) > repeat_untouched (one occurrence replaced, another whole-word occurrence of the same value left).

## What the counts say

**URL** (273 of 312 leaked, 268 untouched): the frozen URL finder saw 292 of the 312 gold URLs and called 39 of them personal (a surrogate; 0 of those leaked). The other 253 it found and left as they are: 253 of them leaked, which with the 20 it never found (20 leaked) is the whole SS count. Nemotron-PII labels every URL as PII; SS's rule protects URLs that point to a person and treats the rest as opaque (audit I1, I8), so this gap is a difference of taxonomy, not a pattern that fails. GLiNER-PII tags 233 url spans for the 312 gold URLs and still leaks 157: 156 of 248 URLs with a path (24 of them partly covered, the path cut off) against 1 of 64 bare hosts; SS 227/248 and 46/64.

**AGE** (14 of 66 leaked): every gold age is a bare number of one or two digits. 4 of the 14 were replaced where they stand as an age and leak only because the same number recurs elsewhere in the record as a whole word (the scorer counts every occurrence). Of the remaining 10, 7 stand behind a Markdown-emphasised field label (`**Age:** 63`: 7 of the 7 ages written that way leaked; the frozen 'age:' cue pattern allows spaces and a colon between the word and the number, not the closing `**`), 1 of 1 are 'N years of age' (the unit patterns know 'years old', 'yo', 'y/o'), and the rest have none of the cues the frozen patterns need (a unit after the number: 3/29 leaked with one, 11/37 without; 'aged'/'age:'/'turned'; a JSON or YAML key; an Age column of a pasted table; 'I'm N'; a name then 'is N'; '(29F)'). By the separator after the nearest 'age' word (SS leaked/n): colon 7/18, no age word 6/36, space 1/12. GLiNER's zero-shot age label takes the field without a cue (1 leaked, a repeat).

**HANDLE** (21 of 102 leaked, 15 untouched, 6 partly covered): 102 of the 102 gold handles have no '@' before them, so SS's '@handle' pattern never applies; its keyword pattern takes a bare word as a handle only behind an explicit label or platform cue, and a plain lowercase word only behind a 'username:'-style label. Leaked: 0/14 behind such a label, 21/88 without one; 19/72 all-lowercase, 2/30 with a capital. The spaCy stage read part of 12 of them as a PERSON and left the rest (6 partly covered). GLiNER-PII's zero-shot username label needs no cue (1 leaked; 112 username spans, 6 outside every gold value).

**ORG** (50 of 265 leaked, 40 untouched, 9 partly covered, 1 repeats): SS's ORG is the spaCy NER stage, which wants context; a synthetic one-word company name leaks 25/57, two words 12/107; a name without a legal suffix 38/131, with one 12/134. SS made 488 ORG edits on the corpus (112 outside every gold value); GLiNER-PII made 1009 organization edits for the same 265 gold names, 419 of them outside every gold value (42 %): its recall on ORG is bought with spurious edits.

**ADDRESS** (15 of 167 leaked, 12 untouched): the frozen street-address rule is house-number-first (number, street words, a street suffix, then unit, city, state, postcode). Leaked: 14/63 addresses that do not start with a number against 1/104 that do; 14/82 in the two international strata against 1 in the US ones. GLiNER-PII leaked 7 (bootstrap +4.8 pp [-0.5, +10.2]: the only one of the five whose interval covers zero).

## Every protect type: leaked / values and the paired cluster bootstrap of SS − GLiNER arm

| type | values | SS | GLiNER-PII | tuned | Δ vs GLiNER-PII | Δ vs tuned |
|---|---|---|---|---|---|---|
| ADDRESS **(SS worse)** | 167 | 15 | 7 | 5 | +4.8 pp [-0.5, +10.2] | +6.0 pp [+1.3, +11.1] * |
| AGE **(SS worse)** | 66 | 14 | 1 | 1 | +19.7 pp [+10.6, +30.5] * | +19.7 pp [+10.2, +29.4] * |
| CREDENTIAL | 250 | 118 | 126 | 114 | -3.2 pp [-9.2, +2.5] | +1.6 pp [-4.3, +7.1] |
| DATE_OF_BIRTH | 174 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] |
| EMAIL | 444 | 5 | 18 | 18 | -2.9 pp [-4.9, -1.2] * | -2.9 pp [-4.8, -1.1] * |
| HANDLE **(SS worse)** | 102 | 21 | 1 | 1 | +19.6 pp [+11.8, +27.8] * | +19.6 pp [+11.4, +27.7] * |
| ID | 1088 | 19 | 567 | 500 | -50.4 pp [-53.2, -47.5] * | -44.2 pp [-47.1, -41.4] * |
| LOCATION | 495 | 106 | 138 | 98 | -6.5 pp [-12.0, -1.2] * | +1.6 pp [-3.2, +6.3] |
| NETWORK | 129 | 1 | 41 | 37 | -31.0 pp [-37.4, -23.4] * | -27.9 pp [-34.2, -20.9] * |
| ORG **(SS worse)** | 265 | 50 | 3 | 1 | +17.7 pp [+13.2, +22.3] * | +18.5 pp [+14.0, +23.2] * |
| PERSON | 890 | 48 | 125 | 103 | -8.6 pp [-12.0, -5.3] * | -6.2 pp [-9.5, -3.1] * |
| PHONE | 245 | 0 | 10 | 6 | -4.1 pp [-6.9, -1.6] * | -2.5 pp [-4.7, -0.4] * |
| URL **(SS worse)** | 312 | 273 | 157 | 157 | +37.2 pp [+31.5, +43.0] * | +37.2 pp [+31.0, +43.1] * |

## How the leaked values got through

| type | arm | leaked | untouched | repeat untouched | partly covered | copied | the arm's edit types overlapping them | edit types on the values it caught |
|---|---|---|---|---|---|---|---|---|
| ADDRESS | ss | 15 | 12 | 0 | 3 | 0 | GPE 2, address 2 | address 171, id_number 2, GPE 1, ORG 1 |
| ADDRESS | gliner_pii | 7 | 7 | 0 | 0 | 0 | – | address 181, location 2 |
| ADDRESS | gliner_pii_tuned | 5 | 5 | 0 | 0 | 0 | – | address 183, location 2 |
| AGE | ss | 14 | 10 | 4 | 0 | 0 | age 4 | age 59 |
| AGE | gliner_pii | 1 | 0 | 1 | 0 | 0 | age 1 | age 76 |
| AGE | gliner_pii_tuned | 1 | 0 | 1 | 0 | 0 | age 1 | age 76 |
| HANDLE | ss | 21 | 15 | 0 | 6 | 0 | PERSON 12 | PERSON 80, handle 64, credential 5, email 1, hostname 1 |
| HANDLE | gliner_pii | 1 | 1 | 0 | 0 | 0 | – | username 94, person 63, email 1 |
| HANDLE | gliner_pii_tuned | 1 | 1 | 0 | 0 | 0 | – | username 94, person 63, email 1 |
| ORG | ss | 50 | 40 | 1 | 9 | 0 | ORG 20, GPE 1 | ORG 309, PERSON 76, GPE 11, address 1 |
| ORG | gliner_pii | 3 | 2 | 1 | 0 | 0 | organization 2 | organization 486, username 2 |
| ORG | gliner_pii_tuned | 1 | 1 | 0 | 0 | 0 | – | organization 493, username 2 |
| URL | ss | 273 | 268 | 0 | 5 | 0 | id_number 3, GPE 2, email 1 | url 47 |
| URL | gliner_pii | 157 | 133 | 0 | 24 | 0 | url 9, organization 8, location 6, username 3, email 1, social security number 1 | url 178, bank account number 3 |
| URL | gliner_pii_tuned | 157 | 110 | 0 | 47 | 0 | url 35, organization 10, location 7, username 3, email 1, ip address 1 | url 178, bank account number 3 |

## The frozen URL finder on the gold URLs

| the finder says | n | leaked by SS |
|---|---|---|
| found personal | 39 | 0 |
| found opaque | 253 | 253 |
| not found | 20 | 20 |

the frozen rule (audit I1, I8): a URL gets a surrogate only when it points to a person (a profile on a social or code host, a personal domain or TLD, a shared document, a page under /team/ or /people/, or a site introduced as the user's own); every other URL is opaque, left as it is, and no pattern or NER stage matches inside it.

## Each arm's edits by edit type (those answering for the five types; all in the json)

| arm | edit type | edits | outside every gold value |
|---|---|---|---|
| ss | url | 47 | 0 (0 %) |
| ss | hostname | 6 | 5 (83 %) |
| ss | age | 73 | 10 (14 %) |
| ss | handle | 77 | 6 (8 %) |
| ss | ORG | 488 | 112 (23 %) |
| ss | address | 202 | 2 (1 %) |
| ss | zip_us | 29 | 0 (0 %) |
| ss | postcode_uk | 1 | 0 (0 %) |
| ss | GPE | 517 | 37 (7 %) |
| ss | PERSON | 1304 | 196 (15 %) |
| gliner_pii | url | 233 | 17 (7 %) |
| gliner_pii | age | 103 | 24 (23 %) |
| gliner_pii | username | 112 | 6 (5 %) |
| gliner_pii | organization | 1009 | 419 (42 %) |
| gliner_pii | address | 223 | 15 (7 %) |
| gliner_pii | location | 347 | 52 (15 %) |
| gliner_pii | person | 1489 | 557 (37 %) |
| gliner_pii_tuned | url | 318 | 37 (12 %) |
| gliner_pii_tuned | age | 114 | 33 (29 %) |
| gliner_pii_tuned | username | 122 | 12 (10 %) |
| gliner_pii_tuned | organization | 1178 | 564 (48 %) |
| gliner_pii_tuned | address | 239 | 27 (11 %) |
| gliner_pii_tuned | location | 426 | 78 (18 %) |
| gliner_pii_tuned | person | 1742 | 749 (43 %) |

## Leak rate inside shape classes of the gold value (n; leaked by SS / GLiNER-PII / tuned)

### ADDRESS

| class | value | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| comma_inside | false | 138 | 11 (8 %) | 5 (4 %) | 3 (2 %) |
| comma_inside | true | 29 | 4 (14 %) | 2 (7 %) | 2 (7 %) |
| has_digit | true | 167 | 15 (9 %) | 7 (4 %) | 5 (3 %) |
| labelled_field | false | 123 | 15 (12 %) | 5 (4 %) | 4 (3 %) |
| labelled_field | true | 44 | 0 (0 %) | 2 (5 %) | 1 (2 %) |
| newline_inside | false | 167 | 15 (9 %) | 7 (4 %) | 5 (3 %) |
| non_ascii | false | 167 | 15 (9 %) | 7 (4 %) | 5 (3 %) |
| occurrences | <=1 | 146 | 13 (9 %) | 6 (4 %) | 4 (3 %) |
| occurrences | >1 | 21 | 2 (10 %) | 1 (5 %) | 1 (5 %) |
| starts_with_digit | false | 63 | 14 (22 %) | 3 (5 %) | 3 (5 %) |
| starts_with_digit | true | 104 | 1 (1 %) | 4 (4 %) | 2 (2 %) |
| stratum | intl-structured | 46 | 6 (13 %) | 2 (4 %) | 2 (4 %) |
| stratum | intl-unstructured | 36 | 8 (22 %) | 1 (3 %) | 1 (3 %) |
| stratum | us-structured | 51 | 0 (0 %) | 3 (6 %) | 1 (2 %) |
| stratum | us-unstructured | 34 | 1 (3 %) | 1 (3 %) | 1 (3 %) |
| tokens | <=3 | 91 | 9 (10 %) | 2 (2 %) | 1 (1 %) |
| tokens | <=5 | 68 | 4 (6 %) | 3 (4 %) | 2 (3 %) |
| tokens | >5 | 8 | 2 (25 %) | 2 (25 %) | 2 (25 %) |
| unit | false | 154 | 11 (7 %) | 5 (3 %) | 3 (2 %) |
| unit | true | 13 | 4 (31 %) | 2 (15 %) | 2 (15 %) |

### AGE

| class | value | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| age_word_sep | colon | 18 | 7 (39 %) | 0 (0 %) | 0 (0 %) |
| age_word_sep | no age word | 36 | 6 (17 %) | 1 (3 %) | 1 (3 %) |
| age_word_sep | space | 12 | 1 (8 %) | 0 (0 %) | 0 (0 %) |
| digits_only | true | 66 | 14 (21 %) | 1 (2 %) | 1 (2 %) |
| emphasis_between | false | 59 | 7 (12 %) | 1 (2 %) | 1 (2 %) |
| emphasis_between | true | 7 | 7 (100 %) | 0 (0 %) | 0 (0 %) |
| labelled_field | false | 54 | 13 (24 %) | 1 (2 %) | 1 (2 %) |
| labelled_field | true | 12 | 1 (8 %) | 0 (0 %) | 0 (0 %) |
| length | <=2 | 66 | 14 (21 %) | 1 (2 %) | 1 (2 %) |
| occurrences | <=1 | 58 | 9 (16 %) | 0 (0 %) | 0 (0 %) |
| occurrences | >1 | 8 | 5 (62 %) | 1 (12 %) | 1 (12 %) |
| stratum | intl-structured | 11 | 4 (36 %) | 0 (0 %) | 0 (0 %) |
| stratum | intl-unstructured | 13 | 2 (15 %) | 0 (0 %) | 0 (0 %) |
| stratum | us-structured | 21 | 7 (33 %) | 0 (0 %) | 0 (0 %) |
| stratum | us-unstructured | 21 | 1 (5 %) | 1 (5 %) | 1 (5 %) |
| unit_inside | false | 66 | 14 (21 %) | 1 (2 %) | 1 (2 %) |
| years_after | false | 37 | 11 (30 %) | 0 (0 %) | 0 (0 %) |
| years_after | true | 29 | 3 (10 %) | 1 (3 %) | 1 (3 %) |
| years_of_age_after | false | 65 | 13 (20 %) | 1 (2 %) | 1 (2 %) |
| years_of_age_after | true | 1 | 1 (100 %) | 0 (0 %) | 0 (0 %) |

### HANDLE

| class | value | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| all_lower | false | 30 | 2 (7 %) | 0 (0 %) | 0 (0 %) |
| all_lower | true | 72 | 19 (26 %) | 1 (1 %) | 1 (1 %) |
| at_prefixed | false | 102 | 21 (21 %) | 1 (1 %) | 1 (1 %) |
| has_digit | false | 66 | 14 (21 %) | 1 (2 %) | 1 (2 %) |
| has_digit | true | 36 | 7 (19 %) | 0 (0 %) | 0 (0 %) |
| has_separator | false | 50 | 7 (14 %) | 1 (2 %) | 1 (2 %) |
| has_separator | true | 52 | 14 (27 %) | 0 (0 %) | 0 (0 %) |
| has_space | false | 102 | 21 (21 %) | 1 (1 %) | 1 (1 %) |
| has_upper | false | 72 | 19 (26 %) | 1 (1 %) | 1 (1 %) |
| has_upper | true | 30 | 2 (7 %) | 0 (0 %) | 0 (0 %) |
| labelled_field | false | 88 | 21 (24 %) | 1 (1 %) | 1 (1 %) |
| labelled_field | true | 14 | 0 (0 %) | 0 (0 %) | 0 (0 %) |
| length | <=10 | 24 | 8 (33 %) | 1 (4 %) | 1 (4 %) |
| length | <=6 | 13 | 2 (15 %) | 0 (0 %) | 0 (0 %) |
| length | >10 | 65 | 11 (17 %) | 0 (0 %) | 0 (0 %) |
| occurrences | <=1 | 83 | 19 (23 %) | 0 (0 %) | 0 (0 %) |
| occurrences | >1 | 19 | 2 (11 %) | 1 (5 %) | 1 (5 %) |
| stratum | intl-structured | 20 | 3 (15 %) | 1 (5 %) | 1 (5 %) |
| stratum | intl-unstructured | 31 | 6 (19 %) | 0 (0 %) | 0 (0 %) |
| stratum | us-structured | 21 | 6 (29 %) | 0 (0 %) | 0 (0 %) |
| stratum | us-unstructured | 30 | 6 (20 %) | 0 (0 %) | 0 (0 %) |

### ORG

| class | value | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| all_caps_token | false | 262 | 49 (19 %) | 3 (1 %) | 1 (0 %) |
| all_caps_token | true | 3 | 1 (33 %) | 0 (0 %) | 0 (0 %) |
| ampersand | false | 241 | 46 (19 %) | 3 (1 %) | 1 (0 %) |
| ampersand | true | 24 | 4 (17 %) | 0 (0 %) | 0 (0 %) |
| has_digit | false | 265 | 50 (19 %) | 3 (1 %) | 1 (0 %) |
| labelled_field | false | 258 | 49 (19 %) | 3 (1 %) | 1 (0 %) |
| labelled_field | true | 7 | 1 (14 %) | 0 (0 %) | 0 (0 %) |
| legal_suffix | false | 131 | 38 (29 %) | 3 (2 %) | 1 (1 %) |
| legal_suffix | true | 134 | 12 (9 %) | 0 (0 %) | 0 (0 %) |
| occurrences | <=1 | 155 | 19 (12 %) | 2 (1 %) | 1 (1 %) |
| occurrences | >1 | 110 | 31 (28 %) | 1 (1 %) | 0 (0 %) |
| possessive_or_apostrophe | false | 265 | 50 (19 %) | 3 (1 %) | 1 (0 %) |
| stratum | intl-structured | 60 | 11 (18 %) | 1 (2 %) | 1 (2 %) |
| stratum | intl-unstructured | 72 | 10 (14 %) | 0 (0 %) | 0 (0 %) |
| stratum | us-structured | 59 | 13 (22 %) | 2 (3 %) | 0 (0 %) |
| stratum | us-unstructured | 74 | 16 (22 %) | 0 (0 %) | 0 (0 %) |
| words | <=1 | 57 | 25 (44 %) | 3 (5 %) | 1 (2 %) |
| words | <=2 | 107 | 12 (11 %) | 0 (0 %) | 0 (0 %) |
| words | <=3 | 79 | 7 (9 %) | 0 (0 %) | 0 (0 %) |
| words | >3 | 22 | 6 (27 %) | 0 (0 %) | 0 (0 %) |

### URL

| class | value | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| has_digit_in_host | false | 282 | 243 (86 %) | 127 (45 %) | 127 (45 %) |
| has_digit_in_host | true | 30 | 30 (100 %) | 30 (100 %) | 30 (100 %) |
| host_labels | <=2 | 201 | 168 (84 %) | 84 (42 %) | 84 (42 %) |
| host_labels | <=3 | 74 | 68 (92 %) | 37 (50 %) | 37 (50 %) |
| host_labels | >3 | 37 | 37 (100 %) | 36 (97 %) | 36 (97 %) |
| ip_host | false | 282 | 243 (86 %) | 127 (45 %) | 127 (45 %) |
| ip_host | true | 30 | 30 (100 %) | 30 (100 %) | 30 (100 %) |
| labelled_field | false | 282 | 244 (87 %) | 141 (50 %) | 141 (50 %) |
| labelled_field | true | 30 | 29 (97 %) | 16 (53 %) | 16 (53 %) |
| occurrences | <=1 | 279 | 247 (89 %) | 141 (51 %) | 141 (51 %) |
| occurrences | >1 | 33 | 26 (79 %) | 16 (48 %) | 16 (48 %) |
| path | false | 64 | 46 (72 %) | 1 (2 %) | 1 (2 %) |
| path | true | 248 | 227 (92 %) | 156 (63 %) | 156 (63 %) |
| port | false | 267 | 232 (87 %) | 121 (45 %) | 121 (45 %) |
| port | true | 45 | 41 (91 %) | 36 (80 %) | 36 (80 %) |
| scheme | true | 312 | 273 (88 %) | 157 (50 %) | 157 (50 %) |
| stratum | intl-structured | 70 | 65 (93 %) | 39 (56 %) | 39 (56 %) |
| stratum | intl-unstructured | 68 | 57 (84 %) | 31 (46 %) | 31 (46 %) |
| stratum | us-structured | 77 | 64 (83 %) | 36 (47 %) | 36 (47 %) |
| stratum | us-unstructured | 97 | 87 (90 %) | 51 (53 %) | 51 (53 %) |
| tld | au | 1 | 1 (100 %) | 1 (100 %) | 1 (100 %) |
| tld | com | 229 | 200 (87 %) | 110 (48 %) | 110 (48 %) |
| tld | gov | 26 | 22 (85 %) | 9 (35 %) | 9 (35 %) |
| tld | io | 2 | 1 (50 %) | 1 (50 %) | 1 (50 %) |
| tld | ip | 30 | 30 (100 %) | 30 (100 %) | 30 (100 %) |
| tld | net | 1 | 1 (100 %) | 0 (0 %) | 0 (0 %) |
| tld | org | 23 | 18 (78 %) | 6 (26 %) | 6 (26 %) |
| www | false | 310 | 271 (87 %) | 156 (50 %) | 156 (50 %) |
| www | true | 2 | 2 (100 %) | 1 (50 %) | 1 (50 %) |

## Per stratum (n; leaked by SS / GLiNER-PII / tuned)

| type | stratum | n | SS | GLiNER-PII | tuned |
|---|---|---|---|---|---|
| ADDRESS | intl-structured | 46 | 6 | 2 | 2 |
| ADDRESS | intl-unstructured | 36 | 8 | 1 | 1 |
| ADDRESS | us-structured | 51 | 0 | 3 | 1 |
| ADDRESS | us-unstructured | 34 | 1 | 1 | 1 |
| AGE | intl-structured | 11 | 4 | 0 | 0 |
| AGE | intl-unstructured | 13 | 2 | 0 | 0 |
| AGE | us-structured | 21 | 7 | 0 | 0 |
| AGE | us-unstructured | 21 | 1 | 1 | 1 |
| HANDLE | intl-structured | 20 | 3 | 1 | 1 |
| HANDLE | intl-unstructured | 31 | 6 | 0 | 0 |
| HANDLE | us-structured | 21 | 6 | 0 | 0 |
| HANDLE | us-unstructured | 30 | 6 | 0 | 0 |
| ORG | intl-structured | 60 | 11 | 1 | 1 |
| ORG | intl-unstructured | 72 | 10 | 0 | 0 |
| ORG | us-structured | 59 | 13 | 2 | 0 |
| ORG | us-unstructured | 74 | 16 | 0 | 0 |
| URL | intl-structured | 70 | 65 | 39 | 39 |
| URL | intl-unstructured | 68 | 57 | 31 | 31 |
| URL | us-structured | 77 | 64 | 36 | 36 |
| URL | us-unstructured | 97 | 87 | 51 | 51 |
