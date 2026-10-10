# Checking the silver labels (100 messages)

The natural slice of the real-data benchmark is labelled by Claude Sonnet 4.6
(`bench/realdata/label.py`). This check measures how far those labels can be
trusted. It takes about two hours with the review page below, and must be
done by a person: the labels it produces stand as the human reference in the
paper.

## Files

```
python -m bench.realdata.label --human-check       # writes bench/realdata/human_check.jsonl
```

`human_check.jsonl` holds 100 real user messages drawn with a fixed seed: about
34 per dataset, half of them labelled by Sonnet as carrying personal data. It is
git-ignored and readable by you only (0600). **Do not commit it, paste it,
share it, or look anything up about the people in it.** Real messages can hold
real personal data even after the publishers' scrubbing.

Each line is one message:

| field | what to do |
|---|---|
| `id`, `dataset`, `text`, `earlier_turns` | read only; `earlier_turns` are the same user's previous turns, for context |
| `task`, `service_query`, `protect`, `sensitive`, `optional`, `keep` | **correct these**; they start as Sonnet's answer |
| `sonnet` | read only; Sonnet's original answer, kept for the comparison |
| `checked` | set to `true` once you have gone over the row |
| `notes` | optional, free text; never copy message text into it |

## The review page (the fast way)

```
python -m bench.realdata.human_check_page          # writes bench/realdata/build/human_check/index.html
```

Open that file in a browser. It is a plain page (no network, nothing sent),
git-ignored and 0600, with the 100 messages in a list on the left and one
message at a time on the right, Sonnet's labels marked in the text. For each
message: select text to add it to `protect` (choose the type), `sensitive`,
`optional` or `keep`; edit or remove the entries Sonnet gave; fix `task` and
`service_query`; tick **checked**; add a note if you want. Every entry is
linted as you type with the benchmark's rule (exact whole-word substring of
the message, known type, listed once, `keep` never overlapping `protect`).
Keys: `j`/`k` next and previous message, `n` next unchecked, `c` toggle
checked. Your work is kept in the browser's local storage between sessions;
"Reset to file" discards it.

When all rows are checked, **Export** downloads `human_check.done.jsonl`.
Move it to `bench/realdata/human_check.done.jsonl` (git-ignored) and run:

```
python -m bench.realdata.human_check_page --check bench/realdata/human_check.done.jsonl
python -m bench.realdata.label --agreement bench/realdata/human_check.done.jsonl --annotator "<your name or role>"
```

The first prints row ids and lint problems only (no text) and confirms the
rows match the source file; the second writes
`bench/results/realdata_label_agreement.json`: per type, the precision and
recall of Sonnet's `protect` and `sensitive` values against yours (same exact
value and type), message-level agreement on "carries personal data" with
Cohen's kappa, the annotator, the checked file's hash and the count of checked
rows per dataset. Only rows with `checked: true` count. Then add the row to
`bench/results/README.md`.

Editing the JSONL by hand instead of using the page works the same way: save
the corrected rows as `bench/realdata/human_check.done.jsonl` and run the two
commands above.

## Rules

Label with `bench/realworld/GUIDE.md` ("The four lists" and "service_query"),
with these additions for real chat text (the same ones Sonnet was given,
`bench/realdata/label.py` `INSTRUCTIONS`):

1. A value is an exact substring of `text`: same case, spelling and spacing, no
   surrounding quotes or punctuation, a whole word, listed once, in one list.
2. A **private person** is the user or someone they know or deal with (family,
   colleague, patient, customer, landlord). Their names, contact details,
   addresses, ids, accounts, and the places, organisations and dates that point
   to them are `protect`.
3. Public figures, authors, companies, products, places in a general question
   and characters from published works are `keep`.
4. Names and details the user invents for a story, a role-play or a template,
   and obvious placeholders ("John Doe", "example.com"), are `optional`.
   Template slots such as `[Your Name]` are not values; leave them out (Sonnet
   may have put one in `keep`, which is fine).
5. In pasted code, logs or configs: credentials, tokens, passwords, keys,
   personal e-mail addresses, and IP addresses or hostnames that identify a
   person's machine are `protect`.
6. `sensitive` is only a special-category fact (health, religion, ethnicity,
   orientation, political view) about a private person. A general question
   about a condition is not sensitive.
7. `keep` is for words a privacy filter might wrongly change; it does not need
   to be complete and does not enter the agreement numbers.

When unsure, choose what a careful privacy reviewer would want a filter to do,
and say why in `notes` (without quoting the text).
