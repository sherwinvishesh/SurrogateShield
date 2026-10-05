"""Build surrogateshield/core/detection/public_people.txt — the gazetteer of
widely known people that the relation gate keeps verbatim in a topical
message (audit I12, gate J2).

    python tools/build_public_people.py                 # fetch from Wikidata (QLever)
    python tools/build_public_people.py --csv dump.csv  # rebuild from a saved fetch

Source: Wikidata (CC0), every human (P31 = Q5) with at least --min-sitelinks
Wikipedia sitelinks, English label. Run once at build time; the library
never goes to the network.

A name is left out when it could just as well be a private person's: when
its first word starts at least --common-first names in the fetched list and
its last word ends at least --common-last of them (so "John Smith", "David
Lee", "Maria Garcia" and "Emma Wilson" are not in; "Alan Greenspan",
"Christine Lagarde" and "Jensen Huang" are), or when one
Faker locale lists its first word as a given name and its last word as a
surname ("Yusuf Demir"), or when every word has at most four letters
("Li Na", "Wang Yi"), unless it has --keep-ambiguous sitelinks or more.
How often a name occurs among notable people stands in for how common it is
in the population; Faker's locale lists cover the populations Wikidata
under-represents. When in doubt a name is left out: the gate then masks it,
which costs a public figure's name, never a private person's. Labels with
digits, brackets or commas, and one-word labels, are left out.

A second section lists surnames that alone stand for one famous person
("Einstein"): at least --surname-sitelinks, borne by at most four people
in the whole list, and not an ordinary English word (_WORDS).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "surrogateshield/core/detection/public_people.txt"
ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
QUERY = (
    "SELECT ?p ?label ?n WHERE {{ "
    "?p <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> . "
    "?p <http://wikiba.se/ontology#sitelinks> ?n . FILTER(?n >= {min}) "
    "?p <http://www.w3.org/2000/01/rdf-schema#label> ?label . FILTER(LANG(?label) = \"en\") }}"
)


# surnames of famous people that are also ordinary words or given names
_WORDS = frozenset("jobs polo hippo bleu teresa joyce byron dietrich luther locke "
                   "grant bush ford gates king lincoln newton franklin washington".split())


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def locale_pairs() -> list:
    """(given names, surnames) per Faker locale, folded."""
    import pkgutil
    from importlib import import_module

    import faker.providers.person as person
    out = []
    for mod in pkgutil.iter_modules(person.__path__):
        P = import_module(f"faker.providers.person.{mod.name}").Provider
        given, family = set(), set()
        for attr, bucket in (("first_names", given), ("first_names_male", given),
                             ("first_names_female", given), ("last_names", family)):
            names = getattr(P, attr, ())
            if isinstance(names, (list, tuple, dict, set, frozenset)):
                bucket.update(fold(n) for n in names)
        if given and family:
            out.append((frozenset(given), frozenset(family)))
    return out


def fetch(min_sitelinks: int) -> str:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": QUERY.format(min=min_sitelinks)})
    req = urllib.request.Request(url, headers={
        "Accept": "text/csv", "User-Agent": "SurrogateShield-gazetteer-build/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="a saved fetch (columns p,label,n)")
    ap.add_argument("--min-sitelinks", type=int, default=30)
    ap.add_argument("--keep-ambiguous", type=int, default=150)
    ap.add_argument("--surname-sitelinks", type=int, default=200)
    ap.add_argument("--common-first", type=int, default=10)
    ap.add_argument("--common-last", type=int, default=8)
    a = ap.parse_args(argv)

    raw = Path(a.csv).read_text("utf-8") if a.csv else fetch(a.min_sitelinks)
    rows = [(r["label"].strip(), int(r["n"])) for r in csv.DictReader(io.StringIO(raw))
            if int(r["n"]) >= a.min_sitelinks]
    if len(rows) < 1000:
        print(f"only {len(rows)} rows — the fetch probably failed", file=sys.stderr)
        return 1
    pairs = locale_pairs()
    first = Counter(fold(l.split()[0]) for l, _ in rows if " " in l)
    last = Counter(fold(l.split()[-1]) for l, _ in rows if " " in l)

    best: dict[str, int] = {}
    for label, n in rows:
        if re.search(r"[\d(),\[\]/]", label) or " " not in label:
            continue
        words = label.split()
        if not all(w[0].isupper() or w.lower() in ("de", "da", "van", "von", "der", "del",
                                                   "la", "le", "di", "du", "bin", "al", "el",
                                                   "y", "the", "of", "dos", "das", "ibn")
                   for w in words):
            continue
        f, l = fold(words[0]), fold(words[-1])
        ambiguous = ((first[f] >= a.common_first and last[l] >= a.common_last)
                     or any(f in g and l in fam for g, fam in pairs)
                     or all(len(w) <= 4 for w in words))
        if ambiguous and n < a.keep_ambiguous:
            continue
        key = fold(label)
        best[key] = max(best.get(key, 0), n)

    surnames: dict[str, int] = {}
    for label, n in rows:
        w = label.split()[-1] if " " in label else ""
        k = fold(w)
        if (n >= a.surname_sitelinks and len(w) >= 4 and w.isalpha() and w[0].isupper()
                and last[k] <= 4 and first[k] < a.common_first and k not in _WORDS):
            surnames[k] = max(surnames.get(k, 0), n)

    lines = [
        "# Widely known people — built by tools/build_public_people.py; do not edit.",
        f"# Source: Wikidata (CC0), humans with >= {a.min_sitelinks} sitelinks, "
        f"fetched {datetime.date.today().isoformat()}.",
        f"# {len(best)} full names, then {len(surnames)} surnames after '## surnames'.",
        *sorted(best),
        "## surnames",
        *sorted(surnames),
    ]
    OUT.write_text("\n".join(lines) + "\n", "utf-8")
    print(f"{len(rows)} rows → {len(best)} names, {len(surnames)} surnames → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
