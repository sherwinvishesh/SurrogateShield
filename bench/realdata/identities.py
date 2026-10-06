"""Fake identities for the injected slice (Phase 3, E2/E3).

One identity per base prompt: a set of fake values, one per chosen GUIDE
``protect`` type (two for ID at most), generated locally from a seeded RNG and
Faker 40.15 (pinned). Nothing here is a real person's data and nothing is
invented by a model: Sonnet only *places* these exact strings (``inject.py``).

Choices that keep the benchmark fair and safe (REALDATA_PROGRESS D5, rule 5):

- Names, streets and postcodes come from Faker locales whose person lists
  SurrogateShield does not read (detection reads only zh_CN / hi_IN person
  lists, zh_CN cities and the en_US lorem list); every value is Latin-script.
  Cities (LOCATION, and the city of an address) come from ``CITIES`` below,
  real cities written for this file, because Faker's are invented
  ("East Jessicaview") and real users name real places.
- No value takes a real provider's token shape (``FORBIDDEN``: ``xox[abp]-``,
  ``shpat_``, ``ghp_``, ``sk-``/``sk_``, ``AKIA``, ``eyJ`` (JWT), PEM, ...);
  credentials are 32- or 48-character hex (not 40, GitHub's old token length),
  a ``zqk_`` token (a prefix no provider uses), three dash-joined lower-case
  groups, or a password from ``WORDS`` (not 40-character base64, an AWS secret
  key's shape). Card numbers pass Luhn and avoid the published test numbers; IBANs
  pass mod 97; phone numbers avoid the 555 and Ofcom drama ranges, so they do
  not read as placeholders.
- Which types an identity carries depends on the prompt's task (``PRIORS``: a
  coding question tends to get a credential and an IP, an advice question an
  address and an ID); ``assign_types`` then tops every type up to ``TARGET``
  prompts per dataset, so each has ≥ 40 after verification drops.
- A value never occurs (whole-word, case-sensitive) in the base text or inside
  another value of the same identity, so its gold occurrences are exactly the
  ones Sonnet places.
- ``shift=True`` (40 prompts per dataset) draws formats a regex may not have
  seen: lower-case / ALL-CAPS / surname-first names, SSN without dashes,
  phones with dots or in words, cards and IBANs spaced in fours, dates and
  ages written out, an e-mail written "name at domain dot com".

**Pool split** (PROMPT_FOR_OPUS_V3 §5.3). ``identity(..., pool="eval")`` (the
sealed test-2) and ``identity(..., pool="train")`` (the tagger's training data)
draw from disjoint halves of every word list a value is built from: first
names, surnames, street names, cities, company names, school towns, mail
providers and password words. Each word (``tokens``: a folded, lower-cased
run of 3+ letters; a name with none is keyed whole) belongs to one half by a
keyed hash (``token_half``), and a candidate value is redrawn until all its
words are in the identity's half. Structural words stay shared (``generic``:
street types, company legal forms, school kinds, flat / suite markers, months,
number words, region codes, link hosts, tlds) so both halves look alike.
E-mail local parts, handles and links are built from the identity's own name,
so they follow it. Digits (phones, IDs, postcodes, house numbers), hex and
random credentials are random draws and are not split. ``pool=None`` (test-1,
dev) is the original generator, draw for draw.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bench import realworld as rw
from bench.realdata.common import TASKS

TYPES = ("PERSON", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "ORG", "DATE_OF_BIRTH", "AGE", "ID", "NETWORK",
         "URL", "HANDLE", "CREDENTIAL")
assert set(TYPES) == rw.PROTECT_TYPES
TARGET = 50                       # prompts per type per dataset before verification
EXCLUSIVE = (("ADDRESS", "LOCATION"), ("DATE_OF_BIRTH", "AGE"))   # one of each pair (a city inside an address)

BASE = {"PERSON": .9, "EMAIL": .35, "PHONE": .3, "ADDRESS": .2, "LOCATION": .25, "ORG": .3, "DATE_OF_BIRTH": .15,
        "AGE": .2, "ID": .2, "NETWORK": .06, "URL": .15, "HANDLE": .15, "CREDENTIAL": .06}
PRIORS: Dict[str, Dict[str, float]] = {t: dict(BASE) for t in TASKS}
PRIORS["coding"].update(CREDENTIAL=.55, NETWORK=.5, EMAIL=.45, URL=.3, HANDLE=.3, ADDRESS=.05, DATE_OF_BIRTH=.05,
                        AGE=.05, LOCATION=.1)
PRIORS["advice"].update(ADDRESS=.35, DATE_OF_BIRTH=.3, AGE=.35, ID=.4, PHONE=.35, LOCATION=.35)
PRIORS["business"].update(ORG=.6, EMAIL=.55, PHONE=.5, URL=.35, ADDRESS=.25)
PRIORS["writing"].update(EMAIL=.4, PHONE=.35, ADDRESS=.3, ORG=.35)
PRIORS["qa"].update(LOCATION=.35, AGE=.3)
PRIORS["translation"].update(ADDRESS=.3, PHONE=.3, DATE_OF_BIRTH=.25)
PRIORS["roleplay"].update(HANDLE=.3, AGE=.3, LOCATION=.3)

# name locale -> weight; the identity's country (phone, address, IDs) follows it, or is drawn for diaspora locales
NAME_LOCALES = {"en_US": 6, "en_GB": 4, "en_IE": 1, "ga_IE": 1, "en_IN": 3, "es_ES": 2, "es_MX": 2, "fr_FR": 2,
                "de_DE": 2, "it_IT": 1, "pt_BR": 2, "nl_NL": 1, "pl_PL": 1, "sv_SE": 1, "tr_TR": 1, "vi_VN": 1,
                "id_ID": 1, "fi_FI": 1, "cs_CZ": 1, "da_DK": 1, "no_NO": 1, "ro_RO": 1, "yo_NG": 2, "sw": 1,
                "zu_ZA": 1, "tw_GH": 1, "hr_HR": 1}
HOME = {"en_US": "US", "en_GB": "UK", "en_IE": "IE", "ga_IE": "IE", "en_IN": "IN", "es_ES": "ES", "es_MX": "MX",
        "fr_FR": "FR", "de_DE": "DE", "pt_BR": "BR", "nl_NL": "NL"}

# country -> Faker locale for streets / postcodes, address layout, real cities (with region where the layout uses one)
COUNTRIES = {
    "US": ("en_US", "{street}, {city}, {region} {postcode}",
           [("Chicago", "IL"), ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"), ("San Antonio", "TX"),
            ("San Diego", "CA"), ("Dallas", "TX"), ("Austin", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"),
            ("Indianapolis", "IN"), ("Seattle", "WA"), ("Denver", "CO"), ("Nashville", "TN"), ("Portland", "OR"),
            ("Las Vegas", "NV"), ("Detroit", "MI"), ("Memphis", "TN"), ("Louisville", "KY"), ("Milwaukee", "WI"),
            ("Albuquerque", "NM"), ("Tucson", "AZ"), ("Sacramento", "CA"), ("Atlanta", "GA"), ("Raleigh", "NC"),
            ("Omaha", "NE"), ("Minneapolis", "MN"), ("Tulsa", "OK"), ("Cleveland", "OH"), ("Tampa", "FL"),
            ("Pittsburgh", "PA"), ("Cincinnati", "OH"), ("Boise", "ID"), ("Madison", "WI"), ("Richmond", "VA"),
            ("Spokane", "WA"), ("Des Moines", "IA"), ("Savannah", "GA"), ("Ann Arbor", "MI"), ("Provo", "UT")]),
    "UK": ("en_GB", "{street}, {city} {postcode}",
           [("Leeds", ""), ("Sheffield", ""), ("Bristol", ""), ("Manchester", ""), ("Liverpool", ""),
            ("Nottingham", ""), ("Leicester", ""), ("Coventry", ""), ("Bradford", ""), ("Cardiff", ""),
            ("Glasgow", ""), ("Edinburgh", ""), ("Belfast", ""), ("Brighton", ""), ("Plymouth", ""),
            ("Southampton", ""), ("Reading", ""), ("Derby", ""), ("Norwich", ""), ("York", ""), ("Swansea", ""),
            ("Aberdeen", ""), ("Exeter", ""), ("Bath", ""), ("Oxford", ""), ("Cambridge", ""), ("Hull", "")]),
    "IE": ("en_IE", "{street}, {city}",
           [("Dublin", ""), ("Cork", ""), ("Galway", ""), ("Limerick", ""), ("Waterford", ""), ("Kilkenny", "")]),
    "CA": ("en_CA", "{street}, {city}, {region} {postcode}",
           [("Toronto", "ON"), ("Ottawa", "ON"), ("Hamilton", "ON"), ("Calgary", "AB"), ("Edmonton", "AB"),
            ("Winnipeg", "MB"), ("Halifax", "NS"), ("Victoria", "BC"), ("Saskatoon", "SK"), ("Kelowna", "BC")]),
    "AU": ("en_AU", "{street}, {city} {region} {postcode}",
           [("Brisbane", "QLD"), ("Perth", "WA"), ("Adelaide", "SA"), ("Hobart", "TAS"), ("Geelong", "VIC"),
            ("Newcastle", "NSW"), ("Wollongong", "NSW"), ("Cairns", "QLD"), ("Darwin", "NT"), ("Ballarat", "VIC")]),
    "DE": ("de_DE", "{street}, {postcode} {city}",
           [("Leipzig", ""), ("Dresden", ""), ("Hannover", ""), ("Nürnberg", ""), ("Bremen", ""), ("Bonn", ""),
            ("Mannheim", ""), ("Freiburg", ""), ("Münster", ""), ("Augsburg", ""), ("Kassel", ""), ("Mainz", "")]),
    "FR": ("fr_FR", "{street}, {postcode} {city}",
           [("Lyon", ""), ("Toulouse", ""), ("Nantes", ""), ("Strasbourg", ""), ("Montpellier", ""), ("Lille", ""),
            ("Rennes", ""), ("Grenoble", ""), ("Dijon", ""), ("Angers", ""), ("Nîmes", ""), ("Tours", "")]),
    "ES": ("es_ES", "{street}, {postcode} {city}",
           [("Valencia", ""), ("Sevilla", ""), ("Zaragoza", ""), ("Málaga", ""), ("Bilbao", ""), ("Murcia", ""),
            ("Granada", ""), ("Alicante", ""), ("Valladolid", ""), ("Salamanca", "")]),
    "NL": ("nl_NL", "{street}, {postcode} {city}",
           [("Utrecht", ""), ("Eindhoven", ""), ("Groningen", ""), ("Tilburg", ""), ("Nijmegen", ""), ("Haarlem", ""),
            ("Leiden", ""), ("Arnhem", ""), ("Delft", ""), ("Breda", "")]),
    "IN": ("en_IN", "{street}, {city} {postcode}",
           [("Pune", ""), ("Jaipur", ""), ("Lucknow", ""), ("Nagpur", ""), ("Indore", ""), ("Bhopal", ""),
            ("Coimbatore", ""), ("Kochi", ""), ("Mysuru", ""), ("Chandigarh", ""), ("Surat", ""), ("Vadodara", "")]),
    "MX": ("es_MX", "{street}, {postcode} {city}",
           [("Guadalajara", ""), ("Monterrey", ""), ("Puebla", ""), ("Querétaro", ""), ("Mérida", ""), ("León", ""),
            ("Morelia", ""), ("Oaxaca", "")]),
    "BR": ("pt_BR", "{street}, {city} - {region}, {postcode}",
           [("Curitiba", "PR"), ("Recife", "PE"), ("Fortaleza", "CE"), ("Salvador", "BA"), ("Campinas", "SP"),
            ("Goiânia", "GO"), ("Belém", "PA"), ("Florianópolis", "SC"), ("Manaus", "AM"), ("Natal", "RN")]),
}
# more real cities, used only with a pool, so that each half has a few per country (test-1 / dev never draw them)
EXTRA_CITIES = {
    "US": [("Fresno", "CA"), ("Wichita", "KS"), ("Lexington", "KY"), ("Anchorage", "AK"), ("Reno", "NV")],
    "UK": [("Stoke", ""), ("Sunderland", ""), ("Wolverhampton", ""), ("Inverness", ""), ("Lancaster", "")],
    "IE": [("Sligo", ""), ("Drogheda", ""), ("Dundalk", ""), ("Athlone", ""), ("Ennis", ""), ("Tralee", ""),
           ("Wexford", ""), ("Letterkenny", ""), ("Navan", ""), ("Carlow", "")],
    "CA": [("Regina", "SK"), ("Kingston", "ON"), ("Windsor", "ON"), ("Moncton", "NB"), ("Fredericton", "NB"),
           ("Guelph", "ON"), ("Kamloops", "BC"), ("Lethbridge", "AB"), ("Nanaimo", "BC")],
    "AU": [("Townsville", "QLD"), ("Toowoomba", "QLD"), ("Launceston", "TAS"), ("Bendigo", "VIC"),
           ("Mackay", "QLD"), ("Bunbury", "WA"), ("Albury", "NSW"), ("Rockhampton", "QLD")],
    "DE": [("Kiel", ""), ("Rostock", ""), ("Erfurt", ""), ("Potsdam", ""), ("Heidelberg", ""), ("Regensburg", ""),
           ("Würzburg", ""), ("Göttingen", ""), ("Ulm", ""), ("Lübeck", "")],
    "FR": [("Reims", ""), ("Limoges", ""), ("Brest", ""), ("Amiens", ""), ("Perpignan", ""), ("Metz", ""),
           ("Besançon", ""), ("Caen", ""), ("Orléans", ""), ("Rouen", "")],
    "ES": [("Córdoba", ""), ("Vigo", ""), ("Gijón", ""), ("Pamplona", ""), ("Santander", ""), ("Oviedo", ""),
           ("Burgos", ""), ("Cádiz", "")],
    "NL": [("Maastricht", ""), ("Zwolle", ""), ("Amersfoort", ""), ("Apeldoorn", ""), ("Enschede", ""),
           ("Leeuwarden", ""), ("Dordrecht", ""), ("Almere", "")],
    "IN": [("Patna", ""), ("Ludhiana", ""), ("Agra", ""), ("Nashik", ""), ("Visakhapatnam", ""), ("Madurai", ""),
           ("Varanasi", ""), ("Ranchi", ""), ("Guwahati", ""), ("Mangaluru", "")],
    "MX": [("Tijuana", ""), ("Chihuahua", ""), ("Toluca", ""), ("Aguascalientes", ""), ("Hermosillo", ""),
           ("Saltillo", ""), ("Culiacán", ""), ("Cancún", ""), ("Veracruz", ""), ("Acapulco", ""), ("Durango", ""),
           ("Mazatlán", ""), ("Zacatecas", ""), ("Tampico", "")],
    "BR": [("Vitória", "ES"), ("Maceió", "AL"), ("Teresina", "PI"), ("Londrina", "PR"), ("Joinville", "SC"),
           ("Uberlândia", "MG"), ("Sorocaba", "SP"), ("Aracaju", "SE"), ("Cuiabá", "MT")],
}
DIASPORA = ("US", "UK", "CA", "AU", "DE", "IE", "NL", "FR")
FREE_MAIL = ("gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "icloud.com", "proton.me", "aol.com",
             "hotmail.co.uk", "gmx.de", "web.de", "orange.fr", "libero.it", "yahoo.co.in", "live.com", "zoho.com")
WORDS = ("maple", "harbor", "copper", "lantern", "meadow", "falcon", "pepper", "violet", "summit", "willow",
         "orbit", "cactus", "marble", "thistle", "juniper", "cobalt", "saffron", "tundra", "quartz", "ember",
         "pelican", "biscuit", "walnut", "glacier", "nectar", "sparrow", "canyon", "mango", "radish", "hazel")
# real provider token prefixes (case-sensitive, at a token start, as providers issue them)
FORBIDDEN = re.compile(r"(?<![A-Za-z0-9])(?:xox[abposr]-|shpat_|shpss_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|"
                       r"npm_|sk-|sk_|rk_|pk_|AKIA|ASIA|eyJ|AIza|tok_|whsec_|SG\.|hf_|-----BEGIN)")
TEST_CARDS = {"4111111111111111", "4242424242424242", "5555555555554444", "5105105105105100", "4012888888881881",
              "378282246310005", "371449635398431", "4000056655665556", "5200828282828210"}
ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth twenty-first twenty-second "
            "twenty-third twenty-fourth twenty-fifth twenty-sixth twenty-seventh twenty-eighth twenty-ninth "
            "thirtieth thirty-first").split()
MONTHS = ("January February March April May June July August September October November December").split()
POOLS = ("eval", "train")
POOL_KEY = "surrogateshield-identity-pool-v1"
SCHOOLS = ('High School', 'Community College', 'Academy', 'Middle School', 'Institute of Technology')


# ── helpers ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def faker(locale: str):
    from faker import Faker
    return Faker(locale)


def fk(locale: str, rng: random.Random):
    f = faker(locale)
    f.seed_instance(rng.getrandbits(32))
    return f


def latin(s: str) -> bool:
    return all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in s)


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("ß", "ss").replace("ı", "i").replace("ł", "l").replace("Ł", "L"))
    return "".join(c for c in s if c.isascii() and (c.isalnum()))


def number_words(n: int) -> str:
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    t, o = divmod(n, 10)
    return TENS[t] + ("-" + ONES[o] if o else "")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    s = iban.replace(" ", "")
    s = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


def spaced(s: str, k: int = 4) -> str:
    return " ".join(s[i:i + k] for i in range(0, len(s), k))


# ── the evaluation / training pool split (§5.3) ──────────────────────────────

_RUN = re.compile(r"[^\W\d_]+")
_FAKER_WORDS = ("street_suffixes", "street_suffixes_long", "street_suffixes_short", "street_prefixes",
                "secondary_address_formats", "street_name_formats", "street_address_formats", "company_suffixes",
                "company_prefixes", "company_preffixes", "company_types", "company_adjectives", "formats")


def _runs(s: str) -> List[str]:
    return [w for w in (ascii_fold(r).lower() for r in _RUN.findall(s)) if w]


@lru_cache(maxsize=None)
def generic() -> frozenset:
    """Structural words shared by both pools: the literal words of Faker's
    street / company vocabularies and templates for every locale used here
    (placeholders removed), school kinds, months, number words, ordinals,
    region codes, particles, link hosts and top-level domains."""
    words = set()
    for loc in sorted({v[0] for v in COUNTRIES.values()}):
        for prov in faker(loc).providers:
            if ".address." not in type(prov).__module__ and ".company." not in type(prov).__module__:
                continue
            for attr in _FAKER_WORDS:
                v = getattr(prov, attr, None)
                for x in (v.keys() if isinstance(v, dict) else v) if isinstance(v, (tuple, list, dict)) else ():
                    if isinstance(x, str):
                        words.update(_runs(re.sub(r"\{\{.*?\}\}", " ", x)))
    words.update(_runs(" ".join(SCHOOLS + tuple(MONTHS) + tuple(ONES) + tuple(TEENS) + tuple(TENS[2:])
                                + tuple(ORDINALS))))
    words.update(r.lower() for v in list(COUNTRIES.values()) for _c, r in v[2] if r)
    words.update(r.lower() for v in EXTRA_CITIES.values() for _c, r in v if r)
    words.update("""the and of at dot de del della der des di da das do dos du la las le los san santa saint st
                 van von ter ten fort port new north south east west upper lower mount
                 com net org edu gov info biz dev app www http https mail
                 linkedin github calendly instagram drive google netlify file view min""".split())
    return frozenset(words)


def tokens(s: str) -> List[str]:
    """The pool-bearing words of *s*: folded, lower-cased runs of 3+ letters
    that are not ``generic``."""
    g = generic()
    return [w for w in _runs(s) if len(w) >= 3 and w not in g]


def name_keys(s: str) -> List[str]:
    """A name's pool words; a name with no 3-letter word (``Li``, ``Ng``) is keyed whole."""
    return tokens(s) or ["".join(_runs(s))]


def token_half(tok: str) -> str:
    """``eval`` or ``train``, by a keyed hash of the word (half the first byte's range each)."""
    return POOLS[hashlib.sha256(f"{POOL_KEY}:{tok}".encode()).digest()[0] >> 7]


def in_pool(keys: Iterable[str], pool: Optional[str]) -> bool:
    return pool is None or all(token_half(k) == pool for k in keys)


def cities(country: str, pool: Optional[str] = None) -> List[Tuple[str, str]]:
    if pool is None:
        return COUNTRIES[country][2]
    return [cr for cr in COUNTRIES[country][2] + EXTRA_CITIES.get(country, []) if in_pool(name_keys(cr[0]), pool)]


def mail_key(domain: str) -> List[str]:
    """A mail provider is keyed by its name (``outlook`` of ``outlook.com``), generic or not."""
    return ["".join(_runs(domain.split(".")[0]))]


def free_mail(pool: Optional[str] = None) -> Sequence[str]:
    return FREE_MAIL if pool is None else [d for d in FREE_MAIL if in_pool(mail_key(d), pool)]


def words(pool: Optional[str] = None) -> Sequence[str]:
    """Password words, keyed whole: several (``harbor``, ``summit``) are also street suffixes."""
    return WORDS if pool is None else [w for w in WORDS if in_pool([w], pool)]


# ── one value per type ───────────────────────────────────────────────────────

class Ctx:
    """What the values of one identity share. With a *pool*, every word a
    value is built from comes from that half (``take``), and the words used
    are kept in ``pool_tokens``."""

    def __init__(self, rng: random.Random, shift: bool, pool: Optional[str] = None):
        self.rng, self.shift, self.pool = rng, shift, pool
        self.pool_tokens: set = set()
        self.name_locale = rng.choices(list(NAME_LOCALES), weights=list(NAME_LOCALES.values()))[0]
        self.country = HOME.get(self.name_locale) or rng.choice(DIASPORA)
        for i in range(100000):
            if pool is not None and i and i % 3000 == 0:      # e.g. vi_VN: its two space-free first names are both eval
                self.name_locale = rng.choices(list(NAME_LOCALES), weights=list(NAME_LOCALES.values()))[0]
                self.country = HOME.get(self.name_locale) or rng.choice(DIASPORA)
            f = fk(self.name_locale, rng)
            first, last = f.first_name(), f.last_name()
            if latin(first + last) and " " not in first and len(ascii_fold(first)) >= 2 and len(ascii_fold(last)) >= 2 \
                    and self.take(first, last, names=True):
                break
        else:
            raise RuntimeError(f"no {self.name_locale} name in the {pool} pool")
        self.first, self.last = first, last
        self.slug_first, self.slug_last = ascii_fold(first).lower(), ascii_fold(last).lower()
        self.org: Optional[str] = None
        self.city: Optional[Tuple[str, str]] = None

    def take(self, *texts: str, names: bool = False, keys: Sequence[str] = ()) -> bool:
        """True (and the words recorded) when every pool word of *texts* (and
        every key in *keys*) is in this identity's pool; always True without
        one, consuming no randomness."""
        if self.pool is None:
            return True
        keys = list(keys) + [k for t in texts for k in (name_keys(t) if names else tokens(t))]
        if not in_pool(keys, self.pool):
            return False
        self.pool_tokens.update(keys)
        return True


def person(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    name = f"{c.first} {c.last}"
    if rng.random() < .15:
        name = f"{c.first} {rng.choice(string.ascii_uppercase)}. {c.last}"
    if not c.shift:
        return name, "plain"
    k = rng.choice(("lower", "upper", "surname-first"))
    return {"lower": name.lower(), "upper": name.upper(), "surname-first": f"{c.last}, {c.first}"}[k], k


def email(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    f, l = c.slug_first, c.slug_last
    yy = str(rng.randint(1, 99)).zfill(2)
    local = rng.choice((f"{f}.{l}", f"{f[0]}{l}", f"{f}{l}{yy}", f"{l}.{f}", f"{f}_{l}{yy}", f"{f}.{l}{yy}",
                        f"{f}{l[0]}"))
    if c.org and rng.random() < .4:
        dom = (ascii_fold(c.org.split()[0]).lower() or "mail") + rng.choice((".com", ".co", ".io", ".org"))
    else:
        dom = rng.choice(free_mail(c.pool))
        if c.pool is not None:
            c.take(keys=mail_key(dom))
    if c.shift and rng.random() < .5:
        return f"{local.replace('.', ' dot ')} at {dom.replace('.', ' dot ')}", "spelled"
    return f"{local}@{dom}", "plain"


US_AREA = (206, 212, 213, 303, 305, 312, 314, 404, 412, 415, 503, 512, 602, 612, 614, 615, 617, 646, 702, 704, 713,
           718, 720, 773, 801, 813, 818, 832, 917, 919, 929, 952)


def _digits(rng, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def phone(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    if c.country in ("US", "CA") or (c.country not in PHONES):
        a = rng.choice(US_AREA)
        while True:
            b = rng.randint(201, 989)
            if b != 555 and b % 100 != 11:
                break
        d = _digits(rng, 4)
        if c.shift:
            k = rng.choice(("dots", "words"))
            if k == "dots":
                return f"{a}.{b}.{d}", k
            return ", ".join(" ".join(ONES[int(x)] for x in part) for part in (str(a), str(b), d)), k
        return rng.choice((f"({a}) {b}-{d}", f"{a}-{b}-{d}", f"+1 {a} {b} {d}", f"{a}{b}{d}")), "plain"
    fmt = rng.choice(PHONES[c.country])
    num = "".join(rng.choice(string.digits) if ch == "#" else ch for ch in fmt)
    if c.shift:
        return re.sub(r"[ \-()]+", ".", num).replace("+.", "+").strip("."), "dots"
    return num, "plain"


PHONES = {"UK": ("07### ######", "+44 7### ######", "020 3### ####", "+44 161 ### ####", "0113 ### ####"),
          "IE": ("087 ### ####", "+353 86 ### ####"), "AU": ("04## ### ###", "+61 4## ### ###"),
          "DE": ("+49 151 ########", "0176 ########", "+49 30 #######"), "FR": ("06 ## ## ## ##", "+33 7 ## ## ## ##"),
          "ES": ("+34 6## ### ###", "6## ## ## ##"), "NL": ("06 ########", "+31 6 ########"),
          "IN": ("+91 9#### #####", "98### #####"), "MX": ("+52 55 #### ####", "+52 33 #### ####"),
          "BR": ("+55 11 9####-####", "(21) 9####-####")}


def city(c: Ctx) -> Tuple[str, str]:
    if c.city is None:
        c.city = c.rng.choice(cities(c.country if c.country in COUNTRIES else "US", c.pool))
        c.take(c.city[0], names=True)
    return c.city


def address(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    country = c.country if c.country in COUNTRIES else "US"
    loc, layout, _ = COUNTRIES[country]
    town, region = city(c)
    for _ in range(50 if c.pool is None else 2000):
        f = fk(loc, rng)
        street = f.street_address().replace("\n", ", ")
        if latin(street) and not re.search(r"\b(PSC|USNS|USNV|USS|APO|FPO|DPO|Unit|Box)\b", street) \
                and c.take(street):
            break
    else:
        if c.pool is not None:
            raise RuntimeError(f"no {loc} street in the {c.pool} pool")
    postcode = f.postcode() if country != "IE" else ""
    value = layout.format(street=street, city=town, region=region, postcode=postcode).strip().rstrip(",")
    return value, "plain"


def location(c: Ctx) -> Tuple[str, str]:
    return city(c)[0], "plain"


def org(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    if c.org is None:
        if rng.random() < .7:
            loc = COUNTRIES.get(c.country, COUNTRIES["US"])[0]
            for _ in range(50 if c.pool is None else 2000):
                name = fk(loc, rng).company()
                if latin(name) and len(name) <= 40 and c.take(name):
                    break
            else:
                if c.pool is not None:
                    raise RuntimeError(f"no {loc} company in the {c.pool} pool")
            c.org = name
        else:
            if rng.random() < .5:
                town = city(c)[0]
            else:
                town = fk("en_US", rng).last_name()
                while not c.take(town, names=True):          # only with a pool
                    town = fk("en_US", rng).last_name()
            c.org = f"{town} {rng.choice(SCHOOLS)}"
    return c.org, "plain"


def dob(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    y, m, d = rng.randint(1950, 2006), rng.randint(1, 12), rng.randint(1, 28)
    if c.shift:
        k = rng.choice(("ordinal-of", "month-ordinal"))
        if k == "ordinal-of":
            return f"the {ORDINALS[d - 1]} of {MONTHS[m - 1]} {y}", k
        return f"{MONTHS[m - 1]} {ORDINALS[d - 1]}, {y}", k
    us = c.country == "US"
    opts = [f"{m:02d}/{d:02d}/{y}" if us else f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}",
            f"{MONTHS[m - 1]} {d}, {y}" if us else f"{d} {MONTHS[m - 1]} {y}"]
    return rng.choice(opts), "plain"


def age(c: Ctx) -> Tuple[str, str]:
    n = c.rng.randint(18, 79)
    if c.shift:
        return number_words(n), "words"
    return str(n), "plain"


def card(rng) -> str:
    while True:
        prefix = rng.choice(("4", "51", "52", "53", "54", "55"))
        body = prefix + _digits(rng, 15 - len(prefix))
        check = next(str(x) for x in range(10) if luhn_ok(body + str(x)))
        num = body + check
        if num not in TEST_CARDS and len(set(num)) > 3:
            return num


def ssn(rng) -> str:
    while True:
        a = rng.randint(1, 899)
        if a != 666:
            return f"{a:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


IBAN_LOCALE = {"UK": "en_GB", "IE": "en_IE", "DE": "de_DE", "FR": "fr_FR", "ES": "es_ES", "NL": "nl_NL",
               "BR": "pt_BR"}


def id_value(c: Ctx, kind: str) -> Tuple[str, str]:
    rng = c.rng
    if kind == "ssn":
        v = ssn(rng)
        if c.shift:
            return v.replace("-", ""), "ssn-nodash"
        return v, "ssn"
    if kind == "card":
        v = card(rng)
        if c.shift:
            return spaced(v), "card-spaced"
        return rng.choice((v, "-".join(v[i:i + 4] for i in range(0, 16, 4)))), "card"
    if kind == "iban":
        loc = IBAN_LOCALE.get(c.country, "de_DE")
        v = fk(loc, rng).iban()
        assert iban_ok(v)
        if c.shift:
            return spaced(v), "iban-spaced"
        return v, "iban"
    if kind == "passport":
        v = {"US": lambda: _digits(rng, 9), "UK": lambda: _digits(rng, 9),
             "IN": lambda: rng.choice("JKLMNPRSTUVWZ") + _digits(rng, 7),
             "DE": lambda: "C" + "".join(rng.choice("CFGHJKLMNPRTVWXYZ0123456789") for _ in range(8)),
             "FR": lambda: _digits(rng, 2) + "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + _digits(rng, 5),
             }.get(c.country, lambda: rng.choice(string.ascii_uppercase) + _digits(rng, 8))()
        return v, "passport"
    if kind == "nino":
        a = rng.choice("ABCEGHJKLMNOPRSTWXYZ")
        b = rng.choice("ABCEGHJKLMNPRSTWXYZ")
        if a + b in ("BG", "GB", "NK", "KN", "TN", "NT", "ZZ"):
            a, b = "A", "B"
        d = _digits(rng, 6)
        return f"{a}{b} {d[:2]} {d[2:4]} {d[4:]} {rng.choice('ABCD')}", "nino"
    if kind == "nhs":
        while True:
            body = str(rng.randint(1, 9)) + _digits(rng, 8)
            r = 11 - sum(int(x) * (10 - i) for i, x in enumerate(body)) % 11
            if r == 11:
                r = 0
            if r != 10:
                v = body + str(r)
                return f"{v[:3]} {v[3:6]} {v[6:]}", "nhs"
    if kind == "policy":
        p = rng.choice(("POL", "HMO", "MBR", "PLN", "INS"))
        return f"{p}-{_digits(rng, rng.choice((7, 8, 9)))}", "policy"
    if kind == "license":
        return rng.choice(string.ascii_uppercase) + _digits(rng, rng.choice((7, 8, 12))), "license"
    if kind == "student":
        return rng.choice(("S", "U", "EMP-", "ID ")) + _digits(rng, rng.choice((6, 7, 8))), "student"
    raise ValueError(kind)


ID_KINDS = {"advice": ("ssn", "card", "iban", "passport", "policy", "nhs", "nino", "license"),
            "business": ("iban", "card", "student", "policy"),
            "coding": ("student", "license", "ssn"),
            "translation": ("passport", "license", "iban", "nino"),
            "writing": ("passport", "ssn", "policy", "student", "license"),
            "qa": ("ssn", "passport", "card", "policy"),
            "roleplay": ("student", "license", "passport"),
            "other": ("ssn", "card", "iban", "passport", "policy", "student", "license")}
COUNTRY_IDS = {"ssn": ("US",), "nino": ("UK",), "nhs": ("UK",), "iban": tuple(IBAN_LOCALE)}
ID_WEIGHT = {"ssn": 4, "card": 3, "iban": 3, "nino": 3, "nhs": 2, "passport": 2, "policy": 1, "license": 1,
             "student": 1}


def ids(c: Ctx, task: str) -> List[Tuple[str, str]]:
    """One ID (two in 30 % of identities): any kind valid for the identity's
    country, weighted by ``ID_WEIGHT`` and tripled for the kinds typical of
    the task; shifted identities draw from the kinds with a shifted format."""
    rng = c.rng
    kinds = sorted(k for k in ID_WEIGHT if c.country in COUNTRY_IDS.get(k, (c.country,)))
    if c.shift:
        kinds = [k for k in kinds if k in ("ssn", "card", "iban")]
    typical = ID_KINDS.get(task, ID_KINDS["other"])
    w = {k: ID_WEIGHT[k] * (3 if k in typical else 1) for k in kinds}
    chosen = rng.choices(kinds, weights=[w[k] for k in kinds])
    if rng.random() < .3 and len(kinds) > 1:
        rest = [k for k in kinds if k != chosen[0]]
        chosen += rng.choices(rest, weights=[w[k] for k in rest])
    return [id_value(c, k) for k in chosen]


def network(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    r = rng.random()
    if r < .7:
        return fk("en_US", rng).ipv4_public(), "ipv4"
    if r < .8:
        return fk("en_US", rng).ipv6(), "ipv6"
    return ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6)), "mac"


def url(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    f, l = c.slug_first, c.slug_last
    n = rng.randint(1, 99)
    doc = "".join(rng.choice(string.ascii_letters + string.digits + "-_") for _ in range(33))
    return rng.choice((f"https://www.linkedin.com/in/{f}-{l}-{n}", f"https://github.com/{f}{l}{n}",
                       f"https://{f}{l}.dev", f"https://calendly.com/{f}-{l}/30min", f"https://www.instagram.com/{f}.{l}{n}/",
                       f"https://drive.google.com/file/d/1{doc}/view", f"https://{f}-{l}.netlify.app")), "plain"


def handle(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    f, l = c.slug_first, c.slug_last
    n = rng.randint(1, 99)
    return "@" + rng.choice((f"{f}_{l}{n}", f"{f}{l[0]}{n}", f"{l}.{f}", f"{f}{l}", f"the_{f}{n}", f"{f[0]}{l}_{n}")), "plain"


def credential(c: Ctx) -> Tuple[str, str]:
    rng = c.rng
    for _ in range(100):
        k = rng.choice(("hex", "zqk", "password", "secret"))
        if k == "hex":
            v = "".join(rng.choice("0123456789abcdef") for _ in range(rng.choice((32, 48))))
        elif k == "zqk":
            v = "zqk_" + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(28))
        elif k == "password":
            ws = words(c.pool)
            a, mark, b = rng.choice(ws), rng.choice("!#$%&*?"), rng.choice(ws)
            v = a.capitalize() + mark + b + _digits(rng, rng.randint(2, 4))
            c.take(keys=(a, b))
        else:
            v = "-".join("".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8)) for _ in range(3))
        if not FORBIDDEN.search(v):
            return v, k
    raise RuntimeError("no credential without a provider shape")


MAKERS = {"PERSON": person, "EMAIL": email, "PHONE": phone, "ADDRESS": address, "LOCATION": location, "ORG": org,
          "DATE_OF_BIRTH": dob, "AGE": age, "NETWORK": network, "URL": url, "HANDLE": handle,
          "CREDENTIAL": credential}
ORDER = ("ORG", "LOCATION", "ADDRESS", "PERSON", "EMAIL", "PHONE", "DATE_OF_BIRTH", "AGE", "ID", "NETWORK", "URL",
         "HANDLE", "CREDENTIAL")        # ORG and the city first: e-mail domains and addresses reuse them


# ── identities ───────────────────────────────────────────────────────────────

def clashes(value: str, others: Iterable[str], text: str) -> bool:
    """True when *value* occurs in *text* or inside / around another value."""
    if rw.occurrences(text, value):
        return True
    for o in others:
        if rw.occurrences(o, value) or rw.occurrences(value, o):
            return True
    return False


def identity(rng: random.Random, types: Sequence[str], task: str, shift: bool = False, text: str = "",
             avoid: Sequence[str] = (), pool: Optional[str] = None) -> dict:
    """One identity with a value for each of *types* (two for ID at most),
    none occurring in *text* or in another of its values, and none inside or
    around a string of *avoid* (the base text's ``keep`` / ``optional``
    values, so a placed value never overlaps one). Redraws the whole identity
    (new person) when a value cannot be placed after a few tries. With a
    *pool* (``eval`` / ``train``), every word comes from that half; the
    identity then also lists them (``pool_tokens``)."""
    if pool is not None and pool not in POOLS:
        raise ValueError(pool)
    for _attempt in range(20):
        c = Ctx(rng, shift, pool)
        values: List[dict] = []
        ok = True
        for t in ORDER:
            if t not in types:
                continue
            for _ in range(10):
                got = ids(c, task) if t == "ID" else [MAKERS[t](c)]
                vals = [v for v, _f in got]
                if len(set(vals)) == len(vals) and not any(FORBIDDEN.search(v) for v in vals) and not any(
                        clashes(v, [x["value"] for x in values] + [w for w in vals if w != v] + list(avoid), text)
                        for v in vals):
                    values += [{"value": v, "type": t, "fmt": f} for v, f in got]
                    break
                if t in ("ORG", "LOCATION", "ADDRESS"):
                    c.org = None if t == "ORG" else c.org
                    c.city = None if t != "ORG" else c.city
            else:
                ok = False
                break
        if ok:
            out = {"locale": c.name_locale, "country": c.country, "shift": shift, "values": values}
            if pool is not None:
                out.update(pool=pool, pool_tokens=sorted(c.pool_tokens))
            return out
    raise RuntimeError("could not build an identity that fits this text")


def capacity(words: int) -> int:
    """How many types a prompt of *words* words can take without the
    rewrite stopping being the user's text (REALDATA_PROGRESS D13)."""
    return max(2, min(7, 2 + words // 20))


def assign_types(rows: Sequence[dict], seed: int, target: int = TARGET) -> List[List[str]]:
    """Types per row (``rows``: ``{"task", "words"}``), drawn from the task
    priors under each row's capacity, then topped up so every type reaches
    *target* rows, adding where the prior is highest and there is room.
    Where short prompts leave no room, the type takes PERSON's place in a
    full row, while PERSON stays in at least ``2 * target`` rows (the priors
    put a name in nine prompts of ten)."""
    rng = random.Random(seed)
    out: List[List[str]] = []
    for r in rows:
        pri = PRIORS.get(r["task"], BASE)
        chosen = [t for t in TYPES if rng.random() < pri[t]]
        for a, b in EXCLUSIVE:
            if a in chosen and b in chosen:
                chosen.remove(rng.choice((a, b)))
        rng.shuffle(chosen)
        cap = capacity(r["words"])
        if "PERSON" in chosen:                       # a name comes first if drawn
            chosen.remove("PERSON")
            chosen.insert(0, "PERSON")
        out.append(chosen[:cap] or ["PERSON"])
    for t in sorted(TYPES, key=lambda t: sum(t in o for o in out)):
        need = target - sum(t in o for o in out)
        if need <= 0:
            continue
        excl = {b for a, b in EXCLUSIVE if a == t} | {a for a, b in EXCLUSIVE if b == t}
        cands = [i for i, o in enumerate(out)
                 if t not in o and not (excl & set(o)) and len(o) < capacity(rows[i]["words"])]
        cands.sort(key=lambda i: (-PRIORS.get(rows[i]["task"], BASE)[t], rng.random()))
        for i in cands[:need]:
            out[i].append(t)
        need -= len(cands[:need])
        spare = sum("PERSON" in o for o in out) - 2 * target
        if need <= 0 or t == "PERSON" or spare <= 0:
            continue
        full = [i for i, o in enumerate(out) if "PERSON" in o and len(o) >= 2 and t not in o and not (excl & set(o))]
        full.sort(key=lambda i: (-PRIORS.get(rows[i]["task"], BASE)[t], rng.random()))
        for i in full[:min(need, spare)]:
            out[i][out[i].index("PERSON")] = t
    return out


def type_counts(identities: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    rows: Dict[str, int] = {t: 0 for t in TYPES}
    vals: Dict[str, int] = {t: 0 for t in TYPES}
    for ident in identities:
        for t in {v["type"] for v in ident["values"]}:
            rows[t] += 1
        for v in ident["values"]:
            vals[v["type"]] += 1
    return {"rows": rows, "values": vals}
