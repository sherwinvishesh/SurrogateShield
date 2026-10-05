"""
generation/places.py — real places for GPE / LOC surrogates (audit I7).

A fictional city ("Port Matthew") leaves the LLM nothing to reason about:
climate, time zone, law and cost of living all depend on the place being
real. A surrogate town is a real mid-sized town of the same country when the
country is known (from the town itself or the message), a US state stays a US
state in the same form, and a lake, mountain, river, valley, park, beach,
island or county stays one.

No surrogate is a major city or country (those are never masked — see
``detection.geo_data`` — so a user may type them for real later) or an
ordinary English word ("Mobile", "Reading", "Nice").
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Sequence, Tuple

from ..detection.geo_data import GEO_PASS_THROUGH, US_STATE_ABBREVS, US_STATES


def _towns(block: str) -> Tuple[str, ...]:
    """Split a '|' list and strip each name: padding in a surrogate marks it as fake."""
    return tuple(t.strip() for t in block.split("|"))


TOWNS: Dict[str, Tuple[str, ...]] = {
    "US": _towns("""Flagstaff|Prescott|Yuma|Boulder|Fort Collins|Pueblo|Salem|Spokane|Tacoma
        |Bellingham|Olympia|Boise|Missoula|Billings|Reno|Provo|Ogden|Santa Fe|Las Cruces
        |Bakersfield|Modesto|Santa Rosa|Redding|Santa Barbara|Pasadena|Ventura|Lubbock|Amarillo
        |Waco|Abilene|Midland|Corpus Christi|Tulsa|Wichita|Topeka|Sioux Falls|Fargo|Des Moines
        |Cedar Rapids|Green Bay|Duluth|Ann Arbor|Grand Rapids|Lansing|Toledo|Akron|Dayton
        |Fort Wayne|Bloomington|Peoria|Champaign|Lexington|Knoxville|Chattanooga|Asheville
        |Durham|Wilmington|Greenville|Savannah|Macon|Tallahassee|Gainesville|Sarasota|Pensacola
        |Huntsville|Montgomery|Shreveport|Little Rock|Fayetteville|Roanoke|Charlottesville
        |Norfolk|Frederick|Annapolis|Allentown|Erie|Scranton|Lancaster|Trenton|Albany|Syracuse
        |Ithaca|Hartford|New Haven|Worcester|Burlington|Manchester|Anchorage
        |Juneau|Hilo|Cheyenne|Casper|Morgantown"""),
    "GB": _towns("""York|Exeter|Norwich|Cambridge|Oxford|Brighton|Leicester|Nottingham|Sheffield
        |Bristol|Cardiff|Swansea|Plymouth|Southampton|Portsmouth|Coventry|Leeds|Bradford
        |Sunderland|Aberdeen|Dundee|Inverness|Stirling|Belfast|Carlisle|Salisbury|Winchester
        |Canterbury|Cheltenham|Gloucester|Hereford|Shrewsbury|Bournemouth|Ipswich|Colchester"""),
    "CA": ("Halifax", "Kingston", "Saskatoon", "Winnipeg", "Edmonton", "Kelowna", "Sudbury",
           "Moncton", "Fredericton", "Guelph", "Kamloops", "Thunder Bay", "Sherbrooke"),
    "AU": ("Hobart", "Canberra", "Geelong", "Cairns", "Townsville", "Wollongong", "Ballarat",
           "Bendigo", "Toowoomba", "Launceston", "Mackay"),
    "IN": ("Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
           "Vadodara", "Coimbatore", "Kochi", "Mysore", "Surat", "Chandigarh", "Visakhapatnam"),
    "DE": ("Freiburg", "Heidelberg", "Leipzig", "Dresden", "Bremen", "Hanover", "Nuremberg",
           "Stuttgart", "Bonn", "Mainz", "Kiel", "Rostock", "Augsburg", "Münster"),
    "FR": ("Lyon", "Marseille", "Toulouse", "Nantes", "Strasbourg", "Montpellier", "Bordeaux",
           "Lille", "Rennes", "Grenoble", "Dijon", "Angers", "Reims"),
    "ES": ("Valencia", "Seville", "Bilbao", "Zaragoza", "Malaga", "Granada", "Alicante"),
    "IT": ("Bologna", "Turin", "Naples", "Genoa", "Verona", "Palermo", "Pisa", "Padua"),
    "NL": ("Utrecht", "Rotterdam", "Eindhoven", "Groningen", "Leiden", "Haarlem"),
    "BR": ("Curitiba", "Recife", "Fortaleza", "Manaus", "Belém", "Campinas", "Goiânia"),
    "MX": ("Puebla", "Tijuana", "Mérida", "Querétaro", "Oaxaca", "Morelia"),
    "NG": ("Ibadan", "Kano", "Abuja", "Enugu", "Port Harcourt", "Benin City"),
    "KE": ("Mombasa", "Kisumu", "Nakuru", "Eldoret"),
    "JP": ("Kyoto", "Nagoya", "Sapporo", "Fukuoka", "Kobe", "Sendai", "Hiroshima"),
    "CN": ("Chengdu", "Hangzhou", "Wuhan", "Nanjing", "Suzhou", "Xiamen", "Qingdao"),
}

# words in a message that tell the country of an unknown town
_COUNTRY_CUES: Sequence[Tuple[str, str]] = (
    (r"\b(?:UK|U\.K\.|England|Scotland|Wales|Britain|Northern Ireland)\b", "GB"),
    (r"\bCanada\b|\b(?:Ontario|Quebec|Alberta|Manitoba|Nova Scotia|BC)\b", "CA"),
    (r"\bAustralia\b|\b(?:NSW|Queensland|Victoria|Tasmania)\b", "AU"),
    (r"\bIndia\b", "IN"), (r"\bGermany\b|\bDeutschland\b", "DE"), (r"\bFrance\b", "FR"),
    (r"\bSpain\b|\bEspaña\b", "ES"), (r"\bItaly\b|\bItalia\b", "IT"),
    (r"\bNetherlands\b|\bHolland\b", "NL"), (r"\bBrazil\b|\bBrasil\b", "BR"),
    (r"(?<!New )\bMexico\b|\bMéxico\b", "MX"), (r"\bNigeria\b", "NG"), (r"\bKenya\b", "KE"),
    (r"\bJapan\b", "JP"), (r"\bChina\b", "CN"),
    # the message's language when no country is named
    (r"(?i)\b(?:vivo en|me llamo|mi correo|calle)\b", "ES"),
    (r"(?i)\b(?:j'habite|je m'appelle|je vis à|rue)\b", "FR"),
    (r"(?i)\b(?:ich wohne|ich heiße|straße|strasse)\b", "DE"),
    (r"(?i)\b(?:moro em|meu nome|minha)\b", "BR"),
    (r"(?i)\b(?:abito a|mi chiamo|vivo a)\b", "IT"),
    (r"(?i)\b(?:ik woon|mijn naam)\b", "NL"),
)

# local spellings of towns of other countries ("Sevilla" is Seville)
_LOCAL_NAMES = {
    "sevilla": "ES", "málaga": "ES", "córdoba": "ES", "salamanca": "ES", "san sebastián": "ES",
    "nürnberg": "DE", "hannover": "DE", "köln": "DE", "düsseldorf": "DE",
    "napoli": "IT", "torino": "IT", "genova": "IT", "padova": "IT", "firenze": "IT",
    "den haag": "NL", "lisboa": "BR", "são paulo": "BR",
}

FEATURES: Dict[str, Tuple[str, ...]] = {
    "lake": ("Lake Tahoe", "Lake Placid", "Lake Champlain", "Lake Como", "Lake Louise",
             "Crater Lake", "Flathead Lake", "Lake Powell", "Lake Winnipesaukee", "Lake Garda"),
    "mount": ("Mount Hood", "Mount Rainier", "Mount Shasta", "Mount Washington", "Mount Whitney",
              "Pikes Peak", "Ben Nevis", "Mont Blanc", "Mount Kenya", "Mount Fuji"),
    "river": ("Hudson River", "Colorado River", "Columbia River", "Snake River", "Potomac River",
              "Willamette River", "Delaware River", "River Severn", "River Tyne"),
    "valley": ("Napa Valley", "Hudson Valley", "Shenandoah Valley", "Willamette Valley",
               "Coachella Valley", "Ohio Valley"),
    "park": ("Yosemite National Park", "Zion National Park", "Acadia National Park",
             "Glacier National Park", "Olympic National Park", "Peak District National Park"),
    "beach": ("Myrtle Beach", "Daytona Beach", "Venice Beach", "Huntington Beach",
              "Clearwater Beach"),
    "island": ("Mackinac Island", "Martha's Vineyard", "Nantucket", "Isle of Skye",
               "Isle of Wight", "Whidbey Island"),
    "county": ("Orange County", "Maricopa County", "King County", "Travis County",
               "Fairfax County", "Westchester County", "Marin County", "Dane County"),
    "region": ("Pacific Northwest", "New England", "Hill Country", "the Cotswolds",
               "Lake District", "Scottish Highlands", "Hudson Valley", "the Ozarks",
               "the Berkshires", "the Poconos", "Cape Cod", "Outer Banks", "Big Sur",
               "Upper Peninsula"),
}
_FEATURE_WORDS = (
    ("lake", r"\blake\b|\bloch\b"), ("mount", r"\b(?:mount|mt\.?|peak|mountains?|ben)\b"),
    ("river", r"\briver\b"), ("valley", r"\bvalley\b"), ("park", r"\bpark\b"),
    ("beach", r"\bbeach\b"), ("island", r"\b(?:island|isle)s?\b"),
    ("county", r"\b(?:county|parish|shire)\b"),
)

_TOWN_COUNTRY = {t.casefold(): c for c, towns in TOWNS.items() for t in towns}
for _towns in TOWNS.values():
    assert not {t.casefold() for t in _towns} & GEO_PASS_THROUGH, "a surrogate town is a major city"

TOWN_NAMES = frozenset(_TOWN_COUNTRY)


def is_known_place(name: str) -> bool:
    """A US state, a listed town or a major city or country."""
    cf = name.casefold()
    return cf in US_STATES or cf in TOWN_NAMES or cf in GEO_PASS_THROUGH


_STATE_NAMES = tuple(sorted(s.title() for s in US_STATES))
_STATE_ABBRS = tuple(sorted(US_STATE_ABBREVS))


def _case_like(model: str, value: str) -> str:
    if model.isupper() and len(model) > 2:
        return value.upper()
    if model.islower():
        return value.lower()
    return value


def country_of(original: str, context: str = "") -> str:
    known = (_TOWN_COUNTRY.get(original.strip().casefold())
             or _LOCAL_NAMES.get(original.strip().casefold()))
    if known:
        return known
    rest = context.replace(original, " ") if original else context
    for pat, code in _COUNTRY_CUES:
        if re.search(pat, rest):
            return code
    return "US"


def _zh_place(text: str, choose: Callable[[Sequence[str]], str],
              avoid: Callable[[str], bool]) -> Optional[str]:
    """A Chinese city for a city, a district for a district ("南山区" ->
    "永川区"), from Faker's zh_CN lists; the suffix is kept."""
    from faker.providers.address.zh_CN import Provider as A
    suffix = text[-1] if text[-1] in "区县市镇" else ""
    pool = A.districts if suffix in ("区", "县", "镇") else A.cities
    cands = [p + suffix for p in pool if p + suffix != text and not avoid(p + suffix)]
    return choose(sorted(cands)) if cands else None


def real_place(original: str, context: str, choose: Callable[[Sequence[str]], str],
               avoid: Callable[[str], bool]) -> Optional[str]:
    """A real place standing in for *original*, or None when every
    candidate is excluded. *choose* picks from a sequence (the caller's
    seeded RNG); *avoid* rejects a candidate (an original, a surrogate in
    use, a word of the message)."""
    text = original.strip()
    cf = text.casefold().rstrip(".")
    if re.search(r"[\u4e00-\u9fff]", text):
        return _zh_place(text, choose, avoid)
    if text.upper() in US_STATE_ABBREVS and len(text) == 2:
        pool: Sequence[str] = [s for s in _STATE_ABBRS if s != text.upper()]
    elif cf in US_STATES:
        pool = [s for s in _STATE_NAMES if s.casefold() != cf]
    else:
        kind = next((k for k, pat in _FEATURE_WORDS if re.search(pat, cf)), None)
        if kind is not None:
            pool = FEATURES[kind]
        elif len(text.split()) >= 3 and kind is None and not text.istitle():
            pool = FEATURES["region"]
        else:
            pool = TOWNS[country_of(text, context)]
    cands = [p for p in pool if p.casefold() != cf and not avoid(p)]
    if not cands:
        cands = [t for towns in TOWNS.values() for t in towns if t.casefold() != cf and not avoid(t)]
    if not cands:
        return None
    return _case_like(text, choose(cands))
