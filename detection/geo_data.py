# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
detection/geo_data.py — Geographic Pass-Through Whitelist

Defines geographic entities that are NEVER replaced by SurrogateShield,
regardless of query type or detection pipeline settings.

Rationale (k-anonymity framing):
  A US state has a minimum population of ~580,000 (Wyoming).  A country has
  tens of millions.  A major city has hundreds of thousands to millions of
  residents.  None of these alone provide a k-anonymity set small enough to
  constitute PII under any reasonable threat model.

  Replacing them:
    • Destroys answer utility (Claude cannot answer "tax benefits of Websterstad")
    • Provides essentially zero privacy gain (state-level granularity is public)
    • Contradicts the About document: "Phoenix is not identifying"

  This is different from specific addresses (1126 E Apache Blvd) or small towns
  combined with other quasi-identifiers, which ARE worth protecting.

Imported by:
  detection/entity_trace.py
  detection/context_guard.py
  detection/logic.py
"""

from __future__ import annotations

# ── US States (full names, lowercase) ────────────────────────────────────────
US_STATES: frozenset = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
    "kentucky", "louisiana", "maine", "maryland", "massachusetts",
    "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia", "puerto rico", "guam",
})

# ── Major countries (full names + common abbreviations, lowercase) ────────────
MAJOR_COUNTRIES: frozenset = frozenset({
    "united states", "united states of america", "usa", "us",
    "united kingdom", "uk", "great britain", "britain", "england",
    "scotland", "wales", "northern ireland",
    "canada", "australia", "new zealand",
    "germany", "france", "italy", "spain", "portugal",
    "netherlands", "belgium", "switzerland", "austria",
    "sweden", "norway", "denmark", "finland", "iceland",
    "russia", "ukraine", "poland", "czech republic", "hungary",
    "romania", "greece", "turkey",
    "japan", "china", "south korea", "north korea", "taiwan",
    "india", "pakistan", "bangladesh", "sri lanka", "nepal",
    "indonesia", "malaysia", "thailand", "vietnam", "philippines",
    "singapore", "myanmar", "cambodia",
    "brazil", "argentina", "colombia", "chile", "peru", "venezuela",
    "mexico", "cuba", "jamaica",
    "nigeria", "south africa", "kenya", "ghana", "ethiopia", "egypt",
    "morocco", "algeria", "tanzania", "uganda",
    "saudi arabia", "uae", "israel", "iran", "iraq", "jordan",
    "lebanon", "qatar", "kuwait", "bahrain", "oman",
})

# ── Major world cities with population > ~500k (lowercase) ───────────────────
# Includes well-known US cities, world capitals, and other major metros.
# Does NOT include small towns — those are legitimately quasi-identifying.
MAJOR_CITIES: frozenset = frozenset({
    # US — major metros
    "new york", "new york city", "nyc",
    "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin",
    "jacksonville", "fort worth", "columbus", "charlotte", "indianapolis",
    "san francisco", "seattle", "denver", "washington", "nashville",
    "oklahoma city", "el paso", "boston", "portland", "las vegas",
    "memphis", "louisville", "baltimore", "milwaukee", "albuquerque",
    "tucson", "fresno", "sacramento", "kansas city", "mesa",
    "atlanta", "omaha", "colorado springs", "raleigh", "virginia beach",
    "long beach", "minneapolis", "tampa", "honolulu", "miami",
    "orlando", "pittsburgh", "cincinnati", "st. louis", "salt lake city",
    "richmond", "baton rouge", "birmingham",
    # World capitals and major cities
    "london", "paris", "berlin", "madrid", "rome", "amsterdam",
    "brussels", "vienna", "stockholm", "oslo", "copenhagen", "helsinki",
    "zurich", "geneva",
    "moscow", "st. petersburg", "kyiv", "warsaw", "prague", "budapest",
    "bucharest", "athens", "istanbul",
    "beijing", "shanghai", "tokyo", "osaka", "seoul", "taipei",
    "hong kong", "singapore", "bangkok", "jakarta", "manila",
    "mumbai", "delhi", "new delhi", "bangalore", "kolkata", "chennai",
    "karachi", "lahore", "dhaka",
    "sydney", "melbourne", "brisbane", "perth", "auckland",
    "toronto", "vancouver", "montreal", "ottawa", "calgary",
    "mexico city", "guadalajara", "monterrey",
    "sao paulo", "rio de janeiro", "buenos aires", "bogota",
    "lima", "santiago", "caracas",
    "cairo", "lagos", "kinshasa", "johannesburg", "cape town",
    "nairobi", "casablanca",
    "dubai", "abu dhabi", "riyadh", "tel aviv", "jerusalem",
    "tehran", "baghdad", "beirut",
})

# ── Combined whitelist ─────────────────────────────────────────────────────────
# All geographic entities that should NEVER be replaced regardless of context.
GEO_PASS_THROUGH: frozenset = US_STATES | MAJOR_COUNTRIES | MAJOR_CITIES