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

# ── US state / territory USPS abbreviations (uppercase) ──────────────────────
# Used by the address parser: a 2-letter token is only treated as a state when
# it appears UPPERCASE and is in this set, so English words like "or", "in",
# "me", "hi", "ok" can never be mistaken for Oregon, Indiana, Maine, …
US_STATE_ABBREVS: frozenset = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI", "AS", "MP",
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

# ── Countries and territories (lowercase) ─────────────────────────────────────
# Every country, territory and common short form ("russia", "uk"). Detection
# reads a place that is one of these, or a US state, as no identifier
# (``pipeline._misreads``, ``nation``); generation keeps to GEO_PASS_THROUGH.
COUNTRIES: frozenset = frozenset({
    "afghanistan", "albania", "algeria", "american samoa", "andorra", "angola", "anguilla",
    "antarctica", "antigua and barbuda", "argentina", "armenia", "aruba", "australia", "austria",
    "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belgium", "belize",
    "benin", "bermuda", "bhutan", "bolivia", "bosnia and herzegovina", "botswana", "bouvet island",
    "bouvetoya", "brazil", "britain", "british indian ocean territory", "british virgin islands",
    "brunei", "brunei darussalam", "bulgaria", "burkina faso", "burma", "burundi", "cabo verde",
    "cambodia", "cameroon", "canada", "cape verde", "cayman islands", "central african republic",
    "chad", "chagos archipelago", "chile", "china", "christmas island", "cocos islands", "colombia",
    "comoros", "congo", "cook islands", "costa rica", "cote d'ivoire", "croatia", "cuba", "cyprus",
    "czech republic", "czechia", "democratic republic of the congo", "denmark", "djibouti",
    "dominica", "dominican republic", "dr congo", "drc", "east timor", "ecuador", "egypt",
    "el salvador", "england", "equatorial guinea", "eritrea", "estonia", "eswatini", "ethiopia",
    "falkland islands", "faroe islands", "fiji", "finland", "france", "french guiana",
    "french polynesia", "french southern territories", "gabon", "gambia", "georgia", "germany",
    "ghana", "gibraltar", "great britain", "greece", "greenland", "grenada", "guadeloupe", "guam",
    "guatemala", "guernsey", "guinea", "guinea-bissau", "guyana", "haiti",
    "heard island and mcdonald islands", "holland", "holy see", "honduras", "hong kong", "hungary",
    "iceland", "india", "indonesia", "iran", "iraq", "ireland", "isle of man", "israel", "italy",
    "ivory coast", "jamaica", "japan", "jersey", "jordan", "kazakhstan", "keeling", "kenya",
    "kiribati", "korea", "kosovo", "kuwait", "kyrgyz republic", "lao people's democratic republic",
    "laos", "latvia", "lebanon", "lesotho", "liberia", "libya", "libyan arab jamahiriya",
    "liechtenstein", "lithuania", "luxembourg", "macao", "macau", "macedonia", "madagascar",
    "malawi", "malaysia", "maldives", "mali", "malta", "malvinas", "marshall islands", "martinique",
    "mauritania", "mauritius", "mayotte", "mexico", "micronesia", "moldova", "monaco", "mongolia",
    "montenegro", "montserrat", "morocco", "mozambique", "myanmar", "namibia", "nauru", "nepal",
    "netherlands", "netherlands antilles", "new caledonia", "new zealand", "nicaragua", "niger",
    "nigeria", "niue", "norfolk island", "north korea", "north macedonia", "northern ireland",
    "northern mariana islands", "norway", "oman", "pakistan", "palau", "palestine",
    "palestinian territory", "panama", "papua new guinea", "paraguay", "peru", "philippines",
    "pitcairn islands", "poland", "portugal", "puerto rico", "qatar", "republic of the congo",
    "reunion", "romania", "russia", "russian federation", "rwanda", "saint barthelemy",
    "saint helena", "saint kitts and nevis", "saint lucia", "saint martin",
    "saint pierre and miquelon", "saint vincent and the grenadines", "samoa", "san marino",
    "sao tome and principe", "saudi arabia", "scotland", "senegal", "serbia", "seychelles",
    "sierra leone", "singapore", "slovak republic", "slovakia", "slovenia", "solomon islands",
    "somalia", "south africa", "south georgia and the south sandwich islands", "south korea",
    "south sudan", "spain", "sri lanka", "sudan", "suriname", "svalbard & jan mayen islands",
    "swaziland", "sweden", "switzerland", "syria", "syrian arab republic", "taiwan", "tajikistan",
    "tanzania", "thailand", "timor-leste", "togo", "tokelau", "tonga", "trinidad and tobago",
    "tunisia", "turkey", "turkiye", "turkmenistan", "turks and caicos islands", "tuvalu", "uae",
    "uganda", "uk", "ukraine", "united arab emirates", "united kingdom", "united states",
    "united states minor outlying islands", "united states of america",
    "united states virgin islands", "uruguay", "usa", "uzbekistan", "vanuatu", "vatican",
    "vatican city", "vatican city state", "venezuela", "vietnam", "wales", "wallis and futuna",
    "western sahara", "yemen", "zambia", "zimbabwe",
})

# ── Regions larger than a country (lowercase) ────────────────────────────────
REGIONS: frozenset = frozenset({
    "america", "americas", "north america", "south america", "central america", "latin america",
    "europe", "western europe", "eastern europe", "northern europe", "southern europe", "scandinavia",
    "balkans", "asia", "east asia", "south asia", "southeast asia", "central asia", "middle east",
    "africa", "north africa", "west africa", "east africa", "southern africa", "sub-saharan africa",
    "oceania", "caribbean", "european union", "eu",
})
