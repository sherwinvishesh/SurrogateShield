#!/usr/bin/env python3
# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
make_dataset.py — the generated PII evaluation set (audit B.8, gate J11).

Writes, for each split (``dev``, ``test``):

  experiment/synth_<split>.json      runner input: ``[{"input": …}, …]``
  experiment/synth_<split>_key.json  key: ``[{"Question", "Answer-Key", "Spans",
                                     "Keep", "id", "identity_id", "gender",
                                     "lang", "template"}, …]``
  experiment/synth_manifest.json     generator version, seed, n, split sizes

``Spans`` holds every gold value with its character offsets; the evaluators
score on them exactly (``eval_metrics.entry_gold``). ``Keep`` lists the
distractors written into the text on purpose (order numbers, non-birth dates,
versions, public figures …): replacing one is a false positive.

Properties, all checked by ``experiment/lint_key.py``:

* Every row has its own identity: first and last name, gender and pronouns,
  e-mail built from the name, phone, DOB, SSN, address, employer, card, …
  No identity and no identifying value occurs in two rows (B1).
* Every gold value sits at its offsets, and every occurrence of it is
  annotated (B2, B4).
* 25 % of rows are PII-free, most of them with numeric, date, acronym or
  public-name distractors, some non-English (B6).
* dev and test use disjoint templates and disjoint name pools, so test
  measures text the detector was not tuned on. The test split is scored once
  per release.

Deterministic: the same ``--seed`` and ``--n`` give byte-identical files.
Standard library only.

Usage:
    python experiment/make_dataset.py                 # --n 1500 --seed 7
    python experiment/make_dataset.py --out /tmp/x    # write elsewhere
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

GENERATOR_VERSION = 1
TEST_FRACTION = 0.2
NEG_FRACTION = 0.25

# ── name pools (index parity splits them between dev and test) ─────────────

MALE = """James Daniel Rahul Wei Omar Luca Ethan Mateo Arjun Tomasz Kwame Yusuf Hiroshi
Diego Andrei Samuel Karim Felipe Nikolai Tariq Jonas Emeka Haruto Ravi Pedro Malik
Liam Stefan Kofi Aarav Bilal Henrik Joaquin Dmitri Chidi Kenji Anders Ibrahim Marco
Tobias Sanjay Rafael Olumide Jae-won Niklas Vikram Hamza Florian Thabo Arash Benedikt
Cormac Eitan Gustavo Hyun-woo Ismail Jiro Lorenzo Mbeki Nasser Oskar Pranav Quentin""".split()
FEMALE = """Sarah Priya Elena Aisha Mei Chloe Fatima Nora Zoe Ana Amara Yuki Ingrid Lucia
Ji-woo Hannah Leila Camila Svetlana Ngozi Freya Ananya Mariam Isabel Akiko Olga Zainab
Clara Thandiwe Sofia Meera Rania Beatriz Katarzyna Adaeze Hana Astrid Noor Valentina
Kavya Yasmin Greta Folake Sakura Daniela Ines Bongani Esperanza Farah Gabriela Helga
Imani Jasleen Keiko Larisa Maite Nadia Oluwaseun Paloma Rosalind Saoirse Tamar Ulrike""".split()
NEUTRAL = """Alex Jordan Riley Sam Robin Kai Noa Avery Rowan Sasha Quinn Emery""".split()
LAST = """Mitchell Patel Rossi Khan Chen Dubois Nakamura Garcia Okafor Novak Diop Mansoor
Kovalev Ivanova Park Nguyen Schmidt Oliveira Haddad Lindqvist Mensah Kowalski Tanaka
Fernandes Abara Moreau Bianchi Petrov Achebe Svensson Reyes Kapoor Yilmaz Hoffmann
Castillo Adeyemi Morales Johansson Sato Jovanovic Bakker Mahlangu Rahman Fischer
Alvarez Osei Lindgren Ferrari Nowak Watanabe Iyer Chowdhury Eriksen Delgado Ogunleye
Kaplan Varga Horvat Lambert Brennan O'Neill MacLeod D'Souza Van der Berg Al-Sayed
Kim Lee Singh Ward Torres Quinn Becker Ruiz Pham Sharma Nilsen Romero Halvorsen
Gallagher Mwangi Banerjee Esposito Laurent Hartmann Wojcik Takahashi Ncube Sorensen""".replace(
    "Van der Berg", "Van_der_Berg").replace("Al-Sayed", "Al-Sayed").split()
LAST = [n.replace("_", " ") for n in LAST]

FREE_DOMAINS = ["gmail.com", "outlook.com", "proton.me", "yahoo.com", "icloud.com", "hotmail.com"]
WORK_DOMAINS = ["brightwater.io", "kestrel-analytics.com", "northfieldlaw.co.uk", "veltra.de",
                "tidepoolhealth.org", "copperleaf.in", "marlowe-studio.fr", "orbitalfreight.com",
                "sunhollow.edu", "pinegrove-clinic.org", "quayside.nl", "lumenbank.es"]

CITIES = """Tempe Bozeman Burlington Naperville Leeds Eugene Asheville Fresno Durham Salem
Boulder Flagstaff Ithaca Missoula Spokane Tacoma Dayton Lansing Pueblo Provo Galway Utrecht
Leipzig Bergamo Tartu Aarhus Ghent Lyon Graz Porto Kraków Pune Mysuru Kumasi Ibadan Arusha
Cuenca Rosario Valparaíso Puebla Sapporo Daegu Hamilton Ballarat Halifax Kelowna Moncton
Reading Norwich Exeter Dundee Cork Bremen Malmö Tampere Brno Split Coimbra Bilbao""".split()
STREETS = ["Oak Avenue", "Maple Street", "Willow Lane", "Cedar Court", "Pine Road",
           "Foxglove Ct", "Juniper Drive", "Birchwood Way", "Harbor View Rd", "Elm Terrace",
           "Laurel Street", "Sycamore Blvd", "Riverbend Drive", "Hawthorn Close",
           "Quarry Lane", "Linden Avenue", "Aspen Trail", "Magnolia Pl", "Kingsley Road",
           "Orchard Street"]
ORGS = ["Brightwater Logistics", "Kestrel Analytics", "Northfield Dental Group",
        "Tidepool Health Clinic", "Copperleaf Software", "Marlowe Design Studio",
        "Orbital Freight", "Sunhollow Community College", "Pinegrove Family Clinic",
        "Quayside Accounting", "Lumen Credit Union", "Hollis Middle School",
        "Redfern Physiotherapy", "Ashgrove Veterinary Hospital", "Calder & Finch LLP",
        "Meadowlark Daycare", "Ironbridge Fitness", "St. Brigid's Primary School",
        "Westbrook Pharmacy", "Saltmarsh Brewing Co"]

# ── distractors and public names (never gold) ───────────────────────────────

PUBLIC_PEOPLE = ["Taylor Swift", "Barack Obama", "Marie Curie", "Lionel Messi", "Elon Musk",
                 "Serena Williams", "Albert Einstein", "Frida Kahlo", "Napoleon Bonaparte",
                 "Ada Lovelace", "Nelson Mandela", "Leonardo da Vinci"]
PUBLIC_ORGS = ["Google", "Tesla", "Microsoft", "Pfizer", "Toyota", "Netflix", "Amazon",
               "Deloitte", "Spotify", "Nvidia", "the WHO", "NASA"]
PUBLIC_PLACES = ["Rome", "Japan", "the Amazon rainforest", "Lake Baikal", "Kyoto", "Iceland",
                 "the Sahara", "Patagonia", "Delaware", "Wyoming", "Lisbon", "Singapore"]
PRODUCTS = ["iPhone 15", "Galaxy S23", "ThinkPad X1", "PlayStation 5", "Kindle Paperwhite",
            "MacBook Air", "Pixel 8", "Surface Pro 9"]
ACRONYMS = ["HR", "PTO", "KPI", "OKR", "SLA", "EOD", "ETA", "CRM", "SQL", "API", "GDPR", "ROI",
            "HIPAA", "VPN", "DNS", "FAQ"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# ── templates ────────────────────────────────────────────────────────────────
# {field} of SPAN_LABEL is a gold span; other fields are context (pronouns,
# distractors). "lang" defaults to en.

SPAN_LABEL = {
    "name": "name", "name_lower": "name", "name_upper": "name", "first": "name",
    "first_lower": "name", "last": "name", "email": "email", "phone": "phone",
    "ssn": "ssn", "dob": "dob", "street": "address", "city": "gpe", "city_lower": "gpe",
    "zip": "zip", "org": "org", "card": "credit_card", "ip": "ip_address", "api": "api_key",
    "age": "age", "iban": "iban", "handle": "username", "url": "url",
}

POS: List[Tuple[str, str]] = [
    ("en", "Hi, my name is {name} and my email is {email}; please draft a resignation letter for me."),
    ("en", "Write a reminder to {name} at {phone} that {his} dentist appointment is on {date} at 3pm."),
    ("en", "My SSN is {ssn} and my date of birth is {dob}. Can I still claim the earned income credit?"),
    ("en", "I live at {street}, {city} {zip}. What should I bring to the DMV?"),
    ("en", "{name} works at {org} in {city}; email {him} at {email} about the Q3 numbers before Friday."),
    ("en", "Compose a cover letter for {name} ({email}, {phone}) applying to {org} as a data analyst."),
    ("en", "Our client {name} was born on {dob} and {he} moved to {city} last year. Draft a welcome note."),
    ("en", "Please text {phone} and ask {name} whether {he} still lives on {street}."),
    ("en", "Verify that the SSN on file for {name}, {ssn}, matches the W-2 from {org}."),
    ("en", "The invoice should go to {email}; {first} prefers a call on {phone} after 5pm."),
    ("en", "{title} {last} asked me to update {his} card {card}. Write the confirmation for order {order}."),
    ("en", "Our server at {ip} is rejecting the key {api}. What is wrong with this curl command?"),
    ("en", "I'm {name}, I just turned {age} and live in {city}. Suggest a birthday itinerary."),
    ("en", "Find a pharmacy near {street}, {city} that is open now."),
    ("en", "Add {name} <{email}> to the newsletter and note {his} postcode is {zip}."),
    ("en", "hey its {name_lower}, can u remind me what {org} pays for overtime? my number is {phone}"),
    ("en", "URGENT: {name_upper} ({phone}) DID NOT RECEIVE THE REFUND FOR ORDER {order}."),
    ("en", "Thanks,\n{name}\n{org}\n{phone} | {email}"),
    ("en", "From: {name} <{email}>\nSubject: Re: invoice {order}\n\nHi, the total of {price} looks wrong."),
    ("en", '{{"customer": "{name}", "email": "{email}", "phone": "{phone}", "plan": "premium"}}'),
    ("en", "name,email,city\n{name},{email},{city}"),
    ("en", "| Field | Value |\n|---|---|\n| Name | {name} |\n| DOB | {dob} |\n| SSN | {ssn} |"),
    ("en", "My {child} {first} is {age} and goes to {org}. How do I request a school transfer?"),
    ("en", "Can you check whether IBAN {iban} belongs to {name}? I need to send {price} by {date}."),
    ("en", "Follow {handle} on Instagram, that's {first}'s account; {his} portfolio is at {url}."),
    ("en", "{first} {last} said {he} can't make the {date} meeting. Reply politely for me."),
    ("en", "Is it legal for my landlord to keep my deposit? Lease is in my name, {name}, at {street}."),
    ("en", "I was born {dob} and I'm {age}. Which retirement accounts can I open with {org}?"),
    ("en", "Draft an email to HR: {name} (employee since {year}) requests PTO from {date}."),
    ("en", "Login fails for user {handle} from {ip}. Error 403 after upgrading to version {version}."),
    ("en", "Please call {first} back on {phone}, {he} asked about the {product} warranty."),
    ("en", "Write a recommendation letter for {name}, who interned with us at {org} in {city}."),
    ("en", "The card ending {card} was charged twice for {price}. Dispute it for {name}."),
    ("en", "I'm a nurse at {org}. Patient {name}, DOB {dob}, needs a follow-up on {date}."),
    ("en", "{name}'s new address is {street}, {city} {zip}; update the CRM please."),
    ("en", "export API_KEY={api}\n# owner: {email}"),
    ("en", "Translate to Spanish: 'My name is {name} and I live in {city}.'"),
    ("en", "my ssn {ssn} got flagged, irs says dob {dob} doesnt match. help"),
    ("en", "Personal site: {url} — contact {email} or {phone} for commissions."),
    ("en", "Set up a payment of {price} to {iban} (account holder {name}) on {date}."),
    ("en", "My {relative} {first} turns {age} next week, {he} lives in {city}. Gift ideas under {price}?"),
    ("en", "Reset the password for {email}; the request came from {ip} at 02:14."),
    ("en", "Hey {first}! It was great meeting you at {org}. My cell is {phone} if you need anything."),
    ("en", "Applicant: {name}\nDate of birth: {dob}\nSSN: {ssn}\nAddress: {street}, {city} {zip}"),
    ("en", "{first} says {his} Discord is {handle}; add {him} to the server before {date}."),
    ("en", "Can {org} fire {name} for taking sick leave? {He} has worked there since {year}."),
    ("en", "Summarize this ticket: customer {name} ({email}) reports {product} overheating, ticket {ticket}."),
    ("en", "Book a table for 4 under {last}, phone {phone}, {date} at 7pm."),
    ("en", "We need the wire to {name}, IBAN {iban}, before the {date} deadline."),
    ("en", "Our intern {first} {last} ({age}) starts {date}. Draft the onboarding checklist."),
    ("en", "curl -H 'Authorization: Bearer {api}' https://api.example.com/v2/users  # from {ip}"),
    ("en", "Meet {name} outside {org} in {city}; {his} number is {phone}."),
    ("en", "Update shipping for order {order}: {street}, {city} {zip}, recipient {name}."),
    ("en", "I am {age} years old with a DOB of {dob}; does my insurance at {org} cover physio?"),
    ("en", "{name_upper}\n{street}\n{city} {zip}"),
    ("en", "is {email} still {first_lower}'s address or should i use {url}?"),
    ("es", "Hola, soy {name} y vivo en {city}. Mi correo es {email}, ¿me ayudas con una carta?"),
    ("de", "Bitte schreiben Sie an {name} ({email}), die Telefonnummer ist {phone}."),
    ("fr", "Je m'appelle {name}, je suis {ne} le {dob} et j'habite à {city}."),
    ("pt", "Meu nome é {name}, meu telefone é {phone} e trabalho na {org}."),
    ("it", "Ciao, sono {name} di {city}; il mio IBAN è {iban}."),
    ("en", "Can you proofread this bio? '{name} is a {age}-year-old designer at {org} based in {city}.'"),
    ("en", "Our VPN logs show {handle} connecting from {ip} at 03:12; is that suspicious?"),
    ("en", "Please verify payroll: {name}, SSN {ssn}, card {card}, start date {date}."),
    ("en", "Note to self: call {first} ({phone}) about the {product} trade-in."),
    ("en", "{He} said {his} name is {name} and {his} email is {email}; is this a phishing attempt?"),
    ("en", "my gamertag is {handle} lol, add me. irl im {first_lower} from {city_lower}"),
    ("en", "Who is {handle}? They keep tagging {name} in posts about {org}."),
    ("en", "Lost my phone. Old number {phone}, account email {email}, username {handle}."),
    ("en", "Profile: {url}\nTwitter: {handle}\nLocation: {city}"),
    ("en", "Dear {title} {last}, your appointment at {org} on {date} is confirmed. Reply to {email}."),
    ("en", "{name} ({age}) from {city} won the {org} raffle; announce it without the prize amount."),
    ("de", "Mein Name ist {name}, ich wohne in der {street} in {city}. Meine IBAN: {iban}."),
    ("es", "{first} tiene {age} años y estudia en {org}; mi teléfono es {phone}."),
    ("fr", "Envoyez la facture {order} à {name}, {email}, avant le {date}."),
    ("en", "SSH from {ip} as {handle} failed: Permission denied (publickey). Key owner {email}."),
    ("en", "The DOB on {first}'s passport says {dob} but {his} visa says otherwise. What now?"),
    ("en", "Transfer {price} from card {card} to {name}; memo: rent for {street}."),
]

NEG: List[Tuple[str, str]] = [
    ("en", "Our Q3 revenue was {n5} units and we shipped {n4} orders before {date}."),
    ("en", "The build failed on {iso_date} with error code {n10} after {small} hours at the gate."),
    ("en", "Set the timeout to {n5} milliseconds and retry {small} times; the deadline is {slash_date}."),
    ("en", "Upgrade the firmware to version {version} and reboot the router."),
    ("en", "Please finish ticket {ticket} and order {order} by Friday."),
    ("en", "A software company usually needs a marketing team and a consulting firm for audits."),
    ("en", "The policy number is {n9} and the claim total is {price}."),
    ("en", "Explain how photosynthesis works and why leaves are green."),
    ("en", "What is the difference between a list and a tuple in Python?"),
    ("en", "Summarise the plot of a novel about a lighthouse keeper in winter."),
    ("en", "The team of {small} engineers shipped {small} releases in {year} and {small} in {year2}."),
    ("en", "Annual report: {n5} units sold, {n4} returned, net {n5b}."),
    ("en", "Give me the tax benefits of incorporating in Wyoming versus Delaware."),
    ("en", "What restaurants are typical in {pplace} and what wine goes with them?"),
    ("en", "What did {pperson} say about climate change in {year}?"),
    ("en", "Is {porg} stock a good buy after the {date} earnings call?"),
    ("en", "Compare the {product} and the {product2} for battery life."),
    ("en", "Our {acr} review moved to {date}; update the {acr2} dashboard and the {acr3} doc."),
    ("en", "Convert {n3} km to miles and {n3b} kg to pounds."),
    ("en", "Recipe: {small} eggs, {n3} g flour, {n3b} ml milk; bake at 180 C for {small} minutes."),
    ("en", "Write a haiku about {pplace} in autumn."),
    ("en", "Why did {pperson} become famous, in three sentences?"),
    ("en", "for i in range({n3}): total += prices[i] * qty_{small}  # why is this slow?"),
    ("en", "SELECT id, created_at FROM orders WHERE total > {n4} AND status = 'SHIPPED';"),
    ("en", "Meeting moved from {slash_date} to {iso_date}, room {n3}, bring the {acr} slides."),
    ("en", "How many calories are in {n3} grams of rice?"),
    ("en", "Our support line handles {n5} calls a month; how do we cut wait times below {small} minutes?"),
    ("en", "Tracking {order} says delivered on {date} but nothing arrived."),
    ("en", "What's the population of {pplace} and how has it changed since {year}?"),
    ("en", "Explain {acr} vs {acr2} for a small business owner."),
    ("en", "The {product} costs {price}; with a {small}% discount what do I pay?"),
    ("en", "Which is older, {pperson} or {pperson2}?"),
    ("en", "Invoice {order} totals {price}, due {date}; net {small}0 terms."),
    ("en", "Write a limerick about a cat who learns SQL."),
    ("es", "¿Cuáles son las ventajas fiscales de una sociedad limitada en {year}?"),
    ("de", "Wie funktioniert die Photosynthese bei Pflanzen?"),
    ("fr", "Quels sont les meilleurs musées de {pplace} pour les enfants ?"),
    ("pt", "Qual é a diferença entre juros simples e compostos?"),
    ("it", "Come si prepara un buon risotto ai funghi?"),
    ("en", "Grow {small} tomato plants in {small}0 L pots: how much water per day?"),
    ("en", "The {acr} team closed {n4} tickets in {year}; ticket {ticket} is still open."),
    ("es", "¿Qué museos de {pplace} recomiendas para un fin de semana?"),
    ("de", "Bestellung {order} kostet {price}, Lieferung am {iso_date}."),
    ("en", "Error {n10}: connection reset after {n5} ms (build {version})."),
    ("fr", "Combien de calories dans {n3} grammes de pâtes ?"),
    ("en", "Did {pperson} ever visit {pplace}? Answer in two sentences."),
    ("pt", "O pedido {order} chegou em {slash_date}, mas faltam {small} itens."),
    ("en", "Rank {porg}, {pperson} biographies, and {product} reviews by popularity."),
]

TITLES = {"m": ["Mr."], "f": ["Ms.", "Mrs."], "x": ["Mx."]}
PRONOUNS = {"m": ("he", "him", "his"), "f": ("she", "her", "her"), "x": ("they", "them", "their")}
FAMILY = {"m": ("son", "dad", "né"), "f": ("daughter", "mom", "née"), "x": ("kid", "parent", "né·e")}


# ── value makers ─────────────────────────────────────────────────────────────

def _luhn(prefix: str) -> str:
    total = 0
    for i, d in enumerate(reversed(prefix + "0")):
        n = int(d)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return prefix + str((10 - total % 10) % 10)


def _iban(rng: random.Random) -> str:
    country, length = rng.choice([("DE", 18), ("GB", 18), ("FR", 23), ("ES", 20), ("NL", 14)])
    if country == "GB":
        bban = "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + \
            "".join(rng.choice(string.digits) for _ in range(14))
    elif country == "NL":
        bban = "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + \
            "".join(rng.choice(string.digits) for _ in range(10))
    else:
        bban = "".join(rng.choice(string.digits) for _ in range(length))
    digits = "".join(str(int(c, 36)) for c in bban + country + "00")
    check = 98 - int(digits) % 97
    iban = f"{country}{check:02d}{bban}"
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4)) if rng.random() < 0.5 else iban


def _phone(rng: random.Random) -> str:
    a, e, l = rng.randint(201, 989), rng.randint(200, 999), rng.randint(1000, 9999)
    while a % 100 == 11 or e % 100 == 11:
        a, e = rng.randint(201, 989), rng.randint(200, 999)
    return rng.choice([
        f"({a}) {e}-{l}", f"{a}-{e}-{l}", f"+1-{a}-{e}-{l}", f"{a}.{e}.{l}",
        f"+44 7{rng.randint(100, 999)} {rng.randint(100000, 999999)}",
        f"+91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)}",
        f"+49 30 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
        f"+33 6 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
    ])


def _dob(rng: random.Random) -> Tuple[str, int]:
    y, m, d = rng.randint(1948, 2006), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}",
                       f"{d} {MONTHS[m - 1]} {y}", f"{d:02d}.{m:02d}.{y}"]), y


def _ssn(rng: random.Random) -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def _api(rng: random.Random) -> str:
    alnum = string.ascii_letters + string.digits
    return rng.choice([
        "sk-proj-" + "".join(rng.choice(alnum) for _ in range(40)),
        "ghp_" + "".join(rng.choice(alnum) for _ in range(36)),
        "AKIA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16)),
        "xoxb-" + "-".join("".join(rng.choice(string.digits) for _ in range(12)) for _ in range(2))
        + "-" + "".join(rng.choice(alnum) for _ in range(24)),
    ])


class Identities:
    """Identities with unique names and values across both splits (B1)."""

    def __init__(self) -> None:
        self.used: set = set()
        self.next_id = 0

    def _fresh(self, make):
        for _ in range(1000):
            v = make()
            if v.casefold() not in self.used:
                self.used.add(v.casefold())
                return v
        raise RuntimeError("value pool exhausted")

    def make(self, rng: random.Random, part: int) -> Dict[str, str]:
        g = rng.choices(["m", "f", "x"], weights=[47, 47, 6])[0]
        firsts = {"m": MALE, "f": FEMALE, "x": NEUTRAL}[g]
        firsts = [n for i, n in enumerate(firsts) if i % 2 == part] or firsts
        lasts = [n for i, n in enumerate(LAST) if i % 2 == part]
        first, last = self._fresh_name(rng, firsts, lasts)
        he, him, his = PRONOUNS[g]
        dob, year = _dob(rng)
        local = rng.choice(["{f}.{l}", "{f}{l}", "{fi}{l}", "{f}_{l}{n}", "{f}.{l}{n}", "{l}.{f}"]).format(
            f=re.sub(r"[^a-z]", "", first.lower()), l=re.sub(r"[^a-z]", "", last.lower()),
            fi=first[0].lower(), n=rng.randint(1, 99))

        def street() -> str:
            line = f"{rng.randint(10, 9899)} {rng.choice(STREETS)}"
            if rng.random() < 0.15:
                line += rng.choice([f", Apt {rng.randint(1, 40)}{rng.choice('ABCD')}",
                                    f" Unit {rng.randint(2, 30)}"])
            return line

        city = rng.choice(CITIES)
        slug = re.sub(r"[^a-z]+", "-", f"{first} {last}".lower()).strip("-")
        idn = {
            "id": str(self.next_id), "gender": g, "first": first, "last": last,
            "name": f"{first} {last}", "he": he, "him": him, "his": his,
            "He": he.capitalize(), "His": his.capitalize(), "title": rng.choice(TITLES[g]),
            "child": FAMILY[g][0], "relative": FAMILY[g][1], "ne": FAMILY[g][2],
            "email": self._fresh(lambda: f"{local}@{rng.choice(FREE_DOMAINS + WORK_DOMAINS)}"),
            "phone": self._fresh(lambda: _phone(rng)),
            "ssn": self._fresh(lambda: _ssn(rng)),
            "dob": dob, "age": str(2026 - year),
            "street": self._fresh(street), "city": city,
            "zip": self._fresh(lambda: f"{rng.randint(10000, 99999)}"),
            "org": rng.choice(ORGS),
            "card": self._fresh(lambda: self._card(rng)),
            "ip": self._fresh(lambda: f"{rng.randint(11, 223)}.{rng.randint(0, 255)}."
                                      f"{rng.randint(0, 255)}.{rng.randint(1, 254)}"),
            "api": self._fresh(lambda: _api(rng)),
            "iban": self._fresh(lambda: _iban(rng)),
            "handle": self._fresh(lambda: "@" + rng.choice(["{f}_{n}", "{f}{l}", "the_{l}", "{f}.codes"]).format(
                f=re.sub(r"[^a-z]", "", first.lower()), l=re.sub(r"[^a-z]", "", last.lower()),
                n=rng.randint(7, 99))),
            "url": self._fresh(lambda: rng.choice([f"linkedin.com/in/{slug}-{rng.randint(10, 999)}",
                                                   f"https://{slug}.dev", f"github.com/{slug}"])),
        }
        idn["name_lower"], idn["name_upper"] = idn["name"].lower(), idn["name"].upper()
        idn["first_lower"], idn["city_lower"] = first.lower(), city.lower()
        self.next_id += 1
        return idn

    def _fresh_name(self, rng, firsts, lasts):
        for _ in range(5000):
            first, last = rng.choice(firsts), rng.choice(lasts)
            key = f"name:{first} {last}".casefold()
            if key not in self.used:
                self.used.add(key)
                return first, last
        raise RuntimeError("name pool exhausted")

    @staticmethod
    def _card(rng: random.Random) -> str:
        num = _luhn(rng.choice(["4", "51", "52", "53", "54", "55"]) +
                    "".join(str(rng.randint(0, 9)) for _ in range(14)))[:16]
        num = _luhn(num[:15])
        sep = rng.choice(["", " ", "-"])
        return sep.join(num[i:i + 4] for i in range(0, 16, 4)) if sep else num


def _distractors(rng: random.Random) -> Dict[str, str]:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2024, 2027)
    return {
        "date": f"{MONTHS[m - 1]} {d}, {y}" if rng.random() < 0.5 else f"{MONTHS[m - 1]} {d}",
        "iso_date": f"{y}-{m:02d}-{d:02d}", "slash_date": f"{m:02d}/{d:02d}/{y}",
        "year": str(rng.randint(2012, 2025)), "year2": str(rng.randint(2012, 2025)),
        "order": rng.choice([f"#{rng.randint(10000, 99999)}", f"ORD-{rng.randint(100000, 999999)}",
                             f"PO-{rng.randint(1000, 9999)}"]),
        "ticket": rng.choice([f"INC-{rng.randint(1000, 99999)}", f"JIRA-{rng.randint(100, 9999)}",
                              f"#{rng.randint(1000, 9999)}"]),
        "price": rng.choice([f"${rng.randint(5, 4999)}.{rng.randint(0, 99):02d}",
                             f"€{rng.randint(5, 999)}", f"£{rng.randint(5, 999)}"]),
        "version": f"{rng.randint(1, 15)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
        "product": rng.choice(PRODUCTS), "product2": rng.choice(PRODUCTS),
        "n10": "".join(str(rng.randint(0, 9)) for _ in range(10)),
        "n9": "".join(str(rng.randint(0, 9)) for _ in range(9)),
        "n5": str(rng.randint(10000, 99999)), "n5b": str(rng.randint(10000, 99999)),
        "n4": str(rng.randint(1000, 9999)), "n3": str(rng.randint(100, 999)),
        "n3b": str(rng.randint(100, 999)), "small": str(rng.randint(2, 9)),
        "pperson": rng.choice(PUBLIC_PEOPLE), "pperson2": rng.choice(PUBLIC_PEOPLE),
        "porg": rng.choice(PUBLIC_ORGS), "pplace": rng.choice(PUBLIC_PLACES),
        "acr": rng.choice(ACRONYMS), "acr2": rng.choice(ACRONYMS), "acr3": rng.choice(ACRONYMS),
    }


# Distractor fields recorded in "Keep" (a replacement there is a false positive).
_KEEP_FIELDS = {"date", "iso_date", "slash_date", "order", "ticket", "price", "version", "product",
                "product2", "n10", "n9", "n5", "n5b", "n4", "pperson", "pperson2", "porg",
                "pplace", "acr", "acr2", "acr3"}
_FIELD = re.compile(r"\{(\w+)\}")


def render(template: str, values: Dict[str, str]) -> Tuple[str, List[dict], List[str]]:
    """Fill *template*; return (text, gold spans, keep values)."""
    text, spans, keep, pos = "", [], [], 0
    tpl = template.replace("{{", "\0").replace("}}", "\1")
    for m in _FIELD.finditer(tpl):
        text += tpl[pos:m.start()]
        field, val = m.group(1), values[m.group(1)]
        if field in SPAN_LABEL:
            spans.append({"start": len(text), "end": len(text) + len(val),
                          "type": SPAN_LABEL[field], "value": val})
        elif field in _KEEP_FIELDS and val not in keep:
            keep.append(val)
        text += val
        pos = m.end()
    text += tpl[pos:]
    text = text.replace("\0", "{").replace("\1", "}")
    for sp in spans:
        assert text[sp["start"]:sp["end"]] == sp["value"]
    return text, spans, keep


def _only_at_spans(text: str, value: str, spans: List[dict]) -> bool:
    """True if every whole-token occurrence of *value* is inside a gold span."""
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.I)
    return all(any(sp["start"] <= m.start() and m.end() <= sp["end"] for sp in spans)
               for m in pat.finditer(text))


def _key(spans: List[dict]) -> Dict[str, object]:
    key: Dict[str, List[str]] = {}
    for sp in spans:
        vals = key.setdefault(sp["type"], [])
        if sp["value"] not in vals:
            vals.append(sp["value"])
    return {k: v[0] if len(v) == 1 else v for k, v in key.items()}


def split_templates(templates: list, part: int) -> list:
    """Every fourth template is test (part 1); the rest are dev (part 0)."""
    return [(i, t) for i, t in enumerate(templates) if (i % 4 == 3) == (part == 1)]


def generate(n: int, seed: int, strict: bool = True) -> Dict[str, list]:
    """Rows per split. *strict* requires every span type in every split."""
    ids = Identities()
    n_test = round(n * TEST_FRACTION)
    out = {}
    for part, (split, size) in enumerate((("dev", n - n_test), ("test", n_test))):
        rng = random.Random(seed * 1000 + part)
        pos, neg = split_templates(POS, part), split_templates(NEG, part)
        n_neg = round(size * NEG_FRACTION)
        rows = []
        for k in range(size):
            negative = k < n_neg
            t_index, (lang, tpl) = rng.choice(neg if negative else pos)
            idn = None if negative else ids.make(rng, part)
            for _ in range(100):           # a gold value never repeats inside a distractor
                text, spans, keep = render(tpl, dict(_distractors(rng), **(idn or {})))
                if all(_only_at_spans(text, sp["value"], spans) for sp in spans):
                    break
            else:
                raise RuntimeError(f"cannot fill {tpl!r} without a stray gold value")
            rows.append({
                "Question": text, "Answer-Key": _key(spans), "Spans": spans, "Keep": keep,
                "identity_id": None if idn is None else int(idn["id"]),
                "gender": None if idn is None else idn["gender"], "lang": lang,
                "template": f"{'neg' if negative else 'pos'}-{t_index:02d}",
            })
        missing = set(SPAN_LABEL.values()) - {sp["type"] for r in rows for sp in r["Spans"]}
        if missing and strict:
            raise RuntimeError(f"{split} has no {sorted(missing)} spans; add templates")
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["id"] = f"synth-{split}-{i:04d}"
        out[split] = rows
    return out


def write(out_dir: Path, data: Dict[str, list], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synth_manifest.json").write_text(json.dumps(dict(
        meta, splits={k: len(v) for k, v in data.items()}), indent=1) + "\n", encoding="utf-8")
    for split, rows in data.items():
        (out_dir / f"synth_{split}_key.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / f"synth_{split}.json").write_text(
            json.dumps([{"input": r["Question"]} for r in rows], indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1500, help="rows over both splits (default 1500)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    data = generate(a.n, a.seed)
    write(a.out, data, {"script": "experiment/make_dataset.py", "version": GENERATOR_VERSION,
                        "seed": a.seed, "n": a.n})
    for split, rows in data.items():
        neg = sum(1 for r in rows if not r["Spans"])
        print(f"{split}: {len(rows)} rows, {sum(len(r['Spans']) for r in rows)} gold spans, "
              f"{neg} negatives ({neg / len(rows):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
