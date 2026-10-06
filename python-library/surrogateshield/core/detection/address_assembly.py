"""
International street addresses: the layouts the US parser cannot read.

``address_parser`` reads one grammar — house, street, suffix, then city,
state and ZIP.  Most of the world writes an address differently:

  Canada       633 James Green Apt. 500, Edmonton, AB L7K 7J3
  Australia    Level 8 7 Adams Copse, Wollongong NSW 1728
  UK           Flat 74n, Thompson Run, Plymouth PH78 2DT
  India        85/847, Bahl Road, Vadodara 218408
  Germany      Langestraße 91/15, 40718 Mainz
  Netherlands  Yasminelaan 264, 5901 UM Eindhoven
  France       29, rue Julie Bègue, 61582 Lille
  Spain        Pasadizo de Amalia Guijarro 70 Piso 5, 12678 Murcia
  Mexico       Circunvalación Sur Zayas 923 Interior 020, 89933 Guadalajara
  Brazil       Travessa Felipe Barbosa, 62, Campinas - SP, 48872-059

``find`` returns each one as a single ``ParsedAddress`` covering the whole
address. Its ``parts`` field names every piece (house, street name, unit,
city, region, postcode), so the generator replaces all of them. A span that
stops at the city would leave the province and the postcode in the clear.

Four street cores, each grown right (units, then the city / region /
postcode tail) and, for English layouts, left (a unit written first):

  EN        [unit] HOUSE[,] Name…            needs a tail with a region or a
                                             postcode — the English street
                                             types are address_parser's job
  ROMANCE   Type Name…[,] [nº] HOUSE          (es, pt, it, mx)
            HOUSE[,] type Name…               (fr: "29, rue Julie Bègue")
            Type Name…                        no house: needs a postcode tail
  COMPOUND  Name+ending HOUSE                 (de, nl, nordic, fi)

The street-type words come from the postal authorities' published lists,
never from a test-data generator:
  - USPS Publication 28 Appendix C1, through address_parser.
  - The Correios DNE "tipo de logradouro" list.
  - Spain's INE street-type codes.
  - Mexico's INEGI "tipo de vialidad" catalogue.
  - France's BAN "type de voie" list.
  - Italy's DUG list.
A generator that draws its street types from those same lists (Faker does)
will produce types this module knows. DETECTOR_REPORT.md records that as
the generator-knowledge threat.

Every layout needs evidence beyond a capitalised word next to a number:
  - a strong street type (one that is not an English word) with a house
    number; or
  - a postcode or region tail whose shape the country fixes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .address_parser import _ALL_SUFFIXES as _EN_TYPES, _UNAMBIGUOUS_SUFFIXES as _EN_STRONG_TYPES, ParsedAddress
from .geo_data import US_STATE_ABBREVS

# ─────────────────────────────────────────────────────────────────────────────
# Letters and words
# ─────────────────────────────────────────────────────────────────────────────

# upper-case letters: ASCII, Latin-1 and Latin Extended-A ("Łódź", "Šumperk")
_UC = "A-ZÀ-ÖØ-Þ" + "".join(c for c in map(chr, range(0x100, 0x180)) if c.isupper())
_L = r"[^\W\d_]"
# "O'Reilly", "MacEver", "Säuberlich", "Saint-Étienne"
_CAPWORD = rf"[{_UC}]{_L}*(?:['’\-][{_UC}]?{_L}+)*"
_ORDINAL = r"\d{1,3}(?:st|nd|rd|th)"

# lower-case particles inside a street or town name
_JOINERS = ("de", "del", "della", "delle", "dei", "degli", "di", "da", "do", "dos", "das", "du", "des",
            "la", "las", "le", "les", "los", "lo", "y", "e", "i", "van", "von", "der", "den", "ter",
            "ten", "am", "an", "im", "auf", "sur", "sous", "en", "op", "aan", "upon", "on")
_JOIN = r"(?:" + "|".join(sorted(_JOINERS, key=len, reverse=True)) + r")"
_ELIDED = r"(?:[dl]['’])"

# ─────────────────────────────────────────────────────────────────────────────
# Units: a flat, suite or floor, written after the street or before it
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_WORDS = (
    # en: USPS Pub. 28 C2 secondary unit designators, plus UK / AU / IN usage
    "Apartment", "Apt", "Suite", "Ste", "Unit", "Flat", "Studio", "Floor", "Fl", "Level", "Lvl",
    "Room", "Rm", "Lot", "Building", "Bldg", "Shop", "Villa", "Office", "Penthouse", "PH",
    # es / mx / pt: INE and Correios complements
    "Interior", "Int", "Edificio", "Edif", "Departamento", "Depto", "Dpto", "Dto", "Piso", "Planta",
    "Puerta", "Pta", "Apartamento", "Apto", "Bloque", "Bloco", "Bl", "Casa", "Sala", "Andar",
    "Conjunto", "Torre", "Escalera", "Esc", "Portal", "Local", "Oficina", "Lote", "Manzana", "Mz",
    # de / nl / fr
    "Wohnung", "Whg", "Etage", "Stock", "Appartement", "Appt", "Bâtiment", "Bât", "Bat", "Bus",
)
_UNIT_WORD = r"(?:" + "|".join(sorted(_UNIT_WORDS, key=len, reverse=True)) + r")"
_UNIT_ID = r"(?:\d{1,5}[A-Za-z]?(?:[/-]\d{1,4}[A-Za-z]?)?|[A-Z]\d{0,4})(?![\w])"
_NUMBER_MARK = r"(?:#|[Nn][º°o]\.?|No\.?|Nr\.?)"
_UNIT = (rf"(?:(?<![\w])(?i:{_UNIT_WORD})\.?[ \t]?(?:{_NUMBER_MARK}[ \t]?)?{_UNIT_ID}"
         rf"|#[ \t]?{_UNIT_ID}"
         rf"|(?<![\w])\d{{1,2}}[º°ª](?:[ \t]?[A-Z](?![\w]))?)")   # es floor and door: "3º B"
# written before the street: "Flat 74n,", "Level 8 7 Adams Copse", "H.No. 794,"
# (a word on its own: not the "rm" of "confirm 790 Crescent Row")
_HNO = r"(?:H(?:ouse)?|D(?:oor)?|Plot|Flat)\.?[ \t]?No\.?"
_UNIT_BEFORE = (rf"(?:(?<![\w])(?:{_HNO}|(?i:{_UNIT_WORD})\.?)[ \t]?(?:{_NUMBER_MARK}[ \t]?)?"
                rf"(?:\d{{1,5}}[A-Za-z]?(?:[/-]\d{{1,4}}[A-Za-z]?)?|[A-Z]\d{{0,4}})(?![\w]))")
# a unit word with its id is no name word ("Flat 3, …"); alone it can be
# ("Rachel Flat", "Quadra de Casa Grande")
_NOT_NAME = rf"(?!(?i:{_UNIT_WORD})\.?[ \t]?(?:{_NUMBER_MARK}[ \t]?)?{_UNIT_ID})(?!{_HNO})"

# ─────────────────────────────────────────────────────────────────────────────
# Regions and postcodes
# ─────────────────────────────────────────────────────────────────────────────

# Canada Post province and territory symbols; Australia Post state codes;
# the Correios UF list
_CA_PROV = ("AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT")
_AU_STATE = ("NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT")
_BR_UF = ("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB",
          "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO")
_REGION = r"(?:" + "|".join(sorted(set(US_STATE_ABBREVS) | set(_CA_PROV) | set(_AU_STATE),
                                   key=len, reverse=True)) + r")"
_UF = r"(?:" + "|".join(_BR_UF) + r")"

_PC_CA = r"[A-Z]\d[A-Z][ ]?\d[A-Z]\d"
_PC_UK = r"[A-Z]{1,2}\d[A-Z\d]?[ ]?\d[A-Z]{2}"
_PC_US = r"\d{5}(?:-\d{4})?"
_PC_BR = r"\d{5}-?\d{3}"
_END = r"(?![\w\-])"
# after a region: a ZIP, a Canadian, Australian or Brazilian code
_PC_AFTER_REGION = rf"(?:{_PC_CA}|{_PC_BR}|{_PC_US}|\d{{4}}){_END}"
# after a town with no region: UK, Canada, India's six-digit PIN, a US ZIP
_PC_AFTER_TOWN = rf"(?:{_PC_UK}|{_PC_CA}|\d{{6}}|{_PC_US}){_END}"
# before the town (de, fr, es, mx, it, nl, at/ch/be/dk/no, pl, se/cz, pt)
_PC_BEFORE_TOWN = (r"(?:(?:C\.?P\.?|CP|D|F|NL|E|I|A|CH|B)[ \t]?-?[ \t]?)?"
                   r"(?:\d{5}(?:-\d{4})?|\d{4}[ ]?[A-Z]{2}(?![A-Za-z])|\d{4}-\d{3}|\d{3}[ ]\d{2}"
                   r"|\d{2}-\d{3}|\d{4})(?![\w\-])")

# ─────────────────────────────────────────────────────────────────────────────
# Towns
# ─────────────────────────────────────────────────────────────────────────────

# words that open the next field or sentence, never a town
_NOT_TOWN = frozenset("""
    phone tel telephone mobile mob cell fax email e-mail mail web website www url
    name address addr contact attn attention dob date born age id ref order invoice account
    please thanks thank regards cheers best kind sincerely hi hello dear yes no ok
    the and or but so then also just my your our his her their this that these those
    i we you he she it they me us
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november december
""".split())
_TOWN_WORD = rf"(?:{_CAPWORD}|{_ELIDED}{_CAPWORD})"
_TOWN = rf"{_NOT_NAME}{_TOWN_WORD}(?:(?:[ ]|[ ](?:{_JOIN})[ ]){_NOT_NAME}{_TOWN_WORD}){{0,3}}"

# ─────────────────────────────────────────────────────────────────────────────
# Street types (only where the layout needs them)
# ─────────────────────────────────────────────────────────────────────────────

# A type that is no English word makes a house number enough
# ("Calle Mayor 17"); a weak one also needs a postcode tail
# ("Parque Nunes, 76, Fortaleza - CE, 63237-812").
_ROMANCE_STRONG = (
    # es — INE tipos de vía
    "Calle", "C/", "Avenida", "Avda", "Paseo", "Pasaje", "Pasadizo", "Vial", "Callejón", "Calleja",
    "Carretera", "Ctra", "Glorieta", "Rambla", "Travesía", "Cuesta", "Bajada", "Subida", "Camino",
    "Carrer", "Costanilla", "Urbanización", "Urb", "Plazuela",
)
# mx — INEGI tipos de vialidad
_MX_TYPES = ("Ampliación", "Andador", "Calzada", "Cerrada", "Circuito", "Circunvalación", "Continuación",
             "Corredor", "Peatonal", "Periférico", "Privada", "Prolongación", "Retorno", "Viaducto")
# pt — Correios DNE tipos de logradouro
_PT_TYPES = ("Rua", "Travessa", "Rodovia", "Estrada", "Alameda", "Praça", "Largo", "Ladeira", "Beco",
             "Viela")
# it — DUG
_IT_TYPES = ("Viale", "Piazza", "Piazzale", "Vicolo", "Contrada", "Lungomare", "Borgo", "Strada")
_ROMANCE_STRONG += _MX_TYPES + _PT_TYPES + _IT_TYPES
_ROMANCE_WEAK = (
    "Via", "Plaza", "Parque", "Campo", "Vila", "Morro", "Pátio", "Diagonal", "Colônia", "Vereda",
    "Setor", "Quadra", "Jardim", "Vale", "Sítio", "Chácara", "Residencial", "Loteamento", "Núcleo",
    "Fazenda", "Distrito", "Lago", "Lagoa", "Recanto", "Trecho", "Trevo", "Conjunto", "Esplanada",
    "Viaduto", "Estação", "Favela", "Feira", "Aeroporto", "Área", "Condomínio", "Praia", "Passarela",
    "Boulevard", "Bulevar", "Corso", "Eje", "Eje vial", "Sector", "Colonia", "Barrio", "Ronda", "Vial",
    "Cañada", "Acceso",
    "R.", "Av.", "Tv.", "C.",                        # "Dr. R. Smith 3": an initial
)
# fr — BAN types de voie, written lower-case after the house number
_FR_STRONG = ("rue", "chemin", "impasse", "allée", "quai", "ruelle", "faubourg", "venelle", "montée",
              "traverse", "lieu-dit", "sentier")
_FR_WEAK = ("avenue", "av.", "boulevard", "bd", "place", "route", "cours", "passage", "square",
            "esplanade", "promenade", "voie", "cité", "hameau", "résidence", "villa", "parvis",
            "rond-point", "quartier")


def _lower(ws) -> frozenset:
    return frozenset(w.lower() for w in ws)


def _words(ws) -> str:
    return "|".join(re.escape(w) for w in sorted(ws, key=len, reverse=True))


_R_TYPE = rf"(?:{_words(_ROMANCE_STRONG + _ROMANCE_WEAK)})"
_FR_TYPE = rf"(?i:{_words(_FR_STRONG + _FR_WEAK)})"
_R_STRONG = frozenset(w.lower() for w in _ROMANCE_STRONG)
_FR_STRONG_SET = frozenset(_FR_STRONG)

# de / nl / nordic / fi compounds: the generic is the word's ending
# ("Kambsstraße", "Yasminelaan", "Ante-Junck-Allee").  Strong endings are no
# English word ending; weak ones ("ring": "Spring 3", "hof") need a postcode.
_CMP_STRONG = ("straße", "strasse", "weg", "platz", "allee", "gasse", "damm", "chaussee", "steig",
               "straat", "laan", "gracht", "plein", "kade", "singel", "dijk", "dreef", "steeg",
               "gatan", "gaten", "vägen", "veien", "vegen", "stræde", "katu")
_CMP_WEAK = ("str.", "ring", "hof", "ufer", "markt", "graben", "wall", "baan", "gata", "gade", "vej",
             "vei", "tie", "kuja", "torv", "pfad", "stieg", "zeile", "park", "berg", "pad", "boulevard")
_CMP_STRONG_SET = frozenset(_CMP_STRONG)
_COMPOUND = rf"[{_UC}](?:[\w'’\-]|\.(?=-))*?(?i:{_words(_CMP_STRONG + _CMP_WEAK)})"   # "Hans-J.-Beier-Straße"

# ─────────────────────────────────────────────────────────────────────────────
# Grammar
# ─────────────────────────────────────────────────────────────────────────────

_SEP = r"(?:[ \t]*,[ \t]*|[ \t]+)"
_LINE = r"(?:[ \t]*,?[ \t]*\r?\n[ \t]*)"            # a multi-line address
_UNITS = rf"(?P<units>(?:[ \t]*,?[ \t]*{_UNIT}){{0,3}})"

_TAIL = (
    # Brazil: "Campinas - SP, 48872-059"
    rf"(?:(?P<sep_br>{_SEP}|{_LINE})(?P<city_br>{_TOWN})[ \t]+[-–][ \t]+(?P<region_br>{_UF})"
    rf"(?:[ \t]*,?[ \t]*(?:CEP:?[ \t]*)?(?P<pc_br>{_PC_BR}){_END})?(?![\w])"
    # postcode first: ", 40718 Mainz", "\n5901 UM Eindhoven"
    rf"|(?P<sep_pf>[ \t]*,[ \t]*|{_LINE})(?P<pc_pf>{_PC_BEFORE_TOWN})[ \t]+(?P<city_pf>{_TOWN})"
    # a town, then a region and / or a postcode: "Edmonton, AB L7K 7J3",
    # "Cairns QLD 2617", "Plymouth PH78 2DT", "Mysuru 034021", "Ponferrada (24401)"
    rf"|(?P<sep_tw>{_SEP}|{_LINE})(?P<city_tw>{_TOWN})"
    rf"(?:{_SEP}(?P<region_tw>{_REGION})(?![\w])(?:[ \t]*,?[ \t]*\b(?P<pc_rg>{_PC_AFTER_REGION}))?"
    rf"|[ \t]*,?[ \t]*\b(?P<pc_tw>{_PC_AFTER_TOWN})"            # not "Leeds L|S1 4AP"
    rf"|[ \t]*\((?P<pc_br2>\d{{4,5}})\))"
    r")"
)

# Ireland writes no postcode in most mail: a bare town ends the address
# when punctuation, "and", a line break or the message's end follows it — and
# only after a USPS street type ("35 Hennigan Street, Waterford."). A county
# ("Co. Kerry") and an Eircode (routing key + unique identifier, from the
# Eircode alphabet) may follow the town or stand in for it.
_IE_COUNTY = rf"(?:Co\.?|County)[ \t]+{_CAPWORD}"
_EIRCODE = r"(?:[ACDEFHKNPRTVWXY]\d\d|D6W)[ ]?[0-9ACDEFHKNPRTVWXY]{4}(?![\w\-])"
_TOWN_ONLY = (rf"(?P<sep_to>[ \t]*,[ \t]*|{_LINE})"
              rf"(?:(?P<county_only>{_IE_COUNTY})"
              rf"|(?P<city_to>{_TOWN}(?:(?<=Dublin)[ ]\d{{1,2}}W?(?!\w))?)"      # "Dublin 6W": a postal district
              rf"(?:(?:[ \t]*,[ \t]*|{_LINE})(?P<county_to>{_IE_COUNTY}))?)"
              rf"(?:(?:[ \t]*,?[ \t]*|{_LINE})(?P<eir_to>{_EIRCODE}))?"
              r"(?=[ \t]*(?:[.,!?;:)\]\"'，。；：！？、]|\r?\n|$)|[ \t]+(?:[a-z]|&))")   # "…, Cork by Friday"

_NAME_EN = rf"{_NOT_NAME}(?:{_CAPWORD}|{_ORDINAL})(?:[ \t]+{_NOT_NAME}(?:{_CAPWORD}|{_ORDINAL})){{0,4}}"
_NAME_R = (rf"(?:(?:{_JOIN})[ \t]+|{_ELIDED})*{_NOT_NAME}{_CAPWORD}"
           rf"(?:[ \t]+(?:(?:{_JOIN})[ \t]+|{_ELIDED})*{_NOT_NAME}{_CAPWORD}){{0,5}}")

_HOUSE_EN = r"(?<![\w.,$#/\-])\d{1,6}[A-Za-z]?(?:/\d{1,5}[A-Za-z]?)?(?![\w/])"
# mx: an exterior and an interior number (INEGI): "Retorno Sur Adame 907 743"
_HOUSE_R = r"(?:\d{1,5}(?:[ ]?(?:bis|ter|[A-Za-z]))?(?:[ ]\d{1,4}(?![\w/\-]))?|[Ss]/[Nn])(?![\w/])"
_HOUSE_CMP = r"\d{1,4}[a-zA-Z]?(?:[ ]?[-/][ ]?\d{1,4}[a-zA-Z]?)?(?![\w/])"

_EN_RE = re.compile(
    rf"(?=(?:{_UNIT_BEFORE}|{_HOUSE_EN}){_SEP})"      # starts at a unit or a house: never at every word
    rf"(?:(?P<pre>{_UNIT_BEFORE})(?P<pre_sep>{_SEP}))?"
    rf"(?:(?P<house>{_HOUSE_EN})(?P<house_sep>{_SEP}))?"
    rf"(?P<name>{_NAME_EN})"
    rf"{_UNITS}(?:{_TAIL}|{_TOWN_ONLY})"
)
_R1_RE = re.compile(      # Type Name [,] [nº] HOUSE
    rf"(?<![\w])(?P<type>{_R_TYPE})[ \t]+(?P<name>{_NAME_R})"
    rf"(?:(?P<house_sep>[ \t]*,?[ \t]*)(?:{_NUMBER_MARK}[ \t]?)?(?P<house>{_HOUSE_R}))?"
    rf"{_UNITS}(?:{_TAIL})?"
)
_R2_RE = re.compile(      # HOUSE[,] type Name  (fr)
    rf"(?<![\w.,$#/\-])(?P<house>\d{{1,4}}(?:[ ]?(?:bis|ter))?)(?P<house_sep>[ \t]*,?[ \t]+)"
    rf"(?P<type>{_FR_TYPE})[ \t]+(?P<name>{_NAME_R})"
    rf"{_UNITS}(?:{_TAIL})?"
)
_R3_RE = re.compile(      # type Name, postcode Town  (fr, no house)
    rf"(?<![\w])(?P<type>{_FR_TYPE})[ \t]+(?P<name>{_NAME_R}){_UNITS}(?:{_TAIL})"
)
_CMP_RE = re.compile(     # Name+ending HOUSE
    rf"(?<![\w\-])(?P<name>{_COMPOUND})[ \t]+(?P<house>{_HOUSE_CMP})"
    rf"{_UNITS}(?:{_TAIL})?"
)

_YEAR = re.compile(r"(?:1[89]|20)\d\d")
_UNIT_SPLIT = re.compile(rf"[ \t]*,?[ \t]*({_UNIT})")


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def _tail(m: "re.Match") -> Tuple[Optional[str], List[Tuple[str, int, int]]]:
    """The tail's kind and its parts (absolute offsets)."""
    g = m.groupdict()
    parts: List[Tuple[str, int, int]] = []

    def add(role: str, name: str) -> None:
        if g.get(name) is not None:
            parts.append((role, m.start(name), m.end(name)))

    if g.get("city_br"):
        add("city", "city_br"), add("region", "region_br"), add("postcode", "pc_br")
        return "br", parts
    if g.get("city_pf"):
        add("postcode", "pc_pf"), add("city", "city_pf")
        return "pc_first", parts
    if g.get("city_tw"):
        add("city", "city_tw"), add("region", "region_tw")
        add("postcode", "pc_rg"), add("postcode", "pc_tw"), add("postcode", "pc_br2")
        if g.get("region_tw"):
            return "region", parts
        return "town_pc", parts
    if g.get("city_to") or g.get("county_only"):
        add("city", "city_to"), add("region", "county_to"), add("region", "county_only")
        add("postcode", "eir_to")
        return "town_only", parts
    return None, parts


def _town_ok(m: "re.Match") -> bool:
    for name in ("city_br", "city_pf", "city_tw", "city_to"):
        town = m.groupdict().get(name)
        if town:
            if town.isupper() and len(town) <= 3:
                return False                          # "IL" in "5, Chicago IL 60601": a region
            first = re.split(r"[ \-]", town, maxsplit=1)[0].strip("'’").lower()
            return first not in _NOT_TOWN
    return True


def _country(kind: Optional[str], m: "re.Match", family: str) -> Optional[str]:
    g = m.groupdict()
    if kind == "br":
        return "BR"
    region = g.get("region_tw")
    pc = g.get("pc_rg") or g.get("pc_tw") or g.get("pc_pf") or ""
    if g.get("county_to") or g.get("county_only") or g.get("eir_to") \
            or re.fullmatch(r"Dublin \d{1,2}W?", g.get("city_to") or ""):
        return "IE"
    if region:
        if region in _AU_STATE and not re.fullmatch(_PC_US, pc or "x"):
            return "AU"
        if region in _CA_PROV and (not pc or re.fullmatch(_PC_CA, pc)):
            return "CA"
        return "US"
    if pc and re.fullmatch(_PC_UK, pc):
        return "GB"
    if pc and re.fullmatch(_PC_CA, pc):
        return "CA"
    if pc and re.fullmatch(r"\d{6}", pc):
        return "IN"
    if pc and re.fullmatch(r"\d{4}[ ]?[A-Z]{2}", pc):
        return "NL"
    if family == "fr":
        return "FR"
    if family == "romance":
        t = (g.get("type") or "").lower()
        if t in _lower(_MX_TYPES) or re.fullmatch(r"\d{5}-\d{4}", pc):
            return "MX"
        if t in _lower(_PT_TYPES):
            return "BR"
        if t in _lower(_IT_TYPES + ("Via",)):
            return "IT"
        return "ES"
    if family == "compound":
        name = (g.get("name") or "").lower()
        if re.search(r"(?:straat|laan|gracht|plein|kade|singel|dijk|dreef|steeg|baan|pad|boulevard)$", name):
            return "NL"
        if re.search(r"(?:stra(?:ß|ss)e|str\.|weg|platz|allee|gasse|damm|chaussee|steig|ring|hof|ufer"
                     r"|markt|graben|wall|pfad|stieg|zeile|park|berg)$", name):
            return "DE"
    return None


def _units(m: "re.Match") -> List[Tuple[str, int, int]]:
    s = m.start("units")
    return [("unit", s + u.start(1), s + u.end(1)) for u in _UNIT_SPLIT.finditer(m.group("units"))]


def _build(text: str, m: "re.Match", start: int, end: int,
           parts: List[Tuple[str, int, int]], country: Optional[str]) -> ParsedAddress:
    parts = sorted(parts, key=lambda p: p[1])

    def first(role: str) -> Optional[Tuple[str, int, int]]:
        return next((p for p in parts if p[0] == role), None)

    def val(role: str) -> Optional[str]:
        p = first(role)
        return text[p[1]:p[2]] if p else None

    house = first("house")
    units = [text[s:e] for r, s, e in parts if r in ("unit", "pre")]
    return ParsedAddress(
        full_text=text[start:end], start=start, end=end,
        house_number=val("house"),
        house_number_span=(house[1], house[2]) if house else None,
        street_name=val("name"), suffix=val("type"),
        unit=", ".join(units) or None,
        city=val("city"), state=val("region"), zip_code=val("postcode"),
        parts=tuple((r, s - start, e - start) for r, s, e in parts),
        country=country,
    )


def _from_en(text: str, m: "re.Match") -> Optional[ParsedAddress]:
    if not (m.group("pre") or m.group("house")):
        return None
    kind, tail = _tail(m)
    if kind not in ("br", "region", "town_pc", "town_only") or not _town_ok(m):
        return None
    house = m.group("house")
    street_type = m.group("name").split()[-1].lower().rstrip(".")
    if kind == "region" and not m.group("pc_rg") and not (
            house and "," not in m.group("house_sep") and street_type in _EN_TYPES):
        return None                                   # "Building 7 Collapse, New York, NY": no postcode
    if kind == "town_only" and not (
            house and "," not in m.group("house_sep") and not m.group("units")
            and street_type in _EN_STRONG_TYPES):
        return None
    if house and "," in (m.group("house_sep") or "") and "/" not in house and _YEAR.fullmatch(house):
        return None                                   # "In 2019, Thompson Road, …": a year
    parts = [("name", m.start("name"), m.end("name"))] + _units(m) + tail
    if m.group("pre"):
        parts.append(("pre", m.start("pre"), m.end("pre")))
    if house:
        parts.append(("house", m.start("house"), m.end("house")))
    return _build(text, m, m.start(), m.end(), parts, _country(kind, m, "en"))


def _from_romance(text: str, m: "re.Match", family: str) -> Optional[ParsedAddress]:
    kind, tail = _tail(m)
    if kind is not None and not _town_ok(m):
        return None
    t = m.group("type")
    house = m.groupdict().get("house")
    if family == "fr":
        strong = t.lower() in _FR_STRONG_SET
    else:
        strong = t.lower() in _R_STRONG
    if family == "romance" and kind is None and house is None:
        return None
    if kind is None and not (house and strong):
        return None                                   # "via Slack 3 times": a weak type, no postcode
    if kind == "region" or (kind == "town_pc" and family != "romance"):
        return None                                   # an English tail after a Romance street
    parts = [("type", m.start("type"), m.end("type")), ("name", m.start("name"), m.end("name"))]
    parts += _units(m) + tail
    if house:
        parts.append(("house", m.start("house"), m.end("house")))
    end = m.end() if kind else max(e for _r, _s, e in parts)
    return _build(text, m, m.start(), end, parts, _country(kind, m, family))


def _from_compound(text: str, m: "re.Match") -> Optional[ParsedAddress]:
    name = m.group("name")
    ending = next(e for e in sorted(_CMP_STRONG + _CMP_WEAK, key=len, reverse=True)
                  if name.lower().endswith(e))
    stem = name[:len(name) - len(ending)].rstrip("-")
    if len(re.sub(r"[\W\d_]", "", stem)) < 2:
        return None                                   # "Weg 5", "Ring 3"
    kind, tail = _tail(m)
    if kind is not None and not _town_ok(m):
        return None
    if kind not in (None, "pc_first"):
        return None
    if kind is None and ending not in _CMP_STRONG_SET:
        return None                                   # "Spring 3", "Keyring 2": a weak ending
    parts = [("name", m.start("name"), m.end("name")), ("house", m.start("house"), m.end("house"))]
    parts += _units(m) + tail
    end = m.end() if kind else max(e for _r, _s, e in parts)
    return _build(text, m, m.start(), end, parts, _country(kind, m, "compound"))


def _candidates(text: str) -> List[ParsedAddress]:
    out: List[ParsedAddress] = []
    for rx, make in ((_EN_RE, _from_en),
                     (_R1_RE, lambda t, m: _from_romance(t, m, "romance")),
                     (_R2_RE, lambda t, m: _from_romance(t, m, "fr")),
                     (_R3_RE, lambda t, m: _from_romance(t, m, "fr")),
                     (_CMP_RE, _from_compound)):
        pos = 0
        while pos < len(text):
            m = rx.search(text, pos)
            if not m:
                break
            parsed = make(text, m) if m.end() > m.start() else None
            if parsed is not None:
                out.append(parsed)
                pos = parsed.end
            else:
                pos = m.start() + 1
    return out


def find(text: str) -> List[ParsedAddress]:
    """Non-overlapping international addresses in *text*, longest first
    where two layouts read the same words."""
    chosen: List[ParsedAddress] = []
    for p in sorted(_candidates(text), key=lambda p: (-(p.end - p.start), p.start)):
        if all(p.end <= q.start or p.start >= q.end for q in chosen):
            chosen.append(p)
    return sorted(chosen, key=lambda p: p.start)


def parse(fragment: str) -> Optional[ParsedAddress]:
    """The address that spans all of *fragment*, if this module reads one."""
    stripped = fragment.strip()
    lead = fragment.find(stripped) if stripped else 0
    for p in find(fragment):
        if p.start == lead and p.end == lead + len(stripped):
            return p
    return None
