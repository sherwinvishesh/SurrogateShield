"""
generation/mimic.py — MimicGen

Realistic surrogate generation for each PII entity type.
Uses Python Faker. Guarantees no collisions within a session via
a session-level used_surrogates set.
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Dict, FrozenSet, List, Optional, Set

from faker import Faker

from ..detection import address_parser
from ..detection.geo_data import US_STATE_ABBREVS
from ..entities import DetectedEntity

logger = logging.getLogger(__name__)

_fake = Faker()
Faker.seed(None)

# Types whose surrogates mirror the original's exact character shape
# (an ID number swaps to a same-shape ID number).
_SHAPE_TYPES = frozenset({
    "mac_address", "vin", "passport", "id_number", "license_plate",
})


def _aba_check(number: str) -> bool:
    digits = [int(c) for c in number]
    return (
        3 * digits[0] + 7 * digits[1] + digits[2] +
        3 * digits[3] + 7 * digits[4] + digits[5] +
        3 * digits[6] + 7 * digits[7] + digits[8]
    ) % 10 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Address surrogate helpers (module-level, unit-testable without MimicGen)
# ─────────────────────────────────────────────────────────────────────────────

def shift_house_number(
    parsed: "address_parser.ParsedAddress",
    shift_range: int = 1,
    rng: Optional[random.Random] = None,
    forbidden: FrozenSet[str] = frozenset(),
) -> Optional[str]:
    """
    Produce a surrogate address by shifting ONLY the house number.

    The new number token is spliced into the ORIGINAL matched span, so
    commas, newlines, casing, and abbreviations survive byte-for-byte.

    Rules:
      • candidate deltas ±1…±shift_range, random order;
      • a shifted number below 1 is rejected (house number 1 always goes UP —
        fixes the v1 max(1, 0) identity-surrogate bug);
      • leading zeros are preserved ("0123" → "0124");
      • alpha suffixes ride along unchanged ("123A" → "124A");
      • candidates found in *forbidden* (other real addresses in the text,
        already-issued surrogates, shadow-map values) are skipped; the range
        widens up to shift_range+8 before giving up.

    Returns the full surrogate address string, or None when no valid
    candidate exists (caller should fall back to replace mode).
    """
    if parsed is None or parsed.house_number is None:
        return None
    rel = parsed.house_number_relative_span
    if rel is None:
        return None

    match = re.match(r"\d+", parsed.house_number)
    if not match:
        return None
    numeric = match.group()
    n = int(numeric)
    _rng = rng if rng is not None else random

    deltas: List[int] = []
    for d in range(1, max(1, shift_range) + 1):
        deltas.extend((d, -d))
    _rng.shuffle(deltas)
    # Widening tail (deterministic order) for collision escape.
    for d in range(max(1, shift_range) + 1, max(1, shift_range) + 9):
        deltas.extend((d, -d))

    for delta in deltas:
        new_n = n + delta
        if new_n < 1:
            continue
        new_numeric = str(new_n).zfill(len(numeric)) if numeric[0] == "0" else str(new_n)
        new_token = new_numeric + parsed.house_number[len(numeric):]
        s, e = rel
        candidate = parsed.full_text[:s] + new_token + parsed.full_text[e:]
        if candidate != parsed.full_text and candidate not in forbidden:
            return candidate
    return None


class MimicGen:
    """Generates realistic, collision-resistant surrogates for detected PII.

    Args:
        seed: Optional seed for reproducible surrogate generation (tests).
              When None (default), generation is entropy-seeded.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.used_surrogates: Set[str] = set()
        self._rng = random.Random(seed) if seed is not None else random
        if seed is not None:
            self._fake = Faker()
            self._fake.seed_instance(seed)
        else:
            self._fake = _fake

    def _unique(self, generator_fn, max_attempts: int = 50) -> str:
        for _ in range(max_attempts):
            candidate = str(generator_fn())
            if candidate not in self.used_surrogates:
                self.used_surrogates.add(candidate)
                return candidate
        fallback = str(generator_fn()) + "_" + "".join(
            random.choices(string.ascii_lowercase, k=4)
        )
        self.used_surrogates.add(fallback)
        return fallback

    # ── Per-type generators ────────────────────────────────────────────────────

    def _gen_email(self) -> str:
        return _fake.email()

    def _gen_ssn(self) -> str:
        return _fake.ssn()

    def _gen_phone_us(self) -> str:
        return _fake.numerify("+1-###-###-####")

    def _gen_phone_uk(self) -> str:
        return _fake.numerify("+44 7### ######")

    def _gen_phone_intl(self) -> str:
        country_codes = [
            "+49", "+33", "+39", "+34", "+31", "+32", "+41", "+43", "+46",
            "+47", "+48", "+30", "+36", "+351", "+353",
            "+91", "+86", "+81", "+82", "+66", "+65", "+60", "+63",
            "+55", "+52", "+54", "+57", "+56", "+58",
            "+61", "+64",
            "+27", "+20", "+234", "+254", "+971", "+966",
        ]
        code = random.choice(country_codes)
        block1 = _fake.numerify("####")
        block2 = _fake.numerify("######")
        return f"{code} {block1} {block2}"

    def _gen_person(self) -> str:
        return _fake.name()

    # ── Address surrogates (mode-aware) ────────────────────────────────────────

    def _gen_address(
        self,
        entity: Optional[DetectedEntity] = None,
        mode: str = "shift",
        shift_range: int = 1,
        forbidden: FrozenSet[str] = frozenset(),
    ) -> str:
        """
        Address surrogate honoring the configured address mode.

          shift   → house number ±shift_range spliced into the original span
                    (street/city/state/ZIP and all formatting preserved);
          replace → structure-preserving fake: every component replaced by a
                    fake of the same shape, in the same slot.

        Falls back: shift → replace (on collision exhaustion),
        replace without a parse → plain Faker address (last resort).
        """
        parsed = getattr(entity, "parsed", None) if entity is not None else None
        if parsed is None and entity is not None:
            parsed = address_parser.parse(entity.text)

        blocked = frozenset(forbidden | self.used_surrogates)

        if mode == "shift" and parsed is not None:
            shifted = shift_house_number(
                parsed, shift_range=shift_range, rng=self._rng, forbidden=blocked
            )
            if shifted is not None:
                self.used_surrogates.add(shifted)
                return shifted
            logger.warning(
                "[MimicGen] shift collision exhausted for %r — falling back to replace",
                entity.text if entity else "?",
            )

        if parsed is not None:
            for _ in range(20):
                candidate = self._replace_address(parsed)
                if candidate not in blocked:
                    self.used_surrogates.add(candidate)
                    return candidate

        # Last resort: no structure available.
        return self._unique(lambda: self._fake.address().replace("\n", ", "))

    def _replace_address(self, parsed: "address_parser.ParsedAddress") -> str:
        """Build a structure-preserving fake: same components, same separators,
        same shapes — ONE unit, never concatenated fragments."""
        text = parsed.full_text
        replacements = []  # (start, end, replacement) relative to full_text
        cursor = 0

        def locate(value: str) -> Optional[tuple]:
            nonlocal cursor
            idx = text.find(value, cursor)
            if idx < 0:
                return None
            cursor = idx + len(value)
            return (idx, idx + len(value))

        # House / box number: same digit count, never identical.
        if parsed.house_number:
            span = parsed.house_number_relative_span
            cursor = span[1]
            numeric = re.match(r"\d+", parsed.house_number).group()
            new_numeric = self._fake_number_like(numeric)
            replacements.append(
                (span[0], span[0] + len(numeric), new_numeric)
            )

        # Street name (skip for highway forms, whose digits are handled below).
        if parsed.street_name:
            span = locate(parsed.street_name)
            if span:
                if re.search(r"\d", parsed.street_name):
                    # "Highway 50" → fake the route number, keep the road word.
                    route = re.search(r"\d+", parsed.street_name)
                    new_route = self._fake_number_like(route.group())
                    replacements.append(
                        (span[0] + route.start(), span[0] + route.end(), new_route)
                    )
                else:
                    replacements.append((span[0], span[1], self._fake.last_name()))

        # Unit: keep the designator, fake the digits of the id.
        if parsed.unit:
            span = locate(parsed.unit)
            if span:
                for digits in re.finditer(r"\d+", parsed.unit):
                    replacements.append((
                        span[0] + digits.start(),
                        span[0] + digits.end(),
                        self._fake_number_like(digits.group()),
                    ))

        # City: fake city of similar shape.
        if parsed.city:
            span = locate(parsed.city)
            if span:
                replacements.append((span[0], span[1], self._fake.city()))

        # State: different state, same form (2-letter stays 2-letter).
        if parsed.state:
            span = locate(parsed.state)
            if span:
                token = parsed.state.rstrip(".")
                if token.isupper() and len(token) == 2:
                    choices = sorted(US_STATE_ABBREVS - {token})
                    new_state = self._rng.choice(choices)
                else:
                    new_state = self._fake.state()
                replacements.append((span[0], span[0] + len(token), new_state))

        # ZIP: same shape (ZIP+4 stays ZIP+4).
        if parsed.zip_code:
            span = locate(parsed.zip_code)
            if span:
                for digits in re.finditer(r"\d+", parsed.zip_code):
                    replacements.append((
                        span[0] + digits.start(),
                        span[0] + digits.end(),
                        self._fake_number_like(digits.group()),
                    ))

        result = text
        for start, end, value in sorted(replacements, reverse=True):
            result = result[:start] + value + result[end:]
        return result

    def _fake_number_like(self, numeric: str) -> str:
        """Random number with the same digit count, different value,
        leading-zero style preserved."""
        for _ in range(20):
            candidate = "".join(
                str(self._rng.randint(0, 9)) for _ in range(len(numeric))
            )
            if candidate != numeric and (len(numeric) == 1 or candidate[0] != "0"):
                if numeric[0] == "0":
                    candidate = "0" + candidate[1:]
                return candidate
        return str(int(numeric) + 1).zfill(len(numeric))

    def _shape_like(self, original: str) -> str:
        """
        Random string with the same character-class shape as *original*:
        digits→digits, uppercase→uppercase, lowercase→lowercase, separators
        preserved.  Hex-looking strings (IPv6, MAC) stay within hex digits
        so the surrogate remains format-valid.
        """
        stripped = original.replace(":", "").replace("-", "").replace(".", "")
        is_hexish = (
            (":" in original or "." in original)
            and bool(re.fullmatch(r"[0-9A-Fa-f]*", stripped))
        )
        for _ in range(20):
            out = []
            for c in original:
                if c.isdigit():
                    out.append(str(self._rng.randint(0, 9)))
                elif is_hexish and c.isalpha():
                    pool = "abcdef" if c.islower() else "ABCDEF"
                    out.append(self._rng.choice(pool))
                elif c.isupper():
                    out.append(self._rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
                elif c.islower():
                    out.append(self._rng.choice("abcdefghijkmnpqrstuvwxyz"))
                else:
                    out.append(c)
            candidate = "".join(out)
            if candidate != original:
                return candidate
        return original[::-1]

    def _gen_iban_like(self, original: str) -> str:
        """
        Same-country fake IBAN with VALID mod-97 check digits and the
        original grouping/spacing preserved.
        """
        compact = re.sub(r"\s+", "", original)
        country = compact[:2]
        body = "".join(
            str(self._rng.randint(0, 9)) if c.isdigit()
            else self._rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
            for c in compact[4:]
        )
        rearranged = body + country + "00"
        numeric = "".join(str(int(c, 36)) for c in rearranged)
        check = 98 - (int(numeric) % 97)
        candidate = f"{country}{check:02d}{body}"
        out, i = [], 0
        for c in original:
            if c.isspace():
                out.append(c)
            else:
                out.append(candidate[i])
                i += 1
        return "".join(out)

    def _gen_credit_card(self) -> str:
        return _fake.credit_card_number(card_type=None)

    def _gen_dob(self) -> str:
        dob = _fake.date_of_birth(minimum_age=18, maximum_age=80)
        return dob.strftime("%m/%d/%Y")

    def _gen_ip(self) -> str:
        return _fake.ipv4()

    def _gen_zip_us(self) -> str:
        return _fake.zipcode()

    def _gen_postcode_uk(self) -> str:
        return _fake.postcode()

    def _gen_api_key(self) -> str:
        return "sk-" + _fake.lexify("?" * 32)

    def _gen_implicit_location(self) -> str:
        return _fake.city() + " area"

    def _gen_gpe(self) -> str:
        return _fake.city()

    def _gen_loc(self) -> str:
        return _fake.city() + " region"

    def _gen_org(self) -> str:
        return _fake.company()

    def _gen_fac(self) -> str:
        return _fake.company() + " Building"

    def _gen_crypto(self) -> str:
        b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        length = random.randint(26, 34)
        return "1" + "".join(random.choices(b58, k=length - 1))

    def _gen_us_bank_number(self) -> str:
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        for _ in range(100):
            digits = [random.randint(0, 9) for _ in range(8)]
            digits[0] = random.choice([0, 1, 2, 3])
            partial = sum(w * d for w, d in zip(weights, digits))
            check = (10 - (partial % 10)) % 10
            result = "".join(str(d) for d in digits) + str(check)
            if _aba_check(result):
                return result
        return "021000021"

    def _gen_us_driver_license(self) -> str:
        letter = random.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
        digits = _fake.numerify("#######")
        return f"{letter}{digits}"

    def _gen_default(self) -> str:
        return _fake.bothify("??##??##")

    def _gen_gender(self) -> str:
        options = [
            "male", "female", "non-binary",
            "he/him", "she/her", "they/them",
            "gender: male", "gender: female", "sex: male", "sex: female",
        ]
        return random.choice(options)

    # ── Dispatch table ─────────────────────────────────────────────────────────

    _GENERATORS: Dict[str, str] = {
        "email":             "_gen_email",
        "ssn":               "_gen_ssn",
        "phone_us":          "_gen_phone_us",
        "phone_uk":          "_gen_phone_uk",
        "phone_intl":        "_gen_phone_intl",
        "address":           "_gen_address",
        "person":            "_gen_person",
        "PERSON":            "_gen_person",
        "credit_card":       "_gen_credit_card",
        "dob":               "_gen_dob",
        "ip_address":        "_gen_ip",
        "zip_us":            "_gen_zip_us",
        "postcode_uk":       "_gen_postcode_uk",
        "api_key":           "_gen_api_key",
        "crypto":            "_gen_crypto",
        "us_bank_number":    "_gen_us_bank_number",
        "us_driver_license": "_gen_us_driver_license",
        "implicit_location": "_gen_implicit_location",
        "GPE":               "_gen_gpe",
        "LOC":               "_gen_loc",
        "ORG":               "_gen_org",
        "FAC":               "_gen_fac",
        "gender_indicator":  "_gen_gender",
    }

    def generate(
        self,
        entity: DetectedEntity,
        address_mode: str = "shift",
        address_shift_range: int = 1,
        forbidden: FrozenSet[str] = frozenset(),
    ) -> str:
        if entity.type == "gender_indicator":
            surrogate = self._gen_gender()
            logger.debug(f"[MimicGen] gender_indicator: {entity.text!r} → {surrogate!r}")
            return surrogate

        if entity.type == "address":
            surrogate = self._gen_address(
                entity,
                mode=address_mode,
                shift_range=address_shift_range,
                forbidden=forbidden,
            )
            logger.debug(f"[MimicGen] address: {entity.text!r} → {surrogate!r}")
            return surrogate

        # Shape-preserving types: the surrogate keeps the exact format of the
        # original (an ID number swaps to an ID number of the same shape).
        if entity.type == "iban":
            surrogate = self._unique(lambda: self._gen_iban_like(entity.text))
            logger.debug(f"[MimicGen] iban: {entity.text!r} → {surrogate!r}")
            return surrogate
        if entity.type in _SHAPE_TYPES or (
            entity.type == "ip_address" and ":" in entity.text
        ):
            surrogate = self._unique(lambda: self._shape_like(entity.text))
            logger.debug(f"[MimicGen] {entity.type}: {entity.text!r} → {surrogate!r}")
            return surrogate

        method_name = self._GENERATORS.get(entity.type, "_gen_default")
        method = getattr(self, method_name)
        surrogate = self._unique(method)
        logger.debug(f"[MimicGen] {entity.type}: {entity.text!r} → {surrogate!r}")
        return surrogate

    def generate_all(
        self,
        entities: List[DetectedEntity],
        address_mode: str = "shift",
        address_shift_range: int = 1,
        forbidden: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        """
        Generate surrogates for all entities.

        Args:
            entities:            Confirmed PII entities.
            address_mode:        "shift" or "replace" (an "auto" policy must be
                                 resolved by the caller before this point).
            address_shift_range: Max house-number delta for shift mode.
            forbidden:           Extra strings surrogates must never equal
                                 (e.g. shadow-map originals from prior turns).
        """
        blocked: Set[str] = set(forbidden or ())
        # A shifted address must never equal ANOTHER real address in the same
        # message ("789 X" and "790 X" both present → shift must dodge).
        blocked |= {e.text.strip() for e in entities if e.type == "address"}

        mapping: Dict[str, str] = {}
        for ent in entities:
            key = ent.text.strip()
            if key not in mapping:
                mapping[key] = self.generate(
                    ent,
                    address_mode=address_mode,
                    address_shift_range=address_shift_range,
                    forbidden=frozenset(blocked),
                )
                blocked.add(mapping[key])
        logger.info(f"[MimicGen] Generated {len(mapping)} surrogate mappings")
        return mapping
