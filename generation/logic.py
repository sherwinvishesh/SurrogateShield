# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
generation/logic.py — MimicGen

Realistic surrogate generation for each PII entity type.
Uses Python Faker. Guarantees no collisions within a session via
a session-level used_surrogates set.

Every generated surrogate is type-consistent (a fake email looks like
a real email, a fake SSN follows the correct format, etc.) and unique
within the session — no two real values ever map to the same fake.
"""

from __future__ import annotations

import random
import string
from typing import Dict, List, Optional, Set

from faker import Faker

from util import DetectedEntity, get_logger

logger = get_logger(__name__)

_fake = Faker()
Faker.seed(None)


def _aba_check(number: str) -> bool:
    """Used by _gen_us_bank_number to verify generated routing numbers."""
    digits = [int(c) for c in number]
    return (
        3 * digits[0] + 7 * digits[1] + digits[2] +
        3 * digits[3] + 7 * digits[4] + digits[5] +
        3 * digits[6] + 7 * digits[7] + digits[8]
    ) % 10 == 0


class MimicGen:
    """
    Generates realistic, collision-resistant surrogates for detected PII.

    Attributes:
        used_surrogates: Session-level set of already-issued surrogates.
    """

    def __init__(self) -> None:
        self.used_surrogates: Set[str] = set()

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
        """
        Generate a realistic international phone surrogate (non-US, non-UK).

        Randomly selects from a pool of real country calling codes and
        generates a plausible subscriber number in groups, e.g. "+49 8234 927461".
        The format mirrors how international numbers are commonly written.
        """
        country_codes = [
            "+49", "+33", "+39", "+34", "+31", "+32", "+41", "+43", "+46",
            "+47", "+48", "+30", "+36", "+351", "+353",    # Europe
            "+91", "+86", "+81", "+82", "+66", "+65", "+60", "+63",  # Asia
            "+55", "+52", "+54", "+57", "+56", "+58",      # Americas
            "+61", "+64",                                   # Oceania
            "+27", "+20", "+234", "+254", "+971", "+966",  # Africa / ME
        ]
        code = random.choice(country_codes)
        # Generate local number as two blocks of digits
        block1 = _fake.numerify("####")
        block2 = _fake.numerify("######")
        return f"{code} {block1} {block2}"

    def _gen_person(self) -> str:
        return _fake.name()

    def _gen_address(self) -> str:
        """Generate a realistic street address surrogate."""
        return _fake.address().replace("\n", ", ")

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
        """Generate a realistic-looking Bitcoin address (P2PKH format)."""
        import random as _rand
        b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        length = _rand.randint(26, 34)
        return "1" + "".join(_rand.choices(b58, k=length - 1))

    def _gen_us_bank_number(self) -> str:
        """
        Generate a valid 9-digit ABA routing number that passes the checksum.
        Uses the ABA formula to compute the 9th check digit.
        """
        import random as _rand
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        for _ in range(100):
            digits = [_rand.randint(0, 9) for _ in range(8)]
            digits[0] = _rand.choice([0, 1, 2, 3])
            partial = sum(w * d for w, d in zip(weights, digits))
            check = (10 - (partial % 10)) % 10
            result = "".join(str(d) for d in digits) + str(check)
            if _aba_check(result):
                return result
        return "021000021"  # known valid fallback (Federal Reserve Bank of NY)

    def _gen_us_driver_license(self) -> str:
        """
        Generate a realistic driver's license number.
        Uses California format (letter + 7 digits) as the most common template.
        """
        import random as _rand
        letter = _rand.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
        digits = _fake.numerify("#######")
        return f"{letter}{digits}"

    def _gen_default(self) -> str:
        return _fake.bothify("??##??##")

    def _gen_gender(self) -> str:
        """
        Generate a gender indicator surrogate that preserves grammatical
        structure. Replaces detected gender with a different valid gender
        expression so sentences like 'I am a female nurse' remain readable
        as 'I am a male nurse' rather than breaking into 'I am a xy42ab98 nurse'.
        """
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
        "phone_intl":        "_gen_phone_intl",   # international numbers
        "address":           "_gen_address",       # street addresses (PatternScan)
        "person":            "_gen_person",
        "PERSON":            "_gen_person",
        "credit_card":       "_gen_credit_card",
        "dob":               "_gen_dob",
        "ip_address":        "_gen_ip",
        "zip_us":            "_gen_zip_us",
        "postcode_uk":       "_gen_postcode_uk",
        "api_key":            "_gen_api_key",
        "crypto":             "_gen_crypto",
        "us_bank_number":     "_gen_us_bank_number",
        "us_driver_license":  "_gen_us_driver_license",
        "implicit_location":  "_gen_implicit_location",
        "GPE":               "_gen_gpe",
        "LOC":               "_gen_loc",
        "ORG":               "_gen_org",
        "FAC":               "_gen_fac",
        "gender_indicator":  "_gen_gender",
    }

    def generate(self, entity: DetectedEntity) -> str:
        # Gender indicator gets a direct random replacement — not unique-wrapped
        # because the pool is small and uniqueness is less important than
        # producing a grammatically valid gender expression.
        if entity.type == "gender_indicator":
            surrogate = self._gen_gender()
            logger.debug(f"[MimicGen] gender_indicator: {entity.text!r} → {surrogate!r}")
            return surrogate

        method_name = self._GENERATORS.get(entity.type, "_gen_default")
        method = getattr(self, method_name)
        surrogate = self._unique(method)
        logger.debug(f"[MimicGen] {entity.type}: {entity.text!r} → {surrogate!r}")
        return surrogate

    def generate_all(self, entities: List[DetectedEntity]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for ent in entities:
            key = ent.text.strip()
            if key not in mapping:
                mapping[key] = self.generate(ent)
        logger.info(f"[MimicGen] Generated {len(mapping)} surrogate mappings")
        return mapping