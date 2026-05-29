"""
generation/mimic.py — MimicGen

Realistic surrogate generation for each PII entity type.
Uses Python Faker. Guarantees no collisions within a session via
a session-level used_surrogates set.
"""

from __future__ import annotations

import logging
import random
import string
from typing import Dict, List, Set

from faker import Faker

from ..entities import DetectedEntity

logger = logging.getLogger(__name__)

_fake = Faker()
Faker.seed(None)


def _aba_check(number: str) -> bool:
    digits = [int(c) for c in number]
    return (
        3 * digits[0] + 7 * digits[1] + digits[2] +
        3 * digits[3] + 7 * digits[4] + digits[5] +
        3 * digits[6] + 7 * digits[7] + digits[8]
    ) % 10 == 0


class MimicGen:
    """Generates realistic, collision-resistant surrogates for detected PII."""

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

    def _gen_address(self) -> str:
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

    def generate(self, entity: DetectedEntity) -> str:
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
