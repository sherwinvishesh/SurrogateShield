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

    def _gen_default(self) -> str:
        return _fake.bothify("??##??##")

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
        "api_key":           "_gen_api_key",
        "implicit_location": "_gen_implicit_location",
        "GPE":               "_gen_gpe",
        "LOC":               "_gen_loc",
        "ORG":               "_gen_org",
        "FAC":               "_gen_fac",
        "gender_indicator":  "_gen_default",
    }

    def generate(self, entity: DetectedEntity) -> str:
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