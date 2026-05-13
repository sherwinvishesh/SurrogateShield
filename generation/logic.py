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

# ─────────────────────────────────────────────
# Faker instance
# ─────────────────────────────────────────────

_fake = Faker()
Faker.seed(None)   # True randomness across sessions


# ─────────────────────────────────────────────
# MimicGen class
# ─────────────────────────────────────────────

class MimicGen:
    """
    Generates realistic, collision-resistant surrogates for detected PII.

    Attributes:
        used_surrogates: Session-level set of already-issued surrogates.
            No two entities will share the same fake value.
    """

    def __init__(self) -> None:
        """Initialise MimicGen with an empty collision-avoidance set."""
        self.used_surrogates: Set[str] = set()

    def _unique(self, generator_fn, max_attempts: int = 50) -> str:
        """
        Call *generator_fn* until a value not in used_surrogates is produced.

        Args:
            generator_fn:  Callable with no arguments that returns a str.
            max_attempts:  Safety limit to prevent infinite loops.

        Returns:
            A surrogate string that has not been used in this session.
        """
        for _ in range(max_attempts):
            candidate = str(generator_fn())
            if candidate not in self.used_surrogates:
                self.used_surrogates.add(candidate)
                return candidate
        # Fallback: append random suffix to guarantee uniqueness
        fallback = str(generator_fn()) + "_" + "".join(
            random.choices(string.ascii_lowercase, k=4)
        )
        self.used_surrogates.add(fallback)
        return fallback

    # ── Per-type generators ────────────────────────────────────────

    def _gen_email(self) -> str:
        return _fake.email()

    def _gen_ssn(self) -> str:
        return _fake.ssn()

    def _gen_phone_us(self) -> str:
        return _fake.numerify("+1-###-###-####")

    def _gen_phone_uk(self) -> str:
        return _fake.numerify("+44 7### ######")

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

    def _gen_default(self) -> str:
        return _fake.bothify("??##??##")

    # ── Dispatch table ────────────────────────────────────────────

    _GENERATORS: Dict[str, str] = {
        "email": "_gen_email",
        "ssn": "_gen_ssn",
        "phone_us": "_gen_phone_us",
        "phone_uk": "_gen_phone_uk",
        "person": "_gen_person",
        "PERSON": "_gen_person",
        "address": "_gen_address",
        "credit_card": "_gen_credit_card",
        "dob": "_gen_dob",
        "ip_address": "_gen_ip",
        "zip_us": "_gen_zip_us",
        "postcode_uk": "_gen_postcode_uk",
        "api_key": "_gen_api_key",
        "implicit_location": "_gen_implicit_location",
        "GPE": "_gen_gpe",
        "LOC": "_gen_loc",
        "ORG": "_gen_org",
        "FAC": "_gen_fac",
    }

    def generate(self, entity: DetectedEntity) -> str:
        """
        Generate a realistic surrogate for a single DetectedEntity.

        Args:
            entity: The detected PII entity to generate a surrogate for.

        Returns:
            A unique, type-consistent surrogate string.
        """
        method_name = self._GENERATORS.get(entity.type, "_gen_default")
        method = getattr(self, method_name)
        surrogate = self._unique(method)
        logger.debug(
            f"[MimicGen] {entity.type}: {entity.text!r} → {surrogate!r}"
        )
        return surrogate

    def generate_all(
        self,
        entities: List[DetectedEntity],
    ) -> Dict[str, str]:
        """
        Generate surrogates for a list of entities.

        Deduplicates by entity.text so the same original value always
        maps to the same surrogate within this call.

        Args:
            entities: List of DetectedEntity objects to replace.

        Returns:
            Dict mapping original_text → surrogate_text.
        """
        mapping: Dict[str, str] = {}
        for ent in entities:
            key = ent.text.strip()
            if key not in mapping:
                mapping[key] = self.generate(ent)
        logger.info(f"[MimicGen] Generated {len(mapping)} surrogate mappings")
        return mapping
