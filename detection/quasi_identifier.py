"""
detection/quasi_identifier.py — Quasi-Identifier Combination Risk Scorer

Based on Sweeney's k-anonymity research: combinations of seemingly innocuous
fields (ZIP code, DOB, gender) can uniquely re-identify individuals without
any traditional PII like an SSN.

Works purely on entity types — no language model required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from util import DetectedEntity


# Human-readable labels for each entity type used in warnings.
_TYPE_LABELS: Dict[str, str] = {
    "PERSON":           "Name",
    "email":            "Email",
    "ssn":              "SSN",
    "phone_us":         "Phone",
    "ip_address":       "IP Address",
    "zip_us":           "ZIP",
    "postcode_uk":      "Postcode",
    "dob":              "DOB",
    "gender_indicator": "Gender",
    "ORG":              "Employer",
    "GPE":              "Location",
}

# Display order for building label strings (lower = earlier in the string).
_FIELD_SORT_KEY: Dict[str, int] = {
    "PERSON":           0,
    "email":            1,
    "ssn":              2,
    "phone_us":         3,
    "ip_address":       4,
    "zip_us":           5,
    "postcode_uk":      6,
    "dob":              7,
    "gender_indicator": 8,
    "ORG":              9,
    "GPE":              10,
}


QUASI_ID_COMBINATIONS: List[dict] = [
    {
        "name": "ZIP + DOB + Gender",
        "fields": frozenset({"zip_us", "dob", "gender_indicator"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "Sweeney 2000 — 87% of US population uniquely identified by all three fields",
        "partial_reference": "ZIP + DOB together re-identify ~63% of US population (Sweeney 2000)",
    },
    {
        "name": "Postcode + DOB",
        "fields": frozenset({"postcode_uk", "dob"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "UK ICO: postcode + DOB combination is sufficient for re-identification",
        "partial_reference": "UK ICO: postcode + DOB combination is sufficient for re-identification",
    },
    {
        "name": "Name + Employer + City",
        "fields": frozenset({"PERSON", "ORG", "GPE"}),
        "required_count": 3,
        "risk_level": "medium",
        "reference": "Name + employer + city triple uniquely identifies an individual in most cities",
        "partial_reference": "Name + employer + city triple uniquely identifies an individual in most cities",
    },
    {
        "name": "Name + SSN",
        "fields": frozenset({"PERSON", "ssn"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "Name + SSN combination enables direct identity theft and financial fraud",
        "partial_reference": "Name + SSN combination enables direct identity theft and financial fraud",
    },
    {
        "name": "Name + DOB",
        "fields": frozenset({"PERSON", "dob"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "Name + DOB is the standard combination used in identity verification worldwide",
        "partial_reference": "Name + DOB is the standard combination used in identity verification worldwide",
    },
    {
        "name": "Email + Location",
        "fields": frozenset({"email", "GPE"}),
        "required_count": 2,
        "risk_level": "medium",
        "reference": "Email address + location narrows to a specific named individual",
        "partial_reference": "Email address + location narrows to a specific named individual",
    },
    {
        "name": "Phone + Name",
        "fields": frozenset({"PERSON", "phone_us"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "Name + phone number directly identifies an individual",
        "partial_reference": "Name + phone number directly identifies an individual",
    },
    {
        "name": "Phone + Location",
        "fields": frozenset({"phone_us", "GPE"}),
        "required_count": 2,
        "risk_level": "medium",
        "reference": "Phone number + location narrows to a local individual",
        "partial_reference": "Phone number + location narrows to a local individual",
    },
    {
        "name": "IP Address + Name",
        "fields": frozenset({"ip_address", "PERSON"}),
        "required_count": 2,
        "risk_level": "high",
        "reference": "IP address + name enables device-level identification",
        "partial_reference": "IP address + name enables device-level identification",
    },
    {
        "name": "DOB + Location + Employer",
        "fields": frozenset({"dob", "GPE", "ORG"}),
        "required_count": 3,
        "risk_level": "medium",
        "reference": "DOB + location + employer triple is highly specific",
        "partial_reference": "DOB + location + employer triple is highly specific",
    },
]


@dataclass
class QuasiIdMatch:
    combination_name: str
    matched_fields: list
    matched_entities: list
    risk_level: str
    reference: str
    partial_reference: str
    all_fields_matched: bool  # True if ALL fields in the combo were present


def score(entities: List[DetectedEntity]) -> List[QuasiIdMatch]:
    """Return a QuasiIdMatch for every triggered combination."""
    type_to_entities: Dict[str, list] = {}
    for ent in entities:
        type_to_entities.setdefault(ent.type, []).append(ent)

    matches: List[QuasiIdMatch] = []
    for combo in QUASI_ID_COMBINATIONS:
        matched_fields = [f for f in combo["fields"] if f in type_to_entities]
        if len(matched_fields) >= combo["required_count"]:
            matched_entities: list = []
            for f in matched_fields:
                matched_entities.extend(type_to_entities[f])
            all_matched = len(matched_fields) == len(combo["fields"])
            matches.append(QuasiIdMatch(
                combination_name=combo["name"],
                matched_fields=matched_fields,
                matched_entities=matched_entities,
                risk_level=combo["risk_level"],
                reference=combo["reference"],
                partial_reference=combo["partial_reference"],
                all_fields_matched=all_matched,
            ))
    return matches


def format_warning(matches: List[QuasiIdMatch]) -> str:
    """Return human-readable warnings for all triggered combinations."""
    lines = []
    for match in matches:
        # Sort matched fields by display order for a consistent label string.
        labels = " + ".join(
            _TYPE_LABELS.get(f, f)
            for f in sorted(match.matched_fields, key=lambda f: _FIELD_SORT_KEY.get(f, 99))
        )
        # Use the full reference only when ALL fields in the combination fired;
        # use the partial reference when only the minimum required count fired,
        # to avoid claims about fields that were not present.
        ref = match.reference if match.all_fields_matched else match.partial_reference
        lines.append(
            f"⚠  Quasi-identifier risk: {labels} combination detected\n"
            f"    ({ref})"
        )
    return "\n".join(lines)
