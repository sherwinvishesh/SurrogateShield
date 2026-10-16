"""Full-cascade detection tests (spaCy + HuggingFace). Run with:
    python -m pytest tests/ -m heavy
"""

import pytest

pytestmark = pytest.mark.heavy

spacy = pytest.importorskip("spacy")

from detection.logic import run_cascade, deduplicate  # noqa: E402


def _model_available():
    return spacy.util.is_package("en_core_web_lg")


needs_model = pytest.mark.skipif(
    not _model_available(), reason="en_core_web_lg not installed"
)


@needs_model
def test_cascade_person_and_address():
    confirmed, _ = run_cascade(
        "My name is Sarah Mitchell and I live at 789 Crescent Row, Tempe, AZ 85281."
    )
    confirmed = deduplicate(confirmed)
    by_type = {e.type: e.text for e in confirmed}
    assert by_type.get("address") == "789 Crescent Row, Tempe, AZ 85281"
    assert by_type.get("PERSON") == "Sarah Mitchell"


@needs_model
def test_city_inside_address_not_double_detected():
    """The full-span mask must hide the in-address city from spaCy."""
    confirmed, _ = run_cascade(
        "Bill me at 789 Crescent Row, Tempe, AZ 85281 next month."
    )
    confirmed = deduplicate(confirmed)
    gpe_values = [e.text for e in confirmed if e.type in ("GPE", "LOC")]
    assert "Tempe" not in gpe_values
    assert "AZ" not in gpe_values


@needs_model
def test_standalone_city_still_detected_in_personal_context():
    confirmed, _ = run_cascade("I grew up in Tempe before moving away.")
    confirmed = deduplicate(confirmed)
    assert any(e.type == "GPE" and e.text == "Tempe" for e in confirmed)


@needs_model
def test_service_query_skips_locations():
    confirmed, _ = run_cascade(
        "any good restaurants near Tempe?", skip_location_entities=True
    )
    confirmed = deduplicate(confirmed)
    assert not any(e.type in ("GPE", "LOC", "FAC") for e in confirmed)
