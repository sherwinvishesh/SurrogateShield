# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""SentinelLayer for the app: ``surrogateshield.core.detection.pipeline``
with the thresholds and models taken from ``config.py`` (audit F4).

Every other name (``deduplicate``, the post-passes, …) resolves to the
package module.
"""

from typing import Dict, List, Optional, Set, Tuple

import config as _config
from surrogateshield.core.detection import pipeline as _impl
from surrogateshield.core.entities import DetectedEntity


def run_cascade(
    text: str,
    skip_values: Optional[Set[str]] = None,
    skip_location_entities: bool = False,
    timings: Optional[Dict[str, float]] = None,
    **kwargs,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """``pipeline.run_cascade`` with config.py defaults; keyword arguments
    (ablation switches, pii_off, explicit thresholds) win."""
    for key, value in (
        ("spacy_model", _config.SPACY_MODEL),
        ("context_guard_enabled", _config.CONTEXT_GUARD_ENABLED),
        ("entity_trace_high_threshold", _config.ENTITY_TRACE_HIGH_THRESHOLD),
        ("entity_trace_low_threshold", _config.ENTITY_TRACE_LOW_THRESHOLD),
        ("context_guard_threshold", _config.CONTEXT_GUARD_CONFIDENCE_THRESHOLD),
        ("entity_trace_fallback_threshold", _config.ENTITY_TRACE_FALLBACK_THRESHOLD),
        ("context_guard_model", _config.CONTEXT_GUARD_MODEL),
        ("context_guard_device", _config.CONTEXT_GUARD_DEVICE),
    ):
        kwargs.setdefault(key, value)
    return _impl.run_cascade(text, skip_values, skip_location_entities, timings, **kwargs)


def __getattr__(name):
    return getattr(_impl, name)
