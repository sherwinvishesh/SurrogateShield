# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
Lazy singleton wrapper for the Presidio AnalyzerEngine used as the baseline.

The baseline is Presidio exactly as it ships: the engine is built by
``AnalyzerEngineProvider()`` with no configuration files, i.e. the package's
own ``conf/default_analyzer.yaml``, ``conf/spacy.yaml`` (en_core_web_lg,
ORGANIZATION ignored, NORP→NRP) and ``conf/default_recognizers.yaml``. Nothing
is tuned. The only explicit choice is the score threshold, which Presidio
leaves at 0 and which is passed per call (``SCORE_THRESHOLD``) and recorded in
every result via ``baseline_config()``.

``get_analyzer()`` returns None when presidio-analyzer or the spaCy model is
not installed (the comparison is optional), and the reason is available from
``unavailability_reason()`` without loading spaCy a second time.
"""

from __future__ import annotations

import threading

# Explicit, documented threshold. Presidio's own default_score_threshold is 0,
# which keeps 0.01-score pattern hits; 0.4 is the threshold used in Presidio's
# documentation examples. It is applied per analyze() call, not baked in.
SCORE_THRESHOLD = 0.4

_analyzer = None
_load_attempted = False
_load_error: str = ""
_lock = threading.Lock()


def _silence_presidio_loggers() -> None:
    import logging

    for name in (
        "presidio-analyzer",
        "presidio_analyzer",
        "presidio_analyzer.nlp_engine.spacy_nlp_engine",
        "presidio_analyzer.recognizer_registry",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False


def get_analyzer():
    """Return the cached default-config AnalyzerEngine, or None if unavailable."""
    global _analyzer, _load_attempted, _load_error
    with _lock:
        if _load_attempted:
            return _analyzer
        _load_attempted = True
        try:
            _silence_presidio_loggers()
            import warnings

            from presidio_analyzer import AnalyzerEngineProvider

            with warnings.catch_warnings():
                # "NLP recognizer is not in the list ... Adding the default" —
                # informational; the default recognizer is what we want.
                warnings.filterwarnings("ignore", message="NLP recognizer")
                _analyzer = AnalyzerEngineProvider().create_engine()
        except ImportError as e:
            _load_error = (
                f"presidio-analyzer not importable: {e} — run: pip install presidio-analyzer"
            )
            _analyzer = None
        except OSError as e:
            _load_error = (
                f"spaCy model not found: {e} — run: python -m spacy download en_core_web_lg"
            )
            _analyzer = None
        return _analyzer


def is_available() -> bool:
    """Return True if Presidio loaded successfully."""
    return get_analyzer() is not None


def unavailability_reason() -> str:
    """Human-readable reason why Presidio is unavailable ('' if it is available)."""
    get_analyzer()
    return _load_error


def baseline_config() -> dict:
    """Describe the baseline configuration so it can be stored with results."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        ver = version("presidio-analyzer")
    except PackageNotFoundError:
        ver = "not installed"
    return {
        "presidio_analyzer_version": ver,
        "config": "AnalyzerEngineProvider() shipped defaults "
                  "(default_analyzer.yaml, spacy.yaml, default_recognizers.yaml)",
        "score_threshold": SCORE_THRESHOLD,
        "overlap_resolution": "longest span, then highest score",
    }
