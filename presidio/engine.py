# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
Lazy singleton wrapper for Presidio AnalyzerEngine.

Loading spaCy en_core_web_lg takes 3-5 seconds. This module loads it
once and caches it for the lifetime of the process. All other presidio/
modules call get_analyzer() — they never import AnalyzerEngine directly.

Returns None (never raises) if presidio-analyzer is not installed or
if the spaCy model is missing, so callers can degrade gracefully.
"""

_analyzer = None
_load_attempted = False
_load_error: str = ""


def get_analyzer():
    """
    Return a cached AnalyzerEngine, or None if unavailable.
    Thread-safe for single-threaded CLI use (no lock needed).
    """
    global _analyzer, _load_attempted, _load_error
    if _load_attempted:
        return _analyzer
    _load_attempted = True
    try:
        import logging

        # Silence all Presidio loggers completely before importing anything.
        # The spaCy NLP engine uses "presidio-analyzer" (hyphen); other
        # modules use "presidio_analyzer" (underscore).  Setting propagate=False
        # ensures messages never reach the root handler even if levels change.
        for _lg_name in (
            "presidio-analyzer",
            "presidio_analyzer",
            "presidio_analyzer.nlp_engine.spacy_nlp_engine",
            "presidio_analyzer.recognizer_registry",
        ):
            _lg = logging.getLogger(_lg_name)
            _lg.setLevel(logging.CRITICAL)
            _lg.propagate = False

        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import (
            NlpEngineProvider,
            NerModelConfiguration,
        )

        # Only load English-language recognizers — prevents the
        # "Recognizer not added to registry" warnings for es/it/pl
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["en"])

        # Tell Presidio to silently ignore spaCy labels it has no
        # mapping for — prevents PRODUCT/CARDINAL/ORDINAL warnings
        ner_model_config = NerModelConfiguration(
            labels_to_ignore=[
                "CARDINAL", "ORDINAL", "PRODUCT", "EVENT",
                "LANGUAGE", "MONEY", "QUANTITY", "TIME",
                "PERCENT", "WORK_OF_ART", "LAW",
            ]
        )

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{
                "lang_code":        "en",
                "model_name":       "en_core_web_lg",
                "ner_model_config": ner_model_config,
            }],
        }
        provider   = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        _analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )
    except ImportError as e:
        _load_error = f"import error: {e}"
        _analyzer = None
    except OSError as e:
        _load_error = f"spaCy model not found: {e}"
        _analyzer = None
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        _analyzer = None
    return _analyzer


def is_available() -> bool:
    """Return True if Presidio loaded successfully."""
    return get_analyzer() is not None


def unavailability_reason() -> str:
    """Return a human-readable reason why Presidio is unavailable."""
    if _load_error:
        return _load_error
    # Fallback probe (e.g. called before get_analyzer)
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError as e:
        return f"presidio-analyzer not importable: {e} — run: pip install presidio-analyzer"
    try:
        import spacy
        spacy.load("en_core_web_lg")
    except OSError:
        return "spaCy model missing — run: python -m spacy download en_core_web_lg"
    return "Presidio failed to initialize — check logs"
