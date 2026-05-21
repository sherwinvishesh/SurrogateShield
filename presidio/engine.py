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
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(
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
