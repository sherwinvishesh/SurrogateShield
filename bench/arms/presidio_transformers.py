"""Arm ``presidio_transformers``: Presidio with its documented transformers NLP
engine — the package's own ``conf/transformers.yaml`` (spaCy en_core_web_sm for
tokens, label mapping, stride 16, ``aggregation_strategy: max``) with the NER
model set to ``obi/deid_roberta_i2b2`` (instead of the file's
StanfordAIMI/stanford-deidentifier-base). Threshold 0.4, overlaps resolved
longest-then-score, ``[ENTITY_TYPE]`` placeholders. Runs in ``.venv-baselines``.
"""

from __future__ import annotations

from bench.arms.base import cli, hf_revision, resolve_overlaps, versions

ARM = "presidio_transformers"
MODEL = "obi/deid_roberta_i2b2"
THRESHOLD = 0.4


def _conf() -> dict:
    from pathlib import Path

    import presidio_analyzer
    import yaml
    conf = yaml.safe_load((Path(presidio_analyzer.__file__).parent / "conf" / "transformers.yaml").read_text())
    conf["models"][0]["model_name"]["transformers"] = MODEL
    return conf


def load():
    import logging
    import warnings
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    for name in ("presidio-analyzer", "presidio_analyzer"):
        logging.getLogger(name).setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nlp = NlpEngineProvider(nlp_configuration=_conf()).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])

    def fn(text: str, _seed: int):
        res = analyzer.analyze(text=text, language="en", score_threshold=THRESHOLD)
        spans = [(max(0, r.start), min(len(text), r.end), r.entity_type, r.score) for r in res]
        spans = [s for s in spans if text[s[0]:s[1]].strip()]
        return [[s, e, t, f"[{t}]"] for s, e, t, _ in resolve_overlaps(spans)]
    return fn


def config() -> dict:
    return {"nlp_configuration": _conf(), "model_revision": hf_revision(MODEL), "score_threshold": THRESHOLD,
            "overlap_resolution": "longest span, then highest score",
            "replacement": "[ENTITY_TYPE] placeholder",
            "versions": versions("presidio-analyzer", "spacy", "spacy-huggingface-pipelines", "transformers", "torch")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
