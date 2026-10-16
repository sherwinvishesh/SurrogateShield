"""Arm ``llm_guard``: LLM Guard's ``input_scanners.Anonymize`` with a fresh
``Vault`` per message, the default recogniser configuration
(``DEBERTA_AI4PRIVACY_v2_CONF``: Isotonic/deberta-v3-base_finetuned_ai4privacy_v2
plus LLM Guard's regex recognisers, threshold 0.5) and ``use_faker=True``
(supported by the installed 0.3.16). Runs in ``.venv-baselines``.

``scan()`` returns only the sanitised text, so the spans are taken from the
same internal steps it runs (analyze → conflict removal → whitespace merge →
``_anonymize``), and every message is checked to reproduce ``_anonymize``'s
text exactly. LLM Guard's module-level Faker is re-seeded per message so the
run is reproducible; it draws a fresh value per occurrence (its behaviour).

0.3.16 bug: 18 entries of ``anonymize_helpers.faker._entity_faker_map``
(CRYPTO, NRP, UK_NHS, US_PASSPORT, ...) are ``lambda _: ...`` but are called
with no argument, so ``scan()`` raises ``TypeError`` whenever one of those
types is detected (of the default types, CRYPTO). ``fix_faker_arity`` wraps
exactly those entries to pass the argument they ignore, so the values are the
ones the authors wrote; the patched keys are recorded in the meta.

LLM Guard puts its NER pipeline on ``mps`` when Apple's GPU is present, and
MPS float results vary between runs: on the dev split one message gained or
lost six two-character IP_ADDRESS spans from run to run (the DeBERTa span
edges moved, so conflict removal kept different regex hits). The arm pins the
pipeline to the CPU, like every other arm, and records the device. The variation
remains on the CPU: two CPU runs of one configuration on devlarge-wildchat
(62f4356, b35777e) moved IP_ADDRESS spans on 11 of 870 messages (leaks
unchanged).
"""

from __future__ import annotations

from bench.arms.base import apply_edits, cli, versions

ARM = "llm_guard"


def fix_faker_arity(fmap: dict) -> list:
    """Wrap entries that need an argument so ``fmap[k]()`` works; return the keys wrapped."""
    import inspect
    fixed = []
    for k, f in list(fmap.items()):
        try:
            params = inspect.signature(f).parameters
        except (TypeError, ValueError):
            continue
        required = [q for q in params.values() if q.default is inspect.Parameter.empty
                    and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
        if required:
            def call(f=f):
                v = f(None)
                return v(None) if callable(v) else v  # IT_IDENTITY_CARD returns a lambda
            fmap[k] = call
            fixed.append(k)
    return sorted(fixed)


def pin_cpu() -> None:
    """Run the default recogniser's pipeline on the CPU (see the module note)."""
    import torch
    from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
    DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].pipeline_kwargs["device"] = torch.device("cpu")


def load():
    import logging
    from llm_guard.input_scanners import Anonymize
    from llm_guard.input_scanners.anonymize_helpers import faker as lg_faker
    from llm_guard.vault import Vault

    logging.getLogger("llm_guard").setLevel(logging.ERROR)
    fix_faker_arity(lg_faker._entity_faker_map)
    pin_cpu()
    try:
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
    except ImportError:
        pass
    a = Anonymize(Vault(), use_faker=True)

    def fn(text: str, seed: int):
        if not text.strip():
            return []
        res = a._analyzer.analyze(text=Anonymize.remove_single_quotes(text), language=a._language,
                                  entities=a._entity_types, allow_list=a._allowed_names,
                                  score_threshold=a._threshold)
        res = a._remove_conflicts_and_get_text_manipulation_data(res)
        merged = a._merge_entities_with_whitespace_between(text, res)
        lg_faker.fake.seed_instance(seed)
        sanitized, replaced = Anonymize._anonymize(text, merged, Vault(), a._use_faker)
        order = sorted(merged, reverse=True)  # the order _anonymize replaces in
        edits = [[e.start, e.end, e.entity_type, rep] for e, (rep, _orig) in zip(order, replaced)]
        if apply_edits(text, edits) != sanitized:
            raise RuntimeError("edits do not reproduce LLM Guard's sanitised text")
        return edits
    return fn


def config() -> dict:
    from llm_guard.input_scanners.anonymize import DEFAULT_ENTITY_TYPES
    from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
    return {"scanner": "llm_guard.input_scanners.Anonymize(Vault(), use_faker=True)",
            "recognizer_conf": "DEBERTA_AI4PRIVACY_v2_CONF",
            "model": DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].path,
            "model_revision": DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].revision,
            "entity_types": list(DEFAULT_ENTITY_TYPES), "threshold": 0.5,
            "device": "cpu (pinned; LLM Guard would pick mps or cuda)",
            "faker": "module Faker re-seeded per message (seed_instance)",
            "faker_arity_shim": "one-argument lambdas in _entity_faker_map called with a dummy argument (0.3.16 bug)",
            "versions": versions("llm-guard", "presidio-analyzer", "transformers", "torch", "faker")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
