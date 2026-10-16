"""Arm ``presidio_faker``: the ``presidio_default`` detection, each entity
replaced by a Faker value of the same type through Presidio's anonymizer
``custom`` operator. Faker is seeded per message, and a per-message dictionary
maps the same (type, value) to the same surrogate. This is Presidio's
"synthesize" idea without an LLM: Presidio's own synthesize sample calls
OpenAI, which needs a key and is not local. Runs in ``.venv``.
"""

from __future__ import annotations

import sys

from bench.arms.base import ROOT, apply_edits, cli, versions

ARM = "presidio_faker"


def shape_like(value: str, rnd) -> str:
    """Same shape: each digit → a random digit, each letter → a random letter
    of the same case; everything else kept."""
    out = []
    for c in value:
        if c.isdigit():
            out.append(str(rnd.randrange(10)))
        elif c.isalpha() and c.isascii():
            ch = chr(rnd.randrange(26) + ord("a"))
            out.append(ch.upper() if c.isupper() else ch)
        else:
            out.append(c)
    return "".join(out)


def generators(fake) -> dict:
    return {
        "PERSON": lambda v: fake.name(),
        "EMAIL_ADDRESS": lambda v: fake.email(),
        "PHONE_NUMBER": lambda v: fake.phone_number(),
        "LOCATION": lambda v: fake.city(),
        "DATE_TIME": lambda v: fake.date(),
        "NRP": lambda v: fake.country(),
        "ORGANIZATION": lambda v: fake.company(),
        "CREDIT_CARD": lambda v: fake.credit_card_number(),
        "IBAN_CODE": lambda v: fake.iban(),
        "IP_ADDRESS": lambda v: fake.ipv6() if ":" in v else fake.ipv4(),
        "URL": lambda v: fake.url(),
        "US_SSN": lambda v: fake.ssn(),
        "US_BANK_NUMBER": lambda v: fake.bban(),
    }


def load():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from faker import Faker
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import ConflictResolutionStrategy, OperatorConfig, RecognizerResult

    from presidio.detect import detect

    engine = AnonymizerEngine()
    CONFLICT = ConflictResolutionStrategy.MERGE_SIMILAR_OR_CONTAINED  # anonymize()'s default
    fake = Faker("en_US")
    gens = generators(fake)

    def fn(text: str, seed: int):
        ents = detect(text)
        if ents is None:
            raise RuntimeError("presidio-analyzer or en_core_web_lg is not installed")
        if not ents:
            return []
        fake.seed_instance(seed)
        memo: dict = {}

        def surrogate(etype):
            gen = gens.get(etype)

            def op(value: str) -> str:
                key = (etype, value)
                if key not in memo:
                    memo[key] = gen(value) if gen else shape_like(value, fake.random)
                return memo[key]
            return op

        results = [RecognizerResult(e.entity_type, e.start, e.end, e.score) for e in ents]
        ops = {t: OperatorConfig("custom", {"lambda": surrogate(t)}) for t in {e.entity_type for e in ents}}
        out = engine.anonymize(text=text, analyzer_results=results, operators=ops)
        # The spans the engine replaced: its own steps (sort, conflict removal,
        # merge of same-type entities separated only by spaces) on a fresh copy.
        spans = sorted((RecognizerResult(e.entity_type, e.start, e.end, e.score) for e in ents),
                       key=lambda r: (r.start, r.end))
        spans = engine._remove_conflicts_and_get_text_manipulation_data(spans, CONFLICT)
        spans = engine._merge_entities_with_whitespace_between(text, spans)
        items = sorted(out.items, key=lambda i: i.start)
        if len(items) != len(spans):
            raise RuntimeError("anonymizer output does not match its merged spans")
        edits = [[r.start, r.end, r.entity_type, i.text] for r, i in zip(spans, items)]
        if apply_edits(text, edits) != out.text:
            raise RuntimeError("edit alignment does not reproduce the anonymizer's text")
        return edits
    return fn


def config() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from presidio.engine import baseline_config
    return {**baseline_config(),
            "merge": "AnonymizerEngine default: same-type entities separated only by spaces become one",
            "replacement": "presidio_anonymizer custom operator: Faker('en_US') seeded per message, "
                           "per-message (type, value) memo; types without a Faker provider keep their shape "
                           "(digit→digit, letter→letter)",
            "faker_types": sorted(generators(None).keys()),
            "versions": versions("presidio-anonymizer", "faker")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
