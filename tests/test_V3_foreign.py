"""V3 §5.5 — SurrogateShield on non-English prompts (``bench/realdata/foreign.py``).
Model-free: invented messages, a stub send path and a hand-made trace."""

import json
from types import SimpleNamespace

from bench.realdata import foreign as F
from surrogateshield.core.entities import DetectedEntity

DE = ("Kannst du diese E-Mail für mich schreiben? Ich bin nicht sicher, wie ich das sagen soll, "
      "und es ist sehr wichtig für die Arbeit bei uns.")
EN = ("Can you help me write a short note to my landlord about the heating, it has been broken "
      "for a week and nobody has come to fix it yet.")


def _row(h, lang, text, toxic=False, role="user"):
    return {"conversation_hash": h, "language": lang, "toxic": toxic, "redacted": False,
            "conversation": [{"role": role, "content": text}]}


def _runner(text, seed):
    if text.startswith("Kannst"):
        clause = (0, len("Kannst du diese"))
        trace = [{"stage": "ner", "entities": [[*clause, "PERSON", "spacy", "confirmed"]]},
                 {"stage": "relation_gate", "dropped": [[*clause, "junk"]], "entities": []}]
        return SimpleNamespace(edits=[], confirmed=[]), trace
    s = text.index("landlord")
    ent = DetectedEntity(text="landlord", start=s, end=s + 8, type="PERSON", source="pii_tagger")
    return SimpleNamespace(edits=[(s, s + 8, "landlord", "Q")], confirmed=[ent]), []


def test_sample_filters_dedups_excludes_and_is_seeded():
    rows = [_row("a", "German", DE), _row("a", "German", DE), _row("b", "German", DE, toxic=True),
            _row("c", "German", "zu kurz"), _row("d", "German", DE, role="assistant"),
            _row("e", "German", DE), _row("f", "Japanese", DE), _row("g", "English", EN), _row("x", "English", EN)]
    s = F.sample(rows, per=5, exclude={"x"})
    assert [h for h, _t in s["German"]] in (["a", "e"], ["e", "a"]) and [h for h, _t in s["English"]] == ["g"]
    assert s == F.sample(rows, per=5, exclude={"x"}) and s["Dutch"] == [] and "Japanese" not in s
    assert len(F.sample(rows, per=1)["German"]) == 1


def test_foreign_fragment_drops_and_edits_are_counted_without_text(tmp_path):
    rows = [_row("de1", "German", DE), _row("de2", "German", DE + " Danke."), _row("en1", "English", EN)]
    out = tmp_path / "foreign.json"
    assert F.main(["--per", "10", "--out", str(out)], rows=rows, runner=_runner) == 0
    doc = json.loads(out.read_text())
    de, en = doc["results"]["German"], doc["results"]["English"]
    assert de["messages"] == 2 and de["gate_reads"] == "de" and de["gate_guess"] == {"de": 2}
    assert de["foreign_fragment_drops"] == 2 and de["model_drops_by_rule"] == {"junk": 2}
    assert de["edits"] == 0 and de["messages_with_foreign_fragment_drop"]["k"] == 2
    assert en["edits"] == 1 and en["entities_by_type"] == {"PERSON": 1} and en["gate_guess_right"]["k"] == 1
    text = out.read_text() + out.with_suffix(".md").read_text()
    assert "Kannst" not in text and "landlord" not in text and "| German | de | 2 |" in text
