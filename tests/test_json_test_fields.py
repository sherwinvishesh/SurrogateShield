"""Pre-publish audit (Pass 6, item 4): the README's JSON Test field table and
its off-by-default sentence are derived from ``json_tester.OUTPUT_FIELDS``
and ``DEFAULT_FIELDS``, and the attacker's "Generating compatible answers
files" table names fields the JSON Test can write."""
import re
from pathlib import Path

import pytest

from json_tester import DEFAULT_FIELDS, OUTPUT_FIELDS

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _section(title: str, until: str) -> str:
    start = README.index(title)
    return README[start:README.index(until, start)]


def _table_keys(section: str) -> list:
    return re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.M)


def test_readme_field_table_lists_every_output_field_in_order():
    section = _section("#### Available output fields", "### Offline detection evaluation")
    assert _table_keys(section) == [key for key, _ in OUTPUT_FIELDS]


def test_readme_names_exactly_the_fields_that_are_off_by_default():
    section = _section("#### Available output fields", "### Offline detection evaluation")
    sentence = re.search(r"^By default, (.+?) are \*\*off\*\*", section, flags=re.M)
    assert sentence, "the off-by-default sentence is missing"
    named = re.findall(r"`([a-z_]+)`", sentence.group(1))
    assert named == [key for key, _ in OUTPUT_FIELDS if not DEFAULT_FIELDS[key]]


def test_attacker_compatibility_table_names_the_fields_the_attacker_reads():
    """The table lists exactly the answers-file fields attacker.py reads
    (the arm texts and the two "what was redacted" lists), besides
    ``question``, which the key file also carries."""
    source = (Path(__file__).resolve().parent.parent / "attacker.py").read_text(encoding="utf-8")
    read = set(re.findall(r'a\.get\("([a-z_]+)"\)', source)) | set(
        re.findall(r'"(?:ss|presidio)":\s*"([a-z_]+)"', source))
    read.discard("question")
    section = _section("### Generating compatible answers files", "### Test coverage")
    keys = _table_keys(section)
    assert set(keys) == read
    assert set(keys) <= {key for key, _ in OUTPUT_FIELDS}

def test_experiment_readme_field_table_lists_every_output_field_in_order():
    text = (Path(__file__).resolve().parent.parent / "experiment" / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Output Fields"):]
    assert _table_keys(section) == [key for key, _ in OUTPUT_FIELDS]


@pytest.mark.parametrize("path", ["README.md", "experiment/README.md"])
def test_readmes_document_the_extra_row_keys(path):
    text = (Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8")
    for key in ("`address_mode`", "`error`", "`error_type`"):
        assert key in text, (path, key)



# ── Pass 7: the offline evaluator's README section is derived from the code ──

def _offline_eval_source() -> str:
    return (Path(__file__).resolve().parent.parent / "offline_eval.py").read_text(encoding="utf-8")


def test_readme_offline_eval_commands_use_flags_the_parser_has():
    documented = set()
    for line in re.findall(r"^python offline_eval\.py (.+)$", README, flags=re.M):
        documented |= set(re.findall(r"(--[a-z-]+)", line))
    parser = set(re.findall(r'add_argument\("(--[a-z-]+)"', _offline_eval_source()))
    assert documented and documented <= parser, documented - parser


def test_readme_ablation_table_lists_the_evaluators_configurations():
    """The ablation rows are the configurations offline_eval re-runs, in
    order, each with the cascade options it switches off; the evaluation
    screen's row names the command the Ablation panel prints."""
    from evaluator import ABLATION_COMMAND
    from offline_eval import ABLATION_CONFIGS

    section = _section("#### Ablation study (Table 4)", "## Attacker Experiment")
    rows = [r for r in re.findall(r"^\| (.+?) \| (.+?) \|$", section, flags=re.M)
            if r[0] != "Configuration"]
    expected = [(label, ", ".join(f"`{k}={v}`" for k, v in opts.items()) or "none")
                for _name, label, opts in ABLATION_CONFIGS]
    assert rows == expected
    assert "--ablation" in section and "post-hoc" not in section
    assert f"`{ABLATION_COMMAND}`" in _section("#### Evaluation metrics reported", "#### Presidio comparison table")
