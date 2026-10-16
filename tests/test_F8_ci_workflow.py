"""F8 — CI runs the offline test suite with a core coverage gate, and the
test workflow never deploys or publishes."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def test_F8_test_workflow_runs_suite_offline_with_coverage_gate():
    wf = load("tests.yml")
    job = wf["jobs"]["test"]
    assert job["env"]["HF_HUB_OFFLINE"] == "1" and job["env"]["TRANSFORMERS_OFFLINE"] == "1"
    run = " ".join(step.get("run", "") for step in job["steps"])
    assert "pytest tests python-library/tests" in run
    assert "--cov=python-library/surrogateshield/core" in run and "--cov-fail-under=85" in run
    assert wf["permissions"] == {"contents": "read"}


def test_F8_test_workflow_never_deploys_or_publishes():
    text = (WORKFLOWS / "tests.yml").read_text()
    for word in ("gh-pages", "pypi", "twine upload", "secrets.", "contents: write", "actions-gh-pages"):
        assert word not in text


def test_F8_test_workflow_builds_and_checks_the_package():
    """The package job builds the wheel and the sdist and runs twine's check
    (merge checklist); it is a plain job, so a failure fails the run."""
    jobs = load("tests.yml")["jobs"]
    runs = "\n".join(step.get("run", "") for step in jobs["package"]["steps"])
    assert "python -m build python-library" in runs
    assert "twine check --strict" in runs
    assert "upload" not in runs
    for job in jobs.values():
        assert "continue-on-error" not in job
        for step in job["steps"]:
            assert "continue-on-error" not in step
            assert "|| true" not in step.get("run", "")


def test_F8_deploy_workflows_still_only_fire_from_main_or_tags():
    for name in ("deploy.yml", "deploy-lib.yml"):
        on = load(name)[True]          # YAML 1.1 reads the key "on" as True
        assert on["push"]["branches"] == ["main"], name
