"""The release probe must be a diagnostic, and must not look like a result.

It answers one question -- can confirmation behaviour be elicited from the
corrected adapter -- and it runs on a GPU where a careless artifact could be
mistaken for a second baseline. So the things pinned here are mostly about what
it refuses to do: no judge calls, no writing near `results/n600_v1_baseline/`,
and an artifact that says on its face it is not comparable with anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompting.strategies import get_strategy
from scripts.probe_release_behavior import (
    DEFAULT_OUTPUT,
    RELEASE_RULE,
    main,
    probe,
    prompt_variants,
    solved_scenarios,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- what it probes

def test_it_probes_exactly_the_two_solved_scenarios():
    ids = sorted(s.id for s in solved_scenarios())
    assert ids == [
        "js_heldout_solved_debounce_closure",
        "py_heldout_solved_generator_exhausted",
    ]


def test_the_control_variant_is_the_frozen_zero_shot_prompt():
    """The control must be byte-identical to what produced the baseline."""
    assert prompt_variants()["zero_shot"] == get_strategy("zero_shot").system_prompt()


def test_the_treatment_differs_only_by_the_release_rule():
    variants = prompt_variants()
    assert variants["zero_shot_plus_release_rule"] == (
        variants["zero_shot"] + RELEASE_RULE
    )


def test_the_release_rule_states_confirmation_rather_than_questioning():
    assert "confirm" in RELEASE_RULE.lower()
    assert "instead of asking another question" in RELEASE_RULE.lower()


# ------------------------------------------------------- it is not a result

def test_the_artifact_declares_it_is_not_an_experiment():
    report = probe(["mock:demo"])
    assert report["artifact_status"] == "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT"
    assert "not claimable as a result" in report["note"]


def test_it_never_writes_into_the_baseline_directory():
    assert "n600_v1_baseline" not in str(DEFAULT_OUTPUT)
    assert DEFAULT_OUTPUT.parent.name == "solved_state_analysis"


def test_it_makes_no_judge_calls():
    """A probe that spent judge credit would be an experiment, not a probe."""
    source = (REPO_ROOT / "scripts/probe_release_behavior.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("DeterministicJudge", "LLMJudge", "evaluation.judge",
                      "JUDGE_PARAMS", "Evaluator("):
        assert forbidden not in source, f"probe must not reach for {forbidden}"


# ------------------------------------------------------------- it runs offline

def test_it_runs_end_to_end_on_a_mock_model():
    report = probe(["mock:demo"])
    rows = [r for r in report["results"] if "response" in r]
    assert len(rows) == 4  # 2 scenarios x 2 prompt variants
    assert all(r["error"] is None for r in rows)


def test_every_row_records_the_behaviour_flags():
    report = probe(["mock:demo"])
    for row in report["results"]:
        if "response" not in row:
            continue
        assert set(row) >= {"confirms", "asks_question", "explains", "prompt_variant"}


def test_an_unresolvable_model_is_reported_not_raised():
    report = probe(["nonsense:does-not-exist"])
    assert report["results"]
    assert report["results"][0]["error"]


def test_the_summary_counts_confirmations_per_variant():
    report = probe(["mock:demo"])
    text = summarise(report)
    assert "zero_shot" in text
    assert "/2" in text


def test_writing_the_artifact_stays_where_it_is_told(tmp_path):
    destination = tmp_path / "release_probe.json"
    main(["--model", "mock:demo", "--write", "--output", str(destination)])
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT"
    assert len([r for r in payload["results"] if "response" in r]) == 4


def test_it_does_not_write_unless_asked(tmp_path):
    before = DEFAULT_OUTPUT.exists()
    main(["--model", "mock:demo"])
    assert DEFAULT_OUTPUT.exists() == before
