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


def test_the_summary_always_shows_the_four_canonical_conditions():
    """A model that is neither the adapter nor the base leaves them 'not run'.

    The four rows are the question being asked, so they are printed whether or
    not a given invocation filled them in. A missing row would read as a zero.
    """
    text = summarise(probe(["mock:demo"]))
    rows = [
        line for line in text.splitlines()
        if "/" in line and ("corrected adapter" in line or "base model" in line)
    ]
    assert len(rows) == 4, rows
    assert all("not run" in line for line in rows)


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


# -------------------------------------------------- the four-condition summary

ADAPTER = "peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt"
BASE = "hf:Qwen/Qwen3-1.7B"
SOLVED_IDS = (
    "py_heldout_solved_generator_exhausted",
    "js_heldout_solved_debounce_closure",
)


def _synthetic(pairs):
    """A report shaped like a real run, without needing a GPU."""
    from scripts.probe_release_behavior import replication_check
    results = []
    for model, variant, confirms in pairs:
        for scenario_id in SOLVED_IDS:
            results.append({
                "model": model, "prompt_variant": variant,
                "scenario_id": scenario_id, "response": "x",
                "confirms": confirms, "asks_question": True,
                "explains": False, "error": None,
            })
    report = {"results": results}
    report["replication_check"] = replication_check(report)
    return report


def test_model_labels_are_stable_and_readable():
    from scripts.probe_release_behavior import model_label
    assert model_label(ADAPTER) == "corrected adapter"
    assert model_label(BASE) == "base model"


def test_the_summary_reports_all_four_conditions():
    report = _synthetic([
        (ADAPTER, "zero_shot", False),
        (ADAPTER, "zero_shot_plus_release_rule", True),
        (BASE, "zero_shot", False),
        (BASE, "zero_shot_plus_release_rule", True),
    ])
    text = summarise(report)
    for label in ("corrected adapter", "base model"):
        for variant in ("zero_shot", "zero_shot_plus_release_rule"):
            assert label in text and variant in text
    assert "0/2" in text and "2/2" in text


def test_replication_passes_when_condition_a_is_zero():
    report = _synthetic([(ADAPTER, "zero_shot", False)])
    assert report["replication_check"]["reproduces_baseline"] is True
    assert "PASS" in summarise(report)


def test_replication_fails_loudly_when_condition_a_is_not_zero():
    """A probe that does not reproduce the baseline must not be interpreted."""
    report = _synthetic([(ADAPTER, "zero_shot", True)])
    assert report["replication_check"]["reproduces_baseline"] is False
    text = summarise(report)
    assert "FAIL" in text
    assert "Do not interpret" in text


def test_the_expected_replication_value_is_the_measured_baseline():
    from scripts.probe_release_behavior import (
        REPLICATION_CONDITION,
        REPLICATION_EXPECTED,
    )
    assert REPLICATION_CONDITION == ("corrected adapter", "zero_shot")
    assert REPLICATION_EXPECTED == 0


def test_the_artifact_carries_the_replication_check():
    report = probe(["mock:demo"])
    assert "replication_check" in report
