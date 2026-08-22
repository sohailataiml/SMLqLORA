"""The free gate, and the verdict it is not allowed to talk itself out of.

Arm A costs twenty judge calls to evaluate properly, so a cheap gate decides
whether it earns them. The decision is computed from counts before anyone reads
the responses, because "it improved a bit" is exactly the kind of judgement a
tired experimenter makes generously.

The fake-success control carries as much weight as the solved cases. An arm that
confirms more often has not learned recognition if it also agrees with a learner
whose fix does not work -- that is a failure with its own name, not a partial win.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.probe_arm_a_free_gate import BASELINE, FAKE_SUCCESS_ID, classify

REPO_ROOT = Path(__file__).resolve().parent.parent


def _report(training: list[bool], heldout: list[bool],
            control_clean: bool = False, control_ack: bool = False):
    def rows(cleans):
        return [
            {"id": f"e{i}", "scores": {
                "strict_confirmation": False,
                "solved_state_acknowledgement": c,
                "followup_diagnostic_question": False,
                "clean_release": c,
            }}
            for i, c in enumerate(cleans)
        ]
    return {
        "training_examples": rows(training),
        "heldout_solved": rows(heldout),
        "fake_success_control": [{"id": FAKE_SUCCESS_ID, "scores": {
            "strict_confirmation": False,
            "solved_state_acknowledgement": control_ack or control_clean,
            "followup_diagnostic_question": False,
            "clean_release": control_clean,
        }}],
    }


# ------------------------------------------------------------- the verdicts

def test_no_improvement_anywhere_is_no_retention_gain():
    v = classify(_report([False] * 3, [False] * 2))
    assert v["verdict"] == "ARM_A_NO_RETENTION_GAIN"
    assert "do not run paid evaluation" in v["detail"].lower()


def test_training_improvement_alone_can_pass():
    v = classify(_report([True, False, False], [False] * 2))
    assert v["verdict"] == "ARM_A_FREE_GATE_PASS"


def test_heldout_improvement_alone_can_pass():
    v = classify(_report([False] * 3, [True, False]))
    assert v["verdict"] == "ARM_A_FREE_GATE_PASS"


def test_improvement_with_a_broken_control_is_discrimination_fail():
    """Confirming everyone is the failure this control exists to catch."""
    v = classify(_report([True] * 3, [True] * 2, control_clean=True))
    assert v["verdict"] == "ARM_A_DISCRIMINATION_FAIL"
    assert v["control_preserved"] is False


def test_merely_acknowledging_the_fake_success_also_fails():
    """Acknowledging a claimed-but-broken fix is premature confirmation."""
    v = classify(_report([True] * 3, [True] * 2, control_ack=True))
    assert v["verdict"] == "ARM_A_DISCRIMINATION_FAIL"


def test_no_gain_and_a_broken_control_says_both():
    v = classify(_report([False] * 3, [False] * 2, control_clean=True))
    assert v["verdict"] == "ARM_A_NO_RETENTION_GAIN"
    assert "second independent reason" in v["detail"]


def test_a_pass_requires_the_control_to_hold():
    v = classify(_report([True] * 3, [True] * 2))
    assert v["verdict"] == "ARM_A_FREE_GATE_PASS"
    assert v["control_preserved"] is True


# --------------------------------------------------------- frozen baselines

def test_the_baseline_is_the_published_corrected_v1():
    assert BASELINE["training_examples"] == {
        "strict": 1, "acknowledgement": 1, "clean_release": 0, "n": 3}
    assert BASELINE["heldout_solved"] == {
        "strict": 0, "acknowledgement": 0, "clean_release": 0, "n": 2}
    assert BASELINE["fake_success"] == {
        "strict": 0, "acknowledgement": 0, "clean_release": 0, "n": 1}


def test_the_verdict_reports_both_arms_counts():
    v = classify(_report([True, False, False], [False] * 2))
    assert v["arm_a"]["training_examples"]["clean_release"] == 1
    assert v["baseline_corrected_v1"] is BASELINE


# ----------------------------------------------------------- it costs nothing

def test_the_gate_makes_no_judge_calls():
    source = (REPO_ROOT / "scripts/probe_arm_a_free_gate.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("DeterministicJudge", "LLMJudge", "evaluation.judge",
                      "JUDGE_PARAMS", "Evaluator("):
        assert forbidden not in source, f"free gate must not reach for {forbidden}"


def test_it_uses_the_frozen_measure_module():
    from scripts import probe_arm_a_free_gate
    from analysis import retention_measures
    assert probe_arm_a_free_gate.score is retention_measures.score


def test_it_writes_outside_the_corrected_baseline_directory():
    from scripts.probe_arm_a_free_gate import DEFAULT_OUTPUT
    assert "n600_v1_baseline" not in str(DEFAULT_OUTPUT)
    assert DEFAULT_OUTPUT.parent.name == "arm_a_assistant_loss"


@pytest.mark.skipif(
    not (REPO_ROOT / "outputs/socratic-v1-n600-bestckpt/data/train.jsonl").exists(),
    reason="split not present locally",
)
def test_it_runs_end_to_end_on_a_mock_model():
    from scripts.probe_arm_a_free_gate import probe
    report = probe("mock:demo", REPO_ROOT / "outputs/socratic-v1-n600-bestckpt")
    assert len(report["training_examples"]) == 3
    assert len(report["heldout_solved"]) == 2
    assert len(report["fake_success_control"]) == 1
    assert all(r["error"] is None for r in report["training_examples"])
