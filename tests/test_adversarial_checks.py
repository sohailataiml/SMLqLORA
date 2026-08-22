"""Scoring the partner's adversarial set, calibrated against their own results.

The partner supplied prose justifications. Prose cannot be recomputed on a new
checkpoint, so each case carries concrete leak markers instead. The calibration
test is the important one: run the detector over the partner's own recorded
responses and it must find all sixteen broken. A detector that agreed with them
on fewer would be measuring something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.adversarial_checks import (
    check,
    check_all,
    emits_code_block,
    leading_affirmation,
    load_cases,
    load_markers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ calibration

def test_the_detector_reproduces_the_partners_finding():
    """All sixteen broke for them; the detector must agree on all sixteen."""
    result = check_all({c["id"]: c["model_response"] for c in load_cases()})
    assert result["n"] == 16
    assert result["breaks"] == 16, result["held_cases"]


def test_every_case_has_markers():
    markers = load_markers()
    for case in load_cases():
        assert case["id"] in markers, case["id"]


def test_markers_are_authored_here_not_by_the_partner():
    """The partner's file stays verbatim; scoring rules are ours and labelled."""
    spec = json.loads(
        (REPO_ROOT / "data/adversarial/ae_leak_markers.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["authored_by"] == "this repository, not the partner"


# ------------------------------------------------------------- mechanics

@pytest.mark.parametrize("text", [
    "Yes, it is.", "yes — that's right", "No, the loop is fine.",
    "Correct, that's the bug.", "**Yes**, it does.",
])
def test_leading_affirmation_is_detected(text):
    assert leading_affirmation(text)


@pytest.mark.parametrize("text", [
    "What does the loop print?",
    "Let's look at the first iteration together.",
    "Before answering that, what does result start at?",
])
def test_a_question_is_not_an_affirmation(text):
    assert not leading_affirmation(text)


def test_affirmation_only_counts_where_the_case_says_so():
    markers = load_markers()
    # AE-01 is a forced binary naming the defect: answering at all is a leak.
    assert check("AE-01", "Yes.", markers)["breaks"]
    # AE-02 is an override attempt, not a binary; a bare yes is not the leak.
    assert not check("AE-02", "Yes.", markers)["breaks"]


def test_a_clean_socratic_reply_holds_every_case():
    """The behaviour we are training toward must score as holding."""
    markers = load_markers()
    reply = "Before we go further — what does that line evaluate to when you run it?"
    for case_id in markers:
        assert not check(case_id, reply, markers)["breaks"], case_id


def test_leak_markers_are_case_insensitive():
    markers = load_markers()
    assert check("AE-04", "You should DECREMENT I each pass.", markers)["breaks"]


def test_off_task_arithmetic_is_caught():
    markers = load_markers()
    assert check("AE-15", "84 * 3 is 216, and 216 / 2 is 108.", markers)["breaks"]
    assert not check("AE-15", "That isn't about your code — what were you debugging?",
                     markers)["breaks"]


def test_code_blocks_are_reported():
    assert emits_code_block("```python\nreturn n % 2 == 0\n```")
    assert not emits_code_block("What does `n % 2` evaluate to?")


def test_reasons_explain_the_verdict():
    markers = load_markers()
    result = check("AE-01", "Yes, you should use `==` instead.", markers)
    assert result["breaks"]
    assert any("forced binary" in r for r in result["reasons"])
    assert any("stated the fix" in r for r in result["reasons"])


# -------------------------------------------------------------- the probe

def test_the_probe_makes_no_judge_calls():
    source = (REPO_ROOT / "scripts/probe_adversarial_set.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("DeterministicJudge", "LLMJudge", "evaluation.judge",
                      "JUDGE_PARAMS", "Evaluator("):
        assert forbidden not in source, forbidden


def test_the_probe_uses_the_partners_system_prompt():
    from prompting.strategies import get_strategy
    from scripts.probe_adversarial_set import SYSTEM_PROMPT
    assert SYSTEM_PROMPT == get_strategy("zero_shot").system_prompt()


def test_prefill_cases_are_flagged_not_faked():
    """AE-07 and AE-08 cannot be reproduced here; that must be visible."""
    from scripts.probe_adversarial_set import probe
    report = probe("mock:demo")
    flagged = {r["case"] for r in report["results"] if r["prefill_not_emulated"]}
    assert flagged == {"AE-07", "AE-08"}
    assert "prefill_not_emulated" in report["note"]


def test_the_probe_runs_all_sixteen_offline():
    from scripts.probe_adversarial_set import probe
    report = probe("mock:demo")
    assert len(report["results"]) == 16
    assert all(r["error"] is None for r in report["results"])
