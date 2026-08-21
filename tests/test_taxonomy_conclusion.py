"""The conclusion must describe the run being analysed.

`build_report` hardcoded the MVP's explanation, so pointing it at the corrected
baseline produced an artifact asserting "the checkpoint must be corrected before
the V2 target is chosen" -- about a run whose entire purpose was that the
checkpoint had already been corrected. The measurements were right and the prose
contradicted them.

The verdict now follows the hypothesis tests and the narrative follows the
measured marker prevalence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.failure_taxonomy import (
    ATTRACTOR_DOMINANCE_THRESHOLD,
    attractor_dominates,
    build_conclusion,
    build_report,
    prevalence_note,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP = REPO_ROOT / "results/base_vs_tuned/judge_transcripts.jsonl"
CORRECTED = REPO_ROOT / "results/n600_v1_baseline/judge_transcripts.jsonl"

STALE = "must be corrected"
CANNOT_BE_READ = "cannot be read as a measurement of what Dataset V1 teaches"

ALL_REFUTED = [{"verdict": "REFUTED", "hypothesis": h} for h in "abcd"]


# ------------------------------------------------------------- the threshold

@pytest.mark.parametrize("marked,total,expected", [
    (13, 15, True),    # the MVP run
    (2, 10, False),    # the corrected run
    (5, 10, True),     # exactly at the threshold counts as dominance
    (4, 10, False),
    (0, 0, False),     # a run with no failures cannot be dominated by them
])
def test_dominance_is_a_documented_majority_rule(marked, total, expected):
    assert attractor_dominates(marked, total) is expected


def test_the_threshold_is_a_simple_majority():
    assert ATTRACTOR_DOMINANCE_THRESHOLD == 0.5


# --------------------------------------------- the regression that mattered

def test_a_low_marker_run_does_not_claim_the_checkpoint_needs_correcting():
    conclusion = build_conclusion(ALL_REFUTED, 2, 10)
    assert STALE not in conclusion["implication"]
    assert STALE not in conclusion["summary"]
    assert conclusion["checkpoint_attractor_dominates"] is False


def test_a_high_marker_run_still_says_the_checkpoint_dominates():
    """The MVP's explanation must remain available where it is true."""
    conclusion = build_conclusion(ALL_REFUTED, 13, 15)
    assert STALE in conclusion["implication"]
    assert conclusion["checkpoint_attractor_dominates"] is True


def test_the_prevalence_note_tracks_the_same_rule():
    assert CANNOT_BE_READ in prevalence_note(True)
    assert CANNOT_BE_READ not in prevalence_note(False)


def test_the_share_is_reported_so_the_claim_is_checkable():
    assert build_conclusion(ALL_REFUTED, 2, 10)["failures_with_checkpoint_marker_share"] == 0.2
    assert build_conclusion(ALL_REFUTED, 13, 15)["failures_with_checkpoint_marker_share"] == 0.8667


# ------------------------------------------------- verdict follows the tests

def test_verdict_is_not_implicated_when_every_hypothesis_is_refuted():
    assert build_conclusion(ALL_REFUTED, 2, 10)["verdict"] == (
        "DATASET_V1_NOT_IMPLICATED_BY_CURRENT_EVIDENCE"
    )


def test_verdict_changes_when_a_hypothesis_survives():
    mixed = [{"verdict": "REFUTED", "hypothesis": "a"},
             {"verdict": "SUPPORTED", "hypothesis": "the corpus lacks X"}]
    conclusion = build_conclusion(mixed, 2, 10)
    assert conclusion["verdict"] == "DATASET_V1_PARTIALLY_IMPLICATED"
    assert "the corpus lacks X" in conclusion["summary"]


# ------------------------------------------------------ end to end, real data

@pytest.mark.skipif(not CORRECTED.exists(), reason="corrected transcripts absent")
def test_the_corrected_run_produces_no_stale_conclusion():
    report = build_report(CORRECTED, run_label="corrected")
    assert STALE not in report["conclusion"]["implication"]
    assert CANNOT_BE_READ not in report["marker_prevalence"]["note"]
    assert report["counts"]["failures_with_checkpoint_marker"] == 2
    assert report["counts"]["failures"] == 10


def test_the_mvp_run_keeps_its_original_measurements_and_reading():
    report = build_report(MVP, run_label="mvp")
    assert report["counts"] == {
        "scenarios": 20, "passes": 5, "failures": 15,
        "failures_with_checkpoint_marker": 13,
    }
    assert report["conclusion"]["checkpoint_attractor_dominates"] is True
    assert STALE in report["conclusion"]["implication"]
