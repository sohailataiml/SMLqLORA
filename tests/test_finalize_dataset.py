"""Tests for the dataset audit, freeze, human-review export and subsets.

These run entirely on a mock teacher and the offline judge, so the whole
finalization path is exercised without an API key.
"""

from __future__ import annotations

import csv
import json

import pytest

from evaluation.judge import DeterministicJudge
from filtering.dataset_card import render_dataset_card
from filtering.quality_gate import run_quality_gate
from generation.generate import build_teacher, generate
from scripts.finalize_dataset import (
    audit,
    build_freeze,
    learner_state,
    prepare_subsets,
    select_review_sample,
    write_human_review,
)


@pytest.fixture(scope="module")
def gated():
    """A small real run of the whole pipeline on the mock teacher."""
    teacher = build_teacher("mock:teacher", mock=True, dataset_version="vtest")
    candidates, _, _ = generate(count=40, teacher=teacher, verbose=False)
    outcome = run_quality_gate(
        candidates, DeterministicJudge(), dataset_version="vtest"
    )
    return candidates, outcome


# =============================================================================
# Learner state
# =============================================================================


def test_learner_state_distinguishes_the_three_cases(gated):
    candidates, _ = gated
    states = {learner_state(c) for c in candidates}
    assert states <= {"unresolved", "almost_correct", "solved"}


def test_solved_scenarios_are_labelled_solved(gated):
    candidates, _ = gated
    for c in candidates:
        if c.scenario.student_has_solved:
            assert learner_state(c) == "solved"


# =============================================================================
# Audit
# =============================================================================


def test_audit_counts_reconcile(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))

    counts = report["counts"]
    assert counts["accepted"] + counts["rejected"] == len(candidates)
    assert counts["candidates_generated"] == len(candidates)


def test_audit_acceptance_rate_matches_counts(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    expected = round(len(outcome.accepted) / len(candidates), 4)
    assert report["counts"]["acceptance_rate"] == expected


def test_audit_distribution_shares_sum_to_one(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    for key in ("language_share", "difficulty_share", "pressure_type_share"):
        shares = report["distribution"][key]
        assert abs(sum(shares.values()) - 1.0) < 0.01, key


def test_audit_behavioral_coverage_partitions_the_dataset(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    cov = report["behavioral_coverage"]
    total = cov["unresolved_count"] + cov["almost_correct_count"] + cov["solved_count"]
    assert total == len(outcome.accepted)


def test_audit_reports_zero_duplicates_after_the_gate(gated):
    """The gate dedupes; the audit independently confirms none survived."""
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    assert report["diversity"]["exact_duplicates_in_accepted"] == 0
    assert report["diversity"]["near_duplicates_in_accepted"] == 0


def test_audit_checks_contamination_against_every_eval_split(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    assert len(report["contamination"]["checked_against"]) == 3
    assert report["contamination"]["eval_scenarios_checked"] > 0


def test_audit_records_provenance(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    prov = report["provenance"]
    assert prov["teacher_model"]
    assert prov["behavior_spec_version"]
    assert prov["behavior_spec_sha256"]


def test_audit_rejection_rates_never_exceed_one(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    for rate in report["rejections"]["by_reason_rate"].values():
        assert 0.0 <= rate <= 1.0


# =============================================================================
# Nested subsets
# =============================================================================


def test_subsets_are_nested(gated, tmp_path):
    _, outcome = gated
    accepted = outcome.accepted
    sizes = [2, 4, len(accepted)]
    report = prepare_subsets(accepted, sizes, tmp_path)

    assert report["nesting_verified"] is True
    assert all(check["passed"] for check in report["nesting_checks"])


def test_subset_nesting_is_a_prefix_relationship(gated, tmp_path):
    _, outcome = gated
    accepted = outcome.accepted
    report = prepare_subsets(accepted, [2, 4], tmp_path)

    small = json.loads((tmp_path / "subset_2.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    large = json.loads((tmp_path / "subset_4.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert small == large, "the larger subset must start with the smaller one"


def test_oversized_request_adapts_instead_of_duplicating(gated, tmp_path):
    _, outcome = gated
    accepted = outcome.accepted
    huge = len(accepted) + 500
    report = prepare_subsets(accepted, [2, huge], tmp_path)

    assert huge in report["sizes_unavailable"]
    assert max(s["n"] for s in report["subsets"]) == len(accepted)
    assert "no example was duplicated" in report["adaptation_note"]


def test_subsets_write_one_file_per_size(gated, tmp_path):
    _, outcome = gated
    prepare_subsets(outcome.accepted, [2, 4], tmp_path)
    assert (tmp_path / "subset_2.jsonl").exists()
    assert (tmp_path / "subset_4.jsonl").exists()


def test_subsets_record_drift_against_the_full_set(gated, tmp_path):
    _, outcome = gated
    report = prepare_subsets(outcome.accepted, [4, len(outcome.accepted)], tmp_path)
    full = report["subsets"][-1]
    # The full-size subset is the dataset, so its drift is zero by definition.
    assert full["max_abs_drift_vs_full"] == 0.0


# =============================================================================
# Freeze
# =============================================================================


def test_freeze_records_hash_and_provenance(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    freeze = build_freeze(outcome.accepted, report, "vtest")

    assert freeze["frozen"] is True
    assert freeze["accepted_count"] == len(outcome.accepted)
    assert len(freeze["dataset_hash"]) == 64
    assert freeze["behavior_spec_sha256"]


def test_freeze_hash_is_stable_across_calls(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    a = build_freeze(outcome.accepted, report, "vtest")
    b = build_freeze(outcome.accepted, report, "vtest")
    assert a["dataset_hash"] == b["dataset_hash"]


def test_freeze_hash_changes_when_the_dataset_changes(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    full = build_freeze(outcome.accepted, report, "vtest")
    fewer = build_freeze(outcome.accepted[:-1], report, "vtest")
    assert full["dataset_hash"] != fewer["dataset_hash"]


def test_freeze_hash_ignores_input_ordering(gated):
    """Order must not change the hash, or a re-run would look like a new dataset."""
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    forward = build_freeze(outcome.accepted, report, "vtest")
    backward = build_freeze(list(reversed(outcome.accepted)), report, "vtest")
    assert forward["dataset_hash"] == backward["dataset_hash"]


# =============================================================================
# Human review sheet
# =============================================================================


def test_human_review_columns_are_left_blank(gated, tmp_path):
    _, outcome = gated
    path = tmp_path / "human_review.csv"
    write_human_review(outcome.accepted, path, target=10)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows
    assert all(r["human_pass"] == "" for r in rows)
    assert all(r["human_notes"] == "" for r in rows)


def test_human_review_includes_the_required_columns(gated, tmp_path):
    _, outcome = gated
    path = tmp_path / "human_review.csv"
    write_human_review(outcome.accepted, path, target=10)

    header = list(csv.DictReader(path.open(encoding="utf-8")).fieldnames or [])
    for column in ("candidate_id", "language", "bug_category", "pressure_type",
                   "student_state", "conversation", "assistant_response",
                   "automatic_pass", "human_pass", "human_notes"):
        assert column in header


def test_human_review_sample_is_deterministic(gated):
    _, outcome = gated
    a = [e.id for e in select_review_sample(outcome.accepted, target=10)]
    b = [e.id for e in select_review_sample(outcome.accepted, target=10)]
    assert a == b


def test_human_review_sample_has_no_duplicates(gated):
    _, outcome = gated
    sample = select_review_sample(outcome.accepted, target=12)
    ids = [e.id for e in sample]
    assert len(ids) == len(set(ids))


# =============================================================================
# Dataset card
# =============================================================================


def test_dataset_card_renders_from_the_audit(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    report["nested_subsets"] = {"materialized_sizes": [2, 4],
                                "nesting_verified": True,
                                "adaptation_note": ""}
    report["freeze"] = build_freeze(outcome.accepted, report, "vtest")

    card = render_dataset_card(report, "vtest")
    assert "Dataset vtest" in card
    assert str(len(outcome.accepted)) in card
    assert "No model has been trained" in card


def test_dataset_card_makes_no_training_effectiveness_claim(gated):
    candidates, outcome = gated
    report = audit(outcome.accepted, outcome.rejected, len(candidates))
    report["nested_subsets"] = {"materialized_sizes": [], "nesting_verified": True,
                                "adaptation_note": ""}
    report["freeze"] = build_freeze(outcome.accepted, report, "vtest")

    card = render_dataset_card(report, "vtest").lower()
    assert "fine-tuning improved" not in card
    assert "does not claim" in card
