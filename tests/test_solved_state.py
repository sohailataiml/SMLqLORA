"""The solved-state forensic analysis, pinned to the frozen corpus.

The decision this analysis feeds -- whether Dataset V2 is justified -- turns on
counts. So the counts are asserted here against the frozen data rather than
against a previous run's output, and the classification and neighbour ordering
are pinned as deterministic. A number that silently drifts would change a
research conclusion without anyone noticing.

These tests also guard the corpus itself: nothing in this analysis may mutate
Dataset V1, `scenarios/`, `behavior/`, or the MVP evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.solved_state import (
    CORE_SIGNALS,
    confirmation_behaviour,
    heldout_solved_profiles,
    load_heldout_solved,
    load_v1_all,
    load_v1_solved,
    nearest_neighbours,
    recognition_category,
    signal_profile,
    solved_corpus_statistics,
    solved_examples_in_training_split,
    tutor_profile,
    v1_prior_turns,
    v1_student_message,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO_ROOT / "results/n600_v1_baseline/judge_transcripts.jsonl"

needs_transcripts = pytest.mark.skipif(
    not TRANSCRIPTS.exists(), reason="corrected-baseline transcripts absent"
)


# --------------------------------------------------------------- the corpus

def test_exactly_85_solved_examples():
    assert len(load_v1_solved()) == 85


def test_dataset_v1_is_still_600_records():
    assert len(load_v1_all()) == 600


def test_prior_turn_distribution_reproduces_the_frozen_corpus():
    """46/26/13 exchanges, counted here as 2/4/6 conversation messages."""
    stats = solved_corpus_statistics()
    assert stats["prior_turn_distribution"] == {2: 46, 4: 26, 6: 13}


def test_every_solved_example_is_labelled_solved():
    assert all(
        r["dimensions"]["student_progress"] == "solved" for r in load_v1_solved()
    )


def test_solved_examples_are_mostly_in_the_training_split():
    """A demonstration in the validation split was never learned from."""
    split = solved_examples_in_training_split()
    if not split.get("available"):
        pytest.skip("training split not present locally")
    assert split["solved_total"] == 85
    assert split["solved_in_training_split"] == 75
    assert split["solved_in_validation_split"] == 10


# ------------------------------------------------------- the two held-out cases

def test_both_heldout_solved_scenarios_are_found():
    ids = sorted(s["id"] for s in load_heldout_solved())
    assert ids == [
        "js_heldout_solved_debounce_closure",
        "py_heldout_solved_generator_exhausted",
    ]


def test_both_heldout_solved_carry_every_core_signal():
    """They are the clearest possible solved reports, not marginal ones."""
    for held in heldout_solved_profiles():
        assert held["core_signal_count"] == len(CORE_SIGNALS)
        assert held["recognition_category"] == "diagnosis_change_and_success"


def test_heldout_solved_sit_at_a_depth_v1_covers():
    stats = solved_corpus_statistics()
    for held in heldout_solved_profiles():
        assert stats["prior_turn_distribution"].get(held["prior_turns"], 0) > 0


# ------------------------------------------------------------- classification

def test_classification_is_deterministic():
    first = [h["recognition_category"] for h in heldout_solved_profiles()]
    second = [h["recognition_category"] for h in heldout_solved_profiles()]
    assert first == second


def test_categories_are_built_from_the_core_signals():
    assert recognition_category(
        {"states_diagnosis": True, "describes_code_change": True,
         "claims_success": True}
    ) == "diagnosis_change_and_success"
    assert recognition_category(
        {"states_diagnosis": False, "describes_code_change": False,
         "claims_success": False}
    ) == "no_core_signal"


def test_signal_profile_counts_words():
    assert signal_profile("one two three")["word_count"] == 3


def test_every_solved_example_receives_a_category():
    stats = solved_corpus_statistics()
    assert sum(stats["recognition_categories"].values()) == 85


# ------------------------------------------------------------------ neighbours

def test_neighbour_ordering_is_deterministic():
    solved = load_v1_solved()
    corpus = [v1_student_message(r) for r in solved]
    query = heldout_solved_profiles()[0]["student_message"]
    assert nearest_neighbours(query, corpus, 5) == nearest_neighbours(query, corpus, 5)


def test_neighbours_are_sorted_by_descending_similarity():
    solved = load_v1_solved()
    corpus = [v1_student_message(r) for r in solved]
    query = heldout_solved_profiles()[0]["student_message"]
    scores = [s for _, s in nearest_neighbours(query, corpus, 5)]
    assert scores == sorted(scores, reverse=True)


def test_each_heldout_case_has_a_same_category_neighbour():
    """The claim that V1 contains analogous demonstrations, asserted."""
    solved = load_v1_solved()
    corpus = [v1_student_message(r) for r in solved]
    for held in heldout_solved_profiles():
        top = nearest_neighbours(held["student_message"], corpus, 5)
        categories = {
            solved[i]["dimensions"]["bug_category"] for i, _ in top
        }
        assert held["bug_category"] in categories or any(
            recognition_category(signal_profile(corpus[i]))
            == held["recognition_category"]
            for i, _ in top
        )


# ------------------------------------------------- what the tutor targets model

def test_v1_solved_targets_overwhelmingly_confirm():
    stats = solved_corpus_statistics()
    assert stats["tutor_behaviour"]["confirms"] == 82
    assert stats["tutor_behaviour"]["confirms_without_question"] == 79


def test_confirmation_detector_recognises_a_v1_style_release():
    assert tutor_profile("That's exactly right, and your fix is correct.")["confirms"]


def test_confirmation_detector_does_not_fire_on_a_probe():
    profile = tutor_profile("Good, that's the right direction. Now what does apply do?")
    assert profile["asks_question"]
    assert not profile["confirms"]


# ------------------------------------------------------- the corrected model

@needs_transcripts
def test_the_corrected_run_has_twenty_records():
    from analysis.solved_state import corrected_transcripts
    assert len(corrected_transcripts()) == 20


@needs_transcripts
def test_the_corrected_model_never_confirms():
    """The finding the V2 decision rests on."""
    behaviour = confirmation_behaviour()
    assert behaviour["outputs"] == 20
    assert behaviour["confirms"] == 0


# ------------------------------------------------------------ no mutation

FROZEN = [
    "data/versions/v1/selected.jsonl",
    "data/versions/v1/freeze.json",
    "scenarios/heldout.jsonl",
    "scenarios/clean.jsonl",
    "scenarios/adversarial.jsonl",
    "behavior/spec.yaml",
    "results/base_vs_tuned/judge_transcripts.jsonl",
    "results/base_vs_tuned/RESULTS_SUMMARY.json",
    "results/failure_analysis/v1_n600_failure_taxonomy.json",
    "SUBMISSION.md",
]


@pytest.fixture
def frozen_digests():
    return {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in FROZEN}


def test_the_analysis_mutates_nothing_frozen(frozen_digests, tmp_path):
    """Running the whole report must leave every frozen artifact byte-identical."""
    from analysis.solved_state_report import main
    main(["--write", "--output-dir", str(tmp_path)])
    after = {
        p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in FROZEN
    }
    assert after == frozen_digests


def test_the_dataset_still_matches_its_frozen_hash():
    freeze = json.loads(
        (REPO_ROOT / "data/versions/v1/freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["dataset_hash"] == (
        "9121c24e47c7253818040aa40356a67d3a359ddcec057bc5bfc533d6a77e2656"
    )


def test_artifacts_are_written_outside_the_frozen_directories(tmp_path):
    from analysis.solved_state_report import main
    main(["--write", "--output-dir", str(tmp_path)])
    written = {p.name for p in tmp_path.glob("*.json")}
    assert "report.json" in written
    assert "v2_decision.json" in written


# ------------------------------------------------------- counts recompute

def test_report_counts_recompute_from_source(tmp_path):
    from analysis.solved_state_report import main
    main(["--write", "--output-dir", str(tmp_path)])
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["v1_solved_statistics"]["count"] == len(load_v1_solved())
    assert len(report["heldout_solved"]) == len(load_heldout_solved())
    assert sum(
        report["v1_solved_statistics"]["recognition_categories"].values()
    ) == 85
