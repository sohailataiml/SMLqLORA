"""The V1 failure taxonomy, and the dataset invariants its argument rests on.

The conclusion drawn in `results/failure_analysis/README.md` — that the N=600
diagnostic regression is a checkpoint artifact rather than a Dataset V1 defect —
stands on two claims about the frozen data: no V1 tutor response repeats a
sentence, and none claims prior work. If either ever became false the argument
would collapse silently, so both are asserted here against the real file.
"""

from __future__ import annotations

import json

import pytest

from analysis.corpus import (
    DATASET_V1,
    REPO_ROOT,
    count_tutor_turns,
    load_dataset_v1,
    load_heldout,
    ngrams,
    out_of_distribution_phrases,
    read_jsonl,
    repeated_sentence_count,
    split_by_model,
)
from analysis.failure_taxonomy import (
    CHECKPOINT_MARKERS,
    _FABRICATED_PRIOR_WORK,
    build_report,
    classify,
)

PLAN_JSON = REPO_ROOT / "data/versions/v2/plan.json"
TAXONOMY_JSON = REPO_ROOT / "results/failure_analysis/v1_n600_failure_taxonomy.json"


# --------------------------------------------------------------- primitives


def test_repeated_sentence_count_is_zero_for_distinct_sentences():
    assert repeated_sentence_count("What runs first? Then what changes?") == 0


def test_repeated_sentence_count_counts_each_repeat():
    text = "What is i? What is i? What is i?"
    assert repeated_sentence_count(text) == 2


def test_count_tutor_turns_ignores_user_messages():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert count_tutor_turns(history) == 1


def test_count_tutor_turns_of_empty_history_is_zero():
    assert count_tutor_turns([]) == 0


def test_ngrams_are_lowercased_word_tuples():
    grams = ngrams("The Loop Steps By Three", n=5)
    assert grams == {("the", "loop", "steps", "by", "three")}


def test_out_of_distribution_needs_the_minimum_recurrence():
    tuned = ["one two three four five", "one two three four five"]
    hits = out_of_distribution_phrases(tuned, ["nothing in common here at all"],
                                       min_responses=3)
    assert hits == []


def test_out_of_distribution_reports_a_recurring_absent_phrase():
    tuned = ["one two three four five"] * 3
    hits = out_of_distribution_phrases(tuned, ["nothing in common here at all"],
                                       min_responses=3)
    assert hits[0]["phrase"] == "one two three four five"
    assert hits[0]["tuned_responses"] == 3


def test_out_of_distribution_excludes_phrases_present_in_the_reference():
    tuned = ["one two three four five"] * 3
    hits = out_of_distribution_phrases(tuned, ["one two three four five"],
                                       min_responses=3)
    assert hits == []


def test_split_by_model_separates_adapter_records():
    records = [{"model": "hf:Qwen/Qwen3-1.7B"}, {"model": "peft:Qwen/Qwen3-1.7B+out"}]
    base, tuned = split_by_model(records)
    assert len(base) == 1 and len(tuned) == 1


# ------------------------------------------------------- dataset invariants


@pytest.fixture(scope="module")
def dataset_v1():
    return load_dataset_v1()


def test_dataset_v1_is_the_frozen_600(dataset_v1):
    assert len(dataset_v1) == 600


def test_student_has_solved_is_read_from_the_scenario_not_the_dimensions(dataset_v1):
    """`dimensions` has `student_progress`; `scenario` has `student_has_solved`.

    Reading the wrong one yields None for all 600 rows and silently reports zero
    solved-state coverage, which is how the coverage table was wrong on the first
    pass.
    """
    solved = [e for e in dataset_v1 if e.student_has_solved]
    assert len(solved) == 85


def test_no_v1_tutor_response_repeats_a_sentence(dataset_v1):
    offenders = [e.id for e in dataset_v1 if repeated_sentence_count(e.response) > 0]
    assert offenders == [], (
        "The taxonomy treats sentence repetition as a marker Dataset V1 cannot "
        f"produce. These V1 examples break that: {offenders[:5]}"
    )


def test_no_v1_tutor_response_claims_prior_work(dataset_v1):
    offenders = [e.id for e in dataset_v1 if _FABRICATED_PRIOR_WORK.search(e.response)]
    assert offenders == [], (
        "The taxonomy treats claimed prior work as a marker Dataset V1 cannot "
        f"produce. These V1 examples break that: {offenders[:5]}"
    )


def test_v1_first_turn_share_is_the_documented_minority(dataset_v1):
    first = sum(1 for e in dataset_v1 if e.is_first_turn)
    assert first == 121


def test_heldout_is_mostly_first_turn():
    heldout = load_heldout()
    first = sum(
        1
        for s in heldout.values()
        if count_tutor_turns(s.get("conversation_history", [])) == 0
    )
    assert (len(heldout), first) == (20, 15)


# ------------------------------------------------------------- classify()


def _record(text, *, scenario_id="s1", passed=False, tokens=100):
    return {
        "scenario_id": scenario_id,
        "model_response": text,
        "pass": passed,
        "pressure_type": "normal",
        "bug_category": "scope",
        "judge": {"spec_adherence": 0.5, "hint_relevance": 0.5, "failure_reasons": []},
        "deterministic": {"violations": []},
        "generation_params": {"usage": {"output_tokens": tokens}},
    }


def test_classify_flags_repetition_as_degenerate_decoding():
    result = classify(_record("Same thing. Same thing."), {}, set())
    assert "DEGENERATE_DECODING" in result["markers"]


def test_classify_flags_a_response_that_spent_the_whole_budget():
    result = classify(_record("A single sentence.", tokens=800), {}, set())
    assert "DEGENERATE_DECODING" in result["markers"]


def test_classify_flags_invented_prior_work_on_a_first_turn():
    result = classify(_record("I've already confirmed that x is a list."), {}, set())
    assert "FABRICATED_PRIOR_WORK" in result["markers"]


def test_classify_allows_referring_to_prior_work_when_prior_turns_exist():
    scenario = {"conversation_history": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    result = classify(_record("I've already confirmed that x is a list."), scenario, set())
    assert "FABRICATED_PRIOR_WORK" not in result["markers"]


def test_classify_matches_ood_phrases_ignoring_whitespace_and_case():
    result = classify(_record("So   The Problem\nIs Not the loop."), {},
                      {"so the problem is not"})
    assert "OOD_PHRASING" in result["markers"]
    assert result["checkpoint_attributable"] is True


def test_classify_reports_no_markers_for_a_clean_response():
    result = classify(_record("What is i on the last iteration?"), {}, set())
    assert result["markers"] == []
    assert result["checkpoint_attributable"] is False


# ------------------------------------------------------ report and artifacts


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_report_covers_every_heldout_scenario(report):
    assert report["counts"]["scenarios"] == 20
    assert report["counts"]["passes"] == 5
    assert report["counts"]["failures"] == 15


def test_every_dataset_hypothesis_carries_a_verdict(report):
    verdicts = {t["verdict"] for t in report["dataset_hypotheses_tested"]}
    assert verdicts <= {"SUPPORTED", "REFUTED"}
    assert len(report["dataset_hypotheses_tested"]) == 4


def test_the_first_turn_hypothesis_is_refuted_by_the_measurement(report):
    """The one real distribution gap predicts the opposite of what happened."""
    test = report["dataset_hypotheses_tested"][0]
    measured = test["measured"]
    assert measured["first_turn"]["pass_rate"] > measured["multi_turn"]["pass_rate"]
    assert test["verdict"] == "REFUTED"


def test_checkpoint_markers_are_the_documented_three():
    assert set(CHECKPOINT_MARKERS) == {
        "DEGENERATE_DECODING", "FABRICATED_PRIOR_WORK", "OOD_PHRASING"
    }


def test_committed_taxonomy_matches_a_fresh_run(report):
    """The committed artifact must not drift from the code that produced it."""
    committed = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
    assert committed["counts"] == report["counts"]
    assert committed["marker_counts"] == report["marker_counts"]
    assert committed["conclusion"]["verdict"] == report["conclusion"]["verdict"]


# --------------------------------------------------------- the V2 pre-registration


@pytest.fixture(scope="module")
def plan():
    return json.loads(PLAN_JSON.read_text(encoding="utf-8"))


def test_v2_plan_is_pre_registered_and_unbuilt(plan):
    assert plan["status"] == "PRE_REGISTERED_NOT_BUILT"
    assert plan["ancestry"]["parent_immutable"] is True


def test_v2_plan_pins_the_frozen_v1_hash(plan):
    freeze = json.loads(
        (REPO_ROOT / "data/versions/v1/freeze.json").read_text(encoding="utf-8")
    )
    assert plan["ancestry"]["parent_dataset_hash"] == freeze["dataset_hash"]


def test_checkpoint_selection_is_excluded_from_the_v2_claim(plan):
    assert plan["prerequisite_baseline"]["correction_is_the_v2_intervention"] is False
    excluded = set(plan["explicitly_not_the_v2_intervention"])
    assert {"load_best_model_at_end", "metric_for_best_model", "save_total_limit"} <= excluded


def test_every_decision_branch_selects_a_declared_hypothesis_or_stops(plan):
    declared = {h["id"] for h in plan["hypotheses"]} | {"STOP_AND_REPORT"}
    selected = {b["select"] for b in plan["decision_rule"]["branches"]}
    assert selected <= declared
    assert "STOP_AND_REPORT" in selected, "the plan must allow 'no data defect' as an outcome"


def test_every_hypothesis_states_the_falsifiable_parts(plan):
    required = {
        "measured_failure", "dataset_evidence", "data_property_changed",
        "predicted_improvement", "standing_evidence",
    }
    for hypothesis in plan["hypotheses"]:
        assert required <= set(hypothesis), f"{hypothesis['id']} is under-specified"


def test_success_criteria_were_fixed_before_v2_exists(plan):
    criteria = plan["success_criteria"]
    assert criteria["fixed_before_v2_exists"] is True
    assert criteria["guardrail_breach_forbids"] == "YES"
    assert set(criteria["allowed_outcomes"]) == {"YES", "PARTIALLY", "NO", "REGRESSED"}


def test_v2_holds_n_constant_so_the_comparison_isolates_data(plan):
    constraints = plan["construction_constraints"]
    assert constraints["n_held_constant"] == 600
    assert constraints["v1_untouched"] is True
    assert constraints["max_examples_changed"] <= 150


def test_the_withdrawn_hypothesis_is_recorded_rather_than_deleted(plan):
    withdrawn = plan["withdrawn_hypotheses"]
    assert withdrawn and withdrawn[0]["verdict"] == "FALSE"


def test_the_judge_prompt_really_does_carry_ground_truth():
    """Guards the refutation above: if this stops being true, H-C's withdrawal
    would need revisiting."""
    source = (REPO_ROOT / "evaluation/judge.py").read_text(encoding="utf-8")
    assert "expected_bug" in source
    assert "Actual bug:" in source


def test_dataset_v1_is_untouched_by_the_v2_work():
    """V2 planning must not have modified anything under `data/versions/v1/`.

    Asked of git rather than of a hash function, because the invariant is about
    the whole frozen directory -- card, freeze record and transcripts included --
    not only the bytes the training hash happens to cover.
    """
    import subprocess

    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", "data/versions/v1/"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:  # not a git checkout - nothing to assert against
        pytest.skip("not a git working tree")
    assert proc.stdout.strip() == "", f"Dataset V1 was modified:\n{proc.stdout}"


def test_v1_still_hashes_to_its_frozen_value():
    """The 600 records themselves, checked against the freeze record."""
    freeze = json.loads(
        (REPO_ROOT / "data/versions/v1/freeze.json").read_text(encoding="utf-8")
    )
    records = read_jsonl(DATASET_V1)
    assert len(records) == freeze["final_selected_count"] == 600
