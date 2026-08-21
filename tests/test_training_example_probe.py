"""The training-example gate, and the validity conditions it depends on.

This probe decides whether a contrastive Dataset V2 is even worth designing, so
its inputs have to be beyond argument: real examples, provably in the training
split, replayed exactly as the trainer showed them. A probe that quietly
paraphrased the learner turn or strengthened the system prompt would answer a
different question and look identical from outside.

So most of what is pinned here is provenance -- the target text matches frozen
Dataset V1 byte for byte, the system prompt matches the frozen strategy, and the
splits do not overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.solved_state import load_v1_solved, tutor_profile
from prompting.strategies import get_strategy
from scripts.probe_training_example_release import (
    ANCHOR_IDS,
    DEFAULT_OUTPUT,
    DEFAULT_RUN_DIR,
    classify,
    load_split,
    main,
    probe,
    replay,
    select_examples,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = DEFAULT_RUN_DIR / "data/train.jsonl"

needs_split = pytest.mark.skipif(
    not TRAIN.exists(), reason="training split not present locally"
)


@pytest.fixture
def train():
    return load_split(TRAIN)


# ------------------------------------------------------------- the split

@needs_split
def test_the_training_split_is_the_trainers_own_file(train):
    assert len(train) == 540


@needs_split
def test_train_and_validation_do_not_overlap(train):
    validation = load_split(DEFAULT_RUN_DIR / "data/validation.jsonl")
    assert len(validation) == 60
    assert not (set(train) & set(validation))


@needs_split
def test_both_anchors_are_in_the_training_split(train):
    """The probe's whole point is that the model saw these."""
    for example_id in ANCHOR_IDS:
        assert example_id in train


# ------------------------------------------------------------- selection

@needs_split
def test_selection_is_deterministic(train):
    assert select_examples(train) == select_examples(train)


@needs_split
def test_selection_covers_distinct_bug_categories(train):
    chosen = select_examples(train)
    categories = [train[i]["meta"]["bug_category"] for i in chosen]
    assert len(categories) == len(set(categories))


@needs_split
def test_every_selected_example_is_a_solved_case(train):
    for example_id in select_examples(train):
        assert train[example_id]["meta"]["student_has_solved"] is True


@needs_split
def test_selection_starts_with_the_anchors(train):
    assert select_examples(train)[: len(ANCHOR_IDS)] == list(ANCHOR_IDS)


# --------------------------------------------------- replay fidelity

@needs_split
def test_replay_withholds_only_the_target_turn(train):
    row = train[ANCHOR_IDS[0]]
    _, visible, _, target = replay(row)
    assert len(visible) == len(row["messages"]) - 2  # system and target removed
    assert target == row["messages"][-1]["content"]


@needs_split
def test_replay_uses_the_frozen_zero_shot_system_prompt(train):
    """No prompt strengthening: this is the prompt training itself used."""
    system, _, _, _ = replay(train[ANCHOR_IDS[0]])
    assert system == get_strategy("zero_shot").system_prompt()


@needs_split
def test_target_text_matches_frozen_dataset_v1_byte_for_byte(train):
    """Proves nothing was rewritten between the dataset and the probe."""
    frozen = {r["id"]: r["tutor_response"] for r in load_v1_solved()}
    for example_id in select_examples(train):
        _, _, _, target = replay(train[example_id])
        assert target == frozen[example_id]


@needs_split
def test_the_learner_final_message_is_the_last_user_turn(train):
    row = train[ANCHOR_IDS[0]]
    _, _, learner, _ = replay(row)
    expected = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    assert learner == expected


@needs_split
def test_every_selected_target_actually_confirms(train):
    """If a target did not confirm, the probe could not ask its question."""
    for example_id in select_examples(train):
        _, _, _, target = replay(train[example_id])
        assert tutor_profile(target)["confirms"] is True


def test_a_row_without_an_assistant_target_is_rejected():
    row = {"meta": {"id": "x"}, "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    with pytest.raises(ValueError, match="assistant target"):
        replay(row)


# ------------------------------------------------------ the predeclared cases

def _report(confirmations: list[bool]):
    return {"results": [
        {"example_id": f"e{i}", "target_confirms": True,
         "generated_confirms": c}
        for i, c in enumerate(confirmations)
    ]}


def test_all_confirmations_retained_is_case_1():
    verdict = classify(_report([True, True, True]))
    assert verdict["case"] == "CASE_1"
    assert "contrastive Dataset V2" in verdict["conclusion"]


def test_no_confirmations_retained_is_case_2():
    verdict = classify(_report([False, False, False]))
    assert verdict["case"] == "CASE_2"
    assert "Do NOT build V2 yet" in verdict["conclusion"]


def test_partial_retention_is_case_3():
    verdict = classify(_report([True, False, True]))
    assert verdict["case"] == "CASE_3"
    assert verdict["retained"] == ["e0", "e2"]
    assert verdict["lost"] == ["e1"]


def test_no_confirming_targets_is_invalid_not_a_case():
    verdict = classify({"results": [
        {"example_id": "e0", "target_confirms": False, "generated_confirms": False}
    ]})
    assert verdict["case"] == "INVALID"


# ------------------------------------------------------- it is not a result

@needs_split
def test_it_runs_end_to_end_on_a_mock_model():
    report = probe("mock:demo", DEFAULT_RUN_DIR)
    assert report["artifact_status"] == "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT"
    assert len(report["results"]) == 3
    assert all(r["error"] is None for r in report["results"])
    assert report["splits_disjoint"] is True


@needs_split
def test_every_row_records_its_provenance():
    report = probe("mock:demo", DEFAULT_RUN_DIR)
    for row in report["results"]:
        assert row["in_training_split"] is True
        assert row["in_validation_split"] is False
        assert row["system_prompt_is_the_frozen_zero_shot"] is True


def test_it_makes_no_judge_calls():
    source = (REPO_ROOT / "scripts/probe_training_example_release.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("DeterministicJudge", "LLMJudge", "evaluation.judge",
                      "JUDGE_PARAMS", "Evaluator("):
        assert forbidden not in source, f"probe must not reach for {forbidden}"


def test_it_uses_the_same_detector_as_the_release_probe():
    """One definition of 'confirms', or the two probes are not comparable."""
    from scripts import probe_release_behavior
    from scripts import probe_training_example_release
    assert (probe_training_example_release.tutor_profile
            is probe_release_behavior.tutor_profile)


def test_it_never_writes_into_the_baseline_directory():
    assert "n600_v1_baseline" not in str(DEFAULT_OUTPUT)
    assert DEFAULT_OUTPUT.name == "training_example_release_probe.json"


@needs_split
def test_writing_the_artifact_stays_where_it_is_told(tmp_path):
    destination = tmp_path / "probe.json"
    main(["--model", "mock:demo", "--write", "--output", str(destination)])
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT"
    assert payload["verdict"]["case"] in {"CASE_1", "CASE_2", "CASE_3"}


@needs_split
def test_it_does_not_write_unless_asked():
    before = DEFAULT_OUTPUT.exists()
    main(["--model", "mock:demo"])
    assert DEFAULT_OUTPUT.exists() == before
