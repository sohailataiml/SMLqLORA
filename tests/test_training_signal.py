"""Token-weighted training signal, and the checkpoint retention matrix.

The example balance is 75 solved against 465 not. That is not what gradient
descent sees: TRL's defaults leave the loss on the whole sequence, so most of
the signal is the learner's text and an invariant system prompt. These tests pin
the measured proportions, because the ablation design is chosen from them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.training_signal import (
    DEFAULT_RUN_DIR,
    LOSS_MASKING,
    classify_target,
    read_jsonl,
)
from scripts.probe_checkpoint_retention import (
    describe_shape,
    model_specs,
    retention_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = DEFAULT_RUN_DIR / "data/train.jsonl"

needs_split = pytest.mark.skipif(
    not TRAIN.exists(), reason="training split not present locally"
)


# ------------------------------------------------------- what the loss covers

def test_the_recipe_requests_neither_masking_option():
    """Both are left unset, so TRL's defaults decide -- and they mask nothing."""
    assert LOSS_MASKING["assistant_only_loss"]["requested"] is None
    assert LOSS_MASKING["completion_only_loss"]["requested"] is None


def test_trl_defaults_mean_full_sequence_loss():
    assert LOSS_MASKING["assistant_only_loss"]["trl_default"] is False
    assert LOSS_MASKING["completion_only_loss"]["trl_default"] is None
    assert LOSS_MASKING["effective"] == "full_sequence"


def test_the_masking_claim_cites_its_source():
    evidence = LOSS_MASKING["evidence"]
    assert "sft_config.py" in evidence
    assert "sft_trainer.py" in evidence


# -------------------------------------------------- the behavioural imbalance

@needs_split
def test_the_training_split_composition():
    rows = read_jsonl(TRAIN)
    solved = [r for r in rows if r["meta"]["student_has_solved"]]
    assert len(rows) == 540
    assert len(solved) == 75
    assert len(rows) - len(solved) == 465


@needs_split
def test_solved_targets_overwhelmingly_confirm_without_re_questioning():
    rows = read_jsonl(TRAIN)
    solved = [r for r in rows if r["meta"]["student_has_solved"]]
    behaviours = [classify_target(r["messages"][-1]["content"]) for r in solved]
    assert behaviours.count("confirm_no_question") == 69
    assert behaviours.count("confirm_with_question") == 3


@needs_split
def test_unsolved_targets_overwhelmingly_ask_a_question():
    rows = read_jsonl(TRAIN)
    unsolved = [r for r in rows if not r["meta"]["student_has_solved"]]
    behaviours = [classify_target(r["messages"][-1]["content"]) for r in unsolved]
    assert behaviours.count("question_only") == 361


@needs_split
def test_the_two_regimes_are_cleanly_separated():
    """If the data were ambiguous, the model would have an excuse. It is not."""
    rows = read_jsonl(TRAIN)
    solved = [r for r in rows if r["meta"]["student_has_solved"]]
    unsolved = [r for r in rows if not r["meta"]["student_has_solved"]]
    solved_confirm = sum(
        1 for r in solved
        if classify_target(r["messages"][-1]["content"]).startswith("confirm")
    )
    unsolved_confirm = sum(
        1 for r in unsolved
        if classify_target(r["messages"][-1]["content"]).startswith("confirm")
    )
    assert solved_confirm == 72          # 96% of solved
    assert unsolved_confirm == 27        # 5.8% of unsolved


def test_target_classification_is_deterministic():
    text = "That's exactly right, and your fix is correct."
    assert classify_target(text) == classify_target(text) == "confirm_no_question"


def test_a_confirmation_followed_by_a_question_is_its_own_category():
    assert classify_target(
        "That's exactly right. Now what does apply do?"
    ) == "confirm_with_question"


def test_a_bare_question_is_question_only():
    assert classify_target("What does the loop print?") == "question_only"


# --------------------------------------------------- the checkpoint matrix

@needs_split
def test_the_matrix_covers_base_export_and_every_saved_checkpoint():
    labels = [label for label, _ in model_specs(DEFAULT_RUN_DIR)]
    assert labels[0] == "base"
    assert "exported" in labels


def test_checkpoints_are_addressed_as_peft_adapter_directories():
    specs = dict(model_specs(DEFAULT_RUN_DIR))
    assert specs["base"] == "hf:Qwen/Qwen3-1.7B"
    assert specs["exported"].startswith("peft:Qwen/Qwen3-1.7B+")


def _matrix(**counts):
    return {k: {"confirms": v, "of": 3} for k, v in counts.items()}


def test_shape_reports_erosion_within_the_first_epoch():
    shape = describe_shape(_matrix(base=3, **{"checkpoint-34": 1, "checkpoint-102": 0}))
    assert "ERODED_EARLY_THEN_FLAT" in shape


def test_shape_reports_retention_at_epoch_one():
    shape = describe_shape(_matrix(base=1, **{"checkpoint-34": 3, "checkpoint-102": 3}))
    assert "RETAINED_AT_EPOCH_1" in shape


def test_shape_refuses_to_blame_training_when_the_base_lacks_it():
    """If the base model never confirms here, fine-tuning did not remove it."""
    shape = describe_shape(_matrix(base=0, **{"checkpoint-34": 0}))
    assert "BASE_LACKS_IT" in shape


def test_shape_reports_incompleteness_rather_than_guessing():
    assert "INCOMPLETE" in describe_shape(_matrix(**{"checkpoint-68": 2}))


def test_retention_matrix_ignores_rows_whose_target_does_not_confirm():
    report = {"results": [
        {"model_label": "base", "example_id": "a", "target_confirms": True,
         "generated_confirms": True},
        {"model_label": "base", "example_id": "b", "target_confirms": False,
         "generated_confirms": True},
    ]}
    assert retention_matrix(report) == {"base": {"confirms": 1, "of": 1}}
