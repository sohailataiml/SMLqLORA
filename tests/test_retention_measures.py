"""The two frozen retention measures, and the Arm A single-variable guarantee.

These measures are pre-registered: they exist before Arm A produces any output,
so an arm cannot be judged on an instrument tuned after seeing its results. The
worked examples in `SPEC_EXAMPLES` *are* the specification, and the first test
here asserts the implementation matches every one of them.

The distinction the measures exist to draw is acknowledgement versus
encouragement. "You've already identified the issue" asserts correctness;
"the right direction" reports progress. Getting that line wrong in either
direction would silently change every retention number, so both sides are tested.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from analysis.retention_measures import (
    ENCOURAGEMENT_ONLY,
    MEASURE_VERSION,
    SPEC_EXAMPLES,
    acknowledgement,
    clean_release,
    followup_diagnostic_question,
    score,
    strict_confirmation,
    tally,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG = REPO_ROOT / "training/configs/qlora_qwen3_1_7b_t4_bestckpt.yaml"
ARM_A_CONFIG = REPO_ROOT / "training/configs/qlora_qwen3_1_7b_t4_assistant_loss.yaml"


# ------------------------------------------------------- the specification

@pytest.mark.parametrize("example", SPEC_EXAMPLES, ids=lambda e: e["source"][:40])
def test_every_worked_example_scores_as_specified(example):
    got = score(example["text"])
    for measure in ("strict_confirmation", "solved_state_acknowledgement",
                    "followup_diagnostic_question", "clean_release"):
        assert got[measure] == example[measure], (
            f"{measure}: expected {example[measure]}, got {got[measure]} "
            f"for {example['text']!r}"
        )


def test_the_measure_is_versioned():
    assert MEASURE_VERSION == "1.0.0"


# ----------------------------------------------- acknowledgement vs encouragement

@pytest.mark.parametrize("text", [
    "Great! You've already identified the issue.",
    "That's the right change.",
    "You've got the cause.",
    "Yes, you've narrowed it down correctly.",
    "You correctly diagnosed the closure problem.",
    "Your reasoning is correct.",
    "That's exactly it.",
])
def test_acknowledgement_fires_on_asserted_correctness(text):
    assert acknowledgement(text), text


@pytest.mark.parametrize("text", [
    "Good observation — the two versions differ.",
    "Good — that's the right direction.",
    "You're on the right track.",
    "Nice try, but that isn't it.",
    "Good instinct checking the input.",
    "Good, you've already narrowed it down.",
    "That's a good start.",
])
def test_acknowledgement_does_not_fire_on_encouragement(text):
    assert not acknowledgement(text), text


def test_every_documented_exclusion_is_actually_excluded():
    """The exclusion list is a claim; this makes it a checked one."""
    for phrase in ENCOURAGEMENT_ONLY:
        assert not acknowledgement(f"Good — {phrase}."), phrase


def test_strict_confirmation_is_a_subset_of_acknowledgement():
    for example in SPEC_EXAMPLES:
        if example["strict_confirmation"]:
            assert acknowledgement(example["text"])


# --------------------------------------------------------- clean release

def test_acknowledgement_plus_question_is_not_clean_release():
    text = "That's the right change. Now, what does apply() do?"
    assert acknowledgement(text)
    assert followup_diagnostic_question(text)
    assert not clean_release(text)


def test_acknowledgement_without_question_is_clean_release():
    text = "That's the right change. The closure now shares one timer."
    assert clean_release(text)


def test_a_question_alone_is_not_clean_release():
    assert not clean_release("What does the loop print?")


def test_clean_release_is_exactly_its_derivation():
    for example in SPEC_EXAMPLES:
        expected = (
            example["solved_state_acknowledgement"]
            and not example["followup_diagnostic_question"]
        )
        assert clean_release(example["text"]) is expected


# ------------------------------------------------------------- mechanics

def test_measures_are_deterministic():
    text = "Great! You've already identified the issue."
    assert score(text) == score(text)


@pytest.mark.parametrize("value", ["", None])
def test_empty_input_is_handled(value):
    assert not strict_confirmation(value)
    assert not acknowledgement(value)
    assert not clean_release(value)


def test_tally_counts_each_measure():
    counts = tally([
        "That's exactly right, and your fix is correct.",   # strict, clean
        "Great! You've already identified the issue.",       # ack, clean
        "That's the right change. Now what does apply do?",  # ack, not clean
        "What does the loop print?",                         # neither
    ])
    assert counts == {
        "n": 4, "strict_confirmation": 1,
        "solved_state_acknowledgement": 3, "clean_release": 2,
    }


def test_reasons_are_reported_so_counts_can_be_audited():
    reasons = score("Great! You've already identified the issue.")[
        "acknowledgement_reasons"
    ]
    assert "identified_the_problem" in reasons


# --------------------------------------- Arm A is a single-variable change

def _semantic(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_arm_a_config_exists():
    assert ARM_A_CONFIG.exists()


def test_arm_a_differs_from_the_baseline_only_by_assistant_only_loss():
    """The whole validity of the ablation rests on this."""
    base, arm = _semantic(BASELINE_CONFIG), _semantic(ARM_A_CONFIG)

    allowed_renames = {"run_name"}
    top_diff = {
        key for key in set(base) | set(arm)
        if base.get(key) != arm.get(key)
    }
    assert top_diff <= allowed_renames | {"training"}, top_diff

    training_diff = {
        key for key in set(base["training"]) | set(arm["training"])
        if base["training"].get(key) != arm["training"].get(key)
    }
    assert training_diff == {"assistant_only_loss"}, training_diff
    assert arm["training"]["assistant_only_loss"] is True
    assert "assistant_only_loss" not in base["training"]


def test_arm_a_uses_a_distinct_run_name():
    """It must not overwrite the corrected baseline adapter."""
    base, arm = _semantic(BASELINE_CONFIG), _semantic(ARM_A_CONFIG)
    assert arm["run_name"] == "socratic-v1-n600-assistant-loss"
    assert arm["run_name"] != base["run_name"]


@pytest.mark.parametrize("section", ["model", "quantization", "lora", "data"])
def test_arm_a_leaves_every_other_section_identical(section):
    base, arm = _semantic(BASELINE_CONFIG), _semantic(ARM_A_CONFIG)
    assert base.get(section) == arm.get(section)


def test_the_flag_reaches_the_trainer_arguments():
    from training.train import requested_trainer_arguments

    baseline = requested_trainer_arguments(
        {"load_best_model_at_end": True}, {}, Path("x"), has_eval=True
    )
    arm = requested_trainer_arguments(
        {"load_best_model_at_end": True, "assistant_only_loss": True},
        {}, Path("x"), has_eval=True,
    )
    assert "assistant_only_loss" not in baseline
    assert arm["assistant_only_loss"] is True
    assert {
        k for k in set(baseline) | set(arm) if baseline.get(k) != arm.get(k)
    } == {"assistant_only_loss"}


def test_the_flag_is_experimentally_significant():
    """A TRL that cannot express it must fail loudly, not train something else."""
    from training.train import EXPERIMENTALLY_SIGNIFICANT

    assert "assistant_only_loss" in EXPERIMENTALLY_SIGNIFICANT


# ------------------------------------------- the pre-registration artifacts

SPEC_ARTIFACT = REPO_ROOT / "results/solved_state_analysis/retention_measure_spec.json"
RESCORED = REPO_ROOT / "results/solved_state_analysis/retention_rescored.json"
MASKING = REPO_ROOT / "results/solved_state_analysis/assistant_loss_verification.json"


def _json(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(not SPEC_ARTIFACT.exists(), reason="spec artifact absent")
def test_the_spec_records_it_was_created_before_arm_a():
    spec = _json(SPEC_ARTIFACT)
    assert spec["created_before_arm_a"] is True
    assert spec["no_llm_judge"] is True
    assert spec["version"] == MEASURE_VERSION
    assert len(spec["worked_examples"]) == len(SPEC_EXAMPLES)


@pytest.mark.skipif(not RESCORED.exists(), reason="rescore artifact absent")
def test_the_rescoring_reproduces_the_published_strict_counts():
    """Historical numbers must survive re-scoring, or continuity is broken."""
    by_model = _json(RESCORED)["sources"]["checkpoint_matrix"]["by_model"]
    assert by_model["base"]["strict_confirmation"] == 0
    assert by_model["checkpoint-34"]["strict_confirmation"] == 1
    assert by_model["checkpoint-68"]["strict_confirmation"] == 0
    assert by_model["checkpoint-102"]["strict_confirmation"] == 0


@pytest.mark.skipif(not RESCORED.exists(), reason="rescore artifact absent")
def test_the_broader_measure_recovers_the_base_models_acknowledgement():
    by_model = _json(RESCORED)["sources"]["checkpoint_matrix"]["by_model"]
    assert by_model["base"]["solved_state_acknowledgement"] == 3
    assert by_model["base"]["clean_release"] == 2


@pytest.mark.skipif(not RESCORED.exists(), reason="rescore artifact absent")
def test_no_fine_tuned_checkpoint_achieves_clean_release():
    """The baseline Arm A has to beat: zero, everywhere."""
    by_model = _json(RESCORED)["sources"]["checkpoint_matrix"]["by_model"]
    for label in ("checkpoint-34", "exported", "checkpoint-68", "checkpoint-102"):
        assert by_model[label]["clean_release"] == 0, label


@pytest.mark.skipif(not MASKING.exists(), reason="masking artifact absent")
def test_masking_was_verified_before_training():
    report = _json(MASKING)
    assert report["masking_verified"] is True
    assert report["trl_swapped_in_training_template"] is True
    for sample in report["samples"]:
        assert sample["system_prefix_fully_masked"] is True
        assert sample["labels_outside_assistant_are_ignore_index"] is True
        assert sample["labels_inside_assistant_are_real"] is True
        assert sample["learner_text_excluded_from_supervision"] is True


@pytest.mark.skipif(not MASKING.exists(), reason="masking artifact absent")
def test_arm_a_raises_the_release_share_of_the_loss():
    """7.44% under full-sequence loss; recomputed, not assumed, under Arm A."""
    distribution = _json(MASKING)["arm_a_distribution"]
    assert distribution["release_share_of_loss_under_arm_a"] == 0.2824
    assert distribution["release_share_of_loss_under_arm_a"] > 0.0744
