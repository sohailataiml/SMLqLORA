"""Training-format conversion, nested subsets, and config validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_filtering import make_example
from training.dataset import (
    TRAINING_SYSTEM_PROMPT,
    ContaminationError,
    DatasetSplit,
    assert_no_contamination,
    build_dataset,
    dataset_fingerprint,
    nested_subsets,
    stable_order,
    suggested_sweep_sizes,
    to_chat_record,
    write_dataset,
)


def examples(spec, n: int):
    return [
        make_example(spec, example_id=f"gen_v1_{i:05d}",
                     student_message=f"distinct question number {i}")
        for i in range(n)
    ]


# ------------------------------------------------------- chat conversion


def test_chat_record_shape(spec):
    record = to_chat_record(make_example(spec))
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == TRAINING_SYSTEM_PROMPT
    assert "```python" in record["messages"][1]["content"]


def test_training_uses_the_weak_prompt():
    """Behavior must live in the weights, not in an elaborate system prompt."""
    from prompting.strategies import StructuredStrategy, ZeroShotStrategy

    assert TRAINING_SYSTEM_PROMPT == ZeroShotStrategy().system_prompt()
    assert TRAINING_SYSTEM_PROMPT != StructuredStrategy().system_prompt()
    assert len(TRAINING_SYSTEM_PROMPT) < 400


def test_conversion_is_deterministic(spec):
    example = make_example(spec)
    assert to_chat_record(example) == to_chat_record(example)


def test_multi_turn_conversion_preserves_history(spec):
    from evaluation.schemas import Message, Role

    example = make_example(spec)
    scenario = example.scenario.model_copy(
        update={
            "conversation_history": (
                Message(role=Role.USER, content="first attempt"),
                Message(role=Role.ASSISTANT, content="Which index is last?"),
            )
        }
    )
    record = to_chat_record(example.model_copy(update={"scenario": scenario}))
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    # The code appears exactly once, on the first learner turn.
    assert sum(m["content"].count("```python") for m in record["messages"]) == 1


def test_metadata_is_attached(spec):
    meta = to_chat_record(make_example(spec))["meta"]
    for key in ("id", "language", "bug_category", "pressure_type", "dataset_version",
                "teacher_model"):
        assert key in meta


def test_metadata_can_be_omitted(spec):
    assert "meta" not in to_chat_record(make_example(spec), include_metadata=False)


# --------------------------------------------------------------- splits


def test_build_dataset_splits(spec):
    split = build_dataset(examples(spec, 20), validation_fraction=0.1)
    assert len(split.train) == 18
    assert len(split.validation) == 2
    assert len(split) == 20


def test_split_is_deterministic(spec):
    items = examples(spec, 20)
    a = build_dataset(items)
    b = build_dataset(items)
    assert [r["meta"]["id"] for r in a.train] == [r["meta"]["id"] for r in b.train]


def test_order_is_independent_of_input_order(spec):
    items = examples(spec, 20)
    forward = [e.id for e in stable_order(items)]
    backward = [e.id for e in stable_order(list(reversed(items)))]
    assert forward == backward


def test_limit_truncates(spec):
    split = build_dataset(examples(spec, 30), limit=10)
    assert len(split) == 10


def test_limit_beyond_available_is_rejected(spec):
    with pytest.raises(ValueError, match="only 5 are available"):
        build_dataset(examples(spec, 5), limit=50)


def test_empty_input_is_rejected(spec):
    with pytest.raises(ValueError, match="at least one accepted example"):
        build_dataset([])


# ------------------------------------------------------- nested subsets


def test_subsets_are_nested(spec):
    items = examples(spec, 40)
    subsets = nested_subsets(items, [10, 20, 40])
    ids = {size: [e.id for e in group] for size, group in subsets.items()}
    assert ids[10] == ids[20][:10]
    assert ids[20] == ids[40][:20]


def test_subset_larger_than_dataset_is_rejected(spec):
    with pytest.raises(ValueError, match="only 10 accepted examples exist"):
        nested_subsets(examples(spec, 10), [5, 50])


@pytest.mark.parametrize(
    "total,expected",
    [(1200, [125, 250, 500, 1000]), (1000, [125, 250, 500, 1000])],
)
def test_canonical_sweep_sizes(total, expected):
    assert suggested_sweep_sizes(total) == expected


def test_sweep_rescales_for_smaller_datasets():
    sizes = suggested_sweep_sizes(600)
    assert sizes[-1] == 600
    assert len(sizes) >= 3
    assert sizes == sorted(sizes)


def test_sweep_needs_a_minimum(spec):
    with pytest.raises(ValueError, match="at least 8 examples"):
        suggested_sweep_sizes(3)


# -------------------------------------------------------- contamination


def test_contamination_raises_before_training(spec, unsolved_scenario):
    contaminated = make_example(
        spec, code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message,
    )
    with pytest.raises(ContaminationError, match="refusing to build"):
        assert_no_contamination([contaminated], [unsolved_scenario])


def test_build_dataset_refuses_contaminated_input(spec, unsolved_scenario):
    contaminated = make_example(
        spec, code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message,
    )
    with pytest.raises(ContaminationError):
        build_dataset([contaminated], eval_scenarios=[unsolved_scenario])


def test_clean_input_builds(spec, unsolved_scenario):
    items = examples(spec, 12)
    # The fixture scenario shares its code with these examples, so use a
    # genuinely unrelated eval scenario.
    from evaluation.schemas import Scenario

    unrelated = Scenario(
        id="unrelated_eval",
        language="javascript",
        bug_category="async_await",
        difficulty="easy",
        code="async function f(){ const r = fetch(u); return r.json(); }",
        student_message="r.json is not a function",
        expected_bug="missing await",
        expected_fix="await fetch(u)",
        split="heldout",
    )
    assert len(build_dataset(items, eval_scenarios=[unrelated])) == 12


# ------------------------------------------------------------ artifacts


def test_fingerprint_tracks_content(spec):
    a = build_dataset(examples(spec, 12))
    b = build_dataset(examples(spec, 12), limit=8)
    assert dataset_fingerprint(a.train) != dataset_fingerprint(b.train)
    assert dataset_fingerprint(a.train) == dataset_fingerprint(a.train)


def test_write_dataset_emits_jsonl(tmp_path, spec):
    split = build_dataset(examples(spec, 12))
    paths = write_dataset(split, tmp_path)
    rows = [json.loads(l) for l in paths["train"].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(split.train)
    assert "messages" in rows[0]


# ------------------------------------------------- training config shape


def test_shipped_training_config_is_complete():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load(
        (root / "training/configs/qlora_qwen3_1_7b.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["base_model"].startswith("Qwen/")
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert config["lora"]["r"] >= 8
    assert config["lora"]["target_modules"]
    assert config["training"]["seed"] == 42
    assert "accepted_path" in config["data"]


def test_training_config_loads_and_exposes_sections():
    from pathlib import Path

    from training.train import TrainingConfig

    root = Path(__file__).resolve().parent.parent
    config = TrainingConfig.load(root / "training/configs/qlora_qwen3_1_7b.yaml")
    assert config.run_name
    assert config.section("lora")["r"]
    assert config.section("nonexistent") == {}


def test_missing_training_config_is_reported(tmp_path):
    from training.train import TrainingConfig

    with pytest.raises(SystemExit, match="Training config not found"):
        TrainingConfig.load(tmp_path / "absent.yaml")


def test_checkpoint_metadata_records_reproducibility_fields(spec):
    from pathlib import Path

    from training.train import TrainingConfig, build_checkpoint_metadata

    root = Path(__file__).resolve().parent.parent
    config = TrainingConfig.load(root / "training/configs/qlora_qwen3_1_7b.yaml")
    split = build_dataset(examples(spec, 12))

    metadata = build_checkpoint_metadata(
        config,
        run_name="test-run",
        dataset_path="data/accepted/v1.jsonl",
        dataset_version="v1",
        train_rows=split.train,
        validation_rows=split.validation,
        output_dir=Path("outputs/test-run"),
        environment={"cuda": False},
    )
    for key in ("base_model", "base_model_revision", "dataset_version",
                "dataset_fingerprint", "seed", "quantization", "lora",
                "training_arguments", "git_commit", "package_versions",
                "training_system_prompt"):
        assert key in metadata, key
    assert metadata["dataset_train_size"] == len(split.train)


# ---------------------------------------------------------------------------
# Checkpoint selection.
#
# The MVP N=600 run exported its epoch-3 adapter while its own validation curve
# showed epoch 1 was better, because `save_total_limit: 1` pruned the best one
# and nothing recorded which epoch had shipped. Both halves of that -- being able
# to ask for the best checkpoint, and recording which one was exported -- are
# tested here.
# ---------------------------------------------------------------------------


class _FakeTrainerState:
    def __init__(self, **kw):
        self.log_history = kw.get("log_history", [])
        self.best_model_checkpoint = kw.get("best_model_checkpoint")
        self.best_metric = kw.get("best_metric")
        self.global_step = kw.get("global_step", 102)
        self.epoch = kw.get("epoch", 3.0)


def _args(**overrides):
    from training.train import requested_trainer_arguments

    train_cfg = {"num_train_epochs": 3, **overrides}
    return requested_trainer_arguments(
        train_cfg, {"max_seq_length": 2048}, Path("outputs/x"), has_eval=True
    )


def test_checkpoint_selection_is_absent_unless_requested():
    """An unchanged config must produce exactly the arguments it produced before,
    so no existing recipe shifts underneath it."""
    resolved = _args()
    assert "load_best_model_at_end" not in resolved
    assert "metric_for_best_model" not in resolved
    assert "greater_is_better" not in resolved


def test_best_checkpoint_can_be_requested_from_the_config():
    resolved = _args(load_best_model_at_end=True, metric_for_best_model="eval_loss",
                     greater_is_better=False)
    assert resolved["load_best_model_at_end"] is True
    assert resolved["metric_for_best_model"] == "eval_loss"
    assert resolved["greater_is_better"] is False


def test_best_checkpoint_is_refused_when_there_is_no_validation_split():
    """`load_best_model_at_end` needs eval running; asking without it fails
    inside the Trainer, after the model is already on the GPU."""
    from training.train import TrainerArgumentError, requested_trainer_arguments

    with pytest.raises(TrainerArgumentError, match="validation split"):
        requested_trainer_arguments(
            {"load_best_model_at_end": True}, {"max_seq_length": 2048},
            Path("outputs/x"), has_eval=False,
        )


def test_checkpoint_selection_settings_cannot_be_silently_dropped():
    from training.train import EXPERIMENTALLY_SIGNIFICANT

    assert {"load_best_model_at_end", "metric_for_best_model",
            "greater_is_better"} <= EXPERIMENTALLY_SIGNIFICANT


def test_selection_record_names_the_final_checkpoint_when_not_selecting_best():
    from training.train import checkpoint_selection_record

    record = checkpoint_selection_record(
        _FakeTrainerState(), {"load_best_model_at_end": False}
    )
    assert record["exported_checkpoint"] == "final_step"


def test_selection_record_names_the_best_checkpoint_and_its_metric():
    from training.train import checkpoint_selection_record

    state = _FakeTrainerState(
        best_model_checkpoint="outputs/run/checkpoint-34", best_metric=1.97
    )
    record = checkpoint_selection_record(
        state,
        {"load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
         "greater_is_better": False},
    )
    assert record["exported_checkpoint"] == "best_by_metric"
    assert record["best_model_checkpoint"] == "outputs/run/checkpoint-34"
    assert record["best_metric"] == 1.97


def test_selection_record_preserves_the_validation_curve():
    """The MVP run's loss curve survived only as a pasted console log."""
    from training.train import checkpoint_selection_record

    state = _FakeTrainerState(log_history=[
        {"loss": 2.9, "epoch": 0.5},
        {"eval_loss": 1.97, "epoch": 1.0},
        {"eval_loss": 2.785, "epoch": 2.0},
        {"eval_loss": 2.73, "epoch": 3.0},
    ])
    record = checkpoint_selection_record(state, {"load_best_model_at_end": True})
    assert [e["eval_loss"] for e in record["validation_history"]] == [1.97, 2.785, 2.73]
    assert len(record["train_history"]) == 1


def test_the_corrected_config_differs_from_the_mvp_config_only_in_selection():
    """The V1-vs-V2 comparison is only clean if nothing else moved."""
    import yaml

    root = Path(__file__).resolve().parent.parent
    mvp = yaml.safe_load(
        (root / "training/configs/qlora_qwen3_1_7b_t4.yaml").read_text(encoding="utf-8")
    )
    fixed = yaml.safe_load(
        (root / "training/configs/qlora_qwen3_1_7b_t4_bestckpt.yaml").read_text(
            encoding="utf-8")
    )
    for section in ("model", "quantization", "lora", "data"):
        assert mvp[section] == fixed[section], f"{section} must not change"

    selection_keys = {"load_best_model_at_end", "metric_for_best_model",
                      "greater_is_better", "save_total_limit"}
    changed = {
        k for k in set(mvp["training"]) | set(fixed["training"])
        if mvp["training"].get(k) != fixed["training"].get(k)
    }
    assert changed <= selection_keys, f"unexpected training changes: {changed - selection_keys}"
    assert fixed["training"]["load_best_model_at_end"] is True
    assert fixed["training"]["metric_for_best_model"] == "eval_loss"
