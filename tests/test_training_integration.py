"""Guards on the two seams between the frozen dataset and a training result.

Both of these failed silently before: training read the wrong file and still
printed a healthy summary, and an evaluation on a machine without PyTorch wrote
a `pass_rate: 0.0` result stamped REAL_EXPERIMENT_RESULT. Neither raised, so
neither would have been noticed until the numbers were already in a report.

Nothing here touches a GPU, a credential or the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.schemas import ErrorKind, classify_error
from training.dataset import load_accepted
from training.train import (
    DatasetHashMismatchError,
    TrainingConfig,
    source_dataset_hash,
    verify_source_dataset,
)

CONFIG_PATH = REPO_ROOT / "training" / "configs" / "qlora_qwen3_1_7b.yaml"
FREEZE_PATH = REPO_ROOT / "data" / "versions" / "v1" / "freeze.json"


@pytest.fixture(scope="module")
def config() -> TrainingConfig:
    return TrainingConfig.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------- config points at the freeze


def test_config_trains_on_the_frozen_selection_not_the_accepted_pool(config, freeze):
    """The pool has 1055 examples; Dataset V1 is the 600 selected from it."""
    data = config.section("data")
    assert data["accepted_path"] == "data/versions/v1/selected.jsonl"
    path = REPO_ROOT / data["accepted_path"]
    assert path.exists()
    assert len(load_accepted(path)) == freeze["final_selected_count"] == 600


def test_config_pins_the_frozen_dataset_hash(config, freeze):
    assert config.section("data")["expected_dataset_hash"] == freeze["dataset_hash"]


def test_config_pins_a_base_model_revision(config):
    """`main` moves. A run pinned to it cannot be reproduced later."""
    model = config.section("model")
    assert model["base_model"] == "Qwen/Qwen3-1.7B"
    revision = str(model["revision"])
    assert revision != "main", "pin a commit sha before the run"
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


# ------------------------------------------------------------ the hash guard


def test_frozen_dataset_on_disk_still_matches_its_hash(config, freeze):
    examples = load_accepted(REPO_ROOT / config.section("data")["accepted_path"])
    assert source_dataset_hash(examples) == freeze["dataset_hash"]


def test_verify_source_dataset_rejects_a_drifted_dataset(config):
    examples = load_accepted(REPO_ROOT / config.section("data")["accepted_path"])
    with pytest.raises(DatasetHashMismatchError) as excinfo:
        verify_source_dataset(examples, "0" * 64)
    assert "Dataset V2" in str(excinfo.value)


def test_verify_source_dataset_allows_an_unpinned_config(config):
    """An unpinned hash is permitted, so ad-hoc subsets stay runnable."""
    examples = load_accepted(REPO_ROOT / config.section("data")["accepted_path"])
    assert verify_source_dataset(examples, None) == source_dataset_hash(examples)


def test_dropping_one_example_changes_the_hash(config):
    examples = load_accepted(REPO_ROOT / config.section("data")["accepted_path"])
    assert source_dataset_hash(examples[:-1]) != source_dataset_hash(examples)


# ------------------------------- a model that never loaded is not a bad model


@pytest.mark.parametrize(
    "message",
    [
        "MissingDependencyError: The 'transformers/torch' package is required "
        "but not installed. Install it with:  pip install -e '.[train]'",
        "MissingDependencyError: The 'bitsandbytes' package is required but "
        "not installed. Install it with:  pip install -e '.[train]'",
    ],
)
def test_missing_dependency_is_infrastructure_not_a_failed_response(message):
    assert classify_error(message) is ErrorKind.INFRASTRUCTURE


def test_missing_credentials_is_still_infrastructure():
    assert classify_error(
        "MissingCredentialsError: ANTHROPIC_API_KEY is not set"
    ) is ErrorKind.INFRASTRUCTURE


def test_a_genuine_bad_response_is_not_reclassified_as_infrastructure():
    """The guard must not swallow real content failures."""
    assert classify_error("ValueError: model returned malformed JSON") is ErrorKind.UNKNOWN


# ------------------------------------------------- training prompt stays weak


def test_training_uses_the_weak_prompt(config):
    """Under the structured prompt a behavioral gain would be unattributable."""
    from prompting.strategies import StructuredStrategy, ZeroShotStrategy
    from training.dataset import TRAINING_SYSTEM_PROMPT

    assert TRAINING_SYSTEM_PROMPT == ZeroShotStrategy().system_prompt()
    assert TRAINING_SYSTEM_PROMPT != StructuredStrategy().system_prompt()
    assert len(TRAINING_SYSTEM_PROMPT) < 400


# --------------------------------------- the spec hash must not drift on checkout


def test_live_behavior_spec_still_hashes_to_the_frozen_value(freeze):
    """The spec is hashed byte-for-byte, so line endings change the hash.

    With `core.autocrlf` deciding endings per machine, an unchanged spec can
    hash differently after a fresh clone - silently severing every result from
    the behavior it measured. `.gitattributes` pins the ending; this asserts it
    worked, because the failure is invisible otherwise.
    """
    from behavior.spec import load_spec

    assert load_spec().spec_sha256 == freeze["behavior_spec_sha256"]


def test_frozen_spec_hash_matches_the_prompt_ceiling_result(freeze):
    """The completed ablation and the frozen dataset must cite the same spec."""
    ceiling = json.loads(
        (REPO_ROOT / "results" / "prompt_ceiling" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    assert ceiling["behavior_spec_sha256"] == freeze["behavior_spec_sha256"]


def test_config_is_valid_yaml_with_the_sections_the_trainer_reads(config):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for section in ("model", "quantization", "lora", "training", "data", "output"):
        assert section in raw, f"config is missing the {section!r} section"
