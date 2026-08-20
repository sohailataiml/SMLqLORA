"""Which checkpoint actually shipped, proven from the artifact.

`load_best_model_at_end: true` is a request, not a receipt. The MVP N=600 run
exported epoch 3 while its own curve said epoch 1 was best, and nothing in the
output directory recorded that.

Byte-matching answers this cleanly when it succeeds. When it fails it answers
nothing at all: `save_pretrained` re-serialises with its own header and can
rename keys, so the same weights land in a differently-hashed file. These tests
pin the distinction -- a re-serialised BEST checkpoint must verify, and a
re-serialised FINAL checkpoint must still be caught.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from scripts.verify_checkpoint_selection import (
    compare_tensor_sets,
    normalize_adapter_key,
    verify,
)

BEST_STEP, MID_STEP, FINAL_STEP = 34, 68, 102


def _tensors(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.randn(16, 32, generator=generator),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.randn(32, 16, generator=generator),
    }


def _write(directory, tensors, *, adapter_name: str | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    if adapter_name:
        tensors = {
            k.replace(".lora_A.", f".lora_A.{adapter_name}.")
             .replace(".lora_B.", f".lora_B.{adapter_name}."): v
            for k, v in tensors.items()
        }
    save_file(tensors, str(directory / "adapter_model.safetensors"))


@pytest.fixture
def run_dir(tmp_path):
    """A finished run: three checkpoints, epoch 1 best, plus an export slot."""
    out = tmp_path / "socratic-v1-n600-bestckpt"
    out.mkdir()
    for step, seed in ((BEST_STEP, 1), (MID_STEP, 2), (FINAL_STEP, 3)):
        _write(out / f"checkpoint-{step}", _tensors(seed))
    (out / "checkpoint_metadata.json").write_text(json.dumps({
        "checkpoint_selection": {
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 3,
            "best_model_checkpoint": f"/somewhere/checkpoint-{BEST_STEP}",
            "best_metric": 1.9665836095809937,
            "final_global_step": FINAL_STEP,
            "final_epoch": 3.0,
            "validation_history": [
                {"epoch": 1.0, "eval_loss": 1.9665836095809937},
                {"epoch": 2.0, "eval_loss": 2.819046974182129},
                {"epoch": 3.0, "eval_loss": 2.8398325443267822},
            ],
        }
    }), encoding="utf-8")
    return out


# ------------------------------------------------------------- the happy path

def test_a_byte_identical_best_export_verifies(run_dir):
    _write(run_dir, _tensors(1))
    code, report = verify(run_dir)
    assert code == 0
    assert report["verdict"] == "VERIFIED_BEST"
    assert report["match_method"] == "bytes"


# ------------------------------------------- the case that blocked the Colab run

def test_a_reserialised_best_export_still_verifies(run_dir):
    """Same weights, different key spelling and header. Must not read as failure."""
    _write(run_dir, _tensors(1), adapter_name="default")
    code, report = verify(run_dir)
    assert code == 0
    assert report["verdict"] == "VERIFIED_BEST"
    assert report["match_method"] == "tensor"
    assert report["tensor_matches"] == [f"checkpoint-{BEST_STEP}"]


def test_the_tensor_path_reports_a_zero_difference(run_dir):
    _write(run_dir, _tensors(1), adapter_name="default")
    _, report = verify(run_dir)
    best = [c for c in report["tensor_comparisons"]
            if c["checkpoint"] == f"checkpoint-{BEST_STEP}"][0]
    assert best["max_abs_diff"] == 0.0
    assert best["identical"] is True


# ----------------------------------------------- the failure it must still catch

def test_a_reserialised_final_export_is_caught(run_dir):
    """The MVP defect wearing a re-serialisation disguise."""
    _write(run_dir, _tensors(3), adapter_name="default")
    code, report = verify(run_dir)
    assert code == 1
    assert report["verdict"] == "EXPORTED_FINAL_NOT_BEST"
    assert report["tensor_matches"] == [f"checkpoint-{FINAL_STEP}"]


def test_a_byte_identical_final_export_is_caught(run_dir):
    _write(run_dir, _tensors(3))
    code, report = verify(run_dir)
    assert code == 1
    assert report["verdict"] == "EXPORTED_FINAL_NOT_BEST"


def test_an_export_matching_nothing_stays_unconfirmed(run_dir):
    """Weights from nowhere must never be blessed."""
    _write(run_dir, _tensors(99))
    code, report = verify(run_dir)
    assert code == 2
    assert report["verdict"] == "EXPORT_MATCHES_NO_CHECKPOINT"


def test_a_middle_checkpoint_export_is_not_blessed(run_dir):
    """Matching *a* checkpoint is not the same as matching the best one."""
    _write(run_dir, _tensors(2), adapter_name="default")
    code, report = verify(run_dir)
    assert code == 2
    assert report["verdict"] == "EXPORT_MATCHES_NO_CHECKPOINT"


def test_a_near_miss_is_not_treated_as_identical(run_dir):
    """Floating-point 'close enough' is not what is being claimed here."""
    tensors = _tensors(1)
    key = next(iter(tensors))
    tensors[key] = tensors[key] + 1e-7
    _write(run_dir, tensors, adapter_name="default")
    code, report = verify(run_dir)
    assert code == 2
    assert report["verdict"] == "EXPORT_MATCHES_NO_CHECKPOINT"


def test_the_validation_curve_is_reported(run_dir):
    _write(run_dir, _tensors(1))
    _, report = verify(run_dir)
    assert report["best_differs_from_final"] is True
    assert report["best_by_eval_loss"]["epoch"] == 1.0


# ------------------------------------------------------------------- helpers

def test_key_normalisation_strips_the_adapter_name():
    assert (normalize_adapter_key("x.lora_A.default.weight")
            == "x.lora_A.weight")


def test_key_normalisation_leaves_plain_keys_alone():
    assert normalize_adapter_key("x.lora_A.weight") == "x.lora_A.weight"


def test_differing_tensor_sets_are_not_identical():
    a = {"one": torch.zeros(2)}
    b = {"one": torch.zeros(2), "two": torch.zeros(2)}
    assert compare_tensor_sets(a, b)["identical"] is False


def test_shape_mismatch_is_reported_not_crashed():
    a = {"one": torch.zeros(2)}
    b = {"one": torch.zeros(3)}
    result = compare_tensor_sets(a, b)
    assert result["identical"] is False
    assert "shape mismatch" in result["reason"]
