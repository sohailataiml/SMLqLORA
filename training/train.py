"""QLoRA fine-tuning with Transformers + PEFT + TRL.

Everything that could change a result is recorded in `checkpoint_metadata.json`
next to the adapter: base model and revision, dataset version and hash, the exact
training text fingerprint, seed, LoRA config, training arguments, package
versions and the git commit.

The heavy imports are deferred so `--dry-run` works on a machine with no GPU and
no torch — which is how the configuration is validated in CI and how the sweep is
planned before any compute is rented.

Usage:
    python -m training.train --config training/configs/qlora_qwen3_1_7b.yaml --dry-run
    python -m training.train --config training/configs/qlora_qwen3_1_7b.yaml
    python -m training.train --config ... --limit 250 --run-name socratic-n250
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from evaluation.reproducibility import git_commit, git_is_dirty, package_versions  # noqa: E402
from evaluation.schemas import load_scenario_files  # noqa: E402
from training.dataset import (  # noqa: E402
    TRAINING_SYSTEM_PROMPT,
    DatasetSplit,
    build_dataset,
    dataset_fingerprint,
    load_accepted,
    stable_order,
    to_chat_record,
    write_dataset,
)

EVAL_SETS = ("scenarios/clean.jsonl", "scenarios/adversarial.jsonl",
             "scenarios/heldout.jsonl")

#: Minimum free VRAM (GiB) for QLoRA on a ~1.7B model at 2048 tokens.
MIN_VRAM_GIB = 12.0
#: bitsandbytes NF4 kernels require this compute capability.
MIN_COMPUTE_CAPABILITY = (7, 5)


class TrainingUnavailableError(RuntimeError):
    """Raised when the environment cannot run training, with what to do instead."""


class DatasetHashMismatchError(RuntimeError):
    """Raised when the source dataset no longer hashes to its frozen value."""


def source_dataset_hash(examples: Sequence[Any]) -> str:
    """Hash the source file the same way `scripts/finalize_dataset.py` froze it.

    Recomputing it here (rather than trusting the recorded value) is what makes
    a training run traceable: the adapter can be tied to the exact bytes that
    produced it, not to a filename that happened to sit at that path.
    """
    chat = [to_chat_record(e) for e in stable_order(list(examples))]
    return dataset_fingerprint(chat)


def verify_source_dataset(examples: Sequence[Any], expected: str | None) -> str:
    """Refuse to train on a dataset that has drifted from its frozen hash."""
    actual = source_dataset_hash(examples)
    if expected and actual != expected:
        raise DatasetHashMismatchError(
            f"""Dataset on disk does not match the frozen hash - refusing to train.
  expected: {expected}
  actual  : {actual}
Dataset V1 is immutable. If the data genuinely needs to change, that change
is Dataset V2, not an edit to v1."""
        )
    return actual


class AdapterNotWrittenError(RuntimeError):
    """Raised when training finished but left no loadable adapter behind."""


def verify_adapter_written(output_dir: Path) -> Path:
    """Fail loudly if the run produced no adapter.

    An output directory is created before training starts - it holds the built
    dataset and the run metadata - so its existence proves nothing. Without this
    check a crashed run leaves a plausible-looking directory that the evaluator
    then tries to load as a PEFT adapter, failing on every single scenario and
    reporting the result as a model that answered nothing.
    """
    config_file = output_dir / "adapter_config.json"
    if not config_file.exists():
        salvage = sorted(output_dir.glob("checkpoint-*/adapter_config.json"))
        hint = (
            f"\nPer-epoch checkpoints do exist: {[str(p.parent) for p in salvage]}\n"
            f"Evaluate one of those rather than retraining."
            if salvage
            else "\nNo per-epoch checkpoints either - the run never saved."
        )
        raise AdapterNotWrittenError(
            f"Training finished but wrote no adapter to {output_dir}.{hint}"
        )
    weights = [
        p for p in output_dir.iterdir()
        if p.suffix in (".safetensors", ".bin") and "adapter" in p.name
    ]
    if not weights:
        raise AdapterNotWrittenError(
            f"{output_dir} has an adapter_config.json but no adapter weights. "
            f"The checkpoint is incomplete and cannot be evaluated."
        )
    return config_file


class TrainerArgumentError(RuntimeError):
    """Raised when a hyperparameter cannot be expressed in the installed TRL."""


class PrecisionMismatchError(RuntimeError):
    """Raised when trainable parameters cannot be used with the chosen AMP mode."""


def model_load_dtype(
    train_cfg: dict[str, Any], quant_cfg: dict[str, Any] | None = None
) -> str:
    """The dtype the base model's non-quantized weights must load in.

    Omitting `torch_dtype` does not mean "pick something sensible" - it means
    "follow the checkpoint", and Qwen3's config says bfloat16. On a T4 that
    combination is fatal: the LoRA parameters inherit bfloat16, and fp16 mixed
    precision uses a GradScaler whose CUDA unscale kernel has no bfloat16
    implementation. Training dies at the first gradient-clipping step with

        NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
        not implemented for 'BFloat16'

    The load dtype must therefore follow the mixed-precision mode, not the
    checkpoint.
    """
    if train_cfg.get("bf16"):
        return "bfloat16"
    if train_cfg.get("fp16"):
        return "float16"
    # No mixed precision. "auto" would follow the checkpoint back to bfloat16,
    # so match the quantization compute dtype instead - that is the dtype the
    # dequantized matmuls actually run in.
    compute = str((quant_cfg or {}).get("bnb_4bit_compute_dtype", "")).strip()
    return compute or "auto"


def mixed_precision_mode(train_cfg: dict[str, Any]) -> str:
    """The accelerate spelling of the configured precision: bf16 / fp16 / no."""
    if train_cfg.get("bf16"):
        return "bf16"
    if train_cfg.get("fp16"):
        return "fp16"
    return "no"


def apply_mixed_precision_environment(train_cfg: dict[str, Any]) -> str:
    """Tell accelerate the precision explicitly instead of letting it decide.

    transformers builds its Accelerator from ambient state - an `accelerate
    config` file, or ACCELERATE_* environment variables - and that state can
    disagree with the training arguments. When it does, the model is re-cast
    inside `Accelerator.prepare()`, AFTER any check this module performs, which
    is how a run whose parameters were verifiably `uint8 + float16 + float32`
    still died in a bfloat16 GradScaler kernel.

    Setting the variable makes the two agree and makes the choice visible.
    """
    import os

    mode = mixed_precision_mode(train_cfg)
    os.environ["ACCELERATE_MIXED_PRECISION"] = mode
    return mode


def dtype_kwarg_name(loader: Any) -> str:
    """Whichever of `dtype` / `torch_dtype` the installed transformers accepts.

    transformers renamed this. Passing the retired spelling is a deprecation
    warning on one version and a silently ignored argument on another - and
    "silently ignored" means the checkpoint's own dtype wins, which is exactly
    the failure this whole function exists to prevent.
    """
    import inspect

    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return "dtype"
    if "dtype" in params:
        return "dtype"
    if "torch_dtype" in params:
        return "torch_dtype"
    # Both absent means the loader takes **kwargs; prefer the current name.
    return "dtype"


def describe_parameter_dtypes(model: Any, *, trainable_only: bool = False) -> str:
    """A dtype histogram, so the log records what loaded rather than what was asked."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for _, param in model.named_parameters():
        if trainable_only and not getattr(param, "requires_grad", False):
            continue
        counts[str(getattr(param, "dtype", "?")).replace("torch.", "")] += 1
    return ", ".join(f"{name}x{n}" for name, n in counts.most_common()) or "none"


def cast_trainable_parameters_to_fp32(model: Any, train_cfg: dict[str, Any]) -> int:
    """Put every trainable parameter in fp32 when fp16 mixed precision is on.

    This is the standard QLoRA arrangement - frozen 4-bit base, fp32 adapters -
    and it is what makes the GradScaler's job well defined. Relying on the
    loader to have produced the right dtype was not enough: the argument that
    controls it has been renamed, and a version that ignores it leaves the
    adapter in the checkpoint's bfloat16, which the fp16 unscale kernel cannot
    handle. Returns how many tensors were recast.
    """
    targets = trainable_tensors_needing_fp32(model.named_parameters(), train_cfg)
    if not targets:
        return 0
    try:
        import torch
    except ImportError:  # pragma: no cover
        return 0
    wanted = set(targets)
    recast = 0
    for name, param in model.named_parameters():
        if name in wanted:
            param.data = param.data.to(torch.float32)
            recast += 1
    return recast


def trainable_tensors_needing_fp32(
    named_parameters: Any, train_cfg: dict[str, Any]
) -> list[str]:
    """Which trainable tensors must be recast, decided without importing torch.

    Split out from the cast itself so the rule is testable on a machine with no
    training stack, which is where this repository's suite runs.
    """
    if not train_cfg.get("fp16") or train_cfg.get("bf16"):
        return []
    return sorted(
        name for name, param in named_parameters
        if getattr(param, "requires_grad", False)
        and str(getattr(param, "dtype", "")) != "torch.float32"
    )


def bf16_trainable_parameters(named_parameters: Any) -> list[str]:
    """Names of trainable parameters held in bfloat16.

    Compares the dtype by name rather than by identity so this stays testable on
    a machine without torch - which is where the rest of the suite runs.
    """
    return sorted(
        name for name, param in named_parameters
        if getattr(param, "requires_grad", False)
        and str(getattr(param, "dtype", "")) == "torch.bfloat16"
    )


def assert_precision_is_trainable(model: Any, train_cfg: dict[str, Any]) -> None:
    """Catch a dtype/AMP mismatch before the first optimizer step, not during it."""
    if not train_cfg.get("fp16") or train_cfg.get("bf16"):
        return
    offenders = bf16_trainable_parameters(model.named_parameters())
    if offenders:
        raise PrecisionMismatchError(
            f"fp16 mixed precision is enabled, but {len(offenders)} trainable "
            f"parameters are bfloat16 - the fp16 GradScaler cannot unscale those "
            f"on CUDA.\n"
            f"  first offenders: {offenders[:3]}\n"
            f"  the base model must load in float16 when fp16: true."
        )


#: Arguments TRL/transformers have renamed across versions. First name that the
#: installed SFTConfig accepts wins.
TRAINER_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "eval_strategy": ("eval_strategy", "evaluation_strategy"),
    "max_length": ("max_length", "max_seq_length"),
    "warmup_ratio": ("warmup_ratio",),
    "optim": ("optim", "optimizer"),
}

#: Dropping one of these silently would change the experiment, so a version that
#: cannot express them is an error rather than a warning.
EXPERIMENTALLY_SIGNIFICANT = frozenset({
    "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "lr_scheduler_type", "weight_decay", "max_grad_norm",
    "seed", "max_length", "warmup_ratio",
    # Which checkpoint the run exports is as significant as how it was trained.
    # The N=600 MVP run shipped its epoch-3 adapter while its own validation
    # curve showed epoch 1 was better, because these were not configurable and
    # `save_total_limit: 1` pruned the best one. Silently dropping them would
    # reproduce exactly that defect.
    "load_best_model_at_end", "metric_for_best_model", "greater_is_better",
})


def total_train_steps(train_cfg: dict[str, Any], train_examples: int) -> int:
    """Optimizer steps for the whole run, for translating ratios into counts."""
    batch = max(1, int(train_cfg.get("per_device_train_batch_size", 2)))
    accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 8)))
    epochs = float(train_cfg.get("num_train_epochs", 3))
    per_epoch = max(1, -(-train_examples // (batch * accum)))  # ceil
    return max(1, int(per_epoch * epochs))


def _checkpoint_selection_arguments(
    train_cfg: dict[str, Any], *, has_eval: bool
) -> dict[str, Any]:
    """Best-checkpoint export settings, present only when actually requested.

    Emitted conditionally for two reasons. A config that does not ask for this
    produces exactly the arguments it produced before, so no existing recipe
    shifts underneath it. And because these keys are experimentally significant,
    including them unconditionally would turn "this TRL cannot express
    `metric_for_best_model`" into a hard error even for runs that never wanted
    it -- while still erroring, correctly, for runs that did.

    `load_best_model_at_end` also requires evaluation to be running. Asking for
    it without a validation split fails inside the Trainer, after the model is
    already on the GPU, so it is refused here instead.
    """
    if not bool(train_cfg.get("load_best_model_at_end", False)):
        return {}
    if not has_eval:
        raise TrainerArgumentError(
            "load_best_model_at_end needs a validation split to choose a best "
            "checkpoint from, but this run has none. Set "
            "`data.validation_fraction` above 0, or drop the setting."
        )
    return {
        "load_best_model_at_end": True,
        "metric_for_best_model": str(
            train_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": bool(train_cfg.get("greater_is_better", False)),
    }


def requested_trainer_arguments(
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    output_dir: Path,
    *,
    has_eval: bool,
) -> dict[str, Any]:
    """The training arguments this experiment wants, before version translation."""
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(
            train_cfg.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(
            train_cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 0.3)),
        "optim": str(train_cfg.get("optim", "paged_adamw_8bit")),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_strategy": str(train_cfg.get("save_strategy", "epoch")),
        "save_total_limit": int(train_cfg.get("save_total_limit", 1)),
        **_checkpoint_selection_arguments(train_cfg, has_eval=has_eval),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "seed": int(train_cfg.get("seed", 42)),
        "report_to": str(train_cfg.get("report_to", "none")),
        "max_length": int(model_cfg.get("max_seq_length", 2048)),
        "eval_strategy": (
            str(train_cfg.get("eval_strategy", "epoch")) if has_eval else "no"
        ),
    }


def adapt_trainer_arguments(
    config_class: Any, requested: dict[str, Any], total_steps: int
) -> tuple[dict[str, Any], list[str]]:
    """Fit the requested arguments to whatever the installed SFTConfig accepts.

    TRL and transformers rename training arguments between releases. Passing one
    the installed version does not know raises a bare TypeError partway into a
    run - which is how `warmup_ratio` cost a T4 session after the model, the
    quantization and the LoRA adapter had all loaded successfully.

    Pinning a version would only move the problem. Instead the signature is read
    and the arguments translated, with two rules:

    * a rename is followed (`eval_strategy` -> `evaluation_strategy`);
    * anything that would change the experiment and has no home raises, rather
      than being dropped into a silently different recipe.

    Cosmetic arguments (logging, reporting) may be dropped with a note.
    """
    import inspect

    try:
        accepted = set(inspect.signature(config_class.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic config classes
        return dict(requested), ["could not introspect SFTConfig; passing as-is"]
    accepted.discard("self")
    accepted.discard("kwargs")

    resolved: dict[str, Any] = {}
    notes: list[str] = []

    for key, value in requested.items():
        candidates = TRAINER_ARG_ALIASES.get(key, (key,))
        target = next((name for name in candidates if name in accepted), None)

        if target is not None:
            if target != key:
                notes.append(f"{key} -> {target} (renamed in this version)")
            resolved[target] = value
            continue

        # warmup_ratio has an exact equivalent when only step counts survive.
        if key == "warmup_ratio" and "warmup_steps" in accepted:
            steps = int(round(float(value) * total_steps))
            resolved["warmup_steps"] = steps
            notes.append(
                f"warmup_ratio={value} -> warmup_steps={steps} "
                f"({total_steps} total steps); same schedule, different spelling"
            )
            continue

        if key in EXPERIMENTALLY_SIGNIFICANT:
            raise TrainerArgumentError(
                f"The installed TRL's SFTConfig does not accept {key!r}, and it "
                f"cannot be translated. Dropping it would silently change the "
                f"experiment, so this is an error.\n"
                f"  tried: {candidates}\n"
                f"  install a TRL that supports it, or express it differently.\n"
                f"  accepted arguments include: "
                f"{sorted(a for a in accepted if 'warm' in a or 'lr' in a or 'epoch' in a)}"
            )

        notes.append(f"{key} dropped - not accepted by this version (cosmetic)")

    return resolved, notes


def check_trainer_api(train_cfg: dict[str, Any], model_cfg: dict[str, Any],
                      train_examples: int) -> list[str]:
    """Validate the trainer arguments before any GPU time is spent.

    Returns notes, or raises. A no-op when TRL is not installed, so `--dry-run`
    still works on a laptop.
    """
    try:
        from trl import SFTConfig
    except ImportError:
        return ["trl not installed - trainer arguments not checked"]
    apply_mixed_precision_environment(train_cfg)

    requested = requested_trainer_arguments(
        train_cfg, model_cfg, Path("."), has_eval=True
    )
    _, notes = adapt_trainer_arguments(
        SFTConfig, requested, total_train_steps(train_cfg, train_examples)
    )
    return notes


@dataclass
class TrainingConfig:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        p = Path(path)
        if not p.exists():
            raise SystemExit(
                f"Training config not found: {p}\n"
                f"Start from training/configs/qlora_qwen3_1_7b.yaml."
            )
        return cls(raw=yaml.safe_load(p.read_text(encoding="utf-8")), path=p)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})

    @property
    def run_name(self) -> str:
        return str(self.raw.get("run_name", "socratic"))


# =============================================================================
# Environment preflight
# =============================================================================


def check_environment(*, require_gpu: bool = True) -> dict[str, Any]:
    """Report whether this machine can actually run QLoRA, and why not."""
    info: dict[str, Any] = {"torch": None, "cuda": False, "devices": []}

    try:
        import torch
    except ImportError as exc:
        message = (
            "PyTorch is not installed.\n"
            "  Install the training extras:  pip install -e '.[train]'\n"
            "  On a CUDA machine, install the matching torch build first:\n"
            "    https://pytorch.org/get-started/locally/"
        )
        if require_gpu:
            raise TrainingUnavailableError(message) from exc
        # A dry run validates configuration and data; it must not require the
        # training stack to be installed at all.
        info["note"] = "torch not installed (dry run does not need it)"
        return info

    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())

    if not info["cuda"]:
        if require_gpu:
            raise TrainingUnavailableError(
                "No CUDA device is visible, and QLoRA needs one.\n"
                "  bitsandbytes 4-bit kernels require an NVIDIA GPU with compute\n"
                "  capability >= 7.5 (Turing or newer) and ~12GB of VRAM.\n"
                "\n"
                "  Options:\n"
                "    - Free Colab/Kaggle T4:  open notebooks/train_colab.ipynb\n"
                "    - Rented GPU:            pip install -e '.[train]' then re-run\n"
                "    - Validate config only:  add --dry-run"
            )
        return info

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        vram = props.total_memory / (1024 ** 3)
        device = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_gib": round(vram, 1),
            "supports_nf4": capability >= MIN_COMPUTE_CAPABILITY,
            "sufficient_vram": vram >= MIN_VRAM_GIB,
        }
        info["devices"].append(device)

    usable = [d for d in info["devices"] if d["supports_nf4"]]
    if require_gpu and not usable:
        listed = "; ".join(
            f"{d['name']} (cc {d['compute_capability']}, {d['vram_gib']}GiB)"
            for d in info["devices"]
        )
        raise TrainingUnavailableError(
            f"No GPU here supports 4-bit NF4 quantization.\n"
            f"  Found: {listed}\n"
            f"  bitsandbytes needs compute capability >= "
            f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}.\n"
            f"  Use a T4 or newer — notebooks/train_colab.ipynb is set up for that."
        )
    return info


# =============================================================================
# Metadata
# =============================================================================


def build_checkpoint_metadata(
    config: TrainingConfig,
    *,
    run_name: str,
    dataset_path: str,
    dataset_version: str,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    environment: dict[str, Any],
    source_dataset_hash: str | None = None,
) -> dict[str, Any]:
    model_cfg = config.section("model")
    return {
        "run_name": run_name,
        "training_format_version": "1.0.0",
        "base_model": model_cfg.get("base_model"),
        "base_model_revision": model_cfg.get("revision", "main"),
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
        "source_dataset_hash": source_dataset_hash,
        "dataset_train_size": len(train_rows),
        "dataset_validation_size": len(validation_rows),
        "dataset_fingerprint": dataset_fingerprint(list(train_rows)),
        "training_system_prompt": TRAINING_SYSTEM_PROMPT,
        "seed": config.section("training").get("seed", 42),
        "quantization": config.section("quantization"),
        "lora": config.section("lora"),
        "training_arguments": config.section("training"),
        "config_file": str(config.path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "package_versions": package_versions(),
        "environment": environment,
    }


def checkpoint_selection_record(trainer_state: Any, resolved: dict[str, Any]) -> dict:
    """Which checkpoint this run exported, and the curve that justifies it.

    Reconstructing this after the fact was impossible for the N=600 MVP run: the
    adapter shipped without a trainer state, `save_total_limit: 1` had already
    deleted the earlier checkpoints, and the only surviving record of the
    validation curve was a console log pasted into a report. Recording it beside
    the adapter makes the question "which epoch is this?" answerable from the
    artifact instead of from memory.
    """
    load_best = bool(resolved.get("load_best_model_at_end", False))
    history = list(getattr(trainer_state, "log_history", []) or [])
    return {
        "exported_checkpoint": "best_by_metric" if load_best else "final_step",
        "load_best_model_at_end": load_best,
        "metric_for_best_model": resolved.get("metric_for_best_model"),
        "greater_is_better": resolved.get("greater_is_better"),
        "save_strategy": resolved.get("save_strategy"),
        "save_total_limit": resolved.get("save_total_limit"),
        "best_model_checkpoint": getattr(trainer_state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer_state, "best_metric", None),
        "final_global_step": getattr(trainer_state, "global_step", None),
        "final_epoch": getattr(trainer_state, "epoch", None),
        "validation_history": [e for e in history if "eval_loss" in e],
        "train_history": [e for e in history if "loss" in e and "eval_loss" not in e],
    }


# =============================================================================
# Training
# =============================================================================


def prepare_data(
    config: TrainingConfig, *, limit: int | None, accepted_override: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str]:
    data_cfg = config.section("data")
    accepted_path = REPO_ROOT / (
        accepted_override or data_cfg.get("accepted_path", "data/accepted/v1.jsonl")
    )
    if not accepted_path.exists():
        raise SystemExit(
            f"No accepted examples at {accepted_path}.\n"
            f"Run the pipeline first:\n"
            f"  python -m generation.generate --dataset-version v1\n"
            f"  python scripts/filter_data.py --dataset-version v1"
        )

    examples = load_accepted(accepted_path)
    source_hash = verify_source_dataset(
        examples, data_cfg.get("expected_dataset_hash")
    )
    eval_scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])

    split = build_dataset(
        examples,
        eval_scenarios=eval_scenarios,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
        limit=limit if limit is not None else data_cfg.get("limit"),
    )
    version = (
        data_cfg.get("dataset_version")
        or (examples[0].provenance.dataset_version if examples else "unknown")
    )
    return split.train, split.validation, str(accepted_path), version, source_hash


def train(
    config: TrainingConfig,
    *,
    limit: int | None = None,
    run_name: str | None = None,
    dry_run: bool = False,
    accepted_override: str | None = None,
) -> Path:
    run = run_name or config.run_name
    output_dir = REPO_ROOT / config.section("output").get("output_dir", "outputs") / run

    (
        train_rows,
        validation_rows,
        dataset_path,
        dataset_version,
        source_hash,
    ) = prepare_data(config, limit=limit, accepted_override=accepted_override)

    print(f"run              : {run}")
    print(f"base model       : {config.section('model').get('base_model')}")
    print(f"dataset          : {dataset_path} (version {dataset_version})")
    print(f"source hash      : {source_hash[:16]} (verified against freeze)")
    print(f"train / val      : {len(train_rows)} / {len(validation_rows)}")
    print(f"fingerprint      : {dataset_fingerprint(train_rows)[:16]}")
    print(f"output           : {output_dir}")

    environment = check_environment(require_gpu=not dry_run)

    # Validate the trainer arguments against the INSTALLED TRL before loading a
    # model. A rejected argument used to surface as a TypeError after the weights,
    # the quantization and the LoRA adapter had all loaded - minutes of GPU time,
    # and an output directory that looked like a checkpoint but held nothing.
    for note in check_trainer_api(
        config.section("training"), config.section("model"), len(train_rows)
    ):
        print(f"  trainer-arg: {note}")

    metadata = build_checkpoint_metadata(
        config,
        run_name=run,
        dataset_path=dataset_path,
        dataset_version=dataset_version,
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        environment=environment,
        source_dataset_hash=source_hash,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset(
        DatasetSplit(train=list(train_rows), validation=list(validation_rows)),
        output_dir / "data",
    )
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if dry_run:
        print()
        print("DRY RUN — configuration validated, data built, nothing trained.")
        print(f"Metadata written to {output_dir / 'checkpoint_metadata.json'}")
        return output_dir

    selection = _run_training(config, train_rows, validation_rows, output_dir)
    verify_adapter_written(output_dir)
    metadata["checkpoint_selection"] = selection

    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps({**metadata, "completed": True}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nAdapter written to {output_dir}")
    print(
        f"Evaluate it with:\n"
        f"  python eval.py --model "
        f"'peft:{config.section('model').get('base_model')}+{output_dir}' "
        f"--eval-set scenarios/heldout.jsonl"
    )
    return output_dir


def _run_training(
    config: TrainingConfig,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """The actual training loop. Imports the heavy stack lazily.

    Returns the checkpoint-selection record, which the caller folds into
    `checkpoint_metadata.json` beside the adapter.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_cfg = config.section("model")
    quant_cfg = config.section("quantization")
    lora_cfg = config.section("lora")
    train_cfg = config.section("training")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(
        torch, str(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    load_dtype = model_load_dtype(train_cfg, quant_cfg)
    print(f"  model dtype      : {load_dtype} (mixed precision: "
          f"{'bf16' if train_cfg.get('bf16') else 'fp16'})")

    load_kwargs: dict[str, Any] = dict(
        revision=model_cfg.get("revision", "main"),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        attn_implementation=str(model_cfg.get("attn_implementation", "eager")),
    )
    # transformers renamed torch_dtype -> dtype. Passing the wrong one is either
    # a deprecation warning or a silently ignored argument depending on version,
    # and "silently ignored" means the checkpoint's bfloat16 wins.
    load_kwargs[dtype_kwarg_name(AutoModelForCausalLM.from_pretrained)] = (
        load_dtype if load_dtype == "auto" else getattr(torch, load_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], **load_kwargs
    )
    print(f"  loaded dtypes    : {describe_parameter_dtypes(model)}")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True))
    )

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora_cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # The standard QLoRA arrangement: 4-bit frozen base, fp32 trainable adapters.
    # fp16 mixed precision drives a GradScaler, and its CUDA unscale kernel is
    # implemented for fp16 and fp32 but NOT bfloat16 - so a bfloat16 trainable
    # parameter kills the run at the first gradient clip, after everything else
    # has already succeeded. Casting removes the whole class of failure rather
    # than depending on what the loader decided.
    recast = cast_trainable_parameters_to_fp32(model, train_cfg)
    if recast:
        print(f"  recast to fp32   : {recast} trainable tensors")
    print(f"  trainable dtypes : {describe_parameter_dtypes(model, trainable_only=True)}")
    assert_precision_is_trainable(model, train_cfg)

    train_dataset = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
    eval_dataset = (
        Dataset.from_list([{"messages": r["messages"]} for r in validation_rows])
        if validation_rows
        else None
    )

    requested = requested_trainer_arguments(
        train_cfg, model_cfg, output_dir, has_eval=eval_dataset is not None
    )
    resolved, notes = adapt_trainer_arguments(
        SFTConfig, requested, total_train_steps(train_cfg, len(train_rows))
    )
    for note in notes:
        print(f"  trainer-arg: {note}")
    sft_config = SFTConfig(**resolved)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    selection = checkpoint_selection_record(trainer.state, resolved)
    print(f"  exported checkpoint: {selection['exported_checkpoint']}"
          f" (best={selection['best_model_checkpoint']},"
          f" metric={selection['best_metric']})")

    output_cfg = config.section("output")
    if output_cfg.get("push_to_hub") and output_cfg.get("hub_repo_id"):
        trainer.model.push_to_hub(str(output_cfg["hub_repo_id"]))
        tokenizer.push_to_hub(str(output_cfg["hub_repo_id"]))
        print(f"Pushed adapter to https://huggingface.co/{output_cfg['hub_repo_id']}")

    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning.")
    parser.add_argument("--config", default="training/configs/qlora_qwen3_1_7b.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="train on the first N examples (data-efficiency sweep)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--accepted", default=None,
                        help="override the accepted-examples path from the config")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config and build data without touching a GPU")
    args = parser.parse_args(argv)

    config = TrainingConfig.load(REPO_ROOT / args.config
                                 if not Path(args.config).is_absolute()
                                 else args.config)
    try:
        train(config, limit=args.limit, run_name=args.run_name,
              dry_run=args.dry_run, accepted_override=args.accepted)
    except TrainingUnavailableError as exc:
        print(f"\nCannot train here:\n{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
