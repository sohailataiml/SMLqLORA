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
) -> dict[str, Any]:
    model_cfg = config.section("model")
    return {
        "run_name": run_name,
        "training_format_version": "1.0.0",
        "base_model": model_cfg.get("base_model"),
        "base_model_revision": model_cfg.get("revision", "main"),
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
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


# =============================================================================
# Training
# =============================================================================


def prepare_data(
    config: TrainingConfig, *, limit: int | None, accepted_override: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
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
    eval_scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])

    split = build_dataset(
        examples,
        eval_scenarios=eval_scenarios,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
        limit=limit if limit is not None else data_cfg.get("limit"),
    )
    version = examples[0].provenance.dataset_version if examples else "unknown"
    return split.train, split.validation, str(accepted_path), version


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

    train_rows, validation_rows, dataset_path, dataset_version = prepare_data(
        config, limit=limit, accepted_override=accepted_override
    )

    print(f"run              : {run}")
    print(f"base model       : {config.section('model').get('base_model')}")
    print(f"dataset          : {dataset_path} (version {dataset_version})")
    print(f"train / val      : {len(train_rows)} / {len(validation_rows)}")
    print(f"fingerprint      : {dataset_fingerprint(train_rows)[:16]}")
    print(f"output           : {output_dir}")

    environment = check_environment(require_gpu=not dry_run)
    metadata = build_checkpoint_metadata(
        config,
        run_name=run,
        dataset_path=dataset_path,
        dataset_version=dataset_version,
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        environment=environment,
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

    _run_training(config, train_rows, validation_rows, output_dir)

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
) -> None:
    """The actual training loop. Imports the heavy stack lazily."""
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

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision", "main"),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        attn_implementation=str(model_cfg.get("attn_implementation", "eager")),
    )
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

    train_dataset = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
    eval_dataset = (
        Dataset.from_list([{"messages": r["messages"]} for r in validation_rows])
        if validation_rows
        else None
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(train_cfg.get("save_total_limit", 1)),
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=str(train_cfg.get("report_to", "none")),
        max_length=int(model_cfg.get("max_seq_length", 2048)),
        eval_strategy=str(train_cfg.get("eval_strategy", "epoch")) if eval_dataset else "no",
    )

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

    output_cfg = config.section("output")
    if output_cfg.get("push_to_hub") and output_cfg.get("hub_repo_id"):
        trainer.model.push_to_hub(str(output_cfg["hub_repo_id"]))
        tokenizer.push_to_hub(str(output_cfg["hub_repo_id"]))
        print(f"Pushed adapter to https://huggingface.co/{output_cfg['hub_repo_id']}")


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
