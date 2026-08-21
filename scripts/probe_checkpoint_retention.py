#!/usr/bin/env python
"""When did solved-state confirmation get lost -- and was it ever there?

The retention gate found 1 of 3 training examples still confirming under the
exported adapter (checkpoint-34, epoch 1). That leaves the shape of the loss
unknown: the capability may have been weak from the first epoch, or eroded
across epochs, or moved non-monotonically.

Three saved checkpoints and the untuned base model answer that on the same three
training examples, with the same prompts, generation parameters and detector as
the gate. The base row is the reference point: it says what the capability looked
like before any fine-tuning touched it.

    base            ?/3
    checkpoint-34   ?/3   <- the exported adapter
    checkpoint-68   ?/3
    checkpoint-102  ?/3

THIS IS A DIAGNOSTIC PROBE, NOT AN EXPERIMENT. No judge calls, no metric
comparable with N600_V1_BASELINE, nothing claimable as a result. Requires a GPU
for the adapter rows.

    python scripts/probe_checkpoint_retention.py --run-dir outputs/socratic-v1-n600-bestckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.solved_state import tutor_profile  # noqa: E402
from models.adapters import EVAL_PARAMS, ModelError, resolve_model  # noqa: E402
from scripts.probe_training_example_release import (  # noqa: E402
    DEFAULT_RUN_DIR,
    load_split,
    replay,
    select_examples,
)

BASE_MODEL = "hf:Qwen/Qwen3-1.7B"
CHECKPOINTS = ("checkpoint-34", "checkpoint-68", "checkpoint-102")
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/solved_state_analysis/checkpoint_retention_probe.json"
)


def model_specs(run_dir: Path) -> list[tuple[str, str]]:
    """(label, model spec) for the base model and each surviving checkpoint.

    Checkpoints are addressed as PEFT adapter directories, which is what they
    are -- each holds its own adapter_config.json and adapter_model.safetensors.
    """
    try:
        relative = run_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        relative = run_dir.as_posix()

    specs = [("base", BASE_MODEL), ("exported", f"peft:Qwen/Qwen3-1.7B+{relative}")]
    for name in CHECKPOINTS:
        if (run_dir / name).exists():
            specs.append((name, f"peft:Qwen/Qwen3-1.7B+{relative}/{name}"))
    return specs


def probe(run_dir: Path, extra: int = 1) -> dict[str, Any]:
    train = load_split(run_dir / "data/train.jsonl")
    selected = select_examples(train, extra)
    rows: list[dict[str, Any]] = []

    for label, spec in model_specs(run_dir):
        try:
            model = resolve_model(spec)
        except ModelError as exc:
            rows.append({"model_label": label, "model": spec, "error": str(exc)})
            print(f"\n[SKIP] {label}: {exc}", file=sys.stderr)
            continue

        for example_id in selected:
            system, visible, learner, target = replay(train[example_id])
            response = model.generate(visible, system=system, params=EVAL_PARAMS)
            behaviour = tutor_profile(response.text or "")
            rows.append({
                "model_label": label,
                "model": spec,
                "example_id": example_id,
                "bug_category": train[example_id]["meta"]["bug_category"],
                "target_confirms": tutor_profile(target)["confirms"],
                "generated_response": (response.text or "").strip(),
                "generated_confirms": behaviour["confirms"],
                "generated_asks_question": behaviour["asks_question"],
                "error": response.error,
            })
            mark = "CONFIRMS" if behaviour["confirms"] else "no-confirm"
            print(f"[{mark:10}] {label:14} {example_id}")

    return {
        "artifact_status": "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT",
        "note": (
            "No judge calls. Inputs replayed verbatim from the trainer's own "
            "train.jsonl. Not comparable with N600_V1_BASELINE."
        ),
        "question": (
            "Was solved-state confirmation ever present, and when was it lost?"
        ),
        "run_dir": str(run_dir),
        "selected_examples": selected,
        "generation_params": {
            "max_tokens": EVAL_PARAMS.max_tokens,
            "temperature": EVAL_PARAMS.temperature,
            "seed": EVAL_PARAMS.seed,
        },
        "results": rows,
    }


def retention_matrix(report: dict[str, Any]) -> dict[str, Any]:
    """Confirmations per model, over examples whose target confirms."""
    matrix: dict[str, dict[str, int]] = {}
    for row in report.get("results", []):
        if "example_id" not in row or not row.get("target_confirms"):
            continue
        entry = matrix.setdefault(row["model_label"], {"confirms": 0, "of": 0})
        entry["of"] += 1
        entry["confirms"] += int(bool(row["generated_confirms"]))
    return matrix


def describe_shape(matrix: dict[str, dict[str, int]]) -> str:
    """Was the capability eroded, weak from the start, or never present?"""
    def value(label: str) -> int | None:
        return matrix[label]["confirms"] if label in matrix else None

    base, first = value("base"), value("checkpoint-34")
    last = value("checkpoint-102")
    if base is None or first is None:
        return "INCOMPLETE - base and checkpoint-34 are both required"
    if base == 0:
        return (
            "BASE_LACKS_IT - the untuned model does not confirm on these inputs "
            "either, so fine-tuning cannot have removed it here"
        )
    if first < base and last is not None and last <= first:
        return (
            "ERODED_EARLY_THEN_FLAT - most of the loss is already present at "
            "epoch 1 and later epochs do not recover it"
        )
    if first < base:
        return "ERODED_BY_EPOCH_1 - the loss happens within the first epoch"
    return "RETAINED_AT_EPOCH_1 - the loss, if any, happens after epoch 1"


def render(report: dict[str, Any]) -> str:
    matrix = retention_matrix(report)
    order = ["base", "checkpoint-34", "exported", "checkpoint-68", "checkpoint-102"]
    lines = ["", "solved-state confirmation on training examples:", ""]
    for label in order:
        if label in matrix:
            entry = matrix[label]
            note = "  <- exported adapter" if label == "checkpoint-34" else ""
            lines.append(f"  {label:16} {entry['confirms']}/{entry['of']}{note}")
    lines += ["", f"SHAPE: {describe_shape(matrix)}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--extra", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    report = probe(run_dir, args.extra)
    report["retention_matrix"] = retention_matrix(report)
    report["shape"] = describe_shape(report["retention_matrix"])
    print(render(report))

    if args.write:
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
