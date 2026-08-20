#!/usr/bin/env python
"""Prove which checkpoint a training run actually exported.

`load_best_model_at_end: true` is a request, not a receipt. The MVP N=600 run is
the cautionary case: its config said `save_total_limit: 1`, its own validation
curve said epoch 1 was best, and the adapter that shipped was epoch 3 — and none
of that was discoverable from the artifact afterwards, because the earlier
checkpoints had already been deleted and no trainer state was saved.

So this does not read the config and believe it. It hashes the exported adapter
weights and hashes every surviving `checkpoint-*/` adapter, and reports which one
the export byte-matches. If the export matches the checkpoint with the lowest
`eval_loss`, selection worked. If it matches the last checkpoint instead, it did
not, whatever the config claims.

    python scripts/verify_checkpoint_selection.py outputs/socratic-v1-n600-bestckpt

Exit codes: 0 verified, 1 selection did not behave as configured, 2 cannot tell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_weights(directory: Path) -> Path | None:
    for name in ADAPTER_WEIGHT_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def load_adapter_tensors(path: Path) -> dict[str, Any] | None:
    """Read adapter weights as tensors, or None if that is not possible here."""
    try:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return dict(load_file(str(path)))
        import torch
        return dict(torch.load(str(path), map_location="cpu"))
    except Exception:  # pragma: no cover - missing torch/safetensors, or corrupt
        return None


def normalize_adapter_key(key: str) -> str:
    """Strip the PEFT adapter-name segment so two serialisations line up.

    `save_pretrained` on a live model writes `lora_A.default.weight`, while a
    checkpoint written mid-run may write `lora_A.weight`. Same numbers, different
    spelling.
    """
    for marker in (".default.", ".default_0."):
        key = key.replace(marker, ".")
    return key


def compare_tensor_sets(exported: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Numeric comparison of two adapter state dicts, key-name differences aside."""
    import torch

    left = {normalize_adapter_key(k): v for k, v in exported.items()}
    right = {normalize_adapter_key(k): v for k, v in other.items()}
    shared = sorted(set(left) & set(right))
    result: dict[str, Any] = {
        "tensor_count_exported": len(left),
        "tensor_count_other": len(right),
        "shared_keys": len(shared),
        "missing_from_other": sorted(set(left) - set(right))[:5],
        "missing_from_exported": sorted(set(right) - set(left))[:5],
    }
    if not shared or len(shared) != len(left) or len(shared) != len(right):
        result["identical"] = False
        result["reason"] = "the two files do not describe the same set of tensors"
        return result

    max_diff = 0.0
    for key in shared:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            result["identical"] = False
            result["reason"] = f"shape mismatch on {key}: {tuple(a.shape)} vs {tuple(b.shape)}"
            return result
        diff = (a.float() - b.float()).abs().max().item()
        max_diff = max(max_diff, diff)

    result["max_abs_diff"] = max_diff
    result["identical"] = max_diff == 0.0
    return result


def checkpoint_dirs(output_dir: Path) -> list[Path]:
    """Surviving per-epoch checkpoints, ordered by global step."""
    found = [
        p for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and adapter_weights(p) is not None
    ]
    return sorted(found, key=lambda p: int(p.name.split("-")[-1]))


def best_epoch_by_loss(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    entries = [e for e in history if e.get("eval_loss") is not None]
    return min(entries, key=lambda e: e["eval_loss"]) if entries else None


def _load_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "checkpoint_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint_metadata.json in {output_dir}. Either training did not "
            f"finish, or it ran with a build of training/train.py that predates "
            f"checkpoint-selection recording."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify(output_dir: Path) -> tuple[int, dict[str, Any]]:
    """Returns (exit_code, report)."""
    metadata = _load_metadata(output_dir)
    selection = metadata.get("checkpoint_selection") or {}
    history = selection.get("validation_history") or []

    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "requested_load_best_model_at_end": selection.get("load_best_model_at_end"),
        "metric_for_best_model": selection.get("metric_for_best_model"),
        "greater_is_better": selection.get("greater_is_better"),
        "save_total_limit": selection.get("save_total_limit"),
        "trainer_best_model_checkpoint": selection.get("best_model_checkpoint"),
        "trainer_best_metric": selection.get("best_metric"),
        "final_global_step": selection.get("final_global_step"),
        "final_epoch": selection.get("final_epoch"),
        "validation_history": history,
    }

    exported = adapter_weights(output_dir)
    if exported is None:
        report["verdict"] = "NO_EXPORTED_ADAPTER"
        report["detail"] = f"no adapter weights in {output_dir}"
        return 2, report
    export_hash = file_sha256(exported)
    report["exported_adapter"] = exported.name
    report["exported_sha256"] = export_hash

    surviving = checkpoint_dirs(output_dir)
    report["checkpoints_found"] = [p.name for p in surviving]
    matches = []
    for directory in surviving:
        weights = adapter_weights(directory)
        digest = file_sha256(weights)
        step = int(directory.name.split("-")[-1])
        entry = {"checkpoint": directory.name, "step": step, "sha256": digest,
                 "matches_export": digest == export_hash}
        if entry["matches_export"]:
            matches.append(directory.name)
        report.setdefault("checkpoint_hashes", []).append(entry)

    report["export_matches"] = matches
    best = best_epoch_by_loss(history)
    report["best_by_eval_loss"] = best

    if not history:
        report["verdict"] = "NO_VALIDATION_HISTORY"
        report["detail"] = (
            "The run recorded no eval_loss entries, so there is no curve to "
            "select against. Check that eval_strategy ran."
        )
        return 2, report

    losses = [(e.get("epoch"), e["eval_loss"]) for e in history]
    report["eval_loss_by_epoch"] = losses
    final_loss = history[-1]["eval_loss"]
    report["best_differs_from_final"] = bool(best["eval_loss"] < final_loss)

    if not surviving:
        # Nothing to byte-match against. Fall back to the trainer's own claim,
        # and say plainly that this is weaker evidence.
        claimed = selection.get("best_model_checkpoint")
        report["verdict"] = "UNVERIFIED_NO_CHECKPOINT_DIRS"
        report["detail"] = (
            f"No checkpoint-*/ directories survive, so the export cannot be "
            f"byte-matched. The trainer claims best={claimed!r} with metric "
            f"{selection.get('best_metric')!r}. Treat as unconfirmed."
        )
        return 2, report

    last = surviving[-1].name
    claimed_best = selection.get("best_model_checkpoint")
    claimed_name = Path(claimed_best).name if claimed_best else None
    report["trainer_best_checkpoint_name"] = claimed_name

    if claimed_name and claimed_name in matches:
        report["verdict"] = "VERIFIED_BEST"
        report["match_method"] = "bytes"
        report["detail"] = (
            f"The exported adapter byte-matches {claimed_name}, which the trainer "
            f"selected as best by {selection.get('metric_for_best_model')} "
            f"({selection.get('best_metric')})."
        )
        if not report["best_differs_from_final"]:
            report["detail"] += (
                " Note: the best checkpoint IS the final one on this run, so the "
                "correction changed nothing about which weights shipped."
            )
        return 0, report

    if matches == [last]:
        report["verdict"] = "EXPORTED_FINAL_NOT_BEST"
        report["detail"] = (
            f"The exported adapter byte-matches the LAST checkpoint ({last}), not "
            f"the best one ({claimed_name}). Checkpoint selection did not take "
            f"effect. Do not evaluate this adapter as a corrected baseline."
        )
        return 1, report

    # A byte mismatch is not evidence of the wrong weights. `save_pretrained`
    # re-serialises with its own header and may rename keys, so identical numbers
    # can land in a differently-hashed file. Ask the only question that matters:
    # are the VALUES the best checkpoint's values?
    exported_tensors = load_adapter_tensors(exported)
    if exported_tensors is not None:
        numeric_matches = []
        for directory in surviving:
            other = load_adapter_tensors(adapter_weights(directory))
            if other is None:
                continue
            comparison = compare_tensor_sets(exported_tensors, other)
            comparison["checkpoint"] = directory.name
            report.setdefault("tensor_comparisons", []).append(comparison)
            if comparison.get("identical"):
                numeric_matches.append(directory.name)

        report["tensor_matches"] = numeric_matches

        if claimed_name and numeric_matches == [claimed_name]:
            report["verdict"] = "VERIFIED_BEST"
            report["match_method"] = "tensor"
            report["detail"] = (
                f"The exported adapter is not byte-identical to any checkpoint -- "
                f"save_pretrained re-serialised it -- but its WEIGHTS are exactly "
                f"{claimed_name}'s, which the trainer selected as best by "
                f"{selection.get('metric_for_best_model')} "
                f"({selection.get('best_metric')}). Every tensor matches to 0.0 "
                f"absolute difference."
            )
            return 0, report

        if numeric_matches == [last]:
            report["verdict"] = "EXPORTED_FINAL_NOT_BEST"
            report["match_method"] = "tensor"
            report["detail"] = (
                f"The exported adapter's WEIGHTS are the LAST checkpoint's "
                f"({last}), not the best one ({claimed_name}). Checkpoint "
                f"selection did not take effect. Do not evaluate this adapter as "
                f"a corrected baseline."
            )
            return 1, report

    report["verdict"] = "EXPORT_MATCHES_NO_CHECKPOINT"
    report["detail"] = (
        "The exported adapter matches none of the surviving checkpoints, by bytes "
        "or by tensor values. Treat selection as unconfirmed and do not evaluate "
        "this adapter as a corrected baseline."
    )
    return 2, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="the training run's output directory")
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args(argv)

    directory = Path(args.output_dir)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory

    try:
        code, report = verify(directory)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    print(f"Checkpoint selection — {report['output_dir']}\n")
    print(f"  requested best-checkpoint export : "
          f"{report['requested_load_best_model_at_end']}")
    print(f"  metric                           : {report['metric_for_best_model']} "
          f"(greater_is_better={report['greater_is_better']})")
    print(f"  save_total_limit                 : {report['save_total_limit']}")
    print(f"  checkpoints surviving            : {report.get('checkpoints_found')}")
    print()
    for epoch, loss in report.get("eval_loss_by_epoch", []):
        best = report.get("best_by_eval_loss") or {}
        marker = "  <-- best" if loss == best.get("eval_loss") else ""
        print(f"    epoch {epoch}: eval_loss {loss}{marker}")
    print()
    print(f"  trainer's best checkpoint        : "
          f"{report.get('trainer_best_checkpoint_name')}")
    print(f"  best differs from final          : "
          f"{report.get('best_differs_from_final')}")
    print(f"  exported adapter sha256          : {report.get('exported_sha256', '')[:16]}")
    print(f"  export byte-matches              : {report.get('export_matches')}")
    for comparison in report.get("tensor_comparisons", []):
        if "max_abs_diff" in comparison:
            print(f"  tensor diff vs {comparison['checkpoint']:<18}: "
                  f"max |delta| = {comparison['max_abs_diff']} "
                  f"over {comparison['shared_keys']} tensors")
        else:
            print(f"  tensor diff vs {comparison['checkpoint']:<18}: "
                  f"{comparison.get('reason')}")
    if report.get("tensor_matches") is not None:
        print(f"  export tensor-matches            : {report.get('tensor_matches')}")
    print()
    print(f"  VERDICT: {report['verdict']}")
    print(f"  {report.get('detail', '')}")

    if args.json:
        path = Path(args.json)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                        encoding="utf-8")
        print(f"\n  wrote {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
