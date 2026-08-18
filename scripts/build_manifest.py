"""Build the top-level experiment manifest at `results/manifest.json`.

One file that answers "what is the current state of this project, and what
produced each number". It reads the artifacts that actually exist on disk rather
than describing what was intended — so an experiment that has not run shows up as
NOT RUN, and one that ran partially shows up as PARTIAL.

    python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from behavior.spec import load_spec  # noqa: E402
from evaluation.judge import JUDGE_PROMPT_VERSION, judge_prompt_hash  # noqa: E402
from evaluation.reproducibility import (  # noqa: E402
    dependency_hash,
    git_commit,
    git_is_dirty,
    package_versions,
)
from evaluation.schemas import load_scenarios, scenarios_hash  # noqa: E402
from generation.prompts import (  # noqa: E402
    GENERATION_PROMPT_VERSION,
    generation_prompt_hash,
)
from prompting.strategies import all_strategies  # noqa: E402

SCENARIO_FILES = ("clean", "adversarial", "heldout")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def experiment_status(name: str, results_path: Path) -> dict[str, Any]:
    payload = _read_json(results_path)
    if payload is None:
        return {"experiment": name, "status": "NOT_RUN",
                "note": f"no artifacts at {results_path.relative_to(REPO_ROOT)}"}

    entry: dict[str, Any] = {
        "experiment": name,
        "status": payload.get("result_status", "UNKNOWN"),
        "artifacts": str(results_path.parent.relative_to(REPO_ROOT)),
    }
    if gate := payload.get("gate"):
        entry["gate_justified"] = gate.get("justified")
        entry["gate_complete"] = gate.get("experiment_complete")
        entry["gate_caveats"] = gate.get("caveats", [])
    if "minimum_viable_dataset_size" in payload:
        entry["minimum_viable_dataset_size"] = payload["minimum_viable_dataset_size"]
    if deltas := payload.get("deltas_tuned_minus_base"):
        entry["deltas_tuned_minus_base"] = deltas
    return entry


def dataset_status() -> list[dict[str, Any]]:
    versions_dir = REPO_ROOT / "data" / "versions"
    if not versions_dir.exists():
        return []
    out = []
    for directory in sorted(versions_dir.iterdir()):
        report = _read_json(directory / "report.json")
        if report is None:
            continue
        out.append(
            {
                "dataset_version": report.get("dataset_version", directory.name),
                "dataset_hash": report.get("dataset_hash"),
                "candidate_count": report.get("candidate_count"),
                "accepted_count": report.get("accepted_count"),
                "rejected_count": report.get("rejected_count"),
                "acceptance_rate": report.get("acceptance_rate"),
                "teacher_model": (report.get("teacher") or {}).get("teacher_model"),
                "judge_model": (report.get("judge") or {}).get("judge_model"),
                "top_rejections": dict(
                    list((report.get("rejections_by_reason") or {}).items())[:5]
                ),
            }
        )
    return out


def checkpoint_status() -> list[dict[str, Any]]:
    outputs = REPO_ROOT / "outputs"
    if not outputs.exists():
        return []
    found = []
    for directory in sorted(outputs.iterdir()):
        metadata = _read_json(directory / "checkpoint_metadata.json")
        if metadata is None:
            continue
        found.append(
            {
                "run_name": metadata.get("run_name"),
                "base_model": metadata.get("base_model"),
                "base_model_revision": metadata.get("base_model_revision"),
                "dataset_version": metadata.get("dataset_version"),
                "dataset_train_size": metadata.get("dataset_train_size"),
                "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                "seed": metadata.get("seed"),
                "completed": metadata.get("completed", False),
                "path": str(directory.relative_to(REPO_ROOT)),
            }
        )
    return found


def main() -> int:
    spec = load_spec()
    versions = package_versions()

    scenario_hashes = {}
    total_scenarios = 0
    for name in SCENARIO_FILES:
        path = REPO_ROOT / "scenarios" / f"{name}.jsonl"
        if path.exists():
            scenarios = load_scenarios(path)
            scenario_hashes[name] = {
                "count": len(scenarios),
                "hash": scenarios_hash(scenarios),
            }
            total_scenarios += len(scenarios)

    datasets = dataset_status()
    checkpoints = checkpoint_status()

    manifest = {
        "project": "socratic-debug-tutor",
        "manifest_version": "1.0.0",
        "experiment_timestamp": datetime.now(timezone.utc).isoformat(),

        "behavior_spec_version": spec.version,
        "behavior_spec_sha256": spec.spec_sha256,

        "eval_sets": scenario_hashes,
        "eval_set_hash": scenario_hashes.get("heldout", {}).get("hash"),
        "eval_scenarios_total": total_scenarios,

        "prompt_versions": [s.describe() for s in all_strategies(spec)],
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": judge_prompt_hash(spec),
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "generation_prompt_sha256": generation_prompt_hash(spec),

        "datasets": datasets,
        "dataset_version": datasets[-1]["dataset_version"] if datasets else None,
        "dataset_hash": datasets[-1]["dataset_hash"] if datasets else None,

        "base_model": "Qwen/Qwen3-1.7B",
        "base_model_revision": "main (pin a commit sha before final runs)",
        "tuned_model": checkpoints[-1]["path"] if checkpoints else None,
        "tuned_model_revision": (
            checkpoints[-1]["dataset_fingerprint"] if checkpoints else None
        ),
        "checkpoints": checkpoints,

        "experiments": [
            experiment_status("prompt_ceiling",
                              REPO_ROOT / "results/prompt_ceiling/results.json"),
            experiment_status("base_vs_tuned",
                              REPO_ROOT / "results/base_vs_tuned/results.json"),
            experiment_status("data_efficiency",
                              REPO_ROOT / "results/data_efficiency/results.json"),
        ],

        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "dependency_versions": versions,
        "dependency_lock_hash": dependency_hash(versions),
        "python_version": sys.version.split()[0],
    }

    out = REPO_ROOT / "results" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"  behavior spec   v{spec.version} ({spec.spec_sha256[:12]})")
    print(f"  eval scenarios  {total_scenarios}")
    print(f"  datasets        {len(datasets)}")
    print(f"  checkpoints     {len(checkpoints)}")
    for entry in manifest["experiments"]:
        print(f"  {entry['experiment']:<16} {entry['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
