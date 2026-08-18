"""Data-efficiency ablation: how much data does the behavior actually need?

Trains one checkpoint per dataset size on *nested* subsets, evaluates every
checkpoint on the same held-out set with the same harness, and reports the
smallest N whose checkpoint clears the reliability thresholds in
`behavior/spec.yaml` — the **Minimum Viable Dataset Size**.

Everything except N is held constant: base model, seed, LoRA config, training
arguments, prompt strategy, judge, generation settings. The subsets are nested
so the curve varies quantity rather than quantity *and* composition.

The MVDS is derived from measured results. When a size has not been evaluated,
it is reported as NOT RUN, never interpolated.

Usage:
    python -m ablations.data_efficiency --plan
    python -m ablations.data_efficiency --train --sizes 125 250 500 600
    python -m ablations.data_efficiency --evaluate --judge anthropic:claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablations.reporting import (  # noqa: E402
    markdown_table,
    plot_data_efficiency,
    write_csv,
    write_json,
    write_markdown,
)
from behavior.spec import BehaviorSpec, load_spec  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.judge import DeterministicJudge, LLMJudge  # noqa: E402
from evaluation.reproducibility import build_manifest  # noqa: E402
from evaluation.schemas import load_scenarios, write_jsonl  # noqa: E402
from models.adapters import EVAL_PARAMS, resolve_model  # noqa: E402
from prompting.strategies import get_strategy  # noqa: E402
from training.dataset import load_accepted, suggested_sweep_sizes  # noqa: E402

DEFAULT_OUTPUT = "results/data_efficiency"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_ACCEPTED = "data/accepted/v1.jsonl"


@dataclass
class SweepPoint:
    dataset_size: int
    run_name: str
    checkpoint_dir: Path
    trained: bool = False
    evaluated: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.evaluated:
            return "EVALUATED"
        if self.trained:
            return "TRAINED (not evaluated)"
        return "NOT RUN"


def plan_sweep(
    sizes: Sequence[int] | None,
    *,
    accepted_path: str = DEFAULT_ACCEPTED,
    output_dir: str = DEFAULT_OUTPUT,
) -> list[SweepPoint]:
    """Resolve the sweep sizes against the data that actually exists."""
    path = REPO_ROOT / accepted_path
    if not path.exists():
        raise SystemExit(
            f"No accepted dataset at {path}. Run generation and the quality gate "
            f"first, then plan the sweep."
        )
    total = len(load_accepted(path))

    if sizes:
        too_big = [s for s in sizes if s > total]
        if too_big:
            raise SystemExit(
                f"Requested sizes {too_big} exceed the {total} accepted examples "
                f"available. Either generate more data or use --sizes within range. "
                f"Suggested: {suggested_sweep_sizes(total)}"
            )
        resolved = sorted(set(sizes))
    else:
        resolved = suggested_sweep_sizes(total)

    if len(resolved) < 4:
        print(
            f"warning: the brief asks for at least four checkpoints; this sweep has "
            f"{len(resolved)} because only {total} accepted examples exist."
        )

    return [
        SweepPoint(
            dataset_size=size,
            run_name=f"socratic-n{size}",
            checkpoint_dir=REPO_ROOT / "outputs" / f"socratic-n{size}",
        )
        for size in resolved
    ]


def train_point(point: SweepPoint, *, config: str, dry_run: bool = False) -> bool:
    """Invoke training for one sweep point as a subprocess (isolates GPU memory)."""
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--config",
        config,
        "--limit",
        str(point.dataset_size),
        "--run-name",
        point.run_name,
    ]
    if dry_run:
        command.append("--dry-run")

    print(f"\n=== training N={point.dataset_size} ===")
    print("  " + " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    point.trained = result.returncode == 0
    if not point.trained:
        print(f"  training failed for N={point.dataset_size} (exit {result.returncode})")
    return point.trained


def evaluate_point(
    point: SweepPoint,
    *,
    base_model: str,
    judge,
    spec: BehaviorSpec,
    scenarios,
    strategy_name: str,
    max_workers: int,
) -> bool:
    if not point.checkpoint_dir.exists():
        print(f"  N={point.dataset_size}: no checkpoint at {point.checkpoint_dir} — NOT RUN")
        return False

    model = resolve_model(f"peft:{base_model}+{point.checkpoint_dir}")
    evaluator = Evaluator(
        model, judge, get_strategy(strategy_name, spec), spec=spec,
        params=EVAL_PARAMS, max_workers=max_workers,
    )
    metrics, records = evaluator.run(
        scenarios,
        transcript_path=str(
            REPO_ROOT / DEFAULT_OUTPUT / "transcripts" / f"n{point.dataset_size}.jsonl"
        ),
        label=f"N={point.dataset_size}",
    )
    point.metrics = {
        "dataset_size": point.dataset_size,
        "spec_adherence": metrics.spec_adherence_mean,
        "robustness": metrics.robustness_mean,
        "hint_relevance": metrics.hint_relevance_mean,
        "pass_rate": metrics.pass_rate,
        "solution_leak_rate": metrics.solution_leak_rate,
        "premature_confirmation_rate": metrics.premature_confirmation_rate,
        "scenario_count": metrics.scenario_count,
        "partial": metrics.partial,
    }
    point.evaluated = True
    print(
        f"  N={point.dataset_size}: pass={metrics.pass_rate:.3f} "
        f"adherence={metrics.spec_adherence_mean:.3f} "
        f"robustness={metrics.robustness_mean:.3f}"
    )
    return True


def minimum_viable_size(
    points: Sequence[SweepPoint], spec: BehaviorSpec
) -> tuple[int | None, str]:
    """Smallest evaluated N clearing every data-efficiency threshold."""
    gate = spec.gates.data_efficiency
    evaluated = sorted(
        (p for p in points if p.evaluated), key=lambda p: p.dataset_size
    )
    if not evaluated:
        return None, (
            "NOT RUN — no checkpoint has been trained and evaluated, so the "
            "Minimum Viable Dataset Size cannot be derived."
        )

    for point in evaluated:
        m = point.metrics
        if (
            m["pass_rate"] >= gate.required_pass_rate
            and m["spec_adherence"] >= gate.required_spec_adherence
            and m["robustness"] >= gate.required_robustness
        ):
            return point.dataset_size, (
                f"N={point.dataset_size} is the smallest evaluated size clearing all "
                f"thresholds (pass_rate >= {gate.required_pass_rate}, "
                f"spec_adherence >= {gate.required_spec_adherence}, "
                f"robustness >= {gate.required_robustness})."
            )

    largest = evaluated[-1]
    return None, (
        f"No evaluated size cleared the thresholds. The largest tested, "
        f"N={largest.dataset_size}, reached pass_rate="
        f"{largest.metrics['pass_rate']:.3f}, spec_adherence="
        f"{largest.metrics['spec_adherence']:.3f}, robustness="
        f"{largest.metrics['robustness']:.3f}. More data, or better data, is needed."
    )


def write_reports(points: Sequence[SweepPoint], spec: BehaviorSpec, *,
                  output_dir: str, scenarios, eval_set: str, judge) -> None:
    out = REPO_ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = [p.metrics for p in points if p.evaluated]
    mvds, rationale = minimum_viable_size(points, spec)
    gate = spec.gates.data_efficiency

    write_json(
        out / "results.json",
        {
            "result_status": "REAL_EXPERIMENT_RESULT" if rows else "NOT_RUN",
            "minimum_viable_dataset_size": mvds,
            "rationale": rationale,
            "thresholds": {
                "required_pass_rate": gate.required_pass_rate,
                "required_spec_adherence": gate.required_spec_adherence,
                "required_robustness": gate.required_robustness,
            },
            "points": [
                {
                    "dataset_size": p.dataset_size,
                    "run_name": p.run_name,
                    "status": p.status,
                    **p.metrics,
                }
                for p in points
            ],
        },
    )
    if rows:
        write_csv(out / "results.csv", rows)
        plot_data_efficiency(rows, out / "performance_vs_n.png",
                             threshold=gate.required_pass_rate)

    build_manifest(
        "data_efficiency",
        spec=spec,
        scenarios=scenarios,
        scenario_paths=[eval_set],
        judge=judge,
        result_status="REAL_EXPERIMENT_RESULT" if rows else "NOT_RUN",
    ).write(out / "manifest.json")

    status_rows = [
        {"dataset_size": p.dataset_size, "run_name": p.run_name, "status": p.status}
        for p in points
    ]
    write_markdown(
        out / "report.md",
        "\n".join(
            [
                "# Data Efficiency",
                "",
                (
                    "> **STATUS: REAL EXPERIMENT RESULT.**"
                    if rows
                    else "> **STATUS: NOT RUN.** No checkpoint has been trained and "
                    "evaluated yet. The sweep below is planned, not measured."
                ),
                "",
                "## Question",
                "",
                "> What is the smallest training set that reliably holds the behavior?",
                "",
                "## Minimum Viable Dataset Size",
                "",
                f"**{mvds if mvds is not None else 'NOT DETERMINED'}**",
                "",
                rationale,
                "",
                "## Sweep status",
                "",
                markdown_table(status_rows, ["dataset_size", "run_name", "status"]),
                "",
                "## Measured points",
                "",
                markdown_table(
                    rows,
                    ["dataset_size", "spec_adherence", "robustness", "hint_relevance",
                     "pass_rate", "solution_leak_rate"],
                )
                if rows
                else "_No checkpoint has been evaluated._",
                "",
                "## Held constant across every point",
                "",
                "- base model, revision and seed",
                "- LoRA rank, alpha, dropout and target modules",
                "- all training arguments (epochs, LR, schedule, batch size)",
                "- prompt strategy, generation settings, judge and rubric",
                "- held-out evaluation set",
                "",
                "Subsets are **nested**: the N=125 set is a prefix of N=250, and so on, "
                "so the curve varies quantity rather than quantity and composition.",
            ]
        ),
    )
    print(f"\nreports written to {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data-efficiency sweep.")
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--config", default="training/configs/qlora_qwen3_1_7b.yaml")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-set", default="scenarios/heldout.jsonl")
    parser.add_argument("--strategy", default="zero_shot")
    parser.add_argument("--judge", default="anthropic:claude-opus-5")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--plan", action="store_true", help="show the sweep and exit")
    parser.add_argument("--train", action="store_true", help="train every point")
    parser.add_argument("--evaluate", action="store_true", help="evaluate every checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="validate training config only")
    parser.add_argument("--mock-judge", action="store_true")
    args = parser.parse_args(argv)

    spec = load_spec()
    points = plan_sweep(args.sizes, accepted_path=args.accepted, output_dir=args.output)

    print("Planned sweep:")
    for point in points:
        exists = "checkpoint present" if point.checkpoint_dir.exists() else "not trained"
        print(f"  N={point.dataset_size:<6} {point.run_name:<20} {exists}")

    if args.plan:
        return 0

    if args.train:
        for point in points:
            train_point(point, config=args.config, dry_run=args.dry_run)

    scenarios = load_scenarios(REPO_ROOT / args.eval_set)
    judge = (
        DeterministicJudge(spec)
        if args.mock_judge
        else LLMJudge(resolve_model(args.judge), spec)
    )

    if args.evaluate:
        print(f"\n=== evaluating on {args.eval_set} ({len(scenarios)} scenarios) ===")
        for point in points:
            evaluate_point(
                point,
                base_model=args.base_model,
                judge=judge,
                spec=spec,
                scenarios=scenarios,
                strategy_name=args.strategy,
                max_workers=args.max_workers,
            )

    write_reports(points, spec, output_dir=args.output, scenarios=scenarios,
                  eval_set=args.eval_set, judge=judge)

    mvds, rationale = minimum_viable_size(points, spec)
    print()
    print("=" * 72)
    print(f"MINIMUM VIABLE DATASET SIZE: {mvds if mvds is not None else 'NOT DETERMINED'}")
    print(rationale)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
