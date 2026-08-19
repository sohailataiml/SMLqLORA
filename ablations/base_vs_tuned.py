"""Base vs tuned on the held-out set.

Both models are evaluated through the *same* evaluator, the same prompt strategy,
the same generation settings and the same judge. The only difference is the LoRA
adapter. Anything else would make the comparison unfalsifiable.

Default prompt strategy is `zero_shot` — the weak one-line instruction the model
was trained under. That is the point of the experiment: if the tuned model holds
the behavior under a prompt that the base model cannot hold it under, the
behavior is in the weights.

Usage:
    python -m ablations.base_vs_tuned \
        --base hf:Qwen/Qwen3-1.7B \
        --tuned 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1' \
        --judge anthropic:claude-opus-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablations.reporting import (  # noqa: E402
    cells_to_rows,
    markdown_table,
    plot_base_vs_tuned,
    write_csv,
    write_json,
    write_markdown,
)
from behavior.spec import load_spec  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.judge import DeterministicJudge, LLMJudge  # noqa: E402
from evaluation.metrics import breakdown_by_pressure, failure_mode_counts  # noqa: E402
from evaluation.reproducibility import build_manifest  # noqa: E402
from evaluation.schemas import CellMetrics, EvalRecord, load_scenarios, write_jsonl  # noqa: E402
from models.adapters import EVAL_PARAMS, resolve_model  # noqa: E402
from prompting.strategies import get_strategy  # noqa: E402

DEFAULT_OUTPUT = "results/base_vs_tuned"


def run_comparison(
    *,
    base_spec: str,
    tuned_spec: str,
    judge_spec: str,
    eval_set: str = "scenarios/heldout.jsonl",
    strategy_name: str = "zero_shot",
    output_dir: str = DEFAULT_OUTPUT,
    mock_judge: bool = False,
    max_workers: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    spec = load_spec()
    scenarios = load_scenarios(REPO_ROOT / eval_set)
    strategy = get_strategy(strategy_name, spec)
    judge = (
        DeterministicJudge(spec) if mock_judge else LLMJudge(resolve_model(judge_spec), spec)
    )

    out = REPO_ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    cells: list[CellMetrics] = []
    all_records: list[EvalRecord] = []

    for label, model_spec in (("base", base_spec), ("tuned", tuned_spec)):
        model = resolve_model(model_spec)
        if verbose:
            print(f"[{label}] {model.name} over {len(scenarios)} held-out scenarios...")
        evaluator = Evaluator(
            model, judge, strategy, spec=spec, params=EVAL_PARAMS,
            max_workers=max_workers,
        )
        metrics, records = evaluator.run(scenarios, label=label)
        cells.append(metrics)
        all_records.extend(records)
        if verbose:
            print(
                f"      pass_rate={metrics.pass_rate:.3f} "
                f"adherence={metrics.spec_adherence_mean:.3f} "
                f"robustness={metrics.robustness_mean:.3f} "
                f"leak_rate={metrics.solution_leak_rate:.3f}"
            )

    write_jsonl(out / "judge_transcripts.jsonl", all_records)
    rows = cells_to_rows(cells)
    write_csv(out / "results.csv", rows)

    base_cell, tuned_cell = cells[0], cells[1]
    deltas = {
        metric: round(getattr(tuned_cell, metric) - getattr(base_cell, metric), 4)
        for metric in (
            "spec_adherence_mean",
            "robustness_mean",
            "hint_relevance_mean",
            "pass_rate",
            "solution_leak_rate",
            "premature_confirmation_rate",
        )
    }

    payload = {
        "result_status": "REAL_EXPERIMENT_RESULT",
        "eval_set": eval_set,
        "prompt_strategy": strategy_name,
        "cells": rows,
        "deltas_tuned_minus_base": deltas,
        "by_pressure_type": {
            label: breakdown_by_pressure(
                [r for r in all_records if r.model == cell.model and r.was_evaluated]
            )
            for label, cell in (("base", base_cell), ("tuned", tuned_cell))
        },
    }
    write_json(out / "results.json", payload)

    plot_base_vs_tuned(
        [
            {
                "label": cell.label or cell.model,
                "spec_adherence": cell.spec_adherence_mean,
                "robustness": cell.robustness_mean,
                "hint_relevance": cell.hint_relevance_mean,
                "pass_rate": cell.pass_rate,
            }
            for cell in cells
        ],
        out / "base_vs_tuned.png",
    )

    build_manifest(
        "base_vs_tuned",
        spec=spec,
        scenarios=scenarios,
        scenario_paths=[eval_set],
        models=[resolve_model(base_spec), resolve_model(tuned_spec)],
        strategies=[strategy],
        judge=judge,
        generation_params=EVAL_PARAMS.to_dict(),
        result_status="REAL_EXPERIMENT_RESULT",
        base_model=base_spec,
        tuned_model=tuned_spec,
    ).write(out / "manifest.json")

    write_markdown(
        out / "report.md",
        _render(base_cell, tuned_cell, deltas, all_records, eval_set, strategy_name),
    )
    return payload


def _render(base, tuned, deltas, records, eval_set, strategy_name) -> str:
    def examples_for(model_name: str, passed: bool, limit: int = 3) -> str:
        chosen = [
            r for r in records
            if r.model == model_name and r.passed is passed and r.was_evaluated
        ][:limit]
        if not chosen:
            return "_None._"
        blocks = []
        for record in chosen:
            body = record.model_response.strip().replace("\n", "\n> ")[:500]
            codes = ", ".join(record.failure_reasons) or "—"
            blocks.append(
                f"**`{record.scenario_id}`** (pressure=`{record.pressure_type.value}`, "
                f"codes: {codes})\n\n> {body}\n"
            )
        return "\n".join(blocks)

    return "\n".join(
        [
            "# Base vs Tuned",
            "",
            "> **STATUS: REAL EXPERIMENT RESULT.**"
            if not (base.partial or tuned.partial)
            else (
                "> **STATUS: PARTIAL — at least one model could not be measured on "
                "every scenario.** Infrastructure failures are excluded from the "
                "rates below, so the two models have DIFFERENT denominators. Read "
                "the counts table before comparing anything."
            ),
            "",
            f"Held-out set: `{eval_set}`  ",
            f"Prompt strategy: `{strategy_name}` (the weak prompt both models see)  ",
            "",
            "## Counts",
            "",
            "Every rate below is over `measured`, not over the scenario file. A "
            "single number cannot describe both models when one of them errored, "
            "so both denominators are stated. Failure-mode codes are multi-label: "
            "one response can carry several, so a code count may legitimately "
            "exceed `measured`.",
            "",
            markdown_table(
                [
                    {
                        "model": cell.label,
                        "attempted": cell.attempted_count,
                        "measured": cell.scenario_count,
                        "infrastructure_errors": cell.infrastructure_error_count,
                        "subject_calls_ok": cell.successful_subject_calls,
                        "judge_calls_ok": cell.successful_judge_calls,
                    }
                    for cell in (base, tuned)
                ],
                ["model", "attempted", "measured", "infrastructure_errors",
                 "subject_calls_ok", "judge_calls_ok"],
            ),
            "",
            "## Headline",
            "",
            markdown_table(
                [
                    {
                        "metric": key.replace("_mean", ""),
                        "base": getattr(base, key),
                        "tuned": getattr(tuned, key),
                        "delta": value,
                    }
                    for key, value in deltas.items()
                ],
                ["metric", "base", "tuned", "delta"],
            ),
            "",
            "## Robustness under pressure",
            "",
            markdown_table(
                [
                    {
                        "model": cell.label,
                        "clean_pass_rate": cell.clean_pass_rate,
                        "adversarial_pass_rate": cell.adversarial_pass_rate,
                    }
                    for cell in (base, tuned)
                ],
                ["model", "clean_pass_rate", "adversarial_pass_rate"],
            ),
            "",
            "## Failure modes",
            "",
            markdown_table(
                [
                    {"model": "base", **base.failure_modes},
                    {"model": "tuned", **tuned.failure_modes},
                ],
                ["model"] + sorted(set(base.failure_modes) | set(tuned.failure_modes)),
            ),
            "",
            "## Representative BASE failures",
            "",
            examples_for(base.model, passed=False),
            "",
            "## Representative TUNED failures",
            "",
            "_Shown deliberately: the tuned model is not perfect, and cherry-picking "
            "only its successes would misrepresent the result._",
            "",
            examples_for(tuned.model, passed=False),
            "",
            "## Representative TUNED passes",
            "",
            examples_for(tuned.model, passed=True),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base and tuned checkpoints.")
    parser.add_argument("--base", required=True, help="e.g. hf:Qwen/Qwen3-1.7B")
    parser.add_argument("--tuned", required=True,
                        help="e.g. 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1'")
    parser.add_argument("--judge", default="anthropic:claude-opus-5")
    parser.add_argument("--eval-set", default="scenarios/heldout.jsonl")
    parser.add_argument("--strategy", default="zero_shot")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--mock-judge", action="store_true")
    args = parser.parse_args(argv)

    run_comparison(
        base_spec=args.base,
        tuned_spec=args.tuned,
        judge_spec=args.judge,
        eval_set=args.eval_set,
        strategy_name=args.strategy,
        output_dir=args.output,
        mock_judge=args.mock_judge,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
