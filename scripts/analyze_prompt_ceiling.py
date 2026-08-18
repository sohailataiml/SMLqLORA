"""Derive every analysis artifact from stored prompt-ceiling transcripts.

Reads `all_records.jsonl` and writes the failure-mode analysis, the proposed
training distribution, the human-validation sheet and the plots. Makes **zero**
API calls, so it can be re-run freely after any change to the analysis code
without spending anything or altering the underlying evidence.

    python scripts/analyze_prompt_ceiling.py
    python scripts/analyze_prompt_ceiling.py --results-dir results/prompt_ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablations.reporting import write_json  # noqa: E402
from evaluation.failure_analysis import (  # noqa: E402
    STRONG_STRATEGIES,
    UNDERPOWERED_THRESHOLD,
    analyze_failure_modes,
    measured,
    propose_training_distribution,
)
from evaluation.human_validation import export_validation_csv  # noqa: E402
from evaluation.plots import (  # noqa: E402
    plot_adversarial,
    plot_failure_mode_bars,
    plot_metric_by_model_strategy,
)
from evaluation.resume import is_mock_record, read_records  # noqa: E402
from evaluation.schemas import EvalRecord  # noqa: E402


def load_records(path: Path) -> list[EvalRecord]:
    if not path.exists():
        raise SystemExit(
            f"No transcripts at {path}. Run the ablation first "
            f"(`python -m ablations.prompt_ceiling --plan` shows what it costs)."
        )
    # Keep infrastructure failures: completeness accounting has to be able to
    # report a cell that was wiped out rather than let it vanish from the table.
    records, _seen, malformed = read_records(path)
    records = [r for r in records if not is_mock_record(r)]
    if malformed:
        print(f"warning: skipped {malformed} unreadable line(s) in {path.name}")
    if not records:
        raise SystemExit(f"{path} contains no real records to analyze.")
    return records


def completeness(records: Sequence[EvalRecord], expected_per_cell: int = 36) -> dict:
    """Per-cell accounting in the exact terms Step 7 asks for."""
    cells: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        cells.setdefault((record.model, record.prompt_strategy), []).append(record)

    rows = []
    for (model, strategy), group in sorted(cells.items()):
        obs = measured(group)
        rows.append({
            "model": model,
            "prompt_strategy": strategy,
            "requested_count": expected_per_cell,
            "attempted_count": len(group),
            "successful_subject_calls": sum(1 for r in obs if not r.error),
            "successful_judge_calls": sum(1 for r in obs if r.judge is not None),
            "infrastructure_error_count": len(group) - len(obs),
            "valid_evaluation_count": len(obs),
            "status": "COMPLETE" if len(obs) >= expected_per_cell else "PARTIAL",
        })
    return {
        "expected_per_cell": expected_per_cell,
        "cells": rows,
        "cells_complete": sum(1 for r in rows if r["status"] == "COMPLETE"),
        "cells_total": len(rows),
    }


def render_failure_report(analysis: dict, distribution: dict, cover: dict) -> str:
    residual = analysis["residual_under_strong_prompts"]
    lines = [
        "# Failure-Mode Analysis",
        "",
        "> Derived entirely from stored transcripts. No model was called to "
        "produce this document.",
        "",
        f"Measured evaluations: **{analysis['n_measured']}**  ",
        f"Failures: **{analysis['n_failed']}**  ",
        f"Overall pass rate: **{analysis['overall_pass_rate']:.3f}**",
        "",
        f"Cells complete: **{cover['cells_complete']} / {cover['cells_total']}** "
        f"(a cell is complete at {cover['expected_per_cell']} valid evaluations)",
        "",
        "## Reading these numbers",
        "",
        f"Any slice with fewer than {UNDERPOWERED_THRESHOLD} observations is "
        "marked `underpowered`. Those rates are reported because they are the "
        "measurement we have, not because they are conclusive — with n=3 a "
        "single response moves the rate by 33 points.",
        "",
        "## Per-cell completeness",
        "",
        "| model | strategy | requested | attempted | subject ok | judge ok | "
        "infra errors | valid | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in cover["cells"]:
        lines.append(
            f"| {row['model']} | {row['prompt_strategy']} | "
            f"{row['requested_count']} | {row['attempted_count']} | "
            f"{row['successful_subject_calls']} | {row['successful_judge_calls']} | "
            f"{row['infrastructure_error_count']} | "
            f"{row['valid_evaluation_count']} | {row['status']} |"
        )

    lines += [
        "",
        "## Failure modes overall",
        "",
        "| code | count | rate | worst model | worst strategy | worst pressure |",
        "|---|---:|---:|---|---|---|",
    ]
    for mode in analysis["failure_modes"]:
        def top(key: str) -> str:
            groups = mode[key]
            if not groups or groups[0]["count"] == 0:
                return "—"
            g = groups[0]
            flag = "*" if g["underpowered"] else ""
            return f"{g['key']} ({g['count']}/{g['n']}{flag})"

        lines.append(
            f"| `{mode['failure_code']}` | {mode['count']} | {mode['rate']:.3f} | "
            f"{top('by_model')} | {top('by_prompt_strategy')} | "
            f"{top('by_pressure_type')} |"
        )
    lines += ["", "`*` = underpowered slice.", ""]

    lines += [
        "## What survives strong prompting",
        "",
        f"Restricted to {', '.join(STRONG_STRATEGIES)} — the cells where "
        "prompting has already done its work. This residue is what fine-tuning "
        "would have to fix.",
        "",
    ]
    if residual.get("n_measured"):
        lines += [
            f"Measured: **{residual['n_measured']}**, "
            f"failed: **{residual['n_failed']}**, "
            f"pass rate: **{residual['pass_rate']:.3f}**",
            "",
            "| surviving failure mode | count |",
            "|---|---:|",
        ]
        for code, count in residual["surviving_failure_modes"].items():
            lines.append(f"| `{code}` | {count} |")
        lines += [
            "",
            "### Pressure types, worst first (strong prompts only)",
            "",
            "| pressure | n | passes | pass rate | leak rate | |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in residual["pressure_type_ranking"]:
            flag = "underpowered" if row["underpowered"] else ""
            lines.append(
                f"| {row['pressure_type']} | {row['n']} | {row['passes']} | "
                f"{row['pass_rate']:.3f} | {row['solution_leak_rate']:.3f} | "
                f"{flag} |"
            )
    else:
        lines.append("_No strong-prompt records measured._")

    lines += [
        "",
        "## Proposed training distribution",
        "",
        f"Basis: {distribution['basis']} "
        f"(n={distribution['n_records_in_basis']}).",
        "",
        f"Rule: every dimension gets a floor of "
        f"{distribution['rule']['floor_share_per_dimension']:.0%}; the rest is "
        f"allocated in proportion to measured failure rate; no dimension exceeds "
        f"{distribution['rule']['cap_share_per_dimension']:.0%}.",
        "",
        "| dimension | share | observed n | failures | failure rate | |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dim, share in distribution["distribution"].items():
        stat = distribution["measured_inputs"][dim]
        flag = "underpowered" if stat["underpowered"] else ""
        lines.append(
            f"| {dim} | {share:.1%} | {stat['n_observed']} | {stat['failures']} | "
            f"{stat['failure_rate']:.3f} | {flag} |"
        )
    lines += [
        "",
        "This distribution is **provisional**. It is computed from whichever "
        "cells are currently measured, and must be recomputed once the "
        "experiment is complete — re-running this script is the whole procedure.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/prompt_ceiling")
    parser.add_argument("--expected-per-cell", type=int, default=36)
    parser.add_argument("--validation-sample", type=int, default=40)
    args = parser.parse_args(argv)

    out = REPO_ROOT / args.results_dir
    records = load_records(out / "all_records.jsonl")
    obs = measured(records)

    analysis = analyze_failure_modes(records)
    distribution = propose_training_distribution(records)
    cover = completeness(records, expected_per_cell=args.expected_per_cell)

    write_json(out / "failure_modes.json", {**analysis, "completeness": cover})
    write_json(out / "proposed_training_distribution.json", distribution)
    (out / "failure_modes.md").write_text(
        render_failure_report(analysis, distribution, cover), encoding="utf-8"
    )

    n_sample = export_validation_csv(
        records, out / "human_validation.csv", target=args.validation_sample
    )

    plots = [
        plot_metric_by_model_strategy(
            obs, out / "spec_adherence_by_model_strategy.png",
            metric="spec_adherence", title="Spec adherence by model x strategy",
        ),
        plot_metric_by_model_strategy(
            obs, out / "robustness_by_model_strategy.png",
            metric="robustness", title="Robustness by model x strategy",
        ),
        plot_metric_by_model_strategy(
            obs, out / "pass_rate_by_model_strategy.png",
            metric="pass_rate", title="Pass rate by model x strategy",
        ),
        plot_failure_mode_bars(obs, out / "failure_modes_by_strategy.png"),
        plot_adversarial(obs, out / "adversarial_by_pressure.png"),
    ]

    print(f"analyzed {len(obs)} measured records from {len(records)} stored")
    print(f"  failure_modes.json                     "
          f"{len(analysis['failure_modes'])} codes")
    print(f"  proposed_training_distribution.json    "
          f"{len(distribution['distribution'])} dimensions")
    print(f"  human_validation.csv                   {n_sample} rows "
          f"(human columns intentionally blank)")
    print(f"  failure_modes.md")
    for path in plots:
        print(f"  {path.name if path else '(plot skipped — matplotlib absent)'}")
    print(f"\ncells complete: {cover['cells_complete']}/{cover['cells_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
