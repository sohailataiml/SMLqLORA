"""Regenerate experiment reports from saved transcripts — no API calls.

Raw judge transcripts are the evidence; the reports are a rendering of them. When
the analysis changes (a metric is corrected, a denominator policy is fixed), the
reports must be regenerated from the same records rather than by re-running an
expensive experiment.

    python scripts/reanalyze.py --results-dir results/prompt_ceiling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ablations.prompt_ceiling import evaluate_gate, write_reports  # noqa: E402
from behavior.spec import load_spec  # noqa: E402
from evaluation.judge import DeterministicJudge  # noqa: E402
from evaluation.metrics import aggregate, group_by_cell  # noqa: E402
from evaluation.schemas import (  # noqa: E402
    EvalRecord,
    ErrorKind,
    iter_jsonl,
    load_scenario_files,
)
from models.adapters import EVAL_PARAMS  # noqa: E402
from prompting.strategies import all_strategies  # noqa: E402

EVAL_SETS = ("scenarios/clean.jsonl", "scenarios/adversarial.jsonl")


class _RecordedJudge(DeterministicJudge):
    """Stands in for the judge that produced the saved records."""

    def __init__(self, spec, name: str):
        super().__init__(spec, name=name)
        self.prompt_version = "recorded"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-render reports from transcripts.")
    parser.add_argument("--results-dir", default="results/prompt_ceiling")
    parser.add_argument("--records", default=None,
                        help="override the records path (default <results-dir>/all_records.jsonl)")
    args = parser.parse_args(argv)

    out = REPO_ROOT / args.results_dir
    records_path = Path(args.records) if args.records else out / "all_records.jsonl"
    if not records_path.exists():
        raise SystemExit(
            f"No records at {records_path}. Run the experiment first, or pass "
            f"--records with the transcript file to re-analyse."
        )

    spec = load_spec()
    records = [EvalRecord.model_validate(row) for row in iter_jsonl(records_path)]
    print(f"loaded {len(records)} records from {records_path.relative_to(REPO_ROOT)}")

    infra = sum(1 for r in records if r.error_kind is ErrorKind.INFRASTRUCTURE)
    refusals = sum(1 for r in records if r.error_kind is ErrorKind.REFUSAL)
    print(f"  infrastructure failures (excluded from rates): {infra}")
    print(f"  model refusals (counted as failures):          {refusals}")

    cells = []
    unmeasured = []
    for (model, strategy), group in group_by_cell(records).items():
        try:
            cells.append(
                aggregate(group, model=model, prompt_strategy=strategy,
                          label=f"{model} | {strategy}")
            )
        except ValueError:
            unmeasured.append(f"{model} | {strategy}")
            print(f"  NOT MEASURED: {model} | {strategy}")

    if not cells:
        raise SystemExit("No cell has any measurable record.")

    judge_name = next(
        (r.judge.judge_model for r in records if r.judge), "unknown"
    )
    judge = _RecordedJudge(spec, judge_name)
    scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])

    decision = evaluate_gate(
        cells, records, spec, result_status="REAL_EXPERIMENT_RESULT",
        unmeasured_cells=unmeasured,
    )

    write_reports(
        out, cells, records, decision, spec,
        models=[], strategies=all_strategies(spec), judge=judge,
        scenarios=scenarios, scenario_paths=EVAL_SETS,
        result_status=decision.status,
    )

    print()
    print("=" * 72)
    print(decision.headline)
    print(decision.evidence)
    print("=" * 72)
    print(f"reports rewritten in {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
