"""Aggregation of per-example records into reportable metrics.

Deliberately simple and pure: records in, numbers out. Kept separate from the
evaluator so aggregation can be unit-tested without any model, and so the same
functions serve the prompt-ceiling, base-vs-tuned and data-efficiency reports.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from evaluation.schemas import CellMetrics, EvalRecord, PressureType


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(matching: int, total: int) -> float:
    return matching / total if total else 0.0


def failure_mode_counts(records: Iterable[EvalRecord]) -> dict[str, int]:
    """How often each failure code appears across the records."""
    counter: Counter[str] = Counter()
    for record in records:
        for code in record.failure_reasons:
            counter[code] += 1
    return dict(counter.most_common())


def aggregate(
    records: Sequence[EvalRecord],
    *,
    model: str | None = None,
    model_family: str | None = None,
    prompt_strategy: str | None = None,
    label: str = "",
    notes: str = "",
    reused_count: int = 0,
) -> CellMetrics:
    """Collapse one experimental cell into a metrics row.

    Denominator policy, which matters more than it looks:

    * **Model failures stay in.** A refusal, an empty response or a malformed
      answer is the model failing to hold the behavior. Excluding those would
      flatter whichever model is least reliable.
    * **Infrastructure failures come out.** An exhausted quota or a dropped
      connection measures the billing account, not the model. Counting those as
      behavioral failures would let an outage masquerade as a result — so they
      are excluded from every rate and reported separately, and the cell is
      marked `partial` so no report can quietly present it as complete.
    """
    if not records:
        raise ValueError("aggregate() requires at least one record")

    attempted = len(records)
    infrastructure = [r for r in records if not r.was_evaluated]
    records = [r for r in records if r.was_evaluated]
    if not records:
        raise ValueError(
            f"aggregate() received {attempted} record(s), all of which failed for "
            f"infrastructure reasons (e.g. exhausted quota). There is nothing to "
            f"measure — fix the provider account and re-run this cell."
        )

    total = len(records)
    judged = [r for r in records if r.judge is not None]

    adherence = _mean([r.judge.spec_adherence for r in judged])
    robustness_records = [
        r for r in judged if r.pressure_type is not PressureType.NORMAL
    ] or judged
    robustness = _mean([r.judge.robustness for r in robustness_records])
    relevance = _mean([r.judge.hint_relevance for r in judged])

    passes = sum(1 for r in records if r.passed)
    adversarial = [r for r in records if r.pressure_type is not PressureType.NORMAL]
    clean = [r for r in records if r.pressure_type is PressureType.NORMAL]

    def code_rate(code: str) -> float:
        return _rate(sum(1 for r in records if code in r.failure_reasons), total)

    return CellMetrics(
        model=model or records[0].model,
        model_family=model_family or records[0].model_family,
        prompt_strategy=prompt_strategy or records[0].prompt_strategy,
        scenario_count=total,
        attempted_count=attempted,
        infrastructure_error_count=len(infrastructure),
        successful_subject_calls=sum(1 for r in records if not r.error),
        successful_judge_calls=len(judged),
        reused_count=reused_count,
        partial=bool(infrastructure),
        spec_adherence_mean=round(adherence, 4),
        robustness_mean=round(robustness, 4),
        hint_relevance_mean=round(relevance, 4),
        pass_rate=round(_rate(passes, total), 4),
        failure_rate=round(1.0 - _rate(passes, total), 4),
        solution_leak_rate=round(code_rate("SOLUTION_LEAK"), 4),
        premature_confirmation_rate=round(code_rate("PREMATURE_CONFIRMATION"), 4),
        multiple_hints_rate=round(code_rate("MULTIPLE_HINTS"), 4),
        adversarial_pass_rate=(
            round(_rate(sum(1 for r in adversarial if r.passed), len(adversarial)), 4)
            if adversarial
            else None
        ),
        clean_pass_rate=(
            round(_rate(sum(1 for r in clean if r.passed), len(clean)), 4)
            if clean
            else None
        ),
        failure_modes=failure_mode_counts(records),
        error_count=sum(1 for r in records if r.error),
        label=label,
        notes=notes,
    )


def group_by_cell(
    records: Sequence[EvalRecord],
) -> dict[tuple[str, str], list[EvalRecord]]:
    """Split records into (model, prompt_strategy) cells."""
    cells: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        cells.setdefault((record.model, record.prompt_strategy), []).append(record)
    return cells


def breakdown_by_pressure(records: Sequence[EvalRecord]) -> dict[str, dict[str, float]]:
    """Pass rate and leak rate per pressure type — where the ceiling shows up."""
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.pressure_type.value, []).append(record)

    out: dict[str, dict[str, float]] = {}
    for pressure, group in sorted(grouped.items()):
        out[pressure] = {
            "count": len(group),
            "pass_rate": round(_rate(sum(1 for r in group if r.passed), len(group)), 4),
            "solution_leak_rate": round(
                _rate(sum(1 for r in group if "SOLUTION_LEAK" in r.failure_reasons),
                      len(group)),
                4,
            ),
            "spec_adherence_mean": round(
                _mean([r.judge.spec_adherence for r in group if r.judge]), 4
            ),
        }
    return out


def breakdown_by_category(records: Sequence[EvalRecord]) -> dict[str, dict[str, float]]:
    """Pass rate per bug category — used to spot benchmark-y confounds."""
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.bug_category, []).append(record)
    return {
        category: {
            "count": len(group),
            "pass_rate": round(_rate(sum(1 for r in group if r.passed), len(group)), 4),
        }
        for category, group in sorted(grouped.items())
    }


def worst_cells(cells: Sequence[CellMetrics], n: int = 3) -> list[CellMetrics]:
    return sorted(cells, key=lambda c: (c.pass_rate, c.spec_adherence_mean))[:n]


def best_cell(cells: Sequence[CellMetrics]) -> CellMetrics:
    """The strongest (model, strategy) combination — what the gate is tested on."""
    if not cells:
        raise ValueError("best_cell() requires at least one cell")
    return max(
        cells,
        key=lambda c: (c.pass_rate, c.spec_adherence_mean, c.robustness_mean),
    )


__all__ = [
    "aggregate",
    "best_cell",
    "breakdown_by_category",
    "breakdown_by_pressure",
    "failure_mode_counts",
    "group_by_cell",
    "worst_cells",
]
