"""Plots derived from evaluation records.

Complements `ablations/reporting.py`, which plots aggregated `CellMetrics`.
These take raw records so the analysis can slice by pressure type and failure
code without first collapsing to a cell.

Two rules apply to every figure here, because the point of the project is
honest measurement:

* **Rate axes are always fixed to 0-1.** Auto-scaling a y-axis to the data is
  the standard way to make a 4-point difference look like a chasm.
* **Sample sizes are printed on the figure.** A bar built on n=3 looks exactly
  like a bar built on n=48 unless the chart says otherwise.

matplotlib is optional. Every function returns `None` when it is unavailable,
so a missing plotting dependency degrades the report instead of failing the run.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from evaluation.schemas import EvalRecord

METRICS = {
    "spec_adherence": "Spec adherence",
    "robustness": "Robustness",
    "hint_relevance": "Hint relevance",
    "pass_rate": "Pass rate",
}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _ensure(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _metric_value(records: Sequence[EvalRecord], metric: str) -> float:
    if not records:
        return 0.0
    if metric == "pass_rate":
        return sum(1 for r in records if r.passed) / len(records)
    judged = [r for r in records if r.judge is not None]
    if not judged:
        return 0.0
    return sum(getattr(r.judge, metric) for r in judged) / len(judged)


def plot_metric_by_model_strategy(
    records: Sequence[EvalRecord],
    path: str | Path,
    *,
    metric: str = "pass_rate",
    title: str | None = None,
) -> Path | None:
    """Grouped bars, one group per model, one bar per prompt strategy."""
    plt = _matplotlib()
    if plt is None or not records:
        return None

    models = sorted({r.model for r in records})
    strategies = sorted({r.prompt_strategy for r in records})
    groups: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        groups.setdefault((record.model, record.prompt_strategy), []).append(record)

    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(models)), 4.8))

    for index, strategy in enumerate(strategies):
        values, labels = [], []
        for model in models:
            cell = groups.get((model, strategy), [])
            values.append(_metric_value(cell, metric))
            labels.append(len(cell))
        offsets = [i + index * width for i in range(len(models))]
        bars = ax.bar(offsets, values, width=width, label=strategy)
        for bar, n in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"n={n}" if n else "not\nmeasured",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(models))])
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(METRICS.get(metric, metric))
    # Fixed 0-1: rates are comparable only on a fixed scale.
    ax.set_ylim(0, 1.12)
    ax.set_title(title or f"{METRICS.get(metric, metric)} by model and strategy")
    ax.legend(title="strategy", fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_failure_mode_bars(
    records: Sequence[EvalRecord],
    path: str | Path,
    *,
    title: str = "Failure modes by prompt strategy",
) -> Path | None:
    """Stacked horizontal bars: which strategy produces which failure."""
    plt = _matplotlib()
    if plt is None or not records:
        return None

    strategies = sorted({r.prompt_strategy for r in records})
    counts: dict[str, Counter[str]] = {s: Counter() for s in strategies}
    for record in records:
        for code in record.failure_reasons:
            counts[record.prompt_strategy][code] += 1

    codes = sorted(
        {c for counter in counts.values() for c in counter},
        key=lambda c: -sum(counter[c] for counter in counts.values()),
    )
    if not codes:
        return None

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(codes) + 1.5)))
    positions = range(len(codes))
    left = [0.0] * len(codes)
    for strategy in strategies:
        values = [counts[strategy][c] for c in codes]
        ax.barh(list(positions), values, left=left, label=strategy)
        left = [a + b for a, b in zip(left, values)]

    ax.set_yticks(list(positions))
    ax.set_yticklabels(codes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences across measured records (counts, not rates)")
    ax.set_title(title)
    ax.legend(title="strategy", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_adversarial(
    records: Sequence[EvalRecord],
    path: str | Path,
    *,
    title: str = "Pass rate by pressure type",
) -> Path | None:
    """Where the behavior breaks: pass rate per adversarial pressure type."""
    plt = _matplotlib()
    if plt is None or not records:
        return None

    groups: dict[str, list[EvalRecord]] = {}
    for record in records:
        groups.setdefault(record.pressure_type.value, []).append(record)

    rows = sorted(
        (
            (name, sum(1 for r in g if r.passed) / len(g), len(g))
            for name, g in groups.items()
        ),
        key=lambda row: row[1],
    )
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    sizes = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(names) + 1.5)))
    bars = ax.barh(range(len(names)), values)
    for bar, n in zip(bars, sizes):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center", fontsize=7,
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Pass rate")
    ax.set_xlim(0, 1.12)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


__all__ = [
    "METRICS",
    "plot_adversarial",
    "plot_failure_mode_bars",
    "plot_metric_by_model_strategy",
]
