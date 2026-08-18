"""Shared report writers: JSON, CSV, Markdown tables and plots.

Kept dependency-light. matplotlib and pandas are optional; when they are absent
the numeric artifacts are still written and the plot is skipped with a note,
because a missing chart must never cost you the underlying evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.schemas import CellMetrics


def _ensure(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = _ensure(path)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    target = _ensure(path)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _flatten(v) for k, v in row.items()})
    return target


def _flatten(value: Any) -> Any:
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return value


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """A GitHub-flavoured Markdown table. Empty input yields an explicit note."""
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(_flatten(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


CELL_COLUMNS = (
    "model",
    "model_family",
    "prompt_strategy",
    "scenario_count",
    "spec_adherence_mean",
    "robustness_mean",
    "hint_relevance_mean",
    "pass_rate",
    "failure_rate",
    "infrastructure_error_count",
    "partial",
)


def cells_to_rows(cells: Iterable[CellMetrics]) -> list[dict[str, Any]]:
    return [cell.model_dump(mode="json") for cell in cells]


def write_markdown(path: str | Path, text: str) -> Path:
    target = _ensure(path)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


# =============================================================================
# Plots
# =============================================================================


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_prompt_ceiling(cells: Sequence[CellMetrics], path: str | Path) -> Path | None:
    """Grouped bars: pass rate per prompt strategy, one group per model."""
    plt = _matplotlib()
    if plt is None:
        return None

    models = sorted({c.model for c in cells})
    strategies = sorted({c.prompt_strategy for c in cells})
    lookup = {(c.model, c.prompt_strategy): c for c in cells}

    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(models)), 4.5))

    for index, strategy in enumerate(strategies):
        values = [
            lookup[(m, strategy)].pass_rate if (m, strategy) in lookup else 0.0
            for m in models
        ]
        offsets = [i + index * width for i in range(len(models))]
        ax.bar(offsets, values, width=width, label=strategy)

    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(models))])
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt ceiling: spec pass rate by model and prompt strategy")
    ax.legend(title="strategy", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_failure_modes(
    counts: dict[str, int], path: str | Path, *, title: str = "Failure modes"
) -> Path | None:
    plt = _matplotlib()
    if plt is None or not counts:
        return None

    labels = list(counts)[:10]
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 2))
    ax.barh(labels[::-1], values[::-1], color="#b4443c")
    ax.set_xlabel("occurrences")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_data_efficiency(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    threshold: float | None = None,
) -> Path | None:
    """Performance versus training-set size, with the reliability threshold."""
    plt = _matplotlib()
    if plt is None or not rows:
        return None

    ordered = sorted(rows, key=lambda r: r["dataset_size"])
    sizes = [r["dataset_size"] for r in ordered]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in (
        ("pass_rate", "pass rate"),
        ("spec_adherence", "spec adherence"),
        ("robustness", "robustness"),
    ):
        if key in ordered[0]:
            ax.plot(sizes, [r[key] for r in ordered], marker="o", label=label)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="grey", linewidth=1)
        ax.annotate(
            f"threshold {threshold:.2f}",
            xy=(sizes[0], threshold),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color="grey",
        )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("score on held-out set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Data efficiency: behavior reliability vs dataset size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_base_vs_tuned(
    rows: Sequence[dict[str, Any]], path: str | Path
) -> Path | None:
    """Grouped bars comparing base and tuned checkpoints across metrics."""
    plt = _matplotlib()
    if plt is None or not rows:
        return None

    metrics = ["spec_adherence", "robustness", "hint_relevance", "pass_rate"]
    labels = [r["label"] for r in rows]

    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, row in enumerate(rows):
        offsets = [i + index * width for i in range(len(metrics))]
        ax.bar(offsets, [row.get(m, 0.0) for m in metrics], width=width, label=row["label"])

    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(metrics))])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Base vs tuned on the held-out set")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    target = _ensure(path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


__all__ = [
    "CELL_COLUMNS",
    "cells_to_rows",
    "markdown_table",
    "plot_base_vs_tuned",
    "plot_data_efficiency",
    "plot_failure_modes",
    "plot_prompt_ceiling",
    "write_csv",
    "write_json",
    "write_markdown",
]
