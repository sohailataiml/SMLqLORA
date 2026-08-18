"""Prompt-ceiling ablation — the experiment that decides whether to fine-tune.

Design: N frontier models (>= 2 families) x 3 prompt strategies x the same >= 30
scenarios. The same scenario set is used in every cell so differences are
attributable to the model and the prompt, not to the sample.

The output is a GATE DECISION, derived from measured numbers against the
thresholds in `behavior/spec.yaml`:

    best (model, strategy) meets every threshold  -> FINE-TUNING NOT JUSTIFIED
    otherwise                                     -> FINE-TUNING JUSTIFIED

The gate is computed, never assumed. Running with `--mock` produces clearly
labelled `MOCKED` output for pipeline testing; those numbers are not evidence and
the reports say so on every page.

Usage:
    python -m ablations.prompt_ceiling --mock
    python -m ablations.prompt_ceiling \
        --models anthropic:claude-opus-5 openai:gpt-5 \
        --judge anthropic:claude-opus-5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablations.reporting import (  # noqa: E402
    CELL_COLUMNS,
    cells_to_rows,
    markdown_table,
    plot_failure_modes,
    plot_prompt_ceiling,
    write_csv,
    write_json,
    write_markdown,
)
from behavior.spec import BehaviorSpec, load_spec  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.judge import DeterministicJudge, Judge, LLMJudge  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate,
    best_cell,
    breakdown_by_category,
    breakdown_by_pressure,
    failure_mode_counts,
    group_by_cell,
)
from evaluation.reproducibility import build_manifest  # noqa: E402
from evaluation.resume import ResumeIndex  # noqa: E402
from evaluation.schemas import (  # noqa: E402
    CellMetrics,
    EvalRecord,
    Scenario,
    load_scenario_files,
    write_jsonl,
)
from models.adapters import EVAL_PARAMS, ScriptedAdapter, resolve_model  # noqa: E402
from prompting.strategies import all_strategies  # noqa: E402

DEFAULT_MODELS = ("anthropic:claude-opus-5", "openai:gpt-5")
DEFAULT_JUDGE = "anthropic:claude-opus-5"
DEFAULT_SCENARIOS = ("scenarios/clean.jsonl", "scenarios/adversarial.jsonl")
DEFAULT_OUTPUT = "results/prompt_ceiling"


# =============================================================================
# Gate
# =============================================================================


@dataclass
class GateDecision:
    """The experiment's conclusion, with the evidence that produced it."""

    justified: bool
    status: str
    best_cell: dict[str, Any]
    thresholds: dict[str, float]
    shortfalls: list[str] = field(default_factory=list)
    surviving_failure_modes: dict[str, int] = field(default_factory=dict)
    weakest_pressure_types: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""
    #: False when the experiment did not meet the spec's required shape, or when
    #: infrastructure failures left cells incomplete.
    experiment_complete: bool = True
    caveats: list[str] = field(default_factory=list)
    unmeasured_cells: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        verdict = (
            "FINE-TUNING JUSTIFIED" if self.justified else "FINE-TUNING NOT JUSTIFIED"
        )
        if not self.experiment_complete:
            return f"GATE RESULT (PROVISIONAL — INCOMPLETE EXPERIMENT): {verdict}"
        return f"GATE RESULT: {verdict}"


def evaluate_gate(
    cells: Sequence[CellMetrics],
    records: Sequence[EvalRecord],
    spec: BehaviorSpec,
    *,
    result_status: str,
    unmeasured_cells: Sequence[str] = (),
) -> GateDecision:
    """Compare the strongest cell against the configured reliability thresholds.

    A shortfall against the thresholds justifies fine-tuning. But an experiment
    that did not meet the spec's required *shape* — too few model families, or
    cells truncated by infrastructure failures — yields a PROVISIONAL verdict:
    the direction may be right, the claim is not yet supported.
    """
    gate = spec.gates.prompt_ceiling
    thresholds = {
        "required_spec_adherence": gate.required_spec_adherence,
        "required_robustness": gate.required_robustness,
        "required_pass_rate": gate.required_pass_rate,
    }

    strongest = best_cell(cells)
    shortfalls: list[str] = []
    if strongest.spec_adherence_mean < gate.required_spec_adherence:
        shortfalls.append(
            f"spec_adherence {strongest.spec_adherence_mean:.3f} < "
            f"{gate.required_spec_adherence:.2f}"
        )
    if strongest.robustness_mean < gate.required_robustness:
        shortfalls.append(
            f"robustness {strongest.robustness_mean:.3f} < {gate.required_robustness:.2f}"
        )
    if strongest.pass_rate < gate.required_pass_rate:
        shortfalls.append(
            f"pass_rate {strongest.pass_rate:.3f} < {gate.required_pass_rate:.2f}"
        )

    strongest_records = [
        r
        for r in records
        if r.was_evaluated
        and r.model == strongest.model
        and r.prompt_strategy == strongest.prompt_strategy
    ]
    surviving = failure_mode_counts(r for r in strongest_records if not r.passed)
    pressure = breakdown_by_pressure(strongest_records)
    weakest = sorted(
        (
            {"pressure_type": name, **stats}
            for name, stats in pressure.items()
        ),
        key=lambda row: row["pass_rate"],
    )[:3]

    # ---- shape and completeness -------------------------------------------
    caveats: list[str] = []
    measured_families = {c.model_family for c in cells}
    if len(measured_families) < gate.min_model_families:
        caveats.append(
            f"only {len(measured_families)} model family "
            f"({', '.join(sorted(measured_families))}) produced data; the spec "
            f"requires {gate.min_model_families}"
        )
    measured_strategies = {c.prompt_strategy for c in cells}
    if len(measured_strategies) < gate.min_strategies:
        caveats.append(
            f"only {len(measured_strategies)} prompt strategies produced data; "
            f"the spec requires {gate.min_strategies}"
        )
    for cell in cells:
        if cell.partial:
            caveats.append(
                f"cell '{cell.model} | {cell.prompt_strategy}' is partial: "
                f"{cell.scenario_count}/{cell.attempted_count} scenarios measured "
                f"({cell.infrastructure_error_count} lost to infrastructure failures)"
            )
    if cell_count_below := [
        c for c in cells if c.scenario_count < gate.min_scenarios_per_cell
    ]:
        caveats.append(
            f"{len(cell_count_below)} cell(s) measured fewer than "
            f"{gate.min_scenarios_per_cell} scenarios"
        )
    for name in unmeasured_cells:
        caveats.append(f"cell '{name}' produced no usable data at all")

    complete = not caveats
    justified = bool(shortfalls)

    evidence = (
        f"Strongest measured cell: {strongest.model} + {strongest.prompt_strategy} "
        f"over {strongest.scenario_count} scenarios"
        + (
            f" (of {strongest.attempted_count} attempted)"
            if strongest.partial
            else ""
        )
        + ". "
        + (
            "Shortfalls against the configured thresholds: "
            + "; ".join(shortfalls)
            + ". "
            if shortfalls
            else "Met every configured threshold. "
        )
        + (
            f"Failure modes surviving the strongest prompt: "
            f"{', '.join(f'{k} x{v}' for k, v in surviving.items())}."
            if surviving
            else "No failure modes survived the strongest prompt."
        )
        + (
            "\n\nTHIS EXPERIMENT IS INCOMPLETE and the verdict is provisional: "
            + "; ".join(caveats)
            + "."
            if caveats
            else ""
        )
    )

    return GateDecision(
        justified=justified,
        status=("PARTIAL" if caveats and result_status != "MOCKED" else result_status),
        best_cell=strongest.model_dump(mode="json"),
        thresholds=thresholds,
        shortfalls=shortfalls,
        surviving_failure_modes=surviving,
        weakest_pressure_types=weakest,
        evidence=evidence,
        experiment_complete=complete,
        caveats=caveats,
        unmeasured_cells=list(unmeasured_cells),
    )


def validate_experiment_shape(
    models: Sequence[Any], strategies: Sequence[Any], scenarios: Sequence[Scenario],
    spec: BehaviorSpec,
) -> list[str]:
    """Check the experiment meets the spec's minimum shape. Returns problems."""
    gate = spec.gates.prompt_ceiling
    problems: list[str] = []

    families = {m.family for m in models}
    if len(families) < gate.min_model_families:
        problems.append(
            f"needs >= {gate.min_model_families} model families, got "
            f"{len(families)} ({sorted(families)})"
        )
    if len(strategies) < gate.min_strategies:
        problems.append(
            f"needs >= {gate.min_strategies} prompt strategies, got {len(strategies)}"
        )
    if len(scenarios) < gate.min_scenarios_per_cell:
        problems.append(
            f"needs >= {gate.min_scenarios_per_cell} scenarios per cell, got "
            f"{len(scenarios)}"
        )
    return problems


# =============================================================================
# Runner
# =============================================================================

MOCK_RESPONSES = [
    "What is the last index your loop actually visits?",
    "Here's the corrected function:\n\n```python\nfor i in range(len(nums)):\n```",
    "That's exactly right, your fix works now!",
    "Try printing the value on each pass — what do you notice?",
]


def build_models(specs: Sequence[str], *, mock: bool) -> list[Any]:
    if mock:
        return [
            ScriptedAdapter(
                MOCK_RESPONSES[i:] + MOCK_RESPONSES[:i],
                name=f"mock:model-{chr(97 + i)}",
                family=f"mock-family-{chr(97 + i)}",
                revision="mock-1",
            )
            for i in range(max(2, len(specs)))
        ]
    return [resolve_model(spec) for spec in specs]


def build_judge(spec_string: str, *, mock: bool, spec: BehaviorSpec) -> Judge:
    if mock:
        return DeterministicJudge(spec)
    return LLMJudge(resolve_model(spec_string), spec)


def run_experiment(
    *,
    model_specs: Sequence[str] = DEFAULT_MODELS,
    judge_spec: str = DEFAULT_JUDGE,
    scenario_paths: Sequence[str] = DEFAULT_SCENARIOS,
    output_dir: str = DEFAULT_OUTPUT,
    mock: bool = False,
    limit: int | None = None,
    max_workers: int = 4,
    verbose: bool = True,
    resume: bool = True,
) -> GateDecision:
    spec = load_spec()
    scenarios = load_scenario_files([REPO_ROOT / p for p in scenario_paths])
    if limit:
        scenarios = scenarios[:limit]

    models = build_models(model_specs, mock=mock)
    strategies = all_strategies(spec)
    judge = build_judge(judge_spec, mock=mock, spec=spec)

    problems = validate_experiment_shape(models, strategies, scenarios, spec)
    if problems and not mock:
        raise SystemExit(
            "Experiment shape does not satisfy behavior/spec.yaml:\n  - "
            + "\n  - ".join(problems)
        )
    if problems and verbose:
        print(f"[mock] shape warnings (ignored in mock mode): {problems}")

    result_status = "MOCKED" if mock else "REAL_EXPERIMENT_RESULT"
    out = REPO_ROOT / output_dir
    transcripts_dir = out / "judge_transcripts"

    # Prior results are reused rather than re-purchased. Infrastructure failures
    # are deliberately *not* reusable, so a resume retries exactly the calls that
    # an exhausted quota cost us and nothing else.
    index = (
        ResumeIndex.from_file(out / "all_records.jsonl", allow_mock=mock)
        if resume
        else ResumeIndex.empty(allow_mock=mock)
    )
    if verbose:
        print(f"resume: {index.stats.summary()}")
        if index.stats.by_cell:
            for cell_label, count in sorted(index.stats.by_cell.items()):
                print(f"        reusable {count:3}  {cell_label}")
        print()

    cells: list[CellMetrics] = []
    all_records: list[EvalRecord] = []
    unmeasured: list[str] = []
    total_cells = len(models) * len(strategies)

    for cell_index, model in enumerate(models):
        for strategy in strategies:
            evaluator = Evaluator(
                model, judge, strategy, spec=spec, params=EVAL_PARAMS,
                max_workers=max_workers,
            )
            label = f"{model.name} | {strategy.name}"
            if verbose:
                print(
                    f"[{cell_index * len(strategies) + strategies.index(strategy) + 1}"
                    f"/{total_cells}] {label} over {len(scenarios)} scenarios..."
                )

            safe = label.replace("/", "_").replace(":", "_").replace(" | ", "__")

            # The prompt version embeds a hash of the *rendered* prompt, so this
            # is per-scenario. Rendering is local and free; only the calls cost.
            done, todo = index.partition(
                scenarios,
                model=model.name,
                strategy=strategy.name,
                prompt_version_for=lambda s: (
                    f"{strategy.version}+{strategy.render(s).sha256[:12]}"
                ),
            )
            if done and verbose:
                print(f"      reusing {len(done)} prior result(s); "
                      f"{len(todo)} call(s) to make")

            try:
                fresh = (
                    evaluator.evaluate(
                        todo,
                        on_progress=(
                            (lambda done_n, total, rec: None)
                            if not verbose
                            else _progress
                        ),
                    )
                    if todo
                    else []
                )
                records = _merge_in_scenario_order(done, fresh, scenarios)
                write_jsonl(str(transcripts_dir / f"{safe}.jsonl"), records)
                metrics = aggregate(
                    records,
                    model=model.name,
                    model_family=model.family,
                    prompt_strategy=strategy.name,
                    label=label,
                    reused_count=len(done),
                )
            except ValueError as exc:
                # Every call in this cell failed for infrastructure reasons.
                # Record it as unmeasured rather than as a model that scored zero.
                unmeasured.append(label)
                if verbose:
                    print(f"      NOT MEASURED — {exc}")
                continue

            cells.append(metrics)
            all_records.extend(records)
            if verbose:
                partial = (
                    f" PARTIAL {metrics.scenario_count}/{metrics.attempted_count}"
                    if metrics.partial
                    else ""
                )
                print(
                    f"      pass_rate={metrics.pass_rate:.3f} "
                    f"adherence={metrics.spec_adherence_mean:.3f} "
                    f"robustness={metrics.robustness_mean:.3f} "
                    f"model_errors={metrics.error_count}"
                    f"{partial}"
                )

    if not cells:
        raise SystemExit(
            "No cell produced usable data — every call failed for infrastructure "
            "reasons (check provider credit, quota and network). Nothing was "
            "measured, so no gate decision can be made."
        )

    decision = evaluate_gate(
        cells, all_records, spec, result_status=result_status,
        unmeasured_cells=unmeasured,
    )
    write_reports(
        out, cells, all_records, decision, spec, models, strategies, judge,
        scenarios, scenario_paths, result_status,
    )

    if verbose:
        print()
        print("=" * 72)
        print(decision.headline)
        if result_status == "MOCKED":
            print("STATUS: MOCKED — scripted responses. NOT EVIDENCE.")
        print(decision.evidence)
        print("=" * 72)
        print(f"reports written to {out}")

    return decision


def _progress(done: int, total: int, record: EvalRecord) -> None:
    end = "\n" if done == total else "\r"
    print(f"      {done}/{total} scenarios", end=end, flush=True)


class _JudgeRow(BaseModel):
    """One judge verdict, flattened for audit."""

    model_config = {"extra": "forbid"}

    scenario_id: str
    subject_model: str
    subject_model_family: str
    prompt_strategy: str
    prompt_version: str
    pressure_type: str
    student_has_solved: bool
    subject_response: str
    judge_model: str = "unknown"
    judge_model_family: str = "unknown"
    judge_prompt_version: str = "unknown"
    #: True when the judge and the subject come from the same family — the
    #: self-evaluation rows, which are the ones a reviewer should scrutinize.
    #: None when the judge is unidentifiable, which is not the same as False.
    self_judged: bool | None = None
    spec_adherence: float | None = None
    robustness: float | None = None
    hint_relevance: float | None = None
    judge_pass: bool | None = None
    judge_reasoning: str = ""
    failure_reasons: tuple[str, ...] = ()
    final_pass: bool = False
    error_kind: str = "none"


def _judge_family(judge) -> str:
    """Resolve the judging model's family, falling back to the spec prefix.

    `judge_model_family` was added after the first real run, so records written
    before it exists carry "unknown". The model spec (`anthropic:claude-opus-5`)
    already names the family, so derive it rather than reporting "unknown" for
    data we can in fact identify.
    """
    if judge is None:
        return "unknown"
    if judge.judge_model_family not in {"", "unknown"}:
        return judge.judge_model_family
    if ":" in judge.judge_model:
        return judge.judge_model.split(":", 1)[0]
    if judge.judge_model.startswith("deterministic"):
        return "deterministic"
    return "unknown"


def _judge_rows(records: Sequence[EvalRecord]) -> list[_JudgeRow]:
    rows = []
    for record in records:
        judge = record.judge
        family = _judge_family(judge)
        rows.append(_JudgeRow(
            scenario_id=record.scenario_id,
            subject_model=record.model,
            subject_model_family=record.model_family,
            prompt_strategy=record.prompt_strategy,
            prompt_version=record.prompt_version,
            pressure_type=record.pressure_type.value,
            student_has_solved=record.student_has_solved,
            subject_response=record.model_response,
            judge_model=judge.judge_model if judge else "unknown",
            judge_model_family=family,
            judge_prompt_version=judge.judge_prompt_version if judge else "unknown",
            self_judged=(
                None
                if family in {"unknown", "deterministic"}
                else family == record.model_family
            ),
            spec_adherence=judge.spec_adherence if judge else None,
            robustness=judge.robustness if judge else None,
            hint_relevance=judge.hint_relevance if judge else None,
            judge_pass=judge.passed if judge else None,
            judge_reasoning=judge.reasoning if judge else "",
            failure_reasons=record.failure_reasons,
            final_pass=record.passed,
            error_kind=record.error_kind.value,
        ))
    return rows


def plan_run(
    *,
    model_specs: Sequence[str] = DEFAULT_MODELS,
    scenario_paths: Sequence[str] = DEFAULT_SCENARIOS,
    output_dir: str = DEFAULT_OUTPUT,
    resume: bool = True,
) -> dict[str, Any]:
    """Report what a run would purchase, without calling any model.

    Answers the only question worth asking before an expensive run: how many
    calls do I still owe, and for which cells?
    """
    spec = load_spec()
    scenarios = load_scenario_files([REPO_ROOT / p for p in scenario_paths])
    strategies = all_strategies(spec)
    out = REPO_ROOT / output_dir
    index = (
        ResumeIndex.from_file(out / "all_records.jsonl")
        if resume
        else ResumeIndex.empty()
    )

    rows: list[dict[str, Any]] = []
    for model_spec in model_specs:
        for strategy in strategies:
            done, todo = index.partition(
                scenarios,
                model=model_spec,
                strategy=strategy.name,
                prompt_version_for=lambda s, st=strategy: (
                    f"{st.version}+{st.render(s).sha256[:12]}"
                ),
            )
            rows.append({
                "model": model_spec,
                "prompt_strategy": strategy.name,
                "scenarios": len(scenarios),
                "reusable": len(done),
                "to_purchase": len(todo),
            })

    return {
        "resume_enabled": resume,
        "prior_index": index.stats.summary(),
        "cells": rows,
        "subject_calls_to_purchase": sum(r["to_purchase"] for r in rows),
        "subject_calls_if_cold": len(scenarios) * len(strategies) * len(model_specs),
    }


def print_plan(plan: dict[str, Any]) -> None:
    print(f"resume: {plan['prior_index']}\n")
    for row in plan["cells"]:
        print(f"  {row['model']:<28} {row['prompt_strategy']:<26} "
              f"reuse={row['reusable']:3}  purchase={row['to_purchase']:3}")
    print()
    print(f"subject calls to purchase : {plan['subject_calls_to_purchase']}")
    print(f"a cold run would purchase : {plan['subject_calls_if_cold']}")
    print("(judge calls are additional, one per successful subject call)")


def _merge_in_scenario_order(
    reused: Sequence[EvalRecord],
    fresh: Sequence[EvalRecord],
    scenarios: Sequence[Scenario],
) -> list[EvalRecord]:
    """Interleave reused and newly purchased records back into scenario order.

    Order matters for reproducibility: a resumed run and a cold run over the same
    scenario set must produce byte-identical transcripts, or the two are not
    comparable artifacts.
    """
    by_id = {r.scenario_id: r for r in reused}
    by_id.update({r.scenario_id: r for r in fresh})
    return [by_id[s.id] for s in scenarios if s.id in by_id]


# =============================================================================
# Reports
# =============================================================================


def write_reports(
    out: Path,
    cells: Sequence[CellMetrics],
    records: Sequence[EvalRecord],
    decision: GateDecision,
    spec: BehaviorSpec,
    models: Sequence[Any],
    strategies: Sequence[Any],
    judge: Judge,
    scenarios: Sequence[Scenario],
    scenario_paths: Sequence[str],
    result_status: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = cells_to_rows(cells)
    # Behavioral analysis uses measured records only; infrastructure failures
    # describe the billing account, not the model.
    measured = [r for r in records if r.was_evaluated]

    write_json(
        out / "results.json",
        {
            "result_status": result_status,
            "records_total": len(records),
            "records_measured": len(measured),
            "records_lost_to_infrastructure": len(records) - len(measured),
            "gate": decision.__dict__,
            "cells": rows,
            "failure_modes_overall": failure_mode_counts(measured),
            "by_pressure_type": breakdown_by_pressure(measured),
            "by_bug_category": breakdown_by_category(measured),
        },
    )
    write_csv(out / "results.csv", rows)
    write_jsonl(out / "all_records.jsonl", records)
    # One flat judge transcript alongside the per-cell files: every judge
    # verdict with the subject response it graded, so a reviewer can audit the
    # judge without reassembling six files.
    write_jsonl(out / "judge_transcripts.jsonl", _judge_rows(records))

    plot_prompt_ceiling(cells, out / "pass_rate_by_strategy.png")
    plot_failure_modes(
        failure_mode_counts(r for r in measured if not r.passed),
        out / "failure_modes.png",
        title="Failure modes across all measured cells",
    )

    build_manifest(
        "prompt_ceiling",
        spec=spec,
        scenarios=scenarios,
        scenario_paths=scenario_paths,
        models=models,
        strategies=strategies,
        judge=judge,
        generation_params=EVAL_PARAMS.to_dict(),
        result_status=result_status,
    ).write(out / "manifest.json")

    write_markdown(
        out / "report.md",
        render_report(out, cells, records, decision, spec, scenarios, result_status),
    )


def render_report(
    out: Path,
    cells: Sequence[CellMetrics],
    records: Sequence[EvalRecord],
    decision: GateDecision,
    spec: BehaviorSpec,
    scenarios: Sequence[Scenario],
    result_status: str,
) -> str:
    if result_status == "MOCKED":
        banner = (
            "> **STATUS: MOCKED.** These numbers come from scripted responses used "
            "to exercise the pipeline. They are **not experimental evidence** and "
            "must not be cited as results.\n"
        )
    elif decision.status == "PARTIAL":
        banner = (
            "> **STATUS: PARTIAL — REAL BUT INCOMPLETE.** Every number below comes "
            "from live model calls and is real. However the experiment did not meet "
            "the shape required by `behavior/spec.yaml`, so the gate verdict is "
            "**provisional**. See *Caveats* immediately below before citing "
            "anything here.\n"
        )
    else:
        banner = "> **STATUS: REAL EXPERIMENT RESULT.** Produced by live model calls.\n"

    caveat_block = (
        [
            "## Caveats — why this experiment is incomplete",
            "",
            *[f"- {c}" for c in decision.caveats],
            "",
            "Cells reported below are computed **only from scenarios that were "
            "actually measured**. Calls lost to infrastructure failures (exhausted "
            "quota, rate limits, dropped connections) are excluded from every rate "
            "rather than counted as model failures, and are reported separately in "
            "`infrastructure_error_count`.",
            "",
        ]
        if decision.caveats
        else []
    )

    rows = cells_to_rows(cells)
    strongest = best_cell(cells)
    measured = [r for r in records if r.was_evaluated]

    lines = [
        "# Prompt-Ceiling Ablation",
        "",
        banner,
        f"Behavior spec `v{spec.version}` (`{spec.spec_sha256[:12]}`)  ",
        f"Scenarios: **{len(scenarios)}** per cell  ",
        f"Cells: **{len(cells)}** ({len({c.model for c in cells})} models x "
        f"{len({c.prompt_strategy for c in cells})} strategies)  ",
        f"Evaluations measured: **{len(measured)}** of **{len(records)}** attempted",
        "",
        *caveat_block,
        "## Question",
        "",
        "> Can a strong prompt make a frontier model hold this behavior reliably? "
        "If yes, fine-tuning is not justified.",
        "",
        f"## {decision.headline}",
        "",
        decision.evidence,
        "",
        "### Thresholds (configuration, from `behavior/spec.yaml`)",
        "",
        markdown_table(
            [{"threshold": k, "required": v} for k, v in decision.thresholds.items()],
            ["threshold", "required"],
        ),
        "",
        "### Measured, strongest cell",
        "",
        markdown_table(
            [
                {
                    "metric": "spec_adherence",
                    "value": strongest.spec_adherence_mean,
                    "required": decision.thresholds["required_spec_adherence"],
                    "met": strongest.spec_adherence_mean
                    >= decision.thresholds["required_spec_adherence"],
                },
                {
                    "metric": "robustness",
                    "value": strongest.robustness_mean,
                    "required": decision.thresholds["required_robustness"],
                    "met": strongest.robustness_mean
                    >= decision.thresholds["required_robustness"],
                },
                {
                    "metric": "pass_rate",
                    "value": strongest.pass_rate,
                    "required": decision.thresholds["required_pass_rate"],
                    "met": strongest.pass_rate
                    >= decision.thresholds["required_pass_rate"],
                },
            ],
            ["metric", "value", "required", "met"],
        ),
        "",
        "## Results by model and prompt strategy",
        "",
        markdown_table(rows, list(CELL_COLUMNS)),
        "",
        "## Robustness split (clean vs adversarial)",
        "",
        markdown_table(
            rows,
            ["model", "prompt_strategy", "clean_pass_rate", "adversarial_pass_rate",
             "solution_leak_rate", "premature_confirmation_rate"],
        ),
        "",
        "## What survives the strongest prompt?",
        "",
        (
            markdown_table(
                [{"failure_mode": k, "occurrences": v}
                 for k, v in decision.surviving_failure_modes.items()],
                ["failure_mode", "occurrences"],
            )
            if decision.surviving_failure_modes
            else "_Nothing — the strongest cell passed every scenario._"
        ),
        "",
        "### Weakest pressure types under the strongest prompt",
        "",
        markdown_table(
            decision.weakest_pressure_types,
            ["pressure_type", "count", "pass_rate", "solution_leak_rate"],
        ),
        "",
        "## Failure modes across every cell",
        "",
        markdown_table(
            [{"failure_mode": k, "occurrences": v}
             for k, v in failure_mode_counts(measured).items()],
            ["failure_mode", "occurrences"],
        ),
        "",
        "## Pass rate by pressure type (all cells)",
        "",
        markdown_table(
            [{"pressure_type": k, **v} for k, v in breakdown_by_pressure(measured).items()],
            ["pressure_type", "count", "pass_rate", "solution_leak_rate",
             "spec_adherence_mean"],
        ),
        "",
        "## Representative failures",
        "",
        _render_examples(measured),
        "",
        "## Artifacts",
        "",
        "| file | contents |",
        "| --- | --- |",
        "| `results.json` | full results, gate decision, breakdowns |",
        "| `results.csv` | one row per (model x strategy) cell |",
        "| `all_records.jsonl` | every evaluation, including judge reasoning |",
        "| `judge_transcripts/` | per-cell raw transcripts |",
        "| `manifest.json` | provenance: spec, prompts, hashes, versions |",
        "| `pass_rate_by_strategy.png` | grouped bar chart |",
        "| `failure_modes.png` | failure-mode distribution |",
    ]
    return "\n".join(lines)


def _render_examples(records: Sequence[EvalRecord], per_mode: int = 2) -> str:
    """A few failing transcripts per mode, so claims can be spot-checked."""
    by_mode: dict[str, list[EvalRecord]] = {}
    for record in records:
        if record.passed:
            continue
        for code in record.failure_reasons:
            by_mode.setdefault(code, []).append(record)

    if not by_mode:
        return "_No failures to show._"

    blocks: list[str] = []
    for mode, group in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        blocks.append(f"### {mode} ({len(group)} occurrences)\n")
        for record in group[:per_mode]:
            response = record.model_response.strip().replace("\n", "\n> ")
            reasoning = record.judge.reasoning if record.judge else "(no judge verdict)"
            blocks.append(
                f"**`{record.scenario_id}`** — {record.model} / "
                f"{record.prompt_strategy} / pressure=`{record.pressure_type.value}`\n\n"
                f"> {response[:600]}\n\n"
                f"_Judge:_ {reasoning[:400]}\n"
            )
    return "\n".join(blocks)


# =============================================================================
# CLI
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the prompt-ceiling ablation and compute the fine-tuning gate.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="model specs, >= 2 families (e.g. anthropic:claude-opus-5)")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="judge model spec")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per cell (smoke tests only)")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--mock", action="store_true",
                        help="scripted models and offline judge; output labelled MOCKED")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore prior results and re-purchase every call "
                             "(default: reuse successful prior results)")
    parser.add_argument("--preflight", action="store_true",
                        help="check every model is reachable and funded before "
                             "spending anything; abort the run if any fails")
    parser.add_argument("--plan", action="store_true",
                        help="report which calls a run would purchase, then exit "
                             "without contacting any provider")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.plan:
        print_plan(plan_run(
            model_specs=args.models,
            scenario_paths=args.scenarios,
            output_dir=args.output,
            resume=args.resume,
        ))
        return 0

    if args.preflight and not args.mock:
        from scripts.preflight import run as run_preflight

        # dict.fromkeys keeps declaration order while dropping the duplicate
        # when the judge is also a subject model.
        probes = run_preflight(list(dict.fromkeys([*args.models, args.judge])))
        if any(not p.ok for p in probes):
            print("Preflight failed — the experiment was NOT started, "
                  "and nothing was spent.")
            return 2

    decision = run_experiment(
        model_specs=args.models,
        judge_spec=args.judge,
        scenario_paths=args.scenarios,
        output_dir=args.output,
        mock=args.mock,
        limit=args.limit,
        max_workers=args.max_workers,
        verbose=not args.quiet,
        resume=args.resume,
    )
    return 0 if decision.status != "NOT_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
