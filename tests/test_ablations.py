"""Ablation configuration, gate logic, error accounting and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablations.data_efficiency import SweepPoint, minimum_viable_size
from ablations.prompt_ceiling import (
    build_models,
    evaluate_gate,
    validate_experiment_shape,
)
from ablations.reporting import markdown_table, write_csv, write_json
from evaluation.metrics import aggregate
from evaluation.reproducibility import (
    ExperimentManifest,
    build_manifest,
    dependency_hash,
    git_commit,
    package_versions,
)
from evaluation.schemas import ErrorKind, classify_error, load_scenario_files
from prompting.strategies import all_strategies

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------- error classification


@pytest.mark.parametrize(
    "message,expected",
    [
        (None, ErrorKind.NONE),
        ("ModelError: Provider declined the request (stop_reason=refusal, ...)",
         ErrorKind.REFUSAL),
        ("BadRequestError: Your credit balance is too low", ErrorKind.INFRASTRUCTURE),
        ("RateLimitError: 429 insufficient_quota", ErrorKind.INFRASTRUCTURE),
        ("APIConnectionError: connection reset", ErrorKind.INFRASTRUCTURE),
        ("ValueError: something odd", ErrorKind.UNKNOWN),
    ],
)
def test_error_classification(message, expected):
    assert classify_error(message) is expected


def test_infrastructure_failures_leave_the_denominator(spec, unsolved_scenario,
                                                       good_response):
    """A billing outage must never be reported as a model that fails everything."""
    from evaluation.evaluator import Evaluator
    from evaluation.judge import DeterministicJudge
    from models.adapters import ScriptedAdapter
    from prompting.strategies import get_strategy

    evaluator = Evaluator(
        ScriptedAdapter([good_response]), DeterministicJudge(spec),
        get_strategy("zero_shot", spec), spec=spec, max_workers=1,
    )
    good = evaluator.evaluate_scenario(unsolved_scenario)
    broke = good.model_copy(
        update={
            "error": "BadRequestError: Your credit balance is too low",
            "error_kind": ErrorKind.INFRASTRUCTURE,
            "passed": False,
        }
    )
    metrics = aggregate([good, broke])

    assert metrics.scenario_count == 1        # only the measured one counts
    assert metrics.attempted_count == 2
    assert metrics.infrastructure_error_count == 1
    assert metrics.partial is True
    assert metrics.pass_rate == 1.0           # not 0.5


def test_refusals_stay_in_the_denominator(spec, unsolved_scenario, good_response):
    from evaluation.evaluator import Evaluator
    from evaluation.judge import DeterministicJudge
    from models.adapters import ScriptedAdapter
    from prompting.strategies import get_strategy

    evaluator = Evaluator(
        ScriptedAdapter([good_response]), DeterministicJudge(spec),
        get_strategy("zero_shot", spec), spec=spec, max_workers=1,
    )
    good = evaluator.evaluate_scenario(unsolved_scenario)
    refused = good.model_copy(
        update={
            "error": "ModelError: Provider declined the request (stop_reason=refusal)",
            "error_kind": ErrorKind.REFUSAL,
            "passed": False,
        }
    )
    metrics = aggregate([good, refused])
    assert metrics.scenario_count == 2
    assert metrics.pass_rate == 0.5
    assert metrics.partial is False


def test_all_infrastructure_raises_rather_than_reporting_zero(spec, unsolved_scenario,
                                                              good_response):
    from evaluation.evaluator import Evaluator
    from evaluation.judge import DeterministicJudge
    from models.adapters import ScriptedAdapter
    from prompting.strategies import get_strategy

    evaluator = Evaluator(
        ScriptedAdapter([good_response]), DeterministicJudge(spec),
        get_strategy("zero_shot", spec), spec=spec, max_workers=1,
    )
    record = evaluator.evaluate_scenario(unsolved_scenario).model_copy(
        update={"error": "insufficient_quota", "error_kind": ErrorKind.INFRASTRUCTURE}
    )
    with pytest.raises(ValueError, match="all of which failed for infrastructure"):
        aggregate([record])


# ------------------------------------------------- prompt-ceiling shape


def test_shipped_experiment_meets_the_required_shape(spec):
    scenarios = load_scenario_files(
        [REPO_ROOT / "scenarios/clean.jsonl", REPO_ROOT / "scenarios/adversarial.jsonl"]
    )
    models = build_models(["a", "b"], mock=True)
    problems = validate_experiment_shape(models, all_strategies(spec), scenarios, spec)
    assert problems == []


def test_shape_validation_catches_one_family(spec):
    from models.adapters import ScriptedAdapter

    scenarios = load_scenario_files([REPO_ROOT / "scenarios/clean.jsonl"])
    same_family = [
        ScriptedAdapter(["x"], name="a", family="same"),
        ScriptedAdapter(["x"], name="b", family="same"),
    ]
    problems = validate_experiment_shape(
        same_family, all_strategies(spec), scenarios, spec
    )
    assert any("model families" in p for p in problems)
    assert any("scenarios per cell" in p for p in problems)


def _cell(spec, **overrides):
    from evaluation.schemas import CellMetrics

    base = dict(
        model="m", model_family="fam", prompt_strategy="structured_system_prompt",
        scenario_count=36, attempted_count=36, infrastructure_error_count=0,
        partial=False, spec_adherence_mean=0.99, robustness_mean=0.99,
        hint_relevance_mean=0.9, pass_rate=0.99, failure_rate=0.01,
        solution_leak_rate=0.0, premature_confirmation_rate=0.0,
        multiple_hints_rate=0.0,
    )
    base.update(overrides)
    return CellMetrics(**base)


def test_gate_says_not_justified_when_thresholds_are_met(spec):
    cells = [_cell(spec, model_family=f) for f in ("anthropic", "openai")]
    for cell, strategy in zip(cells, ("zero_shot", "few_shot")):
        pass
    cells += [_cell(spec, model_family="openai", prompt_strategy="zero_shot"),
              _cell(spec, model_family="openai", prompt_strategy="few_shot")]
    decision = evaluate_gate(cells, [], spec, result_status="REAL_EXPERIMENT_RESULT")
    assert decision.justified is False
    assert "NOT JUSTIFIED" in decision.headline


def test_gate_says_justified_on_a_shortfall(spec):
    cells = [
        _cell(spec, model_family="anthropic", pass_rate=0.80,
              spec_adherence_mean=0.85, robustness_mean=0.80),
        _cell(spec, model_family="openai", prompt_strategy="zero_shot",
              pass_rate=0.5, spec_adherence_mean=0.5, robustness_mean=0.5),
        _cell(spec, model_family="openai", prompt_strategy="few_shot",
              pass_rate=0.6, spec_adherence_mean=0.6, robustness_mean=0.6),
    ]
    decision = evaluate_gate(cells, [], spec, result_status="REAL_EXPERIMENT_RESULT")
    assert decision.justified is True
    assert decision.shortfalls
    assert "JUSTIFIED" in decision.headline


def test_gate_is_provisional_with_one_family(spec):
    decision = evaluate_gate(
        [_cell(spec, pass_rate=0.5, spec_adherence_mean=0.5, robustness_mean=0.5)],
        [], spec, result_status="REAL_EXPERIMENT_RESULT",
    )
    assert decision.experiment_complete is False
    assert decision.status == "PARTIAL"
    assert "PROVISIONAL" in decision.headline
    assert any("model family" in c for c in decision.caveats)


def test_gate_flags_partial_cells(spec):
    cells = [
        _cell(spec, model_family="anthropic", scenario_count=25, attempted_count=36,
              infrastructure_error_count=11, partial=True, pass_rate=0.5,
              spec_adherence_mean=0.5, robustness_mean=0.5),
        _cell(spec, model_family="openai", prompt_strategy="zero_shot"),
        _cell(spec, model_family="openai", prompt_strategy="few_shot"),
    ]
    decision = evaluate_gate(
        cells, [], spec, result_status="REAL_EXPERIMENT_RESULT",
        unmeasured_cells=["openai:gpt-5 | structured_system_prompt"],
    )
    assert decision.experiment_complete is False
    assert any("is partial" in c for c in decision.caveats)
    assert any("no usable data" in c for c in decision.caveats)


# ------------------------------------------------- data-efficiency logic


def _point(size: int, **metrics) -> SweepPoint:
    point = SweepPoint(size, f"n{size}", Path(f"outputs/n{size}"))
    if metrics:
        point.evaluated = True
        point.trained = True
        point.metrics = {"dataset_size": size, **metrics}
    return point


def test_mvds_is_not_run_without_evaluation(spec):
    size, rationale = minimum_viable_size([_point(125), _point(250)], spec)
    assert size is None
    assert "NOT RUN" in rationale


def test_mvds_picks_the_smallest_clearing_size(spec):
    gate = spec.gates.data_efficiency
    points = [
        _point(125, pass_rate=0.5, spec_adherence=0.5, robustness=0.5),
        _point(250, pass_rate=gate.required_pass_rate,
               spec_adherence=gate.required_spec_adherence,
               robustness=gate.required_robustness),
        _point(500, pass_rate=0.99, spec_adherence=0.99, robustness=0.99),
    ]
    size, rationale = minimum_viable_size(points, spec)
    assert size == 250
    assert "smallest evaluated size" in rationale


def test_mvds_reports_failure_honestly(spec):
    points = [
        _point(125, pass_rate=0.3, spec_adherence=0.3, robustness=0.3),
        _point(250, pass_rate=0.6, spec_adherence=0.6, robustness=0.6),
    ]
    size, rationale = minimum_viable_size(points, spec)
    assert size is None
    assert "No evaluated size cleared" in rationale
    assert "N=250" in rationale


def test_sweep_point_status_strings():
    point = _point(125)
    assert point.status == "NOT RUN"
    point.trained = True
    assert point.status == "TRAINED (not evaluated)"
    point.evaluated = True
    assert point.status == "EVALUATED"


# ------------------------------------------------------- reproducibility


def test_manifest_records_everything_needed_to_trace_a_result(spec):
    from evaluation.judge import DeterministicJudge

    scenarios = load_scenario_files([REPO_ROOT / "scenarios/heldout.jsonl"])
    manifest = build_manifest(
        "unit-test",
        spec=spec,
        scenarios=scenarios,
        scenario_paths=["scenarios/heldout.jsonl"],
        models=build_models(["a", "b"], mock=True),
        strategies=all_strategies(spec),
        judge=DeterministicJudge(spec),
        result_status="MOCKED",
    )
    payload = manifest.to_dict()
    for key in ("behavior_spec_version", "behavior_spec_sha256", "eval_set_hash",
                "eval_set_size", "models", "prompt_versions", "judge_model",
                "judge_prompt_sha256", "git_commit", "dependency_versions",
                "dependency_lock_hash", "python_version", "experiment_timestamp",
                "result_status"):
        assert payload[key] not in (None, "", []), key
    assert payload["eval_set_size"] == len(scenarios)
    assert len(payload["prompt_versions"]) == 3


def test_manifest_writes_json(tmp_path, spec):
    path = build_manifest("t", spec=spec).write(tmp_path / "manifest.json")
    assert json.loads(path.read_text(encoding="utf-8"))["experiment"] == "t"


def test_manifest_rejects_unknown_fields(spec):
    with pytest.raises(AttributeError, match="no field"):
        build_manifest("t", spec=spec, not_a_field=1)


def test_dependency_hash_is_stable():
    versions = package_versions()
    assert dependency_hash(versions) == dependency_hash(versions)
    assert len(dependency_hash(versions)) == 64


def test_git_commit_never_crashes():
    assert isinstance(git_commit(), str)


def test_manifest_marks_status():
    assert ExperimentManifest(experiment="x").result_status == "NOT_RUN"


# ------------------------------------------------------------- reporting


def test_markdown_table_renders_and_handles_empty():
    assert markdown_table([], ["a"]) == "_No rows._"
    table = markdown_table([{"a": 1, "b": 0.5}], ["a", "b"])
    assert "| a | b |" in table
    assert "0.500" in table


def test_csv_writes_a_header(tmp_path):
    path = write_csv(tmp_path / "x.csv", [{"a": 1, "b": {"k": 2}}])
    text = path.read_text(encoding="utf-8")
    assert "a,b" in text
    assert "k=2" in text


def test_json_writer_handles_non_serializable(tmp_path):
    path = write_json(tmp_path / "x.json", {"p": Path("/tmp/x")})
    assert json.loads(path.read_text(encoding="utf-8"))["p"]
