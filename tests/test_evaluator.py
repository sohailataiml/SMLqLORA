"""Evaluator combination rules, transcript completeness, and aggregation."""

from __future__ import annotations

import json

import pytest

from evaluation.evaluator import Evaluator, evaluate_many
from evaluation.judge import DeterministicJudge
from evaluation.metrics import (
    aggregate,
    best_cell,
    breakdown_by_pressure,
    failure_mode_counts,
    group_by_cell,
)
from evaluation.schemas import EvalRecord, iter_jsonl
from models.adapters import FailingAdapter, ScriptedAdapter
from prompting.strategies import get_strategy


def make_evaluator(responses, spec, strategy="structured_system_prompt", **kw):
    return Evaluator(
        ScriptedAdapter(responses),
        DeterministicJudge(spec),
        get_strategy(strategy, spec),
        spec=spec,
        max_workers=1,
        **kw,
    )


# ------------------------------------------------------------ single record


def test_compliant_response_passes(unsolved_scenario, spec, good_response):
    record = make_evaluator([good_response], spec).evaluate_scenario(unsolved_scenario)
    assert record.passed is True
    assert record.failure_reasons == ()
    assert record.error is None


def test_leak_fails_and_names_the_reason(unsolved_scenario, spec, leaking_response):
    record = make_evaluator([leaking_response], spec).evaluate_scenario(unsolved_scenario)
    assert record.passed is False
    assert "SOLUTION_LEAK" in record.failure_reasons


def test_deterministic_violation_overrides_a_lenient_judge(unsolved_scenario, spec,
                                                           leaking_response):
    """Static checks are authoritative: a permissive judge cannot rescue a leak."""

    class AlwaysPassJudge(DeterministicJudge):
        def judge(self, scenario, response, deterministic=None):
            result = super().judge(scenario, response, deterministic)
            return result.model_copy(
                update={"passed": True, "spec_adherence": 1.0, "failure_reasons": ()}
            )

    evaluator = Evaluator(
        ScriptedAdapter([leaking_response]),
        AlwaysPassJudge(spec),
        get_strategy("zero_shot", spec),
        spec=spec,
        max_workers=1,
    )
    record = evaluator.evaluate_scenario(unsolved_scenario)
    assert record.passed is False
    assert "SOLUTION_LEAK" in record.failure_reasons


def test_static_silence_cannot_rescue_a_judge_failure(unsolved_scenario, spec):
    """The reverse asymmetry: quiet static checks do not override the judge."""

    class AlwaysFailJudge(DeterministicJudge):
        def judge(self, scenario, response, deterministic=None):
            result = super().judge(scenario, response, deterministic)
            return result.model_copy(
                update={
                    "passed": False,
                    "spec_adherence": 0.1,
                    "failure_reasons": ("IRRELEVANT_HINT",),
                }
            )

    evaluator = Evaluator(
        ScriptedAdapter(["What colour is your terminal?"]),
        AlwaysFailJudge(spec),
        get_strategy("zero_shot", spec),
        spec=spec,
        max_workers=1,
    )
    record = evaluator.evaluate_scenario(unsolved_scenario)
    assert record.passed is False
    assert "IRRELEVANT_HINT" in record.failure_reasons


def test_model_error_is_recorded_as_a_failure(unsolved_scenario, spec):
    evaluator = Evaluator(
        FailingAdapter(),
        DeterministicJudge(spec),
        get_strategy("zero_shot", spec),
        spec=spec,
        max_workers=1,
    )
    record = evaluator.evaluate_scenario(unsolved_scenario)
    assert record.passed is False
    assert record.error is not None
    assert record.judge is None


def test_empty_response_fails(unsolved_scenario, spec):
    record = make_evaluator([""], spec).evaluate_scenario(unsolved_scenario)
    assert record.passed is False
    assert "EMPTY_RESPONSE" in record.failure_reasons


def test_solved_scenario_expects_confirmation(solved_scenario, spec):
    confirm = "That's exactly right — the original loop stopped one index short."
    assert make_evaluator([confirm], spec).evaluate_scenario(solved_scenario).passed


# ------------------------------------------------------------- transcripts


def test_record_carries_full_provenance(unsolved_scenario, spec, good_response):
    record = make_evaluator([good_response], spec).evaluate_scenario(unsolved_scenario)

    assert record.scenario_id == unsolved_scenario.id
    assert record.model == "mock:scripted"
    assert record.model_family == "mock"
    assert record.model_revision
    assert record.prompt_strategy == "structured_system_prompt"
    assert record.prompt_version
    assert record.behavior_spec_version == spec.version
    assert record.behavior_spec_sha256 == spec.spec_sha256
    assert record.input_messages
    assert record.model_response == good_response
    assert record.deterministic is not None
    assert record.judge is not None
    assert record.timestamp
    assert "max_tokens" in record.generation_params


def test_transcripts_are_written_as_inspectable_jsonl(tmp_path, spec, good_response,
                                                      unsolved_scenario, solved_scenario):
    path = tmp_path / "transcripts.jsonl"
    evaluator = make_evaluator([good_response, "That's exactly right."], spec)
    metrics, records = evaluator.run(
        [unsolved_scenario, solved_scenario], transcript_path=str(path)
    )

    rows = list(iter_jsonl(path))
    assert len(rows) == 2
    assert rows[0]["scenario_id"] == unsolved_scenario.id
    assert "pass" in rows[0]
    assert "judge" in rows[0] and rows[0]["judge"]["reasoning"]
    assert "deterministic" in rows[0]

    # Round-trips back into the typed model.
    assert EvalRecord.model_validate(rows[0]).scenario_id == unsolved_scenario.id
    assert metrics.scenario_count == 2


# --------------------------------------------------------------- batch runs


def test_evaluate_preserves_input_order(spec, unsolved_scenario, solved_scenario,
                                        adversarial_scenario):
    scenarios = [unsolved_scenario, solved_scenario, adversarial_scenario]
    evaluator = Evaluator(
        ScriptedAdapter(lambda msgs: f"Response to {len(msgs)} turns?"),
        DeterministicJudge(spec),
        get_strategy("zero_shot", spec),
        spec=spec,
        max_workers=3,
    )
    records = evaluator.evaluate(scenarios)
    assert [r.scenario_id for r in records] == [s.id for s in scenarios]


def test_empty_scenario_list_is_rejected(spec):
    with pytest.raises(ValueError, match="at least one scenario"):
        make_evaluator(["x"], spec).evaluate([])


def test_progress_callback_fires_once_per_scenario(spec, unsolved_scenario,
                                                   solved_scenario, good_response):
    seen = []
    evaluator = make_evaluator([good_response], spec)
    evaluator.evaluate(
        [unsolved_scenario, solved_scenario],
        on_progress=lambda done, total, rec: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_evaluate_many_produces_one_cell_per_evaluator(tmp_path, spec, unsolved_scenario,
                                                       good_response):
    evaluators = [
        make_evaluator([good_response], spec, strategy=name)
        for name in ("zero_shot", "few_shot", "structured_system_prompt")
    ]
    cells, records = evaluate_many(
        evaluators, [unsolved_scenario], transcript_dir=str(tmp_path)
    )
    assert len(cells) == 3
    assert len(records) == 3
    assert len(list(tmp_path.glob("*.jsonl"))) == 3


def test_describe_collects_every_provenance_field(spec):
    described = make_evaluator(["x"], spec).describe()
    for key in ("model", "family", "revision", "strategy", "judge_model",
                "behavior_spec_version"):
        assert key in described


# -------------------------------------------------------------- aggregation


def _records(spec, unsolved_scenario, adversarial_scenario, good_response,
             leaking_response):
    clean_pass = make_evaluator([good_response], spec).evaluate_scenario(unsolved_scenario)
    clean_fail = make_evaluator([leaking_response], spec).evaluate_scenario(unsolved_scenario)
    adv_fail = make_evaluator([leaking_response], spec).evaluate_scenario(adversarial_scenario)
    return [clean_pass, clean_fail, adv_fail]


def test_aggregate_computes_rates(spec, unsolved_scenario, adversarial_scenario,
                                  good_response, leaking_response):
    records = _records(spec, unsolved_scenario, adversarial_scenario, good_response,
                       leaking_response)
    metrics = aggregate(records)

    assert metrics.scenario_count == 3
    assert metrics.pass_rate == pytest.approx(1 / 3, abs=1e-3)
    assert metrics.failure_rate == pytest.approx(2 / 3, abs=1e-3)
    assert metrics.solution_leak_rate == pytest.approx(2 / 3, abs=1e-3)
    assert metrics.clean_pass_rate == pytest.approx(0.5)
    assert metrics.adversarial_pass_rate == pytest.approx(0.0)
    assert metrics.failure_modes["SOLUTION_LEAK"] == 2


def test_aggregate_keeps_errors_in_the_denominator(spec, unsolved_scenario):
    evaluator = Evaluator(
        FailingAdapter(), DeterministicJudge(spec), get_strategy("zero_shot", spec),
        spec=spec, max_workers=1,
    )
    metrics = aggregate([evaluator.evaluate_scenario(unsolved_scenario)])
    assert metrics.scenario_count == 1
    assert metrics.pass_rate == 0.0
    assert metrics.error_count == 1


def test_aggregate_requires_records():
    with pytest.raises(ValueError, match="at least one record"):
        aggregate([])


def test_failure_mode_counts_are_ordered(spec, unsolved_scenario, adversarial_scenario,
                                         good_response, leaking_response):
    records = _records(spec, unsolved_scenario, adversarial_scenario, good_response,
                       leaking_response)
    counts = failure_mode_counts(records)
    assert list(counts.values()) == sorted(counts.values(), reverse=True)


def test_breakdown_by_pressure_separates_the_cases(spec, unsolved_scenario,
                                                   adversarial_scenario, good_response,
                                                   leaking_response):
    records = _records(spec, unsolved_scenario, adversarial_scenario, good_response,
                       leaking_response)
    breakdown = breakdown_by_pressure(records)
    assert breakdown["normal"]["count"] == 2
    assert breakdown["prompt_injection"]["pass_rate"] == 0.0


def test_group_by_cell_splits_model_and_strategy(spec, unsolved_scenario, good_response):
    records = [
        make_evaluator([good_response], spec, strategy=name).evaluate_scenario(
            unsolved_scenario
        )
        for name in ("zero_shot", "few_shot")
    ]
    assert len(group_by_cell(records)) == 2


def test_best_cell_picks_the_strongest_combination(spec, unsolved_scenario,
                                                   good_response, leaking_response):
    strong = aggregate(
        [make_evaluator([good_response], spec).evaluate_scenario(unsolved_scenario)],
        label="strong",
    )
    weak = aggregate(
        [make_evaluator([leaking_response], spec).evaluate_scenario(unsolved_scenario)],
        label="weak",
    )
    assert best_cell([weak, strong]).label == "strong"
