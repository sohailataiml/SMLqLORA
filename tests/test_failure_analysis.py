"""Failure analysis and the failure-derived training distribution."""

from __future__ import annotations

import pytest

from evaluation.failure_analysis import (
    CAP_SHARE,
    FLOOR_SHARE,
    NORMAL_FLOOR_SHARE,
    TRAINING_DIMENSIONS,
    UNDERPOWERED_THRESHOLD,
    analyze_failure_mode,
    analyze_failure_modes,
    measured,
    pressure_ranking,
    propose_training_distribution,
)
from evaluation.schemas import DeterministicResult, EvalRecord, Message, Role


def make_record(
    *,
    scenario_id: str = "s1",
    model: str = "anthropic:claude-opus-5",
    strategy: str = "few_shot",
    pressure: str = "normal",
    passed: bool = True,
    reasons: tuple[str, ...] = (),
    error: str | None = None,
) -> EvalRecord:
    return EvalRecord(
        scenario_id=scenario_id,
        scenario_split="clean",
        pressure_type=pressure,
        language="python",
        bug_category="loop_boundary",
        difficulty="easy",
        student_has_solved=False,
        model=model,
        model_family=model.split(":")[0],
        model_revision="x",
        prompt_strategy=strategy,
        prompt_version="1.0.0+aaa",
        input_messages=[Message(role=Role.USER, content="why?")],
        model_response="What does the last iteration do?",
        deterministic=DeterministicResult(passed=passed),
        failure_reasons=reasons,
        passed=passed,
        behavior_spec_version="1.0.0",
        error=error,
    )


class TestMeasuredOnly:
    def test_infrastructure_failures_leave_the_denominator(self):
        records = [
            make_record(scenario_id="a"),
            make_record(scenario_id="b", error="insufficient_quota", passed=False),
        ]

        assert len(measured(records)) == 1

    def test_analysis_rejects_a_fully_infrastructure_failed_set(self):
        records = [make_record(error="credit balance is too low", passed=False)]

        with pytest.raises(ValueError, match="infrastructure"):
            analyze_failure_modes(records)

    def test_an_outage_cannot_inflate_a_failure_rate(self):
        # Two real leaks out of two measured is 1.0. Adding eight calls that
        # never reached the model must not turn that into 0.2.
        real = [
            make_record(scenario_id=f"r{i}", passed=False,
                        reasons=("SOLUTION_LEAK",))
            for i in range(2)
        ]
        outage = [
            make_record(scenario_id=f"o{i}", error="insufficient_quota",
                        passed=False)
            for i in range(8)
        ]

        stat = analyze_failure_mode(measured(real + outage), "SOLUTION_LEAK")

        assert stat.n_measured == 2
        assert stat.rate == 1.0


class TestBreakdowns:
    def test_a_mode_is_attributed_to_the_right_strategy(self):
        records = [
            make_record(scenario_id="a", strategy="zero_shot", passed=False,
                        reasons=("SOLUTION_LEAK",)),
            make_record(scenario_id="b", strategy="few_shot"),
        ]

        stat = analyze_failure_mode(records, "SOLUTION_LEAK")
        by_strategy = {g.key: g for g in stat.by_strategy}

        assert by_strategy["zero_shot"].count == 1
        assert by_strategy["few_shot"].count == 0

    def test_small_slices_are_flagged_underpowered(self):
        records = [make_record(scenario_id=f"s{i}") for i in range(3)]

        stat = analyze_failure_mode(records, "SOLUTION_LEAK")

        assert all(g.underpowered for g in stat.by_model)

    def test_large_slices_are_not_flagged(self):
        records = [
            make_record(scenario_id=f"s{i}")
            for i in range(UNDERPOWERED_THRESHOLD + 1)
        ]

        stat = analyze_failure_mode(records, "SOLUTION_LEAK")

        assert not stat.by_model[0].underpowered

    def test_tracked_codes_appear_even_at_zero(self):
        result = analyze_failure_modes([make_record()])
        codes = {m["failure_code"] for m in result["failure_modes"]}

        assert "IRRELEVANT_HINT" in codes, (
            "a mode that never fired must be visibly absent, not missing"
        )

    def test_representative_scenarios_are_stable_across_runs(self):
        records = [
            make_record(scenario_id=f"s{i}", passed=False,
                        reasons=("SOLUTION_LEAK",))
            for i in range(9, -1, -1)
        ]

        first = analyze_failure_mode(records, "SOLUTION_LEAK")
        second = analyze_failure_mode(list(reversed(records)), "SOLUTION_LEAK")

        assert first.representative_scenarios == second.representative_scenarios

    def test_pressure_ranking_is_worst_first(self):
        records = [
            make_record(scenario_id="a", pressure="normal", passed=True),
            make_record(scenario_id="b", pressure="frustrated", passed=False),
        ]

        ranking = pressure_ranking(records)

        assert ranking[0]["pressure_type"] == "frustrated"
        assert ranking[0]["pass_rate"] == 0.0


class TestResidual:
    def test_zero_shot_is_excluded_from_the_residual(self):
        # Zero-shot measures the absence of prompting. Including it would let
        # easily-prompted-away failures dominate the dataset design.
        records = [
            make_record(scenario_id=f"z{i}", strategy="zero_shot", passed=False,
                        reasons=("MULTIPLE_HINTS",))
            for i in range(10)
        ] + [make_record(scenario_id="f1", strategy="few_shot", passed=True)]

        residual = analyze_failure_modes(records)["residual_under_strong_prompts"]

        assert residual["n_measured"] == 1
        assert residual["surviving_failure_modes"] == {}


class TestTrainingDistribution:
    @pytest.fixture
    def skewed(self) -> list[EvalRecord]:
        """almost_correct fails every time; everything else passes."""
        records = []
        for i, dim in enumerate(TRAINING_DIMENSIONS):
            for j in range(4):
                fails = dim == "almost_correct"
                records.append(make_record(
                    scenario_id=f"{dim}{j}",
                    strategy="few_shot",
                    pressure=dim,
                    passed=not fails,
                    reasons=("SOLUTION_LEAK",) if fails else (),
                ))
        return records

    def test_distribution_sums_to_one(self, skewed):
        result = propose_training_distribution(skewed)

        assert result["distribution_sums_to"] == pytest.approx(1.0, abs=1e-6)

    def test_every_dimension_is_represented(self, skewed):
        result = propose_training_distribution(skewed)

        assert set(result["distribution"]) == set(TRAINING_DIMENSIONS)
        assert all(v > 0 for v in result["distribution"].values())

    def test_the_failing_dimension_gets_the_largest_share(self, skewed):
        result = propose_training_distribution(skewed)
        ranked = list(result["distribution"])

        assert ranked[0] == "almost_correct"

    def test_no_dimension_exceeds_the_cap(self, skewed):
        result = propose_training_distribution(skewed)

        assert max(result["distribution"].values()) <= CAP_SHARE + 1e-9, (
            "a dataset dominated by one pressure teaches that pressure, "
            "not the behavior"
        )

    def test_a_dimension_that_never_failed_still_keeps_its_floor(self, skewed):
        result = propose_training_distribution(skewed)

        assert result["distribution"]["time_pressure"] >= FLOOR_SHARE - 1e-9

    def test_normal_keeps_a_higher_floor_than_adversarial_dimensions(self, skewed):
        # The measured failure rates come from frontier models; the student is
        # a 1.7B model that will fail on far easier inputs, so the base case
        # cannot be allocated purely by frontier difficulty.
        result = propose_training_distribution(skewed)

        assert result["distribution"]["normal"] >= NORMAL_FLOOR_SHARE - 1e-9
        assert result["distribution"]["normal"] > FLOOR_SHARE

    def test_an_all_passing_set_falls_back_to_an_even_split(self):
        records = [
            make_record(scenario_id=f"{dim}", strategy="few_shot", pressure=dim)
            for dim in TRAINING_DIMENSIONS
        ]

        result = propose_training_distribution(records)
        shares = set(result["distribution"].values())

        assert len(shares) == 1, (
            "with no measured failures there is no basis for a preference"
        )

    def test_underpowered_dimensions_are_flagged_in_the_output(self, skewed):
        result = propose_training_distribution(skewed)

        assert result["measured_inputs"]["almost_correct"]["underpowered"] is True
