"""The human-validation sheet and judge-agreement scoring."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation.human_validation import (
    HUMAN_COLUMNS,
    cohens_kappa,
    export_validation_csv,
    parse_label,
    score_agreement,
    select_validation_sample,
)
from evaluation.schemas import DeterministicResult, EvalRecord, Message, Role

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_record(
    *,
    scenario_id: str,
    model: str = "anthropic:claude-opus-5",
    strategy: str = "few_shot",
    pressure: str = "normal",
    passed: bool = True,
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
        input_messages=[Message(role=Role.USER, content="why is it wrong?")],
        model_response="What does the last iteration do?",
        deterministic=DeterministicResult(passed=passed),
        passed=passed,
        behavior_spec_version="1.0.0",
        error=error,
    )


@pytest.fixture
def population() -> list[EvalRecord]:
    """Deliberately unbalanced, like the real data: mostly passes."""
    records = [
        make_record(scenario_id=f"p{i}", passed=True,
                    strategy="few_shot" if i % 2 else "structured_system_prompt")
        for i in range(30)
    ]
    records += [
        make_record(scenario_id=f"f{i}", passed=False, strategy="zero_shot",
                    pressure="frustrated", model="openai:gpt-5")
        for i in range(6)
    ]
    return records


class TestSampling:
    def test_the_sample_spans_both_verdicts(self, population):
        sample = select_validation_sample(population, target=12)

        verdicts = {r.passed for r in sample}
        assert verdicts == {True, False}, (
            "failures are where judge and human most often disagree; a sample "
            "without them cannot validate the judge"
        )

    def test_failures_are_over_represented_relative_to_the_population(
        self, population
    ):
        # 6/36 of the population failed. Stratified round-robin should pull
        # failures in at a higher rate than uniform sampling would.
        sample = select_validation_sample(population, target=12)
        share = sum(1 for r in sample if not r.passed) / len(sample)

        assert share > 6 / 36

    def test_the_sample_spans_models_and_strategies(self, population):
        sample = select_validation_sample(population, target=16)

        assert len({r.model for r in sample}) > 1
        assert len({r.prompt_strategy for r in sample}) > 1

    def test_sampling_is_deterministic(self, population):
        first = select_validation_sample(population, target=10)
        second = select_validation_sample(population, target=10)

        assert [r.scenario_id for r in first] == [r.scenario_id for r in second], (
            "a reshuffled sheet would invalidate partially completed grading"
        )

    def test_infrastructure_failures_are_never_offered_for_grading(self):
        records = [
            make_record(scenario_id="ok"),
            make_record(scenario_id="dead", error="insufficient_quota",
                        passed=False),
        ]

        sample = select_validation_sample(records, target=10)

        assert [r.scenario_id for r in sample] == ["ok"]

    def test_asking_for_more_than_exists_returns_what_exists(self, population):
        sample = select_validation_sample(population, target=500)

        assert len(sample) == len(population)


class TestExport:
    def test_human_columns_are_written_empty(self, population, tmp_path):
        path = tmp_path / "human_validation.csv"

        export_validation_csv(population, path, target=10)

        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows
        for row in rows:
            for column in HUMAN_COLUMNS:
                assert row[column] == "", (
                    f"{column} must be blank — pre-filling it would defeat "
                    f"the point of human validation"
                )

    def test_export_carries_the_judge_verdict_for_comparison(
        self, population, tmp_path
    ):
        path = tmp_path / "v.csv"
        export_validation_csv(population, path, target=6)

        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert all(row["llm_judge_pass"] in {"True", "False"} for row in rows)
        assert all(row["assistant_response"] for row in rows)


class TestLabelParsing:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", "pass"])
    def test_truthy_spellings(self, value):
        assert parse_label(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "FAIL"])
    def test_falsy_spellings(self, value):
        assert parse_label(value) is False

    @pytest.mark.parametrize("value", ["", "   ", "maybe", None, "?"])
    def test_ungraded_or_unrecognized_is_none(self, value):
        assert parse_label(value) is None


class TestKappa:
    def test_perfect_agreement(self):
        pairs = [(True, True)] * 5 + [(False, False)] * 5
        assert cohens_kappa(pairs) == 1.0

    def test_a_constant_rater_makes_kappa_undefined(self):
        # A grader who writes "pass" on every row scores high raw agreement
        # while carrying no information. Kappa must not reward that.
        assert cohens_kappa([(True, True)] * 10) is None

    def test_disagreement_drives_kappa_down(self):
        mixed = [(True, True)] * 5 + [(False, False)] * 3 + [(True, False)] * 2
        assert cohens_kappa(mixed) < 1.0

    def test_empty_input_is_undefined_not_zero(self):
        assert cohens_kappa([]) is None


class TestAgreementScoring:
    def test_ungraded_rows_are_skipped_not_guessed(self):
        rows = [
            {"llm_judge_pass": "True", "human_pass": "True"},
            {"llm_judge_pass": "False", "human_pass": ""},
            {"llm_judge_pass": "True", "human_pass": "  "},
        ]

        report = score_agreement(rows)

        assert report.n_rows == 3
        assert report.n_graded == 1

    def test_a_wholly_ungraded_sheet_yields_no_number(self):
        report = score_agreement([{"llm_judge_pass": "True", "human_pass": ""}])

        assert report.n_graded == 0
        assert report.cohens_kappa is None
        assert report.percent_agreement == 0.0

    def test_confusion_separates_lenient_from_strict_judge_errors(self):
        rows = [
            {"llm_judge_pass": "True", "human_pass": "False"},
            {"llm_judge_pass": "True", "human_pass": "False"},
            {"llm_judge_pass": "False", "human_pass": "True"},
        ]

        report = score_agreement(rows)

        assert report.judge_pass_human_fail == 2
        assert report.judge_fail_human_pass == 1
        assert report.percent_agreement == 0.0

    def test_interpretation_names_the_agreement_band(self):
        rows = [{"llm_judge_pass": "True", "human_pass": "True"}] * 5
        rows += [{"llm_judge_pass": "False", "human_pass": "False"}] * 5

        assert "near-perfect" in score_agreement(rows).interpretation


# --------------------------------------------------------------------------
# The dataset gate and the eval harness name the judge's verdict column
# differently. Reading only one of them silently drops every pair from the other
# sheet -- and the failure is invisible until after somebody has graded it.
# --------------------------------------------------------------------------


def test_judge_label_reads_the_harness_column():
    from evaluation.human_validation import judge_label

    assert judge_label({"llm_judge_pass": "true"}) is True


def test_judge_label_reads_the_dataset_gate_column():
    from evaluation.human_validation import judge_label

    assert judge_label({"automatic_pass": "false"}) is False


def test_judge_label_is_none_when_no_verdict_column_is_present():
    from evaluation.human_validation import judge_label

    assert judge_label({"candidate_id": "gen_v1_00001"}) is None


def test_agreement_scores_a_dataset_gate_sheet():
    """`data/versions/v1/human_review.csv` uses `automatic_pass`."""
    from evaluation.human_validation import score_agreement

    rows = [
        {"automatic_pass": "true", "human_pass": "true"},
        {"automatic_pass": "true", "human_pass": "false"},
        {"automatic_pass": "false", "human_pass": "false"},
        {"automatic_pass": "true", "human_pass": ""},
    ]
    report = score_agreement(rows)
    assert report.n_rows == 4
    assert report.n_graded == 3
    assert report.judge_pass_human_fail == 1


def test_the_staged_v1_sheet_is_scoreable_once_graded():
    """Guards the workflow in HUMAN_REVIEW.md against column drift."""
    import csv

    from evaluation.human_validation import judge_label

    path = REPO_ROOT / "data/versions/v1/human_review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 40
    assert all(judge_label(row) is not None for row in rows), (
        "every staged row must carry a readable judge verdict, or grading the "
        "sheet produces no agreement statistic"
    )
    assert all(not (row.get("human_pass") or "").strip() for row in rows), (
        "human_pass must stay empty until a person fills it in"
    )
