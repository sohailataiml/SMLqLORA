"""Resumability: reuse what was paid for, retry what infrastructure ate."""

from __future__ import annotations

import json

import pytest

from evaluation.resume import (
    ResumeIndex,
    is_mock_record,
    read_records,
    record_key,
)
from evaluation.schemas import (
    DeterministicResult,
    ErrorKind,
    EvalRecord,
    Message,
    Role,
    Scenario,
)


@pytest.fixture
def scenarios() -> list[Scenario]:
    """A handful of distinct scenarios; content is irrelevant to resume logic."""
    return [
        Scenario(
            id=f"fixture_resume_{i}",
            language="python",
            bug_category="loop_boundary",
            difficulty="easy",
            code=f"def f{i}(xs):\n    return xs[len(xs)]",
            student_message="It crashes and I do not see why.",
            expected_bug="Index equals the length, which is out of range.",
            expected_fix=f"def f{i}(xs):\n    return xs[len(xs) - 1]",
            split="clean",
        )
        for i in range(5)
    ]


def make_record(
    *,
    scenario_id: str = "s1",
    model: str = "anthropic:claude-opus-5",
    strategy: str = "few_shot",
    prompt_version: str = "1.0.0+abc123",
    error: str | None = None,
    passed: bool = True,
) -> EvalRecord:
    return EvalRecord(
        scenario_id=scenario_id,
        scenario_split="clean",
        pressure_type="normal",
        language="python",
        bug_category="loop_boundary",
        difficulty="easy",
        student_has_solved=False,
        model=model,
        model_family=model.split(":")[0],
        model_revision=model.split(":")[-1],
        prompt_strategy=strategy,
        prompt_version=prompt_version,
        input_messages=[Message(role=Role.USER, content="Why is my sum wrong?")],
        model_response="What does the loop print on the last pass?",
        deterministic=DeterministicResult(passed=passed),
        passed=passed,
        behavior_spec_version="1.0.0",
        error=error,
    )


class TestReusability:
    def test_a_successful_record_is_reused(self):
        index = ResumeIndex([make_record()])
        assert index.has(record_key(make_record()))
        assert len(index) == 1

    def test_an_infrastructure_failure_is_never_reused(self):
        record = make_record(error="RateLimitError: insufficient_quota", passed=False)
        assert record.error_kind is ErrorKind.INFRASTRUCTURE

        index = ResumeIndex([record])

        assert len(index) == 0, "an exhausted quota must be retried, not reused"
        assert index.stats.retry_infrastructure == 1

    def test_a_refusal_is_reused_because_it_is_real_behavior(self):
        # Retrying refusals would resample until the model behaved — that is
        # p-hacking, and it is exactly what this must not do.
        record = make_record(error="stop_reason=refusal", passed=False)
        assert record.error_kind is ErrorKind.REFUSAL

        index = ResumeIndex([record])

        assert len(index) == 1
        assert index.stats.retry_infrastructure == 0

    def test_a_later_success_supersedes_an_earlier_infrastructure_failure(self):
        failed = make_record(error="credit balance is too low", passed=False)
        succeeded = make_record()

        index = ResumeIndex([failed, succeeded])

        assert len(index) == 1
        assert index.stats.retry_infrastructure == 0


class TestKeyIdentity:
    def test_changing_the_prompt_version_invalidates_the_result(self):
        index = ResumeIndex([make_record(prompt_version="1.0.0+aaaaaa")])

        stale = record_key(make_record(prompt_version="1.0.0+bbbbbb"))

        assert not index.has(stale), (
            "a result recorded under a different prompt must not be reused"
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("model", "openai:gpt-5"),
            ("strategy", "zero_shot"),
            ("scenario_id", "s2"),
        ],
    )
    def test_every_key_component_distinguishes_results(self, field, value):
        index = ResumeIndex([make_record()])
        assert not index.has(record_key(make_record(**{field: value})))


class TestMockIsolation:
    def test_mock_records_do_not_satisfy_a_real_run(self):
        index = ResumeIndex([make_record(model="mock:model-a")], allow_mock=False)

        assert len(index) == 0
        assert index.stats.skipped_mock == 1

    def test_real_records_do_not_satisfy_a_mock_run(self):
        index = ResumeIndex([make_record()], allow_mock=True)

        assert len(index) == 0
        assert index.stats.skipped_mock == 1

    def test_is_mock_record_detects_the_scripted_adapters(self):
        assert is_mock_record(make_record(model="mock:model-a"))
        assert not is_mock_record(make_record(model="anthropic:claude-opus-5"))


class TestFileLoading:
    def test_missing_file_is_a_cold_run_not_an_error(self, tmp_path):
        index = ResumeIndex.from_file(tmp_path / "nope.jsonl")
        assert len(index) == 0
        assert "cold run" in index.stats.summary()

    def test_a_truncated_final_line_does_not_lose_the_valid_records(self, tmp_path):
        path = tmp_path / "all_records.jsonl"
        good = make_record().model_dump(mode="json", by_alias=True)
        path.write_text(
            json.dumps(good) + "\n" + '{"scenario_id": "s2", "mod',
            encoding="utf-8",
        )

        index = ResumeIndex.from_file(path)

        assert len(index) == 1
        assert index.stats.skipped_malformed == 1

    def test_read_records_keeps_infrastructure_failures(self, tmp_path):
        # The resume index drops them; completeness reporting must not, or a
        # cell wiped out by an outage vanishes from the accounting entirely.
        path = tmp_path / "all_records.jsonl"
        failed = make_record(error="insufficient_quota", passed=False)
        path.write_text(
            json.dumps(failed.model_dump(mode="json", by_alias=True)) + "\n",
            encoding="utf-8",
        )

        records, seen, malformed = read_records(path)

        assert seen == 1 and malformed == 0
        assert len(records) == 1
        assert records[0].error_kind is ErrorKind.INFRASTRUCTURE
        assert len(ResumeIndex.from_file(path)) == 0


class TestPartition:
    def test_only_the_missing_scenarios_are_owed(self, scenarios):
        subset = scenarios[:4]
        index = ResumeIndex([
            make_record(scenario_id=subset[0].id, prompt_version="v+1"),
            make_record(scenario_id=subset[1].id, prompt_version="v+1"),
        ])

        done, todo = index.partition(
            subset,
            model="anthropic:claude-opus-5",
            strategy="few_shot",
            prompt_version_for=lambda s: "v+1",
        )

        assert [r.scenario_id for r in done] == [subset[0].id, subset[1].id]
        assert [s.id for s in todo] == [subset[2].id, subset[3].id]

    def test_a_cold_index_owes_everything(self, scenarios):
        done, todo = ResumeIndex.empty().partition(
            scenarios[:3],
            model="m",
            strategy="s",
            prompt_version_for=lambda s: "v",
        )
        assert done == []
        assert len(todo) == 3
