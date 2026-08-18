"""Tests for generation resume, top-up sizing and usage metering.

These cover the three capabilities added for the Dataset V1 build. The resume
tests matter most: they are the difference between losing a paid 1200-candidate
run to a dropped connection and losing thirty seconds of it.
"""

from __future__ import annotations

import json

import pytest

from generation.generate import build_teacher, generate
from generation.resume import (
    CandidateWriter,
    check_run_config,
    completed_indices,
    index_of,
    pending_indices,
    read_candidates,
    write_run_config,
)
from generation.topup import plan_topup
from models.adapters import GenerationParams, Message, ModelAdapter, ModelResponse, Role
from models.usage import MeteredAdapter, UsageMeter, merged_totals


def _teacher(version: str = "vtest"):
    return build_teacher("mock:teacher", mock=True, dataset_version=version)


# =============================================================================
# Candidate identity
# =============================================================================


def test_index_of_recovers_plan_index():
    assert index_of("gen_v1_00042") == 42
    assert index_of("gen_vdev_00000") == 0


def test_index_of_returns_none_for_unrecognizable_ids():
    assert index_of("no-trailing-number") is None
    assert index_of("") is None


def test_candidate_ids_are_stable_across_runs(tmp_path):
    """The same plan index must always yield the same id and dimensions."""
    path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first, _, _ = generate(count=6, teacher=_teacher(), verbose=False,
                           candidates_path=path_a)
    second, _, _ = generate(count=6, teacher=_teacher(), verbose=False,
                            candidates_path=path_b)

    assert [e.id for e in first] == [e.id for e in second]
    assert [e.dimensions for e in first] == [e.dimensions for e in second]


# =============================================================================
# Resume
# =============================================================================


def test_resume_skips_completed_candidates(tmp_path):
    path = tmp_path / "candidates.jsonl"
    generate(count=8, teacher=_teacher(), verbose=False, candidates_path=path)

    # Simulate an interruption after 3 candidates.
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:3]), encoding="utf-8")

    _, stats, _ = generate(count=8, teacher=_teacher(), verbose=False,
                           candidates_path=path)

    assert stats.reused == 3
    assert stats.generated_this_run == 5
    assert stats.returned == 8


def test_resume_produces_no_duplicate_logical_records(tmp_path):
    path = tmp_path / "candidates.jsonl"
    generate(count=10, teacher=_teacher(), verbose=False, candidates_path=path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:4]), encoding="utf-8")

    candidates, _, _ = generate(count=10, teacher=_teacher(), verbose=False,
                                candidates_path=path)

    ids = [e.id for e in candidates]
    assert len(ids) == len(set(ids)) == 10


def test_no_resume_repurchases_everything(tmp_path):
    path = tmp_path / "candidates.jsonl"
    generate(count=6, teacher=_teacher(), verbose=False, candidates_path=path)

    _, stats, _ = generate(count=6, teacher=_teacher(), verbose=False,
                           candidates_path=path, resume=False)

    assert stats.reused == 0
    assert stats.generated_this_run == 6


def test_interrupted_run_leaves_a_readable_file(tmp_path):
    """A truncated final line is expected, not corruption of the whole file."""
    path = tmp_path / "candidates.jsonl"
    generate(count=5, teacher=_teacher(), verbose=False, candidates_path=path)

    text = path.read_text(encoding="utf-8")
    path.write_text(text[: int(len(text) * 0.8)], encoding="utf-8")

    candidates, malformed = read_candidates(path)
    assert candidates, "complete lines before the truncation must survive"
    assert malformed <= 1


def test_read_candidates_ignores_duplicate_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    generate(count=3, teacher=_teacher(), verbose=False, candidates_path=path)
    original = path.read_text(encoding="utf-8")
    path.write_text(original + original, encoding="utf-8")  # every row twice

    candidates, _ = read_candidates(path)
    assert len(candidates) == 3


def test_shorter_count_does_not_inherit_out_of_plan_candidates(tmp_path):
    path = tmp_path / "candidates.jsonl"
    generate(count=10, teacher=_teacher(), verbose=False, candidates_path=path)

    candidates, stats, _ = generate(count=4, teacher=_teacher(), verbose=False,
                                    candidates_path=path)

    assert stats.returned == 4
    assert all(index_of(e.id) < 4 for e in candidates)


def test_completed_and_pending_indices_partition_the_plan():
    class _Fake:
        def __init__(self, id_):
            self.id = id_

    done = completed_indices([_Fake("gen_v1_00000"), _Fake("gen_v1_00002")])
    assert done == {0, 2}
    assert pending_indices(5, done) == [1, 3, 4]


def test_writer_flushes_each_record(tmp_path):
    path = tmp_path / "out.jsonl"
    examples, _, _ = generate(count=2, teacher=_teacher(), verbose=False)
    with CandidateWriter(path) as writer:
        writer.write(examples[0])
        # Readable before the context manager closes.
        assert path.read_text(encoding="utf-8").count("\n") == 1
        writer.write(examples[1])
    assert path.read_text(encoding="utf-8").count("\n") == 2


# =============================================================================
# Run-config guard
# =============================================================================


def test_changed_seed_blocks_resume(tmp_path):
    path = tmp_path / "candidates.jsonl"
    write_run_config(path, {"base_seed": 1, "dataset_version": "v1",
                            "generation_prompt_sha256": "abc"})

    problems = check_run_config(path, {"base_seed": 2, "dataset_version": "v1",
                                       "generation_prompt_sha256": "abc"})
    assert problems and "base_seed" in problems[0]


def test_changed_prompt_hash_blocks_resume(tmp_path):
    path = tmp_path / "candidates.jsonl"
    write_run_config(path, {"base_seed": 1, "generation_prompt_sha256": "abc"})

    problems = check_run_config(path, {"base_seed": 1,
                                       "generation_prompt_sha256": "xyz"})
    assert problems and "generation_prompt_sha256" in problems[0]


def test_matching_config_permits_resume(tmp_path):
    path = tmp_path / "candidates.jsonl"
    config = {"base_seed": 1, "dataset_version": "v1",
              "generation_prompt_sha256": "abc"}
    write_run_config(path, config)
    assert check_run_config(path, config) == []


def test_absent_config_permits_resume(tmp_path):
    assert check_run_config(tmp_path / "nothing.jsonl", {"base_seed": 1}) == []


# =============================================================================
# Top-up sizing
# =============================================================================


def test_topup_not_needed_when_target_met():
    result = plan_topup(target=600, accepted=610, observed_rate=0.5)
    assert result.needed is False
    assert result.additional_candidates == 0


def test_topup_sizes_from_observed_rate():
    result = plan_topup(target=600, accepted=400, observed_rate=0.5,
                        margin=0.0, round_to=1)
    assert result.shortfall == 200
    assert result.additional_candidates == 400


def test_topup_applies_margin_and_rounding():
    result = plan_topup(target=600, accepted=500, observed_rate=0.5,
                        margin=0.10, round_to=50)
    # 100 short / 0.5 = 200, +10% = 220, rounded up to 250.
    assert result.raw_estimate == 200
    assert result.additional_candidates == 250


def test_topup_respects_minimum_tranche():
    result = plan_topup(target=600, accepted=599, observed_rate=0.9,
                        min_tranche=50)
    assert result.additional_candidates == 50


def test_topup_refuses_a_zero_acceptance_rate():
    """Buying more candidates at a 0% rate buys more rejections."""
    with pytest.raises(ValueError, match="every candidate was rejected"):
        plan_topup(target=600, accepted=0, observed_rate=0.0)


def test_topup_rejects_impossible_rate():
    with pytest.raises(ValueError):
        plan_topup(target=600, accepted=0, observed_rate=1.5)


def test_lower_observed_rate_requires_more_candidates():
    generous = plan_topup(target=600, accepted=300, observed_rate=0.6)
    stingy = plan_topup(target=600, accepted=300, observed_rate=0.2)
    assert stingy.additional_candidates > generous.additional_candidates


# =============================================================================
# Usage metering
# =============================================================================


class _UsageAdapter(ModelAdapter):
    def __init__(self, usage=None, fail=False):
        super().__init__("test:model", "test", "rev-1")
        self._usage = usage
        self._fail = fail

    def _generate(self, messages, system, params) -> ModelResponse:
        if self._fail:
            raise RuntimeError("simulated outage")
        return ModelResponse(text="ok", model=self.name, revision=self.revision,
                             usage=self._usage or {})


def _call(adapter):
    return adapter.generate([Message(role=Role.USER, content="hi")],
                            params=GenerationParams(max_tokens=8))


def test_meter_accumulates_tokens():
    meter = UsageMeter()
    adapter = MeteredAdapter(
        _UsageAdapter({"input_tokens": 100, "output_tokens": 20}), meter
    )
    _call(adapter)
    _call(adapter)

    totals = meter.totals()["totals"]
    assert totals["input_tokens"] == 200
    assert totals["output_tokens"] == 40
    assert totals["requests"] == 2
    assert totals["total_tokens"] == 240


def test_meter_normalizes_openai_key_names():
    meter = UsageMeter()
    adapter = MeteredAdapter(
        _UsageAdapter({"prompt_tokens": 7, "completion_tokens": 3}), meter
    )
    _call(adapter)
    assert meter.totals()["totals"]["input_tokens"] == 7
    assert meter.totals()["totals"]["output_tokens"] == 3


def test_meter_flags_incomplete_when_usage_is_absent():
    meter = UsageMeter()
    _call(MeteredAdapter(_UsageAdapter({}), meter))
    assert meter.totals()["totals"]["token_counts_incomplete"] is True


def test_meter_records_failed_requests_without_flagging_incomplete():
    meter = UsageMeter()
    _call(MeteredAdapter(_UsageAdapter(fail=True), meter))

    totals = meter.totals()["totals"]
    assert totals["requests"] == 1
    assert totals["failed_requests"] == 1
    # A failed call has no usage block by definition; that is not "incomplete".
    assert totals["token_counts_incomplete"] is False


def test_metering_does_not_change_the_response():
    meter = UsageMeter()
    bare = _UsageAdapter({"input_tokens": 5, "output_tokens": 1})
    wrapped = MeteredAdapter(_UsageAdapter({"input_tokens": 5, "output_tokens": 1}),
                             meter)
    assert _call(bare).text == _call(wrapped).text


def test_meter_never_reports_an_invented_cost():
    meter = UsageMeter()
    _call(MeteredAdapter(_UsageAdapter({"input_tokens": 1}), meter))
    assert meter.totals()["estimated_cost_usd"] == (
        "NOT COMPUTED — no price table is configured"
    )


def test_merged_totals_sums_across_meters():
    teacher, judge = UsageMeter(), UsageMeter()
    _call(MeteredAdapter(_UsageAdapter({"input_tokens": 10}), teacher))
    _call(MeteredAdapter(_UsageAdapter({"input_tokens": 5}), judge))

    assert merged_totals([teacher, judge])["totals"]["input_tokens"] == 15


def test_generation_records_usage_per_candidate(tmp_path):
    """Metering is wired through the real generation path, not just in theory."""
    meter = UsageMeter()
    teacher = build_teacher("mock:teacher", mock=True, dataset_version="vtest",
                            meter=meter)
    generate(count=4, teacher=teacher, verbose=False,
             candidates_path=tmp_path / "c.jsonl")

    assert meter.totals()["totals"]["requests"] == 4


# =============================================================================
# Infrastructure errors are not rejections
# =============================================================================


def test_infrastructure_failures_are_not_written_as_candidates(tmp_path):
    """A provider outage must leave no row, so a resume retries exactly it."""

    class _Broken(ModelAdapter):
        def __init__(self):
            super().__init__("broken:teacher", "mock", "rev")

        def _generate(self, messages, system, params) -> ModelResponse:
            return ModelResponse(text="", model=self.name, error="503 upstream")

    from generation.teacher import Teacher

    path = tmp_path / "candidates.jsonl"
    broken = Teacher(_Broken(), dataset_version="vtest")
    candidates, stats, failures = generate(count=3, teacher=broken, verbose=False,
                                           candidates_path=path)

    assert candidates == []
    assert stats.provider_errors == 3
    assert failures
    # Nothing persisted, so the next run retries all three.
    assert not path.exists() or path.read_text(encoding="utf-8").strip() == ""
