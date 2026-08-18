"""Teacher generation: dimension sampling, prompt construction, strict parsing."""

from __future__ import annotations

import json

import pytest

from generation.generate import build_teacher, generate, mock_payload_for
from generation.prompts import (
    PRESSURE_WEIGHTS,
    build_generation_prompt,
    generation_prompt_hash,
    plan_summary,
    sample_dimensions,
    sample_plan,
)
from generation.schemas import (
    GeneratedExample,
    GenerationDimensions,
    Provenance,
    StudentProgress,
)
from generation.teacher import Teacher, TeacherError, build_example
from models.adapters import ScriptedAdapter


# ------------------------------------------------------------ dimensions


def test_sampling_is_deterministic():
    assert sample_dimensions(42) == sample_dimensions(42)
    assert sample_dimensions(42) != sample_dimensions(43)


def test_plan_is_reproducible():
    assert sample_plan(20, base_seed=7) == sample_plan(20, base_seed=7)


def test_plan_covers_the_space():
    summary = plan_summary(sample_plan(400))
    assert set(summary["language"]) == {"python", "javascript"}
    assert len(summary["bug_category"]) >= 12
    assert set(summary["pressure_type"]) == set(PRESSURE_WEIGHTS)
    assert set(summary["difficulty"]) == {"easy", "medium", "hard"}
    assert len(summary["conversation_turns"]) >= 3


def test_adversarial_pressure_is_over_weighted():
    """Adversarial cases are where the ceiling is, so data must be dense there."""
    summary = plan_summary(sample_plan(600))
    adversarial = sum(
        count for name, count in summary["pressure_type"].items() if name != "normal"
    )
    assert adversarial > summary["pressure_type"]["normal"]


def test_history_dependent_pressure_never_lands_on_a_first_turn():
    for seed in range(300):
        dims = sample_dimensions(seed)
        if dims.pressure_type in ("repeated_answer_request", "fake_success", "solved"):
            assert dims.conversation_turns >= 1, f"seed {seed}"


def test_solved_pressure_implies_solved_progress():
    for seed in range(300):
        dims = sample_dimensions(seed)
        if dims.pressure_type == "solved":
            assert dims.student_progress is StudentProgress.SOLVED


def test_fake_success_never_claims_solved_progress():
    for seed in range(300):
        dims = sample_dimensions(seed)
        if dims.pressure_type == "fake_success":
            assert dims.student_progress is not StudentProgress.SOLVED


# ---------------------------------------------------------------- prompts


def test_prompt_pins_every_dimension(spec):
    dims = sample_dimensions(11)
    prompt = build_generation_prompt(dims, spec, seed=11)
    assert f"language: {dims.language}" in prompt
    assert f"bug category: {dims.bug_category}" in prompt
    assert f"difficulty: {dims.difficulty}" in prompt
    assert f"pressure type: {dims.pressure_type}" in prompt
    assert f"prior exchanges: {dims.conversation_turns}" in prompt


def test_prompt_carries_the_behavior_spec(spec):
    prompt = build_generation_prompt(sample_dimensions(1), spec, seed=1)
    assert spec.statement.strip()[:40] in prompt


def test_solved_prompt_asks_for_confirmation(spec):
    solved = next(
        sample_dimensions(s) for s in range(300)
        if sample_dimensions(s).student_progress is StudentProgress.SOLVED
    )
    prompt = build_generation_prompt(solved, spec, seed=1)
    assert "must CONFIRM the fix" in prompt


def test_unsolved_prompt_forbids_revealing(spec):
    unsolved = next(
        sample_dimensions(s) for s in range(300)
        if sample_dimensions(s).student_progress is not StudentProgress.SOLVED
    )
    prompt = build_generation_prompt(unsolved, spec, seed=1)
    assert "exactly ONE diagnostic question" in prompt


def test_generation_prompt_hash_is_stable(spec):
    assert generation_prompt_hash(spec) == generation_prompt_hash(spec)
    assert len(generation_prompt_hash(spec)) == 64


# ----------------------------------------------------------------- parsing


def _dims(**overrides) -> GenerationDimensions:
    base = dict(
        language="python",
        bug_category="loop_boundary",
        difficulty="easy",
        pressure_type="normal",
        conversation_turns=0,
        learner_competence="beginner",
        learner_frustration="none",
        hint_strength="nudge",
        student_progress="stuck",
    )
    base.update(overrides)
    return GenerationDimensions(**base)


def _payload(**overrides):
    base = {
        "code": "def f(x):\n    return x + 1",
        "conversation_history": [],
        "student_message": "It is off by one.",
        "expected_bug": "adds one too many",
        "expected_fix": "return x",
        "tutor_response": "What does the function return for x = 0?",
    }
    base.update(overrides)
    return base


def _build(payload, dims, spec):
    return build_example(
        payload, dims, seed=1, example_id="gen_test_00001",
        teacher=ScriptedAdapter(["x"]), dataset_version="v1", spec=spec,
    )


def test_valid_payload_becomes_a_candidate(spec):
    example = _build(_payload(), _dims(), spec)
    assert example.scenario.split.value == "train"
    assert example.scenario.source.startswith("teacher:")
    assert example.tutor_response.startswith("What does")


@pytest.mark.parametrize("missing", ["code", "student_message", "expected_bug",
                                     "expected_fix", "tutor_response"])
def test_missing_keys_are_rejected(missing, spec):
    payload = _payload()
    payload[missing] = "   "
    with pytest.raises(TeacherError, match="missing or empty"):
        _build(payload, _dims(), spec)


def test_wrong_history_length_is_rejected(spec):
    with pytest.raises(TeacherError, match="expected 2 prior exchange"):
        _build(_payload(), _dims(conversation_turns=2), spec)


def test_odd_history_is_rejected(spec):
    payload = _payload(conversation_history=[{"role": "user", "content": "hi"}])
    with pytest.raises(TeacherError, match="complete user/assistant pairs"):
        _build(payload, _dims(conversation_turns=1), spec)


def test_bad_history_role_is_rejected(spec):
    payload = _payload(conversation_history=[
        {"role": "system", "content": "hi"}, {"role": "assistant", "content": "yo"},
    ])
    with pytest.raises(TeacherError, match="INVALID_SCHEMA|role="):
        _build(payload, _dims(conversation_turns=1), spec)


def test_code_fences_are_stripped(spec):
    payload = _payload(code="```python\ndef f(x):\n    return x + 1\n```")
    assert not _build(payload, _dims(), spec).scenario.code.startswith("```")


def test_malformed_payload_is_rejected_not_repaired(spec):
    """A teacher that returns rubbish must fail, not be silently patched."""
    with pytest.raises(TeacherError):
        _build(_payload(code=""), _dims(), spec)


# -------------------------------------------------------------- provenance


def test_every_candidate_carries_full_provenance(spec):
    example = _build(_payload(), _dims(), spec)
    p = example.provenance
    assert p.teacher_model
    assert p.generation_prompt_version
    assert len(p.generation_prompt_sha256) == 64
    assert p.template_id.startswith("socratic_v1:")
    assert p.generation_seed == 1
    assert p.dataset_version == "v1"
    assert p.behavior_spec_version == spec.version
    assert p.behavior_spec_sha256 == spec.spec_sha256
    assert p.timestamp


def test_provenance_rejects_empty_required_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Provenance(
            teacher_model="",
            generation_prompt_version="1",
            template_id="t",
            generation_seed=1,
            dataset_version="v1",
            behavior_spec_version="1",
        )


# ------------------------------------------------------------- teacher run


def test_teacher_retries_then_fails_with_a_code(spec):
    teacher = Teacher(ScriptedAdapter(["not json"]), spec=spec, retries=1)
    with pytest.raises(TeacherError, match="failed after 2 attempt"):
        teacher.generate_one(1, _dims(), index=0)


def test_teacher_succeeds_on_a_retry(spec):
    model = ScriptedAdapter(["garbage", json.dumps(_payload())])
    teacher = Teacher(model, spec=spec, retries=1)
    example = teacher.generate_one(1, _dims(), index=3)
    assert example.id == "gen_v1_00003"


def test_generate_records_failures_without_aborting(spec):
    teacher = Teacher(ScriptedAdapter(["nope"]), spec=spec, retries=0)
    candidates, stats, failures = generate(
        count=4, teacher=teacher, max_workers=1, verbose=False
    )
    assert candidates == []
    assert stats.requested == 4
    assert stats.returned == 0
    assert len(failures) == 4


# ------------------------------------------------------------ mock teacher


def test_mock_teacher_satisfies_the_requested_dimensions(spec):
    """`--mock` must exercise the real validator, not sidestep it."""
    teacher = build_teacher("unused", mock=True, dataset_version="vtest")
    candidates, stats, failures = generate(
        count=30, teacher=teacher, max_workers=1, verbose=False
    )
    assert not failures
    assert stats.returned == 30
    languages = {c.scenario.language.value for c in candidates}
    assert languages == {"python", "javascript"}


def test_mock_payload_matches_language_and_turns():
    prompt = "language: javascript\nprior exchanges: 2\n(generation seed 5 —)"
    payload = mock_payload_for(prompt)
    assert "function" in payload["code"]
    assert len(payload["conversation_history"]) == 4


def test_mock_payload_varies_with_the_seed():
    a = mock_payload_for("language: python\nprior exchanges: 0\n(generation seed 1 —)")
    b = mock_payload_for("language: python\nprior exchanges: 0\n(generation seed 4 —)")
    assert a["code"] != b["code"]


def test_to_training_messages_round_trips(spec):
    example = _build(_payload(), _dims(), spec)
    messages = example.to_training_messages()
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == example.tutor_response
    assert "```python" in messages[0]["content"]


def test_content_hash_distinguishes_responses(spec):
    a = _build(_payload(), _dims(), spec)
    b = _build(_payload(tutor_response="A different question entirely?"), _dims(), spec)
    assert a.content_hash() != b.content_hash()
