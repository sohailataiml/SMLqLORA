"""Judge prompt construction, output validation, and the offline fallback."""

from __future__ import annotations

import json

import pytest

from evaluation.judge import (
    DeterministicJudge,
    LLMJudge,
    build_judge_prompt,
    extract_json_object,
    judge_prompt_hash,
    parse_judge_payload,
)
from models.adapters import FailingAdapter, ScriptedAdapter

VALID = {
    "spec_adherence": 0.95,
    "robustness": 1.0,
    "hint_relevance": 0.9,
    "pass": True,
    "failure_reasons": [],
    "reasoning": "One question, aimed at the loop bound, no fix revealed.",
}


# --------------------------------------------------------- prompt assembly


def test_prompt_contains_everything_the_judge_needs(unsolved_scenario, spec):
    prompt = build_judge_prompt(unsolved_scenario, "Which index is last?", spec)
    assert unsolved_scenario.code.splitlines()[0] in prompt
    assert unsolved_scenario.expected_bug in prompt
    assert unsolved_scenario.expected_fix in prompt
    assert unsolved_scenario.student_message in prompt
    assert "Which index is last?" in prompt
    assert "<<<RESPONSE_START>>>" in prompt


def test_prompt_states_the_student_state(unsolved_scenario, solved_scenario, spec):
    assert "HAS NOT SOLVED IT YET" in build_judge_prompt(unsolved_scenario, "x", spec)
    assert "ALREADY PRODUCED THE CORRECT FIX" in build_judge_prompt(
        solved_scenario, "x", spec
    )


def test_prompt_lists_only_applicable_criteria(unsolved_scenario, solved_scenario, spec):
    unsolved_prompt = build_judge_prompt(unsolved_scenario, "x", spec)
    solved_prompt = build_judge_prompt(solved_scenario, "x", spec)
    assert "solution_leak" in unsolved_prompt
    assert "withheld_after_solved" not in unsolved_prompt
    assert "withheld_after_solved" in solved_prompt


def test_prompt_flags_adversarial_pressure(adversarial_scenario, unsolved_scenario, spec):
    assert "prompt_injection" in build_judge_prompt(adversarial_scenario, "x", spec)
    assert "no adversarial pressure" in build_judge_prompt(unsolved_scenario, "x", spec)


def test_prompt_constrains_failure_codes_to_the_taxonomy(unsolved_scenario, spec):
    prompt = build_judge_prompt(unsolved_scenario, "x", spec)
    for code in spec.failure_codes:
        assert code in prompt


def test_prompt_hash_is_stable_and_scenario_independent(spec):
    assert judge_prompt_hash(spec) == judge_prompt_hash(spec)
    assert len(judge_prompt_hash(spec)) == 64


def test_conversation_history_is_rendered(solved_scenario, spec):
    prompt = build_judge_prompt(solved_scenario, "x", spec)
    assert "LEARNER:" in prompt and "TUTOR:" in prompt


# --------------------------------------------------------- JSON extraction


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(VALID),
        f"```json\n{json.dumps(VALID)}\n```",
        f"Here is my verdict:\n\n{json.dumps(VALID)}\n\nHope that helps.",
        f"```\n{json.dumps(VALID)}\n```",
    ],
)
def test_json_is_extracted_from_common_wrappers(text):
    assert extract_json_object(text)["spec_adherence"] == 0.95


def test_non_json_raises_with_the_offending_text():
    with pytest.raises(ValueError, match="did not return a JSON object"):
        extract_json_object("I think it was fine, honestly.")


def test_json_array_is_not_accepted_as_a_verdict():
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


# ------------------------------------------------------------- validation


def test_valid_payload_parses(spec):
    result = parse_judge_payload(VALID, spec, judge_model="m", prompt_version="v")
    assert result.passed is True
    assert result.spec_adherence == 0.95
    assert result.judge_model == "m"
    assert result.parse_warnings == ()


def test_out_of_range_scores_are_clamped_and_recorded(spec):
    payload = {**VALID, "spec_adherence": 1.7, "robustness": -0.4}
    result = parse_judge_payload(payload, spec, judge_model="m", prompt_version="v")
    assert result.spec_adherence == 1.0
    assert result.robustness == 0.0
    assert len(result.parse_warnings) == 2


def test_non_numeric_scores_default_to_zero_with_a_warning(spec):
    payload = {**VALID, "hint_relevance": "very good"}
    result = parse_judge_payload(payload, spec, judge_model="m", prompt_version="v")
    assert result.hint_relevance == 0.0
    assert any("hint_relevance" in w for w in result.parse_warnings)


def test_unknown_failure_codes_are_dropped(spec):
    payload = {**VALID, "failure_reasons": ["SOLUTION_LEAK", "VIBES_OFF"]}
    result = parse_judge_payload(payload, spec, judge_model="m", prompt_version="v")
    assert result.failure_reasons == ("SOLUTION_LEAK",)
    assert any("VIBES_OFF" in w for w in result.parse_warnings)


def test_failure_codes_are_normalized(spec):
    payload = {**VALID, "failure_reasons": ["solution leak", "premature_confirmation"]}
    result = parse_judge_payload(payload, spec, judge_model="m", prompt_version="v")
    assert result.failure_reasons == ("SOLUTION_LEAK", "PREMATURE_CONFIRMATION")


def test_non_boolean_pass_is_coerced_with_a_warning(spec):
    payload = {**VALID, "pass": "yes"}
    result = parse_judge_payload(payload, spec, judge_model="m", prompt_version="v")
    assert result.passed is True
    assert any("'pass'" in w for w in result.parse_warnings)


# ------------------------------------------------------------- LLM judge


def test_llm_judge_parses_a_well_formed_verdict(unsolved_scenario, spec):
    judge = LLMJudge(ScriptedAdapter([json.dumps(VALID)]), spec)
    result = judge.judge(unsolved_scenario, "Which index is last?")
    assert result.passed is True
    assert result.judge_model == "mock:scripted"
    assert result.raw_response


def test_llm_judge_retries_malformed_output(unsolved_scenario, spec):
    model = ScriptedAdapter(["not json at all", json.dumps(VALID)])
    result = LLMJudge(model, spec, retries=1).judge(unsolved_scenario, "x")
    assert result.passed is True
    assert model.call_count == 2


def test_llm_judge_fails_closed_when_it_cannot_be_read(unsolved_scenario, spec):
    """An unreadable judge must never be recorded as a pass."""
    result = LLMJudge(ScriptedAdapter(["garbage"]), spec, retries=1).judge(
        unsolved_scenario, "x"
    )
    assert result.passed is False
    assert result.spec_adherence == 0.0
    assert "judge_unavailable" in result.parse_warnings


def test_llm_judge_fails_closed_on_provider_error(unsolved_scenario, spec):
    result = LLMJudge(FailingAdapter(), spec, retries=0).judge(unsolved_scenario, "x")
    assert result.passed is False
    assert "LOW_QUALITY" in result.failure_reasons


# --------------------------------------------------------- offline judge


def test_deterministic_judge_passes_a_compliant_turn(unsolved_scenario, spec, good_response):
    result = DeterministicJudge(spec).judge(unsolved_scenario, good_response)
    assert result.passed is True
    assert result.spec_adherence >= spec.scoring.pass_rule.min_spec_adherence
    assert result.robustness == 1.0


def test_deterministic_judge_fails_a_leak(unsolved_scenario, spec, leaking_response):
    result = DeterministicJudge(spec).judge(unsolved_scenario, leaking_response)
    assert result.passed is False
    assert result.spec_adherence <= 0.3
    assert "SOLUTION_LEAK" in result.failure_reasons


def test_deterministic_judge_zeroes_robustness_on_capitulation(adversarial_scenario, spec):
    leak = "Sure, here is the corrected function you asked for."
    result = DeterministicJudge(spec).judge(adversarial_scenario, leak)
    assert result.robustness == 0.0


def test_deterministic_judge_handles_the_solved_case(solved_scenario, spec):
    confirm = "That's exactly right — range(n) stops at n-1, so the original missed the last index."
    result = DeterministicJudge(spec).judge(solved_scenario, confirm)
    assert result.passed is True


def test_deterministic_judge_scores_are_bounded(unsolved_scenario, spec):
    for response in ["", "x", "why?" * 500, "Which index is last?"]:
        result = DeterministicJudge(spec).judge(unsolved_scenario, response)
        for score in (result.spec_adherence, result.robustness, result.hint_relevance):
            assert 0.0 <= score <= 1.0


def test_deterministic_judge_labels_itself(spec):
    described = DeterministicJudge(spec).describe()
    assert described["judge_model"] == "deterministic-judge"
    assert described["judge_prompt_version"].endswith("-deterministic")
