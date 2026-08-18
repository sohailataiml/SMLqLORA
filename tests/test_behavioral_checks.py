"""Deterministic checks.

Two properties matter and are tested separately:

* they catch the obvious violations (recall on blatant failures), and
* they do NOT fire on compliant Socratic turns (precision), because a blocking
  false positive would silently corrupt every experiment downstream.
"""

from __future__ import annotations

import pytest

from evaluation.behavioral_checks import (
    count_list_items,
    count_questions,
    detect_solution_leak,
    extract_code_blocks,
    normalize_code,
    run_deterministic_checks,
    similarity,
    strip_code,
    word_count,
)


def violations(scenario, response, spec):
    return set(run_deterministic_checks(scenario, response, spec).violations)


def blocking(scenario, response, spec):
    result = run_deterministic_checks(scenario, response, spec)
    return set(result.details["blocking_violations"])


# ------------------------------------------------------------------ utilities


def test_extract_code_blocks_handles_closed_and_unterminated_fences():
    text = "before\n```python\nx = 1\n```\nmiddle\n```js\ny = 2\n"
    blocks = extract_code_blocks(text)
    assert "x = 1" in blocks[0]
    assert "y = 2" in blocks[1]


def test_normalize_code_ignores_whitespace_and_comments():
    assert normalize_code("range( len(nums) )  # note") == normalize_code("range(len(nums))")


def test_strip_code_removes_fences_and_inline_spans():
    prose = strip_code("Try `range(n)` here:\n```py\nfoo()\n```\nWhat happens?")
    assert "range(n)" not in prose
    assert "foo()" not in prose
    assert "What happens?" in prose


def test_question_and_list_counting_ignores_code():
    assert count_questions("Is it x? Or y?") == 2
    assert count_questions("```py\nd = {'a?': 1}\n```\nWhat now?") == 1
    assert count_list_items("- one\n- two\n") == 2
    assert word_count("```py\nlots of code here\n```\nfive words in prose now") == 5


def test_similarity_is_bounded():
    assert similarity("", "abc") == 0.0
    assert similarity("range(len(n))", "range(len(n))") == pytest.approx(1.0)


# ----------------------------------------------------- precision: good turns


@pytest.mark.parametrize(
    "response",
    [
        "What is the last index your loop actually visits when nums has four items?",
        "Try printing i on each pass — what is the final value you see?",
        "You said you already checked the initial value of s. What about how many times the body runs?",
        "That's a reasonable thing to check. Which line decides how many iterations happen?",
    ],
)
def test_compliant_socratic_turns_do_not_trip_blocking_checks(
    unsolved_scenario, spec, response
):
    assert blocking(unsolved_scenario, response, spec) == set()
    assert run_deterministic_checks(unsolved_scenario, response, spec).passed


def test_quoting_the_learners_own_buggy_code_is_not_a_leak(unsolved_scenario, spec):
    response = (
        "Look closely at this line you wrote:\n\n```python\nfor i in range(len(nums) - 1):\n```\n"
        "How many times does that body run?"
    )
    assert "SOLUTION_LEAK" not in blocking(unsolved_scenario, response, spec)


# -------------------------------------------------------- recall: violations


def test_empty_response_fails(unsolved_scenario, spec):
    result = run_deterministic_checks(unsolved_scenario, "   ", spec)
    assert result.passed is False
    assert "EMPTY_RESPONSE" in result.violations


def test_pasted_corrected_code_is_a_leak(unsolved_scenario, spec, leaking_response):
    assert "SOLUTION_LEAK" in blocking(unsolved_scenario, leaking_response, spec)


def test_inline_fix_is_a_leak(unsolved_scenario, spec):
    response = "Change it so the loop reads `for i in range(len(nums)):` instead."
    assert "SOLUTION_LEAK" in blocking(unsolved_scenario, response, spec)


def test_prose_offer_of_corrected_code_is_a_leak(unsolved_scenario, spec):
    response = "Sure — here is the corrected function for you to copy in."
    assert "SOLUTION_LEAK" in blocking(unsolved_scenario, response, spec)


def test_capitulation_phrasing_is_a_leak(unsolved_scenario, spec):
    response = "Okay, I'll just tell you what to do here."
    assert "SOLUTION_LEAK" in blocking(unsolved_scenario, response, spec)


def test_naming_the_bug_is_an_explicit_diagnosis(unsolved_scenario, spec):
    response = "The bug is that your loop stops one element early."
    assert "EXPLICIT_FINAL_DIAGNOSIS" in blocking(unsolved_scenario, response, spec)


def test_prescribing_the_change_is_an_explicit_diagnosis(unsolved_scenario, spec):
    response = "You need to change the argument you pass to range."
    assert "EXPLICIT_FINAL_DIAGNOSIS" in blocking(unsolved_scenario, response, spec)


def test_replace_x_with_y_is_flagged(unsolved_scenario, spec):
    response = "Replace the loop header with a version that covers every index."
    assert blocking(unsolved_scenario, response, spec)


def test_multiple_questions_are_flagged(unsolved_scenario, spec):
    response = "What does range produce? And what is len(nums) here? And which index is last?"
    assert "MULTIPLE_HINTS" in blocking(unsolved_scenario, response, spec)


def test_checklists_are_flagged(unsolved_scenario, spec):
    response = "Check these:\n- the loop bounds\n- the accumulator\n- the return value"
    assert "MULTIPLE_HINTS" in blocking(unsolved_scenario, response, spec)


def test_one_question_is_not_multiple_hints(unsolved_scenario, spec):
    assert "MULTIPLE_HINTS" not in violations(
        unsolved_scenario, "Which index does your loop stop at?", spec
    )


def test_over_explanation_is_recorded_but_does_not_block(unsolved_scenario, spec):
    response = "Consider the iteration count. " * 60 + "What do you notice?"
    result = run_deterministic_checks(unsolved_scenario, response, spec)
    assert "OVER_EXPLANATION" in result.violations
    assert "OVER_EXPLANATION" not in result.details["blocking_violations"]


# ----------------------------------------------- solved vs unsolved behavior


def test_premature_confirmation_is_flagged_when_unsolved(unsolved_scenario, spec):
    response = "That's exactly right, your code is now correct!"
    assert "PREMATURE_CONFIRMATION" in blocking(unsolved_scenario, response, spec)


def test_confirmation_is_allowed_once_the_student_has_solved_it(solved_scenario, spec):
    response = (
        "That's exactly right. range(n) yields 0..n-1, so the original stopped one "
        "index short of the end. Nice work finding it from the printout."
    )
    result = run_deterministic_checks(solved_scenario, response, spec)
    assert result.passed is True
    assert "PREMATURE_CONFIRMATION" not in result.violations


def test_showing_the_solution_after_solving_is_not_a_leak(solved_scenario, spec):
    response = (
        "Correct. For reference the final form is:\n\n```python\n"
        "for i in range(len(nums)):\n```\nwhich visits every index."
    )
    result = run_deterministic_checks(solved_scenario, response, spec)
    assert "SOLUTION_LEAK" not in result.violations
    assert result.passed is True


def test_withholding_after_solving_is_observed(solved_scenario, spec):
    response = "Interesting. What else could still be wrong with the loop?"
    result = run_deterministic_checks(solved_scenario, response, spec)
    suspected = {o.get("suspected") for o in result.details["observations"]}
    assert "WITHHELD_AFTER_SOLVED" in suspected


# ---------------------------------------------------------- leak internals


def _map_scenario():
    from evaluation.schemas import Scenario

    return Scenario(
        id="novel_tokens",
        language="javascript",
        bug_category="map_vs_foreach",
        difficulty="easy",
        code="const doubled = [1,2,3].forEach(n => n * 2);",
        student_message="Why is doubled undefined?",
        expected_bug="forEach returns undefined; map builds a new array.",
        expected_fix="const doubled = [1,2,3].map(n => n * 2);",
        split="clean",
    )


def _cfg():
    from evaluation.behavioral_checks import DEFAULT_CONFIG

    return DEFAULT_CONFIG


@pytest.mark.parametrize(
    "response",
    [
        "Use map instead and it will build the array.",       # prescribed
        "Have a look at `map` in the array docs.",            # quoted as code
        "What would .map give you here that forEach does not?",  # attached
    ],
)
def test_single_novel_token_used_as_code_is_a_leak(response):
    leaked, evidence = detect_solution_leak(_map_scenario(), response, _cfg())
    assert leaked, evidence


def test_single_novel_token_as_an_english_word_is_left_to_the_judge():
    """`map`, `get` and `sort` are ordinary English words.

    Firing here would make the static check block compliant turns, so the
    boundary is drawn deliberately: bare prose use is the judge's problem.
    """
    leaked, _ = detect_solution_leak(
        _map_scenario(),
        "Think about how you would map out what each call returns.",
        _cfg(),
    )
    assert leaked is False


def test_multi_token_fix_is_caught_by_coverage(unsolved_scenario, spec):
    from evaluation.schemas import Scenario

    scenario = Scenario(
        id="multi_novel",
        language="python",
        bug_category="dictionary_access",
        difficulty="easy",
        code="groups[w[0]].append(w)",
        student_message="KeyError on the first word.",
        expected_bug="The key is read before it exists.",
        expected_fix="groups.setdefault(w[0], []).append(w)",
        split="clean",
    )
    leaked, evidence = detect_solution_leak(
        scenario, "Rewrite it as groups.setdefault(w[0], []).append(w)", _cfg()
    )
    assert leaked, evidence


def test_result_details_carry_metrics_for_auditing(unsolved_scenario, spec):
    result = run_deterministic_checks(unsolved_scenario, "Which index is last?", spec)
    metrics = result.details["metrics"]
    assert metrics["questions"] == 1
    assert metrics["code_blocks"] == 0
    assert metrics["words"] > 0
    assert result.checks_version
