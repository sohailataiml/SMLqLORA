"""Static checks for teacher candidates.

Two layers:

1. the same behavioral checks the evaluator runs — a training example that would
   fail evaluation must never be trained on; and
2. generation-specific integrity checks that only make sense for synthetic data:
   does the snippet parse, is the "fix" actually different from the code, did the
   teacher smuggle the answer into its own conversation history.

Layer 2 matters because a teacher failure mode is producing a *self-consistent
but useless* example — a fix identical to the code, or a history where the
earlier tutor turn already gave the game away.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from behavior.spec import BehaviorSpec, load_spec
from evaluation.behavioral_checks import (
    CheckConfig,
    normalize_code,
    run_deterministic_checks,
)
from evaluation.schemas import DeterministicResult, Role
from generation.schemas import GeneratedExample

#: Bounds on a usable training snippet.
MIN_CODE_CHARS = 25
MAX_CODE_CHARS = 1600
MIN_RESPONSE_CHARS = 15

#: Maximum tutor-response length, **by learner state**, because the spec asks
#: for two different shapes of response and a single bound silently enforces
#: only one of them.
#:
#: Unresolved: exactly one Socratic question, deliberately tight. Measured over
#: tranche 1, real unresolved responses ran a median of 347 characters and a
#: maximum of 626, so 1200 is already generous headroom.
#:
#: Solved: the spec requires the opposite behavior — confirm the fix and explain
#: why it works. The generation prompt says so explicitly ("explain the
#: underlying cause as fully as is useful"). Real solved responses ran a median
#: of 1314 characters, so the old flat 1200-character cap rejected 98.8% of them
#: at the static stage, before the judge ever saw them. That is not a quality
#: standard, it is a rule that contradicts the behavior it is meant to enforce:
#: applied as-is it would starve the dataset of `solved` examples and teach
#: "never confirm an answer", which is the exact failure the spec calls
#: WITHHELD_AFTER_SOLVED.
#:
#: The bound stays a blunt sanity check on either side. Whether an explanation
#: is *too long for its purpose* is a semantic question, and it remains the
#: judge's to answer via OVER_EXPLANATION.
MAX_RESPONSE_CHARS = 1200
MAX_RESPONSE_CHARS_SOLVED = 2400


def max_response_chars(example: GeneratedExample) -> int:
    """Response-length ceiling appropriate to the learner's state."""
    if example.scenario.student_has_solved:
        return MAX_RESPONSE_CHARS_SOLVED
    return MAX_RESPONSE_CHARS


@dataclass(frozen=True)
class IntegrityResult:
    ok: bool
    codes: tuple[str, ...]
    notes: tuple[str, ...]


def _python_parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _looks_like_javascript(code: str) -> bool:
    """Cheap sanity check — we do not ship a JS parser for this."""
    if re.search(r"^\s*(def|elif)\s", code, re.M):
        return False
    return bool(
        re.search(r"(function|=>|const |let |var |class |return|await|\{)", code)
    )


def check_integrity(example: GeneratedExample) -> IntegrityResult:
    """Generation-specific validity, independent of tutoring behavior."""
    codes: list[str] = []
    notes: list[str] = []

    scenario = example.scenario
    code = scenario.code

    if not (MIN_CODE_CHARS <= len(code) <= MAX_CODE_CHARS):
        codes.append("LOW_QUALITY")
        notes.append(f"code length {len(code)} outside [{MIN_CODE_CHARS}, {MAX_CODE_CHARS}]")

    if scenario.language.value == "python" and not _python_parses(code):
        codes.append("INVALID_SCHEMA")
        notes.append("python snippet does not parse; a runtime bug was requested, not a syntax error")
    if scenario.language.value == "javascript" and not _looks_like_javascript(code):
        codes.append("INVALID_SCHEMA")
        notes.append("snippet does not look like javascript")

    ceiling = max_response_chars(example)
    if not (MIN_RESPONSE_CHARS <= len(example.tutor_response) <= ceiling):
        codes.append("LOW_QUALITY")
        notes.append(
            f"tutor_response length {len(example.tutor_response)} out of bounds "
            f"(limit {ceiling} for "
            f"{'solved' if example.scenario.student_has_solved else 'unresolved'})"
        )

    # A "fix" identical to the code teaches nothing.
    if normalize_code(scenario.expected_fix) == normalize_code(code):
        codes.append("INCORRECT_DIAGNOSIS")
        notes.append("expected_fix is identical to the buggy code")

    if normalize_code(scenario.expected_fix) in ("", "none", "n/a"):
        codes.append("INCORRECT_DIAGNOSIS")
        notes.append("expected_fix is empty or a placeholder")

    # The ground-truth bug must actually be described, not restated as the fix.
    if scenario.expected_bug.strip().lower() == scenario.expected_fix.strip().lower():
        codes.append("INCORRECT_DIAGNOSIS")
        notes.append("expected_bug and expected_fix are the same string")

    # The teacher must not answer its own question in the response.
    if example.tutor_response.strip().lower() == scenario.student_message.strip().lower():
        codes.append("LOW_QUALITY")
        notes.append("tutor_response echoes the student message")

    # Prior tutor turns must obey the constraint too, or the model learns from
    # a conversation whose earlier turns already leaked the answer.
    if not scenario.student_has_solved:
        fix_norm = normalize_code(scenario.expected_fix)
        for index, message in enumerate(scenario.conversation_history):
            if message.role is not Role.ASSISTANT:
                continue
            if len(fix_norm) >= 8 and fix_norm in normalize_code(message.content):
                codes.append("SOLUTION_LEAK")
                notes.append(f"conversation_history[{index}] already contains the fix")
                break

    return IntegrityResult(ok=not codes, codes=tuple(dict.fromkeys(codes)),
                           notes=tuple(notes))


def check_behavior(
    example: GeneratedExample,
    spec: BehaviorSpec | None = None,
    config: CheckConfig | None = None,
) -> DeterministicResult:
    """Run the evaluator's own static checks against the candidate's tutor turn."""
    return run_deterministic_checks(
        example.scenario, example.tutor_response, spec or load_spec(), config
    )


def static_screen(
    example: GeneratedExample,
    spec: BehaviorSpec | None = None,
    config: CheckConfig | None = None,
) -> tuple[bool, tuple[str, ...], DeterministicResult, IntegrityResult]:
    """Combined static verdict for one candidate."""
    spec = spec or load_spec()
    integrity = check_integrity(example)
    behavior = check_behavior(example, spec, config)

    codes: list[str] = list(integrity.codes)
    blocking = behavior.details.get("blocking_violations", [])
    for code in blocking:
        if code not in codes:
            codes.append(code)

    # Non-blocking behavioral violations are stricter here than at evaluation
    # time: an over-long tutor turn is acceptable output but poor training data.
    for code in behavior.violations:
        if code == "OVER_EXPLANATION" and code not in codes:
            codes.append(code)

    return (not codes), tuple(codes), behavior, integrity


__all__ = [
    "IntegrityResult",
    "MAX_CODE_CHARS",
    "MAX_RESPONSE_CHARS",
    "MIN_CODE_CHARS",
    "MIN_RESPONSE_CHARS",
    "check_behavior",
    "check_integrity",
    "static_screen",
]
