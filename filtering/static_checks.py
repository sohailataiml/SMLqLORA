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
MAX_RESPONSE_CHARS = 1200


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

    if not (MIN_RESPONSE_CHARS <= len(example.tutor_response) <= MAX_RESPONSE_CHARS):
        codes.append("LOW_QUALITY")
        notes.append(f"tutor_response length {len(example.tutor_response)} out of bounds")

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
