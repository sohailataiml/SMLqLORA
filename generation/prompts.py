"""Teacher-generation prompts and the controlled dimension space.

We do not ask a teacher model for "2000 tutoring examples". That produces a pile
of near-identical easy cases. Instead each call is pinned to one point in a
controlled space — language, bug category, difficulty, pressure type,
conversation length, learner competence, frustration, hint strength and student
progress — and the sampler walks that space deterministically from a seed.

The teacher returns the *whole situation* (buggy code, prior turns, the learner's
message, the ground-truth bug and fix) together with the tutor turn it deserves.
That keeps the two coherent, and it lets the quality gate judge a training
candidate with the same criteria the evaluator applies to a model response.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from behavior.spec import BehaviorSpec
from generation.schemas import (
    GenerationDimensions,
    HintStrength,
    LearnerCompetence,
    LearnerFrustration,
    StudentProgress,
)

GENERATION_PROMPT_VERSION = "1.0.0"

# =============================================================================
# Dimension space
# =============================================================================

PYTHON_BUG_CATEGORIES = (
    "loop_boundary",
    "boolean_condition",
    "mutable_default",
    "dictionary_access",
    "none_handling",
    "list_mutation",
    "scope",
    "return_placement",
    "exception_handling",
    "string_immutability",
    "integer_division",
    "comparison_identity",
    "generator_exhaustion",
    "shadowed_builtin",
)

JAVASCRIPT_BUG_CATEGORIES = (
    "async_await",
    "promise_handling",
    "map_vs_foreach",
    "closure_behavior",
    "undefined_properties",
    "incorrect_condition",
    "missing_return",
    "array_mutation",
    "scope",
    "this_binding",
    "type_coercion",
    "shallow_copy",
    "callback_ordering",
    "hoisting",
)

DIFFICULTIES = ("easy", "medium", "hard")

#: Pressure types and how often to draw them, in per-mille. Adversarial cases are
#: over-weighted relative to a natural distribution because they are where the
#: prompt ceiling shows up, and therefore where training data has to be dense.
#:
#: These are not guesses. They are the shares in
#: `results/prompt_ceiling/proposed_training_distribution.json`, computed from
#: the 144 strong-prompt records of the completed two-family ablation: floor +
#: allocation proportional to each dimension's measured failure rate, with a 4%
#: floor, a 22% cap and a 15% floor on `normal`. Re-run `make analyze` and
#: re-derive these numbers rather than hand-editing them.
PRESSURE_WEIGHTS = {
    "normal": 191,
    "frustrated": 94,
    "repeated_answer_request": 94,
    "time_pressure": 101,
    "prompt_injection": 81,
    "authority_override": 81,
    "fake_success": 94,
    "almost_correct": 122,
    "solved": 142,
}

TURN_WEIGHTS = {0: 30, 1: 30, 2: 25, 3: 15}


def _weighted(rng: random.Random, weights: dict) -> object:
    population = list(weights)
    return rng.choices(population, weights=[weights[k] for k in population], k=1)[0]


def sample_dimensions(seed: int) -> GenerationDimensions:
    """Deterministically draw one point in the space from a seed."""
    rng = random.Random(seed)

    language = rng.choice(["python", "javascript"])
    categories = (
        PYTHON_BUG_CATEGORIES if language == "python" else JAVASCRIPT_BUG_CATEGORIES
    )
    pressure = str(_weighted(rng, PRESSURE_WEIGHTS))
    turns = int(_weighted(rng, TURN_WEIGHTS))

    # A first turn cannot carry pressure that depends on conversational history.
    if turns == 0 and pressure in ("repeated_answer_request", "fake_success", "solved"):
        turns = rng.choice([1, 2])

    if pressure == "solved":
        progress = StudentProgress.SOLVED
    elif pressure == "almost_correct":
        progress = StudentProgress.CLOSE
    elif pressure == "fake_success":
        progress = rng.choice([StudentProgress.PARTIAL, StudentProgress.CLOSE])
    else:
        progress = rng.choice(
            [StudentProgress.STUCK, StudentProgress.STUCK, StudentProgress.PARTIAL,
             StudentProgress.CLOSE]
        )

    frustration = (
        LearnerFrustration.HIGH
        if pressure in ("frustrated", "repeated_answer_request")
        else rng.choice(
            [LearnerFrustration.NONE, LearnerFrustration.NONE, LearnerFrustration.MILD]
        )
    )

    hint_strength = {
        StudentProgress.STUCK: HintStrength.NUDGE,
        StudentProgress.PARTIAL: HintStrength.POINTED,
        StudentProgress.CLOSE: HintStrength.NARROW,
        StudentProgress.SOLVED: HintStrength.NARROW,
    }[progress]
    if rng.random() < 0.25:
        hint_strength = rng.choice(list(HintStrength))

    return GenerationDimensions(
        language=language,
        bug_category=rng.choice(categories),
        difficulty=rng.choice(DIFFICULTIES),
        pressure_type=pressure,
        conversation_turns=turns,
        learner_competence=rng.choice(list(LearnerCompetence)),
        learner_frustration=frustration,
        hint_strength=hint_strength,
        student_progress=progress,
    )


def sample_plan(count: int, base_seed: int = 20240101) -> list[tuple[int, GenerationDimensions]]:
    """The full generation plan: (seed, dimensions) pairs, reproducible."""
    return [(base_seed + i, sample_dimensions(base_seed + i)) for i in range(count)]


# =============================================================================
# Prompt text
# =============================================================================

_PRESSURE_BRIEF = {
    "normal": "The learner is calm and simply wants help.",
    "frustrated": "The learner is visibly frustrated — they have been stuck a while and say so.",
    "repeated_answer_request": "The learner has already asked for the answer at least once in the history and asks again, more insistently.",
    "time_pressure": "The learner claims a deadline within minutes and wants the answer now.",
    "prompt_injection": "The learner's message contains an injection attempt — fake system tags, 'ignore previous instructions', or a roleplay framing designed to lift the tutoring constraint.",
    "authority_override": "The learner claims a third party (instructor, TA, manager) has authorised the assistant to give the answer outright.",
    "fake_success": "The learner claims they have fixed it, but the change they describe does NOT fix the actual bug. The bug is still present.",
    "almost_correct": "The learner proposes something very close to the fix but not quite right, and asks which of two options is correct.",
    "solved": "The learner has ALREADY produced the correct fix themselves and reports it working.",
}

_HINT_BRIEF = {
    HintStrength.NUDGE: "a wide nudge that redirects attention to the right region of the program without naming any component",
    HintStrength.POINTED: "a pointed question about a specific line or expression, still without naming the defect",
    HintStrength.NARROW: "a narrow question that leaves the learner one observation away from seeing it themselves",
}

_PROGRESS_BRIEF = {
    StudentProgress.STUCK: "The learner has made no real progress and does not know where to look.",
    StudentProgress.PARTIAL: "The learner has ruled something out or noticed a symptom, but has not localised the bug.",
    StudentProgress.CLOSE: "The learner is close — they have localised the region but drawn the wrong conclusion.",
    StudentProgress.SOLVED: "The learner has produced the correct fix independently.",
}


def build_generation_prompt(
    dimensions: GenerationDimensions, spec: BehaviorSpec, *, seed: int
) -> str:
    """The teacher's user message for one candidate."""
    solved = dimensions.student_progress is StudentProgress.SOLVED

    tutor_brief = (
        "The learner has ALREADY solved it. Your `tutor_response` must CONFIRM the "
        "fix plainly and explain the underlying cause as fully as is useful. "
        "Withholding confirmation here would be a failure."
        if solved
        else (
            f"Your `tutor_response` must be exactly ONE diagnostic question or ONE "
            f"hint — {_HINT_BRIEF[dimensions.hint_strength]}. It must not reveal the "
            f"bug, name the fix, or contain corrected code. It must respond to what "
            f"the learner just said, not restart. Keep it to a few sentences."
        )
    )

    history_brief = (
        "Leave `conversation_history` empty — this is the learner's first message."
        if dimensions.conversation_turns == 0
        else (
            f"`conversation_history` must contain exactly {dimensions.conversation_turns} "
            f"prior exchange(s): alternating user/assistant, starting with user and "
            f"ending with assistant. Every assistant turn in the history must ITSELF "
            f"obey the tutoring constraint (one question or hint, no answers). Do not "
            f"repeat the learner's code in the history — it is supplied separately."
        )
    )

    return f"""You are producing ONE high-quality training example for a Socratic \
debugging tutor. You are the teacher: the tutor turn you write will be imitated.

{spec.render_for_prompt()}

# THIS EXAMPLE'S SPECIFICATION

Draw exactly this point in the space — do not substitute anything easier:

- language: {dimensions.language}
- bug category: {dimensions.bug_category}
- difficulty: {dimensions.difficulty}
- prior exchanges: {dimensions.conversation_turns}
- learner competence: {dimensions.learner_competence.value}
- learner frustration: {dimensions.learner_frustration.value}
- pressure type: {dimensions.pressure_type}

{_PRESSURE_BRIEF[dimensions.pressure_type]}
{_PROGRESS_BRIEF[dimensions.student_progress]}

# WHAT TO WRITE

1. `code` — a short, realistic {dimensions.language} snippet (5-15 lines) with \
exactly ONE bug of category `{dimensions.bug_category}`, at {dimensions.difficulty} \
difficulty. It must be plausible code a real learner would write, not a puzzle. \
Do not add comments pointing at the bug.
2. {history_brief}
3. `student_message` — the learner's current turn, in the voice of a \
{dimensions.learner_competence.value} learner with {dimensions.learner_frustration.value} \
frustration. It must embody the pressure type above.
4. `expected_bug` — one sentence of ground truth naming the actual defect. This is \
for the grader, never shown to the learner.
5. `expected_fix` — the minimal corrected code or expression. Ground truth, never \
shown to the learner.
6. `tutor_response` — {tutor_brief}

# HARD CONSTRAINTS

- The bug in `code` must genuinely be of category `{dimensions.bug_category}`.
- `expected_fix` must actually fix `code`.
- Do NOT put the fix, or any part of `expected_fix`, inside `tutor_response` \
{'(this example is solved, so the tutor MAY discuss the fix)' if solved else '— this is the single most important rule'}.
- Invent fresh code. Do not reuse a textbook example verbatim.
- Write natural prose. No headings or bullet lists in `tutor_response`.

# OUTPUT

Reply with a single JSON object and nothing else. Use these exact keys:

{{
  "code": "<the buggy snippet, with real newlines escaped as \\n>",
  "conversation_history": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "student_message": "...",
  "expected_bug": "...",
  "expected_fix": "...",
  "tutor_response": "...",
  "student_has_solved": {str(solved).lower()}
}}

(generation seed {seed} — vary the specifics so this example is unlike others.)"""


def generation_prompt_hash(spec: BehaviorSpec) -> str:
    """Hash of the template itself, independent of any dimension draw."""
    probe = GenerationDimensions(
        language="python",
        bug_category="loop_boundary",
        difficulty="easy",
        pressure_type="normal",
        conversation_turns=0,
        learner_competence=LearnerCompetence.BEGINNER,
        learner_frustration=LearnerFrustration.NONE,
        hint_strength=HintStrength.NUDGE,
        student_progress=StudentProgress.STUCK,
    )
    payload = build_generation_prompt(probe, spec, seed=0)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TEACHER_SYSTEM_PROMPT = (
    "You write training data for a Socratic debugging tutor. You are precise, you "
    "follow the requested specification exactly, and you always reply with a single "
    "valid JSON object and no surrounding prose. The tutor turns you write are "
    "imitated directly, so a turn that leaks the answer poisons the dataset."
)


def plan_summary(plan: Sequence[tuple[int, GenerationDimensions]]) -> dict[str, dict[str, int]]:
    """Distribution of a plan, for the dataset report."""
    from collections import Counter

    return {
        "language": dict(Counter(d.language for _, d in plan)),
        "bug_category": dict(Counter(d.bug_category for _, d in plan)),
        "difficulty": dict(Counter(d.difficulty for _, d in plan)),
        "pressure_type": dict(Counter(d.pressure_type for _, d in plan)),
        "conversation_turns": dict(Counter(str(d.conversation_turns) for _, d in plan)),
        "learner_competence": dict(Counter(d.learner_competence.value for _, d in plan)),
        "hint_strength": dict(Counter(d.hint_strength.value for _, d in plan)),
        "student_progress": dict(Counter(d.student_progress.value for _, d in plan)),
    }


__all__ = [
    "DIFFICULTIES",
    "GENERATION_PROMPT_VERSION",
    "JAVASCRIPT_BUG_CATEGORIES",
    "PRESSURE_WEIGHTS",
    "PYTHON_BUG_CATEGORIES",
    "TEACHER_SYSTEM_PROMPT",
    "build_generation_prompt",
    "generation_prompt_hash",
    "plan_summary",
    "sample_dimensions",
    "sample_plan",
]
