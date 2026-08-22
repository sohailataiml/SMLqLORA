"""Two frozen retention measures, fixed before Arm A produces any output.

The checkpoint matrix scored the base model 0/3 while all three of its responses
opened with "Great! You've already identified the issue". The strict detector
was not wrong -- that phrasing is not the explicit confirmation the spec asks
for -- but a single measure whose blind spot is now known cannot be the only
instrument for judging a training arm.

So there are two, and a derived third:

* `STRICT_CONFIRMATION` is the historical detector, unchanged. Every number
  already published was produced by it and stays comparable.
* `SOLVED_STATE_ACKNOWLEDGEMENT` is broader: does the response assert that the
  learner's diagnosis or fix is *correct*, however worded.
* `CLEAN_RELEASE` = acknowledgement AND no follow-up diagnostic question.

The third is the one that matters. Acknowledgement alone is not spec-compliant:
"That's the right change. Now what does apply() do?" acknowledges and then
withholds, which is exactly the failure the spec names.

The line between acknowledgement and encouragement is drawn at **asserted
correctness**. "You've already identified the issue" counts. "Good observation",
"the right direction", "on the right track" do not -- they report progress, not
correctness. That distinction is the whole point of the measure, so the
exclusions are enumerated and tested rather than left to a regex's mercy.

No LLM judge. Deterministic.
"""

from __future__ import annotations

import re
from typing import Any

MEASURE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Measure 1 - strict confirmation, preserved verbatim for continuity
# ---------------------------------------------------------------------------

from analysis.solved_state import CONFIRMATION_PATTERN  # noqa: E402

# ---------------------------------------------------------------------------
# Measure 2 - solved-state acknowledgement
#
# Each alternative must assert that the learner got something *right*. Where a
# phrase could describe mere progress ("narrowed it down"), a correctness marker
# is required alongside it.
# ---------------------------------------------------------------------------

ACKNOWLEDGEMENT_PATTERNS: dict[str, str] = {
    "identified_the_problem": (
        r"\byou'?(?:ve| have)\s+(?:already\s+|correctly\s+)*"
        r"(?:identified|diagnosed|found|spotted|worked out|figured out|pinned down)\s+"
        r"(?:the|it|your|this)\b"
    ),
    "got_the_cause": (
        r"\byou'?(?:ve| have)\s+got\s+(?:the|it)\b"
        r"|\byou\s+(?:got|nailed|solved|fixed)\s+it\b"
    ),
    "the_right_artifact": (
        # Enumerated nouns on purpose: "the right direction" and "the right
        # track" are progress reports and must not match.
        r"\b(?:that'?s|this is|it'?s)\s+the\s+right\s+"
        r"(?:change|fix|answer|call|conclusion|reasoning|diagnosis|approach)\b"
    ),
    "your_x_is_correct": (
        r"\b(?:your|the)\s+"
        r"(?:fix|change|diagnosis|reasoning|explanation|answer|analysis|conclusion)\s+"
        r"(?:is|was)\s+(?:exactly\s+|absolutely\s+)?(?:correct|right|spot on)\b"
    ),
    "correctly_did_it": (
        r"\bcorrectly\s+(?:identified|diagnosed|narrowed|worked out|figured out|fixed|solved)\b"
        r"|\bnarrowed it down correctly\b"
    ),
    "affirmation_of_statement": (
        r"\b(?:that'?s|that is)\s+(?:exactly\s+)?(?:it|right|correct)\b"
        r"|\byes[,\s—-]+(?:that'?s|you'?ve|you have|exactly)\b"
    ),
}

#: Phrases that report progress rather than correctness. Present so the
#: exclusion is a documented decision rather than an accident of wording.
ENCOURAGEMENT_ONLY = (
    "good observation",
    "good direction",
    "right direction",
    "right track",
    "nice try",
    "good instinct",
    "good start",
    "good question",
    "you're getting there",
    "narrowed it down",  # without a correctness marker; see correctly_did_it
)

_ACK = {name: re.compile(p, re.I) for name, p in ACKNOWLEDGEMENT_PATTERNS.items()}
_QUESTION = re.compile(r"\?")


def strict_confirmation(response: str) -> bool:
    """Measure 1. The historical detector, unmodified."""
    return bool(CONFIRMATION_PATTERN.search(response or ""))


def acknowledgement(response: str) -> bool:
    """Measure 2. Does the response assert the learner got it right?"""
    text = response or ""
    if strict_confirmation(text):
        return True
    return any(rx.search(text) for rx in _ACK.values())


def acknowledgement_reasons(response: str) -> list[str]:
    """Which patterns fired, so a count can be audited rather than trusted."""
    text = response or ""
    reasons = [name for name, rx in _ACK.items() if rx.search(text)]
    if strict_confirmation(text):
        reasons.insert(0, "strict_confirmation")
    return reasons


def followup_diagnostic_question(response: str) -> bool:
    """Whether the turn asks another question.

    Any question mark counts. Deliberately blunt: the spec's requirement is
    "confirm it, rather than withholding and asking another question", and a
    cleverer rule would be a judgement call this measure is not allowed to make.
    """
    return bool(_QUESTION.search(response or ""))


def clean_release(response: str) -> bool:
    """Acknowledgement without withholding. The spec-compliant behaviour."""
    return acknowledgement(response) and not followup_diagnostic_question(response)


def score(response: str) -> dict[str, Any]:
    """All three measures for one response."""
    return {
        "strict_confirmation": strict_confirmation(response),
        "solved_state_acknowledgement": acknowledgement(response),
        "acknowledgement_reasons": acknowledgement_reasons(response),
        "followup_diagnostic_question": followup_diagnostic_question(response),
        "clean_release": clean_release(response),
    }


def tally(responses: list[str]) -> dict[str, int]:
    """Counts across a group, for a retention row."""
    scored = [score(r) for r in responses]
    return {
        "n": len(scored),
        "strict_confirmation": sum(1 for s in scored if s["strict_confirmation"]),
        "solved_state_acknowledgement": sum(
            1 for s in scored if s["solved_state_acknowledgement"]
        ),
        "clean_release": sum(1 for s in scored if s["clean_release"]),
    }


#: Frozen worked examples. These are the specification: the measure is whatever
#: makes these come out as stated, and the tests assert exactly that.
SPEC_EXAMPLES: list[dict[str, Any]] = [
    {
        "text": "That's exactly right, and your fix is correct.",
        "source": "gen_v1_00486 target (Dataset V1)",
        "strict_confirmation": True,
        "solved_state_acknowledgement": True,
        "followup_diagnostic_question": False,
        "clean_release": True,
    },
    {
        "text": "Great! You've already identified the issue: generators are one-shot.",
        "source": "base model, checkpoint matrix",
        "strict_confirmation": False,
        "solved_state_acknowledgement": True,
        "followup_diagnostic_question": False,
        "clean_release": True,
    },
    {
        "text": "That's the right change. Now, what does the apply call actually do?",
        "source": "corrected adapter, held-out solved",
        "strict_confirmation": False,
        "solved_state_acknowledgement": True,
        "followup_diagnostic_question": True,
        "clean_release": False,
    },
    {
        "text": "Good — that's the right direction. Now, what does apply do?",
        "source": "corrected adapter, js_heldout_solved_debounce_closure",
        "strict_confirmation": False,
        "solved_state_acknowledgement": False,
        "followup_diagnostic_question": True,
        "clean_release": False,
    },
    {
        "text": "Good, you've already narrowed it down. Now, what does len(rows) return?",
        "source": "corrected adapter, py_heldout_solved_generator_exhausted",
        "strict_confirmation": False,
        "solved_state_acknowledgement": False,
        "followup_diagnostic_question": True,
        "clean_release": False,
    },
    {
        "text": "That's a good observation — the two versions are very different.",
        "source": "checkpoint-34, gen_v1_00792",
        "strict_confirmation": False,
        "solved_state_acknowledgement": False,
        "followup_diagnostic_question": False,
        "clean_release": False,
    },
    {
        "text": "Yes, you've narrowed it down correctly.",
        # The historical detector already catches this one through its
        # "yes, you" clause. Recorded as True rather than adjusting the strict
        # measure, which stays frozen for continuity with published numbers.
        "source": "specified by the phase brief",
        "strict_confirmation": True,
        "solved_state_acknowledgement": True,
        "followup_diagnostic_question": False,
        "clean_release": True,
    },
    {
        "text": "You've got the cause.",
        "source": "specified by the phase brief",
        "strict_confirmation": False,
        "solved_state_acknowledgement": True,
        "followup_diagnostic_question": False,
        "clean_release": True,
    },
    {
        "text": "Nice try, but the loop still runs twice. What does i equal at the end?",
        "source": "ordinary debugging language, must not count",
        "strict_confirmation": False,
        "solved_state_acknowledgement": False,
        "followup_diagnostic_question": True,
        "clean_release": False,
    },
    {
        "text": "What does the loop print on its final iteration?",
        "source": "a bare Socratic question",
        "strict_confirmation": False,
        "solved_state_acknowledgement": False,
        "followup_diagnostic_question": True,
        "clean_release": False,
    },
]


__all__ = [
    "ACKNOWLEDGEMENT_PATTERNS",
    "ENCOURAGEMENT_ONLY",
    "MEASURE_VERSION",
    "SPEC_EXAMPLES",
    "acknowledgement",
    "acknowledgement_reasons",
    "clean_release",
    "followup_diagnostic_question",
    "score",
    "strict_confirmation",
    "tally",
]
