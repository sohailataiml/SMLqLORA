"""Loading and shared text primitives for artifact analysis.

The three corpora that matter are the frozen training set, the held-out eval
set, and the judge transcripts from a base-vs-tuned run. Every function here is
pure and offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_V1 = REPO_ROOT / "data/versions/v1/selected.jsonl"
HELDOUT = REPO_ROOT / "scenarios/heldout.jsonl"
TRANSCRIPTS = REPO_ROOT / "results/base_vs_tuned/judge_transcripts.jsonl"

#: A response is treated as truncated when it spent the whole generation budget.
#: `EVAL_PARAMS.max_tokens` is 800 and no scenario needs anything close to it.
MAX_TOKENS_BUDGET = 800

#: An n-gram must recur across this many tuned responses before it counts as a
#: habit rather than a coincidence of one output.
MIN_TUNED_RESPONSES_FOR_HABIT = 3

#: Phrase length for the out-of-distribution scan. Long enough that a match is a
#: real shared phrasing, short enough to survive small wording differences.
NGRAM_SIZE = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass(frozen=True)
class Example:
    """One training example, flattened to what the analysis actually uses."""

    id: str
    response: str
    pressure_type: str
    bug_category: str
    language: str
    difficulty: str
    student_progress: str
    student_has_solved: bool
    prior_tutor_turns: int

    @property
    def is_first_turn(self) -> bool:
        return self.prior_tutor_turns == 0


def load_dataset_v1(path: Path = DATASET_V1) -> list[Example]:
    out = []
    for row in read_jsonl(path):
        scenario, dims = row["scenario"], row["dimensions"]
        out.append(
            Example(
                id=row["id"],
                response=row["tutor_response"],
                pressure_type=dims["pressure_type"],
                bug_category=dims["bug_category"],
                language=dims["language"],
                difficulty=dims["difficulty"],
                student_progress=dims["student_progress"],
                # `student_has_solved` lives on the scenario, not the dimension
                # block -- `dimensions` has `student_progress` instead, and
                # reading the wrong one silently yields None for all 600 rows.
                student_has_solved=bool(scenario.get("student_has_solved")),
                prior_tutor_turns=count_tutor_turns(scenario["conversation_history"]),
            )
        )
    return out


def count_tutor_turns(history: Sequence[dict[str, Any]]) -> int:
    """Prior assistant turns — the position the response is being generated at."""
    return sum(1 for message in history if message.get("role") == "assistant")


def load_heldout(path: Path = HELDOUT) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path)}


def split_by_model(transcripts: Iterable[dict[str, Any]]) -> tuple[list, list]:
    """Separate base from tuned records by the presence of a PEFT adapter."""
    records = list(transcripts)
    tuned = [r for r in records if "peft" in r["model"]]
    base = [r for r in records if "peft" not in r["model"]]
    return base, tuned


_SENTENCE = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[a-z0-9_']+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def repeated_sentence_count(text: str) -> int:
    """How many sentences are verbatim repeats of an earlier one.

    Zero for every one of the 600 V1 tutor responses, which is what makes a
    non-zero value here evidence about the model rather than about the data.
    """
    seen = sentences(text)
    return len(seen) - len(set(seen))


def ngrams(text: str, n: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def out_of_distribution_phrases(
    tuned_responses: Sequence[str],
    reference_responses: Sequence[str],
    *,
    n: int = NGRAM_SIZE,
    min_responses: int = MIN_TUNED_RESPONSES_FOR_HABIT,
) -> list[dict[str, Any]]:
    """Phrases the tuned model repeats that never occur in its training data.

    A fine-tuned model's habits are supposed to come from its training set. An
    n-gram used across several outputs but absent from all 600 reference
    responses was not learned from them, which points at the checkpoint rather
    than at the data.
    """
    reference = set()
    for text in reference_responses:
        reference |= ngrams(text, n)

    counts: dict[tuple[str, ...], int] = {}
    for text in tuned_responses:
        for gram in ngrams(text, n):
            if gram not in reference:
                counts[gram] = counts.get(gram, 0) + 1

    hits = [
        {"phrase": " ".join(gram), "tuned_responses": count, "v1_responses": 0}
        for gram, count in counts.items()
        if count >= min_responses
    ]
    return sorted(hits, key=lambda h: (-h["tuned_responses"], h["phrase"]))
