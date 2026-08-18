"""Duplicate and near-duplicate detection, plus train/eval contamination checks.

Teachers repeat themselves. Left unchecked, a dataset quietly becomes a hundred
copies of the same off-by-one example, and the data-efficiency curve then
measures repetition rather than coverage.

Exact duplicates are caught by hash. Near-duplicates are caught with Jaccard
similarity over word shingles, restricted by an inverted index so the comparison
is near-linear rather than quadratic.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from evaluation.schemas import Scenario
from generation.schemas import GeneratedExample

SHINGLE_SIZE = 4
DEFAULT_THRESHOLD = 0.72

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Word n-grams. Short texts fall back to their own token set."""
    words = _tokens(text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


def example_text(example: GeneratedExample) -> str:
    """The text that defines whether two examples teach the same thing."""
    return "\n".join(
        [example.scenario.code, example.scenario.student_message, example.tutor_response]
    )


@dataclass
class DedupeResult:
    kept: list[GeneratedExample] = field(default_factory=list)
    duplicates: list[GeneratedExample] = field(default_factory=list)
    exact_count: int = 0
    near_count: int = 0

    @property
    def removed(self) -> int:
        return len(self.duplicates)


def deduplicate(
    examples: Sequence[GeneratedExample],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    shingle_size: int = SHINGLE_SIZE,
) -> DedupeResult:
    """Keep the first occurrence of each distinct example.

    Order is preserved, so the result is deterministic for a given input order.
    """
    result = DedupeResult()
    seen_hashes: dict[str, str] = {}
    index: dict[str, list[int]] = defaultdict(list)
    kept_shingles: list[set[str]] = []

    for example in examples:
        digest = example.content_hash()
        if digest in seen_hashes:
            result.exact_count += 1
            result.duplicates.append(
                example.model_copy(
                    update={
                        "accepted": False,
                        "rejection_codes": ("DUPLICATE",),
                        "duplicate_of": seen_hashes[digest],
                        "gate_notes": "exact duplicate",
                    }
                )
            )
            continue

        current = shingles(example_text(example), shingle_size)
        candidates: set[int] = set()
        for shingle in current:
            candidates.update(index[shingle])

        best_position, best_score = None, 0.0
        for position in candidates:
            score = jaccard(current, kept_shingles[position])
            if score > best_score:
                best_position, best_score = position, score

        if best_position is not None and best_score >= threshold:
            result.near_count += 1
            result.duplicates.append(
                example.model_copy(
                    update={
                        "accepted": False,
                        "rejection_codes": ("DUPLICATE",),
                        "duplicate_of": result.kept[best_position].id,
                        "gate_notes": f"near-duplicate (jaccard={best_score:.3f})",
                    }
                )
            )
            continue

        position = len(result.kept)
        result.kept.append(example)
        kept_shingles.append(current)
        seen_hashes[digest] = example.id
        for shingle in current:
            index[shingle].append(position)

    return result


# =============================================================================
# Contamination
# =============================================================================


@dataclass
class ContaminationReport:
    contaminated_ids: list[str] = field(default_factory=list)
    near_matches: list[dict] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.contaminated_ids and not self.near_matches

    def summary(self) -> str:
        if self.clean:
            return "No training example matches or closely resembles an evaluation scenario."
        return (
            f"{len(self.contaminated_ids)} exact and {len(self.near_matches)} near "
            f"overlap(s) with the evaluation set."
        )


def check_contamination(
    examples: Sequence[GeneratedExample],
    eval_scenarios: Sequence[Scenario],
    *,
    threshold: float = 0.60,
) -> ContaminationReport:
    """Detect training examples that overlap the evaluation scenarios.

    Exact matches use the scenario content hash. Near matches compare the code
    and learner message, which is where leakage actually shows up — the tutor
    turn is expected to differ.
    """
    report = ContaminationReport()
    eval_hashes = {s.content_hash(): s.id for s in eval_scenarios}

    eval_shingles = [
        (s.id, shingles(f"{s.code}\n{s.student_message}")) for s in eval_scenarios
    ]
    index: dict[str, list[int]] = defaultdict(list)
    for position, (_, shingle_set) in enumerate(eval_shingles):
        for shingle in shingle_set:
            index[shingle].append(position)

    for example in examples:
        digest = example.scenario.content_hash()
        if digest in eval_hashes:
            report.contaminated_ids.append(example.id)
            continue

        current = shingles(
            f"{example.scenario.code}\n{example.scenario.student_message}"
        )
        candidates: set[int] = set()
        for shingle in current:
            candidates.update(index[shingle])

        for position in candidates:
            score = jaccard(current, eval_shingles[position][1])
            if score >= threshold:
                report.near_matches.append(
                    {
                        "train_id": example.id,
                        "eval_id": eval_shingles[position][0],
                        "jaccard": round(score, 3),
                    }
                )
                break

    return report


def drop_contaminated(
    examples: Sequence[GeneratedExample], report: ContaminationReport
) -> tuple[list[GeneratedExample], list[GeneratedExample]]:
    """Split into (clean, contaminated), marking the contaminated records."""
    flagged = set(report.contaminated_ids) | {m["train_id"] for m in report.near_matches}
    clean: list[GeneratedExample] = []
    dirty: list[GeneratedExample] = []
    for example in examples:
        if example.id in flagged:
            dirty.append(
                example.model_copy(
                    update={
                        "accepted": False,
                        "rejection_codes": ("CONTAMINATED",),
                        "gate_notes": "overlaps an evaluation scenario",
                    }
                )
            )
        else:
            clean.append(example)
    return clean, dirty


__all__ = [
    "ContaminationReport",
    "DEFAULT_THRESHOLD",
    "DedupeResult",
    "SHINGLE_SIZE",
    "check_contamination",
    "deduplicate",
    "drop_contaminated",
    "example_text",
    "jaccard",
    "shingles",
]
