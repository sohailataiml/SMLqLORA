"""Convert accepted examples into the chat format the base model expects.

Three design decisions worth stating, because they shape what the experiment can
claim:

1. **Training uses the WEAK system prompt.** Examples are trained (and the tuned
   model is later evaluated) under the one-line `zero_shot` instruction, not the
   elaborate structured prompt. The claim under test is that the behavior lives
   in the weights. Training under the structured prompt would only show that a
   model can follow a prompt it was already given.

2. **Subsets are nested.** The N=125 set is a prefix of N=250, which is a prefix
   of N=500. The data-efficiency curve then varies one thing — quantity — rather
   than quantity *and* composition.

3. **Contamination is checked here and raises.** The conversion step is the last
   place before weights change, so it fails loudly rather than warning.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from evaluation.schemas import Scenario, iter_jsonl
from filtering.dedupe import check_contamination
from generation.schemas import GeneratedExample
from prompting.strategies import ZeroShotStrategy, render_conversation

TRAINING_FORMAT_VERSION = "1.0.0"

#: The system prompt baked into every training example. Deliberately weak.
TRAINING_SYSTEM_PROMPT = ZeroShotStrategy().system_prompt()


class ContaminationError(RuntimeError):
    """Raised when a training example overlaps the evaluation set."""


@dataclass
class DatasetSplit:
    train: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.train) + len(self.validation)


def to_chat_record(
    example: GeneratedExample,
    *,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """One example as a `messages` record.

    Deterministic: the same example always produces byte-identical output.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in render_conversation(example.scenario):
        messages.append({"role": message.role.value, "content": message.content})
    messages.append({"role": "assistant", "content": example.tutor_response.strip()})

    record: dict[str, Any] = {"messages": messages}
    if include_metadata:
        record["meta"] = {
            "id": example.id,
            "language": example.scenario.language.value,
            "bug_category": example.scenario.bug_category,
            "difficulty": example.scenario.difficulty.value,
            "pressure_type": example.scenario.pressure_type.value,
            "student_has_solved": example.scenario.student_has_solved,
            "turns": example.scenario.turn_count,
            "dataset_version": example.provenance.dataset_version,
            "teacher_model": example.provenance.teacher_model,
        }
    return record


def stable_order(examples: Sequence[GeneratedExample], seed: int = 13) -> list[GeneratedExample]:
    """A deterministic shuffle keyed on content, not on generation order.

    Sorting by content hash first makes the order independent of how the
    candidates happened to arrive, so re-running generation with different
    concurrency cannot silently change which examples land in the N=125 subset.
    """
    ordered = sorted(examples, key=lambda e: e.content_hash())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def assert_no_contamination(
    examples: Sequence[GeneratedExample],
    eval_scenarios: Sequence[Scenario],
) -> None:
    report = check_contamination(examples, eval_scenarios)
    if not report.clean:
        details = ", ".join(report.contaminated_ids[:5]) or ""
        near = "; ".join(
            f"{m['train_id']}~{m['eval_id']} ({m['jaccard']})"
            for m in report.near_matches[:5]
        )
        raise ContaminationError(
            f"Training data overlaps the evaluation set — refusing to build.\n"
            f"  exact: {details}\n  near: {near}\n"
            f"Re-run the quality gate, which drops contaminated examples, or "
            f"regenerate with a different seed."
        )


def build_dataset(
    examples: Sequence[GeneratedExample],
    *,
    eval_scenarios: Sequence[Scenario] = (),
    validation_fraction: float = 0.1,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
    seed: int = 13,
    limit: int | None = None,
) -> DatasetSplit:
    """Build a train/validation split, optionally truncated to `limit` examples."""
    if not examples:
        raise ValueError("build_dataset() requires at least one accepted example")

    if eval_scenarios:
        assert_no_contamination(examples, eval_scenarios)

    ordered = stable_order(examples, seed=seed)
    if limit is not None:
        if limit > len(ordered):
            raise ValueError(
                f"requested {limit} examples but only {len(ordered)} are available"
            )
        ordered = ordered[:limit]

    validation_size = max(1, int(round(len(ordered) * validation_fraction)))
    validation_size = min(validation_size, max(0, len(ordered) - 1))

    validation = ordered[:validation_size]
    train = ordered[validation_size:]

    return DatasetSplit(
        train=[to_chat_record(e, system_prompt=system_prompt) for e in train],
        validation=[to_chat_record(e, system_prompt=system_prompt) for e in validation],
    )


def nested_subsets(
    examples: Sequence[GeneratedExample], sizes: Sequence[int], *, seed: int = 13
) -> dict[int, list[GeneratedExample]]:
    """Nested subsets for the data-efficiency sweep: each size is a prefix."""
    ordered = stable_order(examples, seed=seed)
    available = len(ordered)
    out: dict[int, list[GeneratedExample]] = {}
    for size in sorted(sizes):
        if size > available:
            raise ValueError(
                f"data-efficiency sweep asks for N={size} but only {available} "
                f"accepted examples exist. Generate more, or lower the sweep sizes."
            )
        out[size] = ordered[:size]
    return out


def suggested_sweep_sizes(total: int) -> list[int]:
    """Log-spaced sizes ending at the full dataset.

    The brief suggests 125/250/500/1000; when the accepted set is smaller than
    that, the sweep is rescaled rather than silently asking for data that does
    not exist.
    """
    if total < 8:
        raise ValueError(f"need at least 8 examples for a sweep, have {total}")
    canonical = [125, 250, 500, 1000]
    if total >= 1000:
        return canonical
    sizes: list[int] = []
    size = total
    while size >= 8 and len(sizes) < 4:
        sizes.append(int(size))
        size = size / 2
    return sorted(set(sizes))


# =============================================================================
# Persistence
# =============================================================================


def write_dataset(split: DatasetSplit, out_dir: str | Path) -> dict[str, Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": directory / "train.jsonl",
        "validation": directory / "validation.jsonl",
    }
    for key, rows in (("train", split.train), ("validation", split.validation)):
        with paths[key].open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paths


def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def dataset_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    """Hash of the exact training text — pinned in the training config."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def load_accepted(path: str | Path) -> list[GeneratedExample]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No accepted examples in {path}")
    return [GeneratedExample.model_validate(row) for row in rows]


__all__ = [
    "ContaminationError",
    "DatasetSplit",
    "TRAINING_FORMAT_VERSION",
    "TRAINING_SYSTEM_PROMPT",
    "assert_no_contamination",
    "build_dataset",
    "dataset_fingerprint",
    "load_accepted",
    "nested_subsets",
    "read_dataset",
    "stable_order",
    "suggested_sweep_sizes",
    "to_chat_record",
    "write_dataset",
]
