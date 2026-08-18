"""Resumability — never pay twice for a call that already succeeded.

The prompt-ceiling ablation is 216 paid subject calls plus as many judge calls.
When a run dies partway through (exhausted credit, a dropped connection, a
Ctrl-C), re-running it from zero wastes money on work that is already on disk.

A result is identified by:

    (model, prompt_strategy, scenario_id, prompt_version)

`prompt_version` is the strategy version *plus a hash of the rendered prompt*,
so this key is not merely an address — it is a statement that the input was
byte-identical. Edit a strategy and every affected key changes, which is the
behavior we want: a stale result must not be silently reused under a new prompt.

Three rules decide reusability, and each exists because the alternative would
corrupt the experiment:

1. **Infrastructure failures are never reused.** They measured the billing
   account, not the model. They are exactly the calls a resume should retry.
2. **Refusals and empty responses are reused.** Those are real model behavior
   and belong in the denominator. Retrying them would quietly resample until
   the model behaved, which is p-hacking with extra steps.
3. **Mock records never satisfy a real run**, and real records are never
   discarded by a mock run. Mixing the two would let scripted text masquerade
   as evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from evaluation.schemas import ErrorKind, EvalRecord, Scenario

#: (model, prompt_strategy, scenario_id, prompt_version)
ResultKey = tuple[str, str, str, str]

#: Model-name prefixes produced by test doubles rather than a paid provider.
MOCK_MODEL_PREFIXES = ("mock:", "scripted:", "failing:")


def is_mock_record(record: EvalRecord) -> bool:
    """True when this record came from a test double, not a paid provider."""
    return record.model.lower().startswith(MOCK_MODEL_PREFIXES)


def read_records(path: str | Path) -> tuple[list[EvalRecord], int, int]:
    """Read a transcript file, tolerating lines truncated by a hard kill.

    Returns `(records, lines_seen, malformed)`. Unlike `ResumeIndex`, this keeps
    **every** record including infrastructure failures — completeness reporting
    has to be able to see the calls that failed, or a cell wiped out by an
    exhausted quota would silently disappear from the accounting instead of
    being reported as unmeasured.
    """
    path = Path(path)
    if not path.exists():
        return [], 0, 0

    records: list[EvalRecord] = []
    malformed = 0
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                records.append(EvalRecord.model_validate(json.loads(line)))
            except Exception:  # noqa: BLE001 — a partial line is not fatal
                malformed += 1
    return records, total, malformed


def record_key(record: EvalRecord) -> ResultKey:
    return (
        record.model,
        record.prompt_strategy,
        record.scenario_id,
        record.prompt_version,
    )


@dataclass(frozen=True)
class ResumeStats:
    """What the index found on disk, for an operator to sanity-check."""

    records_on_disk: int = 0
    reusable: int = 0
    retry_infrastructure: int = 0
    skipped_mock: int = 0
    skipped_malformed: int = 0
    by_cell: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.records_on_disk:
            return "no prior results found — this is a cold run"
        parts = [
            f"{self.records_on_disk} prior records",
            f"{self.reusable} reusable",
            f"{self.retry_infrastructure} to retry (infrastructure)",
        ]
        if self.skipped_mock:
            parts.append(f"{self.skipped_mock} mock (ignored)")
        if self.skipped_malformed:
            parts.append(f"{self.skipped_malformed} unreadable (ignored)")
        return ", ".join(parts)


class ResumeIndex:
    """Prior results, keyed so a runner can ask 'do I still owe this call?'."""

    def __init__(
        self,
        records: Iterable[EvalRecord] = (),
        *,
        allow_mock: bool = False,
        stats: ResumeStats | None = None,
    ):
        self.allow_mock = allow_mock
        self._reusable: dict[ResultKey, EvalRecord] = {}
        self._retry: set[ResultKey] = set()
        counts: Counter[str] = Counter()
        skipped_mock = 0

        for record in records:
            if is_mock_record(record) is not allow_mock:
                # A real run ignores mock rows; a mock run ignores real ones.
                skipped_mock += 1
                continue
            key = record_key(record)
            if record.error_kind is ErrorKind.INFRASTRUCTURE:
                self._retry.add(key)
                continue
            # Later records win: a retry that succeeded supersedes the failure.
            self._reusable[key] = record
            self._retry.discard(key)
            counts[f"{record.model} | {record.prompt_strategy}"] += 1

        base = stats or ResumeStats()
        self.stats = ResumeStats(
            records_on_disk=base.records_on_disk,
            reusable=len(self._reusable),
            retry_infrastructure=len(self._retry),
            skipped_mock=skipped_mock,
            skipped_malformed=base.skipped_malformed,
            by_cell=dict(counts),
        )

    # ------------------------------------------------------------- construction

    @classmethod
    def from_file(
        cls, path: str | Path, *, allow_mock: bool = False
    ) -> "ResumeIndex":
        """Load prior records, tolerating a file truncated by a hard kill."""
        records, total, malformed = read_records(path)
        return cls(
            records,
            allow_mock=allow_mock,
            stats=ResumeStats(records_on_disk=total, skipped_malformed=malformed),
        )

    @classmethod
    def empty(cls, *, allow_mock: bool = False) -> "ResumeIndex":
        return cls((), allow_mock=allow_mock)

    # ----------------------------------------------------------------- querying

    def get(self, key: ResultKey) -> EvalRecord | None:
        return self._reusable.get(key)

    def has(self, key: ResultKey) -> bool:
        return key in self._reusable

    def __len__(self) -> int:
        return len(self._reusable)

    def __iter__(self) -> Iterator[EvalRecord]:
        return iter(self._reusable.values())

    def partition(
        self,
        scenarios: Sequence[Scenario],
        *,
        model: str,
        strategy: str,
        prompt_version_for: "callable",
    ) -> tuple[list[EvalRecord], list[Scenario]]:
        """Split scenarios into (already done, still owed) for one cell.

        `prompt_version_for` renders a scenario's prompt version. It is a
        callable rather than a constant because the version embeds a hash of the
        rendered prompt, which varies per scenario.
        """
        done: list[EvalRecord] = []
        todo: list[Scenario] = []
        for scenario in scenarios:
            key = (model, strategy, scenario.id, prompt_version_for(scenario))
            existing = self._reusable.get(key)
            if existing is not None:
                done.append(existing)
            else:
                todo.append(scenario)
        return done, todo


__all__ = [
    "MOCK_MODEL_PREFIXES",
    "ResultKey",
    "ResumeIndex",
    "ResumeStats",
    "is_mock_record",
    "read_records",
    "record_key",
]
