"""Distribution balancing.

Teachers drift: given free rein they produce mostly `normal` pressure, mostly
easy loop bugs, mostly Python. A dataset skewed that way trains a model that
holds the behavior in the common case and folds under exactly the pressure the
prompt-ceiling experiment showed to be the problem.

Balancing here is a per-axis share cap rather than a full cross-product quota.
With ~600 examples and four axes the cross product has hundreds of cells, so
quota-per-cell would reject almost everything; capping each axis independently
keeps the marginals honest while retaining most of the data.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

from generation.schemas import GeneratedExample

#: Maximum share of the final dataset any single bucket on an axis may occupy.
DEFAULT_CAPS: dict[str, float] = {
    "language": 0.62,
    "bug_category": 0.14,
    "pressure_type": 0.28,
    "difficulty": 0.50,
}

AXES: dict[str, Callable[[GeneratedExample], str]] = {
    "language": lambda e: e.scenario.language.value,
    "bug_category": lambda e: e.scenario.bug_category,
    "pressure_type": lambda e: e.scenario.pressure_type.value,
    "difficulty": lambda e: e.scenario.difficulty.value,
}


@dataclass
class BalanceResult:
    kept: list[GeneratedExample] = field(default_factory=list)
    dropped: list[GeneratedExample] = field(default_factory=list)
    caps: dict[str, int] = field(default_factory=dict)
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return len(self.dropped)


def distribution(
    examples: Sequence[GeneratedExample],
) -> dict[str, dict[str, int]]:
    """Counts per bucket on every balancing axis."""
    return {
        axis: dict(Counter(accessor(e) for e in examples).most_common())
        for axis, accessor in AXES.items()
    }


def balance(
    examples: Sequence[GeneratedExample],
    *,
    caps: dict[str, float] | None = None,
    target_size: int | None = None,
) -> BalanceResult:
    """Greedily keep examples while no axis bucket exceeds its share cap.

    Deterministic in input order. Examples dropped here are recorded with the
    `UNBALANCED` code rather than deleted, so the report can show what the cap
    cost and a later dataset version can loosen it.
    """
    caps = caps or DEFAULT_CAPS
    size = target_size or len(examples)
    if size == 0:
        return BalanceResult()

    absolute_caps = {
        axis: max(1, math.ceil(share * size)) for axis, share in caps.items()
    }

    counts: dict[str, Counter] = defaultdict(Counter)
    result = BalanceResult(caps=absolute_caps)

    for example in examples:
        over: list[str] = []
        for axis, accessor in AXES.items():
            if axis not in caps:
                continue
            bucket = accessor(example)
            if counts[axis][bucket] >= absolute_caps[axis]:
                over.append(f"{axis}={bucket}")

        if over:
            result.dropped.append(
                example.model_copy(
                    update={
                        "accepted": False,
                        "rejection_codes": ("UNBALANCED",),
                        "gate_notes": f"share cap reached for {', '.join(over)}",
                    }
                )
            )
            continue

        for axis, accessor in AXES.items():
            if axis in caps:
                counts[axis][accessor(example)] += 1
        result.kept.append(example)

    result.distributions = distribution(result.kept)
    return result


def conversation_length_distribution(
    examples: Sequence[GeneratedExample],
) -> dict[str, int]:
    return dict(
        sorted(Counter(str(e.scenario.turn_count) for e in examples).items())
    )


__all__ = [
    "AXES",
    "BalanceResult",
    "DEFAULT_CAPS",
    "balance",
    "conversation_length_distribution",
    "distribution",
]
