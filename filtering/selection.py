"""Deterministic, distribution-preserving selection of the final dataset.

The quality gate decides what is *eligible*. When more examples pass than the
dataset targets, something still has to choose which ones ship, and "the first
600 in file order" is the wrong answer: file order is generation order, which
tracks the plan's seed sweep and would over-represent whatever the sampler
happened to emit early.

Selection here is:

* **stratified** on `pressure_type`, the axis the Dataset V1 plan specifies and
  the axis the prompt-ceiling experiment actually measured;
* **quota-allocated** by largest remainder, so integer rounding cannot silently
  drop a small stratum;
* **diversity-interleaved** within each stratum — candidates are drawn
  round-robin across bug categories, so a stratum dominated by one bug category
  in the pool does not become one bug category in the dataset;
* **deterministic** — the same pool and seed always yield the same dataset, and
  the ordering is keyed on content hashes rather than arrival order, so
  re-running generation with different concurrency cannot change the result.

Selection never relaxes quality: it only ever chooses among examples that have
already passed the full gate. Where a stratum cannot meet its quota, the
shortfall is reported rather than back-filled from an easier stratum.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from generation.schemas import GeneratedExample

#: Selection seed. Fixed and recorded in the freeze manifest; changing it
#: produces a different dataset and therefore a different version.
SELECTION_SEED = 20260818
SELECTION_METHOD = (
    "stratified on pressure_type; largest-remainder quotas from the Dataset V1 "
    "plan; within each stratum candidates ordered by content hash then seeded "
    "shuffle, drawn round-robin across bug categories for diversity"
)


@dataclass
class SelectionResult:
    selected: list[GeneratedExample] = field(default_factory=list)
    quotas: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    shortfalls: dict[str, int] = field(default_factory=dict)
    seed: int = SELECTION_SEED
    method: str = SELECTION_METHOD
    target: int = 0

    @property
    def size(self) -> int:
        return len(self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "selected": self.size,
            "seed": self.seed,
            "method": self.method,
            "quotas": self.quotas,
            "available_in_pool": self.available,
            "shortfalls": self.shortfalls,
            "shortfall_total": sum(self.shortfalls.values()),
        }


def largest_remainder(shares: dict[str, float], total: int) -> dict[str, int]:
    """Integer quotas summing exactly to `total`.

    Plain rounding of nine shares can miss the target by several examples and
    systematically starves the smallest stratum. Largest remainder distributes
    the rounding error to whoever was rounded down hardest.
    """
    if total <= 0 or not shares:
        return {k: 0 for k in shares}
    scale = sum(shares.values()) or 1.0
    exact = {k: (v / scale) * total for k, v in shares.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remaining = total - sum(floors.values())
    # Ties broken by key so the result never depends on dict ordering.
    order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in order[:remaining]:
        floors[key] += 1
    return floors


def _deterministic_order(
    examples: Sequence[GeneratedExample], seed: int
) -> list[GeneratedExample]:
    """Content-keyed order, independent of how candidates arrived."""
    ordered = sorted(examples, key=lambda e: e.content_hash())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def _round_robin_by_category(
    examples: Sequence[GeneratedExample], want: int, seed: int
) -> list[GeneratedExample]:
    """Draw `want` examples, cycling bug categories to spread diversity."""
    buckets: dict[str, list[GeneratedExample]] = defaultdict(list)
    for example in _deterministic_order(examples, seed):
        buckets[example.scenario.bug_category].append(example)

    keys = sorted(buckets)
    picked: list[GeneratedExample] = []
    index = 0
    while len(picked) < want and any(buckets[k] for k in keys):
        key = keys[index % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop(0))
        index += 1
    return picked[:want]


def select_balanced(
    pool: Sequence[GeneratedExample],
    target: int,
    shares: dict[str, float],
    *,
    seed: int = SELECTION_SEED,
) -> SelectionResult:
    """Choose ~`target` examples from `pool`, holding the planned mix.

    A stratum with fewer examples than its quota contributes everything it has
    and records a shortfall. The freed quota is **not** redistributed: doing so
    would quietly replace a missing `solved` example with an easier `normal`
    one and paper over exactly the coverage gap worth reporting.
    """
    by_pressure: dict[str, list[GeneratedExample]] = defaultdict(list)
    for example in pool:
        by_pressure[example.scenario.pressure_type.value].append(example)

    quotas = largest_remainder(shares, target)
    available = {k: len(by_pressure.get(k, [])) for k in sorted(shares)}

    selected: list[GeneratedExample] = []
    shortfalls: dict[str, int] = {}
    for pressure in sorted(shares):
        want = quotas.get(pressure, 0)
        have = by_pressure.get(pressure, [])
        if len(have) < want:
            shortfalls[pressure] = want - len(have)
            selected.extend(_deterministic_order(have, seed))
        else:
            selected.extend(_round_robin_by_category(have, want, seed))

    # Stable final ordering, again content-keyed rather than assembly-keyed.
    selected = _deterministic_order(selected, seed)

    return SelectionResult(
        selected=selected, quotas=quotas, available=available,
        shortfalls=shortfalls, seed=seed, target=target,
    )


__all__ = [
    "SELECTION_METHOD",
    "SELECTION_SEED",
    "SelectionResult",
    "largest_remainder",
    "select_balanced",
]
