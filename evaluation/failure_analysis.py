"""Failure-mode analysis — what does the strongest prompt still get wrong?

The prompt-ceiling gate answers *whether* to fine-tune. This module answers the
follow-up question that actually shapes the dataset: **which behaviors survive
the best prompt, and under which pressure?**

Everything here is pure: records in, numbers out. No model calls, no disk. That
keeps the analysis unit-testable and lets it be re-derived from stored
transcripts at any time without spending anything.

Two conventions are load-bearing:

* **Only measured records count.** Infrastructure failures describe the billing
  account, not the model, and never enter a numerator or a denominator.
* **Small cells are labelled, not hidden.** A pressure type with three
  observations gets reported with `n=3` and an `underpowered` flag rather than
  a confident-looking percentage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from evaluation.schemas import EvalRecord

#: Below this many observations a rate is reported but flagged as underpowered.
#: Chosen to match the spec's own `min_scenarios_per_cell` intent: fewer than a
#: handful of observations cannot distinguish a real effect from sampling noise.
UNDERPOWERED_THRESHOLD = 10

#: The failure codes the brief asks to analyze explicitly. Others still appear
#: in the output; these are guaranteed to appear even at a count of zero, so a
#: mode that never fired is visibly absent rather than silently missing.
TRACKED_CODES: tuple[str, ...] = (
    "SOLUTION_LEAK",
    "EXPLICIT_FINAL_DIAGNOSIS",
    "MULTIPLE_HINTS",
    "IRRELEVANT_HINT",
    "OVER_EXPLANATION",
    "PREMATURE_CONFIRMATION",
    "FAILED_TO_ADAPT",
)

#: Strategies treated as "strong prompting" when isolating residual difficulty.
#: Zero-shot is excluded on purpose: it measures the absence of prompting, and
#: including it would let easy-to-prompt-away failures dominate the analysis.
STRONG_STRATEGIES: tuple[str, ...] = ("few_shot", "structured_system_prompt")


def measured(records: Iterable[EvalRecord]) -> list[EvalRecord]:
    """Records where the model was actually observed."""
    return [r for r in records if r.was_evaluated]


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


@dataclass(frozen=True)
class GroupStat:
    """Counts for one slice of the data (a model, a strategy, a pressure type)."""

    key: str
    n: int
    count: int

    @property
    def rate(self) -> float:
        return _rate(self.count, self.n)

    @property
    def underpowered(self) -> bool:
        return self.n < UNDERPOWERED_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "n": self.n,
            "count": self.count,
            "rate": self.rate,
            "underpowered": self.underpowered,
        }


@dataclass(frozen=True)
class FailureModeStat:
    """Everything measured about one failure code."""

    code: str
    n_measured: int
    count: int
    by_model: list[GroupStat] = field(default_factory=list)
    by_strategy: list[GroupStat] = field(default_factory=list)
    by_pressure_type: list[GroupStat] = field(default_factory=list)
    representative_scenarios: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return _rate(self.count, self.n_measured)

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_code": self.code,
            "count": self.count,
            "n_measured": self.n_measured,
            "rate": self.rate,
            "by_model": [g.to_dict() for g in self.by_model],
            "by_prompt_strategy": [g.to_dict() for g in self.by_strategy],
            "by_pressure_type": [g.to_dict() for g in self.by_pressure_type],
            "representative_scenario_ids": self.representative_scenarios,
        }


def _group(
    records: Sequence[EvalRecord], code: str, attr: str
) -> list[GroupStat]:
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for record in records:
        key = getattr(record, attr)
        key = key.value if hasattr(key, "value") else str(key)
        totals[key] += 1
        if code in record.failure_reasons:
            hits[key] += 1
    return sorted(
        (GroupStat(k, totals[k], hits[k]) for k in totals),
        key=lambda g: (-g.rate, g.key),
    )


def analyze_failure_mode(
    records: Sequence[EvalRecord], code: str, *, examples: int = 5
) -> FailureModeStat:
    """Full breakdown for one failure code across every reported dimension."""
    hits = [r for r in records if code in r.failure_reasons]
    return FailureModeStat(
        code=code,
        n_measured=len(records),
        count=len(hits),
        by_model=_group(records, code, "model"),
        by_strategy=_group(records, code, "prompt_strategy"),
        by_pressure_type=_group(records, code, "pressure_type"),
        # Representative rather than cherry-picked: sorted by id so the same
        # records are cited every time the analysis is re-derived.
        representative_scenarios=sorted({r.scenario_id for r in hits})[:examples],
    )


def analyze_failure_modes(
    records: Sequence[EvalRecord], *, codes: Sequence[str] = TRACKED_CODES
) -> dict[str, object]:
    """The Step-8 report: every tracked code, plus any others that fired."""
    obs = measured(records)
    if not obs:
        raise ValueError(
            "No measured records. Every call failed for infrastructure reasons, "
            "so there is nothing to analyze."
        )

    observed_codes = {c for r in obs for c in r.failure_reasons}
    all_codes = list(dict.fromkeys([*codes, *sorted(observed_codes)]))
    stats = [analyze_failure_mode(obs, code) for code in all_codes]

    failures = [r for r in obs if not r.passed]
    strong = [r for r in obs if r.prompt_strategy in STRONG_STRATEGIES]

    return {
        "n_measured": len(obs),
        "n_failed": len(failures),
        "overall_pass_rate": _rate(len(obs) - len(failures), len(obs)),
        "underpowered_threshold": UNDERPOWERED_THRESHOLD,
        "failure_modes": [s.to_dict() for s in stats],
        "residual_under_strong_prompts": _residual(strong),
        "pressure_type_ranking": pressure_ranking(obs),
    }


def _residual(strong: Sequence[EvalRecord]) -> dict[str, object]:
    """What survives few-shot and structured prompting — the dataset's target."""
    if not strong:
        return {"n_measured": 0, "note": "no strong-prompt records measured"}
    failed = [r for r in strong if not r.passed]
    counts = Counter(c for r in failed for c in r.failure_reasons)
    return {
        "strategies": list(STRONG_STRATEGIES),
        "n_measured": len(strong),
        "n_failed": len(failed),
        "pass_rate": _rate(len(strong) - len(failed), len(strong)),
        "surviving_failure_modes": dict(counts.most_common()),
        "pressure_type_ranking": pressure_ranking(strong),
    }


def pressure_ranking(records: Sequence[EvalRecord]) -> list[dict[str, object]]:
    """Pressure types ordered worst-first by pass rate."""
    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pressure_type.value].append(record)

    rows = []
    for pressure, group in grouped.items():
        passes = sum(1 for r in group if r.passed)
        leaks = sum(1 for r in group if "SOLUTION_LEAK" in r.failure_reasons)
        rows.append({
            "pressure_type": pressure,
            "n": len(group),
            "passes": passes,
            "pass_rate": _rate(passes, len(group)),
            "failure_rate": _rate(len(group) - passes, len(group)),
            "solution_leak_rate": _rate(leaks, len(group)),
            "underpowered": len(group) < UNDERPOWERED_THRESHOLD,
        })
    return sorted(rows, key=lambda r: (r["pass_rate"], r["pressure_type"]))


# =============================================================================
# Proposed training distribution (Step 11)
# =============================================================================

#: Every dimension the generator can vary. Each keeps at least `FLOOR_SHARE` so
#: the dataset never loses coverage of a behavior we still have to hold.
TRAINING_DIMENSIONS: tuple[str, ...] = (
    "normal",
    "frustrated",
    "repeated_answer_request",
    "time_pressure",
    "prompt_injection",
    "authority_override",
    "fake_success",
    "almost_correct",
    "solved",
)

#: Guaranteed minimum share per dimension. Nine dimensions x 4% = 36% floor,
#: leaving 64% to be allocated by measured difficulty.
FLOOR_SHARE = 0.04

#: No single dimension may exceed this, however badly it scored. A dataset that
#: is 60% one pressure type teaches that pressure type, not the behavior.
CAP_SHARE = 0.22

#: `normal` gets a much higher floor than the adversarial dimensions, for two
#: reasons that pure failure-rate allocation cannot see:
#:
#: 1. Every adversarial dimension is a *perturbation of* the normal case. A
#:    dataset that is 6% normal teaches a model to resist pressure without ever
#:    teaching it the base behavior being defended.
#: 2. The failure rates driving this allocation were measured on **frontier**
#:    models under strong prompts. The student is a 1.7B model, which will fail
#:    far more broadly and on much easier inputs. Frontier difficulty is a guide
#:    to *relative* emphasis among the hard cases, not evidence that the base
#:    case is already solved for a model three orders of magnitude smaller.
NORMAL_FLOOR_SHARE = 0.15


def propose_training_distribution(
    records: Sequence[EvalRecord],
    *,
    floor: float = FLOOR_SHARE,
    cap: float = CAP_SHARE,
    normal_floor: float = NORMAL_FLOOR_SHARE,
    dimensions: Sequence[str] = TRAINING_DIMENSIONS,
) -> dict[str, object]:
    """Allocate dataset share from *measured* failure mass, not intuition.

    The rule, stated so it can be argued with:

    1. Every dimension receives `floor` share, guaranteeing coverage; `normal`
       receives `normal_floor` instead (see `NORMAL_FLOOR_SHARE` for why).
    2. The remaining budget is split in proportion to each dimension's observed
       failure rate **under strong prompts only** — that is where the residual
       difficulty lives, and it is what fine-tuning has to fix.
    3. No dimension exceeds `cap`; overflow is redistributed to the others.
    4. A dimension never observed under strong prompts receives the floor and is
       flagged, rather than being silently dropped or silently inflated.
    """
    obs = measured(records)
    strong = [r for r in obs if r.prompt_strategy in STRONG_STRATEGIES]
    basis = strong or obs

    stats: dict[str, dict[str, object]] = {}
    weights: dict[str, float] = {}
    for dim in dimensions:
        group = [r for r in basis if r.pressure_type.value == dim]
        failures = sum(1 for r in group if not r.passed)
        rate = _rate(failures, len(group))
        stats[dim] = {
            "n_observed": len(group),
            "failures": failures,
            "failure_rate": rate,
            "underpowered": len(group) < UNDERPOWERED_THRESHOLD,
            "observed": bool(group),
        }
        weights[dim] = rate

    floors = {d: (normal_floor if d == "normal" else floor) for d in dimensions}
    budget = 1.0 - sum(floors.values())
    if budget < 0:
        raise ValueError(
            f"floors sum to {sum(floors.values()):.2f}, which exceeds 100%"
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        # Nothing failed anywhere: fall back to an even split rather than
        # inventing a preference the data does not support.
        shares = {d: 1.0 / len(dimensions) for d in dimensions}
    else:
        shares = {
            d: floors[d] + budget * (w / total_weight) for d, w in weights.items()
        }
        shares = _apply_cap(shares, cap=cap, floor=floor)

    return {
        "basis": (
            "failure rate under strong prompts "
            f"({', '.join(STRONG_STRATEGIES)})"
            if strong
            else "failure rate across all measured records (no strong-prompt "
                 "records available)"
        ),
        "n_records_in_basis": len(basis),
        "rule": {
            "floor_share_per_dimension": floor,
            "floor_share_normal": normal_floor,
            "cap_share_per_dimension": cap,
            "allocation": "floor + proportional to measured failure rate",
            "why_normal_has_a_higher_floor": (
                "Every adversarial dimension is a perturbation of the normal "
                "case, and these failure rates were measured on frontier models "
                "under strong prompts. The student is a 1.7B model that will "
                "fail far more broadly, so frontier difficulty guides relative "
                "emphasis among hard cases; it is not evidence that the base "
                "case is already solved."
            ),
        },
        "measured_inputs": stats,
        "distribution": {d: round(s, 4) for d, s in sorted(
            shares.items(), key=lambda kv: -kv[1]
        )},
        "distribution_sums_to": round(sum(shares.values()), 6),
    }


def _apply_cap(
    shares: dict[str, float], *, cap: float, floor: float
) -> dict[str, float]:
    """Clip to `cap` and redistribute the overflow to uncapped dimensions."""
    shares = dict(shares)
    for _ in range(len(shares)):
        over = {d: s - cap for d, s in shares.items() if s > cap}
        if not over:
            break
        overflow = sum(over.values())
        for d in over:
            shares[d] = cap
        receivers = {d: s for d, s in shares.items() if s < cap}
        if not receivers:
            break
        room = sum(cap - s for s in receivers.values())
        for d, s in receivers.items():
            shares[d] = s + overflow * ((cap - s) / room) if room else s
    return shares


__all__ = [
    "CAP_SHARE",
    "FLOOR_SHARE",
    "FailureModeStat",
    "GroupStat",
    "STRONG_STRATEGIES",
    "TRACKED_CODES",
    "TRAINING_DIMENSIONS",
    "UNDERPOWERED_THRESHOLD",
    "analyze_failure_mode",
    "analyze_failure_modes",
    "measured",
    "pressure_ranking",
    "propose_training_distribution",
]
