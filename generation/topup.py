"""Top-up sizing — how many more candidates to buy, from the observed rate.

The Dataset V1 plan deliberately refuses to guess a single candidate count. It
buys a first tranche, measures the acceptance rate the real teacher and the real
gate actually produce, and sizes the remainder from that measurement.

    shortfall  = target - accepted
    additional = ceil(shortfall / observed_acceptance_rate) * (1 + margin)

The margin exists because acceptance rate is itself estimated from a finite
sample, so sizing exactly at the point estimate lands short about half the time.
It is small and explicit rather than a doubling "to be safe" — overshooting
costs money and biases the final distribution toward whatever the top-up
tranche happened to contain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Safety margin on the top-up. 10% covers ordinary sampling wobble in the
#: observed rate without materially overbuying.
DEFAULT_MARGIN = 0.10

#: Never buy a tranche smaller than this; per-run overhead makes it wasteful.
MIN_TRANCHE = 50


@dataclass(frozen=True)
class TopUp:
    """A sized top-up tranche, with the arithmetic kept inspectable."""

    target: int
    accepted: int
    observed_rate: float
    shortfall: int
    raw_estimate: int
    margin: float
    additional_candidates: int
    needed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_accepted": self.target,
            "accepted_so_far": self.accepted,
            "observed_acceptance_rate": round(self.observed_rate, 4),
            "shortfall": self.shortfall,
            "raw_estimate": self.raw_estimate,
            "safety_margin": self.margin,
            "additional_candidates": self.additional_candidates,
            "needed": self.needed,
            "reason": self.reason,
        }


def plan_topup(
    *,
    target: int,
    accepted: int,
    observed_rate: float,
    margin: float = DEFAULT_MARGIN,
    min_tranche: int = MIN_TRANCHE,
    round_to: int = 50,
) -> TopUp:
    """Size the next tranche from the measured acceptance rate.

    Returns `needed=False` when the target is already met. Raises when the
    observed rate is zero or negative: that means the gate rejected everything,
    which is a signal to investigate the pipeline, not to buy more candidates.
    """
    shortfall = max(0, target - accepted)
    if shortfall == 0:
        return TopUp(
            target=target, accepted=accepted, observed_rate=observed_rate,
            shortfall=0, raw_estimate=0, margin=margin,
            additional_candidates=0, needed=False,
            reason=f"target met: {accepted} >= {target}",
        )

    if observed_rate <= 0:
        raise ValueError(
            "observed acceptance rate is 0 — every candidate was rejected. "
            "Generating more would buy more rejections; investigate the gate "
            "or the teacher before spending again."
        )
    if observed_rate > 1:
        raise ValueError(f"acceptance rate {observed_rate} exceeds 1.0")

    raw = math.ceil(shortfall / observed_rate)
    padded = math.ceil(raw * (1 + margin))
    rounded = int(math.ceil(padded / round_to) * round_to) if round_to > 1 else padded
    additional = max(rounded, min_tranche)

    return TopUp(
        target=target, accepted=accepted, observed_rate=observed_rate,
        shortfall=shortfall, raw_estimate=raw, margin=margin,
        additional_candidates=additional, needed=True,
        reason=(
            f"need {shortfall} more accepted at an observed rate of "
            f"{observed_rate:.3f} -> {raw} candidates, "
            f"+{int(margin * 100)}% margin, rounded to {round_to}"
        ),
    )


__all__ = ["DEFAULT_MARGIN", "MIN_TRANCHE", "TopUp", "plan_topup"]
