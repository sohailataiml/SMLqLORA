"""API usage accounting — what a run actually consumed.

The prompt-ceiling ablation could report call counts but not tokens, because
`ModelResponse.usage` was populated by the providers and then discarded. Data
generation is the first phase expensive enough that "we spent roughly this
much" is not a good enough answer, so usage is metered here and written into the
run artifacts.

The design goal is that metering cannot change behavior. `MeteredAdapter` wraps
an existing adapter and delegates; it does not retry, transform, or suppress
anything. Removing the wrapper leaves the pipeline byte-identical.

Cost is deliberately **not** computed unless prices are supplied. Inventing a
dollar figure from a half-remembered price list would be worse than reporting
tokens and saying `NOT COMPUTED`.

    meter = UsageMeter()
    model = MeteredAdapter(resolve_model("anthropic:claude-opus-5"), meter)
    ...
    meter.totals()
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from models.adapters import GenerationParams, Message, ModelAdapter, ModelResponse

#: Token keys providers report under. Anthropic and OpenAI are normalized onto
#: `input_tokens` / `output_tokens` by their adapters; cache keys differ.
_INPUT_KEYS = ("input_tokens", "prompt_tokens")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens")
_CACHE_READ_KEYS = ("cache_read_input_tokens", "cached_tokens")
_CACHE_WRITE_KEYS = ("cache_creation_input_tokens",)


def _first_int(usage: dict[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


@dataclass
class ModelUsage:
    """Everything one model consumed during a run."""

    model: str
    family: str = "unknown"
    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: True when at least one response carried no usage block, so the token
    #: totals below are a floor rather than an exact count.
    incomplete: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "family": self.family,
            "requests": self.requests,
            "failed_requests": self.failed_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "token_counts_incomplete": self.incomplete,
        }


class UsageMeter:
    """Thread-safe accumulation of provider usage across a run.

    Generation and filtering both run on thread pools, so every mutation is
    guarded. The meter never raises: a provider that reports usage in an
    unexpected shape yields a zero contribution and sets `incomplete`, rather
    than aborting a run that has already been paid for.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, ModelUsage] = {}

    def record(
        self,
        model: str,
        family: str,
        usage: dict[str, Any] | None,
        *,
        ok: bool = True,
    ) -> None:
        usage = usage or {}
        with self._lock:
            entry = self._by_model.get(model)
            if entry is None:
                entry = ModelUsage(model=model, family=family)
                self._by_model[model] = entry
            entry.requests += 1
            if not ok:
                entry.failed_requests += 1
            if not usage:
                # A failed call legitimately has no usage block; only successful
                # calls with no usage make the totals a floor.
                if ok:
                    entry.incomplete = True
                return
            entry.input_tokens += _first_int(usage, _INPUT_KEYS)
            entry.output_tokens += _first_int(usage, _OUTPUT_KEYS)
            entry.cache_read_tokens += _first_int(usage, _CACHE_READ_KEYS)
            entry.cache_write_tokens += _first_int(usage, _CACHE_WRITE_KEYS)

    def by_model(self) -> list[ModelUsage]:
        with self._lock:
            return sorted(self._by_model.values(), key=lambda u: u.model)

    def totals(self) -> dict[str, Any]:
        """Per-model usage plus a grand total, ready to serialize."""
        entries = self.by_model()
        return {
            "by_model": [e.to_dict() for e in entries],
            "totals": {
                "requests": sum(e.requests for e in entries),
                "failed_requests": sum(e.failed_requests for e in entries),
                "input_tokens": sum(e.input_tokens for e in entries),
                "output_tokens": sum(e.output_tokens for e in entries),
                "cache_read_tokens": sum(e.cache_read_tokens for e in entries),
                "cache_write_tokens": sum(e.cache_write_tokens for e in entries),
                "total_tokens": sum(e.total_tokens for e in entries),
                "token_counts_incomplete": any(e.incomplete for e in entries),
            },
            "estimated_cost_usd": "NOT COMPUTED — no price table is configured",
        }

    def merge(self, other: "UsageMeter") -> None:
        """Fold another meter in, for a run assembled from several tranches."""
        for entry in other.by_model():
            with self._lock:
                mine = self._by_model.get(entry.model)
                if mine is None:
                    self._by_model[entry.model] = ModelUsage(**vars(entry))
                    continue
                mine.requests += entry.requests
                mine.failed_requests += entry.failed_requests
                mine.input_tokens += entry.input_tokens
                mine.output_tokens += entry.output_tokens
                mine.cache_read_tokens += entry.cache_read_tokens
                mine.cache_write_tokens += entry.cache_write_tokens
                mine.incomplete = mine.incomplete or entry.incomplete


def merged_totals(meters: Sequence[UsageMeter]) -> dict[str, Any]:
    """Combined totals across several meters (e.g. teacher + judge)."""
    combined = UsageMeter()
    for meter in meters:
        combined.merge(meter)
    return combined.totals()


class MeteredAdapter(ModelAdapter):
    """Delegating wrapper that records usage. Changes no behavior.

    Wrapping at `_generate` rather than `generate` is deliberate: the base
    class keeps ownership of error conversion, latency stamping and message
    validation, so a metered adapter and a bare one take exactly the same code
    path through `ModelAdapter.generate`.
    """

    def __init__(self, inner: ModelAdapter, meter: UsageMeter):
        super().__init__(inner.name, inner.family, inner.revision)
        self.inner = inner
        self.meter = meter

    def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        try:
            response = self.inner._generate(messages, system, params)
        except Exception:
            # The base class turns this into an error response; record the
            # request so failed calls are visible in the accounting.
            self.meter.record(self.name, self.family, None, ok=False)
            raise
        self.meter.record(
            self.name, self.family, response.usage, ok=response.ok
        )
        return response

    def describe(self) -> dict[str, str]:
        return self.inner.describe()


__all__ = [
    "MeteredAdapter",
    "ModelUsage",
    "UsageMeter",
    "merged_totals",
]
