"""Provider-independent model interface.

The evaluator must not care whether a response came from a frontier API, a local
Hugging Face checkpoint, or a scripted test double. Everything it needs is::

    response = model.generate(messages, system=...)

Model specs are strings so experiments can be driven from the CLI:

    anthropic:claude-opus-5
    openai:gpt-5
    hf:Qwen/Qwen3-1.7B
    peft:Qwen/Qwen3-1.7B+outputs/socratic-v1
    mock:scripted
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from evaluation.schemas import Message, Role

# =============================================================================
# Errors — each explains how to fix the problem, not just that one exists.
# =============================================================================


class ModelError(RuntimeError):
    """Base class for adapter problems."""


class MissingCredentialsError(ModelError):
    def __init__(self, provider: str, env_var: str):
        super().__init__(
            f"No credentials for provider {provider!r}. Set {env_var} in your "
            f"environment or in a .env file (see .env.example). "
            f"Experiments requiring this provider cannot run without it."
        )
        self.provider = provider
        self.env_var = env_var


class InferenceError(ModelError):
    """A local model failed while generating - OOM, device assert, decode error.

    Distinct from a bad response: no response exists at all. The evaluator must
    treat this as infrastructure, because scoring it as behavior lets a GPU
    problem masquerade as a model that answered nothing.
    """


class MissingDependencyError(ModelError):
    def __init__(self, package: str, extra: str):
        super().__init__(
            f"The {package!r} package is required but not installed. "
            f"Install it with:  pip install -e '.[{extra}]'"
        )


class UnsupportedProviderError(ModelError):
    def __init__(self, provider: str, known: Sequence[str]):
        super().__init__(
            f"Unsupported model provider {provider!r}. "
            f"Known providers: {', '.join(sorted(known))}. "
            f"Model specs look like 'anthropic:claude-opus-5' or 'hf:Qwen/Qwen3-1.7B'."
        )


class ModelNotFoundError(ModelError):
    pass


# =============================================================================
# Generation parameters
# =============================================================================


@dataclass(frozen=True)
class GenerationParams:
    """Decoding settings. Recorded verbatim in every transcript."""

    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": list(self.stop),
            **({"extra": self.extra} if self.extra else {}),
        }


# Evaluation defaults: as close to deterministic as each provider allows. Recent
# Anthropic models reject sampling parameters outright (see SAMPLING_UNSUPPORTED
# below), so `temperature=None` means "use the provider default".
EVAL_PARAMS = GenerationParams(max_tokens=800, temperature=0.0, seed=1234)


@dataclass(frozen=True)
class ModelResponse:
    """One completion, plus everything needed to reproduce and audit it."""

    text: str
    model: str
    revision: str = "unknown"
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# =============================================================================
# Adapter base
# =============================================================================


class ModelAdapter(ABC):
    """Common interface implemented by every backend."""

    #: Human-readable identifier, e.g. "anthropic:claude-opus-5".
    name: str
    #: Model family, used to enforce "at least two families" in the ablation.
    family: str
    #: Pinned revision (API snapshot, HF commit sha, or adapter checkpoint hash).
    revision: str

    def __init__(self, name: str, family: str, revision: str = "unknown"):
        self.name = name
        self.family = family
        self.revision = revision

    @abstractmethod
    def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        """Backend-specific call. Exceptions here are converted by `generate`."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        params: GenerationParams | None = None,
        raise_on_error: bool = False,
    ) -> ModelResponse:
        """Generate one response.

        Errors are returned as a `ModelResponse` with `error` set rather than
        raised, so a single flaky call does not abort a 180-cell experiment. The
        error is recorded in the transcript and counted in the metrics.
        """
        params = params or EVAL_PARAMS
        self._validate_messages(messages)
        started = time.perf_counter()
        try:
            response = self._generate(messages, system, params)
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            if raise_on_error:
                raise
            return ModelResponse(
                text="",
                model=self.name,
                revision=self.revision,
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        if response.latency_s == 0.0:
            response = replace(response, latency_s=time.perf_counter() - started)
        return response

    @staticmethod
    def _validate_messages(messages: Sequence[Message]) -> None:
        if not messages:
            raise ValueError("generate() requires at least one message")
        if messages[0].role is not Role.USER:
            raise ValueError(
                f"conversation must start with a user turn, got "
                f"{messages[0].role.value!r}"
            )
        if messages[-1].role is not Role.USER:
            raise ValueError(
                "conversation must end with a user turn (the learner's message)"
            )

    def describe(self) -> dict[str, str]:
        return {"model": self.name, "family": self.family, "revision": self.revision}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name} rev={self.revision}>"


# =============================================================================
# Test doubles — no network, fully deterministic.
# =============================================================================


class ScriptedAdapter(ModelAdapter):
    """Returns canned responses. The backbone of the offline test suite."""

    def __init__(
        self,
        responses: Sequence[str] | Callable[[Sequence[Message]], str],
        *,
        name: str = "mock:scripted",
        family: str = "mock",
        revision: str = "scripted-1",
    ):
        super().__init__(name=name, family=family, revision=revision)
        self._responses = responses
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    def _generate(self, messages, system, params) -> ModelResponse:
        if callable(self._responses):
            text = self._responses(list(messages))
        else:
            if not self._responses:
                raise ModelError("ScriptedAdapter was given no responses")
            text = self._responses[self._calls % len(self._responses)]
        self._calls += 1
        return ModelResponse(
            text=text,
            model=self.name,
            revision=self.revision,
            usage={"input_tokens": 0, "output_tokens": 0},
            raw={"system": system, "params": params.to_dict()},
        )


class FailingAdapter(ModelAdapter):
    """Always errors. Used to test error accounting in the evaluator."""

    def __init__(self, message: str = "simulated provider outage"):
        super().__init__(name="mock:failing", family="mock", revision="failing-1")
        self._message = message

    def _generate(self, messages, system, params) -> ModelResponse:
        raise ModelError(self._message)


# =============================================================================
# Registry
# =============================================================================

_PROVIDERS: dict[str, Callable[..., ModelAdapter]] = {}


def register_provider(prefix: str, factory: Callable[..., ModelAdapter]) -> None:
    _PROVIDERS[prefix] = factory


def known_providers() -> list[str]:
    return sorted(_PROVIDERS)


def resolve_model(spec: str, **kwargs: Any) -> ModelAdapter:
    """Build an adapter from a spec string.

    Grammar::

        <provider>:<model-id>[@<revision>]

    `peft` takes a composite id: `peft:<base-model>+<adapter-path>`.
    """
    # Load the built-ins first so every error message can list what is available.
    _ensure_builtin_providers()

    if not isinstance(spec, str) or not spec.strip():
        raise UnsupportedProviderError(str(spec), known_providers())

    spec = spec.strip()
    # A bare Hugging Face repo id is the spelling a grader is handed - it is what
    # the submission document and the Hub page both show. Accepting it only in
    # eval.py meant the same id failed in the ablation runners, so normalization
    # belongs here, where every entry point goes through it.
    if ":" not in spec and "/" in spec and not spec.startswith((".", "/", "\\")):
        spec = f"hf:{spec}"
    if ":" not in spec:
        raise UnsupportedProviderError(spec, known_providers())

    provider, _, remainder = spec.partition(":")
    provider = provider.strip().lower()

    if provider not in _PROVIDERS:
        raise UnsupportedProviderError(provider, known_providers())

    model_id, _, revision = remainder.partition("@")
    if not model_id.strip():
        raise ModelNotFoundError(
            f"Model spec {spec!r} has no model id. Expected '<provider>:<model-id>'."
        )
    return _PROVIDERS[provider](
        model_id.strip(), revision=revision.strip() or None, **kwargs
    )


_builtins_loaded = False


def _ensure_builtin_providers() -> None:
    """Import provider modules lazily so unit tests never touch provider SDKs."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from models import providers  # noqa: F401  (registers anthropic/openai/mock)

    try:
        from models import local_hf  # noqa: F401  (registers hf/peft)
    except Exception:  # pragma: no cover - torch absent is the normal case
        pass


__all__ = [
    "EVAL_PARAMS",
    "FailingAdapter",
    "GenerationParams",
    "InferenceError",
    "MissingCredentialsError",
    "MissingDependencyError",
    "ModelAdapter",
    "ModelError",
    "ModelNotFoundError",
    "ModelResponse",
    "ScriptedAdapter",
    "UnsupportedProviderError",
    "known_providers",
    "register_provider",
    "resolve_model",
]
