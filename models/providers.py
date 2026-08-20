"""Frontier API adapters (Anthropic, OpenAI) behind the common interface.

Provider quirks live here and nowhere else. In particular:

* Recent Anthropic models (Opus 4.7+, Opus 5, Sonnet 5, Fable 5) **reject**
  ``temperature`` / ``top_p`` / ``top_k`` with a 400. The adapter drops them
  rather than letting an experiment fail halfway through.
* Those same models think by default, and ``max_tokens`` caps thinking *plus*
  visible text — so a tight token budget silently truncates the answer. The
  adapter raises the floor when thinking is active.
* OpenAI reasoning models similarly reject non-default sampling parameters. The
  adapter retries once without them instead of dropping the sample.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from models.adapters import (
    GenerationParams,
    MissingCredentialsError,
    MissingDependencyError,
    ModelAdapter,
    ModelError,
    ModelResponse,
    ScriptedAdapter,
    register_provider,
)
from models.credentials import resolve_credential_conflicts
from evaluation.schemas import Message, Role

# Anthropic model families that reject sampling parameters and think by default.
_ANTHROPIC_NO_SAMPLING = re.compile(
    r"^claude-(opus-5|opus-4-(7|8)|sonnet-5|fable-5|mythos-5)", re.I
)
# OpenAI reasoning families that reject non-default sampling parameters.
_OPENAI_NO_SAMPLING = re.compile(r"^(o\d|gpt-5)", re.I)

#: Floor for max_tokens when the provider may spend tokens on hidden reasoning.
THINKING_MAX_TOKENS_FLOOR = 4096


def _load_dotenv_once(env_var: str | None = None) -> None:
    """Load `.env`, then refuse to guess when it disagrees with the shell.

    `load_dotenv(override=False)` is the right precedence but an invisible one:
    a stale shell key silently outranks a funded key in `.env`, and every
    downstream verdict is then about the wrong account. When `env_var` is given
    and the two sources hold different values, this raises rather than let a
    paid call pick one on its own. With no conflict, nothing changes.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional
        pass
    else:
        load_dotenv(override=False)
    if env_var is not None:
        resolve_credential_conflicts([env_var])


def _split_system(
    messages: Sequence[Message], system: str | None
) -> tuple[str | None, list[dict[str, str]]]:
    """Separate a system prompt from the conversation turns."""
    system_parts = [system] if system else []
    turns: list[dict[str, str]] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            system_parts.append(msg.content)
        else:
            turns.append({"role": msg.role.value, "content": msg.content})
    combined = "\n\n".join(p for p in system_parts if p and p.strip()) or None
    return combined, turns


# =============================================================================
# Anthropic
# =============================================================================


class AnthropicAdapter(ModelAdapter):
    """Claude models via the official Anthropic SDK."""

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model_id: str = "claude-opus-5",
        revision: str | None = None,
        *,
        api_key: str | None = None,
        client: Any = None,
        thinking: dict[str, Any] | None = None,
        effort: str | None = None,
        max_retries: int = 3,
        timeout: float = 600.0,
    ):
        super().__init__(
            name=f"anthropic:{model_id}",
            family="anthropic",
            revision=revision or model_id,
        )
        self.model_id = model_id
        self.thinking = thinking
        self.effort = effort
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout

    # ------------------------------------------------------------------ setup

    @property
    def client(self):
        if self._client is None:
            _load_dotenv_once(self.ENV_VAR)
            try:
                import anthropic
            except ImportError as exc:
                raise MissingDependencyError("anthropic", "providers") from exc
            key = self._api_key or os.environ.get(self.ENV_VAR)
            if not key:
                raise MissingCredentialsError("anthropic", self.ENV_VAR)
            self._client = anthropic.Anthropic(
                api_key=key, max_retries=self._max_retries, timeout=self._timeout
            )
        return self._client

    @property
    def rejects_sampling_params(self) -> bool:
        return bool(_ANTHROPIC_NO_SAMPLING.match(self.model_id))

    @property
    def thinks_by_default(self) -> bool:
        """Whether hidden reasoning shares the max_tokens budget."""
        if self.thinking is not None:
            return self.thinking.get("type") != "disabled"
        return bool(_ANTHROPIC_NO_SAMPLING.match(self.model_id))

    # ----------------------------------------------------------------- request

    def _build_kwargs(
        self, messages: Sequence[Message], system: str | None, params: GenerationParams
    ) -> dict[str, Any]:
        system_prompt, turns = _split_system(messages, system)
        max_tokens = params.max_tokens
        if self.thinks_by_default:
            max_tokens = max(max_tokens, THINKING_MAX_TOKENS_FLOOR)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if params.stop:
            kwargs["stop_sequences"] = list(params.stop)
        if not self.rejects_sampling_params:
            if params.temperature is not None:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        kwargs.update(params.extra.get("anthropic", {}))
        return kwargs

    def _generate(self, messages, system, params) -> ModelResponse:
        kwargs = self._build_kwargs(messages, system, params)
        message = self.client.messages.create(**kwargs)

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        if message.stop_reason == "refusal":
            raise ModelError(
                f"Provider declined the request (stop_reason=refusal, "
                f"details={getattr(message, 'stop_details', None)})"
            )
        usage = {
            "input_tokens": getattr(message.usage, "input_tokens", None),
            "output_tokens": getattr(message.usage, "output_tokens", None),
        }
        return ModelResponse(
            text=text.strip(),
            model=self.name,
            revision=getattr(message, "model", self.revision),
            usage=usage,
            raw={
                "stop_reason": message.stop_reason,
                "request": {k: v for k, v in kwargs.items() if k != "messages"},
            },
        )


# =============================================================================
# OpenAI
# =============================================================================


class OpenAIAdapter(ModelAdapter):
    """GPT models via the official OpenAI SDK (chat completions)."""

    ENV_VAR = "OPENAI_API_KEY"

    def __init__(
        self,
        model_id: str = "gpt-5",
        revision: str | None = None,
        *,
        api_key: str | None = None,
        client: Any = None,
        max_retries: int = 3,
        timeout: float = 600.0,
    ):
        super().__init__(
            name=f"openai:{model_id}", family="openai", revision=revision or model_id
        )
        self.model_id = model_id
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._sampling_rejected = bool(_OPENAI_NO_SAMPLING.match(model_id))

    @property
    def client(self):
        if self._client is None:
            _load_dotenv_once(self.ENV_VAR)
            try:
                import openai
            except ImportError as exc:
                raise MissingDependencyError("openai", "providers") from exc
            key = self._api_key or os.environ.get(self.ENV_VAR)
            if not key:
                raise MissingCredentialsError("openai", self.ENV_VAR)
            self._client = openai.OpenAI(
                api_key=key, max_retries=self._max_retries, timeout=self._timeout
            )
        return self._client

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        system: str | None,
        params: GenerationParams,
        *,
        with_sampling: bool,
    ) -> dict[str, Any]:
        system_prompt, turns = _split_system(messages, system)
        payload = list(turns)
        if system_prompt:
            payload.insert(0, {"role": "system", "content": system_prompt})

        kwargs: dict[str, Any] = {"model": self.model_id, "messages": payload}
        # Reasoning models use `max_completion_tokens`; the budget must also
        # cover hidden reasoning tokens, so raise the floor as for Anthropic.
        if self._sampling_rejected:
            kwargs["max_completion_tokens"] = max(
                params.max_tokens, THINKING_MAX_TOKENS_FLOOR
            )
        else:
            kwargs["max_tokens"] = params.max_tokens
        if with_sampling and not self._sampling_rejected:
            if params.temperature is not None:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.seed is not None:
                kwargs["seed"] = params.seed
        if params.stop:
            kwargs["stop"] = list(params.stop)
        kwargs.update(params.extra.get("openai", {}))
        return kwargs

    def _generate(self, messages, system, params) -> ModelResponse:
        kwargs = self._build_kwargs(messages, system, params, with_sampling=True)
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not self._is_sampling_rejection(exc):
                raise
            # The model rejects non-default sampling parameters; remember that
            # and retry once so the experiment cell is not lost.
            self._sampling_rejected = True
            kwargs = self._build_kwargs(messages, system, params, with_sampling=False)
            completion = self.client.chat.completions.create(**kwargs)

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = {}
        if completion.usage:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        return ModelResponse(
            text=text.strip(),
            model=self.name,
            revision=getattr(completion, "model", self.revision),
            usage=usage,
            raw={
                "finish_reason": choice.finish_reason,
                "request": {k: v for k, v in kwargs.items() if k != "messages"},
            },
        )

    @staticmethod
    def _is_sampling_rejection(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in ("temperature", "top_p", "unsupported_value", "max_tokens")
        )


# =============================================================================
# Registration
# =============================================================================


def _anthropic_factory(model_id: str, revision: str | None = None, **kw):
    return AnthropicAdapter(model_id, revision=revision, **kw)


def _openai_factory(model_id: str, revision: str | None = None, **kw):
    return OpenAIAdapter(model_id, revision=revision, **kw)


def _mock_factory(model_id: str, revision: str | None = None, **kw):
    responses = kw.pop("responses", None) or [
        "What does the loop print on its final iteration?"
    ]
    return ScriptedAdapter(
        responses, name=f"mock:{model_id}", revision=revision or "scripted-1", **kw
    )


register_provider("anthropic", _anthropic_factory)
register_provider("openai", _openai_factory)
register_provider("mock", _mock_factory)


__all__ = ["AnthropicAdapter", "OpenAIAdapter"]
