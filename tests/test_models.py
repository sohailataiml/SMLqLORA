"""Model adapter contract, registry, and provider quirks.

No test here makes a network call. Provider classes are exercised with injected
fake clients so the request-shaping logic (which is where the real bugs live) is
covered without credentials.
"""

from __future__ import annotations

import types

import pytest

from evaluation.schemas import Message, Role
from models.adapters import (
    EVAL_PARAMS,
    FailingAdapter,
    GenerationParams,
    MissingCredentialsError,
    ModelError,
    ScriptedAdapter,
    UnsupportedProviderError,
    resolve_model,
)
from models.providers import AnthropicAdapter, OpenAIAdapter

USER = [Message(role=Role.USER, content="hello")]


# ----------------------------------------------------------------- contract


def test_scripted_adapter_cycles_responses():
    model = ScriptedAdapter(["a", "b"])
    assert [model.generate(USER).text for _ in range(3)] == ["a", "b", "a"]
    assert model.call_count == 3


def test_callable_responses_see_the_conversation():
    model = ScriptedAdapter(lambda msgs: f"saw {len(msgs)} messages")
    assert model.generate(USER).text == "saw 1 messages"


def test_errors_are_returned_not_raised_by_default():
    response = FailingAdapter().generate(USER)
    assert response.ok is False
    assert "simulated provider outage" in response.error
    assert response.text == ""


def test_errors_can_be_raised_when_asked():
    with pytest.raises(ModelError):
        FailingAdapter().generate(USER, raise_on_error=True)


def test_latency_is_recorded():
    assert ScriptedAdapter(["x"]).generate(USER).latency_s >= 0.0


@pytest.mark.parametrize(
    "messages,match",
    [
        ([], "at least one message"),
        ([Message(role=Role.ASSISTANT, content="hi")], "must start with a user turn"),
        (
            [Message(role=Role.USER, content="hi"), Message(role=Role.ASSISTANT, content="yo")],
            "must end with a user turn",
        ),
    ],
)
def test_malformed_conversations_are_rejected(messages, match):
    with pytest.raises(ValueError, match=match):
        ScriptedAdapter(["x"]).generate(messages)


def test_describe_reports_identity():
    described = ScriptedAdapter(["x"], name="mock:demo", revision="r1").describe()
    assert described == {"model": "mock:demo", "family": "mock", "revision": "r1"}


# ----------------------------------------------------------------- registry


def test_resolve_mock_model():
    model = resolve_model("mock:demo")
    assert model.family == "mock"
    assert model.name == "mock:demo"


def test_resolve_anthropic_does_not_require_credentials_until_use():
    model = resolve_model("anthropic:claude-opus-5@snapshot-1")
    assert model.family == "anthropic"
    assert model.revision == "snapshot-1"


@pytest.mark.parametrize("spec", ["", "   ", "no-provider-here", "unknown:thing"])
def test_bad_specs_raise_with_guidance(spec):
    """The error must name the providers that *are* available, not an empty list."""
    with pytest.raises(UnsupportedProviderError) as excinfo:
        resolve_model(spec)
    message = str(excinfo.value)
    assert "Known providers" in message
    for provider in ("anthropic", "openai", "mock"):
        assert provider in message


def test_missing_model_id_is_reported():
    from models.adapters import ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="no model id"):
        resolve_model("anthropic:")


def test_missing_credentials_message_names_the_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("models.providers._load_dotenv_once", lambda *_: None)
    model = AnthropicAdapter("claude-opus-5")
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        _ = model.client


# --------------------------------------------------------- provider quirks


class _FakeAnthropicClient:
    def __init__(self):
        self.captured = None
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="  a question?  ")],
            stop_reason="end_turn",
            model="claude-opus-5-snapshot",
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def test_anthropic_drops_sampling_params_for_models_that_reject_them():
    """Opus 5 / Sonnet 5 / Opus 4.7+ return 400 if temperature is sent."""
    client = _FakeAnthropicClient()
    model = AnthropicAdapter("claude-opus-5", client=client)
    assert model.rejects_sampling_params is True

    model.generate(USER, system="sys", params=EVAL_PARAMS, raise_on_error=True)
    assert "temperature" not in client.captured
    assert "top_p" not in client.captured
    assert client.captured["system"] == "sys"


def test_anthropic_keeps_sampling_params_for_older_models():
    client = _FakeAnthropicClient()
    model = AnthropicAdapter("claude-haiku-4-5", client=client)
    assert model.rejects_sampling_params is False

    model.generate(USER, params=GenerationParams(max_tokens=100, temperature=0.3))
    assert client.captured["temperature"] == 0.3


def test_anthropic_raises_the_token_floor_when_thinking_is_on():
    client = _FakeAnthropicClient()
    model = AnthropicAdapter("claude-opus-5", client=client)
    model.generate(USER, params=GenerationParams(max_tokens=200))
    # Thinking shares the budget, so a tight cap would truncate the answer.
    assert client.captured["max_tokens"] >= 4096


def test_anthropic_does_not_inflate_tokens_when_thinking_is_disabled():
    client = _FakeAnthropicClient()
    model = AnthropicAdapter(
        "claude-opus-5", client=client, thinking={"type": "disabled"}
    )
    assert model.thinks_by_default is False
    model.generate(USER, params=GenerationParams(max_tokens=200))
    assert client.captured["max_tokens"] == 200


def test_anthropic_strips_whitespace_and_records_usage():
    model = AnthropicAdapter("claude-opus-5", client=_FakeAnthropicClient())
    response = model.generate(USER)
    assert response.text == "a question?"
    assert response.revision == "claude-opus-5-snapshot"
    assert response.usage["input_tokens"] == 10


def test_anthropic_refusal_becomes_an_error():
    class Refusing(_FakeAnthropicClient):
        def _create(self, **kwargs):
            return types.SimpleNamespace(
                content=[], stop_reason="refusal", model="m",
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=0),
                stop_details=types.SimpleNamespace(category="cyber"),
            )

    response = AnthropicAdapter("claude-opus-5", client=Refusing()).generate(USER)
    assert response.ok is False
    assert "refusal" in response.error


class _FakeOpenAIClient:
    def __init__(self, reject_sampling: bool = False):
        self.calls: list[dict] = []
        self.reject_sampling = reject_sampling
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_sampling and "temperature" in kwargs:
            raise RuntimeError("Unsupported value: 'temperature' does not support 0.0")
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="an answer"),
                    finish_reason="stop",
                )
            ],
            model="gpt-test",
            usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


def test_openai_retries_once_without_sampling_params():
    client = _FakeOpenAIClient(reject_sampling=True)
    model = OpenAIAdapter("gpt-4.1", client=client)
    response = model.generate(USER, params=EVAL_PARAMS, raise_on_error=True)

    assert response.text == "an answer"
    assert len(client.calls) == 2
    assert "temperature" in client.calls[0]
    assert "temperature" not in client.calls[1]


def test_openai_reasoning_models_skip_sampling_and_use_completion_tokens():
    client = _FakeOpenAIClient()
    OpenAIAdapter("gpt-5", client=client).generate(USER, params=EVAL_PARAMS)
    assert "temperature" not in client.calls[0]
    assert client.calls[0]["max_completion_tokens"] >= 4096


def test_openai_places_the_system_prompt_first():
    client = _FakeOpenAIClient()
    OpenAIAdapter("gpt-4.1", client=client).generate(USER, system="be terse")
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "be terse"}


def test_non_sampling_errors_still_propagate():
    class Broken(_FakeOpenAIClient):
        def _create(self, **kwargs):
            raise RuntimeError("connection reset by peer")

    response = OpenAIAdapter("gpt-4.1", client=Broken()).generate(USER)
    assert response.ok is False
    assert "connection reset" in response.error


# ------------------------------------------------------------- local models


def test_local_adapter_refuses_large_models_without_opt_in():
    from models.local_hf import LocalHFAdapter

    with pytest.raises(ModelError, match="Refusing to load"):
        LocalHFAdapter("meta-llama/Llama-3.1-70B")


def test_local_adapter_accepts_small_models():
    from models.local_hf import LocalHFAdapter

    model = LocalHFAdapter("Qwen/Qwen3-1.7B")
    assert model.name == "hf:Qwen/Qwen3-1.7B"
    assert model.family == "local-hf"


def test_peft_spec_requires_a_base_and_an_adapter():
    with pytest.raises(ModelError, match="base-model.*adapter-path"):
        resolve_model("peft:Qwen/Qwen3-1.7B")


def test_peft_spec_parses():
    model = resolve_model("peft:Qwen/Qwen3-1.7B+outputs/run-1")
    assert model.adapter_path == "outputs/run-1"
    assert model.name == "peft:Qwen/Qwen3-1.7B+outputs/run-1"


def test_generation_params_serialize_for_the_transcript():
    payload = GenerationParams(max_tokens=64, temperature=0.0, seed=3).to_dict()
    assert payload["max_tokens"] == 64
    assert payload["seed"] == 3
