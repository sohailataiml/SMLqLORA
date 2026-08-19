"""Local Hugging Face adapters: the base model and the QLoRA-tuned checkpoint.

Both are the same class — a tuned model is the base model plus a PEFT adapter —
so base-vs-tuned evaluation runs through identical code paths and identical
generation settings. That symmetry is what makes the comparison meaningful.

Imports of torch/transformers are deferred to first use so the unit test suite
runs on a CPU-only machine with neither installed.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from models.adapters import (
    GenerationParams,
    InferenceError,
    MissingDependencyError,
    ModelAdapter,
    ModelError,
    ModelResponse,
    register_provider,
)
from evaluation.schemas import Message, Role

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"

#: Guardrail against silently pulling a very large model onto a laptop.
MAX_UNCONFIRMED_PARAMS_B = 5.0


def _messages_to_chat(
    messages: Sequence[Message], system: str | None
) -> list[dict[str, str]]:
    chat: list[dict[str, str]] = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        chat.append({"role": msg.role.value, "content": msg.content})
    return chat


def _resolve_dtype(dtype: str) -> Any:
    """Pick a load dtype the current GPU can actually compute in.

    `"auto"` follows the checkpoint, which for Qwen3 is bfloat16 - and a T4 is
    Turing, which has no bfloat16 unit. Loading bf16 there is either very slow or
    raises from kernels that have no bf16 implementation. On such a GPU this
    resolves to float16 instead, which Turing does support natively.
    """
    if dtype != "auto":
        return dtype
    try:
        import torch
    except ImportError:
        return "auto"
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return "auto"


def _guess_param_billions(model_id: str) -> float | None:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id)
    return float(match.group(1)) if match else None


class LocalHFAdapter(ModelAdapter):
    """A causal LM loaded with transformers, optionally with a PEFT adapter."""

    def __init__(
        self,
        model_id: str = DEFAULT_BASE_MODEL,
        revision: str | None = None,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        allow_large: bool = False,
        model: Any = None,
        tokenizer: Any = None,
    ):
        label = f"peft:{model_id}+{adapter_path}" if adapter_path else f"hf:{model_id}"
        super().__init__(
            name=label,
            family="local-hf",
            revision=revision or "main",
        )
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = model
        self._tokenizer = tokenizer

        size = _guess_param_billions(model_id)
        if size is not None and size > MAX_UNCONFIRMED_PARAMS_B and not allow_large:
            raise ModelError(
                f"Refusing to load {model_id!r} (~{size}B parameters) without an "
                f"explicit opt-in. This project targets small instruction-tuned "
                f"models. Pass allow_large=True if you really mean it."
            )

    # ------------------------------------------------------------------ loading

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError("transformers/torch", "train") from exc

        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }
        # Skipping torch_dtype does NOT mean "auto" - it means float32, which is
        # 4x the memory of Qwen3's bfloat16 checkpoint (8.1 GiB instead of 2.0)
        # and enough to exhaust a 16 GiB T4 partway through a run.
        kwargs["torch_dtype"] = _resolve_dtype(self.dtype)

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError("bitsandbytes", "train") from exc
            import torch

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        if self.adapter_path:
            if not os.path.exists(self.adapter_path) and "/" not in self.adapter_path:
                raise ModelError(
                    f"Adapter path {self.adapter_path!r} does not exist. Pass a local "
                    f"checkpoint directory or a Hugging Face repo id."
                )
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise MissingDependencyError("peft", "train") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.eval()

        self._model = model.eval()

    # --------------------------------------------------------------- inference

    def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        self._ensure_loaded()
        import torch

        chat = _messages_to_chat(messages, system)
        prompt = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if _supports_thinking_flag(self.model_id) else {}),
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if params.temperature is None or params.temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=params.temperature,
                top_p=params.top_p if params.top_p is not None else 1.0,
            )
        if params.seed is not None:
            torch.manual_seed(params.seed)

        try:
            with torch.no_grad():
                output = self._model.generate(**inputs, **gen_kwargs)
        except Exception as exc:
            # An OOM or a device-side assert means no response exists. Raising a
            # typed error keeps it out of the behavioral denominator instead of
            # scoring the model as having answered nothing.
            raise InferenceError(
                f"generation failed for {self.name}: {type(exc).__name__}: {exc}"
            ) from exc

        input_length = int(inputs["input_ids"].shape[-1])
        generated = output[0][input_length:]
        generated_length = int(generated.shape[-1])

        try:
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            raw_text = self._tokenizer.decode(generated, skip_special_tokens=False)
        except Exception as exc:
            raise InferenceError(
                f"decode failed for {self.name}: {type(exc).__name__}: {exc}"
            ) from exc

        # Keep enough evidence in the transcript to tell "the model generated
        # nothing" apart from "the model generated text that post-processing
        # threw away". Without these fields both look identical downstream.
        return ModelResponse(
            text=text.strip(),
            model=self.name,
            revision=self.revision,
            usage={
                "input_tokens": input_length,
                "output_tokens": generated_length,
            },
            raw={
                "generation_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
                "input_length": input_length,
                "output_length": int(output[0].shape[-1]),
                "generated_length": generated_length,
                "first_generated_token_ids": [int(t) for t in generated[:8]],
                "decoded_with_special_tokens": raw_text[:600],
                "decoded_chars": len(text),
                "stripped_to_empty": bool(raw_text.strip() and not text.strip()),
            },
        )


def _supports_thinking_flag(model_id: str) -> bool:
    """Qwen3 chat templates accept `enable_thinking`; most others do not."""
    return "qwen3" in model_id.lower()


def adapter_base_model(model_id: str, revision: str | None = None) -> str | None:
    """The base model a PEFT adapter repo/directory was trained on, or None.

    A published QLoRA checkpoint is an adapter, not a model: a few tens of MB of
    LoRA weights that are meaningless without the base they were trained against.
    `AutoModelForCausalLM.from_pretrained` cannot load one.

    Graders run `eval.py --model <hf-repo-id>`, so the adapter case has to just
    work rather than requiring a `peft:base+adapter` incantation they have no
    reason to know. Reading `base_model_name_or_path` out of the adapter config
    is how the repo declares what it needs.
    """
    import json

    local = os.path.join(model_id, "adapter_config.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(
            model_id, "adapter_config.json", revision=revision or None
        )
    except Exception:
        return None  # not an adapter repo, or not reachable - treat as a full model
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("base_model_name_or_path")


def _hf_factory(model_id: str, revision: str | None = None, **kw):
    """Load a full model, or transparently resolve an adapter to base+adapter."""
    base = adapter_base_model(model_id, revision)
    if base:
        # The adapter pins its own base; `revision` names the ADAPTER's commit,
        # so it must not be applied to the base model as well.
        return LocalHFAdapter(base, adapter_path=model_id, **kw)
    return LocalHFAdapter(model_id, revision=revision, **kw)


def _peft_factory(model_id: str, revision: str | None = None, **kw):
    """`peft:<base-model>+<adapter-path>`."""
    if "+" not in model_id:
        raise ModelError(
            f"PEFT model spec {model_id!r} must be '<base-model>+<adapter-path>', "
            f"e.g. 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1'."
        )
    base, _, adapter = model_id.partition("+")
    return LocalHFAdapter(base.strip(), revision=revision, adapter_path=adapter.strip(), **kw)


register_provider("hf", _hf_factory)
register_provider("peft", _peft_factory)


__all__ = ["DEFAULT_BASE_MODEL", "LocalHFAdapter"]
