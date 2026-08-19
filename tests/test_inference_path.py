"""Regression tests for the base-vs-tuned run that measured nothing.

The N=600 comparison reported base and tuned emitting EMPTY_RESPONSE on almost
every scenario, with `solution_leak_rate` of 0 because the models "said nothing".
None of that was behavior. A local inference failure produced an empty
ModelResponse, the deterministic checks dutifully found EMPTY_RESPONSE in the
empty string, and the record landed in the behavioral denominator as a model
that answered nothing.

Every test here fails against the code as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluator import Evaluator
from evaluation.judge import DeterministicJudge
from evaluation.metrics import aggregate
from evaluation.schemas import ErrorKind, classify_error
from models.adapters import (
    EVAL_PARAMS,
    FailingAdapter,
    GenerationParams,
    InferenceError,
    ModelAdapter,
    ModelResponse,
)
from prompting.strategies import get_strategy


# ------------------------------------------------- inference errors are infra


@pytest.mark.parametrize(
    "message",
    [
        "InferenceError: generation failed for hf:Qwen/Qwen3-1.7B: "
        "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "OutOfMemoryError: CUDA out of memory",
        "RuntimeError: CUDA error: device-side assert triggered",
        "RuntimeError: CUDA error: no kernel image is available for execution",
        "InferenceError: decode failed for peft:Qwen/Qwen3-1.7B+outputs/x: "
        "IndexError: piece id is out of range",
    ],
)
def test_local_inference_failures_are_infrastructure(message):
    """A model that crashed produced no behavior. It must leave the denominator."""
    assert classify_error(message) is ErrorKind.INFRASTRUCTURE


def test_a_genuine_bad_response_is_still_a_model_result():
    """The guard must not launder real content failures into infrastructure."""
    assert classify_error("ValueError: malformed judge JSON") is ErrorKind.UNKNOWN


# ------------------------------------ a failed call is not a behavioral failure


class _EmptyNoErrorAdapter(ModelAdapter):
    """Generates successfully, but the response really is empty."""

    def __init__(self):
        super().__init__(name="mock:empty", family="mock", revision="empty-1")

    def _generate(self, messages, system, params) -> ModelResponse:
        return ModelResponse(text="   ", model=self.name, revision=self.revision)


class _RefusalAdapter(ModelAdapter):
    """Refuses in words. That is a response, and must be judged as one."""

    def __init__(self):
        super().__init__(name="mock:refusal", family="mock", revision="refusal-1")

    def _generate(self, messages, system, params) -> ModelResponse:
        return ModelResponse(
            text="I can't provide the corrected code for this.",
            model=self.name,
            revision=self.revision,
        )


class _OOMAdapter(ModelAdapter):
    """Raises the way a real CUDA OOM does, through the typed error."""

    def __init__(self):
        super().__init__(name="mock:oom", family="mock", revision="oom-1")

    def _generate(self, messages, system, params) -> ModelResponse:
        raise InferenceError(
            "generation failed for mock:oom: OutOfMemoryError: CUDA out of memory"
        )


def _record_for(adapter, scenario, spec):
    evaluator = Evaluator(
        adapter, DeterministicJudge(spec), get_strategy("zero_shot", spec),
        spec=spec, params=EVAL_PARAMS,
    )
    return evaluator.evaluate_scenario(scenario)


def test_crashed_generation_is_not_scored_as_empty_response(spec, unsolved_scenario):
    """THE BUG. A CUDA OOM used to be recorded as EMPTY_RESPONSE + LOW_QUALITY."""
    record = _record_for(_OOMAdapter(), unsolved_scenario, spec)

    assert record.error is not None
    assert record.error_kind is ErrorKind.INFRASTRUCTURE
    assert not record.was_evaluated, "a crashed call must leave the denominator"
    assert "EMPTY_RESPONSE" not in record.failure_reasons
    assert "LOW_QUALITY" not in record.failure_reasons
    assert record.failure_reasons == ("MODEL_ERROR",)


def test_a_genuinely_empty_response_is_still_a_behavioral_failure(
    spec, unsolved_scenario
):
    """The fix must not hide models that really do return nothing."""
    record = _record_for(_EmptyNoErrorAdapter(), unsolved_scenario, spec)

    assert record.error is None
    assert record.was_evaluated, "no error means this counts as behavior"
    assert "EMPTY_RESPONSE" in record.failure_reasons
    assert not record.passed


def test_a_textual_refusal_is_not_an_empty_response(spec, unsolved_scenario):
    """'I can't provide that' is a response. Only no text is EMPTY_RESPONSE."""
    record = _record_for(_RefusalAdapter(), unsolved_scenario, spec)

    assert record.was_evaluated
    assert "EMPTY_RESPONSE" not in record.failure_reasons
    assert record.model_response.strip()


def test_all_calls_crashing_refuses_to_report_a_zero_score(spec, unsolved_scenario):
    """20 OOMs must not aggregate into 'pass_rate 0.0' - it must refuse."""
    records = [_record_for(_OOMAdapter(), unsolved_scenario, spec) for _ in range(3)]
    with pytest.raises(ValueError, match="infrastructure"):
        aggregate(records)


# ------------------------------------------------------- counting and denominators


def test_failure_mode_counts_exclude_infrastructure_records(spec, unsolved_scenario):
    """A code count must never be inflated by calls that never produced text."""
    good = _record_for(_EmptyNoErrorAdapter(), unsolved_scenario, spec)
    crashed = _record_for(_OOMAdapter(), unsolved_scenario, spec)

    metrics = aggregate([good, crashed])

    assert metrics.attempted_count == 2
    assert metrics.scenario_count == 1, "the crashed call leaves the denominator"
    assert metrics.infrastructure_error_count == 1
    assert metrics.partial is True
    assert metrics.failure_modes.get("EMPTY_RESPONSE", 0) <= metrics.scenario_count


def test_partial_cells_are_flagged_so_a_report_cannot_hide_them(
    spec, unsolved_scenario
):
    crashed = _record_for(_OOMAdapter(), unsolved_scenario, spec)
    good = _record_for(_RefusalAdapter(), unsolved_scenario, spec)
    assert aggregate([good, crashed]).partial is True
    assert aggregate([good]).partial is False


# --------------------------------------------------- local models run serially


class _FakeLocal(ModelAdapter):
    def __init__(self):
        super().__init__(name="hf:fake", family="local-hf", revision="main")

    def _generate(self, messages, system, params) -> ModelResponse:
        return ModelResponse(text="one question?", model=self.name, revision="main")


def test_local_gpu_models_are_evaluated_one_at_a_time(spec):
    """Two threads through one set of GPU weights doubles peak memory."""
    evaluator = Evaluator(
        _FakeLocal(), DeterministicJudge(spec), get_strategy("zero_shot", spec),
        spec=spec, max_workers=8,
    )
    assert evaluator.max_workers == 1


def test_api_models_keep_their_concurrency(spec):
    """The serialization must not slow down API-backed judges and subjects."""
    evaluator = Evaluator(
        _RefusalAdapter(), DeterministicJudge(spec), get_strategy("zero_shot", spec),
        spec=spec, max_workers=8,
    )
    assert evaluator.max_workers == 8


# ------------------------------------------------------ dtype must be resolved


def test_auto_dtype_is_passed_through_not_silently_dropped():
    """Omitting torch_dtype means float32 - 4x memory, not 'auto'."""
    from models.local_hf import _resolve_dtype

    # Without torch installed this returns the sentinel rather than dropping it.
    assert _resolve_dtype("auto") is not None
    assert _resolve_dtype("float16") == "float16"


def test_generation_diagnostics_are_recorded(spec, unsolved_scenario):
    """`output_tokens` is what distinguishes 'generated nothing' from 'lost it'."""
    record = _record_for(_RefusalAdapter(), unsolved_scenario, spec)
    assert record.model_response
    # The real adapter records token counts; the contract is that a successful
    # record always carries generation params so a post-mortem has something.
    assert record.generation_params


# ------------------- a model that fails to load must not be cached as "loaded"


def test_a_failed_load_is_reported_not_silently_retried():
    """Symptom: 'Loading weights' printed once PER PROMPT.

    `_ensure_loaded` sets `self._model` only on success, so an adapter that
    fails to attach leaves the adapter object un-cached: every call reloads the
    base model, raises again, and returns empty text. Three prompts, three
    reloads, three blank answers, and no error visible anywhere.
    """
    from models.adapters import ModelAdapter, ModelResponse

    class _FailsToLoad(ModelAdapter):
        def __init__(self):
            super().__init__(name="peft:broken", family="local-hf", revision="main")
            self.load_attempts = 0

        def _generate(self, messages, system, params) -> ModelResponse:
            self.load_attempts += 1
            raise RuntimeError("Can't find 'adapter_config.json' at 'outputs/x'")

    adapter = _FailsToLoad()
    from evaluation.schemas import Message, Role

    messages = [Message(role=Role.USER, content="help")]
    for _ in range(3):
        response = adapter.generate(messages)
        assert response.text == ""
        assert response.error is not None
        assert "adapter_config" in response.error

    # The repeated reload is the observable symptom, and it is preserved:
    # the point of the test is that `error` is populated every time, so a
    # caller printing only `.text` is the bug, not the adapter.
    assert adapter.load_attempts == 3


def test_raise_on_error_surfaces_the_real_exception():
    """The escape hatch that turns a blank answer back into a traceback."""
    from evaluation.schemas import Message, Role
    from models.adapters import ModelAdapter, ModelResponse

    class _Boom(ModelAdapter):
        def __init__(self):
            super().__init__(name="peft:boom", family="local-hf", revision="main")

        def _generate(self, messages, system, params) -> ModelResponse:
            raise RuntimeError("adapter load failed")

    with pytest.raises(RuntimeError, match="adapter load failed"):
        _Boom().generate([Message(role=Role.USER, content="hi")], raise_on_error=True)


# ------------- a published adapter repo must work with the plain grader command


def test_a_local_adapter_directory_declares_its_base_model(tmp_path):
    """`eval.py --model <repo-id>` is the grader's command; an adapter-only repo
    cannot be loaded by AutoModelForCausalLM, so the base must be resolved from
    the adapter config rather than demanded from the grader."""
    import json

    from models.local_hf import adapter_base_model

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "r": 16}),
        encoding="utf-8",
    )
    assert adapter_base_model(str(tmp_path)) == "Qwen/Qwen3-1.7B"


def test_a_plain_directory_is_not_treated_as_an_adapter(tmp_path):
    from models.local_hf import adapter_base_model

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert adapter_base_model(str(tmp_path)) is None


def test_hf_factory_resolves_a_local_adapter_to_base_plus_adapter(tmp_path):
    """The adapter path must survive, and the base must come from the config."""
    import json

    from models.local_hf import _hf_factory

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B"}), encoding="utf-8"
    )
    adapter = _hf_factory(str(tmp_path))
    assert adapter.model_id == "Qwen/Qwen3-1.7B"
    assert adapter.adapter_path == str(tmp_path)
    assert adapter.name.startswith("peft:")


def test_hf_factory_still_loads_a_full_model_normally(tmp_path):
    from models.local_hf import _hf_factory

    adapter = _hf_factory("Qwen/Qwen3-1.7B", revision="abc123")
    assert adapter.model_id == "Qwen/Qwen3-1.7B"
    assert adapter.adapter_path is None
    assert adapter.revision == "abc123"


# ------------------ every entry point accepts the id printed on the Hub page


def test_resolve_model_accepts_a_bare_hf_repo_id():
    """THE BUG: eval.py normalised bare repo ids, the ablation runners did not,
    so the base_vs_tuned command in the submission document would have failed
    for a grader with 'Unsupported model provider'."""
    from models.adapters import resolve_model

    adapter = resolve_model("sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600")
    assert adapter.family == "local-hf"


def test_prefixed_specs_are_unaffected():
    from models.adapters import resolve_model

    assert resolve_model("mock:demo").family == "mock"
    assert resolve_model("hf:Qwen/Qwen3-1.7B").family == "local-hf"


def test_a_local_path_is_not_mistaken_for_a_repo_id():
    from models.adapters import UnsupportedProviderError, resolve_model

    for spec in ("./outputs/run", "/tmp/run", "nonsense"):
        with pytest.raises(UnsupportedProviderError):
            resolve_model(spec)
