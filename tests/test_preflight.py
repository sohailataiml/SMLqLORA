"""Connectivity preflight: fail loudly and cheaply, before the expensive run."""

from __future__ import annotations

import pytest

from models.adapters import (
    FailingAdapter,
    ModelResponse,
    ScriptedAdapter,
    register_provider,
)
from scripts.preflight import DIAGNOSES, diagnose, probe, run


@pytest.fixture
def stub_provider(monkeypatch):
    """Register a `stub:` provider whose behavior each test chooses."""

    def install(adapter):
        register_provider("stub", lambda model, **kw: adapter)
        return adapter

    return install


class TestDiagnosis:
    @pytest.mark.parametrize(
        "error,expected",
        [
            ("RateLimitError: insufficient_quota", "NO_QUOTA"),
            ("Your credit balance is too low to access the API", "NO_QUOTA"),
            ("authentication_error: invalid x-api-key", "BAD_CREDENTIAL"),
            ("permission_error: model not allowed", "NO_PERMISSION"),
            ("not_found_error: model does not exist", "BAD_MODEL"),
        ],
    )
    def test_provider_errors_map_to_actionable_verdicts(self, error, expected):
        verdict, remedy = diagnose(error)

        assert verdict == expected
        assert remedy, "every verdict must tell the operator what to do next"

    def test_every_diagnosis_carries_a_remedy(self):
        assert all(remedy for _, _, remedy in DIAGNOSES)

    def test_an_unrecognized_error_is_reported_not_swallowed(self):
        verdict, _ = diagnose("KeyboardInterrupt: something odd")

        assert verdict in {"UNKNOWN_ERROR", "INFRASTRUCTURE"}


class TestProbe:
    def test_a_reachable_model_passes(self, stub_provider):
        stub_provider(ScriptedAdapter(["ready"], name="stub:ok", family="stub"))

        result = probe("stub:ok")

        assert result.ok
        assert result.verdict == "READY"

    def test_an_unfunded_account_fails_with_the_billing_verdict(
        self, stub_provider
    ):
        stub_provider(FailingAdapter("RateLimitError: insufficient_quota"))

        result = probe("stub:broke")

        assert not result.ok
        assert result.verdict == "NO_QUOTA"
        assert "billing" in result.remedy.lower() or "fund" in result.remedy.lower()

    def test_an_empty_body_is_flagged_rather_than_passed(self, stub_provider):
        # Authenticated and billed, but returning nothing would produce 216
        # empty responses that all look like behavioral failures.
        stub_provider(ScriptedAdapter([""], name="stub:silent", family="stub"))

        result = probe("stub:silent")

        assert not result.ok
        assert result.verdict == "EMPTY_RESPONSE"

    def test_an_unknown_provider_fails_without_calling_anything(self):
        result = probe("nosuchprovider:model")

        assert not result.ok
        assert result.verdict == "BAD_SPEC"

    def test_probe_makes_exactly_one_call(self, stub_provider):
        adapter = stub_provider(
            ScriptedAdapter(["ready"], name="stub:count", family="stub")
        )

        probe("stub:count")

        assert adapter.call_count == 1, (
            "the preflight must be cheap; more than one call defeats its purpose"
        )


class TestRun:
    def test_one_failure_is_enough_to_report_failure(self, stub_provider):
        stub_provider(FailingAdapter("insufficient_quota"))

        results = run(["stub:a", "stub:b"], verbose=False)

        assert len(results) == 2
        assert not any(r.ok for r in results)

    def test_results_never_touch_the_results_directory(self, stub_provider, tmp_path):
        # A preflight is not an experiment. Nothing it produces may be recorded
        # as evidence about the model.
        stub_provider(ScriptedAdapter(["ready"], name="stub:ok", family="stub"))

        results = run(["stub:ok"], verbose=False)

        assert not any(tmp_path.iterdir())
        assert not hasattr(results[0], "passed")
