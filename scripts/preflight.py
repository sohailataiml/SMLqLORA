"""Connectivity preflight — one cheap call per provider before an expensive run.

The prompt-ceiling ablation costs real money and takes real time. A wrong model
id, an unfunded account or a missing key should surface in one call, not on call
217. This makes exactly one minimal request per configured model and reports,
per provider, whether the credential exists, authenticates, names a real model,
and has usable quota.

Nothing here is an experiment. No result produced by this script is ever written
into `results/` or counted toward any metric.

    python scripts/preflight.py
    python scripts/preflight.py --models anthropic:claude-opus-5 openai:gpt-5
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.schemas import ErrorKind, Message, Role, classify_error  # noqa: E402
from models.providers import _load_dotenv_once  # noqa: E402
from models.adapters import (  # noqa: E402
    GenerationParams,
    MissingCredentialsError,
    MissingDependencyError,
    ModelNotFoundError,
    UnsupportedProviderError,
    resolve_model,
)

#: Deliberately tiny. This is a dial tone, not a behavioral test.
PREFLIGHT_PARAMS = GenerationParams(max_tokens=16, temperature=0.0, seed=1234)
PREFLIGHT_PROMPT = "Reply with the single word: ready"

#: Env var each provider prefix reads, for a presence check that never prints
#: the value itself.
CREDENTIAL_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

#: Provider error text -> (verdict, what the operator should actually do).
DIAGNOSES: tuple[tuple[str, str, str], ...] = (
    ("insufficient_quota", "NO_QUOTA",
     "Account has no usable credit. Add funds at "
     "https://platform.openai.com/settings/organization/billing/overview"),
    ("credit balance is too low", "NO_QUOTA",
     "Account has no usable credit. Add funds at "
     "https://console.anthropic.com/settings/billing"),
    ("billing", "BILLING",
     "Billing problem on the account - check the provider console."),
    ("authentication_error", "BAD_CREDENTIAL",
     "The key was rejected. Check it is current and belongs to the funded org."),
    ("invalid_api_key", "BAD_CREDENTIAL",
     "The key was rejected. Check it is current and belongs to the funded org."),
    ("invalid x-api-key", "BAD_CREDENTIAL",
     "The key was rejected. Check it is current and belongs to the funded org."),
    ("permission_error", "NO_PERMISSION",
     "The key authenticates but may not access this model."),
    ("not_found_error", "BAD_MODEL",
     "The model id was not found for this account."),
    ("model_not_found", "BAD_MODEL",
     "The model id was not found for this account."),
    ("does not exist", "BAD_MODEL",
     "The model id was not found for this account."),
    ("rate_limit", "RATE_LIMITED",
     "Rate limited on the very first call - raise the limit or lower "
     "--max-workers before running the full experiment."),
)


@dataclass
class ProbeResult:
    """The outcome of one provider dial tone."""

    model_spec: str
    credential_env: str | None
    credential_present: bool
    ok: bool
    verdict: str
    detail: str
    remedy: str = ""
    latency_s: float = 0.0
    reply: str = ""

    @property
    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.model_spec:<32} {self.verdict}"


def diagnose(error: str) -> tuple[str, str]:
    """Map a provider error string onto a verdict and a concrete remedy."""
    lowered = error.lower()
    for marker, verdict, remedy in DIAGNOSES:
        if marker in lowered:
            return verdict, remedy
    if classify_error(error) is ErrorKind.INFRASTRUCTURE:
        return "INFRASTRUCTURE", "Transient or network-level failure - retry."
    return "UNKNOWN_ERROR", "Unrecognized provider error; read the detail above."


def probe(model_spec: str) -> ProbeResult:
    """Make exactly one minimal call against `model_spec`."""
    # The provider adapters load .env lazily, inside their constructors. Do it
    # up front here, or the credential-presence check below reads a still-empty
    # environment and reports a false MISSING.
    _load_dotenv_once()

    provider = model_spec.split(":", 1)[0].lower()
    env_var = CREDENTIAL_ENV.get(provider)
    present = bool(os.environ.get(env_var)) if env_var else True

    if env_var and not present:
        return ProbeResult(
            model_spec, env_var, False, False, "NO_CREDENTIAL",
            f"{env_var} is not set in the environment.",
            f"Set {env_var} in .env (see .env.example).",
        )

    try:
        adapter = resolve_model(model_spec)
    except MissingCredentialsError as exc:
        return ProbeResult(model_spec, env_var, present, False,
                           "NO_CREDENTIAL", str(exc), f"Set {env_var}.")
    except MissingDependencyError as exc:
        return ProbeResult(model_spec, env_var, present, False,
                           "NO_SDK", str(exc),
                           'pip install -e ".[providers]"')
    except (UnsupportedProviderError, ModelNotFoundError) as exc:
        return ProbeResult(model_spec, env_var, present, False,
                           "BAD_SPEC", str(exc),
                           "Fix the model spec passed to --models.")

    response = adapter.generate(
        [Message(role=Role.USER, content=PREFLIGHT_PROMPT)],
        params=PREFLIGHT_PARAMS,
    )
    if response.error:
        verdict, remedy = diagnose(response.error)
        return ProbeResult(model_spec, env_var, present, False, verdict,
                           response.error, remedy, response.latency_s)

    # An empty body is not a credential problem, but it is not a dial tone
    # either - flag it rather than let the full run discover it.
    if not response.text.strip():
        return ProbeResult(
            model_spec, env_var, present, False, "EMPTY_RESPONSE",
            "Authenticated and billed, but the model returned no text.",
            "Check max_tokens and any thinking-budget interaction.",
            response.latency_s,
        )

    return ProbeResult(
        model_spec, env_var, present, True, "READY",
        f"authenticated, model resolved, quota usable "
        f"(revision={response.revision})",
        latency_s=response.latency_s,
        reply=response.text.strip()[:60],
    )


def run(model_specs: Sequence[str], *, verbose: bool = True) -> list[ProbeResult]:
    results = [probe(spec) for spec in model_specs]
    if verbose:
        print("Connectivity preflight - one minimal call per model.")
        print("This is not an experiment; nothing here is recorded as a result.\n")
        for result in results:
            print(result.line)
            print(f"       credential : {result.credential_env or 'n/a'} "
                  f"{'present' if result.credential_present else 'MISSING'}")
            print(f"       detail     : {result.detail}")
            if result.reply:
                print(f"       reply      : {result.reply!r} "
                      f"({result.latency_s:.2f}s)")
            if result.remedy:
                print(f"       remedy     : {result.remedy}")
            print()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["anthropic:claude-opus-5", "openai:gpt-5"])
    args = parser.parse_args(argv)

    results = run(args.models)
    failed = [r for r in results if not r.ok]
    if failed:
        print("=" * 72)
        print(f"PREFLIGHT FAILED for {len(failed)} of {len(results)} models.")
        print("The full experiment was NOT started. Nothing was spent.")
        for r in failed:
            print(f"  - {r.model_spec}: {r.verdict} - {r.remedy or r.detail}")
        print("=" * 72)
        return 1

    print("=" * 72)
    print(f"PREFLIGHT PASSED - all {len(results)} models reachable and funded.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
