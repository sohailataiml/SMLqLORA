"""Which credential is actually in play, without ever printing it.

`load_dotenv(override=False)` means an environment variable silently wins over
the same key in `.env`. That is the right precedence, but it is invisible: when
two different keys are configured, every diagnostic reports on whichever one the
environment happened to hold, and says nothing about the other.

That cost a real round trip on this project. `preflight.py` reported `NO_QUOTA`
against an unfunded key exported by the shell, while a funded key sat in `.env`
being ignored. The verdict was accurate and the conclusion drawn from it -- "the
account has no credit" -- was wrong, because it was the wrong account.

So a credential check has to answer two questions, not one: is there a key, and
*which* key. Values are never returned or logged; a fingerprint is enough to tell
two keys apart and useless to anyone who steals it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from models.adapters import MissingCredentialsError, ModelError

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = REPO_ROOT / ".env"

#: Characters of the key's tail to show. Enough to recognise a key you are
#: looking at in a console, far too few to reconstruct one.
TAIL_CHARS = 4

#: Characters of the SHA-256 digest to show.
DIGEST_CHARS = 8


def fingerprint(value: str | None) -> str:
    """A stable, non-reversible label for a secret."""
    value = (value or "").strip()
    if not value:
        return "absent"
    digest = hashlib.sha256(value.encode()).hexdigest()[:DIGEST_CHARS]
    return f"...{value[-TAIL_CHARS:]}/{digest}"


@dataclass(frozen=True)
class Credential:
    """Where a credential came from, and what else was available."""

    env_var: str
    present: bool
    source: str
    fingerprint: str
    #: True when `.env` holds a *different* value that precedence is discarding.
    shadowed: bool = False
    shadowed_fingerprint: str | None = None

    @property
    def summary(self) -> str:
        if not self.present:
            return f"{self.env_var} not set"
        return f"{self.env_var} {self.fingerprint} (from {self.source})"

    @property
    def warning(self) -> str | None:
        """The message worth interrupting someone for, or None."""
        if not self.shadowed:
            return None
        return (
            f"{self.env_var} is set in BOTH the environment and .env, with "
            f"DIFFERENT values. The environment wins "
            f"(load_dotenv(override=False)), so this run used "
            f"{self.fingerprint} and ignored {self.shadowed_fingerprint} in "
            f".env. If the verdict above looks wrong for your account, you are "
            f"probably looking at the other key."
        )


def _dotenv_value(env_var: str, dotenv_path: Path | None = None) -> str | None:
    """Read one key straight out of `.env` without touching `os.environ`."""
    path = dotenv_path or DOTENV_PATH
    if not path.exists():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - optional dependency
        return _parse_dotenv_line(path, env_var)
    return dotenv_values(path).get(env_var)


def _parse_dotenv_line(path: Path, env_var: str) -> str | None:
    """Minimal fallback parser for when python-dotenv is not installed."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == env_var:
            return value.strip().strip('"').strip("'")
    return None


def describe_credential(
    env_var: str, *, dotenv_path: Path | None = None
) -> Credential:
    """Report which value of `env_var` is in effect and what it shadows.

    Safe to call before or after `load_dotenv`. Precedence is inferred from the
    values rather than from call ordering: because `load_dotenv(override=False)`
    never overwrites, an effective value that differs from the one in `.env` can
    only have come from the real environment.
    """
    effective = (os.environ.get(env_var) or "").strip() or None
    from_file = (_dotenv_value(env_var, dotenv_path) or "").strip() or None

    if effective is None and from_file is None:
        return Credential(env_var, False, "nowhere", "absent")

    if effective is None:
        # `.env` exists but has not been loaded into the environment yet.
        return Credential(env_var, True, ".env (not yet loaded)",
                          fingerprint(from_file))

    if from_file is None:
        return Credential(env_var, True, "environment", fingerprint(effective))

    if effective == from_file:
        return Credential(env_var, True, "environment and .env (identical)",
                          fingerprint(effective))

    return Credential(
        env_var, True, "environment (.env ignored)", fingerprint(effective),
        shadowed=True, shadowed_fingerprint=fingerprint(from_file),
    )


# =============================================================================
# Resolving a conflict, rather than only reporting it
# =============================================================================

#: Set this to say which source wins when the two disagree. Deliberately
#: explicit: the whole point is that no paid call should ever pick silently.
SOURCE_ENV_VAR = "SMLQLORA_CREDENTIAL_SOURCE"

#: The only two answers that mean anything.
SOURCE_ENVIRONMENT = "environment"
SOURCE_DOTENV = "dotenv"
VALID_SOURCES = (SOURCE_ENVIRONMENT, SOURCE_DOTENV)

#: Every credential a paid run can spend against.
PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


class CredentialConflictError(MissingCredentialsError):
    """Two different values are configured for one credential.

    Subclasses `MissingCredentialsError` on purpose. Every caller that already
    handles a missing credential -- `eval.py`, the evaluator, the preflight --
    handles an ambiguous one the same way and for the same reason: the run must
    not proceed until a human says which key to spend.
    """

    def __init__(self, credential: "Credential"):
        ModelError.__init__(self, self._message(credential))
        self.provider = credential.env_var
        self.env_var = credential.env_var
        self.credential = credential

    @staticmethod
    def _message(credential: "Credential") -> str:
        lines = (
            f"{credential.env_var} is set in BOTH the shell environment "
            f"and .env, with DIFFERENT values, so which account this run "
            f"would bill is ambiguous.",
            f"  environment : {credential.fingerprint}  (would win today)",
            f"  .env        : {credential.shadowed_fingerprint}  (ignored)",
            "Refusing to guess. Resolve it in one of these ways:",
            f"  - unset the shell variable:  unset {credential.env_var}",
            f"  - use the .env value once:   "
            f"{SOURCE_ENV_VAR}={SOURCE_DOTENV}",
            f"  - use the shell value once:  "
            f"{SOURCE_ENV_VAR}={SOURCE_ENVIRONMENT}",
            "This check exists because a preflight once reported NO_QUOTA "
            "against an unfunded shell key while a funded key sat unused "
            "in .env.",
        )
        return chr(10).join(lines)


def _requested_source(source: str | None) -> str | None:
    """Read the operator's explicit choice, rejecting anything unrecognised."""
    raw = source if source is not None else os.environ.get(SOURCE_ENV_VAR)
    raw = (raw or "").strip().lower() or None
    if raw is not None and raw not in VALID_SOURCES:
        raise ValueError(
            f"{SOURCE_ENV_VAR}={raw!r} is not a valid credential source. "
            f"Use one of: {', '.join(VALID_SOURCES)}."
        )
    return raw


def resolve_credential_conflicts(
    env_vars: "Sequence[str]" = PROVIDER_ENV_VARS,
    *,
    dotenv_path: Path | None = None,
    source: str | None = None,
) -> list[Credential]:
    """Refuse to continue while it is ambiguous which key a run would spend.

    Returns the credentials it inspected. Raises `CredentialConflictError` when
    the shell and `.env` disagree and nobody has said which wins.

    When there is no conflict this does nothing at all, so ordinary
    environment-variable precedence is completely unchanged. `.env` never
    overrides the shell implicitly; it overrides only when explicitly selected
    via `SOURCE_ENV_VAR`, and only for the variables actually in conflict.
    """
    requested = _requested_source(source)
    inspected: list[Credential] = []

    for env_var in env_vars:
        credential = describe_credential(env_var, dotenv_path=dotenv_path)
        inspected.append(credential)
        if not credential.shadowed:
            continue
        if requested is None:
            raise CredentialConflictError(credential)
        if requested == SOURCE_DOTENV:
            value = _dotenv_value(env_var, dotenv_path)
            if value is not None:
                os.environ[env_var] = value.strip()
        # SOURCE_ENVIRONMENT: the shell value already wins; nothing to do.

    return inspected


__all__ = [
    "Credential",
    "CredentialConflictError",
    "PROVIDER_ENV_VARS",
    "SOURCE_DOTENV",
    "SOURCE_ENVIRONMENT",
    "SOURCE_ENV_VAR",
    "describe_credential",
    "fingerprint",
    "resolve_credential_conflicts",
]

