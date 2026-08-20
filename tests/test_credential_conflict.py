"""A shadowed key must stop a paid run, not steer it.

`load_dotenv(override=False)` lets a stale shell variable outrank a funded key
in `.env`. Reporting that was not enough: the report was a warning attached to a
`NO_QUOTA` verdict, and the verdict is what people read. These tests pin the
stronger rule -- when the two sources disagree, nothing spends money until a
human says which key wins.

The value of a key is never asserted on here, only its fingerprint, because a
test that hardcodes a real secret is the same defect wearing a different hat.
"""

from __future__ import annotations

import pytest

from models.adapters import MissingCredentialsError
from models.credentials import (
    SOURCE_DOTENV,
    SOURCE_ENV_VAR,
    SOURCE_ENVIRONMENT,
    CredentialConflictError,
    fingerprint,
    resolve_credential_conflicts,
)

KEY_SHELL = "sk-ant-api03-" + "a" * 80 + "AAAA"
KEY_DOTENV = "sk-ant-api03-" + "b" * 80 + "BBBB"
VAR = "TEST_PROVIDER_API_KEY"


@pytest.fixture
def dotenv(tmp_path):
    def write(value: str | None):
        path = tmp_path / ".env"
        body = "UNRELATED=1\n" + (f"{VAR}={value}\n" if value is not None else "")
        path.write_text(body, encoding="utf-8")
        return path
    return write


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)


# ------------------------------------------------------------------ the fault

def test_conflicting_values_refuse_to_resolve(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    with pytest.raises(CredentialConflictError):
        resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))


def test_the_refusal_names_both_keys_and_leaks_neither(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    with pytest.raises(CredentialConflictError) as caught:
        resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))
    message = str(caught.value)
    assert fingerprint(KEY_SHELL) in message
    assert fingerprint(KEY_DOTENV) in message
    assert KEY_SHELL not in message
    assert KEY_DOTENV not in message


def test_the_refusal_says_how_to_resolve_it(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    with pytest.raises(CredentialConflictError) as caught:
        resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))
    message = str(caught.value)
    assert f"unset {VAR}" in message
    assert SOURCE_ENV_VAR in message
    assert SOURCE_DOTENV in message


def test_callers_that_handle_a_missing_key_also_catch_this(monkeypatch, dotenv):
    """eval.py catches MissingCredentialsError; ambiguity must land there too."""
    monkeypatch.setenv(VAR, KEY_SHELL)
    with pytest.raises(MissingCredentialsError):
        resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))


# ------------------------------------------------------- normal behaviour kept

def test_shell_only_is_untouched(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(None))
    import os
    assert os.environ[VAR] == KEY_SHELL


def test_identical_values_are_not_a_conflict(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_SHELL))


def test_nothing_configured_is_not_a_conflict(dotenv):
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(None))


def test_dotenv_only_is_not_a_conflict(dotenv):
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))


def test_a_missing_dotenv_file_is_not_a_conflict(monkeypatch, tmp_path):
    monkeypatch.setenv(VAR, KEY_SHELL)
    resolve_credential_conflicts([VAR], dotenv_path=tmp_path / "absent.env")


# ------------------------------------------------------------ explicit choice

def test_dotenv_can_be_selected_explicitly(monkeypatch, dotenv):
    import os
    monkeypatch.setenv(VAR, KEY_SHELL)
    monkeypatch.setenv(SOURCE_ENV_VAR, SOURCE_DOTENV)
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))
    assert os.environ[VAR] == KEY_DOTENV


def test_environment_can_be_selected_explicitly(monkeypatch, dotenv):
    import os
    monkeypatch.setenv(VAR, KEY_SHELL)
    monkeypatch.setenv(SOURCE_ENV_VAR, SOURCE_ENVIRONMENT)
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))
    assert os.environ[VAR] == KEY_SHELL


def test_the_choice_is_case_insensitive(monkeypatch, dotenv):
    import os
    monkeypatch.setenv(VAR, KEY_SHELL)
    monkeypatch.setenv(SOURCE_ENV_VAR, "DotEnv")
    resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))
    assert os.environ[VAR] == KEY_DOTENV


def test_an_unrecognised_choice_is_rejected(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_SHELL)
    monkeypatch.setenv(SOURCE_ENV_VAR, "whichever")
    with pytest.raises(ValueError, match="not a valid credential source"):
        resolve_credential_conflicts([VAR], dotenv_path=dotenv(KEY_DOTENV))


def test_selecting_a_source_does_not_touch_unconflicted_vars(monkeypatch, dotenv):
    """The escape hatch is narrow: only variables actually in conflict move."""
    import os
    monkeypatch.setenv(SOURCE_ENV_VAR, SOURCE_DOTENV)
    monkeypatch.setenv("OTHER_KEY", "left-alone")
    resolve_credential_conflicts(["OTHER_KEY"], dotenv_path=dotenv(KEY_DOTENV))
    assert os.environ["OTHER_KEY"] == "left-alone"


# ------------------------------------------- the path a real evaluation takes

@pytest.fixture
def repo_dotenv(monkeypatch, tmp_path):
    """Point the credentials module at a throwaway .env."""
    def write(env_var: str, value: str):
        path = tmp_path / ".env"
        path.write_text(f"{env_var}={value}\n", encoding="utf-8")
        monkeypatch.setattr("models.credentials.DOTENV_PATH", path)
        return path
    return write


def test_the_anthropic_adapter_refuses_a_shadowed_key(monkeypatch, repo_dotenv):
    """This is the path `eval.py --judge anthropic:...` actually walks."""
    from models.providers import AnthropicAdapter

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_SHELL)
    repo_dotenv("ANTHROPIC_API_KEY", KEY_DOTENV)

    adapter = AnthropicAdapter("claude-opus-5")
    with pytest.raises(CredentialConflictError):
        _ = adapter.client


def test_the_openai_adapter_refuses_a_shadowed_key(monkeypatch, repo_dotenv):
    from models.providers import OpenAIAdapter

    monkeypatch.setenv("OPENAI_API_KEY", KEY_SHELL)
    repo_dotenv("OPENAI_API_KEY", KEY_DOTENV)

    adapter = OpenAIAdapter("gpt-5")
    with pytest.raises(CredentialConflictError):
        _ = adapter.client


def test_one_provider_conflict_does_not_block_the_other(monkeypatch, repo_dotenv):
    """An OpenAI conflict must not stop an Anthropic-only judging run."""
    from models.providers import AnthropicAdapter

    monkeypatch.setenv("OPENAI_API_KEY", KEY_SHELL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_SHELL)
    repo_dotenv("OPENAI_API_KEY", KEY_DOTENV)

    adapter = AnthropicAdapter("claude-opus-5")
    assert adapter.client is not None


def test_preflight_reports_a_conflict_as_a_conflict(monkeypatch, repo_dotenv):
    """Not as NO_QUOTA. The wrong verdict about the wrong account is the bug."""
    from scripts.preflight import probe

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_SHELL)
    repo_dotenv("ANTHROPIC_API_KEY", KEY_DOTENV)

    result = probe("anthropic:claude-opus-5")
    assert result.verdict == "CREDENTIAL_CONFLICT"
    assert not result.ok
    assert KEY_SHELL not in result.detail
    assert KEY_DOTENV not in result.detail
