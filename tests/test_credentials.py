"""Which credential is in play, and never leaking it.

`load_dotenv(override=False)` lets an environment variable silently outrank the
same key in `.env`. On this project that produced a `NO_QUOTA` verdict against an
unfunded key exported by the shell while a funded key sat unused in `.env` — the
verdict was correct and the conclusion drawn from it was wrong, because it was
about the wrong account.

The shadowing case is therefore the one that matters most here.
"""

from __future__ import annotations

import re

import pytest

from models.credentials import Credential, describe_credential, fingerprint

KEY_A = "sk-ant-api03-" + "a" * 80 + "AAAA"
KEY_B = "sk-ant-api03-" + "b" * 80 + "BBBB"
VAR = "TEST_PROVIDER_API_KEY"


@pytest.fixture
def dotenv(tmp_path):
    def write(value: str | None):
        path = tmp_path / ".env"
        path.write_text(
            f"OTHER_THING=1\n# a comment\n"
            + (f"{VAR}={value}\n" if value is not None else ""),
            encoding="utf-8",
        )
        return path
    return write


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


# ------------------------------------------------------------- fingerprint


def test_fingerprint_never_contains_the_secret():
    printed = fingerprint(KEY_A)
    assert KEY_A not in printed
    assert KEY_A[:20] not in printed


def test_fingerprint_shows_only_a_short_tail_and_digest():
    assert re.fullmatch(r"\.\.\.[A-Za-z0-9_-]{4}/[0-9a-f]{8}", fingerprint(KEY_A))


def test_fingerprint_distinguishes_two_keys():
    assert fingerprint(KEY_A) != fingerprint(KEY_B)


def test_fingerprint_is_stable():
    assert fingerprint(KEY_A) == fingerprint(KEY_A)


def test_fingerprint_ignores_surrounding_whitespace():
    assert fingerprint(f"  {KEY_A}\n") == fingerprint(KEY_A)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_absent_values_fingerprint_as_absent(value):
    assert fingerprint(value) == "absent"


# ------------------------------------------------------ describe_credential


def test_nothing_configured_anywhere(dotenv):
    result = describe_credential(VAR, dotenv_path=dotenv(None))
    assert result.present is False
    assert result.source == "nowhere"
    assert result.warning is None


def test_environment_only(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_A)
    result = describe_credential(VAR, dotenv_path=dotenv(None))
    assert result.present and result.source == "environment"
    assert result.fingerprint == fingerprint(KEY_A)
    assert result.warning is None


def test_dotenv_only_before_it_has_been_loaded(dotenv):
    result = describe_credential(VAR, dotenv_path=dotenv(KEY_A))
    assert result.present
    assert result.source == ".env (not yet loaded)"
    assert result.fingerprint == fingerprint(KEY_A)
    assert result.warning is None


def test_matching_values_are_not_reported_as_a_conflict(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_A)
    result = describe_credential(VAR, dotenv_path=dotenv(KEY_A))
    assert result.shadowed is False
    assert result.warning is None
    assert "identical" in result.source


def test_differing_values_are_reported_as_shadowed(monkeypatch, dotenv):
    """The case that cost a round trip: two keys, only one of them consulted."""
    monkeypatch.setenv(VAR, KEY_A)
    result = describe_credential(VAR, dotenv_path=dotenv(KEY_B))
    assert result.shadowed is True
    assert result.fingerprint == fingerprint(KEY_A)
    assert result.shadowed_fingerprint == fingerprint(KEY_B)


def test_the_shadow_warning_names_both_keys_and_neither_value(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_A)
    warning = describe_credential(VAR, dotenv_path=dotenv(KEY_B)).warning
    assert warning is not None
    assert fingerprint(KEY_A) in warning and fingerprint(KEY_B) in warning
    assert KEY_A not in warning and KEY_B not in warning


def test_whitespace_differences_alone_do_not_raise_a_conflict(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, f" {KEY_A} ")
    assert describe_credential(VAR, dotenv_path=dotenv(KEY_A)).shadowed is False


def test_a_missing_dotenv_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv(VAR, KEY_A)
    result = describe_credential(VAR, dotenv_path=tmp_path / "nope.env")
    assert result.source == "environment"


def test_summary_never_contains_the_secret(monkeypatch, dotenv):
    monkeypatch.setenv(VAR, KEY_A)
    summary = describe_credential(VAR, dotenv_path=dotenv(KEY_B)).summary
    assert VAR in summary and KEY_A not in summary and KEY_B not in summary


def test_summary_of_an_absent_credential_says_so(dotenv):
    assert "not set" in describe_credential(VAR, dotenv_path=dotenv(None)).summary


# ------------------------------------------------------- preflight wiring


def test_preflight_reports_the_credential_it_used(monkeypatch, capsys):
    """A verdict about the wrong account reads as an answer, so the report has
    to name which credential produced it."""
    from scripts import preflight

    credential = Credential(
        "ANTHROPIC_API_KEY", True, "environment (.env ignored)",
        "...9QAA/9914ba76", shadowed=True, shadowed_fingerprint="...XAAA/d3b5c538",
    )
    result = preflight.ProbeResult(
        "anthropic:claude-opus-5", "ANTHROPIC_API_KEY", True, False,
        "NO_QUOTA", "credit balance is too low", "Add funds.",
        credential=credential,
    )
    monkeypatch.setattr(preflight, "probe", lambda spec: result)
    preflight.run(["anthropic:claude-opus-5"])

    out = capsys.readouterr().out
    assert "...9QAA/9914ba76" in out
    assert "WARNING" in out
    assert "...XAAA/d3b5c538" in out


def test_preflight_stays_quiet_when_there_is_no_conflict(monkeypatch, capsys):
    from scripts import preflight

    credential = Credential("ANTHROPIC_API_KEY", True, "environment",
                            "...9QAA/9914ba76")
    result = preflight.ProbeResult(
        "anthropic:claude-opus-5", "ANTHROPIC_API_KEY", True, True,
        "READY", "authenticated", "", 1.0, "ready", credential=credential,
    )
    monkeypatch.setattr(preflight, "probe", lambda spec: result)
    preflight.run(["anthropic:claude-opus-5"])
    assert "WARNING" not in capsys.readouterr().out


def test_probe_result_still_renders_without_a_credential(monkeypatch, capsys):
    """Older call sites construct ProbeResult with no credential description."""
    from scripts import preflight

    result = preflight.ProbeResult(
        "mock:demo", None, True, True, "READY", "fine", "", 0.1, "ready",
    )
    monkeypatch.setattr(preflight, "probe", lambda spec: result)
    preflight.run(["mock:demo"])
    assert "credential : n/a" in capsys.readouterr().out
