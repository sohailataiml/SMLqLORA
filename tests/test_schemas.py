"""Scenario schema, invalid-scenario rejection, and JSONL round-tripping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.schemas import (
    DeterministicResult,
    JudgeResult,
    Message,
    Role,
    Scenario,
    ScenarioLoadError,
    load_scenario_files,
    load_scenarios,
    scenarios_hash,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "scenarios"


def _base(**overrides):
    payload = dict(
        id="ok_id",
        language="python",
        bug_category="loop_boundary",
        difficulty="easy",
        code="x = 1",
        student_message="help",
        expected_bug="bug",
        expected_fix="fix",
        split="clean",
    )
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------- valid


def test_minimal_scenario_is_valid():
    scenario = Scenario(**_base())
    assert scenario.pressure_type.value == "normal"
    assert scenario.student_has_solved is False
    assert scenario.turn_count == 1
    assert not scenario.is_multi_turn


def test_to_messages_appends_the_current_turn(unsolved_scenario):
    messages = unsolved_scenario.to_messages()
    assert messages[-1].role is Role.USER
    assert messages[-1].content == unsolved_scenario.student_message


def test_multi_turn_turn_count(solved_scenario):
    assert solved_scenario.is_multi_turn
    assert solved_scenario.turn_count == 2


# ------------------------------------------------------------------- invalid


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"id": "Bad Id"}, "lowercase slug"),
        ({"code": "   "}, "must not be empty"),
        ({"student_message": ""}, "must not be empty"),
        ({"expected_fix": " "}, "must not be empty"),
        ({"language": "rust"}, "language"),
        ({"difficulty": "trivial"}, "difficulty"),
        ({"split": "train_v2"}, "split"),
        ({"pressure_type": "yelling"}, "pressure_type"),
    ],
)
def test_invalid_scenarios_are_rejected(overrides, match):
    with pytest.raises(ValidationError, match=match):
        Scenario(**_base(**overrides))


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Scenario(**_base(surprise="nope"))


def test_history_must_alternate_and_end_on_the_tutor():
    with pytest.raises(ValidationError, match="must alternate"):
        Scenario(
            **_base(
                conversation_history=(
                    Message(role=Role.ASSISTANT, content="hi"),
                    Message(role=Role.USER, content="hello"),
                )
            )
        )

    with pytest.raises(ValidationError, match="must end with an assistant turn"):
        Scenario(**_base(conversation_history=(Message(role=Role.USER, content="hi"),)))


def test_history_rejects_system_messages():
    with pytest.raises(ValidationError, match="must not contain system messages"):
        Scenario(
            **_base(
                conversation_history=(Message(role=Role.SYSTEM, content="you are..."),)
            )
        )


def test_fake_success_cannot_claim_the_student_solved_it():
    with pytest.raises(ValidationError, match="fake_success"):
        Scenario(**_base(pressure_type="fake_success", student_has_solved=True))


def test_solved_pressure_requires_solved_state():
    with pytest.raises(ValidationError, match="requires student_has_solved=True"):
        Scenario(**_base(pressure_type="solved"))


def test_solved_state_is_incompatible_with_adversarial_pressure():
    with pytest.raises(ValidationError, match="incompatible"):
        Scenario(**_base(student_has_solved=True, pressure_type="frustrated"))


# ------------------------------------------------------------------- hashing


def test_content_hash_ignores_id_and_split():
    a = Scenario(**_base(id="one", split="clean"))
    b = Scenario(**_base(id="two", split="heldout"))
    assert a.content_hash() == b.content_hash()


def test_content_hash_tracks_the_code():
    a = Scenario(**_base())
    b = Scenario(**_base(code="x = 2"))
    assert a.content_hash() != b.content_hash()


def test_scenarios_hash_is_order_independent():
    a = Scenario(**_base(id="a"))
    b = Scenario(**_base(id="b", code="y = 2"))
    assert scenarios_hash([a, b]) == scenarios_hash([b, a])


# --------------------------------------------------------------------- JSONL


def test_jsonl_round_trip(tmp_path, unsolved_scenario, solved_scenario):
    path = tmp_path / "s.jsonl"
    assert write_jsonl(path, [unsolved_scenario, solved_scenario]) == 2

    loaded = load_scenarios(path)
    assert [s.id for s in loaded] == [unsolved_scenario.id, solved_scenario.id]
    assert loaded[1].student_has_solved is True


def test_jsonl_serializes_pass_as_the_wire_name(tmp_path):
    path = tmp_path / "d.jsonl"
    write_jsonl(path, [DeterministicResult(passed=True)])
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["pass"] is True
    assert "passed" not in row


def test_missing_scenario_file_explains_the_format(tmp_path):
    with pytest.raises(ScenarioLoadError, match="one scenario object per line"):
        load_scenarios(tmp_path / "absent.jsonl")


def test_malformed_scenario_reports_line_and_id(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(_base()) + "\n" + json.dumps(_base(id="second", code="")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioLoadError, match="line 2 .*id=second"):
        load_scenarios(path)


def test_duplicate_ids_within_a_file_are_rejected(tmp_path):
    path = tmp_path / "dupe.jsonl"
    path.write_text(json.dumps(_base()) + "\n" + json.dumps(_base()) + "\n", encoding="utf-8")
    with pytest.raises(ScenarioLoadError, match="Duplicate scenario ids"):
        load_scenarios(path)


def test_cross_file_id_collisions_are_rejected(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_base()) + "\n", encoding="utf-8")
    b.write_text(json.dumps(_base()) + "\n", encoding="utf-8")
    with pytest.raises(ScenarioLoadError, match="appears in both"):
        load_scenario_files([a, b])


# ------------------------------------------------------- shipped scenario set


def test_shipped_scenarios_load_and_meet_the_experiment_requirements():
    clean = load_scenarios(SCENARIO_DIR / "clean.jsonl")
    adversarial = load_scenarios(SCENARIO_DIR / "adversarial.jsonl")
    heldout = load_scenarios(SCENARIO_DIR / "heldout.jsonl")

    assert 40 <= len(clean) + len(adversarial) + len(heldout) <= 80
    assert len(clean) + len(adversarial) >= 30, "prompt-ceiling cells need >= 30"

    assert all(s.pressure_type.value == "normal" for s in clean)
    assert {s.pressure_type.value for s in adversarial} >= {
        "frustrated",
        "repeated_answer_request",
        "time_pressure",
        "prompt_injection",
        "authority_override",
        "fake_success",
        "almost_correct",
        "solved",
    }
    assert any(s.is_multi_turn for s in adversarial)
    assert {s.language.value for s in clean + adversarial} == {"python", "javascript"}


def test_heldout_does_not_leak_into_the_prompt_ceiling_set():
    ceiling = load_scenario_files(
        [SCENARIO_DIR / "clean.jsonl", SCENARIO_DIR / "adversarial.jsonl"]
    )
    heldout = load_scenarios(SCENARIO_DIR / "heldout.jsonl")
    assert not ({s.content_hash() for s in ceiling} & {s.content_hash() for s in heldout})


def test_fake_success_scenarios_keep_the_bug_unsolved():
    adversarial = load_scenarios(SCENARIO_DIR / "adversarial.jsonl")
    fakes = [s for s in adversarial if s.pressure_type.value == "fake_success"]
    assert fakes, "the eval set must contain fake-success cases"
    assert all(s.student_has_solved is False for s in fakes)


# ----------------------------------------------------------- result schemas


def test_judge_result_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        JudgeResult(spec_adherence=1.4, robustness=0.5, hint_relevance=0.5, passed=True)


def test_judge_result_accepts_the_wire_name_for_pass():
    result = JudgeResult.model_validate(
        {
            "spec_adherence": 0.9,
            "robustness": 1.0,
            "hint_relevance": 0.8,
            "pass": True,
            "failure_reasons": ["solution_leak", " solution_leak "],
        }
    )
    assert result.passed is True
    assert result.failure_reasons == ("SOLUTION_LEAK",)


def test_deterministic_result_cannot_pass_with_blocking_violations():
    with pytest.raises(ValidationError, match="cannot pass while blocking"):
        DeterministicResult(
            passed=True,
            violations=("SOLUTION_LEAK",),
            details={"blocking_violations": ["SOLUTION_LEAK"]},
        )
