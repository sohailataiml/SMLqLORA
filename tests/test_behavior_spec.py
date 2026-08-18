"""The behavior spec is the contract. If it parses wrong, everything downstream
measures the wrong thing."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from behavior.spec import AppliesWhen, BehaviorSpec, CriterionKind, load_spec


def test_spec_loads_and_is_versioned(spec):
    assert spec.version
    assert spec.id == "socratic_debug_tutor"
    assert len(spec.spec_sha256) == 64


def test_required_criteria_are_present(spec):
    required = {
        "solution_leak",
        "explicit_final_diagnosis",
        "exactly_one_hint_or_question",
        "hint_relevant",
        "adapted_to_student_state",
        "premature_confirmation",
    }
    assert required <= {c.id for c in spec.criteria}


def test_student_state_flips_applicable_criteria(spec):
    unsolved = {c.id for c in spec.applicable_criteria(False)}
    solved = {c.id for c in spec.applicable_criteria(True)}

    # Withholding the answer is required before the fix and forbidden after it.
    assert "solution_leak" in unsolved
    assert "solution_leak" not in solved
    assert "withheld_after_solved" in solved
    assert "withheld_after_solved" not in unsolved


def test_blocking_codes_differ_by_state(spec):
    assert "PREMATURE_CONFIRMATION" in spec.blocking_failure_codes(False)
    assert "PREMATURE_CONFIRMATION" not in spec.blocking_failure_codes(True)


def test_every_criterion_failure_code_is_in_the_taxonomy(spec):
    assert {c.failure_code for c in spec.criteria} <= set(spec.failure_codes)


def test_over_explanation_is_non_blocking(spec):
    """A long answer is a quality problem, not an automatic failure."""
    assert spec.criterion("over_explanation").blocking is False


def test_adversarial_pressure_types_cover_the_scenario_enum(spec):
    from evaluation.schemas import PressureType

    expected = {p.value for p in PressureType} - {"normal", "solved"}
    assert expected == set(spec.robustness.adversarial_pressure_types)


def test_gates_are_configuration_not_conclusions(spec):
    gate = spec.gates.prompt_ceiling
    assert 0.0 <= gate.required_spec_adherence <= 1.0
    assert gate.min_scenarios_per_cell >= 30
    assert gate.min_model_families >= 2
    assert gate.min_strategies >= 3


def test_render_for_prompt_mentions_every_criterion(spec):
    rendered = spec.render_for_prompt()
    for criterion in spec.criteria:
        assert criterion.id in rendered


def test_missing_spec_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Behavior spec not found"):
        load_spec(tmp_path / "nope.yaml")


def test_duplicate_criterion_ids_are_rejected(spec):
    payload = spec.model_dump(mode="python")
    payload["criteria"] = [payload["criteria"][0], payload["criteria"][0]]
    with pytest.raises(ValidationError, match="duplicate criterion ids"):
        BehaviorSpec.model_validate(payload)


def test_unknown_failure_code_is_rejected(spec):
    payload = spec.model_dump(mode="python")
    payload["criteria"][0]["failure_code"] = "NOT_IN_TAXONOMY"
    with pytest.raises(ValidationError, match="absent from the taxonomy"):
        BehaviorSpec.model_validate(payload)


def test_spec_file_is_valid_yaml_mapping():
    from behavior.spec import SPEC_PATH

    data = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "criteria" in data


def test_applies_when_matching():
    assert AppliesWhen.ALWAYS.matches(True)
    assert AppliesWhen.ALWAYS.matches(False)
    assert AppliesWhen.SOLVED.matches(True)
    assert not AppliesWhen.SOLVED.matches(False)
    assert AppliesWhen.UNSOLVED.matches(False)
    assert not AppliesWhen.UNSOLVED.matches(True)


def test_criteria_kinds_are_well_formed(spec):
    for criterion in spec.criteria:
        assert criterion.kind in (CriterionKind.VIOLATION, CriterionKind.SCORE)
        if criterion.kind is CriterionKind.SCORE:
            assert not criterion.blocking or criterion.id == "withheld_after_solved"
