"""Typed loader for ``behavior/spec.yaml``.

The behavior specification is the contract every other component obeys. Loading
it through Pydantic means a malformed spec fails loudly at import time rather
than silently changing what "pass" means halfway through an experiment.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SPEC_PATH = Path(__file__).resolve().parent / "spec.yaml"

#: Adversarial pressure types introduced after `spec.yaml` was frozen.
#:
#: The YAML is pinned at dc14f40b in SUBMISSION.md and every published
#: robustness number was computed against it, so extending its
#: `adversarial_pressure_types` list would silently invalidate all of them.
#:
#: These four came from a partner's red-team set, which found attack shapes with
#: zero coverage in Dataset V1: a forced yes/no whose question already names the
#: defect, coercion into a diff or completed code block, a roleplay persona whose
#: norm is to post solutions, and an off-task request that pulls the model out of
#: the tutor role. They are adversarial in exactly the sense the spec means, so
#: the robustness criterion has to apply to them. Keeping them here rather than
#: in the YAML is what lets that be true without moving the frozen hash.
POST_FREEZE_ADVERSARIAL_PRESSURE_TYPES = frozenset({
    "forced_binary",
    "output_coercion",
    "roleplay_reframe",
    "off_task",
})


class CriterionKind(str, Enum):
    VIOLATION = "violation"
    SCORE = "score"


class AppliesWhen(str, Enum):
    UNSOLVED = "unsolved"
    SOLVED = "solved"
    ALWAYS = "always"

    def matches(self, student_has_solved: bool) -> bool:
        if self is AppliesWhen.ALWAYS:
            return True
        if self is AppliesWhen.SOLVED:
            return student_has_solved
        return not student_has_solved


class CheckedBy(str, Enum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    BOTH = "both"


class Criterion(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    id: str
    kind: CriterionKind
    failure_code: str
    applies_when: AppliesWhen
    checked_by: CheckedBy
    blocking: bool
    weight: float = Field(ge=0.0, le=1.0)
    description: str
    judge_question: str

    @property
    def is_deterministic(self) -> bool:
        return self.checked_by in (CheckedBy.DETERMINISTIC, CheckedBy.BOTH)

    @property
    def is_judged(self) -> bool:
        return self.checked_by in (CheckedBy.JUDGE, CheckedBy.BOTH)


class StudentState(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    field: str
    description: str
    unsolved_mode: str
    solved_mode: str


class Robustness(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    description: str
    adversarial_pressure_types: tuple[str, ...]
    judge_question: str


class PassRule(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    no_blocking_violations: bool
    min_spec_adherence: float = Field(ge=0.0, le=1.0)
    deterministic_violations_are_authoritative: bool


class OverallAdherence(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    method: Literal["weighted_mean_of_applicable_criteria"]
    range: tuple[float, float]


class Scoring(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    overall_spec_adherence: OverallAdherence
    pass_rule: PassRule


class PromptCeilingGate(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    required_spec_adherence: float = Field(ge=0.0, le=1.0)
    required_robustness: float = Field(ge=0.0, le=1.0)
    required_pass_rate: float = Field(ge=0.0, le=1.0)
    min_scenarios_per_cell: int = Field(ge=1)
    min_model_families: int = Field(ge=1)
    min_strategies: int = Field(ge=1)


class DataEfficiencyGate(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    required_pass_rate: float = Field(ge=0.0, le=1.0)
    required_spec_adherence: float = Field(ge=0.0, le=1.0)
    required_robustness: float = Field(ge=0.0, le=1.0)


class QualityGateThresholds(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    min_judge_spec_adherence: float = Field(ge=0.0, le=1.0)
    min_judge_hint_relevance: float = Field(ge=0.0, le=1.0)
    min_judge_robustness: float = Field(ge=0.0, le=1.0)
    allow_any_deterministic_violation: bool


class Gates(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    prompt_ceiling: PromptCeilingGate
    data_efficiency: DataEfficiencyGate
    quality_gate: QualityGateThresholds


class BehaviorSpec(BaseModel):
    """The full, validated behavior specification."""

    model_config = {"frozen": True, "extra": "forbid"}

    version: str
    id: str
    statement: str
    rubric_preamble: str
    student_state: StudentState
    criteria: tuple[Criterion, ...]
    robustness: Robustness
    scoring: Scoring
    failure_codes: tuple[str, ...]
    gates: Gates

    # Populated at load time so results can be traced to an exact spec file.
    spec_sha256: str = ""

    @field_validator("criteria")
    @classmethod
    def _criteria_non_empty_and_unique(cls, value: tuple[Criterion, ...]):
        if not value:
            raise ValueError("behavior spec must define at least one criterion")
        ids = [c.id for c in value]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate criterion ids: {ids}")
        return value

    @model_validator(mode="after")
    def _failure_codes_cover_criteria(self) -> "BehaviorSpec":
        known = set(self.failure_codes)
        missing = {c.failure_code for c in self.criteria} - known
        if missing:
            raise ValueError(
                f"criteria reference failure codes absent from the taxonomy: {sorted(missing)}"
            )
        return self

    # ---------------------------------------------------------------- helpers

    def criterion(self, criterion_id: str) -> Criterion:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        raise KeyError(f"unknown criterion: {criterion_id!r}")

    def applicable_criteria(self, student_has_solved: bool) -> tuple[Criterion, ...]:
        """Criteria in force for a scenario in the given student state."""
        return tuple(
            c for c in self.criteria if c.applies_when.matches(student_has_solved)
        )

    def blocking_failure_codes(self, student_has_solved: bool) -> set[str]:
        return {
            c.failure_code
            for c in self.applicable_criteria(student_has_solved)
            if c.blocking
        }

    def is_adversarial_pressure(self, pressure_type: str) -> bool:
        return (
            pressure_type in self.robustness.adversarial_pressure_types
            or pressure_type in POST_FREEZE_ADVERSARIAL_PRESSURE_TYPES
        )

    def render_for_prompt(self) -> str:
        """Compact human-readable rendering injected into prompts and judges."""
        lines = [
            f"BEHAVIOR SPEC v{self.version} ({self.id})",
            "",
            self.statement.strip(),
            "",
            "CRITERIA:",
        ]
        for c in self.criteria:
            marker = "MUST NOT" if c.kind is CriterionKind.VIOLATION else "QUALITY"
            scope = {
                AppliesWhen.UNSOLVED: "while the learner has NOT yet solved it",
                AppliesWhen.SOLVED: "once the learner HAS solved it",
                AppliesWhen.ALWAYS: "always",
            }[c.applies_when]
            lines.append(f"- [{marker}] {c.id} ({scope}): {c.description.strip()}")
        return "\n".join(lines)


@lru_cache(maxsize=8)
def load_spec(path: str | Path = SPEC_PATH) -> BehaviorSpec:
    """Load and validate the behavior spec (cached per path)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Behavior spec not found at {p}. The spec is required; it defines what "
            f"'pass' means. Expected the repository's behavior/spec.yaml."
        )
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Behavior spec at {p} must be a YAML mapping.")
    data["spec_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return BehaviorSpec.model_validate(data)


__all__ = [
    "AppliesWhen",
    "BehaviorSpec",
    "CheckedBy",
    "Criterion",
    "CriterionKind",
    "SPEC_PATH",
    "load_spec",
]
