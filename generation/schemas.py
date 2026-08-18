"""Schemas for teacher-generated training data.

A generated example is a *scenario plus the tutor turn it deserves*. Because the
scenario is a full `Scenario`, the quality gate can judge a training candidate
with exactly the same criteria the evaluation harness uses on a model response.
Training data and evaluation therefore cannot drift apart.

Every record carries provenance: which teacher produced it, under which prompt
version, from which template and dimension draw, at what time, for which dataset
version. Records with missing provenance are rejected, not repaired.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from evaluation.schemas import DeterministicResult, JudgeResult, Scenario

GENERATION_SCHEMA_VERSION = "1.0.0"


# =============================================================================
# Controlled dimensions
# =============================================================================


class LearnerCompetence(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LearnerFrustration(str, Enum):
    NONE = "none"
    MILD = "mild"
    HIGH = "high"


class HintStrength(str, Enum):
    NUDGE = "nudge"          # widen attention, name no component
    POINTED = "pointed"      # name the region, not the defect
    NARROW = "narrow"        # a single observation away from the answer


class StudentProgress(str, Enum):
    STUCK = "stuck"
    PARTIAL = "partial"
    CLOSE = "close"
    SOLVED = "solved"


class GenerationDimensions(BaseModel):
    """One point in the controlled generation space.

    Sampling explicitly across these rather than asking a teacher for "2000
    examples" is what keeps the dataset from collapsing onto one bug, one
    register and one hint style.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    language: str
    bug_category: str
    difficulty: str
    pressure_type: str
    conversation_turns: int = Field(ge=0, le=4)
    learner_competence: LearnerCompetence
    learner_frustration: LearnerFrustration
    hint_strength: HintStrength
    student_progress: StudentProgress

    def cell_key(self) -> tuple[str, ...]:
        """The balancing key: the axes that must not skew."""
        return (self.language, self.bug_category, self.pressure_type, self.difficulty)

    def slug(self) -> str:
        return "-".join(
            [
                self.language[:2],
                re.sub(r"[^a-z0-9]+", "", self.bug_category)[:12],
                self.pressure_type[:6],
                str(self.conversation_turns),
                self.student_progress.value[:4],
            ]
        )


# =============================================================================
# Provenance
# =============================================================================


class Provenance(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    teacher_model: str
    teacher_revision: str = "unknown"
    generation_prompt_version: str
    generation_prompt_sha256: str = ""
    template_id: str
    generation_seed: int
    dataset_version: str
    behavior_spec_version: str
    behavior_spec_sha256: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("teacher_model", "generation_prompt_version", "template_id",
                     "dataset_version")
    @classmethod
    def _required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provenance field must not be empty")
        return v


# =============================================================================
# Candidate
# =============================================================================


class RejectionCode(str, Enum):
    SOLUTION_LEAK = "SOLUTION_LEAK"
    EXPLICIT_FINAL_DIAGNOSIS = "EXPLICIT_FINAL_DIAGNOSIS"
    MULTIPLE_HINTS = "MULTIPLE_HINTS"
    IRRELEVANT_HINT = "IRRELEVANT_HINT"
    INCORRECT_DIAGNOSIS = "INCORRECT_DIAGNOSIS"
    OVER_EXPLANATION = "OVER_EXPLANATION"
    PREMATURE_CONFIRMATION = "PREMATURE_CONFIRMATION"
    WITHHELD_AFTER_SOLVED = "WITHHELD_AFTER_SOLVED"
    DUPLICATE = "DUPLICATE"
    LOW_QUALITY = "LOW_QUALITY"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNBALANCED = "UNBALANCED"
    CONTAMINATED = "CONTAMINATED"
    GENERATION_ERROR = "GENERATION_ERROR"


class GeneratedExample(BaseModel):
    """One teacher candidate, before or after the quality gate."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    id: str
    scenario: Scenario
    tutor_response: str
    dimensions: GenerationDimensions
    provenance: Provenance

    # Filled in by the quality gate.
    accepted: bool | None = None
    rejection_codes: tuple[str, ...] = ()
    deterministic: DeterministicResult | None = None
    judge: JudgeResult | None = None
    duplicate_of: str | None = None
    gate_notes: str = ""

    @field_validator("tutor_response")
    @classmethod
    def _response_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tutor_response must not be empty")
        return v

    # ---------------------------------------------------------------- helpers

    def content_hash(self) -> str:
        """Identity for exact-duplicate detection and contamination checks."""
        payload = "\n".join(
            [
                self.scenario.content_hash(),
                re.sub(r"\s+", " ", self.tutor_response.strip().lower()),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_training_messages(self) -> list[dict[str, str]]:
        """Chat-format turns: conversation in, tutor turn out."""
        from prompting.strategies import render_conversation

        messages = [
            {"role": m.role.value, "content": m.content}
            for m in render_conversation(self.scenario)
        ]
        messages.append({"role": "assistant", "content": self.tutor_response.strip()})
        return messages

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.scenario.language.value,
            "bug_category": self.scenario.bug_category,
            "difficulty": self.scenario.difficulty.value,
            "pressure_type": self.scenario.pressure_type.value,
            "turns": self.scenario.turn_count,
            "student_has_solved": self.scenario.student_has_solved,
            "accepted": self.accepted,
            "rejection_codes": list(self.rejection_codes),
        }


class GenerationBatchStats(BaseModel):
    """What one generation run produced, before filtering."""

    model_config = {"extra": "forbid"}

    requested: int = 0
    returned: int = 0
    parse_failures: int = 0
    schema_failures: int = 0
    provider_errors: int = 0
    #: Candidates reused from a prior interrupted run rather than re-purchased.
    reused: int = 0
    #: Candidates actually bought during this invocation.
    generated_this_run: int = 0
    #: Provider token accounting, when the run was metered.
    usage: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str = ""
    teacher_model: str = ""
    elapsed_s: float = 0.0
    failure_examples: list[str] = Field(default_factory=list)


__all__ = [
    "GENERATION_SCHEMA_VERSION",
    "GeneratedExample",
    "GenerationBatchStats",
    "GenerationDimensions",
    "HintStrength",
    "LearnerCompetence",
    "LearnerFrustration",
    "Provenance",
    "RejectionCode",
    "StudentProgress",
]
