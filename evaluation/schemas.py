"""Strongly typed schemas shared by evaluation, generation and filtering.

Everything that crosses a module boundary or gets written to disk goes through
one of these models. A malformed scenario or an out-of-range judge score is an
error, not a silently-coerced value.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# =============================================================================
# Enumerations
# =============================================================================


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PressureType(str, Enum):
    """How the learner is pushing on the tutoring constraint."""

    NORMAL = "normal"
    FRUSTRATED = "frustrated"
    REPEATED_ANSWER_REQUEST = "repeated_answer_request"
    TIME_PRESSURE = "time_pressure"
    PROMPT_INJECTION = "prompt_injection"
    AUTHORITY_OVERRIDE = "authority_override"
    FAKE_SUCCESS = "fake_success"
    ALMOST_CORRECT = "almost_correct"
    SOLVED = "solved"


class ErrorKind(str, Enum):
    """Why a call produced no response.

    The distinction is load-bearing. A `refusal` is the model declining — a real
    behavioral data point that belongs in the denominator. An `infrastructure`
    failure (exhausted quota, rate limit, dropped connection) says nothing about
    the model and must be excluded from rates, or a billing outage silently
    reads as a model that never passes anything.
    """

    NONE = "none"
    REFUSAL = "refusal"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


_INFRASTRUCTURE_MARKERS = (
    "credit balance",
    "insufficient_quota",
    "exceeded your current quota",
    "rate_limit",
    "ratelimiterror",
    "429",
    "apiconnectionerror",
    "apitimeouterror",
    "connection reset",
    "connection error",
    "timed out",
    "overloaded",
    "503",
    "502",
    "internalservererror",
    "authenticationerror",
    "permissiondeniederror",
    "missingcredentials",
)


def classify_error(error: str | None) -> ErrorKind:
    """Bucket a provider error string. Pure and testable."""
    if not error:
        return ErrorKind.NONE
    lowered = error.lower()
    if "refusal" in lowered or "stop_reason=refusal" in lowered:
        return ErrorKind.REFUSAL
    if any(marker in lowered for marker in _INFRASTRUCTURE_MARKERS):
        return ErrorKind.INFRASTRUCTURE
    return ErrorKind.UNKNOWN


class Split(str, Enum):
    """Which pool a scenario belongs to. Splits never mix."""

    CLEAN = "clean"
    ADVERSARIAL = "adversarial"
    HELDOUT = "heldout"
    TRAIN = "train"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# =============================================================================
# Conversation
# =============================================================================


class Message(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message content must not be empty")
        return v


# =============================================================================
# Scenario
# =============================================================================

_ID_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


class Scenario(BaseModel):
    """One evaluation situation: broken code, a conversation, a learner turn."""

    model_config = {"frozen": True, "extra": "forbid"}

    id: str
    language: Language
    bug_category: str
    difficulty: Difficulty
    code: str
    conversation_history: tuple[Message, ...] = ()
    student_message: str
    expected_bug: str
    expected_fix: str
    student_has_solved: bool = False
    pressure_type: PressureType = PressureType.NORMAL
    source: str = "handwritten"
    split: Split
    notes: str = ""

    # ------------------------------------------------------------- validators

    @field_validator("id")
    @classmethod
    def _id_slug(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(
                f"scenario id {v!r} must be a lowercase slug (a-z, 0-9, '_' or '-')"
            )
        return v

    @field_validator("code", "student_message", "expected_bug", "expected_fix",
                     "bug_category")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v

    @field_validator("conversation_history")
    @classmethod
    def _history_well_formed(cls, v: tuple[Message, ...]) -> tuple[Message, ...]:
        """History must be prior turns only: alternating, ending on the tutor."""
        if not v:
            return v
        expected = Role.USER
        for i, msg in enumerate(v):
            if msg.role is Role.SYSTEM:
                raise ValueError(
                    "conversation_history must not contain system messages; the "
                    "system prompt is supplied by the prompt strategy"
                )
            if msg.role is not expected:
                raise ValueError(
                    f"conversation_history must alternate user/assistant starting "
                    f"with user; index {i} is {msg.role.value}, expected "
                    f"{expected.value}"
                )
            expected = Role.ASSISTANT if expected is Role.USER else Role.USER
        if v[-1].role is not Role.ASSISTANT:
            raise ValueError(
                "conversation_history must end with an assistant turn; the new "
                "learner turn belongs in `student_message`"
            )
        return v

    @model_validator(mode="after")
    def _state_consistent_with_pressure(self) -> "Scenario":
        if self.pressure_type is PressureType.SOLVED and not self.student_has_solved:
            raise ValueError(
                "pressure_type 'solved' requires student_has_solved=True"
            )
        if self.pressure_type is PressureType.FAKE_SUCCESS and self.student_has_solved:
            raise ValueError(
                "pressure_type 'fake_success' means the learner CLAIMS a fix they "
                "did not make; student_has_solved must be False"
            )
        if self.student_has_solved and self.pressure_type not in (
            PressureType.SOLVED,
            PressureType.NORMAL,
        ):
            raise ValueError(
                f"student_has_solved=True is incompatible with pressure_type "
                f"{self.pressure_type.value!r}"
            )
        return self

    # ---------------------------------------------------------------- helpers

    @property
    def is_multi_turn(self) -> bool:
        return len(self.conversation_history) > 0

    @property
    def turn_count(self) -> int:
        """Number of learner turns including the current one."""
        return len([m for m in self.conversation_history if m.role is Role.USER]) + 1

    def content_hash(self) -> str:
        """Stable hash over the semantic content (not the id or split).

        Used for train/eval contamination detection: two scenarios with the same
        code + student message are the same scenario however they are labelled.
        """
        payload = "\n".join(
            [
                _normalize_ws(self.code),
                _normalize_ws(self.student_message),
                "\n".join(_normalize_ws(m.content) for m in self.conversation_history),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_messages(self) -> list[Message]:
        """Conversation as sent to a model, excluding any system prompt."""
        return [*self.conversation_history, Message(role=Role.USER,
                                                    content=self.student_message)]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# =============================================================================
# Deterministic check result
# =============================================================================


class DeterministicResult(BaseModel):
    """Outcome of the static, non-LLM behavioral checks."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    passed: bool = Field(
        validation_alias=AliasChoices("passed", "pass"), serialization_alias="pass"
    )
    violations: tuple[str, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)
    checks_version: str = "1.0.0"

    @model_validator(mode="after")
    def _pass_matches_violations(self) -> "DeterministicResult":
        # `violations` may include non-blocking observations, so `passed` is set
        # by the checker, not inferred here. But passing with a blocking
        # violation recorded would be a contradiction the checker must resolve.
        if self.passed and self.details.get("blocking_violations"):
            raise ValueError(
                "DeterministicResult cannot pass while blocking_violations is non-empty"
            )
        return self


# =============================================================================
# Judge result
# =============================================================================


class JudgeResult(BaseModel):
    """Structured verdict from an LLM judge (or a deterministic fake judge)."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    spec_adherence: float = Field(ge=0.0, le=1.0)
    robustness: float = Field(ge=0.0, le=1.0)
    hint_relevance: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(
        validation_alias=AliasChoices("passed", "pass"), serialization_alias="pass"
    )
    failure_reasons: tuple[str, ...] = ()
    reasoning: str = ""

    judge_model: str = "unknown"
    #: Recorded separately from `judge_model` so self-evaluation bias is
    #: detectable by grouping: a record whose judge family equals its subject
    #: family was graded by its own kind.
    judge_model_family: str = "unknown"
    judge_prompt_version: str = "unknown"
    parse_warnings: tuple[str, ...] = ()
    raw_response: str = ""

    @field_validator("failure_reasons")
    @classmethod
    def _codes_upper(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(code.strip().upper() for code in v if code.strip()))


# =============================================================================
# Per-example evaluation record (the raw judge transcript row)
# =============================================================================


class EvalRecord(BaseModel):
    """Everything needed to audit a single model/scenario evaluation."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    scenario_id: str
    scenario_split: Split
    pressure_type: PressureType
    language: Language
    bug_category: str
    difficulty: Difficulty
    student_has_solved: bool

    model: str
    model_family: str
    model_revision: str
    prompt_strategy: str
    prompt_version: str

    input_messages: tuple[Message, ...]
    model_response: str

    deterministic: DeterministicResult
    judge: JudgeResult | None = None

    failure_reasons: tuple[str, ...] = ()
    passed: bool = Field(
        validation_alias=AliasChoices("passed", "pass"), serialization_alias="pass"
    )

    behavior_spec_version: str
    behavior_spec_sha256: str = ""
    generation_params: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def _derive_error_kind(self) -> "EvalRecord":
        # Records written before `error_kind` existed still carry the string.
        if self.error and self.error_kind is ErrorKind.NONE:
            object.__setattr__(self, "error_kind", classify_error(self.error))
        return self

    @property
    def is_adversarial(self) -> bool:
        return self.pressure_type is not PressureType.NORMAL

    @property
    def was_evaluated(self) -> bool:
        """False when infrastructure prevented the model from being measured."""
        return self.error_kind is not ErrorKind.INFRASTRUCTURE


# =============================================================================
# Aggregated metrics
# =============================================================================


class CellMetrics(BaseModel):
    """Metrics for one (model x prompt strategy) cell, or one checkpoint."""

    model_config = {"extra": "forbid"}

    model: str
    model_family: str
    prompt_strategy: str
    #: Records that actually measured the model (excludes infrastructure failures).
    #: This is the denominator of every behavioral rate below.
    scenario_count: int
    #: Scenarios attempted, including those lost to infrastructure failures.
    attempted_count: int = 0
    infrastructure_error_count: int = 0
    #: Subject calls that returned a body (no error of any kind).
    successful_subject_calls: int = 0
    #: Judge calls that returned a parseable verdict. Lower than
    #: `successful_subject_calls` when the judge itself failed.
    successful_judge_calls: int = 0
    #: Records served from a previous run rather than re-purchased.
    reused_count: int = 0
    #: True when infrastructure failures make this cell incomplete.
    partial: bool = False

    spec_adherence_mean: float
    robustness_mean: float
    hint_relevance_mean: float
    pass_rate: float
    failure_rate: float

    solution_leak_rate: float
    premature_confirmation_rate: float
    multiple_hints_rate: float

    adversarial_pass_rate: float | None = None
    clean_pass_rate: float | None = None

    failure_modes: dict[str, int] = Field(default_factory=dict)
    error_count: int = 0
    label: str = ""
    notes: str = ""


# =============================================================================
# JSONL helpers
# =============================================================================


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Write pydantic models as JSONL (alias-serialized). Returns row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json", by_alias=True),
                                ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def append_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json", by_alias=True),
                                ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: malformed JSON — {exc}") from exc


class ScenarioLoadError(ValueError):
    """Raised with a per-line explanation when a scenario file is malformed."""


def load_scenarios(path: str | Path, *, strict: bool = True) -> list[Scenario]:
    """Load and validate scenarios from a JSONL file.

    Raises ScenarioLoadError with the offending line number and the validation
    problem, so a malformed eval set is diagnosable rather than mysterious.
    """
    p = Path(path)
    if not p.exists():
        raise ScenarioLoadError(
            f"Scenario file not found: {p}\n"
            f"Expected a JSONL file (one scenario object per line). "
            f"See scenarios/clean.jsonl for the format."
        )
    scenarios: list[Scenario] = []
    problems: list[str] = []
    for lineno, obj in enumerate(iter_jsonl(p), start=1):
        try:
            scenarios.append(Scenario.model_validate(obj))
        except ValidationError as exc:
            sid = obj.get("id", "<no id>") if isinstance(obj, dict) else "<not an object>"
            problems.append(f"  line {lineno} (id={sid}): {_short_validation(exc)}")
    if problems:
        msg = f"{len(problems)} malformed scenario(s) in {p}:\n" + "\n".join(problems)
        if strict:
            raise ScenarioLoadError(msg)
    if not scenarios:
        raise ScenarioLoadError(f"No valid scenarios found in {p}")

    ids = [s.id for s in scenarios]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ScenarioLoadError(f"Duplicate scenario ids in {p}: {sorted(dupes)}")
    return scenarios


def load_scenario_files(paths: Sequence[str | Path]) -> list[Scenario]:
    """Load several scenario files, rejecting cross-file id collisions."""
    out: list[Scenario] = []
    seen: dict[str, str] = {}
    for path in paths:
        for scenario in load_scenarios(path):
            if scenario.id in seen:
                raise ScenarioLoadError(
                    f"Scenario id {scenario.id!r} appears in both {seen[scenario.id]} "
                    f"and {path}"
                )
            seen[scenario.id] = str(path)
            out.append(scenario)
    return out


def _short_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenarios_hash(scenarios: Sequence[Scenario]) -> str:
    """Order-independent hash of a scenario set."""
    digest = hashlib.sha256()
    for h in sorted(s.content_hash() for s in scenarios):
        digest.update(h.encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "CellMetrics",
    "DeterministicResult",
    "Difficulty",
    "ErrorKind",
    "EvalRecord",
    "classify_error",
    "JudgeResult",
    "Language",
    "Message",
    "PressureType",
    "Role",
    "Scenario",
    "ScenarioLoadError",
    "Split",
    "append_jsonl",
    "file_sha256",
    "iter_jsonl",
    "load_scenario_files",
    "load_scenarios",
    "scenarios_hash",
    "write_jsonl",
]
