"""The teacher: turns one dimension draw into one validated candidate.

Parsing is strict. A teacher response that does not produce a schema-valid
`Scenario` is a rejected candidate with a recorded reason, never a silently
patched one — repairing malformed generations is how quiet bias enters a dataset.
"""

from __future__ import annotations

import json
import re
from typing import Any

from behavior.spec import BehaviorSpec, load_spec
from evaluation.judge import extract_json_object
from evaluation.schemas import Message, Role, Scenario, Split
from generation.prompts import (
    GENERATION_PROMPT_VERSION,
    TEACHER_SYSTEM_PROMPT,
    build_generation_prompt,
    generation_prompt_hash,
)
from generation.schemas import (
    GeneratedExample,
    GenerationDimensions,
    Provenance,
    StudentProgress,
)
from models.adapters import GenerationParams, ModelAdapter

#: Teachers need room for hidden reasoning plus a full JSON object.
TEACHER_PARAMS = GenerationParams(max_tokens=6000, temperature=1.0, seed=None)

_REQUIRED_KEYS = (
    "code",
    "student_message",
    "expected_bug",
    "expected_fix",
    "tutor_response",
)


class TeacherError(RuntimeError):
    """A candidate could not be produced. Carries a rejection code."""

    def __init__(self, message: str, code: str = "GENERATION_ERROR"):
        super().__init__(message)
        self.code = code


def _clean_code_fence(text: str) -> str:
    """Teachers sometimes wrap the snippet in a fence despite instructions."""
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z]*\n(.*?)```$", stripped, re.S)
    return match.group(1).strip() if match else stripped


def _parse_history(raw: Any, expected_turns: int) -> tuple[Message, ...]:
    if raw in (None, "", []):
        if expected_turns:
            raise TeacherError(
                f"expected {expected_turns} prior exchange(s), got none",
                "INVALID_SCHEMA",
            )
        return ()
    if not isinstance(raw, list):
        raise TeacherError("conversation_history must be a list", "INVALID_SCHEMA")

    messages: list[Message] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TeacherError(
                f"conversation_history[{index}] is not an object", "INVALID_SCHEMA"
            )
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            raise TeacherError(
                f"conversation_history[{index}] has role={role!r} and "
                f"{len(content)} chars of content",
                "INVALID_SCHEMA",
            )
        messages.append(Message(role=Role(role), content=content))

    if len(messages) % 2 != 0:
        raise TeacherError(
            f"conversation_history has {len(messages)} messages; prior exchanges "
            f"must be complete user/assistant pairs",
            "INVALID_SCHEMA",
        )
    return tuple(messages)


def build_example(
    payload: dict[str, Any],
    dimensions: GenerationDimensions,
    *,
    seed: int,
    example_id: str,
    teacher: ModelAdapter,
    dataset_version: str,
    spec: BehaviorSpec,
) -> GeneratedExample:
    """Validate a parsed teacher payload into a candidate."""
    missing = [k for k in _REQUIRED_KEYS if not str(payload.get(k, "")).strip()]
    if missing:
        raise TeacherError(f"missing or empty keys: {missing}", "INVALID_SCHEMA")

    solved = dimensions.student_progress is StudentProgress.SOLVED
    history = _parse_history(payload.get("conversation_history"),
                             dimensions.conversation_turns)

    try:
        scenario = Scenario(
            id=example_id,
            language=dimensions.language,
            bug_category=dimensions.bug_category,
            difficulty=dimensions.difficulty,
            code=_clean_code_fence(str(payload["code"])),
            conversation_history=history,
            student_message=str(payload["student_message"]).strip(),
            expected_bug=str(payload["expected_bug"]).strip(),
            expected_fix=str(payload["expected_fix"]).strip(),
            student_has_solved=solved,
            pressure_type=dimensions.pressure_type,
            source=f"teacher:{teacher.name}",
            split=Split.TRAIN,
        )
    except Exception as exc:  # pydantic ValidationError and friends
        raise TeacherError(f"scenario failed validation: {exc}", "INVALID_SCHEMA") from exc

    return GeneratedExample(
        id=example_id,
        scenario=scenario,
        tutor_response=str(payload["tutor_response"]).strip(),
        dimensions=dimensions,
        provenance=Provenance(
            teacher_model=teacher.name,
            teacher_revision=teacher.revision,
            generation_prompt_version=GENERATION_PROMPT_VERSION,
            generation_prompt_sha256=generation_prompt_hash(spec),
            template_id=f"socratic_v1:{dimensions.slug()}",
            generation_seed=seed,
            dataset_version=dataset_version,
            behavior_spec_version=spec.version,
            behavior_spec_sha256=spec.spec_sha256,
        ),
    )


class Teacher:
    """Frontier model wrapped as a candidate factory."""

    def __init__(
        self,
        model: ModelAdapter,
        *,
        spec: BehaviorSpec | None = None,
        dataset_version: str = "v1",
        params: GenerationParams | None = None,
        retries: int = 1,
    ):
        self.model = model
        self.spec = spec or load_spec()
        self.dataset_version = dataset_version
        self.params = params or TEACHER_PARAMS
        self.retries = retries

    def generate_one(
        self, seed: int, dimensions: GenerationDimensions, *, index: int
    ) -> GeneratedExample:
        prompt = build_generation_prompt(dimensions, self.spec, seed=seed)
        example_id = f"gen_{self.dataset_version}_{index:05d}"
        last_error = "no attempt made"
        # Carried out of the loop so the final error keeps the *cause*. Without
        # this every failure surfaces as GENERATION_ERROR and a malformed
        # teacher payload gets counted as a provider outage, which would let a
        # content problem masquerade as an infrastructure one.
        last_code = "GENERATION_ERROR"

        for _ in range(self.retries + 1):
            response = self.model.generate(
                [Message(role=Role.USER, content=prompt)],
                system=TEACHER_SYSTEM_PROMPT,
                params=self.params,
            )
            if not response.ok:
                last_error = response.error or "unknown provider error"
                last_code = "INFRASTRUCTURE"
                continue
            try:
                payload = extract_json_object(response.text)
            except ValueError as exc:
                last_error = f"unparseable JSON: {exc}"
                last_code = "UNPARSEABLE"
                continue
            try:
                return build_example(
                    payload,
                    dimensions,
                    seed=seed,
                    example_id=example_id,
                    teacher=self.model,
                    dataset_version=self.dataset_version,
                    spec=self.spec,
                )
            except TeacherError as exc:
                last_error = str(exc)
                last_code = exc.code
                continue

        raise TeacherError(
            f"{example_id}: teacher failed after {self.retries + 1} attempt(s) "
            f"({dimensions.slug()}): {last_error}",
            last_code,
        )

    def describe(self) -> dict[str, str]:
        return {
            **self.model.describe(),
            "generation_prompt_version": GENERATION_PROMPT_VERSION,
            "generation_prompt_sha256": generation_prompt_hash(self.spec)[:16],
            "dataset_version": self.dataset_version,
        }


__all__ = ["TEACHER_PARAMS", "Teacher", "TeacherError", "build_example"]
