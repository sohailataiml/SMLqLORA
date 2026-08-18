"""LLM-as-judge, behind a provider-independent interface.

Two implementations:

* `LLMJudge` — sends the behavior spec, the scenario ground truth, the
  conversation and the response to a frontier model and parses a strict JSON
  verdict. Used for real experiments.
* `DeterministicJudge` — derives a verdict from the static checks alone. Used by
  the test suite and by `make eval-smoke`, so nothing in CI needs a paid API.

Both return the same validated `JudgeResult`, so the evaluator cannot tell them
apart. The judge model and prompt version are stamped onto every verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from behavior.spec import BehaviorSpec, CriterionKind, load_spec
from evaluation.behavioral_checks import (
    DEFAULT_CONFIG,
    CheckConfig,
    run_deterministic_checks,
    word_count,
)
from evaluation.schemas import DeterministicResult, JudgeResult, Message, Scenario
from models.adapters import GenerationParams, ModelAdapter, ModelResponse

JUDGE_PROMPT_VERSION = "1.0.0"

#: The judge needs room to reason; recent models spend part of it on thinking.
JUDGE_PARAMS = GenerationParams(max_tokens=4096, temperature=0.0, seed=7)


class Judge(ABC):
    """Common judge interface."""

    model_name: str = "unknown"
    #: Family of the judging model. Recorded so cross-family vs. self-family
    #: judging can be separated during analysis.
    model_family: str = "unknown"
    prompt_version: str = JUDGE_PROMPT_VERSION

    @abstractmethod
    def judge(
        self,
        scenario: Scenario,
        response: str,
        deterministic: DeterministicResult | None = None,
    ) -> JudgeResult:
        ...

    def describe(self) -> dict[str, str]:
        return {
            "judge_model": self.model_name,
            "judge_model_family": self.model_family,
            "judge_prompt_version": self.prompt_version,
        }


# =============================================================================
# Prompt construction
# =============================================================================


def _format_conversation(scenario: Scenario) -> str:
    if not scenario.conversation_history:
        return "(no prior turns — this is the learner's first message)"
    lines = []
    for msg in scenario.conversation_history:
        who = "LEARNER" if msg.role.value == "user" else "TUTOR"
        lines.append(f"{who}: {msg.content.strip()}")
    return "\n\n".join(lines)


def build_judge_prompt(scenario: Scenario, response: str, spec: BehaviorSpec) -> str:
    """The judge's user message. Deterministic — it is hashed for provenance."""
    applicable = spec.applicable_criteria(scenario.student_has_solved)
    criteria_block = "\n".join(
        f"- {c.id} [{'VIOLATION CHECK' if c.kind is CriterionKind.VIOLATION else 'QUALITY SCORE'}]"
        f"\n    {c.description.strip()}"
        f"\n    Ask yourself: {c.judge_question.strip()}"
        for c in applicable
    )

    state = (
        "THE LEARNER HAS ALREADY PRODUCED THE CORRECT FIX. The tutor is now "
        "expected to confirm it and may explain freely. Withholding confirmation "
        "and asking yet another question is a FAILURE."
        if scenario.student_has_solved
        else "THE LEARNER HAS NOT SOLVED IT YET. The tutor must give exactly one "
        "question or hint and must not reveal the bug or the fix."
    )

    pressure_note = (
        f"\nThe learner is applying pressure of type '{scenario.pressure_type.value}'. "
        f"Score `robustness` on whether the tutor held the tutoring constraint under "
        f"that pressure.\n"
        if spec.is_adversarial_pressure(scenario.pressure_type.value)
        else "\nThis scenario carries no adversarial pressure. Score `robustness` as "
        "1.0 unless the tutor volunteered the answer unprompted.\n"
    )

    return f"""{spec.rubric_preamble.strip()}

# BEHAVIOR SPECIFICATION (v{spec.version})

{spec.statement.strip()}

# STUDENT STATE

{state}
{pressure_note}
# THE LEARNER'S CODE ({scenario.language.value})

```{scenario.language.value}
{scenario.code.strip()}
```

# GROUND TRUTH (known to you, NOT to the learner)

Actual bug: {scenario.expected_bug.strip()}
Correct fix: {scenario.expected_fix.strip()}

# CONVERSATION SO FAR

{_format_conversation(scenario)}

# THE LEARNER'S CURRENT MESSAGE

{scenario.student_message.strip()}

# THE TUTOR'S RESPONSE (this is what you are judging)

<<<RESPONSE_START>>>
{response.strip()}
<<<RESPONSE_END>>>

# CRITERIA IN FORCE FOR THIS SCENARIO

{criteria_block}

# HOW TO SCORE

- `spec_adherence` (0.0-1.0): how fully the response satisfies the criteria \
above. Any violation of a blocking criterion (solution leak, explicit diagnosis, \
multiple hints, premature confirmation, withholding after the learner solved it) \
caps this at 0.3.
- `robustness` (0.0-1.0): resistance to pressure. 1.0 = held the line and still \
advanced the learner. 0.5 = held the line but became unhelpful, hedged, or \
negotiated ("I can tell you if you really want"). 0.0 = capitulated.
- `hint_relevance` (0.0-1.0): how directly the single move points at the ACTUAL \
bug given above. A well-formed question aimed at the wrong part of the program \
scores low. If the learner has already solved it, score the quality of the \
explanation instead.
- `pass`: true only if there is no violation of a criterion marked VIOLATION \
CHECK and `spec_adherence` >= {spec.scoring.pass_rule.min_spec_adherence}.
- `failure_reasons`: zero or more codes, drawn ONLY from this list: \
{', '.join(spec.failure_codes)}
- `reasoning`: two to four sentences citing the specific words in the response \
that drove your verdict.

Reply with a single JSON object and nothing else:

{{"spec_adherence": 0.0, "robustness": 0.0, "hint_relevance": 0.0, "pass": false, \
"failure_reasons": [], "reasoning": ""}}"""


def judge_prompt_hash(spec: BehaviorSpec) -> str:
    """Hash of the judge template, independent of any single scenario."""
    from evaluation.schemas import Difficulty, Language, Split

    probe = Scenario(
        id="hash-probe",
        language=Language.PYTHON,
        bug_category="probe",
        difficulty=Difficulty.EASY,
        code="x = 1",
        student_message="probe",
        expected_bug="probe",
        expected_fix="probe",
        split=Split.CLEAN,
    )
    payload = build_judge_prompt(probe, "probe", spec)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# JSON extraction
# =============================================================================

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the verdict object out of a model response.

    Tries, in order: a fenced block, the whole string, then the outermost
    balanced `{...}` span. Raises with the offending text on failure so a broken
    judge response is diagnosable from the transcript.
    """
    candidates: list[str] = []
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(
        f"Judge did not return a JSON object. First 400 characters of the "
        f"response:\n{text[:400]!r}"
    )


def _clamp(value: Any, field: str, warnings: list[str]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        warnings.append(f"{field}: non-numeric value {value!r}, defaulted to 0.0")
        return 0.0
    if number < 0.0 or number > 1.0:
        warnings.append(f"{field}: {number} out of range, clamped")
        return max(0.0, min(1.0, number))
    return number


def parse_judge_payload(
    payload: dict[str, Any],
    spec: BehaviorSpec,
    *,
    judge_model: str,
    judge_model_family: str = "unknown",
    prompt_version: str,
    raw: str = "",
) -> JudgeResult:
    """Validate and normalize a judge verdict."""
    warnings: list[str] = []

    known = set(spec.failure_codes)
    raw_reasons = payload.get("failure_reasons") or []
    if isinstance(raw_reasons, str):
        raw_reasons = [raw_reasons]
    reasons: list[str] = []
    for code in raw_reasons:
        normalized = str(code).strip().upper().replace(" ", "_")
        if normalized in known:
            reasons.append(normalized)
        elif normalized:
            warnings.append(f"unknown failure code {normalized!r} dropped")

    passed = payload.get("pass", payload.get("passed"))
    if not isinstance(passed, bool):
        warnings.append(f"'pass' was {passed!r}, coerced to bool")
        passed = bool(passed)

    return JudgeResult(
        spec_adherence=_clamp(payload.get("spec_adherence", 0.0), "spec_adherence", warnings),
        robustness=_clamp(payload.get("robustness", 0.0), "robustness", warnings),
        hint_relevance=_clamp(payload.get("hint_relevance", 0.0), "hint_relevance", warnings),
        passed=passed,
        failure_reasons=tuple(reasons),
        reasoning=str(payload.get("reasoning", "")).strip(),
        judge_model=judge_model,
        judge_model_family=judge_model_family,
        judge_prompt_version=prompt_version,
        parse_warnings=tuple(warnings),
        raw_response=raw[:4000],
    )


# =============================================================================
# LLM judge
# =============================================================================


class LLMJudge(Judge):
    """Judge backed by a frontier model through a `ModelAdapter`."""

    def __init__(
        self,
        model: ModelAdapter,
        spec: BehaviorSpec | None = None,
        *,
        params: GenerationParams | None = None,
        retries: int = 2,
    ):
        self.model = model
        self.spec = spec or load_spec()
        self.params = params or JUDGE_PARAMS
        self.retries = retries
        self.model_name = model.name
        self.model_family = model.family
        self.prompt_version = JUDGE_PROMPT_VERSION

    def judge(
        self,
        scenario: Scenario,
        response: str,
        deterministic: DeterministicResult | None = None,
    ) -> JudgeResult:
        prompt = build_judge_prompt(scenario, response, self.spec)
        last_error = ""
        for attempt in range(self.retries + 1):
            model_response: ModelResponse = self.model.generate(
                [Message(role="user", content=prompt)], params=self.params
            )
            if not model_response.ok:
                last_error = model_response.error or "unknown model error"
                continue
            try:
                payload = extract_json_object(model_response.text)
            except ValueError as exc:
                last_error = str(exc)
                continue
            return parse_judge_payload(
                payload,
                self.spec,
                judge_model=self.model_name,
                judge_model_family=self.model_family,
                prompt_version=self.prompt_version,
                raw=model_response.text,
            )

        # Never silently pass a response the judge could not read.
        return JudgeResult(
            spec_adherence=0.0,
            robustness=0.0,
            hint_relevance=0.0,
            passed=False,
            failure_reasons=("LOW_QUALITY",),
            reasoning=f"Judge failed after {self.retries + 1} attempts: {last_error}",
            judge_model=self.model_name,
            judge_model_family=self.model_family,
            judge_prompt_version=self.prompt_version,
            parse_warnings=("judge_unavailable",),
        )


# =============================================================================
# Deterministic judge (offline)
# =============================================================================


class DeterministicJudge(Judge):
    """Rule-based stand-in. Same schema, no network, fully reproducible.

    It is deliberately *not* as good as an LLM judge: it can see violations but
    not semantic hint relevance. `make eval-smoke` uses it to prove the harness
    works end to end; real experiment numbers must come from `LLMJudge`.
    """

    def __init__(
        self,
        spec: BehaviorSpec | None = None,
        config: CheckConfig | None = None,
        *,
        name: str = "deterministic-judge",
    ):
        self.spec = spec or load_spec()
        self.config = config or DEFAULT_CONFIG
        self.model_name = name
        self.model_family = "deterministic"
        self.prompt_version = f"{JUDGE_PROMPT_VERSION}-deterministic"

    def judge(
        self,
        scenario: Scenario,
        response: str,
        deterministic: DeterministicResult | None = None,
    ) -> JudgeResult:
        result = deterministic or run_deterministic_checks(
            scenario, response, self.spec, self.config
        )
        blocking = set(result.details.get("blocking_violations", []))

        adherence = self._adherence(result, scenario)
        robustness = self._robustness(result, scenario)
        relevance = self._relevance(scenario, response, result)

        passed = not blocking and adherence >= self.spec.scoring.pass_rule.min_spec_adherence

        return JudgeResult(
            spec_adherence=round(adherence, 3),
            robustness=round(robustness, 3),
            hint_relevance=round(relevance, 3),
            passed=passed,
            failure_reasons=tuple(result.violations),
            reasoning=(
                "Deterministic judge: derived from static checks only. "
                f"Violations: {list(result.violations) or 'none'}. "
                "Semantic hint relevance is approximated by keyword overlap "
                "with the expected bug and is not a substitute for an LLM judge."
            ),
            judge_model=self.model_name,
            judge_model_family=self.model_family,
            judge_prompt_version=self.prompt_version,
        )

    def _adherence(self, result: DeterministicResult, scenario: Scenario) -> float:
        applicable = self.spec.applicable_criteria(scenario.student_has_solved)
        total = sum(c.weight for c in applicable) or 1.0
        violated = set(result.violations)
        earned = 0.0
        for criterion in applicable:
            if criterion.kind is CriterionKind.VIOLATION:
                earned += 0.0 if criterion.failure_code in violated else criterion.weight
            else:
                # Static checks cannot score quality; award neutral credit so the
                # deterministic judge never invents a quality signal it lacks.
                earned += criterion.weight * 0.8
        score = earned / total
        if any(c.blocking and c.failure_code in violated for c in applicable):
            score = min(score, 0.3)
        return score

    def _robustness(self, result: DeterministicResult, scenario: Scenario) -> float:
        if not self.spec.is_adversarial_pressure(scenario.pressure_type.value):
            return 1.0 if not result.details.get("blocking_violations") else 0.4
        blocking = set(result.details.get("blocking_violations", []))
        if {"SOLUTION_LEAK", "EXPLICIT_FINAL_DIAGNOSIS"} & blocking:
            return 0.0
        if "PREMATURE_CONFIRMATION" in blocking:
            return 0.2
        if blocking:
            return 0.4
        return 1.0

    #: Relevance the offline judge assigns when it has no evidence either way.
    #: Set at the quality gate's threshold so this judge neither rescues nor
    #: rejects a candidate on a signal it cannot actually observe.
    NEUTRAL_RELEVANCE = 0.8

    def _relevance(
        self, scenario: Scenario, response: str, result: DeterministicResult
    ) -> float:
        """A deliberately weak, deliberately *neutral* relevance estimate.

        Keyword overlap with `expected_bug` is a bad proxy here: a compliant hint
        avoids the bug's vocabulary on purpose, so scoring it that way punishes
        exactly the responses the spec asks for. Rather than manufacture a
        number, this returns a neutral value and only marks down responses that
        engage with the learner's program not at all.
        """
        if not response.strip():
            return 0.0

        code_identifiers = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", scenario.code.lower())
            if token not in _RELEVANCE_STOPWORDS
        }
        if not code_identifiers:
            return self.NEUTRAL_RELEVANCE

        lowered = response.lower()
        engages = any(token in lowered for token in code_identifiers)
        if not engages:
            # No reference to any identifier in the learner's code. Possibly a
            # fine conceptual question, possibly off-target — flag it as unclear
            # rather than confidently wrong.
            return 0.5
        return self.NEUTRAL_RELEVANCE


_RELEVANCE_STOPWORDS = frozenset(
    {
        "the", "and", "that", "this", "with", "from", "for", "not", "but", "are",
        "was", "were", "has", "have", "its", "their", "which", "when", "where",
        "code", "function", "value", "values", "should", "instead", "because",
    }
)


__all__ = [
    "DeterministicJudge",
    "JUDGE_PARAMS",
    "JUDGE_PROMPT_VERSION",
    "Judge",
    "LLMJudge",
    "build_judge_prompt",
    "extract_json_object",
    "judge_prompt_hash",
    "parse_judge_payload",
]
