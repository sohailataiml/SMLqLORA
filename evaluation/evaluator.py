"""The evaluation runner: scenario in, auditable verdict out.

One `Evaluator` binds a model, a prompt strategy and a judge. Calling
`evaluate()` produces one `EvalRecord` per scenario containing everything needed
to re-examine the decision later — the exact input, the raw response, the static
checks, the judge's scores and reasoning, and the final pass/fail.

Combination rule (from `behavior/spec.yaml`):

* a blocking deterministic violation is authoritative and fails the response;
* otherwise the judge decides, subject to the spec's adherence threshold.

Deterministic checks can only fail a response, never rescue one. That asymmetry
is deliberate: the static rules are conservative, so when they fire they are
almost certainly right, but their silence proves nothing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence

from behavior.spec import BehaviorSpec, load_spec
from evaluation.behavioral_checks import CheckConfig, run_deterministic_checks
from evaluation.judge import Judge
from evaluation.metrics import aggregate
from evaluation.schemas import (
    CellMetrics,
    EvalRecord,
    Scenario,
    classify_error,
    write_jsonl,
)
from models.adapters import EVAL_PARAMS, GenerationParams, ModelAdapter
from prompting.strategies import PromptStrategy


class Evaluator:
    """Runs one (model x prompt strategy) cell against a scenario set."""

    def __init__(
        self,
        model: ModelAdapter,
        judge: Judge,
        strategy: PromptStrategy,
        *,
        spec: BehaviorSpec | None = None,
        params: GenerationParams | None = None,
        check_config: CheckConfig | None = None,
        max_workers: int = 4,
    ):
        self.model = model
        self.judge = judge
        self.strategy = strategy
        self.spec = spec or load_spec()
        self.params = params or EVAL_PARAMS
        self.check_config = check_config
        # A local GPU model is one set of weights on one device. Running two
        # threads through it does not halve wall-clock - it doubles peak
        # activation memory on a card that is already nearly full, and races the
        # per-call torch.manual_seed that is supposed to make runs deterministic.
        # Concurrency helps for API-backed models, where the wait is network.
        if getattr(model, "family", "") == "local-hf":
            max_workers = 1
        self.max_workers = max(1, max_workers)

    # ------------------------------------------------------------------ single

    def evaluate_scenario(self, scenario: Scenario) -> EvalRecord:
        prompt = self.strategy.render(scenario)
        response = self.model.generate(
            prompt.messages, system=prompt.system, params=self.params
        )

        deterministic = run_deterministic_checks(
            scenario, response.text, self.spec, self.check_config
        )

        judge_result = None
        if response.ok:
            judge_result = self.judge.judge(scenario, response.text, deterministic)

        passed, reasons = self._combine(scenario, deterministic, judge_result, response.error)

        return EvalRecord(
            scenario_id=scenario.id,
            scenario_split=scenario.split,
            pressure_type=scenario.pressure_type,
            language=scenario.language,
            bug_category=scenario.bug_category,
            difficulty=scenario.difficulty,
            student_has_solved=scenario.student_has_solved,
            model=self.model.name,
            model_family=self.model.family,
            model_revision=response.revision or self.model.revision,
            prompt_strategy=self.strategy.name,
            prompt_version=f"{self.strategy.version}+{prompt.sha256[:12]}",
            input_messages=prompt.messages,
            model_response=response.text,
            deterministic=deterministic,
            judge=judge_result,
            failure_reasons=reasons,
            passed=passed,
            behavior_spec_version=self.spec.version,
            behavior_spec_sha256=self.spec.spec_sha256,
            generation_params={
                **self.params.to_dict(),
                "latency_s": round(response.latency_s, 3),
                "usage": response.usage,
            },
            error=response.error,
            error_kind=classify_error(response.error),
        )

    def _combine(
        self,
        scenario: Scenario,
        deterministic,
        judge_result,
        error: str | None,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = list(deterministic.violations)

        if error is not None:
            # The call never produced a response, so the deterministic checks ran
            # on an empty string and "found" EMPTY_RESPONSE. Attributing that to
            # the model is how a CUDA OOM came to look like a tutor that answered
            # nothing 20 times. A failed call gets one code: MODEL_ERROR.
            return False, ("MODEL_ERROR",)

        blocking = deterministic.details.get("blocking_violations", [])
        if blocking and self.spec.scoring.pass_rule.deterministic_violations_are_authoritative:
            return False, tuple(dict.fromkeys(reasons))

        if judge_result is None:
            reasons.append("LOW_QUALITY")
            return False, tuple(dict.fromkeys(reasons))

        for code in judge_result.failure_reasons:
            if code not in reasons:
                reasons.append(code)

        judge_blocking = self.spec.blocking_failure_codes(scenario.student_has_solved)
        passed = (
            judge_result.passed
            and judge_result.spec_adherence >= self.spec.scoring.pass_rule.min_spec_adherence
            and not (set(judge_result.failure_reasons) & judge_blocking)
        )
        return passed, tuple(dict.fromkeys(reasons))

    # -------------------------------------------------------------------- many

    def evaluate(
        self,
        scenarios: Sequence[Scenario],
        *,
        on_progress: Callable[[int, int, EvalRecord], None] | None = None,
    ) -> list[EvalRecord]:
        """Evaluate every scenario, preserving input order."""
        if not scenarios:
            raise ValueError("evaluate() requires at least one scenario")

        records: list[EvalRecord | None] = [None] * len(scenarios)

        if self.max_workers == 1:
            for index, scenario in enumerate(scenarios):
                record = self.evaluate_scenario(scenario)
                records[index] = record
                if on_progress:
                    on_progress(index + 1, len(scenarios), record)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(self.evaluate_scenario, scenario): index
                    for index, scenario in enumerate(scenarios)
                }
                completed = 0
                for future in as_completed(futures):
                    index = futures[future]
                    record = future.result()
                    records[index] = record
                    completed += 1
                    if on_progress:
                        on_progress(completed, len(scenarios), record)

        return [r for r in records if r is not None]

    # ---------------------------------------------------------------- reporting

    def run(
        self,
        scenarios: Sequence[Scenario],
        *,
        transcript_path: str | None = None,
        label: str = "",
        on_progress: Callable[[int, int, EvalRecord], None] | None = None,
    ) -> tuple[CellMetrics, list[EvalRecord]]:
        """Evaluate, optionally persist transcripts, and aggregate."""
        records = self.evaluate(scenarios, on_progress=on_progress)
        if transcript_path:
            write_jsonl(transcript_path, records)
        metrics = aggregate(
            records,
            model=self.model.name,
            model_family=self.model.family,
            prompt_strategy=self.strategy.name,
            label=label,
        )
        return metrics, records

    def describe(self) -> dict[str, str]:
        return {
            **self.model.describe(),
            **self.strategy.describe(),
            **self.judge.describe(),
            "behavior_spec_version": self.spec.version,
            "behavior_spec_sha256": self.spec.spec_sha256[:16],
        }


def evaluate_many(
    evaluators: Iterable[Evaluator],
    scenarios: Sequence[Scenario],
    *,
    transcript_dir: str | None = None,
) -> tuple[list[CellMetrics], list[EvalRecord]]:
    """Run several cells over the same scenario set (the ablation workhorse)."""
    all_metrics: list[CellMetrics] = []
    all_records: list[EvalRecord] = []
    for evaluator in evaluators:
        path = None
        if transcript_dir:
            safe = f"{evaluator.model.name}__{evaluator.strategy.name}".replace(
                "/", "_"
            ).replace(":", "_")
            path = f"{transcript_dir.rstrip('/')}/{safe}.jsonl"
        metrics, records = evaluator.run(scenarios, transcript_path=path)
        all_metrics.append(metrics)
        all_records.extend(records)
    return all_metrics, all_records


__all__ = ["Evaluator", "evaluate_many"]
