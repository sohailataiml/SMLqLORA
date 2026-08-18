"""Judging teacher candidates.

Reuses the evaluation judge unchanged. A training example is held to the *same*
rubric a model response is held to, plus the stricter numeric thresholds in
`behavior/spec.yaml -> gates.quality_gate`: it is not enough for a candidate to
scrape a pass, it has to be a good demonstration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from behavior.spec import BehaviorSpec, load_spec
from evaluation.judge import Judge
from generation.schemas import GeneratedExample


def judge_candidate(
    example: GeneratedExample,
    judge: Judge,
    spec: BehaviorSpec,
) -> tuple[GeneratedExample, tuple[str, ...]]:
    """Judge one candidate and return it with any rejection codes."""
    thresholds = spec.gates.quality_gate
    result = judge.judge(example.scenario, example.tutor_response, example.deterministic)

    codes: list[str] = list(result.failure_reasons)

    if result.spec_adherence < thresholds.min_judge_spec_adherence:
        if "LOW_QUALITY" not in codes:
            codes.append("LOW_QUALITY")
    if result.hint_relevance < thresholds.min_judge_hint_relevance:
        if "IRRELEVANT_HINT" not in codes:
            codes.append("IRRELEVANT_HINT")
    if spec.is_adversarial_pressure(example.scenario.pressure_type.value):
        if result.robustness < thresholds.min_judge_robustness:
            if "LOW_QUALITY" not in codes:
                codes.append("LOW_QUALITY")
    if not result.passed and not codes:
        codes.append("LOW_QUALITY")

    notes = (
        f"judge adherence={result.spec_adherence:.2f} "
        f"robustness={result.robustness:.2f} relevance={result.hint_relevance:.2f}"
    )
    updated = example.model_copy(
        update={
            "judge": result,
            "gate_notes": (example.gate_notes + " | " + notes).strip(" |"),
        }
    )
    return updated, tuple(dict.fromkeys(codes))


def judge_candidates(
    examples: Sequence[GeneratedExample],
    judge: Judge,
    spec: BehaviorSpec | None = None,
    *,
    max_workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[GeneratedExample, tuple[str, ...]]]:
    """Judge many candidates, preserving input order."""
    spec = spec or load_spec()
    if not examples:
        return []

    results: list[tuple[GeneratedExample, tuple[str, ...]] | None] = [None] * len(examples)

    if max_workers == 1:
        for index, example in enumerate(examples):
            results[index] = judge_candidate(example, judge, spec)
            if on_progress:
                on_progress(index + 1, len(examples))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(judge_candidate, example, judge, spec): index
                for index, example in enumerate(examples)
            }
            done = 0
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                done += 1
                if on_progress:
                    on_progress(done, len(examples))

    return [r for r in results if r is not None]


__all__ = ["judge_candidate", "judge_candidates"]
