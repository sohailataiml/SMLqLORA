"""Judging teacher candidates.

Reuses the evaluation judge unchanged. A training example is held to the *same*
rubric a model response is held to, plus the stricter numeric thresholds in
`behavior/spec.yaml -> gates.quality_gate`: it is not enough for a candidate to
scrape a pass, it has to be a good demonstration.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from behavior.spec import BehaviorSpec, load_spec
from evaluation.judge import Judge
from evaluation.schemas import DeterministicResult, JudgeResult, Scenario
from generation.schemas import GeneratedExample


#: Marker returned instead of rejection codes when the judge never ran. The
#: caller must hold the candidate aside rather than reject it — an outage
#: describes the billing account, not the candidate.
UNJUDGED = "__UNJUDGED__"


def judge_unavailable(result) -> bool:
    """True when the verdict is a fail-closed placeholder, not a judgment."""
    return "judge_unavailable" in (result.parse_warnings or ())


def judge_candidate(
    example: GeneratedExample,
    judge: Judge,
    spec: BehaviorSpec,
) -> tuple[GeneratedExample, tuple[str, ...]]:
    """Judge one candidate and return it with any rejection codes.

    A judge that could not be reached returns `(example, (UNJUDGED,))`. The
    judge itself fails *closed* — the right default when a verdict is required
    — but a quality gate must not turn "we could not ask" into "the candidate
    is bad". Tranche 1 of Dataset V1 made the cost of that conflation concrete:
    an exhausted API credit produced 410 candidates carrying LOW_QUALITY and
    IRRELEVANT_HINT, which would have been frozen into the dataset report as a
    35% quality failure rate that never happened.
    """
    thresholds = spec.gates.quality_gate
    result = judge.judge(example.scenario, example.tutor_response, example.deterministic)

    if judge_unavailable(result):
        updated = example.model_copy(
            update={
                "judge": result,
                "gate_notes": (
                    example.gate_notes + " | judge unavailable (infrastructure)"
                ).strip(" |"),
            }
        )
        return updated, (UNJUDGED,)

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


class CachedJudge(Judge):
    """Reuse verdicts already paid for; only call the real judge for the rest.

    Judging 1190 candidates costs real money, and an outage part-way through
    should not force re-purchasing the verdicts that succeeded. Cache keys are
    candidate content hashes, so an edited candidate correctly misses the cache
    instead of silently inheriting a stale verdict.

    Placeholder verdicts from an unreachable judge are **never** cached — those
    are exactly the calls a resumed run needs to retry.
    """

    def __init__(self, inner: Judge, verdicts: dict[str, JudgeResult],
                 journal: "VerdictJournal | None" = None):
        self.inner = inner
        self.verdicts = verdicts
        self.journal = journal
        self.model_name = inner.model_name
        self.model_family = inner.model_family
        self.prompt_version = inner.prompt_version
        self.hits = 0
        self.misses = 0

    def judge(
        self,
        scenario: Scenario,
        response: str,
        deterministic: DeterministicResult | None = None,
    ) -> JudgeResult:
        key = _verdict_key(scenario, response)
        cached = self.verdicts.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = self.inner.judge(scenario, response, deterministic)
        if self.journal is not None:
            self.journal.append(key, result)
        return result

    def describe(self) -> dict[str, str]:
        return self.inner.describe()


def _verdict_key(scenario: Scenario, response: str) -> str:
    import hashlib
    import re

    normalized = re.sub(r"\s+", " ", response.strip())
    payload = scenario.content_hash() + "\n" + normalized
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_verdict_cache(
    examples: Sequence[GeneratedExample],
) -> dict[str, JudgeResult]:
    """Index usable prior verdicts. Unreachable-judge placeholders are skipped."""
    cache: dict[str, JudgeResult] = {}
    for example in examples:
        result = example.judge
        if result is None or judge_unavailable(result):
            continue
        cache[_verdict_key(example.scenario, example.tutor_response)] = result
    return cache


class VerdictJournal:
    """Append each judge verdict to disk the moment it is paid for.

    The gate previously persisted only at the end of a run. That was survivable
    when the judge failed *softly* — placeholders still came back and the run
    finished — but a hard interruption part-way through would have discarded
    every verdict already bought. With 525 outstanding calls that is a real
    loss, so verdicts are journalled as they arrive and folded back into the
    cache on the next run.

    Thread-safe: the gate judges on a pool.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.written = 0

    def append(self, key: str, result: JudgeResult) -> None:
        # Never journal a placeholder: the next run must retry those, not
        # inherit them from cache.
        if judge_unavailable(result):
            return
        row = {"key": key, "verdict": result.model_dump(mode="json")}
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            self.written += 1


def load_journal(path) -> dict[str, JudgeResult]:
    """Read journalled verdicts back into a cache. Bad lines are skipped."""
    target = Path(path)
    if not target.exists():
        return {}
    out: dict[str, JudgeResult] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                result = JudgeResult.model_validate(row["verdict"])
            except Exception:  # noqa: BLE001 — a partial write is expected
                continue
            if not judge_unavailable(result):
                out[row["key"]] = result
    return out
