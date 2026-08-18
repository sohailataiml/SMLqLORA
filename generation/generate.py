"""Teacher data generation driver.

Walks a deterministic plan over the controlled dimension space, calls the teacher
once per point, and writes every candidate — including failures — to
`data/candidates/`. Filtering is a separate step so a generation run is never
lost to a filtering bug, and so the gate can be re-run with different thresholds
without paying for generation again.

Usage:
    python -m generation.generate --count 1400 --dataset-version v1
    python -m generation.generate --count 12 --mock --dataset-version vdev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior.spec import load_spec  # noqa: E402
from evaluation.schemas import write_jsonl  # noqa: E402
from generation.prompts import plan_summary, sample_plan  # noqa: E402
from generation.schemas import GeneratedExample, GenerationBatchStats  # noqa: E402
from generation.teacher import Teacher, TeacherError  # noqa: E402
from models.adapters import ModelAdapter, ScriptedAdapter, resolve_model  # noqa: E402

DEFAULT_TEACHER = "anthropic:claude-opus-5"

_MOCK_PYTHON = (
    "def total(nums, factor):\n"
    "    s = 0\n"
    "    for i in range(len(nums) - 1):\n"
    "        s += nums[i] * factor\n"
    "    return s"
)
_MOCK_JAVASCRIPT = (
    "function total(nums, factor) {\n"
    "  let s = 0;\n"
    "  for (let i = 0; i < nums.length - 1; i++) {\n"
    "    s += nums[i] * factor;\n"
    "  }\n"
    "  return s;\n"
    "}"
)


def mock_payload_for(prompt: str) -> dict:
    """Build a mock candidate that actually satisfies the requested dimensions.

    The mock teacher reads the language, turn count and solved-state back out of
    its own prompt. Without that the mock fails validation for reasons that have
    nothing to do with the pipeline, and `--mock` stops being a useful test.
    """
    import re as _re

    language = "javascript" if "language: javascript" in prompt else "python"
    turns_match = _re.search(r"prior exchanges: (\d+)", prompt)
    turns = int(turns_match.group(1)) if turns_match else 0
    solved = '"student_has_solved": true' in prompt
    seed_match = _re.search(r"generation seed (\d+)", prompt)
    seed = int(seed_match.group(1)) if seed_match else 0

    # Vary the surface so the mock exercises dedupe and balancing realistically
    # rather than collapsing into one repeated example.
    subject = ["nums", "scores", "rows", "values", "items", "prices", "counts"][seed % 7]
    helper = ["total", "tally", "summarize", "accumulate", "aggregate"][seed % 5]
    symptom = [
        "the result is always a little too small",
        "the final entry never shows up",
        "I lose the last record every time",
        "the totals are consistently short by one item",
        "the output misses whatever is at the end",
    ][seed % 5]

    code = (
        _MOCK_PYTHON if language == "python" else _MOCK_JAVASCRIPT
    ).replace("nums", subject).replace("total", helper)

    history = []
    probes = [
        "How many times does the body actually run?",
        "What is the last position the loop reaches?",
        "Which line decides where the loop stops?",
        f"What does the length of {subject} tell you about the bound?",
    ]
    for turn in range(turns):
        history.append(
            {"role": "user",
             "content": f"I checked the accumulator on attempt {turn + 1} and it looked fine."}
        )
        history.append({"role": "assistant", "content": probes[(seed + turn) % len(probes)]})

    if solved:
        student = (
            f"I widened the bound to cover every index of {subject}, and {helper} is "
            f"correct now."
        )
        tutor = (
            f"That's exactly it. The original bound stopped one position short, so the "
            f"final element of {subject} never reached the accumulator. Nice work "
            f"tracing it from the counts."
        )
    else:
        student = f"When I call {helper}, {symptom}. What am I missing?"
        tutor = (
            f"If {subject} has four entries, what is the last index your loop "
            f"actually visits?"
        )

    return {
        "code": code,
        "conversation_history": history,
        "student_message": student,
        "expected_bug": (
            f"The loop bound stops one index early, so the final element of "
            f"{subject} is never added."
        ),
        "expected_fix": (
            f"for i in range(len({subject})):" if language == "python"
            else f"for (let i = 0; i < {subject}.length; i++) {{"
        ),
        "tutor_response": tutor,
        "student_has_solved": solved,
    }


def build_teacher(spec_string: str, *, mock: bool, dataset_version: str) -> Teacher:
    spec = load_spec()
    model: ModelAdapter
    if mock:
        model = ScriptedAdapter(
            lambda messages: json.dumps(mock_payload_for(messages[-1].content)),
            name="mock:teacher",
            family="mock",
            revision="mock-1",
        )
    else:
        model = resolve_model(spec_string)
    return Teacher(model, spec=spec, dataset_version=dataset_version)


def generate(
    *,
    count: int,
    teacher: Teacher,
    base_seed: int = 20240101,
    max_workers: int = 4,
    verbose: bool = True,
) -> tuple[list[GeneratedExample], GenerationBatchStats, list[str]]:
    """Run the plan. Returns (candidates, stats, failure messages)."""
    plan = sample_plan(count, base_seed=base_seed)
    stats = GenerationBatchStats(
        requested=count,
        dataset_version=teacher.dataset_version,
        teacher_model=teacher.model.name,
    )
    candidates: list[GeneratedExample | None] = [None] * count
    failures: list[str] = []
    started = time.perf_counter()

    def work(index: int) -> tuple[int, GeneratedExample | None, str | None]:
        seed, dimensions = plan[index]
        try:
            return index, teacher.generate_one(seed, dimensions, index=index), None
        except TeacherError as exc:
            return index, None, f"[{exc.code}] {exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, i) for i in range(count)]
        done = 0
        for future in as_completed(futures):
            index, example, error = future.result()
            done += 1
            if example is not None:
                candidates[index] = example
            else:
                failures.append(error or "unknown failure")
                if error and "INVALID_SCHEMA" in error:
                    stats.schema_failures += 1
                elif error and "unparseable JSON" in error:
                    stats.parse_failures += 1
                else:
                    stats.provider_errors += 1
            if verbose and (done % 10 == 0 or done == count):
                ok = sum(1 for c in candidates if c is not None)
                print(
                    f"  {done}/{count} attempted, {ok} valid, {len(failures)} failed",
                    end="\r" if done != count else "\n",
                    flush=True,
                )

    produced = [c for c in candidates if c is not None]
    stats.returned = len(produced)
    stats.elapsed_s = round(time.perf_counter() - started, 2)
    stats.failure_examples = failures[:20]
    return produced, stats, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate teacher training candidates.")
    parser.add_argument("--count", type=int, default=1400,
                        help="candidates to attempt (the gate rejects aggressively)")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--base-seed", type=int, default=20240101)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-dir", default="data/candidates")
    parser.add_argument("--mock", action="store_true",
                        help="scripted teacher; produces clearly-labelled mock candidates")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the dimension distribution and exit without calling a model")
    args = parser.parse_args(argv)

    if args.plan_only:
        plan = sample_plan(args.count, base_seed=args.base_seed)
        print(json.dumps(plan_summary(plan), indent=2))
        return 0

    teacher = build_teacher(args.teacher, mock=args.mock,
                            dataset_version=args.dataset_version)
    print(f"Teacher: {teacher.model.name} (dataset {args.dataset_version})")
    print(f"Attempting {args.count} candidates with {args.max_workers} workers...")

    candidates, stats, failures = generate(
        count=args.count,
        teacher=teacher,
        base_seed=args.base_seed,
        max_workers=args.max_workers,
    )

    out_dir = REPO_ROOT / args.output_dir
    candidates_path = out_dir / f"{args.dataset_version}.jsonl"
    write_jsonl(candidates_path, candidates)

    stats_path = out_dir / f"{args.dataset_version}.stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats.model_dump(), indent=2) + "\n", encoding="utf-8")

    print()
    print(f"valid candidates : {stats.returned}/{stats.requested}")
    print(f"parse failures   : {stats.parse_failures}")
    print(f"schema failures  : {stats.schema_failures}")
    print(f"provider errors  : {stats.provider_errors}")
    print(f"elapsed          : {stats.elapsed_s}s")
    print(f"written          : {candidates_path.relative_to(REPO_ROOT)}")
    if failures:
        print(f"\nfirst failures:\n  " + "\n  ".join(failures[:5]))

    print("\nNext: python scripts/filter_data.py "
          f"--dataset-version {args.dataset_version}")
    return 0 if stats.returned else 1


if __name__ == "__main__":
    raise SystemExit(main())
