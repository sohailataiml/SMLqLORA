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
from generation.prompts import (  # noqa: E402
    GENERATION_PROMPT_VERSION,
    generation_prompt_hash,
    plan_summary,
    sample_plan,
)
from generation.resume import (  # noqa: E402
    CandidateWriter,
    check_run_config,
    completed_indices,
    pending_indices,
    read_candidates,
    write_run_config,
)
from generation.schemas import GeneratedExample, GenerationBatchStats  # noqa: E402
from generation.teacher import Teacher, TeacherError  # noqa: E402
from models.adapters import ModelAdapter, ScriptedAdapter, resolve_model  # noqa: E402
from models.usage import MeteredAdapter, UsageMeter  # noqa: E402

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


def build_teacher(spec_string: str, *, mock: bool, dataset_version: str,
                  meter: UsageMeter | None = None) -> Teacher:
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
    if meter is not None:
        model = MeteredAdapter(model, meter)
    return Teacher(model, spec=spec, dataset_version=dataset_version)


def generate(
    *,
    count: int,
    teacher: Teacher,
    base_seed: int = 20240101,
    max_workers: int = 4,
    verbose: bool = True,
    candidates_path: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[GeneratedExample], GenerationBatchStats, list[str]]:
    """Run the plan. Returns (candidates, stats, failure messages).

    When `candidates_path` is given the run is **resumable**: completed
    candidates are appended to that file as they arrive, and a restart skips
    plan indices already present. Because a candidate's plan index fixes its
    dimension draw, resuming cannot shift the distribution, and a retried
    infrastructure failure cannot create a second record for one plan point.
    """
    plan = sample_plan(count, base_seed=base_seed)
    stats = GenerationBatchStats(
        requested=count,
        dataset_version=teacher.dataset_version,
        teacher_model=teacher.model.name,
    )
    failures: list[str] = []
    started = time.perf_counter()

    prior: list[GeneratedExample] = []
    if candidates_path is not None and resume:
        prior, malformed = read_candidates(candidates_path)
        if malformed and verbose:
            print(f"  skipped {malformed} unreadable line(s) from a prior run")
    done_indices = completed_indices(prior)
    # Never reuse a row outside the current plan; a shorter --count must not
    # inherit candidates from a longer earlier run.
    prior = [e for e in prior if (idx := _index_of(e)) is not None and idx < count]
    done_indices = {i for i in done_indices if i < count}
    todo = pending_indices(count, done_indices)

    if verbose and done_indices:
        print(f"  resuming: {len(done_indices)} prior candidate(s) reused, "
              f"{len(todo)} to generate")

    stats.reused = len(done_indices)

    def work(index: int) -> tuple[int, GeneratedExample | None, str | None]:
        seed, dimensions = plan[index]
        try:
            return index, teacher.generate_one(seed, dimensions, index=index), None
        except TeacherError as exc:
            return index, None, f"[{exc.code}] {exc}"

    fresh: list[GeneratedExample] = []
    writer_cm = (
        CandidateWriter(candidates_path) if candidates_path is not None
        else _NullWriter()
    )
    with writer_cm as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(work, i) for i in todo]
            done = 0
            for future in as_completed(futures):
                index, example, error = future.result()
                done += 1
                if example is not None:
                    fresh.append(example)
                    writer.write(example)
                else:
                    failures.append(error or "unknown failure")
                    # `work()` prefixes the propagated cause as "[CODE] ...", so
                    # a malformed teacher payload is never filed as an outage.
                    if error and error.startswith("[INFRASTRUCTURE]"):
                        stats.infrastructure_errors += 1
                    elif error and error.startswith("[UNPARSEABLE]"):
                        stats.parse_failures += 1
                    elif error and error.startswith("[INVALID_SCHEMA]"):
                        stats.schema_failures += 1
                    else:
                        stats.provider_errors += 1
                if verbose and (done % 10 == 0 or done == len(todo)):
                    print(
                        f"  {done}/{len(todo)} attempted, {len(fresh)} valid, "
                        f"{len(failures)} failed",
                        end="\r" if done != len(todo) else "\n",
                    )

    produced = sorted(
        prior + fresh, key=lambda e: (_index_of(e) if _index_of(e) is not None else 0)
    )
    stats.returned = len(produced)
    stats.generated_this_run = len(fresh)
    stats.elapsed_s = round(time.perf_counter() - started, 2)
    stats.failure_examples = failures[:20]
    return produced, stats, failures


def _index_of(example: GeneratedExample) -> int | None:
    from generation.resume import index_of

    return index_of(example.id)


class _NullWriter:
    """Stand-in when no candidates path is given (tests, --plan-only)."""

    def __enter__(self) -> "_NullWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def write(self, example: GeneratedExample) -> None:
        return None


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
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore prior candidates and re-purchase every call")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the dimension distribution and exit without calling a model")
    args = parser.parse_args(argv)

    if args.plan_only:
        plan = sample_plan(args.count, base_seed=args.base_seed)
        print(json.dumps(plan_summary(plan), indent=2))
        return 0

    meter = UsageMeter()
    teacher = build_teacher(args.teacher, mock=args.mock,
                            dataset_version=args.dataset_version, meter=meter)

    out_dir = REPO_ROOT / args.output_dir
    candidates_path = out_dir / f"{args.dataset_version}.jsonl"

    # Reusing rows generated under a different seed or prompt would silently mix
    # two dimension spaces, so a mismatch stops the run instead.
    run_config = {
        "base_seed": args.base_seed,
        "dataset_version": args.dataset_version,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "generation_prompt_sha256": generation_prompt_hash(load_spec()),
        "teacher_model": teacher.model.name,
    }
    if not args.no_resume:
        problems = check_run_config(candidates_path, run_config)
        if problems:
            print("Cannot resume — the cached candidates came from a different run:")
            for problem in problems:
                print(f"  - {problem}")
            print("Re-run with --no-resume to start fresh (this re-purchases "
                  "every candidate), or restore the original settings.")
            return 1

    print(f"Teacher: {teacher.model.name} (dataset {args.dataset_version})")
    print(f"Attempting {args.count} candidates with {args.max_workers} workers...")

    candidates, stats, failures = generate(
        count=args.count,
        teacher=teacher,
        base_seed=args.base_seed,
        max_workers=args.max_workers,
        candidates_path=candidates_path,
        resume=not args.no_resume,
    )

    stats.usage = meter.totals()
    write_run_config(candidates_path, run_config)

    stats_path = out_dir / f"{args.dataset_version}.stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats.model_dump(), indent=2) + "\n", encoding="utf-8")

    print()
    print(f"valid candidates : {stats.returned}/{stats.requested}")
    print(f"reused (resume)  : {stats.reused}")
    print(f"bought this run  : {stats.generated_this_run}")
    print(f"parse failures   : {stats.parse_failures}")
    print(f"schema failures  : {stats.schema_failures}")
    print(f"infra errors     : {stats.infrastructure_errors}")
    print(f"provider errors  : {stats.provider_errors}")
    tokens = stats.usage.get("totals", {})
    if tokens.get("requests"):
        print(f"tokens           : {tokens.get('input_tokens', 0):,} in / "
              f"{tokens.get('output_tokens', 0):,} out "
              f"over {tokens.get('requests', 0)} request(s)")
    print(f"elapsed          : {stats.elapsed_s}s")
    print(f"written          : {candidates_path.relative_to(REPO_ROOT)}")
    if failures:
        print(f"\nfirst failures:\n  " + "\n  ".join(failures[:5]))

    print("\nNext: python scripts/filter_data.py "
          f"--dataset-version {args.dataset_version}")
    return 0 if stats.returned else 1


if __name__ == "__main__":
    raise SystemExit(main())
