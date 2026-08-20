#!/usr/bin/env python
"""One-command evaluation. The entry point a grader runs.

    python eval.py --model <model-spec> --eval-set scenarios/heldout.jsonl

Model specs:
    anthropic:claude-opus-5                     frontier API
    openai:gpt-5                                frontier API
    hf:Qwen/Qwen3-1.7B                          local base model
    peft:Qwen/Qwen3-1.7B+outputs/socratic-v1    local tuned adapter
    <hf-repo-id>                                shorthand, resolved as hf:
    mock:demo                                   offline, for smoke tests

Every failure mode reports what is wrong and how to fix it: a missing model, an
unreadable eval set, absent credentials, an unknown provider, or a malformed
scenario file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablations.reporting import cells_to_rows, markdown_table, write_csv, write_json
from behavior.spec import load_spec
from evaluation.evaluator import Evaluator
from evaluation.judge import DeterministicJudge, LLMJudge
from evaluation.metrics import breakdown_by_pressure, failure_mode_counts
from evaluation.reproducibility import build_manifest
from evaluation.schemas import ScenarioLoadError, load_scenarios, write_jsonl
from models.adapters import (
    EVAL_PARAMS,
    MissingCredentialsError,
    MissingDependencyError,
    ModelError,
    UnsupportedProviderError,
    resolve_model,
)
from prompting.strategies import get_strategy

KNOWN_PREFIXES = ("anthropic:", "openai:", "hf:", "peft:", "mock:")


def normalize_model_spec(spec: str) -> str:
    """Accept a bare Hugging Face repo id as shorthand for `hf:<id>`."""
    if any(spec.startswith(prefix) for prefix in KNOWN_PREFIXES):
        return spec
    if "/" in spec and not spec.startswith(("/", ".")):
        return f"hf:{spec}"
    return spec


def _fail(message: str, code: int = 2) -> int:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    return code



def _describe_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    A finished evaluation must not exit non-zero because its last line could not
    phrase a path. Twenty judge calls have already been paid for by this point.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a model against the Socratic Debug Tutor behavior spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
                        help="model spec or Hugging Face repo id")
    parser.add_argument("--eval-set", default="scenarios/heldout.jsonl",
                        help="JSONL scenario file (default: scenarios/heldout.jsonl)")
    parser.add_argument("--strategy", default="zero_shot",
                        help="prompt strategy: zero_shot | few_shot | structured_system_prompt")
    parser.add_argument("--judge", default="anthropic:claude-opus-5",
                        help="judge model spec")
    parser.add_argument("--offline-judge", action="store_true",
                        help="use the deterministic judge (no API calls, weaker signal)")
    parser.add_argument("--output", default=None,
                        help="directory for results (default: results/eval/<model>)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = load_spec()

    # ---- eval set --------------------------------------------------------
    eval_path = Path(args.eval_set)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    try:
        scenarios = load_scenarios(eval_path)
    except ScenarioLoadError as exc:
        return _fail(f"could not load the evaluation set.\n{exc}")
    if args.limit:
        scenarios = scenarios[: args.limit]

    # ---- model -----------------------------------------------------------
    model_spec = normalize_model_spec(args.model)
    try:
        model = resolve_model(model_spec)
    except UnsupportedProviderError as exc:
        return _fail(str(exc))
    except ModelError as exc:
        return _fail(f"could not resolve model {args.model!r}.\n{exc}")

    # ---- judge -----------------------------------------------------------
    try:
        judge = (
            DeterministicJudge(spec)
            if args.offline_judge
            else LLMJudge(resolve_model(args.judge), spec)
        )
    except (UnsupportedProviderError, ModelError) as exc:
        return _fail(f"could not resolve judge {args.judge!r}.\n{exc}")

    strategy_name = args.strategy
    try:
        strategy = get_strategy(strategy_name, spec)
    except KeyError as exc:
        return _fail(str(exc).strip("'"))

    safe_name = model.name.replace("/", "_").replace(":", "_").replace("+", "__")
    # Resolve a relative --output against the repo, exactly as --eval-set is
    # resolved above. Without this the artifacts land wherever the caller happens
    # to be standing, and the summary line below cannot describe where they went.
    out = Path(args.output) if args.output else REPO_ROOT / "results" / "eval" / safe_name
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"model      : {model.name} (family={model.family})")
        print(f"eval set   : {eval_path.relative_to(REPO_ROOT)} ({len(scenarios)} scenarios)")
        print(f"strategy   : {strategy_name}")
        print(f"judge      : {judge.describe()['judge_model']}")
        print(f"spec       : v{spec.version} ({spec.spec_sha256[:12]})")
        print()

    evaluator = Evaluator(
        model, judge, strategy, spec=spec, params=EVAL_PARAMS,
        max_workers=args.max_workers,
    )

    def progress(done: int, total: int, record) -> None:
        if not args.quiet:
            print(f"  {done}/{total} scenarios", end="\r" if done != total else "\n",
                  flush=True)

    try:
        metrics, records = evaluator.run(
            scenarios,
            transcript_path=str(out / "judge_transcripts.jsonl"),
            label=model.name,
            on_progress=progress,
        )
    except MissingCredentialsError as exc:
        return _fail(str(exc))
    except MissingDependencyError as exc:
        return _fail(str(exc))
    except ValueError as exc:
        return _fail(
            f"nothing could be measured.\n{exc}\n"
            f"Every call failed for infrastructure reasons — check provider credit, "
            f"quota and network before treating this as a model result."
        )

    measured = [r for r in records if r.was_evaluated]
    rows = cells_to_rows([metrics])
    write_csv(out / "results.csv", rows)
    write_json(
        out / "results.json",
        {
            "result_status": "REAL_EXPERIMENT_RESULT",
            "model": model.describe(),
            "eval_set": str(eval_path.relative_to(REPO_ROOT)),
            "prompt_strategy": strategy_name,
            "metrics": rows[0],
            "failure_modes": failure_mode_counts(measured),
            "by_pressure_type": breakdown_by_pressure(measured),
        },
    )
    build_manifest(
        "eval",
        spec=spec,
        scenarios=scenarios,
        scenario_paths=[str(eval_path.relative_to(REPO_ROOT))],
        models=[model],
        strategies=[strategy],
        judge=judge,
        generation_params=EVAL_PARAMS.to_dict(),
        result_status="REAL_EXPERIMENT_RESULT",
    ).write(out / "manifest.json")

    print()
    print(markdown_table(
        [
            {"metric": "scenarios measured", "value": metrics.scenario_count},
            {"metric": "spec adherence", "value": metrics.spec_adherence_mean},
            {"metric": "robustness", "value": metrics.robustness_mean},
            {"metric": "hint relevance", "value": metrics.hint_relevance_mean},
            {"metric": "pass rate", "value": metrics.pass_rate},
            {"metric": "solution leak rate", "value": metrics.solution_leak_rate},
            {"metric": "premature confirmation rate",
             "value": metrics.premature_confirmation_rate},
            {"metric": "clean pass rate", "value": metrics.clean_pass_rate},
            {"metric": "adversarial pass rate", "value": metrics.adversarial_pass_rate},
        ],
        ["metric", "value"],
    ))
    if metrics.partial:
        print(
            f"\nWARNING: PARTIAL — {metrics.infrastructure_error_count} of "
            f"{metrics.attempted_count} calls failed for infrastructure reasons and "
            f"were excluded from every rate above."
        )
    print(f"\nfailure modes: {failure_mode_counts(measured) or 'none'}")
    print(f"artifacts    : {_describe_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
