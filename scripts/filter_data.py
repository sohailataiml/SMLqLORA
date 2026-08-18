"""Run the quality gate over generated candidates and cut a dataset version.

    python scripts/filter_data.py --dataset-version v1
    python scripts/filter_data.py --dataset-version vdev --mock

Reads `data/candidates/<version>.jsonl`, writes `data/accepted/`,
`data/rejected/` and `data/versions/<version>/` with a full dataset report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from behavior.spec import load_spec  # noqa: E402
from evaluation.judge import DeterministicJudge, LLMJudge  # noqa: E402
from evaluation.schemas import iter_jsonl, load_scenario_files  # noqa: E402
from filtering.judge import CachedJudge, build_verdict_cache  # noqa: E402
from filtering.quality_gate import run_quality_gate, write_dataset_version  # noqa: E402
from generation.schemas import GeneratedExample  # noqa: E402
from generation.topup import plan_topup  # noqa: E402
from models.adapters import resolve_model  # noqa: E402
from models.usage import MeteredAdapter, UsageMeter  # noqa: E402

EVAL_SETS = ("scenarios/clean.jsonl", "scenarios/adversarial.jsonl",
             "scenarios/heldout.jsonl")


def load_candidates(path: Path) -> list[GeneratedExample]:
    if not path.exists():
        raise SystemExit(
            f"No candidates at {path}.\n"
            f"Generate them first:  python -m generation.generate "
            f"--dataset-version {path.stem}"
        )
    examples: list[GeneratedExample] = []
    problems = 0
    for lineno, row in enumerate(iter_jsonl(path), start=1):
        try:
            examples.append(GeneratedExample.model_validate(row))
        except Exception as exc:  # noqa: BLE001
            problems += 1
            if problems <= 3:
                print(f"  warning: candidate on line {lineno} failed to load: {exc}")
    if problems:
        print(f"  {problems} candidate(s) could not be loaded and were skipped")
    if not examples:
        raise SystemExit(f"No loadable candidates in {path}")
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the data quality gate.")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--candidates", default=None,
                        help="override the candidates path")
    parser.add_argument("--judge", default="anthropic:claude-opus-5")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--mock", action="store_true",
                        help="use the offline deterministic judge (no API calls)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--target-accepted", type=int, default=600,
                        help="target size; used only to size a top-up tranche")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore prior judge verdicts and re-purchase all of them")
    args = parser.parse_args(argv)

    spec = load_spec()
    candidates_path = Path(args.candidates) if args.candidates else (
        REPO_ROOT / "data" / "candidates" / f"{args.dataset_version}.jsonl"
    )
    candidates = load_candidates(candidates_path)
    eval_scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])

    meter = UsageMeter()
    judge = (
        DeterministicJudge(spec)
        if args.mock
        else LLMJudge(MeteredAdapter(resolve_model(args.judge), meter), spec)
    )

    # Reuse verdicts already paid for. An outage part-way through judging must
    # not force re-purchasing the calls that succeeded.
    cache_hits = 0
    if not args.no_resume:
        version_dir = REPO_ROOT / "data" / "versions" / args.dataset_version
        prior: list = []
        for name in ("accepted.jsonl", "rejected.jsonl", "unjudged.jsonl"):
            path = version_dir / name
            if path.exists():
                prior.extend(load_candidates(path))
        if prior:
            verdicts = build_verdict_cache(prior)
            if verdicts:
                judge = CachedJudge(judge, verdicts)
                cache_hits = len(verdicts)
                print(f"Resuming        : {cache_hits} prior judge verdict(s) reusable")

    print(f"Candidates      : {len(candidates)}")
    print(f"Judge           : {judge.describe()['judge_model']}")
    print(f"Eval scenarios  : {len(eval_scenarios)} (contamination reference)")
    print()

    def progress(stage: str, survived: int, entering: int) -> None:
        dropped = entering - survived
        print(f"  {stage:<15} {entering:>5} -> {survived:>5}  (-{dropped})")

    outcome = run_quality_gate(
        candidates,
        judge,
        spec=spec,
        eval_scenarios=eval_scenarios,
        dataset_version=args.dataset_version,
        max_workers=args.max_workers,
        teacher_description=(
            candidates[0].provenance.model_dump() if candidates else {}
        ),
        on_progress=progress,
        notes=args.notes,
    )

    paths = write_dataset_version(
        outcome, repo_root=REPO_ROOT, dataset_version=args.dataset_version
    )

    report = outcome.report
    assert report is not None
    print()
    print(f"accepted        : {report.accepted_count}")
    print(f"rejected        : {report.rejected_count}")
    print(f"unjudged (infra): {report.unjudged_count}")
    print(f"acceptance rate : {report.acceptance_rate:.1%} of {report.judged_count} judged")
    if not report.complete:
        print(f"STATUS          : INCOMPLETE — {report.unjudged_count} candidate(s) "
              f"never reached the judge. Do not freeze this version.")
    print(f"dataset hash    : {report.dataset_hash[:16]}")
    print(f"contamination   : {report.contamination_summary}")
    print()
    print("top rejection reasons:")
    for reason, count in list(report.rejections_by_reason.items())[:8]:
        print(f"  {reason:<28} {count}")
    tokens = meter.totals()
    if tokens["totals"]["requests"]:
        t = tokens["totals"]
        print()
        print(f"judge tokens    : {t['input_tokens']:,} in / "
              f"{t['output_tokens']:,} out over {t['requests']} request(s)")
        usage_path = (REPO_ROOT / "data" / "versions" / args.dataset_version
                      / "judge_usage.json")
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")

    # Size the next tranche from what the gate actually accepted, rather than
    # from an assumed rate. Reported only; generation stays a separate command.
    print()
    if report.accepted_count >= args.target_accepted:
        print(f"top-up          : not needed "
              f"({report.accepted_count} >= {args.target_accepted})")
    elif report.acceptance_rate > 0:
        topup = plan_topup(target=args.target_accepted,
                           accepted=report.accepted_count,
                           observed_rate=report.acceptance_rate)
        print(f"top-up          : {topup.additional_candidates} more candidate(s)")
        print(f"                  {topup.reason}")
        topup_path = (REPO_ROOT / "data" / "versions" / args.dataset_version
                      / "topup.json")
        topup_path.parent.mkdir(parents=True, exist_ok=True)
        topup_path.write_text(json.dumps(topup.to_dict(), indent=2) + "\n",
                              encoding="utf-8")
    else:
        print("top-up          : NOT SIZED — acceptance rate is 0; "
              "investigate the gate before spending again")

    print()
    for name, path in paths.items():
        print(f"  {name:<18} {Path(path).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
