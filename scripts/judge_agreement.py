"""Score human-vs-LLM-judge agreement once the validation sheet is graded.

Reads `human_validation.csv`, uses only the rows whose `human_pass` column is
filled in, and reports percent agreement plus Cohen's kappa.

Ungraded rows are skipped, never guessed. If nobody has graded anything yet the
script says so and exits without producing a number, because an agreement
statistic over zero labels is not a small result — it is no result.

    python scripts/judge_agreement.py
    python scripts/judge_agreement.py --csv results/prompt_ceiling/human_validation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.human_validation import score_agreement_csv  # noqa: E402

DEFAULT_CSV = "results/prompt_ceiling/human_validation.csv"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output", default=None,
                        help="write the report as JSON next to the CSV")
    args = parser.parse_args(argv)

    path = REPO_ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    if not path.exists():
        print(f"No validation sheet at {path}.")
        print("Generate one with: python scripts/analyze_prompt_ceiling.py")
        return 1

    report = score_agreement_csv(path)

    print(f"Judge-vs-human agreement — {path.name}")
    print(f"  rows in sheet    : {report.n_rows}")
    print(f"  rows graded      : {report.n_graded}")

    if report.n_graded == 0:
        print()
        print("NOT YET GRADED — no `human_pass` cell is filled in.")
        print("Agreement is undefined until a human grades the sheet. Nothing")
        print("was estimated or inferred.")
        return 2

    print(f"  agreements       : {report.n_agree}")
    print(f"  percent agreement: {report.percent_agreement:.1%}")
    print(f"  Cohen's kappa    : "
          f"{'undefined' if report.cohens_kappa is None else report.cohens_kappa}")
    print()
    print("  confusion (judge x human)")
    print(f"    both pass            : {report.both_pass}")
    print(f"    both fail            : {report.both_fail}")
    print(f"    judge pass/human fail: {report.judge_pass_human_fail}  "
          f"(judge too lenient)")
    print(f"    judge fail/human pass: {report.judge_fail_human_pass}  "
          f"(judge too strict)")
    print()
    print(f"  {report.interpretation}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
