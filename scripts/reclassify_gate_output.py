"""Re-derive a dataset version's verdicts from stored judge results.

Tranche 1 of Dataset V1 was filtered while the Anthropic account ran out of
credit. The judge fails closed by design — the right default when a verdict is
required — but the gate then recorded 410 unreachable-judge placeholders as
`LOW_QUALITY` / `IRRELEVANT_HINT` rejections. Frozen, that would have published
a 35% quality-failure rate that never happened.

This script re-applies the *corrected* classification to the records already on
disk. It makes **zero API calls**: every judge verdict it reads was already paid
for, and candidates whose judge never ran are moved out of `rejected` into
`unjudged` so a resumed run retries exactly them.

    python scripts/reclassify_gate_output.py --dataset-version v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.schemas import write_jsonl  # noqa: E402
from generation.schemas import GeneratedExample  # noqa: E402


def load(path: Path) -> list[GeneratedExample]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(GeneratedExample.model_validate(json.loads(line)))
    return out


def was_unjudged(example: GeneratedExample) -> bool:
    judge = example.judge
    if judge is None:
        return False
    return "judge_unavailable" in (judge.parse_warnings or ())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args(argv)

    version_dir = REPO_ROOT / "data" / "versions" / args.dataset_version
    accepted = load(version_dir / "accepted.jsonl")
    rejected = load(version_dir / "rejected.jsonl")

    if not accepted and not rejected:
        raise SystemExit(f"No gate output in {version_dir}")

    still_rejected = [e for e in rejected if not was_unjudged(e)]
    unjudged = [
        e.model_copy(update={
            "accepted": None,
            "rejection_codes": (),
            "gate_notes": "judge unavailable (infrastructure) — retry, not a rejection",
        })
        for e in rejected if was_unjudged(e)
    ]

    candidates_path = (REPO_ROOT / "data" / "candidates"
                       / f"{args.dataset_version}.jsonl")
    total = sum(1 for line in candidates_path.read_text(encoding="utf-8").splitlines()
                if line.strip()) if candidates_path.exists() else (
        len(accepted) + len(rejected))

    judged = total - len(unjudged)
    reasons: Counter[str] = Counter()
    for example in still_rejected:
        for code in example.rejection_codes:
            reasons[code] += 1

    write_jsonl(version_dir / "rejected.jsonl", still_rejected)
    write_jsonl(version_dir / "unjudged.jsonl", unjudged)
    write_jsonl(REPO_ROOT / "data" / "rejected" / f"{args.dataset_version}.jsonl",
                still_rejected)

    summary = {
        "dataset_version": args.dataset_version,
        "complete": not unjudged,
        "candidates": total,
        "judged": judged,
        "accepted": len(accepted),
        "rejected": len(still_rejected),
        "unjudged_infrastructure": len(unjudged),
        "acceptance_rate_of_judged": round(len(accepted) / judged, 4) if judged else 0.0,
        "acceptance_rate_of_all_candidates_MISLEADING": round(
            len(accepted) / total, 4) if total else 0.0,
        "rejections_by_reason": dict(reasons.most_common()),
        "note": (
            "unjudged candidates are an infrastructure outcome (the judge could "
            "not be reached) and are excluded from the acceptance-rate "
            "denominator. They must be re-judged before this version is frozen."
        ),
    }
    (version_dir / "gate_status.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Dataset {args.dataset_version} — reclassified from stored verdicts")
    print(f"  candidates            : {total}")
    print(f"  judged                : {judged}")
    print(f"  accepted              : {len(accepted)}")
    print(f"  rejected (real)       : {len(still_rejected)}")
    print(f"  unjudged (infra)      : {len(unjudged)}")
    print(f"  acceptance rate       : {summary['acceptance_rate_of_judged']:.1%} "
          f"of judged")
    print(f"  (as previously shown  : "
          f"{summary['acceptance_rate_of_all_candidates_MISLEADING']:.1%} "
          f"— depressed by the outage)")
    print()
    print("  rejections by reason:")
    for reason, count in reasons.most_common():
        print(f"    {reason:<28} {count}")
    print()
    print(f"  COMPLETE: {summary['complete']}")
    if unjudged:
        print(f"  {len(unjudged)} candidate(s) still need a judge verdict. "
              f"Do not freeze this version yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
