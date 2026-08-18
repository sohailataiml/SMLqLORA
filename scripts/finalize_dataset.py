"""Audit, freeze and package a dataset version.

The quality gate decides what is *in* the dataset. This script answers the
separate question a reviewer actually asks: **is the resulting dataset good
enough to spend GPU time on?** It computes the audit, exports a blind
human-review sheet, writes the dataset card, freezes the version behind a hash
manifest, and prepares the nested data-efficiency subsets.

It makes **zero API calls** and never mutates accepted/rejected data, so it can
be re-run freely after any change to the reporting code.

    python scripts/finalize_dataset.py --dataset-version v1

Freezing is the one irreversible-feeling step, so it is explicit: once
`freeze.json` exists, re-running reports a mismatch instead of silently
restamping a dataset that has changed underneath it. A corrected dataset is v2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior.spec import load_spec  # noqa: E402
from evaluation.schemas import load_scenario_files, write_jsonl  # noqa: E402
from filtering.dataset_card import render_dataset_card  # noqa: E402
from filtering.dedupe import check_contamination, deduplicate  # noqa: E402
from filtering.selection import select_balanced  # noqa: E402
from generation.schemas import GeneratedExample  # noqa: E402
from prompting.strategies import render_conversation  # noqa: E402
from training.dataset import (  # noqa: E402
    dataset_fingerprint,
    nested_subsets,
    stable_order,
    to_chat_record,
)

EVAL_SETS = ("scenarios/clean.jsonl", "scenarios/adversarial.jsonl",
             "scenarios/heldout.jsonl")

#: Pressure types where the learner is actively trying to extract the answer.
#: Coverage here is the whole point of the dataset, so it is reported by name.
ANSWER_SEEKING = ("repeated_answer_request", "time_pressure", "frustrated")
INJECTION_AUTHORITY = ("prompt_injection", "authority_override")

SWEEP_SIZES = (125, 250, 500, 600)
HUMAN_REVIEW_TARGET = 40


# =============================================================================
# Loading
# =============================================================================


def load_examples(path: Path) -> list[GeneratedExample]:
    if not path.exists():
        raise SystemExit(f"No dataset at {path}. Run the quality gate first.")
    out: list[GeneratedExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(GeneratedExample.model_validate(json.loads(line)))
    if not out:
        raise SystemExit(f"{path} contains no examples.")
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (tests use tmp dirs)."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace(chr(92), "/")
    except ValueError:
        return str(path)


def _counts(values: Sequence[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def _shares(counter: dict[str, int]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 4) for k, v in counter.items()}


# =============================================================================
# Audit
# =============================================================================


def learner_state(example: GeneratedExample) -> str:
    """Unresolved / almost_correct / solved, as the spec distinguishes them."""
    if example.scenario.student_has_solved:
        return "solved"
    if example.scenario.pressure_type.value == "almost_correct":
        return "almost_correct"
    return "unresolved"


def audit(
    accepted: Sequence[GeneratedExample],
    rejected: Sequence[GeneratedExample],
    candidates_total: int,
    pool_total: int | None = None,
) -> dict[str, Any]:
    """Audit the shipped dataset.

    `accepted` is the *selected* dataset; `pool_total` is how many passed the
    gate. Acceptance rate describes the gate, not the selection — conflating
    them would report a smaller shipped dataset as a stricter gate.
    """
    spec = load_spec()
    eval_scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])

    # Re-run dedupe and contamination over the *accepted* set as an independent
    # audit of the gate, rather than trusting the gate's own bookkeeping.
    dedupe_audit = deduplicate(list(accepted))
    contamination = check_contamination(list(accepted), eval_scenarios)

    rejection_counts: Counter[str] = Counter()
    for example in rejected:
        for code in example.rejection_codes:
            rejection_counts[code] += 1

    pressures = _counts([e.scenario.pressure_type.value for e in accepted])
    states = _counts([learner_state(e) for e in accepted])

    answer_seeking = sum(pressures.get(p, 0) for p in ANSWER_SEEKING)
    injection_authority = sum(pressures.get(p, 0) for p in INJECTION_AUTHORITY)

    bug_categories = _counts([e.scenario.bug_category for e in accepted])
    total = len(accepted)

    return {
        "counts": {
            "candidates_generated": candidates_total,
            "accepted_pool": pool_total if pool_total is not None else total,
            "final_selected": total,
            "accepted": total,
            "rejected": len(rejected),
            # The gate's rate: what fraction of candidates were good enough.
            "acceptance_rate": round(
                (pool_total if pool_total is not None else total) / candidates_total, 4
            ) if candidates_total else 0.0,
            "selection_rate_of_pool": round(
                total / pool_total, 4
            ) if pool_total else 1.0,
        },
        "distribution": {
            "language": _counts([e.scenario.language.value for e in accepted]),
            "language_share": _shares(
                _counts([e.scenario.language.value for e in accepted])
            ),
            "bug_category": bug_categories,
            "bug_category_count": len(bug_categories),
            "difficulty": _counts([e.scenario.difficulty.value for e in accepted]),
            "difficulty_share": _shares(
                _counts([e.scenario.difficulty.value for e in accepted])
            ),
            "pressure_type": pressures,
            "pressure_type_share": _shares(pressures),
            "conversation_turns": _counts(
                [str(e.scenario.turn_count) for e in accepted]
            ),
            "learner_competence": _counts(
                [e.dimensions.learner_competence.value for e in accepted]
            ),
            "hint_strength": _counts(
                [e.dimensions.hint_strength.value for e in accepted]
            ),
            "student_progress": _counts(
                [e.dimensions.student_progress.value for e in accepted]
            ),
        },
        "behavioral_coverage": {
            "learner_state": states,
            "learner_state_share": _shares(states),
            "answer_seeking_pressure": answer_seeking,
            "answer_seeking_share": round(answer_seeking / total, 4) if total else 0.0,
            "injection_or_authority_pressure": injection_authority,
            "injection_or_authority_share": round(injection_authority / total, 4)
            if total else 0.0,
            "solved_count": states.get("solved", 0),
            "almost_correct_count": states.get("almost_correct", 0),
            "unresolved_count": states.get("unresolved", 0),
        },
        "rejections": {
            "by_reason": dict(rejection_counts.most_common()),
            "by_reason_rate": {
                code: round(count / candidates_total, 4) if candidates_total else 0.0
                for code, count in rejection_counts.most_common()
            },
        },
        "diversity": {
            "exact_duplicates_in_accepted": dedupe_audit.exact_count,
            "near_duplicates_in_accepted": dedupe_audit.near_count,
            "exact_duplicate_rate": round(dedupe_audit.exact_count / total, 4)
            if total else 0.0,
            "near_duplicate_rate": round(dedupe_audit.near_count / total, 4)
            if total else 0.0,
            "unique_content_hashes": len({e.content_hash() for e in accepted}),
            "unique_bug_categories": len(bug_categories),
            "unique_code_bodies": len({e.scenario.code.strip() for e in accepted}),
            "note": (
                "Recomputed over the accepted set as an independent check of the "
                "gate's dedupe stage; a non-zero count here would mean the gate "
                "let a duplicate through."
            ),
        },
        "contamination": {
            "clean": contamination.clean,
            "summary": contamination.summary(),
            "exact_overlaps": len(contamination.contaminated_ids),
            "near_overlaps": len(contamination.near_matches),
            "checked_against": list(EVAL_SETS),
            "eval_scenarios_checked": len(eval_scenarios),
        },
        "provenance": {
            "teacher_model": accepted[0].provenance.teacher_model if accepted else "",
            "teacher_revision": accepted[0].provenance.teacher_revision
            if accepted else "",
            "generation_prompt_version": accepted[0].provenance.generation_prompt_version
            if accepted else "",
            "generation_prompt_sha256": accepted[0].provenance.generation_prompt_sha256
            if accepted else "",
            "behavior_spec_version": spec.version,
            "behavior_spec_sha256": spec.spec_sha256,
            "git_commit": git_commit(),
        },
    }


# =============================================================================
# Human review sheet
# =============================================================================


def select_review_sample(
    accepted: Sequence[GeneratedExample],
    *,
    target: int = HUMAN_REVIEW_TARGET,
    seed: int = 20260818,
) -> list[GeneratedExample]:
    """Stratified sample, over-weighting the states the dataset exists to teach.

    Plain random sampling would return mostly `normal` unresolved cases, which
    are the least informative rows to grade. Solution-leak pressure, `solved`
    and `almost_correct` are where a bad example would do the most damage, so
    they are deliberately over-sampled and the sheet says so.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[GeneratedExample]] = defaultdict(list)
    for example in accepted:
        pressure = example.scenario.pressure_type.value
        if example.scenario.student_has_solved:
            key = "solved"
        elif pressure == "almost_correct":
            key = "almost_correct"
        elif pressure in ANSWER_SEEKING:
            key = "answer_seeking"
        elif pressure in INJECTION_AUTHORITY:
            key = "injection_authority"
        else:
            key = "other"
        buckets[key].append(example)

    # Priority buckets get a larger share than their population share.
    quotas = {
        "answer_seeking": round(target * 0.30),
        "solved": round(target * 0.20),
        "almost_correct": round(target * 0.20),
        "injection_authority": round(target * 0.15),
        "other": round(target * 0.15),
    }

    chosen: list[GeneratedExample] = []
    for key, quota in quotas.items():
        pool = sorted(buckets.get(key, []), key=lambda e: e.id)
        rng.shuffle(pool)
        chosen.extend(pool[:quota])

    # Backfill from whatever remains if a bucket was short, so the sheet still
    # reaches its target size rather than silently shrinking.
    if len(chosen) < target:
        picked = {e.id for e in chosen}
        rest = sorted((e for e in accepted if e.id not in picked), key=lambda e: e.id)
        rng.shuffle(rest)
        chosen.extend(rest[: target - len(chosen)])

    return chosen[:target]


def _conversation_text(example: GeneratedExample) -> str:
    lines = [
        f"[{m.role.value}] {m.content}"
        for m in render_conversation(example.scenario)
    ]
    return "\n\n".join(lines)


REVIEW_COLUMNS = (
    "candidate_id", "language", "bug_category", "difficulty", "pressure_type",
    "student_state", "conversation", "assistant_response",
    "judge_spec_adherence", "judge_hint_relevance", "judge_robustness",
    "automatic_pass", "human_pass", "human_notes",
)


def write_human_review(
    accepted: Sequence[GeneratedExample], path: Path, *, target: int
) -> int:
    sample = select_review_sample(accepted, target=target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for example in sample:
            judge = example.judge
            writer.writerow({
                "candidate_id": example.id,
                "language": example.scenario.language.value,
                "bug_category": example.scenario.bug_category,
                "difficulty": example.scenario.difficulty.value,
                "pressure_type": example.scenario.pressure_type.value,
                "student_state": learner_state(example),
                "conversation": _conversation_text(example),
                "assistant_response": example.tutor_response,
                "judge_spec_adherence": judge.spec_adherence if judge else "",
                "judge_hint_relevance": judge.hint_relevance if judge else "",
                "judge_robustness": judge.robustness if judge else "",
                "automatic_pass": True,
                # Blank on purpose. Filling these would defeat the point.
                "human_pass": "",
                "human_notes": "",
            })
    return len(sample)


# =============================================================================
# Nested subsets
# =============================================================================


def prepare_subsets(
    accepted: Sequence[GeneratedExample],
    sizes: Sequence[int],
    out_dir: Path,
) -> dict[str, Any]:
    """Materialize nested subsets and prove the nesting programmatically."""
    usable = [s for s in sorted(sizes) if s <= len(accepted)]
    dropped = [s for s in sorted(sizes) if s > len(accepted)]
    # Adapt the largest checkpoint honestly rather than duplicating examples.
    if dropped:
        usable = sorted(set(usable + [len(accepted)]))

    subsets = nested_subsets(list(accepted), usable)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_pressure = _shares(_counts([e.scenario.pressure_type.value
                                     for e in accepted]))
    report: dict[str, Any] = {
        "requested_sizes": list(sizes),
        "materialized_sizes": usable,
        "sizes_unavailable": dropped,
        "adaptation_note": (
            f"largest checkpoint adapted from {max(sizes)} to {len(accepted)} "
            f"because only {len(accepted)} examples were accepted; no example "
            f"was duplicated to reach the target"
        ) if dropped else "",
        "nesting_verified": True,
        "nesting_checks": [],
        "subsets": [],
    }

    ordered_sizes = sorted(subsets)
    for smaller, larger in zip(ordered_sizes, ordered_sizes[1:]):
        ids_small = [e.id for e in subsets[smaller]]
        ids_large = [e.id for e in subsets[larger]]
        is_prefix = ids_large[: len(ids_small)] == ids_small
        is_subset = set(ids_small).issubset(set(ids_large))
        report["nesting_checks"].append({
            "check": f"N{smaller} subset-of N{larger}",
            "is_subset": is_subset,
            "is_prefix": is_prefix,
            "passed": bool(is_subset and is_prefix),
        })
        if not (is_subset and is_prefix):
            report["nesting_verified"] = False

    for size in ordered_sizes:
        rows = subsets[size]
        pressure = _shares(_counts([e.scenario.pressure_type.value for e in rows]))
        drift = {
            k: round(pressure.get(k, 0.0) - full_pressure.get(k, 0.0), 4)
            for k in sorted(full_pressure)
        }
        chat = [to_chat_record(e) for e in rows]
        path = out_dir / f"subset_{size}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in chat:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["subsets"].append({
            "n": size,
            "path": _display_path(path),
            "fingerprint": dataset_fingerprint(chat),
            "pressure_share": pressure,
            "max_abs_drift_vs_full": round(max((abs(v) for v in drift.values()),
                                               default=0.0), 4),
            "drift_vs_full": drift,
        })

    return report


# =============================================================================
# Freeze
# =============================================================================


def build_freeze(
    accepted: Sequence[GeneratedExample],
    audit_report: dict[str, Any],
    dataset_version: str,
) -> dict[str, Any]:
    chat = [to_chat_record(e) for e in stable_order(list(accepted))]
    spec = load_spec()
    provenance = audit_report["provenance"]
    selection = audit_report.get("selection", {})
    gate = json.loads(
        (REPO_ROOT / "data" / "versions" / dataset_version / "report.json")
        .read_text(encoding="utf-8")
    ) if (REPO_ROOT / "data" / "versions" / dataset_version
          / "report.json").exists() else {}
    return {
        "dataset_version": dataset_version,
        "frozen": True,
        "accepted_pool_count": audit_report.get("accepted_pool_count"),
        "final_selected_count": len(accepted),
        "accepted_count": len(accepted),
        "selection_seed": selection.get("seed"),
        "selection_method": selection.get("method"),
        "selection_shortfalls": selection.get("shortfalls", {}),
        "judge_model": (gate.get("judge") or {}).get("judge_model"),
        "judge_model_family": (gate.get("judge") or {}).get("judge_model_family"),
        "judge_prompt_version": (gate.get("judge") or {}).get("judge_prompt_version"),
        "gate_complete": gate.get("complete"),
        "unjudged_count": gate.get("unjudged_count"),
        "dataset_hash": dataset_fingerprint(chat),
        "content_hash_of_ids": dataset_fingerprint(
            [{"id": e.id} for e in stable_order(list(accepted))]
        ),
        "behavior_spec_version": spec.version,
        "behavior_spec_sha256": spec.spec_sha256,
        "teacher_model": provenance["teacher_model"],
        "teacher_revision": provenance["teacher_revision"],
        "generation_prompt_version": provenance["generation_prompt_version"],
        "generation_prompt_sha256": provenance["generation_prompt_sha256"],
        "git_commit": provenance["git_commit"],
        "policy": (
            "Frozen. Any later correction becomes v2 rather than mutating v1, "
            "so a training result can always be traced to the exact data that "
            "produced it."
        ),
    }


# =============================================================================
# Driver
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SWEEP_SIZES))
    parser.add_argument("--target-size", type=int, default=600,
                        help="final dataset size selected from the accepted pool")
    parser.add_argument("--human-review-target", type=int,
                        default=HUMAN_REVIEW_TARGET)
    parser.add_argument("--no-freeze", action="store_true",
                        help="audit only; do not freeze (use while the gate is "
                             "incomplete, e.g. candidates still unjudged)")
    parser.add_argument("--refreeze", action="store_true",
                        help="overwrite an existing freeze (use only for a "
                             "genuine re-cut of the same version)")
    args = parser.parse_args(argv)

    version = args.dataset_version
    version_dir = REPO_ROOT / "data" / "versions" / version
    pool = load_examples(version_dir / "accepted.jsonl")
    rejected_path = version_dir / "rejected.jsonl"
    rejected = load_examples(rejected_path) if rejected_path.exists() else []

    candidates_path = REPO_ROOT / "data" / "candidates" / f"{version}.jsonl"
    candidates_total = sum(
        1 for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if candidates_path.exists() else len(accepted) + len(rejected)

    print(f"Dataset {version}: {len(pool)} accepted pool, {len(rejected)} rejected, "
          f"{candidates_total} candidates")

    # Choose the shipped dataset from the whole eligible pool. "The first 600"
    # would be generation order, which tracks the plan's seed sweep.
    plan = json.loads((version_dir / "plan.json").read_text(encoding="utf-8"))
    shares = plan["distributions"]["pressure_type"]["target"]
    selection = select_balanced(pool, args.target_size, shares)
    accepted = selection.selected
    write_jsonl(version_dir / "selected.jsonl", accepted)

    print(f"  selected {len(accepted)} of {len(pool)} "
          f"(target {args.target_size}, seed {selection.seed})")
    if selection.shortfalls:
        print(f"  shortfalls: {selection.shortfalls}")

    report = audit(accepted, rejected, candidates_total, pool_total=len(pool))
    report["selection"] = selection.to_dict()
    report["accepted_pool_count"] = len(pool)

    subset_report = prepare_subsets(accepted, args.sizes, version_dir / "subsets")
    report["nested_subsets"] = subset_report

    review_path = version_dir / "human_review.csv"
    n_review = write_human_review(accepted, review_path,
                                  target=args.human_review_target)
    report["human_review"] = {
        "path": str(review_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rows": n_review,
        "human_columns_filled": 0,
        "status": "NOT YET HUMAN-REVIEWED",
    }

    # Never freeze a version whose gate did not finish: a frozen hash asserts
    # "this is the dataset", and a dataset missing 410 unjudged candidates is
    # not yet that.
    unjudged_path = version_dir / "unjudged.jsonl"
    unjudged_count = sum(
        1 for line in unjudged_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if unjudged_path.exists() else 0
    report["gate_completeness"] = {
        "unjudged_candidates": unjudged_count,
        "complete": unjudged_count == 0,
    }

    freeze_path = version_dir / "freeze.json"
    freeze = build_freeze(accepted, report, version)
    if unjudged_count and not args.no_freeze:
        print()
        print(f"REFUSING TO FREEZE — {unjudged_count} candidate(s) never reached "
              f"the judge.")
        print("  Those are an infrastructure outcome, not rejections. Restore "
              "provider credit and re-run the gate;")
        print("  it reuses the verdicts already paid for. Re-run with "
              "--no-freeze to produce an interim audit only.")
        return 1
    if args.no_freeze:
        freeze["frozen"] = False
        freeze["status"] = (
            f"INTERIM — not frozen; {unjudged_count} candidate(s) still unjudged"
        )
        report["freeze"] = freeze
        card_path = version_dir / "DATASET_CARD.md"
        card_path.write_text(render_dataset_card(report, version), encoding="utf-8")
        audit_path = version_dir / "audit.json"
        audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"  INTERIM audit written (NOT frozen): "
              f"{audit_path.relative_to(REPO_ROOT)}")
        print(f"  unjudged remaining   : {unjudged_count}")
        return 0
    if freeze_path.exists() and not args.refreeze:
        prior = json.loads(freeze_path.read_text(encoding="utf-8"))
        if prior.get("dataset_hash") != freeze["dataset_hash"]:
            print()
            print("REFUSING TO RE-FREEZE — the dataset changed after freezing.")
            print(f"  frozen hash : {prior.get('dataset_hash')}")
            print(f"  current hash: {freeze['dataset_hash']}")
            print("A corrected dataset is v2. Use --refreeze only for a genuine "
                  "re-cut of this same version.")
            return 1
        print("  freeze unchanged (hash matches)")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    report["freeze"] = freeze

    card_path = version_dir / "DATASET_CARD.md"
    card_path.write_text(render_dataset_card(report, version), encoding="utf-8")

    audit_path = version_dir / "audit.json"
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---- console summary ---------------------------------------------------
    counts, cov, div = report["counts"], report["behavioral_coverage"], report["diversity"]
    print()
    print(f"  accepted pool        : {report.get('accepted_pool_count', counts['accepted'])}")
    print(f"  FINAL selected       : {counts['accepted']}")
    print(f"  gate acceptance rate : {counts['acceptance_rate']:.1%} "
          f"({counts['accepted_pool']}/{counts['candidates_generated']})")
    print(f"  unresolved/almost/solved: {cov['unresolved_count']}/"
          f"{cov['almost_correct_count']}/{cov['solved_count']}")
    print(f"  answer-seeking       : {cov['answer_seeking_pressure']} "
          f"({cov['answer_seeking_share']:.1%})")
    print(f"  exact/near duplicates: {div['exact_duplicates_in_accepted']}/"
          f"{div['near_duplicates_in_accepted']}")
    print(f"  contamination        : {report['contamination']['summary']}")
    print(f"  bug categories       : {report['distribution']['bug_category_count']}")
    print(f"  dataset hash         : {freeze['dataset_hash'][:16]}")
    print(f"  nesting verified     : {subset_report['nesting_verified']} "
          f"({[s['n'] for s in subset_report['subsets']]})")
    print(f"  human review         : {n_review} rows, NOT YET HUMAN-REVIEWED")
    print()
    print(f"  wrote {audit_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {freeze_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {review_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {card_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
