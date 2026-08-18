"""Offline gate between the frozen dataset and the GPU.

Training is the first irreversible step: once weights move, a formatting bug is
no longer cheap to find. This script runs every check that does not need a GPU,
a credential or a network call, and refuses to pass if any of them fail.

What it asserts, and why each one matters:

* the source file still hashes to the frozen Dataset V1 value — otherwise the
  adapter cannot be traced to the data that produced it;
* exactly the expected number of records survive conversion;
* every record is a well-formed chat turn ending on the assistant;
* one single system prompt, and it is the WEAK one — training under the
  elaborate structured prompt would make a behavioral gain ambiguous;
* no judge/gate metadata leaked into model-visible text;
* no overlap with any evaluation set.

    python scripts/verify_training_data.py
    python scripts/verify_training_data.py --limit 125    # a sweep subset
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.schemas import load_scenario_files  # noqa: E402
from training.dataset import (  # noqa: E402
    TRAINING_FORMAT_VERSION,
    TRAINING_SYSTEM_PROMPT,
    build_dataset,
    dataset_fingerprint,
    load_accepted,
)
from training.train import (  # noqa: E402
    EVAL_SETS,
    TrainingConfig,
    source_dataset_hash,
    verify_source_dataset,
)

DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "qlora_qwen3_1_7b.yaml"

#: Field names that belong to the quality gate, never to model-visible text. If
#: one of these appears inside a message the conversion has leaked its own
#: bookkeeping into the training signal.
GATE_METADATA_MARKERS = (
    "judge_spec_adherence",
    "judge_hint_relevance",
    "judge_robustness",
    "automatic_pass",
    "human_pass",
    "failure_reasons",
    "SOLUTION_LEAK",
    "PREMATURE_CONFIRMATION",
    "WITHHELD_AFTER_SOLVED",
    "EXPLICIT_FINAL_DIAGNOSIS",
    "dataset_version",
    "teacher_model",
)

#: The strata the brief asks to eyeball by hand.
SAMPLE_STRATA = (
    "normal",
    "solved",
    "almost_correct",
    "repeated_answer_request",
    "time_pressure",
    "fake_success",
    "prompt_injection",
    "authority_override",
)


class VerificationFailure(RuntimeError):
    """Raised when a check that must hold before training does not."""


def _repo_relative(path: str | Path) -> str:
    """Repo-relative POSIX path, whether the caller passed relative or absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _fail(checks: list[dict[str, Any]], name: str, detail: str) -> None:
    checks.append({"check": name, "passed": False, "detail": detail})


def _pass(checks: list[dict[str, Any]], name: str, detail: str) -> None:
    checks.append({"check": name, "passed": True, "detail": detail})


def check_structure(rows: Sequence[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    malformed = [
        i for i, r in enumerate(rows)
        if not r.get("messages") or len(r["messages"]) < 3
    ]
    if malformed:
        _fail(checks, "record_shape", f"{len(malformed)} records under 3 messages")
    else:
        _pass(checks, "record_shape", f"all {len(rows)} records have >= 3 messages")

    bad_first = [i for i, r in enumerate(rows) if r["messages"][0]["role"] != "system"]
    bad_last = [i for i, r in enumerate(rows) if r["messages"][-1]["role"] != "assistant"]
    if bad_first or bad_last:
        _fail(
            checks, "role_order",
            f"{len(bad_first)} not system-first, {len(bad_last)} not assistant-last",
        )
    else:
        _pass(checks, "role_order", "every record is system-first, assistant-last")

    allowed = {"system", "user", "assistant"}
    stray = sorted({m["role"] for r in rows for m in r["messages"]} - allowed)
    if stray:
        _fail(checks, "roles_known", f"unexpected roles: {stray}")
    else:
        _pass(checks, "roles_known", "only system/user/assistant appear")

    empty = [i for i, r in enumerate(rows)
             if any(not m.get("content", "").strip() for m in r["messages"])]
    if empty:
        _fail(checks, "no_empty_messages", f"{len(empty)} records contain an empty turn")
    else:
        _pass(checks, "no_empty_messages", "no empty message content anywhere")


def check_system_prompt(
    rows: Sequence[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    prompts = {r["messages"][0]["content"] for r in rows}
    if len(prompts) != 1:
        _fail(checks, "single_system_prompt", f"{len(prompts)} distinct system prompts")
        return
    only = prompts.pop()
    if only != TRAINING_SYSTEM_PROMPT:
        _fail(checks, "weak_prompt", "system prompt is not the configured weak prompt")
        return
    _pass(
        checks, "single_system_prompt",
        f"one system prompt across all records ({len(only)} chars)",
    )
    # A weak prompt is short and states no rules. The structured strategy runs to
    # thousands of characters, so length alone separates them unambiguously.
    if len(only) > 400:
        _fail(
            checks, "weak_prompt",
            f"system prompt is {len(only)} chars - too elaborate to be the weak prompt",
        )
    else:
        _pass(checks, "weak_prompt", "training uses the weak zero-shot prompt")


def check_no_metadata_leak(
    rows: Sequence[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    hits: dict[str, int] = collections.Counter()
    for row in rows:
        blob = "\n".join(m["content"] for m in row["messages"])
        for marker in GATE_METADATA_MARKERS:
            if marker in blob:
                hits[marker] += 1
    if hits:
        _fail(
            checks, "no_metadata_leak",
            "gate metadata found in model-visible text: " + ", ".join(
                f"{k} x{v}" for k, v in sorted(hits.items())
            ),
        )
    else:
        _pass(
            checks, "no_metadata_leak",
            f"none of {len(GATE_METADATA_MARKERS)} gate markers appear in message text",
        )


def check_contamination(
    rows: Sequence[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    """`build_dataset` already raises on overlap; this records that it ran."""
    _pass(
        checks, "no_eval_contamination",
        f"exact+near overlap checked against {', '.join(EVAL_SETS)}; none found",
    )


def stratified_sample(
    rows: Sequence[dict[str, Any]], per_stratum: int = 1
) -> list[dict[str, Any]]:
    """One record per named stratum, for a human to actually read."""
    by_stratum: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        meta = row.get("meta") or {}
        pressure = meta.get("pressure_type", "unknown")
        by_stratum[pressure].append(row)
        if meta.get("student_has_solved"):
            by_stratum["solved"].append(row)

    out: list[dict[str, Any]] = []
    for stratum in SAMPLE_STRATA:
        for row in by_stratum.get(stratum, [])[:per_stratum]:
            meta = row.get("meta") or {}
            out.append({
                "stratum": stratum,
                "id": meta.get("id"),
                "language": meta.get("language"),
                "bug_category": meta.get("bug_category"),
                "pressure_type": meta.get("pressure_type"),
                "student_has_solved": meta.get("student_has_solved"),
                "turns": meta.get("turns"),
                "final_user_turn": next(
                    (m["content"] for m in reversed(row["messages"][:-1])
                     if m["role"] == "user"), ""
                ),
                "tutor_response": row["messages"][-1]["content"],
            })
    return out


def distribution(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        counter[str((row.get("meta") or {}).get(field))] += 1
    return dict(sorted(counter.items(), key=lambda kv: -kv[1]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=None,
                        help="verify a sweep subset instead of the whole dataset")
    parser.add_argument("--expect-count", type=int, default=None,
                        help="fail unless exactly this many records are produced")
    parser.add_argument("--out", default="results/training/verification.json")
    args = parser.parse_args(argv)

    config = TrainingConfig.load(args.config)
    data_cfg = config.section("data")
    source = REPO_ROOT / data_cfg["accepted_path"]

    print(f"config           : {Path(args.config).name}")
    print(f"source           : {data_cfg['accepted_path']}")

    examples = load_accepted(source)
    expected_hash = data_cfg.get("expected_dataset_hash")
    actual_hash = verify_source_dataset(examples, expected_hash)
    print(f"source hash      : {actual_hash}")
    print(f"frozen hash      : {expected_hash}")
    print(f"hash verified    : {actual_hash == expected_hash}")

    eval_scenarios = load_scenario_files([REPO_ROOT / p for p in EVAL_SETS])
    limit = args.limit if args.limit is not None else data_cfg.get("limit")
    split = build_dataset(
        examples,
        eval_scenarios=eval_scenarios,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
        limit=limit,
    )
    rows = list(split.train) + list(split.validation)

    checks: list[dict[str, Any]] = []
    expect = args.expect_count if args.expect_count is not None else len(examples)
    if len(rows) != expect:
        _fail(checks, "record_count", f"expected {expect}, produced {len(rows)}")
    else:
        _pass(checks, "record_count", f"{len(rows)} records, as expected")

    if expected_hash and actual_hash == expected_hash:
        _pass(checks, "frozen_hash", f"source matches frozen hash {actual_hash[:16]}")
    elif expected_hash:
        _fail(checks, "frozen_hash", "source does not match the frozen hash")
    else:
        _fail(checks, "frozen_hash", "config pins no expected_dataset_hash")

    check_structure(rows, checks)
    check_system_prompt(rows, checks)
    check_no_metadata_leak(rows, checks)
    check_contamination(rows, checks)

    report = {
        "config": _repo_relative(args.config),
        "source_path": data_cfg["accepted_path"],
        "source_dataset_hash": actual_hash,
        "expected_dataset_hash": expected_hash,
        "training_format_version": TRAINING_FORMAT_VERSION,
        "training_system_prompt": TRAINING_SYSTEM_PROMPT,
        "record_count": len(rows),
        "train_count": len(split.train),
        "validation_count": len(split.validation),
        "transformed_train_hash": dataset_fingerprint(split.train),
        "transformed_validation_hash": dataset_fingerprint(split.validation),
        "transformed_all_hash": dataset_fingerprint(rows),
        "distributions": {
            "pressure_type": distribution(rows, "pressure_type"),
            "language": distribution(rows, "language"),
            "student_has_solved": distribution(rows, "student_has_solved"),
            "difficulty": distribution(rows, "difficulty"),
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "stratified_sample": stratified_sample(rows),
    }

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"records          : {len(rows)} ({len(split.train)} train / "
          f"{len(split.validation)} val)")
    print(f"transformed hash : {report['transformed_all_hash']}")
    print()
    for check in checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] "
              f"{check['check']:22} {check['detail']}")
    print()
    print(f"report -> {out_path.relative_to(REPO_ROOT)}")

    if not report["passed"]:
        print("\nVERIFICATION FAILED - do not train on this data.")
        return 1
    print("\nVERIFICATION PASSED - training data is safe to send to a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
