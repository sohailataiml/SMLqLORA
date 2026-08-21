#!/usr/bin/env python
"""Can the corrected adapter still confirm on an example it trained on?

The release probe established that the adapter confirms in 0 of 20 held-out
outputs while the base model confirms when told the rule. That leaves two very
different explanations, and they imply opposite next steps:

* the behaviour is in the weights but does not reach nearby unseen cases -- a
  generalisation/discrimination failure, which a contrastive Dataset V2 could
  plausibly address;
* the behaviour is not in the weights at all, having been overwritten during
  optimisation -- in which case no amount of data fixes it and the training
  recipe is the thing to change.

One measurement separates them: feed the adapter an example it was actually
trained on and see whether it reproduces that example's confirmation.

The inputs are taken from `outputs/<run>/data/train.jsonl`, the exact file the
trainer consumed, so training-split membership is a fact about the file rather
than an inference. The conversation is replayed as the trainer saw it -- same
system prompt, same turns, target turn withheld -- with no rewriting of learner
or target text.

THIS IS A DIAGNOSTIC PROBE, NOT AN EXPERIMENT. No judge calls, no metric
comparable with N600_V1_BASELINE, nothing claimable as a result.

    python scripts/probe_training_example_release.py \
        --model 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.solved_state import tutor_profile  # noqa: E402
from evaluation.schemas import Message, Role  # noqa: E402
from models.adapters import EVAL_PARAMS, ModelError, resolve_model  # noqa: E402
from prompting.strategies import get_strategy  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "outputs/socratic-v1-n600-bestckpt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/solved_state_analysis/training_example_release_probe.json"
)

#: The two examples the forensic analysis found nearest to the held-out solved
#: failures. If the behaviour survives anywhere, it should survive here.
ANCHOR_IDS = ("gen_v1_00486", "gen_v1_00792")

#: How many further solved training examples to add, drawn deterministically
#: from bug categories the anchors do not already cover.
EXTRA_EXAMPLES = 1


def load_split(path: Path) -> dict[str, dict]:
    """The trainer's own file, keyed by example id."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["meta"]["id"]: row for row in rows}


def select_examples(train: dict[str, dict], extra: int = EXTRA_EXAMPLES) -> list[str]:
    """Anchors first, then further solved examples from uncovered categories.

    Deterministic: candidates are considered in sorted id order, so the same
    split always yields the same selection.
    """
    chosen = [i for i in ANCHOR_IDS if i in train]
    covered = {train[i]["meta"]["bug_category"] for i in chosen}
    for example_id in sorted(train):
        if len(chosen) >= len(ANCHOR_IDS) + extra:
            break
        meta = train[example_id]["meta"]
        if not meta.get("student_has_solved"):
            continue
        if example_id in chosen or meta["bug_category"] in covered:
            continue
        chosen.append(example_id)
        covered.add(meta["bug_category"])
    return chosen


def replay(row: dict) -> tuple[str, list[Message], str, str]:
    """Rebuild what the trainer showed the model, and what it asked for back.

    Returns (system prompt, visible turns, learner's final message, target).
    """
    messages = row["messages"]
    if messages[0]["role"] != "system":
        raise ValueError(f"{row['meta']['id']}: expected a system turn first")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{row['meta']['id']}: expected an assistant target last")

    system = messages[0]["content"]
    target = messages[-1]["content"]
    visible = [
        Message(role=Role(m["role"]), content=m["content"])
        for m in messages[1:-1]
    ]
    learner_final = next(
        (m["content"] for m in reversed(messages[1:-1]) if m["role"] == "user"), ""
    )
    return system, visible, learner_final, target


def probe(
    model_spec: str, run_dir: Path, extra: int = EXTRA_EXAMPLES
) -> dict[str, Any]:
    train = load_split(run_dir / "data/train.jsonl")
    validation = load_split(run_dir / "data/validation.jsonl")
    if set(train) & set(validation):
        raise ValueError("train and validation splits overlap; the probe is invalid")

    selected = select_examples(train, extra)
    zero_shot = get_strategy("zero_shot").system_prompt()

    try:
        model = resolve_model(model_spec)
    except ModelError as exc:
        return {
            "artifact_status": "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT",
            "error": str(exc),
            "model": model_spec,
        }

    results: list[dict[str, Any]] = []
    for example_id in selected:
        row = train[example_id]
        system, visible, learner_final, target = replay(row)
        response = model.generate(visible, system=system, params=EVAL_PARAMS)

        target_behaviour = tutor_profile(target)
        generated_behaviour = tutor_profile(response.text or "")
        results.append({
            "example_id": example_id,
            "in_training_split": True,
            "in_validation_split": example_id in validation,
            "bug_category": row["meta"]["bug_category"],
            "language": row["meta"]["language"],
            "student_has_solved": row["meta"]["student_has_solved"],
            "turns": row["meta"]["turns"],
            "system_prompt_is_the_frozen_zero_shot": system == zero_shot,
            "learner_final_message": learner_final,
            "target_response": target,
            "generated_response": (response.text or "").strip(),
            "error": response.error,
            "target_confirms": target_behaviour["confirms"],
            "generated_confirms": generated_behaviour["confirms"],
            "generated_asks_question": generated_behaviour["asks_question"],
            "generated_explains": generated_behaviour["explains"],
        })

    return {
        "artifact_status": "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT",
        "note": (
            "No judge calls. Inputs are replayed verbatim from the trainer's own "
            "train.jsonl. Not comparable with N600_V1_BASELINE and not claimable "
            "as a result."
        ),
        "question": (
            "Does the corrected adapter reproduce solved-state confirmation on "
            "examples it was trained on?"
        ),
        "model": model_spec,
        "run_dir": str(run_dir),
        "train_split_size": len(train),
        "validation_split_size": len(validation),
        "splits_disjoint": True,
        "selected_examples": selected,
        "generation_params": {
            "max_tokens": EVAL_PARAMS.max_tokens,
            "temperature": EVAL_PARAMS.temperature,
            "seed": EVAL_PARAMS.seed,
        },
        "results": results,
    }


def classify(report: dict[str, Any]) -> dict[str, Any]:
    """CASE 1 / 2 / 3, fixed before the run and applied mechanically."""
    rows = [r for r in report.get("results", []) if r.get("target_confirms")]
    total = len(rows)
    confirmed = sum(1 for r in rows if r["generated_confirms"])

    if total == 0:
        case, conclusion = "INVALID", (
            "No selected example has a confirming target, so the probe cannot "
            "ask its question."
        )
    elif confirmed == total:
        case, conclusion = "CASE_1", (
            "The solved-release behaviour is still present in the fine-tuned "
            "weights but does not generalise reliably to nearby unseen cases. "
            "This supports a data-discrimination/generalisation hypothesis and "
            "makes a contrastive Dataset V2 scientifically defensible."
        )
    elif confirmed == 0:
        case, conclusion = "CASE_2", (
            "The behaviour was not merely underrepresented; it was suppressed or "
            "overwritten by optimisation. Do NOT build V2 yet. Investigate "
            "training dynamics: learning rate, epochs, checkpoint timing, "
            "system-prompt invariance, assistant-only loss, the balance between "
            "unresolved and solved behaviour, and catastrophic forgetting."
        )
    else:
        case, conclusion = "CASE_3", (
            "Behaviour retention is unstable across training examples. Do not "
            "claim a clean data hypothesis. Report which examples retain "
            "confirmation and which do not, and identify the distinguishing "
            "factors."
        )

    return {
        "case": case,
        "examples_with_confirming_target": total,
        "generated_confirmations": confirmed,
        "conclusion": conclusion,
        "retained": [r["example_id"] for r in rows if r["generated_confirms"]],
        "lost": [r["example_id"] for r in rows if not r["generated_confirms"]],
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for row in report.get("results", []):
        lines += [
            "",
            "=" * 78,
            f"example id       : {row['example_id']}",
            f"in training split: {row['in_training_split']}",
            f"bug category     : {row['bug_category']} ({row['language']})",
            f"frozen prompt    : {row['system_prompt_is_the_frozen_zero_shot']}",
            "",
            f"learner (final)  : {row['learner_final_message']}",
            "",
            f"TARGET           : {row['target_response']}",
            "",
            f"GENERATED        : {row['generated_response']}",
            "",
            f"target confirms      : {row['target_confirms']}",
            f"generated confirms   : {row['generated_confirms']}",
            f"generated asks a Q   : {row['generated_asks_question']}",
        ]
    verdict = report.get("verdict") or {}
    lines += [
        "",
        "=" * 78,
        f"confirmations reproduced: {verdict.get('generated_confirmations')}"
        f"/{verdict.get('examples_with_confirming_target')} training examples",
        f"retained : {verdict.get('retained')}",
        f"lost     : {verdict.get('lost')}",
        "",
        f"VERDICT: {verdict.get('case')}",
        f"  {verdict.get('conclusion')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--extra", type=int, default=EXTRA_EXAMPLES)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    report = probe(args.model, run_dir, args.extra)
    if "error" in report:
        print(f"\nERROR: {report['error']}\n", file=sys.stderr)
        return 2
    report["verdict"] = classify(report)
    print(render(report))

    if args.write:
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
