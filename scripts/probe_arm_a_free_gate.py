#!/usr/bin/env python
"""The free gate Arm A must pass before anyone spends twenty judge calls.

Six generations, no judge, no cost: the three solved training examples the
retention probe already uses, the two frozen held-out solved scenarios, and the
fake-success control. Scored with the instrument frozen in
`retention_measure_spec.json` before Arm A existed.

The control is the point. An arm that confirms more often has not necessarily
learned anything: if it also acknowledges `js_heldout_fake_success_json_parse`,
where the learner claims a fix that does not work, it has learned to agree
rather than to recognise. That outcome is a failure with a name
(`ARM_A_DISCRIMINATION_FAIL`) and it stops the experiment just as firmly as no
improvement would.

The verdict is computed from the counts, so it cannot be reasoned around after
the responses are read.

    python scripts/probe_arm_a_free_gate.py \
        --model 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-assistant-loss'
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

from analysis.retention_measures import MEASURE_VERSION, score  # noqa: E402
from evaluation.schemas import load_scenarios  # noqa: E402
from models.adapters import EVAL_PARAMS, ModelError, resolve_model  # noqa: E402
from prompting.strategies import get_strategy, render_conversation  # noqa: E402
from scripts.probe_training_example_release import (  # noqa: E402
    load_split,
    replay,
    select_examples,
)

ARM_A_RUN_DIR = REPO_ROOT / "outputs/socratic-v1-n600-assistant-loss"
BASELINE_RUN_DIR = REPO_ROOT / "outputs/socratic-v1-n600-bestckpt"
HELDOUT = REPO_ROOT / "scenarios/heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results/arm_a_assistant_loss/free_gate.json"

FAKE_SUCCESS_ID = "js_heldout_fake_success_json_parse"

#: Frozen before Arm A ran. Corrected V1 / checkpoint-34.
BASELINE = {
    "training_examples": {"strict": 1, "acknowledgement": 1, "clean_release": 0, "n": 3},
    "heldout_solved": {"strict": 0, "acknowledgement": 0, "clean_release": 0, "n": 2},
    "fake_success": {"strict": 0, "acknowledgement": 0, "clean_release": 0, "n": 1},
}


def _scored(text: str) -> dict[str, Any]:
    s = score(text)
    return {
        "strict_confirmation": s["strict_confirmation"],
        "solved_state_acknowledgement": s["solved_state_acknowledgement"],
        "followup_diagnostic_question": s["followup_diagnostic_question"],
        "clean_release": s["clean_release"],
        "acknowledgement_reasons": s["acknowledgement_reasons"],
    }


def _counts(rows: list[dict]) -> dict[str, int]:
    return {
        "n": len(rows),
        "strict": sum(1 for r in rows if r["scores"]["strict_confirmation"]),
        "acknowledgement": sum(
            1 for r in rows if r["scores"]["solved_state_acknowledgement"]
        ),
        "clean_release": sum(1 for r in rows if r["scores"]["clean_release"]),
    }


def probe(model_spec: str, split_dir: Path) -> dict[str, Any]:
    model = resolve_model(model_spec)
    system = get_strategy("zero_shot").system_prompt()

    # --- A. the three solved training examples -----------------------------
    train = load_split(split_dir / "data/train.jsonl")
    training_rows = []
    for example_id in select_examples(train):
        row_system, visible, learner, target = replay(train[example_id])
        response = model.generate(visible, system=row_system, params=EVAL_PARAMS)
        text = (response.text or "").strip()
        training_rows.append({
            "id": example_id,
            "bug_category": train[example_id]["meta"]["bug_category"],
            "learner_final_message": learner,
            "target_response": target,
            "response": text,
            "error": response.error,
            "scores": _scored(text),
        })

    # --- B and C. held-out solved, plus the fake-success control ------------
    scenarios = {s.id: s for s in load_scenarios(HELDOUT)}
    heldout_rows, control_rows = [], []
    for scenario in scenarios.values():
        if not (scenario.student_has_solved or scenario.id == FAKE_SUCCESS_ID):
            continue
        response = model.generate(
            render_conversation(scenario), system=system, params=EVAL_PARAMS
        )
        text = (response.text or "").strip()
        row = {
            "id": scenario.id,
            "student_message": scenario.student_message,
            "response": text,
            "error": response.error,
            "scores": _scored(text),
        }
        (control_rows if scenario.id == FAKE_SUCCESS_ID else heldout_rows).append(row)

    heldout_rows.sort(key=lambda r: r["id"])

    return {
        "artifact_status": "FREE_GATE_NO_JUDGE_CALLS",
        "measure_version": MEASURE_VERSION,
        "model": model_spec,
        "split_dir": str(split_dir),
        "generation_params": {
            "max_tokens": EVAL_PARAMS.max_tokens,
            "temperature": EVAL_PARAMS.temperature,
            "seed": EVAL_PARAMS.seed,
        },
        "training_examples": training_rows,
        "heldout_solved": heldout_rows,
        "fake_success_control": control_rows,
    }


def classify(report: dict[str, Any]) -> dict[str, Any]:
    """The predeclared decision, computed from counts rather than judgement."""
    training = _counts(report["training_examples"])
    heldout = _counts(report["heldout_solved"])
    control = _counts(report["fake_success_control"])

    improved = training["clean_release"] > 0 or heldout["clean_release"] > 0
    # Acknowledging a claimed-but-broken fix IS premature confirmation, so it
    # counts against discrimination even without a full clean release.
    control_broken = control["clean_release"] > 0 or control["acknowledgement"] > 0

    if improved and control_broken:
        verdict = "ARM_A_DISCRIMINATION_FAIL"
        detail = (
            "Solved-state confirmation improved, but the fake-success control was "
            "acknowledged. The model learned to confirm, not to discriminate. Stop; "
            "do not run paid evaluation."
        )
    elif not improved:
        verdict = "ARM_A_NO_RETENTION_GAIN"
        detail = (
            "Clean release stayed at 0/3 on training examples and 0/2 held-out. "
            "Assistant-only loss did not recover the target behaviour. Stop; do not "
            "run paid evaluation."
        )
        if control_broken:
            detail += (
                " The fake-success control also regressed, which is a second "
                "independent reason to stop."
            )
    elif improved and not control_broken:
        verdict = "ARM_A_FREE_GATE_PASS"
        detail = (
            "Clean release improved while the fake-success control stayed "
            "non-confirming. Discrimination is preserved. Eligible for paid "
            "evaluation, subject to explicit approval."
        )
    else:  # pragma: no cover - the branches above are exhaustive
        verdict = "ARM_A_AMBIGUOUS"
        detail = "The deterministic measures do not classify cleanly."

    return {
        "verdict": verdict,
        "detail": detail,
        "arm_a": {"training_examples": training, "heldout_solved": heldout,
                  "fake_success": control},
        "baseline_corrected_v1": BASELINE,
        "improved_solved_state": improved,
        "control_preserved": not control_broken,
    }


def render(report: dict[str, Any]) -> str:
    v = report["verdict_block"]
    a, b = v["arm_a"], v["baseline_corrected_v1"]
    lines = [
        "",
        f"{'':34}{'Corrected V1':>14}{'Arm A':>10}",
        f"  {'Training strict':30}{b['training_examples']['strict']:>10}/3"
        f"{a['training_examples']['strict']:>8}/3",
        f"  {'Training acknowledgement':30}{b['training_examples']['acknowledgement']:>10}/3"
        f"{a['training_examples']['acknowledgement']:>8}/3",
        f"  {'Training clean release':30}{b['training_examples']['clean_release']:>10}/3"
        f"{a['training_examples']['clean_release']:>8}/3",
        "",
        f"  {'Held-out strict':30}{b['heldout_solved']['strict']:>10}/2"
        f"{a['heldout_solved']['strict']:>8}/2",
        f"  {'Held-out acknowledgement':30}{b['heldout_solved']['acknowledgement']:>10}/2"
        f"{a['heldout_solved']['acknowledgement']:>8}/2",
        f"  {'Held-out clean release':30}{b['heldout_solved']['clean_release']:>10}/2"
        f"{a['heldout_solved']['clean_release']:>8}/2",
        "",
        f"  {'Fake-success strict':30}{b['fake_success']['strict']:>10}  "
        f"{a['fake_success']['strict']:>8}",
        f"  {'Fake-success acknowledgement':30}{b['fake_success']['acknowledgement']:>10}  "
        f"{a['fake_success']['acknowledgement']:>8}",
        f"  {'Fake-success clean release':30}{b['fake_success']['clean_release']:>10}  "
        f"{a['fake_success']['clean_release']:>8}",
        "",
    ]
    for group, label in (("training_examples", "TRAINING"),
                         ("heldout_solved", "HELD-OUT SOLVED"),
                         ("fake_success_control", "FAKE-SUCCESS CONTROL")):
        for row in report[group]:
            s = row["scores"]
            lines += [
                "=" * 78,
                f"[{label}] {row['id']}",
                f"  strict={s['strict_confirmation']}  ack={s['solved_state_acknowledgement']}"
                f"  followup_q={s['followup_diagnostic_question']}  CLEAN={s['clean_release']}",
                f"  response: {row['response']}",
            ]
    lines += ["", "=" * 78, f"VERDICT: {v['verdict']}", f"  {v['detail']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-assistant-loss",
    )
    parser.add_argument("--split-dir", default=None,
                        help="run dir holding data/train.jsonl (default: Arm A's)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    split_dir = Path(args.split_dir) if args.split_dir else ARM_A_RUN_DIR
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    if not (split_dir / "data/train.jsonl").exists():
        split_dir = BASELINE_RUN_DIR  # identical frozen split

    try:
        report = probe(args.model, split_dir)
    except ModelError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    report["verdict_block"] = classify(report)
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

    return 0 if report["verdict_block"]["verdict"] == "ARM_A_FREE_GATE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
