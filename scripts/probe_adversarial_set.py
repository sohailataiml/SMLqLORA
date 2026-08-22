#!/usr/bin/env python
"""Run the partner's 16 adversarial prompts against a model. No judge calls.

The partner tested revision 16d60373 -- the published MVP adapter, which is the
epoch-3 checkpoint we later proved was the wrong export. Seven of their sixteen
responses carry that checkpoint's signature attractor. So before generating any
training data against these failures, it is worth knowing which of them survive
on the corrected baseline: a failure that is already gone is not something
Dataset V2 needs to fix.

Scoring is deterministic (`analysis.adversarial_checks`) and was calibrated
against the partner's own responses, where it reproduces 16/16 broken.

    python scripts/probe_adversarial_set.py \
        --model 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt' \
        --model hf:Qwen/Qwen3-1.7B --write
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

from analysis.adversarial_checks import check_all, load_cases  # noqa: E402
from evaluation.schemas import Message, Role  # noqa: E402
from models.adapters import EVAL_PARAMS, ModelError, resolve_model  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "results/adversarial/partner_ae_v1_probe.json"

#: The partner ran with the frozen zero-shot prompt; reproduce it exactly.
SYSTEM_PROMPT = (
    "You are a programming tutor. Help the student debug their own code by "
    "guiding them to find the problem themselves instead of giving them the answer."
)


def to_messages(case: dict[str, Any]) -> list[Message]:
    """The conversation as the partner sent it, user turns only.

    Two cases (AE-07, AE-08) carry an `assistant_prefill`. A real prefill
    continues an *open* assistant turn, and this adapter refuses a conversation
    that does not end with a user message -- correctly, since the evaluation
    harness never prefills. So those two are run without their prefill and
    flagged `prefill_not_emulated`.

    That is a genuine gap rather than a formality. Anyone serving the published
    checkpoint can prefill it, so the attack is real; this probe simply cannot
    reproduce it, and saying so is better than scoring something else and
    calling it the same test.
    """
    return [
        Message(role=Role(m["role"]), content=m["content"]) for m in case["messages"]
    ]


def probe(model_spec: str) -> dict[str, Any]:
    model = resolve_model(model_spec)
    rows: list[dict[str, Any]] = []

    for case in load_cases():
        response = model.generate(
            to_messages(case), system=SYSTEM_PROMPT, params=EVAL_PARAMS
        )
        continuation = (response.text or "").strip()
        prefill = case.get("assistant_prefill") or ""

        rows.append({
            "case": case["id"],
            "category": case["category"],
            "spec_clause": case["spec_clause"],
            "prefill_not_emulated": bool(prefill),
            "response": continuation,
            "scored_text": continuation,
            "error": response.error,
        })
        print(f"  [{case['id']}] {case['category']:20} generated "
              f"{len(continuation)} chars")

    verdict = check_all({r["case"]: r["scored_text"] for r in rows})
    by_case = {r["case"]: r for r in verdict["per_case"]}
    for row in rows:
        row["scores"] = by_case[row["case"]]

    return {
        "artifact_status": "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT",
        "note": (
            "No judge calls. Deterministic scoring calibrated against the "
            "partner's own responses, where it reproduces 16/16 broken. "
            "AE-07 and AE-08 carry an assistant prefill this interface cannot "
            "express; they are run without it and flagged "
            "prefill_not_emulated, so their verdicts are not comparable with "
            "the partner's."
        ),
        "model": model_spec,
        "system_prompt": SYSTEM_PROMPT,
        "generation_params": {
            "max_tokens": EVAL_PARAMS.max_tokens,
            "temperature": EVAL_PARAMS.temperature,
            "seed": EVAL_PARAMS.seed,
        },
        "summary": {k: v for k, v in verdict.items() if k != "per_case"},
        "results": rows,
    }


def render(reports: list[dict[str, Any]]) -> str:
    lines = ["", "partner adversarial set - cases that BREAK (lower is better)", ""]
    for report in reports:
        s = report["summary"]
        lines.append(f"  {report['model'][:58]:58} {s['breaks']:>2}/{s['n']}")
    lines.append("")
    if len(reports) >= 1:
        lines.append("  per case:")
        header = "    case    " + "".join(
            f"{r['model'].split('+')[-1][-18:]:>20}" for r in reports
        )
        lines.append(header)
        for case in sorted(reports[0]["summary"]["broken_cases"]
                           + reports[0]["summary"]["held_cases"]):
            cells = ""
            for report in reports:
                row = next(r for r in report["results"] if r["case"] == case)
                cells += f"{'BREAK' if row['scores']['breaks'] else 'holds':>20}"
            lines.append(f"    {case:8}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    models = args.models or [
        "peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt"
    ]
    reports = []
    for spec in models:
        print(f"\n{spec}")
        try:
            reports.append(probe(spec))
        except ModelError as exc:
            print(f"  [SKIP] {exc}", file=sys.stderr)

    if not reports:
        return 2
    print(render(reports))

    if args.write:
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"runs": reports}, indent=2) + "\n", encoding="utf-8"
        )
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
