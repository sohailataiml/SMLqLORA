#!/usr/bin/env python
"""Is the missing confirmation behaviour absent, or merely unelicited?

The corrected N=600 baseline confirms in 0 of 20 held-out outputs while its
solved-state training targets confirm in 82 of 85. That measurement is what
made `V2_NOT_JUSTIFIED` the verdict, and it leaves one question open: the model
was *shown* the behaviour, so does it *have* it?

Two prompts and two models separate the possibilities:

* the adapter confirms once the release rule is stated -> the behaviour is
  latent and the deficit is elicitation. No dataset change would fix it.
* only the base model confirms -> fine-tuning suppressed a capability the base
  had. A training-dynamics finding, not a data gap.
* nothing confirms under any prompt -> a capability floor at 1.7B.

Only the last leaves any room for a data intervention, and even then it argues
against one.

THIS IS A DIAGNOSTIC PROBE, NOT AN EXPERIMENT. It makes no judge calls, writes
nothing into `results/n600_v1_baseline/`, and produces no number that is
comparable with the baseline or claimable as a result. It answers one yes/no
question about elicitation.

    python scripts/probe_release_behavior.py \
        --model 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt' \
        --model hf:Qwen/Qwen3-1.7B
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
from evaluation.schemas import load_scenarios  # noqa: E402
from models.adapters import (  # noqa: E402
    EVAL_PARAMS,
    ModelError,
    resolve_model,
)
from prompting.strategies import get_strategy, render_conversation  # noqa: E402

HELDOUT = REPO_ROOT / "scenarios/heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results/solved_state_analysis/release_probe.json"

#: Taken from `behavior/spec.yaml`'s `withheld_after_solved` rule rather than
#: invented here, so the probe tests the spec's own requirement.
RELEASE_RULE = (
    " If the student has already produced the correct fix, confirm plainly that "
    "it is correct instead of asking another question. You may then explain why "
    "it works."
)


def prompt_variants() -> dict[str, str]:
    """The frozen zero-shot prompt, and the same prompt plus the release rule."""
    base = get_strategy("zero_shot").system_prompt()
    return {"zero_shot": base, "zero_shot_plus_release_rule": base + RELEASE_RULE}


def solved_scenarios() -> list[Any]:
    return [s for s in load_scenarios(HELDOUT) if s.student_has_solved]


def probe(model_specs: list[str]) -> dict[str, Any]:
    scenarios = solved_scenarios()
    variants = prompt_variants()
    results: list[dict[str, Any]] = []

    for spec in model_specs:
        try:
            model = resolve_model(spec)
        except ModelError as exc:
            print(f"\n[SKIP] {spec}: {exc}\n", file=sys.stderr)
            results.append({"model": spec, "error": str(exc)})
            continue

        for variant, system in variants.items():
            for scenario in scenarios:
                response = model.generate(
                    render_conversation(scenario),
                    system=system,
                    params=EVAL_PARAMS,
                )
                behaviour = tutor_profile(response.text or "")
                row = {
                    "model": spec,
                    "prompt_variant": variant,
                    "scenario_id": scenario.id,
                    "error": response.error,
                    "response": (response.text or "").strip(),
                    "confirms": behaviour["confirms"],
                    "asks_question": behaviour["asks_question"],
                    "explains": behaviour["explains"],
                }
                results.append(row)
                mark = "CONFIRMS" if row["confirms"] else "no-confirm"
                print(f"[{mark:10}] {spec.split('+')[-1][:28]:28} "
                      f"{variant:28} {scenario.id}")
                print(f"    {row['response'][:220]}")

    return {
        "artifact_status": "DIAGNOSTIC_PROBE_NOT_AN_EXPERIMENT",
        "note": (
            "No judge calls. Not comparable with N600_V1_BASELINE and not "
            "claimable as a result. Answers only whether confirmation behaviour "
            "can be elicited."
        ),
        "release_rule_appended": RELEASE_RULE.strip(),
        "generation_params": {
            "max_tokens": EVAL_PARAMS.max_tokens,
            "temperature": EVAL_PARAMS.temperature,
            "seed": EVAL_PARAMS.seed,
        },
        "results": results,
    }


def summarise(report: dict[str, Any]) -> str:
    rows = [r for r in report["results"] if "response" in r]
    lines = ["", "confirmation counts (out of 2 solved scenarios):"]
    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["model"], row["prompt_variant"])
        seen[key] = seen.get(key, 0) + int(bool(row["confirms"]))
    for (model, variant), count in sorted(seen.items()):
        lines.append(f"  {count}/2  {model}  [{variant}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", dest="models", default=None,
        help="repeatable; e.g. peft:Qwen/Qwen3-1.7B+outputs/... or hf:Qwen/Qwen3-1.7B",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    models = args.models or [
        "peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600-bestckpt",
        "hf:Qwen/Qwen3-1.7B",
    ]
    report = probe(models)
    print(summarise(report))

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
