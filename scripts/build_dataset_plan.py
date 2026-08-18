"""Emit the Dataset V1 generation plan — derived, never hand-typed.

Step 8 of the experiment brief asks for a machine-readable plan that makes this
sentence checkable rather than aspirational:

    "We measured where prompting failed, then deliberately generated training
     examples targeting those failures."

So every distribution written here is *read back* from artifacts produced by the
completed prompt-ceiling ablation and from the generator's own sampler, rather
than being asserted in prose:

* pressure-type shares come from `proposed_training_distribution.json`
* the realized language / bug-category / difficulty / turn mix comes from
  `sample_plan()` itself, so the plan cannot drift from what generation will do
* target failure modes come from the residual that survived strong prompting

Makes zero API calls and generates no candidates.

    python scripts/build_dataset_plan.py
    python scripts/build_dataset_plan.py --target-accepted 600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior.spec import load_spec  # noqa: E402
from filtering.dedupe import DEFAULT_THRESHOLD as DEDUPE_THRESHOLD  # noqa: E402
from generation.prompts import (  # noqa: E402
    GENERATION_PROMPT_VERSION,
    PRESSURE_WEIGHTS,
    generation_prompt_hash,
    plan_summary,
    sample_plan,
)

#: The only acceptance rate this project has ever *measured* is 33%, from the
#: 200-candidate mock dry run. That teacher is deliberately repetitive, so most
#: of its rejections were duplicates — a real teacher should dedupe far less and
#: leak somewhat more. Rather than pick a number and hope, the plan sizes a
#: first tranche, measures the real rate, and computes the top-up from it.
MEASURED_MOCK_ACCEPTANCE = 0.33
OPTIMISTIC_ACCEPTANCE = 0.45

#: Nested sweep checkpoints. Fixed here so the dataset is sized to serve them.
SWEEP_SIZES = (125, 250, 500, 600)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _shares(counter: dict[str, int]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 4)
            for k, v in sorted(counter.items(), key=lambda kv: -kv[1])}


def build_plan(target_accepted: int, tranche_1: int) -> dict:
    spec = load_spec()
    ceiling = json.loads(
        (REPO_ROOT / "results/prompt_ceiling/proposed_training_distribution.json")
        .read_text(encoding="utf-8")
    )
    failures = json.loads(
        (REPO_ROOT / "results/prompt_ceiling/failure_modes.json")
        .read_text(encoding="utf-8")
    )
    residual = failures["residual_under_strong_prompts"]

    # Realized mix, read out of the sampler that generation will actually use.
    realized = plan_summary(sample_plan(tranche_1))

    conservative = -(-target_accepted * 100 // int(MEASURED_MOCK_ACCEPTANCE * 100))
    optimistic = -(-target_accepted * 100 // int(OPTIMISTIC_ACCEPTANCE * 100))

    return {
        "dataset_version": "v1",
        "status": "PLAN ONLY — no candidate has been generated",
        "derived_from": {
            "experiment": "results/prompt_ceiling",
            "experiment_status": (
                "REAL_EXPERIMENT_RESULT, 216/216 evaluations, 6/6 cells complete"
            ),
            "gate_decision": "FINE-TUNING JUSTIFIED",
            "git_commit": git_commit(),
        },

        "target_accepted_count": target_accepted,

        "estimated_candidate_count": {
            "strategy": "two tranches — measure the real acceptance rate, then top up",
            "tranche_1_candidates": tranche_1,
            "expected_accepted_from_tranche_1": {
                "at_measured_mock_rate_0.33": int(tranche_1 * MEASURED_MOCK_ACCEPTANCE),
                "at_optimistic_rate_0.45": int(tranche_1 * OPTIMISTIC_ACCEPTANCE),
            },
            "single_pass_equivalent": {
                "conservative_at_0.33": conservative,
                "optimistic_at_0.45": optimistic,
            },
            "top_up_rule": (
                "after tranche 1, accepted_shortfall / observed_acceptance_rate, "
                "rounded up to the next 50. Generation is seeded and resumable, so "
                "a top-up extends the plan rather than regenerating it."
            ),
            "why_not_a_single_number": (
                "The only measured acceptance rate (0.33) came from a deliberately "
                "repetitive mock teacher whose rejections were dominated by dedupe. "
                "Committing ~1800 candidates up front on that number risks "
                "overspending by ~450 candidates if the real rate is nearer 0.45."
            ),
        },

        "distributions": {
            "pressure_type": {
                "target": ceiling["distribution"],
                "realized_in_plan": _shares(realized["pressure_type"]),
                "source": (
                    "measured failure rate under strong prompts, floored and capped"
                ),
            },
            "language": _shares(realized["language"]),
            "bug_category": _shares(realized["bug_category"]),
            "difficulty": _shares(realized["difficulty"]),
            "conversation_turns": _shares(realized["conversation_turns"]),
            "learner_competence": _shares(realized["learner_competence"]),
            "hint_strength": _shares(realized["hint_strength"]),
            "student_progress": _shares(realized["student_progress"]),
        },

        "allocation_rule": ceiling["rule"],

        "target_failure_modes": {
            "basis": (
                "failures that survived strong prompting in the completed "
                f"ablation ({residual['n_failed']} of {residual['n_measured']} "
                "strong-prompt evaluations failed)"
            ),
            "surviving_failure_modes": residual["surviving_failure_modes"],
            "primary_target": "SOLUTION_LEAK",
            "why_primary": (
                "Every one of the 4 failures in the strongest cell "
                "(openai:gpt-5 + structured_system_prompt) was a SOLUTION_LEAK, "
                "and they cluster under answer-seeking pressure "
                "(time_pressure, frustrated, repeated_answer_request). "
                "MULTIPLE_HINTS is more frequent overall but is largely "
                "prompt-fixable, so it is a secondary target."
            ),
            "secondary_targets": [
                "MULTIPLE_HINTS", "EXPLICIT_FINAL_DIAGNOSIS", "WITHHELD_AFTER_SOLVED",
            ],
            "hardest_pressure_types_under_strong_prompts": [
                {k: p[k] for k in ("pressure_type", "n", "pass_rate", "underpowered")}
                for p in residual["pressure_type_ranking"][:3]
            ],
        },

        "teacher_model": "anthropic:claude-opus-5",
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "generation_prompt_sha256": generation_prompt_hash(spec),
        "base_seed": 20240101,
        "pressure_weights_per_mille": dict(PRESSURE_WEIGHTS),

        "quality_gate": {
            "stages": [
                "schema", "static_checks", "llm_judge", "dedupe",
                "contamination", "balance",
            ],
            "judge_model": "anthropic:claude-opus-5",
            "min_judge_spec_adherence":
                spec.gates.quality_gate.min_judge_spec_adherence,
            "min_judge_hint_relevance":
                spec.gates.quality_gate.min_judge_hint_relevance,
            "min_judge_robustness":
                spec.gates.quality_gate.min_judge_robustness,
            "allow_any_deterministic_violation":
                spec.gates.quality_gate.allow_any_deterministic_violation,
            "dedupe_jaccard_threshold": DEDUPE_THRESHOLD,
            "rejected_examples_retained": "data/rejected/v1.jsonl",
        },

        "data_efficiency_sweep": {
            "sizes": list(SWEEP_SIZES),
            "nested": True,
            "nesting_guarantee": (
                "training.dataset.nested_subsets() sorts accepted examples by "
                "content_hash, applies a seeded shuffle (seed=13), then takes "
                "prefixes. Prefixes of one fixed ordering are nested by "
                "construction: 125 subset-of 250 subset-of 500 subset-of 600. "
                "Because the ordering is a uniform shuffle of the whole accepted "
                "set, each prefix is a simple random sample and reproduces the "
                "dataset distribution up to sampling error — it is not "
                "stratified, so small-N shares will wobble."
            ),
            "ordering_is_generation_order_independent": (
                "sorting on content_hash before shuffling means concurrency, "
                "retries and top-up tranches cannot change which examples land "
                "in the N=125 subset."
            ),
            "command": (
                "python -m ablations.data_efficiency --train --sizes 125 250 500 600"
            ),
            "status": "NOT RUN — awaiting authorization",
        },

        "not_yet_done": [
            "no candidates generated", "no quality gate run",
            "no dataset built", "no training run", "no evaluation run",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accepted", type=int, default=600)
    parser.add_argument("--tranche-1", type=int, default=1200)
    parser.add_argument("--output", default="data/versions/v1/plan.json")
    args = parser.parse_args(argv)

    plan = build_plan(args.target_accepted, args.tranche_1)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    est = plan["estimated_candidate_count"]
    print(f"Dataset {plan['dataset_version']} plan -> {args.output}")
    print(f"  target accepted     : {plan['target_accepted_count']}")
    print(f"  tranche 1 candidates: {est['tranche_1_candidates']}")
    print(f"  single-pass equiv   : "
          f"{est['single_pass_equivalent']['conservative_at_0.33']} (at 0.33) .. "
          f"{est['single_pass_equivalent']['optimistic_at_0.45']} (at 0.45)")
    print(f"  primary target      : {plan['target_failure_modes']['primary_target']}")
    print(f"  sweep               : {plan['data_efficiency_sweep']['sizes']} (nested)")
    print("\nNo candidates generated. `make generate-data` is a separate, paid step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
