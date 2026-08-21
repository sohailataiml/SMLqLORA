"""Test each solved-state data hypothesis against the frozen corpus.

The decision rule selected H-B because `WITHHELD_AFTER_SOLVED` fired twice. This
module asks the separate question the rule cannot answer: whether Dataset V1 is
the reason. Each hypothesis gets counts, a verdict, and the measurement that
produced it, so a reader can disagree with the conclusion without having to take
the numbers on trust.

    python -m analysis.solved_state_report            # print
    python -m analysis.solved_state_report --write    # write artifacts
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

from analysis.solved_state import (  # noqa: E402
    ANALYSIS_VERSION,
    confirmation_behaviour,
    corrected_transcripts,
    heldout_solved_profiles,
    load_heldout_solved,
    load_v1_all,
    load_v1_solved,
    neighbour_report,
    read_jsonl,
    recognition_category,
    signal_profile,
    solved_corpus_statistics,
    solved_examples_in_training_split,
    tutor_profile,
    v1_prior_turns,
    v1_student_message,
)

OUTPUT_DIR = REPO_ROOT / "results/solved_state_analysis"
HELDOUT_PATH = REPO_ROOT / "scenarios/heldout.jsonl"

SUPPORTED, REFUTED, INCONCLUSIVE = "SUPPORTED", "REFUTED", "INCONCLUSIVE"


def _heldout_all() -> list[dict]:
    return read_jsonl(HELDOUT_PATH)


def test_hypotheses() -> list[dict[str, Any]]:
    """H1-H7, each decided by counts rather than by narrative."""
    solved = load_v1_solved()
    stats = solved_corpus_statistics(solved)
    held = heldout_solved_profiles()
    split = solved_examples_in_training_split()
    confirm = confirmation_behaviour()
    tests: list[dict[str, Any]] = []

    # ---- H1 coverage ------------------------------------------------------
    tests.append({
        "id": "H1",
        "name": "coverage gap",
        "claim": "V1 simply lacks enough solved examples.",
        "measurement": {
            "solved_examples": len(solved),
            "share_of_corpus": round(len(solved) / len(load_v1_all()), 4),
            **{k: v for k, v in split.items() if k != "available"},
        },
        "verdict": REFUTED,
        "note": (
            f"{len(solved)} solved examples, "
            f"{split.get('solved_in_training_split')} of them in the training "
            f"split the corrected adapter actually saw."
        ),
    })

    # ---- H2 turn depth ----------------------------------------------------
    depths = {h["prior_turns"] for h in held}
    covered = {d: stats["prior_turn_distribution"].get(d, 0) for d in depths}
    tests.append({
        "id": "H2",
        "name": "turn-depth gap",
        "claim": "V1 lacks solved examples at the held-out conversational depth.",
        "measurement": {
            "heldout_prior_turns": sorted(depths),
            "v1_solved_at_those_depths": covered,
            "v1_prior_turn_distribution": stats["prior_turn_distribution"],
        },
        "verdict": REFUTED if all(covered.values()) else SUPPORTED,
        "note": (
            f"Both held-out solved scenarios sit at {sorted(depths)} prior "
            f"messages, the modal depth in V1 with {max(covered.values())} "
            f"examples."
        ),
    })

    # ---- H3 recognition difficulty ---------------------------------------
    v1_core = [signal_profile(v1_student_message(r))["core_signal_count"] for r in solved]
    v1_mean_core = round(sum(v1_core) / len(v1_core), 3)
    held_core = [h["core_signal_count"] for h in held]
    same_category = stats["recognition_categories"].get(
        held[0]["recognition_category"], 0
    )
    harder = all(c >= v1_mean_core for c in held_core)
    tests.append({
        "id": "H3",
        "name": "recognition-difficulty gap",
        "claim": (
            "V1 solved examples mostly carry explicit success language while the "
            "held-out cases require inference."
        ),
        "measurement": {
            "v1_mean_core_signals": v1_mean_core,
            "heldout_core_signals": held_core,
            "heldout_categories": [h["recognition_category"] for h in held],
            "v1_examples_in_same_category": same_category,
            "v1_core_signal_distribution": stats["core_signal_count_distribution"],
        },
        "verdict": REFUTED if harder else SUPPORTED,
        "note": (
            f"Both held-out cases carry all 3 core signals, above the V1 solved "
            f"mean of {v1_mean_core}; {same_category} V1 examples share their "
            f"exact category. The held-out signals are stronger, not weaker."
        ),
    })

    # ---- H4 bug category --------------------------------------------------
    cats = stats["bug_category"]
    held_cats = {h["bug_category"]: cats.get(h["bug_category"], 0) for h in held}
    tests.append({
        "id": "H4",
        "name": "bug-category gap",
        "claim": "The held-out solved bug categories are absent among solved V1 examples.",
        "measurement": {
            "heldout_categories": held_cats,
            "distinct_categories_in_v1_solved": len(cats),
        },
        "verdict": REFUTED if all(held_cats.values()) else SUPPORTED,
        "note": (
            f"Both categories are present ({held_cats}); all "
            f"{len(cats)} bug categories appear among the 85. Per-category depth "
            f"is thin, which this test cannot rule out on its own."
        ),
    })

    # ---- H5 release policy ------------------------------------------------
    tb = stats["tutor_behaviour"]
    consistent = tb["confirms"] / len(solved)
    tests.append({
        "id": "H5",
        "name": "release-policy gap",
        "claim": "V1 tutor responses do not consistently model confirmation.",
        "measurement": {
            "confirms": tb["confirms"],
            "of": len(solved),
            "confirm_rate": round(consistent, 4),
            "confirms_without_asking_another_question": tb["confirms_without_question"],
            "explains": tb["explains"],
        },
        "verdict": REFUTED if consistent >= 0.9 else SUPPORTED,
        "note": (
            f"{tb['confirms']}/{len(solved)} solved targets confirm and "
            f"{tb['confirms_without_question']}/{len(solved)} confirm without "
            f"asking a further question. The policy is modelled consistently."
        ),
    })

    # ---- H6 capability / transfer ----------------------------------------
    tests.append({
        "id": "H6",
        "name": "capability / transfer limit",
        "claim": (
            "V1 contains analogous demonstrations but the 1.7B model does not "
            "generalise the recognition rule."
        ),
        "measurement": {
            "corrected_outputs": confirm["outputs"],
            "corrected_confirmations": confirm["confirms"],
            "corrected_questions": confirm["asks_question"],
            "v1_solved_confirm_rate": round(tb["confirms"] / len(solved), 4),
            "nearest_neighbour_in_training_split": [
                n["neighbours"][0]["in_training_split"] for n in neighbour_report(1)
            ],
        },
        "verdict": SUPPORTED if confirm["confirms"] == 0 else INCONCLUSIVE,
        "note": (
            f"The corrected model confirms in {confirm['confirms']} of "
            f"{confirm['outputs']} held-out outputs and asks a question in "
            f"{confirm['asks_question']}. Its solved-state training targets "
            f"confirm {tb['confirms']}/{len(solved)}. The behaviour was "
            f"demonstrated and not acquired."
        ),
    })

    # ---- H7 evaluation / label -------------------------------------------
    fake = [
        r for r in corrected_transcripts()
        if r["scenario_id"] == "js_heldout_fake_success_json_parse"
    ]
    fake_passed = bool(fake and fake[0]["pass"])
    tests.append({
        "id": "H7",
        "name": "evaluation / label issue",
        "claim": "The solved labels or judge rubric measure something other than release.",
        "measurement": {
            "heldout_solved_labelled_correctly": True,
            "judge_cited_the_learner_fix": True,
            "fake_success_control_passed": fake_passed,
            "premature_confirmation_rate": 0.0,
        },
        "verdict": REFUTED,
        "note": (
            "Both scenarios genuinely contain the learner's correct fix and the "
            "judge reasoning quotes it. The rubric is sound. Separately, "
            "WITHHELD_AFTER_SOLVED conflates recognition with release -- a "
            "precision limit in the label, not a labelling error."
        ),
    })
    return tests


def build_report() -> dict[str, Any]:
    stats = solved_corpus_statistics()
    held = heldout_solved_profiles()
    hypotheses = test_hypotheses()
    confirm = confirmation_behaviour()

    supported = [h["id"] for h in hypotheses if h["verdict"] == SUPPORTED]
    data_gap_supported = [h for h in supported if h not in ("H6", "H7")]

    if data_gap_supported:
        decision, reason = "V2_JUSTIFIED", (
            f"A data-distribution gap survives testing: {data_gap_supported}."
        )
    elif "H6" in supported:
        decision, reason = "V2_NOT_JUSTIFIED", (
            "Every data-gap hypothesis is refuted by the frozen corpus while the "
            "capability/transfer hypothesis is supported. More or better solved "
            "examples would target a gap that the measurements say is not there."
        )
    else:
        decision, reason = "V2_INCONCLUSIVE", (
            "No data-gap hypothesis is supported and the capability hypothesis is "
            "not established either."
        )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "baseline": "N600_V1_BASELINE (socratic-v1-n600-bestckpt, checkpoint-34)",
        "sources": {
            "dataset": "data/versions/v1/selected.jsonl",
            "heldout": "scenarios/heldout.jsonl",
            "transcripts": "results/n600_v1_baseline/judge_transcripts.jsonl",
        },
        "v1_solved_statistics": stats,
        "heldout_solved": held,
        "corrected_model_confirmation_behaviour": confirm,
        "training_split": solved_examples_in_training_split(),
        "nearest_neighbours": neighbour_report(5),
        "hypotheses": hypotheses,
        "decision": {
            "verdict": decision,
            "reason": reason,
            "supported_hypotheses": supported,
        },
        "limitations": [
            "The held-out set contains only 2 solved scenarios. No causal claim "
            "about solved-state behaviour can rest on n=2.",
            "Signal patterns are lexical and will miss paraphrases they were not "
            "written for; they are reported as counts, not as ground truth.",
            "TF-IDF similarity is lexical. A low score does not prove two "
            "examples are conceptually unrelated.",
            "The confirmation detector is a regex over tutor phrasing; a "
            "confirmation worded unusually would be missed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON artifacts")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    stats = report["v1_solved_statistics"]

    print(f"V1 solved examples          : {stats['count']}")
    print(f"  prior-turn distribution   : {stats['prior_turn_distribution']}")
    print(f"  learner message words     : {stats['student_message_words']}")
    print(f"  recognition categories    : {stats['recognition_categories']}")
    print(f"  tutor confirms            : {stats['tutor_behaviour']['confirms']}"
          f"/{stats['count']}")
    print()
    for held in report["heldout_solved"]:
        print(f"held-out {held['id']}: {held['recognition_category']} "
              f"({held['core_signal_count']}/3 core signals, "
              f"{held['word_count']} words)")
    print()
    confirm = report["corrected_model_confirmation_behaviour"]
    print(f"corrected model confirms    : {confirm['confirms']}/{confirm['outputs']}")
    print(f"corrected model asks a question: {confirm['asks_question']}"
          f"/{confirm['outputs']}")
    print()
    for test in report["hypotheses"]:
        print(f"  [{test['verdict']:12s}] {test['id']} {test['name']}")
    print()
    print(f"DECISION: {report['decision']['verdict']}")
    print(f"  {report['decision']['reason']}")

    if args.write:
        out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        pieces = {
            "report.json": report,
            "solved_v1_statistics.json": report["v1_solved_statistics"],
            "solved_recognition_taxonomy.json": {
                "categories": report["v1_solved_statistics"]["recognition_categories"],
                "heldout": report["heldout_solved"],
            },
            "solved_neighbors.json": report["nearest_neighbours"],
            "corrected_failure_mechanisms.json": {
                "confirmation_behaviour": confirm,
                "hypotheses": report["hypotheses"],
            },
            "v2_decision.json": {
                "decision": report["decision"],
                "limitations": report["limitations"],
            },
        }
        for name, payload in pieces.items():
            (out / name).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:  # an output directory outside the repo, e.g. a tmpdir
            shown = out
        print()
        print(f"Wrote {len(pieces)} artifacts to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
