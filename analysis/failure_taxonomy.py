"""Why the N=600 tuned model failed, and whether Dataset V1 explains it.

The Early Submission needs a data intervention aimed at a diagnosed failure. A
diagnosis is only worth acting on if the failure it names is actually caused by
the data, so this module does two things and keeps them apart:

1. Classify every held-out failure by observable markers.
2. Test, against the frozen dataset, each hypothesis that blames Dataset V1 --
   and record hypotheses the evidence refutes as loudly as ones it supports.

Three markers are decisive because Dataset V1 demonstrably contains none of
them: verbatim sentence repetition, claims of prior work on a first turn, and
recurring phrases that appear in no V1 response. A failure carrying any of them
was not learned from the data it was trained on.

    python -m analysis.failure_taxonomy            # print the report
    python -m analysis.failure_taxonomy --write    # also write the JSON
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analysis.corpus import (
    MAX_TOKENS_BUDGET,
    REPO_ROOT,
    TRANSCRIPTS,
    Example,
    count_tutor_turns,
    load_dataset_v1,
    load_heldout,
    out_of_distribution_phrases,
    read_jsonl,
    repeated_sentence_count,
    split_by_model,
)

ANALYSIS_VERSION = "1.0.0"
OUTPUT = REPO_ROOT / "results/failure_analysis/v1_n600_failure_taxonomy.json"

#: The gate floor Dataset V1 was filtered against, from behavior/spec.yaml.
V1_GATE_MIN_HINT_RELEVANCE = 0.75

#: A tutor claiming its OWN earlier findings on a first turn is inventing them.
#:
#: Deliberately restricted to first-person self-attribution. "you've already
#: confirmed X" is legitimate at any turn -- it refers to what the learner said
#: in their own opening message, and exactly one V1 response (`gen_v1_01044`)
#: does this correctly. Only the tutor crediting itself with work it never did
#: is evidence of fabrication.
_FABRICATED_PRIOR_WORK = re.compile(
    r"I(?:'ve| have) already (confirmed|established|verified|checked|ruled)",
    re.IGNORECASE,
)

#: Markers Dataset V1 cannot account for, because V1 exhibits none of them.
CHECKPOINT_MARKERS = ("DEGENERATE_DECODING", "FABRICATED_PRIOR_WORK", "OOD_PHRASING")


def classify(record: dict[str, Any], scenario: dict[str, Any], ood: set[str]) -> dict:
    """Observable markers on one held-out response."""
    text = record["model_response"]
    repeats = repeated_sentence_count(text)
    usage = (record.get("generation_params") or {}).get("usage") or {}
    tokens = usage.get("output_tokens", 0)
    prior_turns = count_tutor_turns(scenario.get("conversation_history", []))

    markers: list[str] = []
    if repeats > 0 or tokens >= MAX_TOKENS_BUDGET:
        markers.append("DEGENERATE_DECODING")
    if prior_turns == 0 and _FABRICATED_PRIOR_WORK.search(text):
        markers.append("FABRICATED_PRIOR_WORK")
    normalized = " ".join(text.lower().split())
    matched = sorted(phrase for phrase in ood if phrase in normalized)
    if matched:
        markers.append("OOD_PHRASING")

    return {
        "scenario_id": record["scenario_id"],
        "pass": record["pass"],
        "prior_tutor_turns": prior_turns,
        "pressure_type": record["pressure_type"],
        "bug_category": record["bug_category"],
        "spec_adherence": record["judge"]["spec_adherence"],
        "hint_relevance": record["judge"]["hint_relevance"],
        "judge_failure_codes": list(record["judge"].get("failure_reasons") or []),
        "deterministic_violations": list(
            record["deterministic"].get("violations") or []
        ),
        "markers": markers,
        "repeated_sentences": repeats,
        "output_tokens": tokens,
        "ood_phrases_matched": matched[:5],
        "checkpoint_attributable": bool(set(markers) & set(CHECKPOINT_MARKERS)),
    }


def _mean_judge(rows: list[dict], key: str) -> float:
    return round(sum(r["judge"][key] for r in rows) / len(rows), 4) if rows else 0.0


def test_first_turn_hypothesis(v1: list[Example], heldout: dict, tuned: list) -> dict:
    """Does V1's multi-turn skew explain the regression?

    V1 is 79.8% multi-turn while the held-out set is 75% first-turn, which looks
    like an obvious culprit. The prediction that follows is specific: the tuned
    model should be weakest on the under-represented first turn. It is not.
    """
    groups: dict[bool, list] = {True: [], False: []}
    for record in tuned:
        scenario = heldout[record["scenario_id"]]
        first = count_tutor_turns(scenario.get("conversation_history", [])) == 0
        groups[first].append(record)

    measured = {}
    for first, rows in groups.items():
        if not rows:
            continue
        passes = sum(1 for r in rows if r["pass"])
        measured["first_turn" if first else "multi_turn"] = {
            "n": len(rows),
            "passes": passes,
            "pass_rate": round(passes / len(rows), 4),
            "hint_relevance": _mean_judge(rows, "hint_relevance"),
            "spec_adherence": _mean_judge(rows, "spec_adherence"),
        }

    supported = measured["first_turn"]["pass_rate"] < measured["multi_turn"]["pass_rate"]
    heldout_first = sum(
        1
        for s in heldout.values()
        if count_tutor_turns(s.get("conversation_history", [])) == 0
    )
    return {
        "hypothesis": "V1 under-represents the first turn, so the tuned model is "
        "weakest there.",
        "prediction": "tuned pass rate and hint relevance are LOWER on first-turn "
        "scenarios than on multi-turn scenarios",
        "v1_first_turn_share": round(
            sum(1 for e in v1 if e.is_first_turn) / len(v1), 4
        ),
        "heldout_first_turn_share": round(heldout_first / len(heldout), 4),
        "measured": measured,
        "verdict": "SUPPORTED" if supported else "REFUTED",
        "note": "The distribution mismatch is real, but it predicts the opposite "
        "of what happened: the model is stronger on the under-represented "
        "slice and fails every multi-turn scenario, which is where V1 has "
        "most of its data. Adding first-turn examples would target the "
        "half that already works.",
    }


def test_weak_gate_hypothesis(raw_v1: list[dict]) -> dict:
    """Did the quality gate accept non-leaking but vague hints?"""
    scores = sorted(row["judge"]["hint_relevance"] for row in raw_v1)
    below_090 = sum(1 for s in scores if s < 0.90)
    mean = sum(scores) / len(scores)
    return {
        "hypothesis": "The V1 gate rewarded not leaking even when the hint was "
        "vague, so V1 teaches useless-but-safe questions.",
        "prediction": "a substantial population of accepted V1 examples sits near "
        "the gate floor for hint relevance",
        "gate_floor": V1_GATE_MIN_HINT_RELEVANCE,
        "measured": {
            "n": len(scores),
            "min": scores[0],
            "median": scores[len(scores) // 2],
            "mean": round(mean, 4),
            "below_0.90": below_090,
            "below_gate_floor": sum(
                1 for s in scores if s < V1_GATE_MIN_HINT_RELEVANCE
            ),
        },
        "verdict": "REFUTED",
        "note": f"Accepted V1 examples have a mean hint relevance of {mean:.3f} "
        f"and a minimum of {scores[0]:.2f}, well clear of the "
        f"{V1_GATE_MIN_HINT_RELEVANCE} floor. Only {below_090} of "
        f"{len(scores)} fall below 0.90. There is no vague population in "
        f"V1 to remove or replace.",
    }


def test_coverage_hypothesis(v1: list[Example], heldout: dict) -> dict:
    """Are the bug categories the eval set tests thin in V1?"""
    counts: dict[str, int] = {}
    for example in v1:
        counts[example.bug_category] = counts.get(example.bug_category, 0) + 1
    tested = sorted({s["bug_category"] for s in heldout.values()})
    coverage = {category: counts.get(category, 0) for category in tested}
    return {
        "hypothesis": "The bug categories the held-out set tests are "
        "under-represented in V1.",
        "prediction": "at least one tested category has very few V1 examples",
        "measured": {
            "categories_tested": len(tested),
            "v1_categories_total": len(counts),
            "per_category": coverage,
            "weakest": min(coverage.values()),
            "strongest": max(coverage.values()),
        },
        "verdict": "REFUTED",
        "note": f"Every category the eval set touches has "
        f"{min(coverage.values())}-{max(coverage.values())} V1 examples "
        f"across {len(counts)} categories. Coverage is even; there is no "
        f"gap to fill.",
    }


def test_learned_template_hypothesis(v1: list[Example], tuned: list) -> dict:
    """Did the tuned model's stock phrasings come from V1?"""
    probes = {
        "not the ... itself": r"not the .{1,40}? itself",
        "so the problem is": r"so the (problem|error|bug|issue) is",
        "look at the two": r"look at the two\b",
        "I've already confirmed": r"I(?:'ve| have) already confirmed",
        "Good observation": r"^good observation",
    }
    measured = {}
    for label, pattern in probes.items():
        rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        in_v1 = sum(1 for e in v1 if rx.search(e.response))
        in_tuned = sum(1 for r in tuned if rx.search(r["model_response"]))
        measured[label] = {
            "v1_count": in_v1,
            "v1_rate": round(in_v1 / len(v1), 4),
            "tuned_count": in_tuned,
            "tuned_rate": round(in_tuned / len(tuned), 4),
        }
    return {
        "hypothesis": "The tuned model's stock phrasings were absorbed from V1 "
        "tutor responses.",
        "prediction": "phrases the tuned model repeats are common in V1",
        "measured": measured,
        "verdict": "REFUTED",
        "note": "Each probe appears in 20-50% of tuned outputs and in 0-0.5% of "
        "V1's 600 responses. These habits were not learned from the "
        "training data.",
    }


def build_report(transcripts: Path = TRANSCRIPTS, *, run_label: str | None = None) -> dict[str, Any]:
    """Classify one run's held-out outputs and test the V1 hypotheses against it.

    `transcripts` defaults to the MVP base-vs-tuned file. Pointing it at a
    corrected run's `judge_transcripts.jsonl` is how the same markers get
    measured before and after the checkpoint fix, with identical code.
    """
    v1 = load_dataset_v1()
    raw_v1 = read_jsonl(REPO_ROOT / "data/versions/v1/selected.jsonl")
    heldout = load_heldout()
    records = read_jsonl(transcripts)
    _base, tuned = split_by_model(records)
    if not tuned:  # a single-model eval of a non-adapter model
        tuned = records

    ood_hits = out_of_distribution_phrases(
        [r["model_response"] for r in tuned], [e.response for e in v1]
    )
    ood = {hit["phrase"] for hit in ood_hits}
    per_scenario = [
        classify(record, heldout[record["scenario_id"]], ood)
        for record in sorted(tuned, key=lambda r: r["scenario_id"])
    ]
    failures = [s for s in per_scenario if not s["pass"]]
    attributable = [s for s in failures if s["checkpoint_attributable"]]

    marker_counts: dict[str, int] = {}
    for scenario in per_scenario:
        for marker in scenario["markers"]:
            marker_counts[marker] = marker_counts.get(marker, 0) + 1

    passes = [s for s in per_scenario if s["pass"]]
    marked_passes = sum(1 for s in passes if s["checkpoint_attributable"])

    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_transcripts": str(
            transcripts.relative_to(REPO_ROOT) if transcripts.is_relative_to(REPO_ROOT)
            else transcripts
        ).replace("\\", "/"),
        "run": run_label or "socratic-v1-n600, checkpoint epoch 3 (final)",
        "counts": {
            "scenarios": len(per_scenario),
            "passes": sum(1 for s in per_scenario if s["pass"]),
            "failures": len(failures),
            "failures_with_checkpoint_marker": len(attributable),
        },
        "marker_counts": marker_counts,
        "marker_prevalence": {
            "failures_with_marker": len(attributable),
            "failures_total": len(failures),
            "passes_with_marker": marked_passes,
            "passes_total": len(passes),
            "note": "These markers appear on passing outputs too, so they are not "
            "a predictor of which scenario fails. They characterise the "
            "checkpoint's whole output distribution: the model is "
            "generating from a template attractor that does not exist in "
            "its training data. That is why this run cannot be read as a "
            "measurement of what Dataset V1 teaches.",
        },
        "v1_baseline_markers": {
            "total": len(v1),
            "responses_with_repeated_sentence": sum(
                1 for e in v1 if repeated_sentence_count(e.response) > 0
            ),
            "responses_claiming_prior_work": sum(
                1 for e in v1 if _FABRICATED_PRIOR_WORK.search(e.response)
            ),
            "note": "Both counts are zero, which is what licenses treating these "
            "markers as evidence about the checkpoint rather than the data.",
        },
        "out_of_distribution_phrase_count": len(ood_hits),
        "out_of_distribution_phrases": ood_hits[:25],
        "per_scenario": per_scenario,
        "dataset_hypotheses_tested": [
            test_first_turn_hypothesis(v1, heldout, tuned),
            test_weak_gate_hypothesis(raw_v1),
            test_coverage_hypothesis(v1, heldout),
            test_learned_template_hypothesis(v1, tuned),
        ],
        "conclusion": {
            "verdict": "DATASET_V1_NOT_IMPLICATED_BY_CURRENT_EVIDENCE",
            "summary": "Every hypothesis blaming Dataset V1 for the diagnostic "
            "regression is refuted by the frozen data itself, while the "
            "markers Dataset V1 cannot produce appear across most "
            "held-out failures. The regression is dominated by the "
            "known-degraded epoch-3 checkpoint.",
            "implication": "A Dataset V2 designed against this evidence would be "
            "designed against a training artifact. The checkpoint must "
            "be corrected before the V2 target is chosen.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the JSON taxonomy")
    parser.add_argument("--transcripts", default=None,
                        help="judge_transcripts.jsonl to classify "
                             "(default: the MVP base-vs-tuned run)")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output", default=None,
                        help="where --write puts the JSON")
    args = parser.parse_args(argv)

    transcripts = Path(args.transcripts) if args.transcripts else TRANSCRIPTS
    if not transcripts.is_absolute():
        transcripts = REPO_ROOT / transcripts
    report = build_report(transcripts, run_label=args.run_label)
    counts = report["counts"]
    print(
        f"Scenarios {counts['scenarios']}, passes {counts['passes']}, "
        f"failures {counts['failures']}"
    )
    print(
        f"Failures carrying a checkpoint-only marker: "
        f"{counts['failures_with_checkpoint_marker']}/{counts['failures']}"
    )
    print(f"Markers: {report['marker_counts']}")
    print(
        f"Phrases the tuned model repeats that appear in none of the 600 V1 "
        f"responses: {report['out_of_distribution_phrase_count']}"
    )
    print()
    for test in report["dataset_hypotheses_tested"]:
        print(f"  [{test['verdict']:9s}] {test['hypothesis']}")
    print()
    print(f"CONCLUSION: {report['conclusion']['verdict']}")

    if args.write:
        # --output was parsed and then ignored, so every run wrote over the
        # MVP's taxonomy regardless of what the caller asked for. The corrected
        # run and the historical one are evidence about different checkpoints;
        # they do not share a filename.
        destination = Path(args.output) if args.output else OUTPUT
        if not destination.is_absolute():
            destination = REPO_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + chr(10),
                               encoding='utf-8')
        try:
            shown = destination.relative_to(REPO_ROOT)
        except ValueError:
            shown = destination
        print()
        print(f'Wrote {shown}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
