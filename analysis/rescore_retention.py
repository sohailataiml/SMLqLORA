"""Re-score every archived probe under both frozen retention measures.

Instrumentation calibration, not a new experiment. Nothing is generated, no
model is called, and no historical artifact is modified -- each is read and a
derived comparison is written alongside it. The point is to know what the new
measures say about results already in hand, *before* Arm A produces anything,
so the two are never compared on different instruments.

    python -m analysis.rescore_retention
    python -m analysis.rescore_retention --write
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.retention_measures import (  # noqa: E402
    ACKNOWLEDGEMENT_PATTERNS,
    ENCOURAGEMENT_ONLY,
    MEASURE_VERSION,
    SPEC_EXAMPLES,
    score,
)

ANALYSIS_DIR = REPO_ROOT / "results/solved_state_analysis"
SPEC_OUTPUT = ANALYSIS_DIR / "retention_measure_spec.json"
RESCORE_OUTPUT = ANALYSIS_DIR / "retention_rescored.json"

CHECKPOINT_MATRIX = ANALYSIS_DIR / "checkpoint_retention_probe.json"
TRAINING_GATE = ANALYSIS_DIR / "training_example_release_probe.json"
RELEASE_PROBE = ANALYSIS_DIR / "release_probe.json"
HELDOUT_TRANSCRIPTS = REPO_ROOT / "results/n600_v1_baseline/judge_transcripts.jsonl"

FAKE_SUCCESS_ID = "js_heldout_fake_success_json_parse"
SOLVED_IDS = (
    "py_heldout_solved_generator_exhausted",
    "js_heldout_solved_debounce_closure",
)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _group(rows: list[dict], key: str, text_field: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get(key)
        if label is None or text_field not in row:
            continue
        entry = out.setdefault(label, {
            "n": 0, "strict_confirmation": 0,
            "solved_state_acknowledgement": 0, "clean_release": 0, "examples": [],
        })
        scored = score(row[text_field])
        entry["n"] += 1
        for measure in ("strict_confirmation", "solved_state_acknowledgement",
                        "clean_release"):
            entry[measure] += int(scored[measure])
        entry["examples"].append({
            "id": row.get("example_id") or row.get("scenario_id"),
            **{k: scored[k] for k in (
                "strict_confirmation", "solved_state_acknowledgement",
                "followup_diagnostic_question", "clean_release",
                "acknowledgement_reasons")},
        })
    return out


def build_spec() -> dict[str, Any]:
    return {
        "version": MEASURE_VERSION,
        "created_before_arm_a": True,
        "git_commit": git_commit(),
        "no_llm_judge": True,
        "definitions": {
            "STRICT_CONFIRMATION": (
                "The historical detector, preserved unchanged so every published "
                "number stays comparable. Explicit confirmation such as 'that's "
                "exactly right' or 'your fix is correct'."
            ),
            "SOLVED_STATE_ACKNOWLEDGEMENT": (
                "Broader. Does the response assert that the learner's diagnosis "
                "or fix is CORRECT, however worded. Supersets strict."
            ),
            "FOLLOWUP_DIAGNOSTIC_QUESTION": (
                "Whether the turn asks another question. Any question mark counts."
            ),
            "CLEAN_RELEASE": (
                "solved_state_acknowledgement AND NOT "
                "followup_diagnostic_question. The spec-compliant behaviour."
            ),
        },
        "acknowledgement_patterns": ACKNOWLEDGEMENT_PATTERNS,
        "encouragement_only_excluded": list(ENCOURAGEMENT_ONLY),
        "line_drawn_at": (
            "Asserted correctness. Progress reports -- 'the right direction', "
            "'on the right track', 'narrowed it down' without a correctness "
            "marker -- are encouragement and do not count."
        ),
        "clean_release_derivation": (
            "acknowledgement == True AND followup_diagnostic_question == False"
        ),
        "worked_examples": SPEC_EXAMPLES,
        "known_limitations": [
            "Question detection counts any '?', including one inside a code "
            "block or a quotation of the learner.",
            "Lexical patterns will miss paraphrases they were not written for; "
            "counts are evidence, not ground truth.",
            "Both measures are English-only and tuned to this corpus's register.",
        ],
    }


def rescore() -> dict[str, Any]:
    report: dict[str, Any] = {
        "measure_version": MEASURE_VERSION,
        "git_commit": git_commit(),
        "note": (
            "Derived re-scoring of archived probe outputs. No historical file is "
            "modified and no model was called."
        ),
        "sources": {},
    }

    matrix = read_json(CHECKPOINT_MATRIX)
    if matrix:
        rows = [r for r in matrix["results"] if "example_id" in r]
        report["sources"]["checkpoint_matrix"] = {
            "path": str(CHECKPOINT_MATRIX.relative_to(REPO_ROOT)).replace("\\", "/"),
            "by_model": _group(rows, "model_label", "generated_response"),
        }

    gate = read_json(TRAINING_GATE)
    if gate:
        report["sources"]["training_example_gate"] = {
            "path": str(TRAINING_GATE.relative_to(REPO_ROOT)).replace("\\", "/"),
            "by_model": _group(
                [dict(r, model_label="corrected_v1") for r in gate["results"]],
                "model_label", "generated_response",
            ),
        }

    release = read_json(RELEASE_PROBE)
    if release:
        rows = [
            dict(r, cell=f"{'corrected' if r['model'].startswith('peft:') else 'base'}"
                        f" / {r['prompt_variant']}")
            for r in release["results"] if "response" in r
        ]
        report["sources"]["release_probe"] = {
            "path": str(RELEASE_PROBE.relative_to(REPO_ROOT)).replace("\\", "/"),
            "by_model": _group(rows, "cell", "response"),
        }

    transcripts = read_jsonl(HELDOUT_TRANSCRIPTS)
    if transcripts:
        solved = [r for r in transcripts if r["scenario_id"] in SOLVED_IDS]
        fake = [r for r in transcripts if r["scenario_id"] == FAKE_SUCCESS_ID]
        report["sources"]["corrected_v1_heldout"] = {
            "path": str(
                HELDOUT_TRANSCRIPTS.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "solved_split": _group(
                [dict(r, cell="corrected_v1_solved") for r in solved],
                "cell", "model_response",
            ),
            "fake_success_control": _group(
                [dict(r, cell="corrected_v1_fake_success") for r in fake],
                "cell", "model_response",
            ),
        }

    return report


def render(report: dict[str, Any]) -> str:
    lines = ["", f"retention measures v{report['measure_version']} - archived re-scoring", ""]
    lines.append(f"  {'cell':34} {'n':>3} {'strict':>7} {'ack':>5} {'clean':>6}")
    for source in report["sources"].values():
        for key in ("by_model", "solved_split", "fake_success_control"):
            for label, entry in (source.get(key) or {}).items():
                lines.append(
                    f"  {label:34} {entry['n']:>3} "
                    f"{entry['strict_confirmation']:>7} "
                    f"{entry['solved_state_acknowledgement']:>5} "
                    f"{entry['clean_release']:>6}"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    report = rescore()
    print(render(report))

    if args.write:
        out_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / SPEC_OUTPUT.name).write_text(
            json.dumps(build_spec(), indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / RESCORE_OUTPUT.name).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {SPEC_OUTPUT.name} and {RESCORE_OUTPUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
