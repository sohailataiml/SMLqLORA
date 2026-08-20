"""A finished evaluation must not exit non-zero over a path it cannot phrase.

The corrected N=600 run passed `--output results/n600_v1_baseline`, a relative
path. Every artifact was written and all twenty judge calls were paid for, and
then the last line of main() crashed formatting the summary: `relative_to()`
cannot express a relative path against an absolute repo root.

Cosmetic in a notebook, where the next cell simply read the file back. Not
cosmetic to anything that checks an exit code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(output: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "eval.py", "--model", "mock:demo",
         "--eval-set", "scenarios/heldout.jsonl", "--offline-judge",
         "--output", output],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )


def test_a_relative_output_path_does_not_crash_a_finished_run(tmp_path):
    relative = Path("results") / "eval" / "_test_relative_output"
    proc = _run(str(relative))
    try:
        assert proc.returncode == 0, proc.stderr[-800:]
        assert "Traceback" not in proc.stderr
    finally:
        target = REPO_ROOT / relative
        for child in sorted(target.glob("*")):
            child.unlink()
        if target.exists():
            target.rmdir()


def test_a_relative_output_resolves_against_the_repo_not_the_cwd():
    """Same rule --eval-set already follows, so artifacts are findable."""
    relative = Path("results") / "eval" / "_test_relative_resolve"
    target = REPO_ROOT / relative
    proc = _run(str(relative))
    try:
        assert proc.returncode == 0, proc.stderr[-800:]
        assert (target / "results.json").exists()
        payload = json.loads((target / "results.json").read_text(encoding="utf-8"))
        assert payload["metrics"]["scenario_count"] == 20
        assert payload["result_status"] == "REAL_EXPERIMENT_RESULT"
    finally:
        for child in sorted(target.glob("*")):
            child.unlink()
        if target.exists():
            target.rmdir()


def test_the_summary_line_names_where_the_artifacts_went():
    relative = Path("results") / "eval" / "_test_relative_summary"
    target = REPO_ROOT / relative
    proc = _run(str(relative))
    try:
        assert proc.returncode == 0, proc.stderr[-800:]
        assert "artifacts" in proc.stdout
        assert "_test_relative_summary" in proc.stdout
    finally:
        for child in sorted(target.glob("*")):
            child.unlink()
        if target.exists():
            target.rmdir()
