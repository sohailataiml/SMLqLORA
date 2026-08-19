"""Run what a grader runs, and say whether it worked.

The submission document promises specific commands. This checks those promises
rather than trusting them: it extracts nothing, assumes nothing, and reports
each documented entry point as PASS or FAIL.

Two tiers, because a grader has a GPU and this repository's test machine may not:

* **offline** — everything that needs no GPU and no credentials: the model spec
  resolves, the eval set loads, the harness runs end to end on a mock model with
  the offline judge, and the pinned hashes in SUBMISSION.md match the artifacts.
* **live** — the real commands against the published Hugging Face checkpoint.
  Skipped unless --live is passed, because it downloads a model and needs a GPU.

    python scripts/grader_dry_run.py
    python scripts/grader_dry_run.py --live          # on a GPU box
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBMISSION = REPO_ROOT / "SUBMISSION.md"
HF_REPO = "sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600"

results: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    def decorate(fn: Callable) -> Callable:
        def run() -> None:
            try:
                detail = fn() or "ok"
                results.append((name, True, str(detail)))
            except Exception as exc:  # noqa: BLE001 — this IS the report
                results.append((name, False, f"{type(exc).__name__}: {exc}"))
        run.__name__ = fn.__name__
        return run
    return decorate


# ----------------------------------------------------------------- offline


@check("SUBMISSION.md exists and pins every hash")
def submission_pins() -> str:
    text = SUBMISSION.read_text(encoding="utf-8")
    required = {
        "model commit": r"16d60373[0-9a-f]*",
        "base revision": r"70d244cc[0-9a-f]*",
        "dataset hash": r"9121c24e[0-9a-f]*",
        "spec hash": r"dc14f40b[0-9a-f]*",
        "eval set hash": r"a30abe2a[0-9a-f]*",
        "hf repo": re.escape(HF_REPO),
    }
    missing = [k for k, pat in required.items() if not re.search(pat, text)]
    if missing:
        raise AssertionError(f"not pinned in SUBMISSION.md: {missing}")
    return f"all {len(required)} pins present"


@check("pinned dataset hash matches the frozen dataset")
def dataset_hash_matches() -> str:
    freeze = json.loads(
        (REPO_ROOT / "data/versions/v1/freeze.json").read_text(encoding="utf-8")
    )
    text = SUBMISSION.read_text(encoding="utf-8")
    if freeze["dataset_hash"] not in text:
        raise AssertionError("SUBMISSION.md cites a dataset hash that is not the freeze")
    return freeze["dataset_hash"][:16]


@check("pinned spec hash matches the live behavior spec")
def spec_hash_matches() -> str:
    from behavior.spec import load_spec

    live = load_spec().spec_sha256
    if live not in SUBMISSION.read_text(encoding="utf-8"):
        raise AssertionError(
            f"live spec hashes to {live[:16]}, which SUBMISSION.md does not cite"
        )
    return live[:16]


@check("the published model id resolves without a prefix")
def bare_repo_id_resolves() -> str:
    from models.adapters import resolve_model

    adapter = resolve_model(HF_REPO)
    if adapter.family != "local-hf":
        raise AssertionError(f"resolved to {adapter.family}, expected local-hf")
    return adapter.name


@check("the held-out eval set loads and hashes as documented")
def eval_set_loads() -> str:
    from evaluation.schemas import load_scenarios, scenarios_hash

    scenarios = load_scenarios(REPO_ROOT / "scenarios/heldout.jsonl")
    digest = scenarios_hash(scenarios)
    if digest not in SUBMISSION.read_text(encoding="utf-8"):
        raise AssertionError(f"eval set hashes to {digest[:16]}, not the documented value")
    return f"{len(scenarios)} scenarios, {digest[:16]}"


@check("eval.py runs end to end (mock model, offline judge)")
def eval_py_offline() -> str:
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", "mock:demo",
         "--eval-set", "scenarios/heldout.jsonl", "--offline-judge"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(f"exit {proc.returncode}: {proc.stderr[-400:]}")
    if "scenarios measured" not in proc.stdout:
        raise AssertionError("no results table in stdout")
    return "results table produced"


@check("eval.py explains itself on a bad model id")
def eval_py_bad_model() -> str:
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", "nonsense",
         "--eval-set", "scenarios/heldout.jsonl", "--offline-judge"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode == 0:
        raise AssertionError("a bad model id should not exit 0")
    blob = (proc.stdout + proc.stderr).lower()
    if "traceback" in blob:
        raise AssertionError("failed with a stack trace rather than an explanation")
    return "explained, no traceback"


@check("eval.py explains itself on a missing eval set")
def eval_py_bad_set() -> str:
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", "mock:demo",
         "--eval-set", "scenarios/does_not_exist.jsonl", "--offline-judge"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode == 0:
        raise AssertionError("a missing eval set should not exit 0")
    if "traceback" in (proc.stdout + proc.stderr).lower():
        raise AssertionError("failed with a stack trace rather than an explanation")
    return "explained, no traceback"


@check("an unseen scenario file works (staff held-out simulation)")
def unseen_eval_set() -> str:
    """The staff set is one this repository has never seen. Simulate that by
    running against `clean.jsonl`, which is not the documented held-out file."""
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", "mock:demo",
         "--eval-set", "scenarios/clean.jsonl", "--offline-judge", "--limit", "4"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(f"exit {proc.returncode}: {proc.stderr[-400:]}")
    return "harness is not tied to one eval file"


@check("raw judge transcripts are present and complete")
def transcripts_present() -> str:
    path = REPO_ROOT / "results/base_vs_tuned/judge_transcripts.jsonl"
    if not path.exists():
        raise AssertionError("judge_transcripts.jsonl is missing")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    reasoned = sum(1 for r in records if (r.get("judge") or {}).get("reasoning"))
    if reasoned != len(records):
        raise AssertionError(f"only {reasoned}/{len(records)} records carry judge reasoning")
    return f"{len(records)} records, all with reasoning"


@check("published metrics recompute from the raw transcripts")
def metrics_recompute() -> str:
    base = REPO_ROOT / "results/base_vs_tuned"
    records = [
        json.loads(l)
        for l in (base / "judge_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    published = json.loads((base / "results.json").read_text(encoding="utf-8"))
    for cell in published["cells"]:
        rows = [r for r in records if r["model"] == cell["model"]]
        if not rows:
            raise AssertionError(f"no transcripts for {cell['model']}")
        passes = sum(1 for r in rows if r["pass"])
        rate = round(passes / len(rows), 4)
        if abs(rate - cell["pass_rate"]) > 1e-4:
            raise AssertionError(
                f"{cell['label']}: transcripts give {rate}, results.json says "
                f"{cell['pass_rate']}"
            )
    return "pass rates match for every cell"


@check("the offline test suite passes")
def tests_pass() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stdout[-500:])
    summary = [l for l in proc.stdout.splitlines() if "passed" in l or "failed" in l]
    return summary[-1].strip() if summary else "suite passed"


# -------------------------------------------------------------------- live


@check("LIVE: eval.py against the published Hugging Face checkpoint")
def live_eval() -> str:
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", HF_REPO,
         "--eval-set", "scenarios/heldout.jsonl", "--offline-judge", "--limit", "3"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise AssertionError(f"exit {proc.returncode}: {proc.stderr[-600:]}")
    if "scenarios measured" not in proc.stdout:
        raise AssertionError("no results table")
    return "pulled from the Hub and generated"


OFFLINE = [
    submission_pins, dataset_hash_matches, spec_hash_matches, bare_repo_id_resolves,
    eval_set_loads, eval_py_offline, eval_py_bad_model, eval_py_bad_set,
    unseen_eval_set, transcripts_present, metrics_recompute, tests_pass,
]
LIVE = [live_eval]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also run the real commands against the Hub checkpoint")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the full suite (it is the slowest check)")
    args = parser.parse_args(argv)

    checks = [c for c in OFFLINE if not (args.skip_tests and c is tests_pass)]
    if args.live:
        checks = checks + LIVE

    print("Running what a grader runs.\n")
    for fn in checks:
        fn()

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:{width}}  {detail}")

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED — a grader would hit this:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"All {len(results)} checks passed.")
    if not args.live:
        print("Note: the live Hub checkpoint was NOT exercised. Re-run with --live "
              "on a GPU machine to test the command a grader actually types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
