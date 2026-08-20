"""The three-way comparison, base-transcript reuse, and checkpoint verification.

The comparison table is the artifact the Early Submission conclusion rests on, so
its BASE and MVP V1 columns are asserted against the *published* MVP report. An
earlier draft of `cell_metrics` recomputed the metrics by hand and produced a
base solution-leak rate of 0.750 against the published 0.450 -- it counted only
the judge's failure codes, where the published figure counts the combined
deterministic-plus-judge codes, and it averaged robustness over all scenarios
rather than the adversarial ones. Both would have silently contradicted the
submitted result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from analysis.compare_runs import (
    apply_decision_rule,
    build,
    cell_metrics,
    load_cell,
    render,
    reuse_is_valid,
)
from analysis.corpus import TRANSCRIPTS, load_heldout
from scripts.verify_checkpoint_selection import best_epoch_by_loss, verify

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "results/base_vs_tuned"

#: Straight from results/base_vs_tuned/report.md, which is submitted and frozen.
PUBLISHED = {
    "BASE": {
        "spec_adherence": 0.045, "robustness": 0.233, "hint_relevance": 0.573,
        "pass_rate": 0.0, "solution_leak_rate": 0.45,
        "premature_confirmation_rate": 0.0, "passes": 0, "solution_leaks": 9,
    },
    "MVP_V1": {
        "spec_adherence": 0.459, "robustness": 0.678, "hint_relevance": 0.408,
        "pass_rate": 0.25, "solution_leak_rate": 0.0,
        "premature_confirmation_rate": 0.05, "passes": 5, "solution_leaks": 0,
    },
}

PUBLISHED_FAILURE_MODES = {
    "BASE": {"SOLUTION_LEAK": 9, "MULTIPLE_HINTS": 17, "OVER_EXPLANATION": 12,
             "EXPLICIT_FINAL_DIAGNOSIS": 11, "INCORRECT_DIAGNOSIS": 2,
             "FAILED_TO_ADAPT": 1, "LOW_QUALITY": 2},
    "MVP_V1": {"SOLUTION_LEAK": 0, "MULTIPLE_HINTS": 5, "OVER_EXPLANATION": 1,
               "EXPLICIT_FINAL_DIAGNOSIS": 1, "INCORRECT_DIAGNOSIS": 4,
               "FAILED_TO_ADAPT": 5, "IRRELEVANT_HINT": 3, "LOW_QUALITY": 5,
               "WITHHELD_AFTER_SOLVED": 1, "PREMATURE_CONFIRMATION": 1,
               "DUPLICATE": 1},
}


@pytest.fixture(scope="module")
def heldout():
    return load_heldout()


@pytest.fixture(scope="module")
def cells(heldout):
    return {
        "BASE": cell_metrics(load_cell(TRANSCRIPTS, adapter=False), heldout),
        "MVP_V1": cell_metrics(load_cell(TRANSCRIPTS, adapter=True), heldout),
    }


@pytest.mark.parametrize("cell", ["BASE", "MVP_V1"])
def test_headline_metrics_match_the_published_report(cells, cell):
    for metric, expected in PUBLISHED[cell].items():
        assert cells[cell][metric] == pytest.approx(expected, abs=1e-3), (
            f"{cell}.{metric} is {cells[cell][metric]}, but the submitted report "
            f"says {expected}"
        )


@pytest.mark.parametrize("cell", ["BASE", "MVP_V1"])
def test_failure_mode_counts_match_the_published_report(cells, cell):
    counts = cells[cell]["failure_modes"]
    for code, expected in PUBLISHED_FAILURE_MODES[cell].items():
        assert counts.get(code, 0) == expected, f"{cell}.{code}"


def test_both_cells_were_fully_measured(cells):
    for cell in ("BASE", "MVP_V1"):
        assert cells[cell]["n"] == 20
        assert cells[cell]["infrastructure_errors"] == 0
        assert cells[cell]["empty_responses"] == 0


def test_splits_partition_the_eval_set(cells):
    for cell in cells.values():
        splits = cell["by_split"]
        assert splits["clean"]["n"] + splits["adversarial"]["n"] == 20
        assert splits["first_turn"]["n"] + splits["multi_turn"]["n"] == 20
        assert splits["solved"]["n"] == 2


# ------------------------------------------------------------ base reuse


def _manifest() -> dict:
    return json.loads((MVP_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_identical_manifests_permit_reuse():
    valid, problems = reuse_is_valid(_manifest(), _manifest())
    assert valid and problems == []


@pytest.mark.parametrize("field", [
    "eval_set_hash", "behavior_spec_sha256", "judge_model", "judge_prompt_sha256",
])
def test_a_changed_evaluation_path_refuses_reuse(field):
    corrected = _manifest()
    corrected[field] = "something-else"
    valid, problems = reuse_is_valid(_manifest(), corrected)
    assert not valid
    assert any(field in p for p in problems)


def test_changed_generation_settings_refuse_reuse():
    corrected = _manifest()
    corrected["generation_params"] = {**corrected["generation_params"], "seed": 999}
    valid, problems = reuse_is_valid(_manifest(), corrected)
    assert not valid
    assert any("seed" in p for p in problems)


# ------------------------------------------------------- end-to-end build


@pytest.fixture
def corrected_dir(tmp_path):
    """Stand the MVP tuned cell in as a 'corrected' run to exercise the plumbing."""
    directory = tmp_path / "n600_v1_baseline"
    directory.mkdir()
    records = [
        line for line in
        (MVP_DIR / "judge_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and "peft" in line
    ]
    (directory / "judge_transcripts.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8")
    shutil.copy(MVP_DIR / "manifest.json", directory / "manifest.json")
    return directory


def test_build_produces_three_cells(corrected_dir):
    report = build(corrected_dir)
    assert set(report["cells"]) == {"BASE", "MVP_V1", "CORRECTED_V1"}
    assert report["base_reuse_valid"] is True
    assert report["result_status"] == "REAL_EXPERIMENT_RESULT"


def test_build_refuses_a_missing_corrected_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="No corrected-run transcripts"):
        build(tmp_path / "absent")


def test_render_emits_every_requested_row(corrected_dir):
    table = render(build(corrected_dir))
    for label in ("spec adherence", "robustness", "hint relevance", "pass rate",
                  "solution leak rate", "premature confirmation", "adversarial",
                  "solved", "first_turn", "multi_turn"):
        assert label in table


# ----------------------------------------------------------- decision rule


def _cell(**overrides):
    base = {
        "hint_relevance": 0.8,
        "failure_modes": {},
        "by_split": {
            "first_turn": {"n": 15, "passes": 8, "pass_rate": 0.53},
            "multi_turn": {"n": 5, "passes": 3, "pass_rate": 0.60},
            "solved": {"n": 2, "passes": 2, "pass_rate": 1.0},
        },
    }
    base.update(overrides)
    return base


def test_branch_one_fires_on_a_multi_turn_gap():
    cell = _cell(by_split={
        "first_turn": {"n": 15, "passes": 9, "pass_rate": 0.60},
        "multi_turn": {"n": 5, "passes": 1, "pass_rate": 0.20},
        "solved": {"n": 2, "passes": 2, "pass_rate": 1.0},
    })
    assert apply_decision_rule(cell)["selected"] == "H-A"


def test_branch_two_fires_on_withholding_after_solved():
    assert apply_decision_rule(
        _cell(failure_modes={"WITHHELD_AFTER_SOLVED": 1})
    )["selected"] == "H-B"


def test_branch_three_fires_on_wrong_diagnosis_codes():
    cell = _cell(failure_modes={"INCORRECT_DIAGNOSIS": 3, "IRRELEVANT_HINT": 1})
    assert apply_decision_rule(cell)["selected"] == "H-C"


def test_branch_three_fires_when_hint_relevance_is_below_base():
    assert apply_decision_rule(_cell(hint_relevance=0.4))["selected"] == "H-C"


def test_no_branch_fires_when_nothing_is_wrong():
    result = apply_decision_rule(_cell())
    assert result["selected"] == "STOP_AND_REPORT"
    assert result["hypothesis"] is None


def test_the_selected_hypothesis_comes_from_the_committed_plan():
    result = apply_decision_rule(_cell(failure_modes={"WITHHELD_AFTER_SOLVED": 1}))
    assert result["hypothesis"]["id"] == "H-B"
    assert "solved-state" in result["hypothesis"]["name"]


def test_the_mvp_run_would_select_h_a(cells):
    """Sanity anchor: the known-degraded checkpoint fires branch 1."""
    assert apply_decision_rule(cells["MVP_V1"])["selected"] == "H-A"


# ------------------------------------------- checkpoint selection verifier


def test_best_epoch_is_the_minimum_eval_loss():
    history = [
        {"epoch": 1.0, "eval_loss": 1.97},
        {"epoch": 2.0, "eval_loss": 2.785},
        {"epoch": 3.0, "eval_loss": 2.73},
    ]
    assert best_epoch_by_loss(history)["epoch"] == 1.0


def test_best_epoch_of_an_empty_history_is_none():
    assert best_epoch_by_loss([]) is None


def _write_run(tmp_path, *, exported: bytes, checkpoints: dict[str, bytes],
               best: str | None, history: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_bytes(exported)
    for name, blob in checkpoints.items():
        directory = run / name
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(blob)
    (run / "checkpoint_metadata.json").write_text(json.dumps({
        "checkpoint_selection": {
            "load_best_model_at_end": best is not None,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 3,
            "best_model_checkpoint": f"/x/{best}" if best else None,
            "best_metric": 1.97,
            "validation_history": history,
        }
    }), encoding="utf-8")
    return run


HISTORY = [
    {"epoch": 1.0, "eval_loss": 1.97},
    {"epoch": 2.0, "eval_loss": 2.785},
    {"epoch": 3.0, "eval_loss": 2.73},
]


def test_verifier_confirms_an_export_matching_the_best_checkpoint(tmp_path):
    run = _write_run(
        tmp_path, exported=b"BEST",
        checkpoints={"checkpoint-34": b"BEST", "checkpoint-68": b"MID",
                     "checkpoint-102": b"LAST"},
        best="checkpoint-34", history=HISTORY,
    )
    code, report = verify(run)
    assert code == 0
    assert report["verdict"] == "VERIFIED_BEST"
    assert report["best_differs_from_final"] is True


def test_verifier_catches_an_export_that_is_actually_the_final_checkpoint(tmp_path):
    """The MVP failure mode: config asks for best, adapter is the last one."""
    run = _write_run(
        tmp_path, exported=b"LAST",
        checkpoints={"checkpoint-34": b"BEST", "checkpoint-102": b"LAST"},
        best="checkpoint-34", history=HISTORY,
    )
    code, report = verify(run)
    assert code == 1
    assert report["verdict"] == "EXPORTED_FINAL_NOT_BEST"


def test_verifier_will_not_guess_without_a_validation_curve(tmp_path):
    run = _write_run(tmp_path, exported=b"X", checkpoints={"checkpoint-1": b"X"},
                     best="checkpoint-1", history=[])
    code, report = verify(run)
    assert code == 2
    assert report["verdict"] == "NO_VALIDATION_HISTORY"


def test_verifier_reports_when_no_checkpoints_survive(tmp_path):
    """save_total_limit: 1 leaves nothing to byte-match against."""
    run = _write_run(tmp_path, exported=b"X", checkpoints={}, best="checkpoint-102",
                     history=HISTORY)
    code, report = verify(run)
    assert code == 2
    assert report["verdict"] == "UNVERIFIED_NO_CHECKPOINT_DIRS"


def test_verifier_notes_when_best_and_final_coincide(tmp_path):
    run = _write_run(
        tmp_path, exported=b"LAST",
        checkpoints={"checkpoint-34": b"A", "checkpoint-102": b"LAST"},
        best="checkpoint-102",
        history=[{"epoch": 1.0, "eval_loss": 2.7}, {"epoch": 2.0, "eval_loss": 1.9}],
    )
    code, report = verify(run)
    assert code == 0
    assert report["best_differs_from_final"] is False
    assert "changed nothing" in report["detail"]


def test_verifier_requires_metadata(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint_metadata.json"):
        verify(tmp_path / "empty")
