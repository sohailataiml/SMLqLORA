"""The corrected-baseline notebook is the only path to the N=600 answer.

It runs interactively on a machine this suite never sees, so nothing here can
execute it. What can be checked is everything that would waste a GPU session or
spend judge credit wrongly: that every code cell parses, that the guards which
must stop the run are present, and that no cell points at an MVP artifact.

A syntax error in cell 20 costs 36 minutes of training before it surfaces.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks/train_corrected_baseline.ipynb"

RUN = "socratic-v1-n600-bestckpt"
CONFIG = "training/configs/qlora_qwen3_1_7b_t4_bestckpt.yaml"
OUTPUT = "results/n600_v1_baseline"

#: Paths the MVP submission depends on. The notebook may read them; it must never
#: name them as a write destination.
MVP_ARTIFACTS = (
    "outputs/socratic-v1-n600",
    "results/base_vs_tuned",
    "data/versions/v1/selected.jsonl",
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def all_source(notebook) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"])


def test_notebook_is_valid_nbformat(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_every_code_cell_parses(code_cells):
    """A syntax error here costs a 36-minute training run to discover."""
    transform = None
    try:
        from IPython.core.inputtransformer2 import TransformerManager

        transform = TransformerManager().transform_cell
    except ImportError:  # pragma: no cover - IPython absent
        pytest.skip("IPython not installed; cannot expand ! and % magics")

    for index, source in enumerate(code_cells):
        try:
            ast.parse(transform(source))
        except SyntaxError as exc:
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_it_trains_the_corrected_config_into_a_new_directory(all_source):
    assert CONFIG in all_source
    assert f'RUN    = "{RUN}"' in all_source


def test_it_refuses_to_reuse_the_mvp_run_name(all_source):
    assert 'assert RUN != "socratic-v1-n600"' in all_source


def test_training_checks_the_exit_code_rather_than_piping_to_tee(code_cells):
    """`!python ... | tee log` reports tee's status, so a crash looks like success.

    Checked against executable lines only — the notebook comments explain the
    hazard by name, and matching those would flag the warning as the defect.
    """
    executable = "\n".join(
        line for source in code_cells for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "assert returncode == 0" in executable
    assert "| tee" not in executable


def test_checkpoint_selection_is_verified_not_assumed(all_source):
    assert "scripts/verify_checkpoint_selection.py" in all_source
    assert 'assert verdict == "VERIFIED_BEST"' in all_source


def test_the_trl_in_use_must_support_checkpoint_selection(all_source):
    """Without these the run silently exports the final checkpoint again."""
    for argument in ("load_best_model_at_end", "metric_for_best_model",
                     "greater_is_better"):
        assert argument in all_source


def test_config_invariants_are_asserted_before_the_gpu_is_used(all_source):
    assert 'expected = {"load_best_model_at_end", "metric_for_best_model",' in all_source
    assert "assert changed == expected" in all_source


def test_generation_is_smoke_checked_before_judge_credit_is_spent(all_source):
    assert "assert empty == 0" in all_source
    # The three cases Step 4 asks for.
    assert '"normal"' in all_source
    assert '"adversarial"' in all_source
    assert '"solved"' in all_source


def test_the_smoke_check_prints_errors_and_usage_not_just_text(all_source):
    """An adapter that failed to load looks exactly like a tutor with nothing
    to say, unless the error and token counts are shown."""
    assert "response.error" in all_source
    assert "response.usage" in all_source


def test_the_smoke_check_counts_the_attractor_markers(all_source):
    assert "so the problem is not" in all_source
    assert "repeated_sentences" in all_source


def test_quota_is_checked_before_the_paid_run(all_source):
    assert "scripts/preflight.py" in all_source
    preflight = all_source.index("scripts/preflight.py")
    evaluation = all_source.index("--judge anthropic:claude-opus-5")
    assert preflight < evaluation, "preflight must come before the paid evaluation"


def test_results_go_to_a_new_directory(all_source):
    assert f'OUT = "{OUTPUT}"' in all_source


def test_it_never_writes_over_an_mvp_artifact(code_cells):
    for index, source in enumerate(code_cells):
        for artifact in MVP_ARTIFACTS:
            for verb in (f"cp -r {artifact}", f"rm -rf {artifact}", f"> {artifact}"):
                assert verb not in source, f"cell {index} would clobber {artifact}"


def test_it_does_not_publish_to_hugging_face(all_source):
    assert "push_to_hub" not in all_source
    assert "!huggingface-cli upload" not in all_source


def test_it_stops_before_dataset_v2(all_source):
    assert "Stop here" in all_source
    assert "Do **not** build Dataset V2" in all_source


def test_the_config_it_names_exists_and_is_the_corrected_one():
    import yaml

    config = yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))
    assert config["training"]["load_best_model_at_end"] is True
    assert config["training"]["metric_for_best_model"] == "eval_loss"
    assert config["training"]["save_total_limit"] >= 3


def test_the_scripts_it_calls_exist(all_source):
    for script in ("scripts/verify_checkpoint_selection.py",
                   "scripts/verify_training_data.py", "scripts/preflight.py"):
        assert script in all_source
        assert (REPO_ROOT / script).exists(), f"{script} is referenced but missing"


def test_the_modules_it_calls_are_importable(all_source):
    assert "analysis.compare_runs" in all_source
    assert "analysis.failure_taxonomy" in all_source
    import analysis.compare_runs  # noqa: F401
    import analysis.failure_taxonomy  # noqa: F401


def test_the_evaluation_is_archived_before_the_notebook_ends(code_cells):
    """A recycled runtime once took twenty paid judge transcripts with it.

    Archiving only at the end is archiving only if nothing goes wrong in
    between, and the thing that goes wrong is the runtime itself.
    """
    evaluation = next(i for i, s in enumerate(code_cells)
                      if "eval.py" in s and "--judge" in s)
    archive = next(i for i, s in enumerate(code_cells) if "EVAL_ARCHIVE" in s)
    assert archive == evaluation + 1, (
        "the archive step must come directly after the paid evaluation, "
        f"not {archive - evaluation} cells later"
    )


def test_a_finished_run_can_be_restored_without_retraining(all_source):
    """38 minutes of T4 time should survive a lost VM."""
    assert "SAVED = Path(" in all_source
    assert "Train it with section 6 instead" in all_source


def test_restoring_refuses_to_overwrite_an_existing_adapter(all_source):
    assert 'if (DEST / "adapter_model.safetensors").exists():' in all_source
    assert "leaving it untouched" in all_source
