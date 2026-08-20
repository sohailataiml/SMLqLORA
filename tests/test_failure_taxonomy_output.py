"""`--output` must decide where the taxonomy lands.

It was parsed and then ignored: every `--write` overwrote
`results/failure_analysis/v1_n600_failure_taxonomy.json`, the MVP's committed
evidence, no matter what the caller passed. Running the corrected baseline's
taxonomy therefore destroyed the historical one and silently failed to produce
the file the notebook then tries to archive.

The MVP run and the corrected run are evidence about two different checkpoints.
They do not share a filename.
"""

from __future__ import annotations

import json

import pytest

from analysis.failure_taxonomy import OUTPUT, main

MVP_TRANSCRIPTS = "results/base_vs_tuned/judge_transcripts.jsonl"


@pytest.fixture
def mvp_taxonomy_is_restored():
    """Never let a test leave the committed MVP artifact modified."""
    before = OUTPUT.read_bytes() if OUTPUT.exists() else None
    yield
    if before is None:
        if OUTPUT.exists():
            OUTPUT.unlink()
    else:
        OUTPUT.write_bytes(before)


def test_output_decides_where_the_json_is_written(tmp_path, mvp_taxonomy_is_restored):
    destination = tmp_path / "n600_v1_baseline_taxonomy.json"
    code = main(["--transcripts", MVP_TRANSCRIPTS, "--write",
                 "--output", str(destination)])
    assert code == 0
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["counts"]["scenarios"] == 20


def test_output_does_not_touch_the_mvp_taxonomy(tmp_path, mvp_taxonomy_is_restored):
    """The regression that mattered: writing elsewhere must leave the MVP alone."""
    before = OUTPUT.read_bytes()
    destination = tmp_path / "elsewhere.json"
    main(["--transcripts", MVP_TRANSCRIPTS, "--write", "--output", str(destination)])
    assert OUTPUT.read_bytes() == before


def test_a_relative_output_resolves_against_the_repo(tmp_path, mvp_taxonomy_is_restored):
    relative = "results/failure_analysis/_test_relative_taxonomy.json"
    from analysis.failure_taxonomy import REPO_ROOT
    target = REPO_ROOT / relative
    try:
        main(["--transcripts", MVP_TRANSCRIPTS, "--write", "--output", relative])
        assert target.exists()
    finally:
        if target.exists():
            target.unlink()


def test_without_output_it_still_writes_the_default(mvp_taxonomy_is_restored):
    code = main(["--transcripts", MVP_TRANSCRIPTS, "--write"])
    assert code == 0
    assert OUTPUT.exists()


def test_the_message_names_the_file_it_actually_wrote(tmp_path, capsys,
                                                      mvp_taxonomy_is_restored):
    destination = tmp_path / "named.json"
    main(["--transcripts", MVP_TRANSCRIPTS, "--write", "--output", str(destination)])
    out = capsys.readouterr().out
    assert "v1_n600_failure_taxonomy.json" not in out
