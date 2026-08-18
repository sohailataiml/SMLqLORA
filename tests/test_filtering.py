"""Quality gate: acceptance, rejection, reason tracking, dedupe, contamination."""

from __future__ import annotations

import pytest

from evaluation.judge import DeterministicJudge
from evaluation.schemas import Scenario
from filtering.balance import DEFAULT_CAPS, balance, distribution
from filtering.dedupe import (
    check_contamination,
    deduplicate,
    drop_contaminated,
    jaccard,
    shingles,
)
from filtering.quality_gate import (
    dataset_hash,
    render_report,
    run_quality_gate,
    write_dataset_version,
)
from filtering.static_checks import check_integrity, static_screen
from generation.schemas import GeneratedExample, GenerationDimensions, Provenance

GOOD_RESPONSE = "If nums has four items, what is the last index your loop visits?"
LEAK_RESPONSE = "Here's the corrected code:\n```python\nfor i in range(len(nums)):\n```"


def make_example(
    spec,
    *,
    example_id="gen_v1_00001",
    code="def total(nums):\n    s = 0\n    for i in range(len(nums) - 1):\n        s += nums[i]\n    return s",
    response=GOOD_RESPONSE,
    language="python",
    bug_category="loop_boundary",
    difficulty="easy",
    pressure_type="normal",
    student_message="My sum is too small. Why?",
    expected_fix="for i in range(len(nums)):",
    solved=False,
) -> GeneratedExample:
    scenario = Scenario(
        id=example_id,
        language=language,
        bug_category=bug_category,
        difficulty=difficulty,
        code=code,
        student_message=student_message,
        expected_bug="The loop stops one index early so the last element is skipped.",
        expected_fix=expected_fix,
        student_has_solved=solved,
        pressure_type=pressure_type,
        source="teacher:test",
        split="train",
    )
    return GeneratedExample(
        id=example_id,
        scenario=scenario,
        tutor_response=response,
        dimensions=GenerationDimensions(
            language=language,
            bug_category=bug_category,
            difficulty=difficulty,
            pressure_type=pressure_type,
            conversation_turns=0,
            learner_competence="beginner",
            learner_frustration="none",
            hint_strength="nudge",
            student_progress="solved" if solved else "stuck",
        ),
        provenance=Provenance(
            teacher_model="mock:teacher",
            generation_prompt_version="1.0.0",
            template_id="socratic_v1:test",
            generation_seed=1,
            dataset_version="v1",
            behavior_spec_version=spec.version,
        ),
    )


# --------------------------------------------------------- static checks


def test_good_example_passes_static_screen(spec):
    ok, codes, _, _ = static_screen(make_example(spec), spec)
    assert ok, codes


def test_leaking_example_is_rejected(spec):
    ok, codes, _, _ = static_screen(make_example(spec, response=LEAK_RESPONSE), spec)
    assert not ok
    assert "SOLUTION_LEAK" in codes


def test_unparseable_python_is_rejected(spec):
    example = make_example(spec, code="def broken(:\n    pass")
    result = check_integrity(example)
    assert "INVALID_SCHEMA" in result.codes


def test_fix_identical_to_code_is_rejected(spec):
    example = make_example(spec, code="x = 1", expected_fix="x = 1")
    assert "INCORRECT_DIAGNOSIS" in check_integrity(example).codes


def test_language_mismatch_is_caught(spec):
    example = make_example(spec, language="javascript")  # code is Python
    assert "INVALID_SCHEMA" in check_integrity(example).codes


def test_leaky_history_is_rejected(spec):
    from evaluation.schemas import Message, Role

    example = make_example(spec)
    scenario = example.scenario.model_copy(
        update={
            "conversation_history": (
                Message(role=Role.USER, content="help"),
                Message(
                    role=Role.ASSISTANT,
                    content="Use for i in range(len(nums)): instead.",
                ),
            )
        }
    )
    result = check_integrity(example.model_copy(update={"scenario": scenario}))
    assert "SOLUTION_LEAK" in result.codes


def test_overlong_response_is_rejected(spec):
    example = make_example(spec, response="word " * 400 + "?")
    ok, codes, _, _ = static_screen(example, spec)
    assert not ok
    assert "OVER_EXPLANATION" in codes


# ---------------------------------------------------------------- dedupe


def test_jaccard_and_shingles():
    assert jaccard(shingles("a b c d e"), shingles("a b c d e")) == pytest.approx(1.0)
    assert jaccard(shingles("a b c d"), shingles("w x y z")) == 0.0


def test_exact_duplicates_are_removed(spec):
    a = make_example(spec, example_id="gen_v1_00001")
    b = make_example(spec, example_id="gen_v1_00002")
    result = deduplicate([a, b])
    assert len(result.kept) == 1
    assert result.exact_count == 1
    assert result.duplicates[0].rejection_codes == ("DUPLICATE",)
    assert result.duplicates[0].duplicate_of == "gen_v1_00001"


def test_near_duplicates_are_removed(spec):
    a = make_example(spec, example_id="gen_v1_00001")
    b = make_example(
        spec, example_id="gen_v1_00002",
        student_message="My sum is too small. Why is that?",
    )
    result = deduplicate([a, b])
    assert len(result.kept) == 1
    assert result.near_count == 1
    assert "near-duplicate" in result.duplicates[0].gate_notes


def test_distinct_examples_are_kept(spec):
    a = make_example(spec, example_id="gen_v1_00001")
    b = make_example(
        spec,
        example_id="gen_v1_00002",
        code="const doubled = items.forEach(n => n * 2);",
        language="javascript",
        bug_category="map_vs_foreach",
        student_message="doubled is undefined and I cannot see why.",
        expected_fix="const doubled = items.map(n => n * 2);",
        response="What does forEach hand back when it finishes?",
    )
    assert len(deduplicate([a, b]).kept) == 2


def test_dedupe_is_order_deterministic(spec):
    items = [make_example(spec, example_id=f"gen_v1_0000{i}") for i in range(1, 4)]
    assert [e.id for e in deduplicate(items).kept] == [
        e.id for e in deduplicate(items).kept
    ]


# --------------------------------------------------------- contamination


def test_contamination_detects_an_eval_scenario(spec, unsolved_scenario):
    example = make_example(
        spec,
        code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message,
    )
    report = check_contamination([example], [unsolved_scenario])
    assert not report.clean
    assert example.id in report.contaminated_ids


def test_contamination_detects_near_overlap(spec, unsolved_scenario):
    example = make_example(
        spec,
        code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message + " Any ideas at all?",
    )
    report = check_contamination([example], [unsolved_scenario])
    assert not report.clean


def test_clean_training_data_passes(spec, unsolved_scenario):
    example = make_example(
        spec,
        code="const x = items.forEach(n => n);",
        language="javascript",
        bug_category="map_vs_foreach",
        student_message="Totally different question about forEach.",
        expected_fix="const x = items.map(n => n);",
    )
    report = check_contamination([example], [unsolved_scenario])
    assert report.clean
    assert "No training example" in report.summary()


def test_drop_contaminated_marks_records(spec, unsolved_scenario):
    example = make_example(
        spec, code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message,
    )
    report = check_contamination([example], [unsolved_scenario])
    clean, dirty = drop_contaminated([example], report)
    assert clean == []
    assert dirty[0].rejection_codes == ("CONTAMINATED",)


# --------------------------------------------------------------- balance


def test_balance_caps_a_dominant_bucket(spec):
    examples = [
        make_example(spec, example_id=f"gen_v1_{i:05d}", bug_category="loop_boundary",
                     student_message=f"question number {i}")
        for i in range(50)
    ]
    result = balance(examples, caps={"bug_category": 0.2})
    assert len(result.kept) <= 11
    assert all(e.rejection_codes == ("UNBALANCED",) for e in result.dropped)


def test_balance_keeps_a_diverse_set(spec):
    examples = [
        make_example(spec, example_id=f"gen_v1_{i:05d}",
                     bug_category=f"category_{i % 10}",
                     student_message=f"question number {i}")
        for i in range(40)
    ]
    result = balance(examples, caps={"bug_category": 0.2})
    assert len(result.kept) == 40


def test_distribution_reports_every_axis(spec):
    dist = distribution([make_example(spec)])
    assert set(dist) >= {"language", "bug_category", "pressure_type", "difficulty"}


# ------------------------------------------------------------ full gate


def test_gate_accepts_good_and_rejects_bad(spec):
    good = make_example(spec, example_id="gen_v1_00001")
    bad = make_example(
        spec, example_id="gen_v1_00002", response=LEAK_RESPONSE,
        student_message="Different question so it is not a duplicate.",
    )
    outcome = run_quality_gate(
        [good, bad], DeterministicJudge(spec), spec=spec, max_workers=1
    )
    assert [e.id for e in outcome.accepted] == ["gen_v1_00001"]
    assert outcome.rejected[0].id == "gen_v1_00002"
    assert "SOLUTION_LEAK" in outcome.rejected[0].rejection_codes


def test_gate_report_tracks_every_number(spec):
    examples = [
        make_example(spec, example_id=f"gen_v1_{i:05d}",
                     student_message=f"unique question number {i}")
        for i in range(6)
    ]
    examples.append(
        make_example(spec, example_id="gen_v1_00099", response=LEAK_RESPONSE,
                     student_message="another distinct question entirely")
    )
    outcome = run_quality_gate(
        examples, DeterministicJudge(spec), spec=spec, max_workers=1,
        dataset_version="v9",
    )
    report = outcome.report
    assert report.dataset_version == "v9"
    assert report.candidate_count == 7
    assert report.accepted_count + report.rejected_count == 7
    assert 0.0 <= report.acceptance_rate <= 1.0
    assert report.rejections_by_reason
    assert report.rejections_by_stage
    assert report.language_distribution
    assert report.conversation_length_distribution
    assert len(report.dataset_hash) == 64
    assert report.judge["judge_model"] == "deterministic-judge"
    assert report.thresholds["min_judge_spec_adherence"]


def test_gate_stages_run_in_order(spec):
    seen = []
    run_quality_gate(
        [make_example(spec)], DeterministicJudge(spec), spec=spec, max_workers=1,
        on_progress=lambda stage, survived, entering: seen.append(stage),
    )
    assert seen == ["schema", "static_checks", "llm_judge", "dedupe",
                    "contamination", "balance"]


def test_rejected_examples_are_kept_not_deleted(spec, tmp_path):
    bad = make_example(spec, response=LEAK_RESPONSE)
    outcome = run_quality_gate(
        [bad], DeterministicJudge(spec), spec=spec, max_workers=1, dataset_version="v1"
    )
    paths = write_dataset_version(outcome, repo_root=tmp_path, dataset_version="v1")
    assert paths["rejected"].exists()
    assert paths["rejected"].read_text(encoding="utf-8").strip()
    assert paths["report_json"].exists()
    assert paths["report_md"].exists()


def test_dataset_hash_is_order_independent(spec):
    a = make_example(spec, example_id="gen_v1_00001")
    b = make_example(spec, example_id="gen_v1_00002", student_message="another one")
    assert dataset_hash([a, b]) == dataset_hash([b, a])


def test_report_renders_markdown(spec):
    outcome = run_quality_gate(
        [make_example(spec)], DeterministicJudge(spec), spec=spec, max_workers=1
    )
    markdown = render_report(outcome.report)
    assert "# Dataset" in markdown
    assert "## Funnel" in markdown
    assert "## Rejections by reason" in markdown


def test_contaminated_candidate_never_reaches_the_dataset(spec, unsolved_scenario):
    contaminated = make_example(
        spec, code=unsolved_scenario.code,
        student_message=unsolved_scenario.student_message,
    )
    outcome = run_quality_gate(
        [contaminated], DeterministicJudge(spec), spec=spec,
        eval_scenarios=[unsolved_scenario], max_workers=1,
    )
    assert outcome.accepted == []
    assert "CONTAMINATED" in outcome.rejected[0].rejection_codes
