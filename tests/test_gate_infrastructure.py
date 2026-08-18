"""An unreachable judge must never look like a bad candidate.

Tranche 1 of Dataset V1 made this concrete: the Anthropic account ran out of
credit part-way through judging, and the gate recorded 410 unreachable-judge
placeholders as `LOW_QUALITY` / `IRRELEVANT_HINT` rejections. Frozen, that would
have published a 35% quality-failure rate that never happened, and it would have
depressed the reported acceptance rate from 74.1% to 48.6%.
"""

from __future__ import annotations

import pytest

from evaluation.judge import DeterministicJudge, Judge
from evaluation.schemas import DeterministicResult, JudgeResult, Scenario
from filtering.judge import (
    UNJUDGED,
    CachedJudge,
    build_verdict_cache,
    judge_candidate,
    judge_unavailable,
)
from filtering.quality_gate import run_quality_gate
from generation.generate import build_teacher, generate


@pytest.fixture(scope="module")
def candidates():
    teacher = build_teacher("mock:teacher", mock=True, dataset_version="vtest")
    rows, _, _ = generate(count=20, teacher=teacher, verbose=False)
    return rows


class _UnreachableJudge(Judge):
    """Mimics the fail-closed placeholder a dead judge produces."""

    model_name = "dead:judge"
    model_family = "dead"

    def judge(self, scenario: Scenario, response: str,
              deterministic: DeterministicResult | None = None) -> JudgeResult:
        return JudgeResult(
            spec_adherence=0.0, robustness=0.0, hint_relevance=0.0,
            passed=False, failure_reasons=("LOW_QUALITY",),
            reasoning="Judge failed after 3 attempts: credit balance is too low",
            judge_model=self.model_name, judge_model_family=self.model_family,
            parse_warnings=("judge_unavailable",),
        )


class _FlakyJudge(Judge):
    """Judges the first `ok` candidates, then goes down — like a spent balance."""

    model_name = "flaky:judge"
    model_family = "flaky"

    def __init__(self, ok: int):
        self.ok = ok
        self.calls = 0
        self._real = DeterministicJudge()

    def judge(self, scenario: Scenario, response: str,
              deterministic: DeterministicResult | None = None) -> JudgeResult:
        self.calls += 1
        if self.calls > self.ok:
            return _UnreachableJudge().judge(scenario, response, deterministic)
        return self._real.judge(scenario, response, deterministic)


# =============================================================================
# Detection
# =============================================================================


def test_judge_unavailable_is_detected():
    result = _UnreachableJudge().judge(None, "x") if False else JudgeResult(
        spec_adherence=0.0, robustness=0.0, hint_relevance=0.0, passed=False,
        parse_warnings=("judge_unavailable",),
    )
    assert judge_unavailable(result) is True


def test_a_real_verdict_is_not_flagged_unavailable():
    result = JudgeResult(spec_adherence=0.9, robustness=1.0, hint_relevance=0.9,
                         passed=True)
    assert judge_unavailable(result) is False


def test_unreachable_judge_yields_the_unjudged_marker(candidates):
    from behavior.spec import load_spec

    _, codes = judge_candidate(candidates[0], _UnreachableJudge(), load_spec())
    assert codes == (UNJUDGED,)


def test_unreachable_judge_does_not_yield_quality_codes(candidates):
    from behavior.spec import load_spec

    _, codes = judge_candidate(candidates[0], _UnreachableJudge(), load_spec())
    assert "LOW_QUALITY" not in codes
    assert "IRRELEVANT_HINT" not in codes


# =============================================================================
# Gate routing
# =============================================================================


def test_unjudged_candidates_are_not_rejected(candidates):
    outcome = run_quality_gate(candidates, _UnreachableJudge(),
                               dataset_version="vtest")
    assert outcome.accepted == []
    assert outcome.rejected == []
    assert len(outcome.unjudged) == len(candidates)


def test_acceptance_rate_excludes_unjudged_candidates(candidates):
    """The denominator is judged candidates, so an outage cannot depress it."""
    half = len(candidates) // 2
    outcome = run_quality_gate(candidates, _FlakyJudge(ok=half),
                               dataset_version="vtest")
    report = outcome.report

    assert report.unjudged_count > 0
    assert report.judged_count == len(candidates) - report.unjudged_count
    expected = round(report.accepted_count / report.judged_count, 4)
    assert report.acceptance_rate == expected


def test_report_is_marked_incomplete_when_any_candidate_is_unjudged(candidates):
    outcome = run_quality_gate(candidates, _FlakyJudge(ok=3),
                               dataset_version="vtest")
    assert outcome.report.complete is False


def test_report_is_complete_when_every_candidate_was_judged(candidates):
    outcome = run_quality_gate(candidates, DeterministicJudge(),
                               dataset_version="vtest")
    assert outcome.report.complete is True
    assert outcome.report.unjudged_count == 0


def test_unjudged_candidates_never_enter_the_accepted_set(candidates):
    outcome = run_quality_gate(candidates, _FlakyJudge(ok=5),
                               dataset_version="vtest")
    accepted_ids = {e.id for e in outcome.accepted}
    unjudged_ids = {e.id for e in outcome.unjudged}
    assert accepted_ids.isdisjoint(unjudged_ids)


def test_unjudged_do_not_appear_in_rejection_reasons(candidates):
    outcome = run_quality_gate(candidates, _UnreachableJudge(),
                               dataset_version="vtest")
    assert outcome.report.rejections_by_reason == {}


# =============================================================================
# Judge resume
# =============================================================================


def test_cached_judge_reuses_prior_verdicts(candidates):
    first = run_quality_gate(candidates, DeterministicJudge(),
                             dataset_version="vtest")
    cache = build_verdict_cache(first.accepted + first.rejected)

    inner = _FlakyJudge(ok=0)  # would fail every call if actually consulted
    cached = CachedJudge(inner, cache)
    second = run_quality_gate(candidates, cached, dataset_version="vtest")

    assert second.report.unjudged_count == 0
    assert len(second.accepted) == len(first.accepted)
    assert cached.hits > 0


def test_verdict_cache_skips_unavailable_placeholders(candidates):
    outcome = run_quality_gate(candidates, _UnreachableJudge(),
                               dataset_version="vtest")
    # Placeholders must not be cached, or a resume would never retry them.
    assert build_verdict_cache(outcome.unjudged) == {}


def test_verdict_cache_misses_when_the_response_changes(candidates):
    outcome = run_quality_gate(candidates, DeterministicJudge(),
                               dataset_version="vtest")
    cache = build_verdict_cache(outcome.accepted)
    if not outcome.accepted:
        pytest.skip("no accepted candidates to mutate")

    edited = outcome.accepted[0].model_copy(
        update={"tutor_response": "a completely different tutor turn"}
    )
    cached = CachedJudge(DeterministicJudge(), cache)
    cached.judge(edited.scenario, edited.tutor_response)
    assert cached.misses == 1, "an edited candidate must not inherit a verdict"


def test_cached_judge_describes_the_underlying_judge(candidates):
    cached = CachedJudge(DeterministicJudge(), {})
    assert cached.describe() == DeterministicJudge().describe()


# =============================================================================
# Response-length bound is state-aware
# =============================================================================


def test_solved_responses_get_a_larger_length_ceiling(candidates):
    """The spec asks solved-state replies to confirm *and* explain.

    A flat ceiling calibrated on one-question unresolved replies rejected 98.8%
    of solved candidates in tranche 1 — before the judge saw them — which would
    have taught the model never to confirm an answer.
    """
    from filtering.static_checks import (
        MAX_RESPONSE_CHARS,
        MAX_RESPONSE_CHARS_SOLVED,
        max_response_chars,
    )

    solved = next(c for c in candidates if c.scenario.student_has_solved)
    unresolved = next(c for c in candidates if not c.scenario.student_has_solved)

    assert max_response_chars(solved) == MAX_RESPONSE_CHARS_SOLVED
    assert max_response_chars(unresolved) == MAX_RESPONSE_CHARS
    assert MAX_RESPONSE_CHARS_SOLVED > MAX_RESPONSE_CHARS


def test_a_long_confirmation_passes_static_checks_when_solved(candidates):
    from behavior.spec import load_spec
    from filtering.static_checks import static_screen

    solved = next(c for c in candidates if c.scenario.student_has_solved)
    long_confirmation = solved.tutor_response + " " + ("Here is why it works. " * 55)
    assert 1200 < len(long_confirmation) <= 2400

    candidate = solved.model_copy(update={"tutor_response": long_confirmation})
    _, codes, _, integrity = static_screen(candidate, load_spec())
    assert not any("length" in note for note in integrity.notes), integrity.notes


def test_the_same_length_still_fails_when_unresolved(candidates):
    """The tight bound must remain tight where one question is the behavior."""
    from behavior.spec import load_spec
    from filtering.static_checks import static_screen

    unresolved = next(c for c in candidates if not c.scenario.student_has_solved)
    rambling = unresolved.tutor_response + " " + ("Also consider this. " * 70)
    assert len(rambling) > 1200

    candidate = unresolved.model_copy(update={"tutor_response": rambling})
    _, codes, _, integrity = static_screen(candidate, load_spec())
    assert "LOW_QUALITY" in codes


def test_solved_ceiling_is_not_unbounded(candidates):
    from behavior.spec import load_spec
    from filtering.static_checks import static_screen

    solved = next(c for c in candidates if c.scenario.student_has_solved)
    essay = "word " * 1200
    candidate = solved.model_copy(update={"tutor_response": essay})
    _, codes, _, _ = static_screen(candidate, load_spec())
    assert "LOW_QUALITY" in codes


# =============================================================================
# Verdict journal — crash safety for paid calls
# =============================================================================


def test_journal_persists_verdicts_as_they_are_bought(candidates, tmp_path):
    from filtering.judge import VerdictJournal, load_journal

    journal = VerdictJournal(tmp_path / "journal.jsonl")
    cached = CachedJudge(DeterministicJudge(), {}, journal=journal)
    for candidate in candidates[:5]:
        cached.judge(candidate.scenario, candidate.tutor_response)

    # Readable immediately, without any run having finished.
    recovered = load_journal(tmp_path / "journal.jsonl")
    assert len(recovered) == 5


def test_journal_never_records_an_unavailable_verdict(candidates, tmp_path):
    """Journalling a placeholder would stop the next run retrying it."""
    from filtering.judge import VerdictJournal, load_journal

    journal = VerdictJournal(tmp_path / "journal.jsonl")
    cached = CachedJudge(_UnreachableJudge(), {}, journal=journal)
    cached.judge(candidates[0].scenario, candidates[0].tutor_response)

    assert load_journal(tmp_path / "journal.jsonl") == {}


def test_journalled_verdicts_are_reusable_after_an_interruption(candidates, tmp_path):
    from filtering.judge import VerdictJournal, load_journal

    path = tmp_path / "journal.jsonl"
    first = CachedJudge(DeterministicJudge(), {}, journal=VerdictJournal(path))
    for candidate in candidates[:4]:
        first.judge(candidate.scenario, candidate.tutor_response)

    # A later run loads the journal and must not re-buy those calls.
    second = CachedJudge(_FlakyJudge(ok=0), load_journal(path))
    for candidate in candidates[:4]:
        second.judge(candidate.scenario, candidate.tutor_response)

    assert second.hits == 4
    assert second.misses == 0


def test_journal_survives_a_truncated_final_line(tmp_path):
    from filtering.judge import load_journal

    path = tmp_path / "journal.jsonl"
    path.write_text('{"key": "a", "verdict": {"spec_adherence": 0.9, '
                    '"robustness": 1.0, "hint_relevance": 0.9, "passed": true}}\n'
                    '{"key": "b", "verdi',
                    encoding="utf-8")
    recovered = load_journal(path)
    assert set(recovered) == {"a"}
