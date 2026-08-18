"""Tests for the final deterministic, distribution-preserving selection."""

from __future__ import annotations

import pytest

from evaluation.judge import DeterministicJudge
from filtering.quality_gate import run_quality_gate
from filtering.selection import largest_remainder, select_balanced
from generation.generate import build_teacher, generate

SHARES = {
    "normal": 0.191, "solved": 0.142, "almost_correct": 0.122,
    "time_pressure": 0.101, "frustrated": 0.094,
    "repeated_answer_request": 0.094, "fake_success": 0.094,
    "prompt_injection": 0.081, "authority_override": 0.081,
}


@pytest.fixture(scope="module")
def pool():
    teacher = build_teacher("mock:teacher", mock=True, dataset_version="vtest")
    candidates, _, _ = generate(count=200, teacher=teacher, verbose=False)
    return run_quality_gate(candidates, DeterministicJudge(),
                            dataset_version="vtest").accepted


# =============================================================================
# Quota arithmetic
# =============================================================================


def test_largest_remainder_sums_exactly_to_target():
    quotas = largest_remainder(SHARES, 600)
    assert sum(quotas.values()) == 600


def test_largest_remainder_handles_awkward_totals():
    for total in (7, 13, 99, 601):
        assert sum(largest_remainder(SHARES, total).values()) == total


def test_largest_remainder_never_starves_a_small_stratum():
    quotas = largest_remainder(SHARES, 600)
    assert all(count > 0 for count in quotas.values())


def test_largest_remainder_is_order_independent():
    reversed_shares = dict(reversed(list(SHARES.items())))
    assert largest_remainder(SHARES, 600) == largest_remainder(reversed_shares, 600)


def test_largest_remainder_of_zero_is_all_zero():
    assert set(largest_remainder(SHARES, 0).values()) == {0}


# =============================================================================
# Determinism
# =============================================================================


def test_selection_is_deterministic(pool):
    a = select_balanced(pool, 40, SHARES)
    b = select_balanced(pool, 40, SHARES)
    assert [e.id for e in a.selected] == [e.id for e in b.selected]


def test_selection_ignores_pool_ordering(pool):
    """File order is generation order; it must not influence the dataset."""
    a = select_balanced(pool, 40, SHARES)
    b = select_balanced(list(reversed(pool)), 40, SHARES)
    assert {e.id for e in a.selected} == {e.id for e in b.selected}


def test_a_different_seed_selects_differently(pool):
    a = select_balanced(pool, 40, SHARES, seed=1)
    b = select_balanced(pool, 40, SHARES, seed=2)
    assert [e.id for e in a.selected] != [e.id for e in b.selected]


def test_selection_returns_no_duplicates(pool):
    result = select_balanced(pool, 60, SHARES)
    ids = [e.id for e in result.selected]
    assert len(ids) == len(set(ids))


def test_every_selected_example_came_from_the_pool(pool):
    result = select_balanced(pool, 40, SHARES)
    pool_ids = {e.id for e in pool}
    assert all(e.id in pool_ids for e in result.selected)


# =============================================================================
# Distribution
# =============================================================================


def test_selection_approximates_the_target_shares(pool):
    result = select_balanced(pool, 60, SHARES)
    if result.shortfalls:
        pytest.skip("mock pool cannot fill every stratum")

    total = result.size
    for pressure, share in SHARES.items():
        got = sum(1 for e in result.selected
                  if e.scenario.pressure_type.value == pressure) / total
        assert abs(got - share) < 0.05, f"{pressure}: {got:.3f} vs {share}"


def test_shortfalls_are_reported_not_backfilled(pool):
    """A missing solved example must not be replaced by an easier normal one."""
    thin = [e for e in pool if e.scenario.pressure_type.value != "solved"]
    result = select_balanced(thin, 60, SHARES)

    assert "solved" in result.shortfalls
    assert result.shortfalls["solved"] > 0
    # The freed quota is not handed to another stratum.
    assert result.size == 60 - sum(result.shortfalls.values())


def test_selection_never_exceeds_the_target(pool):
    result = select_balanced(pool, 30, SHARES)
    assert result.size <= 30


def test_selection_spreads_bug_categories(pool):
    """Round-robin drawing must not collapse a stratum onto one bug category."""
    result = select_balanced(pool, 60, SHARES)
    categories = {e.scenario.bug_category for e in result.selected}
    pool_categories = {e.scenario.bug_category for e in pool}
    # At this size it should reach most of what the pool offers.
    assert len(categories) >= min(len(pool_categories), 5)


def test_result_records_the_seed_and_method(pool):
    result = select_balanced(pool, 20, SHARES)
    payload = result.to_dict()
    assert payload["seed"]
    assert "stratified" in payload["method"]
    assert payload["selected"] == result.size


def test_selecting_more_than_the_pool_takes_everything(pool):
    result = select_balanced(pool, 10_000, SHARES)
    assert result.size == len(pool)
    assert sum(result.shortfalls.values()) > 0
