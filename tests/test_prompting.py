"""Prompt strategy rendering and versioning."""

from __future__ import annotations

import pytest

from evaluation.schemas import Role
from prompting.strategies import (
    STRATEGIES,
    FewShotStrategy,
    StructuredStrategy,
    ZeroShotStrategy,
    all_strategies,
    get_strategy,
    render_conversation,
)


def test_three_distinct_strategies_are_registered():
    assert set(STRATEGIES) == {"zero_shot", "few_shot", "structured_system_prompt"}


def test_strategies_have_distinct_system_prompts():
    prompts = {s.name: s.system_prompt() for s in all_strategies()}
    assert len(set(prompts.values())) == 3


def test_prompt_hashes_are_distinct_and_stable():
    hashes = {s.name: s.prompt_hash() for s in all_strategies()}
    assert len(set(hashes.values())) == 3
    assert hashes == {s.name: s.prompt_hash() for s in all_strategies()}


def test_effort_increases_across_strategies():
    zero = len(ZeroShotStrategy().system_prompt())
    few = len(FewShotStrategy().system_prompt())
    structured = len(StructuredStrategy().system_prompt())
    assert zero < few
    assert zero < structured


def test_structured_prompt_carries_the_behavior_spec(spec):
    prompt = StructuredStrategy(spec).system_prompt()
    for criterion in spec.criteria:
        assert criterion.id in prompt
    assert "EXACTLY ONE MOVE" in prompt
    assert "COUNTER-PRESSURE POLICY" in prompt


def test_structured_prompt_addresses_each_pressure_type(spec):
    prompt = StructuredStrategy(spec).system_prompt().lower()
    for phrase in ["just give me the answer", "instructor", "ignore the tutoring rules",
                   "30 seconds", "i fixed it"]:
        assert phrase in prompt


def test_few_shot_includes_solved_and_adversarial_exemplars():
    prompt = FewShotStrategy().system_prompt()
    assert prompt.count("--- Example") == 6
    assert "instructor" in prompt.lower()
    assert "Fixed it!" in prompt


def test_render_attaches_code_to_the_first_user_turn(unsolved_scenario):
    rendered = ZeroShotStrategy().render(unsolved_scenario)
    assert len(rendered.messages) == 1
    assert "```python" in rendered.messages[0].content
    assert unsolved_scenario.student_message in rendered.messages[0].content


def test_render_multi_turn_keeps_history_and_shows_code_once(solved_scenario):
    messages = render_conversation(solved_scenario)
    assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert messages[0].content.count("```python") == 1
    assert "```" not in messages[-1].content
    assert messages[-1].content == solved_scenario.student_message


def test_rendered_prompt_records_provenance(unsolved_scenario):
    rendered = StructuredStrategy().render(unsolved_scenario)
    assert rendered.strategy == "structured_system_prompt"
    assert rendered.version
    assert len(rendered.sha256) == 64


def test_rendered_messages_satisfy_the_adapter_contract(unsolved_scenario, solved_scenario):
    from models.adapters import ModelAdapter

    for scenario in (unsolved_scenario, solved_scenario):
        for strategy in all_strategies():
            messages = strategy.render(scenario).messages
            ModelAdapter._validate_messages(messages)  # raises if malformed


def test_unknown_strategy_lists_the_available_ones():
    with pytest.raises(KeyError, match="zero_shot"):
        get_strategy("chain_of_thought")


def test_describe_is_reportable():
    described = get_strategy("few_shot").describe()
    assert set(described) == {"strategy", "version", "description", "sha256"}
