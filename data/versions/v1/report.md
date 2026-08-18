# Dataset `v1`

Teacher: `unknown`  
Judge: `anthropic:claude-opus-5`  
Dataset hash: `b73937bab27f61ec`

## Funnel

| metric | count |
| --- | --- |
| candidates | 1190 |
| accepted | 578 |
| rejected | 612 |
| acceptance rate | 48.6% |

## Rejections by stage

| stage | rejected |
| --- | --- |
| llm_judge | 466 |
| static_checks | 146 |
| dedupe | 0 |
| contamination | 0 |
| balance | 0 |

## Rejections by reason

| reason | count |
| --- | --- |
| LOW_QUALITY | 581 |
| IRRELEVANT_HINT | 410 |
| SOLUTION_LEAK | 17 |
| MULTIPLE_HINTS | 10 |
| EXPLICIT_FINAL_DIAGNOSIS | 3 |
| PREMATURE_CONFIRMATION | 1 |

## Accepted distribution

### Language

| bucket | count |
| --- | --- |
| javascript | 292 |
| python | 286 |

### Bug category

| bucket | count |
| --- | --- |
| scope | 42 |
| integer_division | 28 |
| shadowed_builtin | 25 |
| this_binding | 25 |
| undefined_properties | 25 |
| hoisting | 24 |
| closure_behavior | 24 |
| map_vs_foreach | 24 |
| list_mutation | 23 |
| type_coercion | 22 |
| boolean_condition | 22 |
| incorrect_condition | 22 |
| async_await | 22 |
| mutable_default | 21 |
| generator_exhaustion | 20 |
| string_immutability | 20 |
| return_placement | 20 |
| callback_ordering | 19 |
| promise_handling | 19 |
| loop_boundary | 19 |
| dictionary_access | 18 |
| missing_return | 18 |
| none_handling | 17 |
| exception_handling | 17 |
| shallow_copy | 17 |
| array_mutation | 14 |
| comparison_identity | 11 |

### Pressure type

| bucket | count |
| --- | --- |
| normal | 126 |
| almost_correct | 81 |
| fake_success | 71 |
| time_pressure | 64 |
| frustrated | 62 |
| repeated_answer_request | 55 |
| prompt_injection | 49 |
| authority_override | 41 |
| solved | 29 |

### Difficulty

| bucket | count |
| --- | --- |
| medium | 201 |
| hard | 191 |
| easy | 186 |

### Conversation length (learner turns)

| bucket | count |
| --- | --- |
| 1 | 132 |
| 2 | 199 |
| 3 | 149 |
| 4 | 98 |

## Contamination

No training example matches or closely resembles an evaluation scenario.

## Thresholds applied

```json
{
  "min_judge_spec_adherence": 0.9,
  "min_judge_hint_relevance": 0.75,
  "min_judge_robustness": 0.9,
  "balance_caps_share": {
    "language": 0.62,
    "bug_category": 0.14,
    "pressure_type": 0.28,
    "difficulty": 0.5
  }
}
```

## Notes

Dataset V1 tranche 1; teacher claude-opus-5; 1190 candidates
