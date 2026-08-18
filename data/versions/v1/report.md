# Dataset `v1`

Teacher: `unknown`  
Judge: `anthropic:claude-opus-5`  
Dataset hash: `af9aae8eec6db87c`

## Funnel

| metric | count |
| --- | --- |
| candidates | 1190 |
| accepted | 1055 |
| rejected | 135 |
| acceptance rate | 88.7% |

## Rejections by stage

| stage | rejected |
| --- | --- |
| llm_judge | 104 |
| static_checks | 31 |
| dedupe | 0 |
| contamination | 0 |
| balance | 0 |

## Rejections by reason

| reason | count |
| --- | --- |
| LOW_QUALITY | 104 |
| SOLUTION_LEAK | 17 |
| MULTIPLE_HINTS | 10 |
| EXPLICIT_FINAL_DIAGNOSIS | 3 |
| PREMATURE_CONFIRMATION | 1 |
| IRRELEVANT_HINT | 1 |

## Accepted distribution

### Language

| bucket | count |
| --- | --- |
| javascript | 529 |
| python | 526 |

### Bug category

| bucket | count |
| --- | --- |
| scope | 78 |
| shadowed_builtin | 52 |
| incorrect_condition | 51 |
| type_coercion | 49 |
| async_await | 45 |
| map_vs_foreach | 43 |
| integer_division | 43 |
| boolean_condition | 42 |
| exception_handling | 42 |
| this_binding | 41 |
| return_placement | 40 |
| list_mutation | 40 |
| closure_behavior | 38 |
| string_immutability | 37 |
| dictionary_access | 37 |
| undefined_properties | 37 |
| callback_ordering | 36 |
| mutable_default | 35 |
| generator_exhaustion | 34 |
| hoisting | 33 |
| missing_return | 32 |
| shallow_copy | 31 |
| array_mutation | 30 |
| comparison_identity | 29 |
| promise_handling | 28 |
| loop_boundary | 27 |
| none_handling | 25 |

### Pressure type

| bucket | count |
| --- | --- |
| normal | 204 |
| solved | 162 |
| almost_correct | 129 |
| fake_success | 107 |
| time_pressure | 105 |
| frustrated | 99 |
| repeated_answer_request | 95 |
| prompt_injection | 80 |
| authority_override | 74 |

### Difficulty

| bucket | count |
| --- | --- |
| medium | 366 |
| easy | 352 |
| hard | 337 |

### Conversation length (learner turns)

| bucket | count |
| --- | --- |
| 1 | 216 |
| 2 | 379 |
| 3 | 298 |
| 4 | 162 |

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

Dataset V1 resumed filtering; solved-state bound corrected
