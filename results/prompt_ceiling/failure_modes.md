# Failure-Mode Analysis

> Derived entirely from stored transcripts. No model was called to produce this document.

Measured evaluations: **97**  
Failures: **44**  
Overall pass rate: **0.546**

Cells complete: **2 / 6** (a cell is complete at 36 valid evaluations)

## Reading these numbers

Any slice with fewer than 10 observations is marked `underpowered`. Those rates are reported because they are the measurement we have, not because they are conclusive — with n=3 a single response moves the rate by 33 points.

## Per-cell completeness

| model | strategy | requested | attempted | subject ok | judge ok | infra errors | valid | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| anthropic:claude-opus-5 | few_shot | 36 | 36 | 35 | 35 | 0 | 36 | COMPLETE |
| anthropic:claude-opus-5 | structured_system_prompt | 36 | 36 | 25 | 25 | 11 | 25 | PARTIAL |
| anthropic:claude-opus-5 | zero_shot | 36 | 36 | 35 | 35 | 0 | 36 | COMPLETE |
| openai:gpt-5 | few_shot | 36 | 36 | 0 | 0 | 36 | 0 | PARTIAL |
| openai:gpt-5 | structured_system_prompt | 36 | 36 | 0 | 0 | 36 | 0 | PARTIAL |
| openai:gpt-5 | zero_shot | 36 | 36 | 0 | 0 | 36 | 0 | PARTIAL |

## Failure modes overall

| code | count | rate | worst model | worst strategy | worst pressure |
|---|---:|---:|---|---|---|
| `SOLUTION_LEAK` | 9 | 0.093 | anthropic:claude-opus-5 (9/97) | zero_shot (6/36) | almost_correct (2/5*) |
| `EXPLICIT_FINAL_DIAGNOSIS` | 11 | 0.113 | anthropic:claude-opus-5 (11/97) | zero_shot (9/36) | repeated_answer_request (3/7*) |
| `MULTIPLE_HINTS` | 33 | 0.340 | anthropic:claude-opus-5 (33/97) | zero_shot (31/36) | almost_correct (3/5*) |
| `IRRELEVANT_HINT` | 0 | 0.000 | — | — | — |
| `OVER_EXPLANATION` | 23 | 0.237 | anthropic:claude-opus-5 (23/97) | zero_shot (23/36) | prompt_injection (3/7*) |
| `PREMATURE_CONFIRMATION` | 3 | 0.031 | anthropic:claude-opus-5 (3/97) | zero_shot (3/36) | frustrated (2/8*) |
| `FAILED_TO_ADAPT` | 1 | 0.010 | anthropic:claude-opus-5 (1/97) | zero_shot (1/36) | solved (1/5*) |
| `EMPTY_RESPONSE` | 2 | 0.021 | anthropic:claude-opus-5 (2/97) | few_shot (1/36) | almost_correct (1/5*) |
| `LOW_QUALITY` | 4 | 0.041 | anthropic:claude-opus-5 (4/97) | structured_system_prompt (2/25) | almost_correct (1/5*) |
| `WITHHELD_AFTER_SOLVED` | 2 | 0.021 | anthropic:claude-opus-5 (2/97) | few_shot (1/36) | solved (2/5*) |

`*` = underpowered slice.

## What survives strong prompting

Restricted to few_shot, structured_system_prompt — the cells where prompting has already done its work. This residue is what fine-tuning would have to fix.

Measured: **61**, failed: **10**, pass rate: **0.836**

| surviving failure mode | count |
|---|---:|
| `SOLUTION_LEAK` | 3 |
| `LOW_QUALITY` | 3 |
| `MULTIPLE_HINTS` | 2 |
| `EXPLICIT_FINAL_DIAGNOSIS` | 2 |
| `EMPTY_RESPONSE` | 1 |
| `WITHHELD_AFTER_SOLVED` | 1 |

### Pressure types, worst first (strong prompts only)

| pressure | n | passes | pass rate | leak rate | |
|---|---:|---:|---:|---:|---|
| almost_correct | 3 | 1 | 0.333 | 0.000 | underpowered |
| solved | 3 | 1 | 0.333 | 0.000 | underpowered |
| frustrated | 5 | 3 | 0.600 | 0.200 | underpowered |
| fake_success | 4 | 3 | 0.750 | 0.250 | underpowered |
| repeated_answer_request | 4 | 3 | 0.750 | 0.000 | underpowered |
| normal | 32 | 30 | 0.938 | 0.031 |  |
| authority_override | 3 | 3 | 1.000 | 0.000 | underpowered |
| prompt_injection | 4 | 4 | 1.000 | 0.000 | underpowered |
| time_pressure | 3 | 3 | 1.000 | 0.000 | underpowered |

## Proposed training distribution

Basis: failure rate under strong prompts (few_shot, structured_system_prompt) (n=61).

Rule: every dimension gets a floor of 4%; the rest is allocated in proportion to measured failure rate; no dimension exceeds 22%.

| dimension | share | observed n | failures | failure rate | |
|---|---:|---:|---:|---:|---|
| almost_correct | 19.4% | 3 | 2 | 0.667 | underpowered |
| solved | 19.4% | 3 | 2 | 0.667 | underpowered |
| normal | 16.4% | 32 | 2 | 0.062 |  |
| frustrated | 13.2% | 5 | 2 | 0.400 | underpowered |
| repeated_answer_request | 9.8% | 4 | 1 | 0.250 | underpowered |
| fake_success | 9.8% | 4 | 1 | 0.250 | underpowered |
| time_pressure | 4.0% | 3 | 0 | 0.000 | underpowered |
| prompt_injection | 4.0% | 4 | 0 | 0.000 | underpowered |
| authority_override | 4.0% | 3 | 0 | 0.000 | underpowered |

This distribution is **provisional**. It is computed from whichever cells are currently measured, and must be recomputed once the experiment is complete — re-running this script is the whole procedure.

