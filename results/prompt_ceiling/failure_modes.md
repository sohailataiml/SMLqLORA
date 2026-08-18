# Failure-Mode Analysis

> Derived entirely from stored transcripts. No model was called to produce this document.

Measured evaluations: **216**  
Failures: **114**  
Overall pass rate: **0.472**

Cells complete: **6 / 6** (a cell is complete at 36 valid evaluations)

## Reading these numbers

Any slice with fewer than 10 observations is marked `underpowered`. Those rates are reported because they are the measurement we have, not because they are conclusive — with n=3 a single response moves the rate by 33 points.

## Per-cell completeness

| model | strategy | requested | attempted | subject ok | judge ok | infra errors | valid | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| anthropic:claude-opus-5 | few_shot | 36 | 36 | 35 | 35 | 0 | 36 | COMPLETE |
| anthropic:claude-opus-5 | structured_system_prompt | 36 | 36 | 36 | 36 | 0 | 36 | COMPLETE |
| anthropic:claude-opus-5 | zero_shot | 36 | 36 | 35 | 35 | 0 | 36 | COMPLETE |
| openai:gpt-5 | few_shot | 36 | 36 | 36 | 36 | 0 | 36 | COMPLETE |
| openai:gpt-5 | structured_system_prompt | 36 | 36 | 36 | 36 | 0 | 36 | COMPLETE |
| openai:gpt-5 | zero_shot | 36 | 36 | 36 | 36 | 0 | 36 | COMPLETE |

## Failure modes overall

| code | count | rate | worst model | worst strategy | worst pressure |
|---|---:|---:|---|---|---|
| `SOLUTION_LEAK` | 24 | 0.111 | openai:gpt-5 (15/108) | zero_shot (16/72) | almost_correct (4/12) |
| `EXPLICIT_FINAL_DIAGNOSIS` | 17 | 0.079 | anthropic:claude-opus-5 (12/108) | zero_shot (12/72) | repeated_answer_request (6/18) |
| `MULTIPLE_HINTS` | 92 | 0.426 | openai:gpt-5 (59/108) | zero_shot (63/72) | almost_correct (7/12) |
| `IRRELEVANT_HINT` | 0 | 0.000 | — | — | — |
| `OVER_EXPLANATION` | 30 | 0.139 | anthropic:claude-opus-5 (23/108) | zero_shot (30/72) | almost_correct (3/12) |
| `PREMATURE_CONFIRMATION` | 3 | 0.014 | anthropic:claude-opus-5 (3/108) | zero_shot (3/72) | frustrated (2/18) |
| `FAILED_TO_ADAPT` | 2 | 0.009 | anthropic:claude-opus-5 (2/108) | structured_system_prompt (1/72) | solved (2/12) |
| `EMPTY_RESPONSE` | 2 | 0.009 | anthropic:claude-opus-5 (2/108) | few_shot (1/72) | almost_correct (1/12) |
| `INCORRECT_DIAGNOSIS` | 2 | 0.009 | anthropic:claude-opus-5 (1/108) | structured_system_prompt (1/72) | solved (2/12) |
| `LOW_QUALITY` | 4 | 0.018 | anthropic:claude-opus-5 (4/108) | structured_system_prompt (2/72) | almost_correct (1/12) |
| `WITHHELD_AFTER_SOLVED` | 6 | 0.028 | anthropic:claude-opus-5 (3/108) | few_shot (3/72) | solved (6/12) |

`*` = underpowered slice.

## What survives strong prompting

Restricted to few_shot, structured_system_prompt — the cells where prompting has already done its work. This residue is what fine-tuning would have to fix.

Measured: **144**, failed: **45**, pass rate: **0.688**

| surviving failure mode | count |
|---|---:|
| `MULTIPLE_HINTS` | 29 |
| `SOLUTION_LEAK` | 8 |
| `EXPLICIT_FINAL_DIAGNOSIS` | 5 |
| `WITHHELD_AFTER_SOLVED` | 4 |
| `LOW_QUALITY` | 3 |
| `EMPTY_RESPONSE` | 1 |
| `FAILED_TO_ADAPT` | 1 |
| `INCORRECT_DIAGNOSIS` | 1 |

### Pressure types, worst first (strong prompts only)

| pressure | n | passes | pass rate | leak rate | |
|---|---:|---:|---:|---:|---|
| solved | 8 | 3 | 0.375 | 0.000 | underpowered |
| almost_correct | 8 | 4 | 0.500 | 0.125 | underpowered |
| time_pressure | 8 | 5 | 0.625 | 0.125 | underpowered |
| fake_success | 12 | 8 | 0.667 | 0.083 |  |
| frustrated | 12 | 8 | 0.667 | 0.167 |  |
| repeated_answer_request | 12 | 8 | 0.667 | 0.083 |  |
| authority_override | 8 | 6 | 0.750 | 0.000 | underpowered |
| normal | 64 | 48 | 0.750 | 0.031 |  |
| prompt_injection | 12 | 9 | 0.750 | 0.000 |  |

## Proposed training distribution

Basis: failure rate under strong prompts (few_shot, structured_system_prompt) (n=144).

Rule: every dimension gets a floor of 4%; the rest is allocated in proportion to measured failure rate; no dimension exceeds 22%.

| dimension | share | observed n | failures | failure rate | |
|---|---:|---:|---:|---:|---|
| normal | 19.1% | 64 | 16 | 0.250 |  |
| solved | 14.2% | 8 | 5 | 0.625 | underpowered |
| almost_correct | 12.2% | 8 | 4 | 0.500 | underpowered |
| time_pressure | 10.1% | 8 | 3 | 0.375 | underpowered |
| frustrated | 9.4% | 12 | 4 | 0.333 |  |
| repeated_answer_request | 9.4% | 12 | 4 | 0.333 |  |
| fake_success | 9.4% | 12 | 4 | 0.333 |  |
| prompt_injection | 8.1% | 12 | 3 | 0.250 |  |
| authority_override | 8.1% | 8 | 2 | 0.250 | underpowered |

This distribution is **provisional**. It is computed from whichever cells are currently measured, and must be recomputed once the experiment is complete — re-running this script is the whole procedure.

