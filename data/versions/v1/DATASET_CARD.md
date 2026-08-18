# Dataset v1 — Socratic Debug Tutor

**Status: BUILT, AUDITED and FROZEN. No model has been trained on it.**

| | |
| --- | --- |
| Version | `v1` |
| Accepted examples | **600** |
| Candidates generated | 1190 |
| Rejected | 135 |
| Acceptance rate | 88.7% |
| Dataset hash | `9121c24e47c7253818040aa40356a67d` |
| Behavior spec | `1.0.0` (`dc14f40b94d622d1`) |
| Teacher | `anthropic:claude-opus-5` (`claude-opus-5`) |
| Generation prompt | `1.0.0` (`6dc7c82da5780f61`) |
| Git commit | `963290efc078` |

## Purpose

Teach a small model to give Socratic debugging guidance: help a learner find a
bug themselves, without revealing the solution before they have solved it — and
to recognize when they *have* solved it and confirm plainly.

The behavior has two halves, and a dataset that taught only the first would be
actively harmful:

```
UNRESOLVED  ->  withhold the solution, ask exactly one useful question
SOLVED      ->  recognize the correct fix, confirm it, explain why it works
```

## Behavior Spec

Every example was judged against behavior spec
`1.0.0` (`dc14f40b94d622d1`) — the
same spec, and the same criteria, the evaluation harness applies to a model
response. Training data and evaluation therefore cannot drift apart.

## Why this dataset exists

The completed Prompt-Ceiling Ablation measured whether prompting alone suffices:
two frontier model families x three prompting strategies x 36 scenarios,
216/216 valid evaluations, zero infrastructure errors. **No configuration
cleared the predeclared thresholds** (adherence / robustness / pass rate
>= 0.95). The strongest cell, `openai:gpt-5` + `structured_system_prompt`,
reached adherence 0.874, robustness 0.902, pass rate 0.889 — and **all four of
its failures were `SOLUTION_LEAK`**, clustered under answer-seeking pressure.

That residual is what this dataset targets. The pressure mix below is not a
guess: it is computed from where strong prompting actually failed.

## Generation

Scenario selection is **controlled by the generator, not the teacher**. A
deterministic seeded plan walks the dimension space — language x bug category x
difficulty x pressure type x learner competence x conversation length x hint
strength x student progress — and the teacher is asked only to write the ideal
tutoring interaction for the point it is handed. Asking a teacher for "1200
examples" would have produced mostly easy Python loop bugs under no pressure.

Generation is seeded and resumable. A candidate's identity is its plan index, so
an interrupted run resumes without re-purchasing completed work and without
creating two records for one plan point. Infrastructure failures are never
persisted, so they are retried rather than mistaken for quality rejections.

## Filtering

Every accepted example passed the full quality gate. Each stage can only reject:

```
schema -> static checks -> LLM judge -> dedupe -> contamination -> balance
```

The order is deliberate: cheap deterministic stages run before the paid judge,
dedupe runs *after* judging so the survivor of a near-duplicate pair is one that
passed, and balancing runs last on already-good data. **Static silence proves
nothing** — deterministic checks can fail a response but never rescue one the
judge rejected.

## Distribution

### Pressure type

| pressure type | count | share |
| --- | ---: | ---: |
| `normal` | 114 | 19.0% |
| `solved` | 85 | 14.2% |
| `almost_correct` | 73 | 12.2% |
| `time_pressure` | 61 | 10.2% |
| `frustrated` | 57 | 9.5% |
| `fake_success` | 57 | 9.5% |
| `repeated_answer_request` | 57 | 9.5% |
| `prompt_injection` | 48 | 8.0% |
| `authority_override` | 48 | 8.0% |

### Language

| language | count | share |
| --- | ---: | ---: |
| `javascript` | 304 | 50.7% |
| `python` | 296 | 49.3% |

### Difficulty

| difficulty | count | share |
| --- | ---: | ---: |
| `easy` | 212 | 35.3% |
| `medium` | 201 | 33.5% |
| `hard` | 187 | 31.2% |

### Conversation length (prior turns)

| prior turns | count | share |
| --- | ---: | ---: |
| `2` | 221 | 36.8% |
| `3` | 159 | 26.5% |
| `1` | 121 | 20.2% |
| `4` | 99 | 16.5% |

### Bug categories

27 distinct categories. Top 12:

| bug category | count | share |
| --- | ---: | ---: |
| `async_await` | 28 | 4.7% |
| `callback_ordering` | 27 | 4.5% |
| `array_mutation` | 27 | 4.5% |
| `closure_behavior` | 26 | 4.3% |
| `boolean_condition` | 26 | 4.3% |
| `exception_handling` | 25 | 4.2% |
| `dictionary_access` | 24 | 4.0% |
| `integer_division` | 24 | 4.0% |
| `incorrect_condition` | 23 | 3.8% |
| `scope` | 22 | 3.7% |
| `string_immutability` | 22 | 3.7% |
| `comparison_identity` | 22 | 3.7% |

## Behavioral coverage

| learner state | count | share |
| --- | ---: | ---: |
| unresolved | 442 | 73.7% |
| almost correct | 73 | 12.2% |
| solved | 85 | 14.2% |

| pressure family | count | share |
| --- | ---: | ---: |
| answer-seeking (`repeated_answer_request`, `time_pressure`, `frustrated`) | 175 | 29.2% |
| injection / authority | 96 | 16.0% |

`solved` coverage is load-bearing. Without it the dataset would teach "never
confirm an answer under any circumstances", which is not the behavior. The
behavior is "do not reveal the answer *before* the learner solves it".

## Rejections

135 candidates were rejected and **kept**, with their reasons,
in `rejected.jsonl`. The rejection pile is the evidence for how aggressively the
data was filtered, and the first place to look if a trained model misbehaves.

| rejection reason | count | share |
| --- | ---: | ---: |
| `LOW_QUALITY` | 104 | 76.5% |
| `SOLUTION_LEAK` | 17 | 12.5% |
| `MULTIPLE_HINTS` | 10 | 7.4% |
| `EXPLICIT_FINAL_DIAGNOSIS` | 3 | 2.2% |
| `PREMATURE_CONFIRMATION` | 1 | 0.7% |
| `IRRELEVANT_HINT` | 1 | 0.7% |

`SOLUTION_LEAK` rejections: **17**. `WITHHELD_AFTER_SOLVED`
rejections: **0**. Those are the two failures the behavior is
defined by, caught in training data by the same codes the evaluator uses.

## Diversity

| | |
| --- | --- |
| Exact duplicates in accepted | 0 |
| Near duplicates in accepted | 0 |
| Unique content hashes | 600 / 600 |
| Unique code bodies | 599 / 600 |
| Distinct bug categories | 27 |

Recomputed over the accepted set as an independent audit of the gate's dedupe
stage, rather than copied from the gate's own bookkeeping.

## Contamination

No training example matches or closely resembles an evaluation scenario.

Checked against all 56
evaluation scenarios across `clean`, `adversarial` and `heldout`. Exact
overlaps: 0. Near overlaps:
0.

## Nested subsets (prepared, NOT trained)

Sizes [125, 250, 500, 600]. Nesting verified programmatically:
**True**. Each subset is a prefix of one fixed
content-hash-ordered shuffle, so a smaller subset is literally contained in
every larger one. 

## Known limitations

1. **Synthetic learners.** Every learner turn was written by a model imitating a
   stuck, frustrated or hurried student. Real learners are messier.
2. **The teacher's ceiling is the dataset's ceiling.** These are filtered
   frontier outputs, so a tuned model inherits the teacher's blind spots. The
   gate can only reject; it cannot create quality that was never generated.
3. **The judge is an LLM and its agreement with humans is not established.**
   `human_review.csv` is exported and **not yet graded**. Until it is, every
   acceptance decision rests on an unvalidated judge.
4. **Self-judging.** Teacher and judge are the same model family, so a
   systematic blind spot could pass its own output unchallenged.
5. **Two languages, finite bug taxonomy.** Python and JavaScript only, across
   27 bug categories. Nothing here supports claims
   about other languages or unusual defect classes.
6. **Small per-cell counts.** Spread across many dimension combinations, any
   individual cell holds few examples.
7. **No training claim.** Nothing has been trained on this data. This card
   describes data quality only.

## What this dataset does NOT claim

It does not claim that fine-tuning improves the model, because no model has been
trained. The strongest claim currently supported is that the data was generated
against a measured failure distribution and survived a documented quality gate.
