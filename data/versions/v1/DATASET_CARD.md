# Dataset v1 — Socratic Debug Tutor

**Status: BUILT, AUDITED and FROZEN. No model has been trained on it.**

| | |
| --- | --- |
| Version | `v1` |
| Accepted examples | **578** |
| Candidates generated | 1190 |
| Rejected | 202 |
| Acceptance rate | 48.6% |
| Dataset hash | `21ad0b83aac539f0a4c2f0fdfb67c7c0` |
| Behavior spec | `1.0.0` (`dc14f40b94d622d1`) |
| Teacher | `anthropic:claude-opus-5` (`claude-opus-5`) |
| Generation prompt | `1.0.0` (`6dc7c82da5780f61`) |
| Git commit | `5e440e0379e2` |

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
| `normal` | 126 | 21.8% |
| `almost_correct` | 81 | 14.0% |
| `fake_success` | 71 | 12.3% |
| `time_pressure` | 64 | 11.1% |
| `frustrated` | 62 | 10.7% |
| `repeated_answer_request` | 55 | 9.5% |
| `prompt_injection` | 49 | 8.5% |
| `authority_override` | 41 | 7.1% |
| `solved` | 29 | 5.0% |

### Language

| language | count | share |
| --- | ---: | ---: |
| `javascript` | 292 | 50.5% |
| `python` | 286 | 49.5% |

### Difficulty

| difficulty | count | share |
| --- | ---: | ---: |
| `medium` | 201 | 34.8% |
| `hard` | 191 | 33.0% |
| `easy` | 186 | 32.2% |

### Conversation length (prior turns)

| prior turns | count | share |
| --- | ---: | ---: |
| `2` | 199 | 34.4% |
| `3` | 149 | 25.8% |
| `1` | 132 | 22.8% |
| `4` | 98 | 17.0% |

### Bug categories

27 distinct categories. Top 12:

| bug category | count | share |
| --- | ---: | ---: |
| `scope` | 42 | 7.3% |
| `integer_division` | 28 | 4.8% |
| `shadowed_builtin` | 25 | 4.3% |
| `this_binding` | 25 | 4.3% |
| `undefined_properties` | 25 | 4.3% |
| `hoisting` | 24 | 4.2% |
| `closure_behavior` | 24 | 4.2% |
| `map_vs_foreach` | 24 | 4.2% |
| `list_mutation` | 23 | 4.0% |
| `type_coercion` | 22 | 3.8% |
| `boolean_condition` | 22 | 3.8% |
| `incorrect_condition` | 22 | 3.8% |

## Behavioral coverage

| learner state | count | share |
| --- | ---: | ---: |
| unresolved | 468 | 81.0% |
| almost correct | 81 | 14.0% |
| solved | 29 | 5.0% |

| pressure family | count | share |
| --- | ---: | ---: |
| answer-seeking (`repeated_answer_request`, `time_pressure`, `frustrated`) | 181 | 31.3% |
| injection / authority | 90 | 15.6% |

`solved` coverage is load-bearing. Without it the dataset would teach "never
confirm an answer under any circumstances", which is not the behavior. The
behavior is "do not reveal the answer *before* the learner solves it".

## Rejections

202 candidates were rejected and **kept**, with their reasons,
in `rejected.jsonl`. The rejection pile is the evidence for how aggressively the
data was filtered, and the first place to look if a trained model misbehaves.

| rejection reason | count | share |
| --- | ---: | ---: |
| `LOW_QUALITY` | 171 | 84.7% |
| `SOLUTION_LEAK` | 17 | 8.4% |
| `MULTIPLE_HINTS` | 10 | 5.0% |
| `EXPLICIT_FINAL_DIAGNOSIS` | 3 | 1.5% |
| `PREMATURE_CONFIRMATION` | 1 | 0.5% |

`SOLUTION_LEAK` rejections: **17**. `WITHHELD_AFTER_SOLVED`
rejections: **0**. Those are the two failures the behavior is
defined by, caught in training data by the same codes the evaluator uses.

## Diversity

| | |
| --- | --- |
| Exact duplicates in accepted | 0 |
| Near duplicates in accepted | 0 |
| Unique content hashes | 578 / 578 |
| Unique code bodies | 574 / 578 |
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

Sizes [125, 250, 500, 578]. Nesting verified programmatically:
**True**. Each subset is a prefix of one fixed
content-hash-ordered shuffle, so a smaller subset is literally contained in
every larger one. largest checkpoint adapted from 600 to 578 because only 578 examples were accepted; no example was duplicated to reach the target

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
