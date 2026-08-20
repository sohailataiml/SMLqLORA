# Failure analysis — N=600, Dataset V1, epoch-3 checkpoint

> **Finding: the diagnostic regression cannot currently be attributed to Dataset
> V1.** Every hypothesis that blames the data is refuted by the frozen data
> itself, and 17 of the 20 held-out outputs carry phrasing that appears in none
> of V1's 600 tutor responses. Designing Dataset V2 against this evidence would
> be designing against a training artifact.

Reproduce with:

```bash
python -m analysis.failure_taxonomy --write
```

Machine-readable: [`v1_n600_failure_taxonomy.json`](v1_n600_failure_taxonomy.json).
Source: the 40 committed judge transcripts at
`results/base_vs_tuned/judge_transcripts.jsonl`, unmodified.

---

## What was measured

20 held-out scenarios, tuned model, 5 passes and 15 failures. The MVP report
attributed the failures to "learned the policy, not the competence" — the model
withholds correctly but misdiagnoses. That description is accurate as a
description. This analysis asks the next question: **is Dataset V1 the cause?**

## Three markers Dataset V1 cannot produce

| marker | tuned outputs | Dataset V1 |
| --- | ---: | ---: |
| verbatim sentence repetition | 1 / 20 | **0 / 600** |
| tutor claims its own prior work on a first turn | 4 / 20 | **0 / 600** |
| repeats a phrase absent from all V1 responses | 16 / 20 | — |

The third is the strongest, because it is derived rather than probed. Taking
every 5-gram in the tuned outputs and subtracting every 5-gram in V1's 600 tutor
responses leaves **10 phrases the model uses across three or more outputs and
that occur nowhere in its training data**:

| times | phrase |
| ---: | --- |
| 8 | `so the problem is not` |
| 8 | `the problem is not the` |
| 7 | `let's look at the two` |
| 4 | `good observation let's look at` |
| 4 | `good so the problem is` |
| 4 | `i've already confirmed that the` |
| 3 | `and what is the value` |
| 3 | `going to give you the` |
| 3 | `to give you the answer` |

A fine-tuned model's habits are supposed to come from its fine-tuning set. These
did not.

**These markers appear on passes too** — on 4 of the 5. So they do not predict
*which* scenario fails. What they establish is broader and more damaging to the
measurement: **17 of 20 outputs are generated from a template attractor that does
not exist in the training distribution**, and 13 of the 15 failures are among
them. This run therefore cannot be read as a measurement of what Dataset V1
teaches.

Three outputs carry no marker at all: `js_heldout_authority_override_equality`
(a pass), `js_heldout_parseint_radix` (`MULTIPLE_HINTS`,
`PREMATURE_CONFIRMATION`, `LOW_QUALITY`) and
`py_heldout_repeated_request_set_order` (`IRRELEVANT_HINT`, `LOW_QUALITY`, hint
relevance 0.10). Those two failures are the only ones this analysis cannot
attribute to the checkpoint.

## Four hypotheses blaming Dataset V1 — all refuted

### H1 — V1 under-represents the first turn · **REFUTED**

Real mismatch: V1 is **20.2%** first-turn (121/600); the held-out set is
**75%** first-turn (15/20). It is the one genuine distribution gap in the data.

It predicts the opposite of what happened:

| | n | passes | pass rate | hint relevance | spec adherence |
| --- | ---: | ---: | ---: | ---: | ---: |
| first turn (under-represented) | 15 | 5 | **0.333** | 0.448 | 0.521 |
| multi-turn (over-represented) | 5 | 0 | **0.000** | 0.290 | 0.270 |

The model is *stronger* on the slice V1 barely covers and fails **every**
multi-turn scenario, which is where 79.8% of V1 lives. Adding first-turn
examples would target the half that already works.

### H2 — the quality gate rewarded vague-but-safe hints · **REFUTED**

Accepted V1 examples: mean hint relevance **0.950**, minimum **0.82**, against a
gate floor of 0.75. Only **14** of 600 fall below 0.90, and none fall below the floor. The rejected pool confirms
the gate was doing its job — 104 rejections carry `LOW_QUALITY`.

There is no population of vague non-leaking examples in V1 to replace.

### H3 — the tested bug categories are thin in V1 · **REFUTED**

All 13 categories the held-out set touches have **19–27** V1 examples across 27
categories. Coverage is even. There is no gap to fill.

### H4 — the stock phrasings were absorbed from V1 · **REFUTED**

| probe | in V1 (600) | in tuned (20) |
| --- | ---: | ---: |
| `not the … itself` | 0 (0.0%) | 10 (50%) |
| `so the problem is` | 3 (0.5%) | 10 (50%) |
| `look at the two` | 1 (0.2%) | 7 (35%) |
| `I've already confirmed` | **0 (0.0%)** | 4 (20%) |
| `Good observation` | 2 (0.3%) | 6 (30%) |

## What the evidence does point at

The training run's own validation curve, recorded in
`results/base_vs_tuned/RESULTS_SUMMARY.json`:

| | eval loss | token accuracy | entropy |
| --- | ---: | ---: | ---: |
| epoch 1 | **1.97** | 0.584 | 3.02 |
| epoch 2 | 2.785 | 0.514 | 5.15 |
| epoch 3 | 2.730 | 0.515 | 4.85 |

The evaluated adapter is epoch 3. Collapse onto a small set of stock phrases,
fabricated prior context, and one output that repeats a sentence 17 times are the
behaviours of an overtrained adapter, and they line up with a validation loss
that rose 39% after epoch 1 and never recovered.

**This is a diagnosis of the measurement, not of the data.** It says the current
numbers are a lower bound on N=600 and cannot ground a data intervention. It does
not say Dataset V1 is adequate — that remains unmeasured.

## Consequence for Dataset V2

The Early Submission requires a data change addressing a diagnosed failure. No
data-attributable failure has survived scrutiny. The honest sequence is:

1. Re-train N=600 with best-checkpoint selection (a correctness fix, not the V2
   intervention, and not claimable as the Early Submission improvement).
2. Re-evaluate to establish a canonical `N600_V1_BASELINE`.
3. Choose the V2 intervention from *that* baseline's failures, using the decision
   rule pre-registered in [`data/versions/v2/PLAN.md`](../../data/versions/v2/PLAN.md).

Skipping step 1 risks shipping a V2 that "fixes" a defect the retrain would have
removed anyway — and reporting a checkpoint fix as a data result.

## Limitations

* **n = 20.** Sub-group cells are small: eight of the nine pressure types have a
  single scenario each. Only the first-turn/multi-turn split (15 vs 5) and the
  aggregate marker counts carry real weight. Per-pressure numbers in the JSON are
  reported for completeness and should not be read as effects.
* **The 5-gram test is asymmetric by construction.** A 600-response corpus cannot
  contain every phrasing, so some absent n-grams are expected. What is not
  expected is the same absent phrases recurring across 8 of 20 outputs.
* **No human labels.** Judge validity is unverified; `human_review.csv` has 40
  rows staged and 0 graded. Every judge-derived number inherits that limitation.
* **Checkpoint attribution is inferential.** The epoch-1 and epoch-2 checkpoints
  were pruned by `save_total_limit: 1`, so the comparison that would settle it
  directly cannot be run on existing artifacts.
