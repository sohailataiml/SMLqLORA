# Base vs Tuned — Dataset V1, N=600

> **STATUS: REAL EXPERIMENT RESULT.**
> **OUTCOME: MIXED RESULT.**
> **Caveat that governs everything below: the evaluated checkpoint is the
> epoch-3 adapter, and epoch 1 was measurably better. See "The confound".**

Held-out set: `scenarios/heldout.jsonl` (20 scenarios,
hash `a30abe2a9be7df5420e01197ba700b9e582fefb06fa3d0a0855351c1fbb5f048`)
Prompt strategy: `zero_shot` — the weak prompt, identical for both models
Judge: `anthropic:claude-opus-5`
Base: `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
Tuned: the same weights plus `outputs/socratic-v1-n600`

## Counts

Both models were measured on every scenario. No infrastructure errors, so the
two denominators are equal and the deltas below are like-for-like.

| model | attempted | measured | infrastructure errors | subject calls ok | judge calls ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 20 | 20 | 0 | 20 | 20 |
| tuned | 20 | 20 | 0 | 20 | 20 |

## Headline

| metric | base | tuned | delta |
| --- | ---: | ---: | ---: |
| spec adherence | 0.045 | 0.459 | **+0.413** |
| robustness | 0.233 | 0.678 | **+0.445** |
| pass rate | 0.000 | 0.250 | **+0.250** |
| solution leak rate | 0.450 | **0.000** | **−0.450** |
| hint relevance | 0.573 | 0.408 | **−0.164** |
| premature confirmation rate | 0.000 | 0.050 | +0.050 |

Raw counts: base passed **0 of 20**. Tuned passed **5 of 20**.

## Pressure

| model | clean (11) | adversarial (9) |
| --- | ---: | ---: |
| base | 0.000 (0/11) | 0.000 (0/9) |
| tuned | 0.364 (4/11) | 0.111 (1/9) |

Adversarial handling improved but remains weak: one scenario in nine.

## Failure modes

| code | base | tuned | direction |
| --- | ---: | ---: | --- |
| SOLUTION_LEAK | 9 | **0** | eliminated |
| MULTIPLE_HINTS | 17 | 5 | large improvement |
| OVER_EXPLANATION | 12 | 1 | large improvement |
| EXPLICIT_FINAL_DIAGNOSIS | 11 | 1 | large improvement |
| INCORRECT_DIAGNOSIS | 2 | 4 | **worse** |
| FAILED_TO_ADAPT | 1 | 5 | **worse** |
| IRRELEVANT_HINT | 0 | 3 | **worse** |
| LOW_QUALITY | 2 | 5 | **worse** |
| WITHHELD_AFTER_SOLVED | 0 | 1 | ~unchanged |
| PREMATURE_CONFIRMATION | 0 | 1 | ~unchanged |
| DUPLICATE | 0 | 1 | ~unchanged |

Codes are multi-label; one response can carry several.

## What the tuned model learned, and what it did not

**It learned the policy.** Every failure mode about *revealing* the answer
collapsed. Solution leaks went from 9 occurrences to none, over-explanation from
12 to 1, explicit final diagnosis from 11 to 1. Under the same weak prompt that
the prompt-ceiling ablation showed could not buy this behavior, the tuned model
withholds reliably.

**It did not learn the competence.** The failures moved rather than
disappearing, from *saying too much* to *saying the wrong thing*. Hint relevance
fell 0.573 → 0.408, and INCORRECT_DIAGNOSIS, FAILED_TO_ADAPT, IRRELEVANT_HINT
and LOW_QUALITY all rose. The responses are well-formed Socratic questions that
frequently misidentify the bug:

> "Good, so the problem is **not the loop itself**, but the way you read the
> items." — on `py_heldout_range_step`, where the loop *is* the bug.

> "**I've already confirmed that** the list is not a function, so the problem is
> not the function itself." — nothing had been confirmed.

That second phrasing recurs across outputs, including ones the judge passed. It
reads as a tic absorbed from teacher responses that referenced genuine earlier
findings, reproduced in contexts where no earlier finding exists.

**The refusal pathology did not appear.** The obvious worry about a zero leak
rate is that the model bought it by never confirming anything. It did not:
WITHHELD_AFTER_SOLVED is 1 and PREMATURE_CONFIRMATION is 1. Solved-state
handling is essentially intact, which is the two-sided invariant holding.

## The confound

The evaluated adapter is **epoch 3**. The training run's own validation curve:

| | eval loss | token accuracy | entropy |
| --- | ---: | ---: | ---: |
| epoch 1 | **1.97** | 0.584 | 3.02 |
| epoch 2 | 2.785 | 0.514 | 5.15 |
| epoch 3 | 2.730 | 0.515 | 4.85 |

The model degraded after epoch 1 and never recovered. `save_strategy: epoch`
with `save_total_limit: 1` and no `load_best_model_at_end` means the *last*
checkpoint was kept and the best one pruned.

Vague, confidently-worded, wrong-but-well-shaped questions are exactly what a
high-entropy degraded checkpoint produces. The regression this report measures
and the degradation the loss curve records are the same shape.

**So N=600's actual capability has not been measured. What was measured is a
damaged instance of it.**

## Outcome: MIXED RESULT

Improvement on the primary thesis is unambiguous — solution leakage, the failure
that survived the strongest prompt in the prompt-ceiling ablation, went to zero.
But pass rate improved while hint relevance deteriorated, which is the textbook
mixed outcome, and absolute performance remains low at 5/20.

## Recommendation

**Do not run the N=125/250/500 sweep.** It would trace a data-efficiency curve
whose y-axis is contaminated by a checkpoint-selection defect.

**Do not build Dataset V2 yet.** The evidence does not isolate the data as the
cause. INCORRECT_DIAGNOSIS is equally consistent with a degraded checkpoint.

**Do re-train N=600 with best-checkpoint selection, then re-evaluate.** Same
data, same hyperparameters, same seed; add `load_best_model_at_end: true` and
`metric_for_best_model: eval_loss`, and raise `save_total_limit`. One 35-minute
run separates "V1 cannot teach diagnostic accuracy" from "the wrong epoch was
saved" — the cheapest decisive experiment available.

This is a correctness fix rather than hyperparameter tuning: keeping the best
checkpoint by validation loss is standard practice, not a search for a better
number. It is nonetheless a change made after seeing a result, and is recorded
here as such.

## Statistical uncertainty

**NOT COMPUTED.** No bootstrap is implemented in this harness. At N=20 the pass
rate of 0.250 is 5 successes and its confidence interval is wide. The leak-rate
delta (9/20 → 0/20) is the only result large enough to be robust at this N;
treat the smaller deltas, especially the hint-relevance regression, as
directional rather than established.

## Provenance

* Dataset V1 — 600 examples, hash
  `9121c24e47c7253818040aa40356a67d3a359ddcec057bc5bfc533d6a77e2656`
* Transformed training data — `f22b4ea52b585767155632b872c1c57cfc80d3dbb44519b743ee6453a3784e04`
* Behavior spec 1.0.0 — `dc14f40b94d622d1…`
* Training prompt — `zero_shot`, `cae5bdada1ece4b7…`
* The invalid first attempt is preserved at
  [`results/base_vs_tuned_invalid_run1/`](../base_vs_tuned_invalid_run1/INVALID_RUN.md)

**Raw artifacts committed.** `results.json`, `results.csv`,
`judge_transcripts.jsonl`, `manifest.json` and `base_vs_tuned.png` are in this
directory. `report_generated.md` is the harness's own output, kept unedited
beside this analysis.

Every figure above was recomputed from the 40 raw records and matched exactly:
base 0/20 passes and 9/20 leaks, tuned 5/20 passes and 0 leaks, zero empty
responses and zero errors on either model, all 40 judged with reasoning, spec
hash `dc14f40b…` on every record and `zero_shot` for both. A grader can repeat
that check without re-running anything.
