# Dataset V2 — pre-registered plan

**Status: PRE-REGISTERED, NOT BUILT.** No V2 example exists. Nothing in this
document may be revised after the V2 evaluation is seen; revisions before then
must be committed with a dated entry in "Amendments" below.

Machine-readable: [`plan.json`](plan.json).

---

## Why this plan is conditional, and why that is not evasion

The Early Submission improvement must come from a data change addressing a
diagnosed failure. The N=600 run produced an obvious candidate diagnosis — hint
relevance fell 0.573 → 0.408 while solution leakage went to zero — and the
temptation is to write the V2 hypothesis against it immediately.

That diagnosis does not survive testing. Per
[`results/failure_analysis/README.md`](../../../results/failure_analysis/README.md):

* Four hypotheses blaming Dataset V1 were tested against the frozen data. **All
  four are refuted.** Bug-category coverage is even (19–27 per category), the
  quality gate left no vague population (mean hint relevance 0.950, minimum
  0.82), the tuned model's stock phrasings appear in 0–0.5% of V1, and the one
  real distribution gap predicts the *opposite* of what happened.
* **17 of 20 held-out outputs** carry phrasing found in none of V1's 600 tutor
  responses, including 10 distinct 5-grams recurring across three or more
  outputs; 13 of the 15 failures are among them. The checkpoint is generating
  from a template attractor absent from its own training data.
* The evaluated adapter is epoch 3, with validation loss 39% worse than epoch 1.
  `save_total_limit: 1` without `load_best_model_at_end` pruned the best
  checkpoint.

So the observed regression is, on present evidence, a property of the checkpoint
rather than of Dataset V1. Writing a V2 hypothesis against it now would target a
training artifact, and a V2 that "fixed" it would be reporting a checkpoint fix
as a data result.

**What this plan does instead:** pre-register three fully-specified hypotheses
with a decision rule that selects among them from the corrected baseline, and
fix every success threshold *now*, before that baseline is measured. This is
stronger pre-registration than committing to one hypothesis on evidence known to
be contaminated — the thresholds cannot be tuned to the result, because they are
written down before the result exists.

## Prerequisite — the canonical baseline

`N600_V1_BASELINE` does not yet exist. It is defined as:

> Dataset V1, N=600, unchanged recipe **except** `load_best_model_at_end: true`,
> `metric_for_best_model: eval_loss`, `greater_is_better: false`,
> `save_total_limit: 3` — evaluated on `scenarios/heldout.jsonl`
> (`a30abe2a…`), `zero_shot` prompt, `anthropic:claude-opus-5` judge, behavior
> spec `dc14f40b…`, base `Qwen/Qwen3-1.7B@70d244cc…`.

Those four settings are a **checkpoint-selection correctness fix, not the V2
intervention.** They are not claimable as the Early Submission improvement and
are excluded from every V2 claim. V2 is trained with the identical corrected
recipe, so the V1↔V2 comparison isolates data.

The epoch-1 and epoch-2 adapters from the MVP run were deleted by
`save_total_limit: 1`, so this baseline requires one training run. It cannot be
recovered from existing artifacts.

## Pre-registered hypotheses

Each is stated in the required falsifiable form. Exactly one will be selected by
the decision rule below.

### H-A — multi-turn adaptation

> Dataset V1 successfully reduced solution leakage, but **failure to adapt to a
> learner's established findings across turns** remains, because **every
> assistant turn in V1's synthetic conversation histories is a well-aimed teacher
> hint, so the model never sees a tutor recovering from a prior move that
> missed**. Dataset V2 will change **the conversation histories of a subset of
> multi-turn examples so the prior assistant turn is misaimed and the target
> response must redirect without restating it**. If the hypothesis is correct,
> **multi-turn pass rate and `FAILED_TO_ADAPT` counts** should improve while
> solution leakage remains at or near the V1 level.

Standing evidence: the tuned model passed **0 of 5** multi-turn scenarios and
**5 of 15** first-turn ones, despite V1 being 79.8% multi-turn. Plentiful data
that does not buy competence is the signature of data teaching the wrong thing.

### H-B — solved-state release

> Dataset V1 successfully reduced solution leakage, but **withholding after the
> learner has already produced the fix** remains, because **V1's 85 solved-state
> examples are 14.2% of the data and all sit at turn 2 or later, giving the model
> a weak and narrow signal for the one case where the constraint inverts**.
> Dataset V2 will change **the proportion and turn-position spread of
> solved-state examples**. If the hypothesis is correct, **`WITHHELD_AFTER_SOLVED`
> and solved-split pass rate** should improve while solution leakage and
> `PREMATURE_CONFIRMATION` remain at or near the V1 level.

Standing evidence: both solved-state held-out scenarios failed
(`WITHHELD_AFTER_SOLVED` on one, degenerate confirmation on the other). This is
the two-sided invariant that makes the behavior spec hard, and it is the thinnest
slice of V1.

### H-C — hint-policy consistency

> Dataset V1 successfully reduced solution leakage, but **hints aimed at the
> wrong part of the program** remain, because **V1 mixes three different hint
> policies — `narrow` (234), `nudge` (206), `pointed` (160) — with nothing in the
> learner-visible input indicating which applies, so the target for any given
> scenario is ambiguous**. Dataset V2 will change **the hint-strength
> distribution to a single consistent policy, regenerating the off-policy
> examples at the retained strength**. If the hypothesis is correct, **hint
> relevance and `INCORRECT_DIAGNOSIS` / `IRRELEVANT_HINT` counts** should improve
> while solution leakage remains at or near the V1 level.

Standing evidence: `hint_strength` is a generation dimension recorded in every
V1 example, but it is not derivable from the scenario the model sees. A 1.7B
model must therefore fit three policies to one input distribution. This is a
checkpoint-independent property of the data, verifiable now.

> **A fourth candidate was drafted and withdrawn.** "The V1 gate never checked
> hints against the scenario's `expected_bug`" is false: `evaluation/judge.py`
> puts `Actual bug:` / `Correct fix:` in the judge prompt and defines
> `hint_relevance` as "how directly the single move points at the ACTUAL bug
> given above. A well-formed question aimed at the wrong part of the program
> scores low." The gate did check diagnostic grounding, and V1 scores a mean of
> 0.950 on it. Recorded here rather than deleted, because it is the hypothesis
> most likely to be proposed next.

## Decision rule — fixed before the baseline is measured

Applied to `N600_V1_BASELINE` in this order; the first match fires:

| # | Condition on the corrected baseline | Selected |
| --- | --- | --- |
| 1 | multi-turn pass rate is more than 0.20 below first-turn pass rate | **H-A** |
| 2 | else if `WITHHELD_AFTER_SOLVED` ≥ 1 **or** solved-split pass rate = 0 | **H-B** |
| 3 | else if hint relevance is below the base model's 0.573 **or** `INCORRECT_DIAGNOSIS` + `IRRELEVANT_HINT` ≥ 4 | **H-C** |
| 4 | else — no data-attributable failure survives the correction | **STOP and report** |

Branch 4 is a real possible outcome and will be reported as one rather than
worked around. If it fires, the Early Submission reports that the MVP's apparent
data defect was a checkpoint defect, with the corrected numbers as evidence, and
V2 is proposed but not claimed.

## Success criteria — fixed before V2 exists

Let `B` = the metric on `N600_V1_BASELINE` and `V` = the same metric on V2.

**Primary (the selected hypothesis's target metric):**

* `V − B ≥ +0.10` absolute, **and** the change is at least 2 scenarios out of 20.

**Guardrails — any breach means the intervention is not claimed as a win:**

| guardrail | bound |
| --- | --- |
| solution leak rate | `V ≤ 0.05` (V1 achieved 0.000; one scenario in 20 is tolerated) |
| spec adherence | `V ≥ B − 0.05` |
| pass rate | `V ≥ B` |
| premature confirmation rate | `V ≤ 0.10` |
| infrastructure errors | 0 (any non-zero invalidates the run) |

**Reporting rule:** at n = 20, a delta below 0.10 is not claimable and will be
reported as directional. Raw counts accompany every rate. The outcome is stated
as one of `YES` / `PARTIALLY` / `NO` / `REGRESSED`, and a guardrail breach
forbids `YES`.

## Construction constraints

* **N is held at 600.** V2 replaces examples rather than adding them, so the
  comparison isolates data *content*, not data *quantity*. The data-efficiency
  curve is the separate experiment that varies N.
* **Smallest defensible intervention.** Cap: **150 of 600 examples changed
  (25%)**. Exceeding the cap requires a written justification in Amendments
  before V2 is built.
* **V1 is not touched.** V2 is a new directory with its own hash, freeze record
  and dataset card. Every changed example records its V1 ancestor id and a reason
  code.
* **Same everything else.** Teacher `anthropic:claude-opus-5`, behavior spec
  `1.0.0` / `dc14f40b…`, same schema, same dedupe, same contamination check
  against all three scenario files, same stratified selection seed policy.
* V2 must ship a V1↔V2 distribution comparison across pressure type, bug
  category, language, difficulty, turn position and student progress.

## What is explicitly not the V2 intervention

`load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`,
`save_total_limit`, and any other training-configuration or checkpoint-selection
change. These are corrections applied identically to the V1 baseline and to V2.

## Amendments

*(none — the plan is as first committed)*
