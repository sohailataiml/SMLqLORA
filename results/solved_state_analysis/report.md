# Why the corrected baseline still fails both solved-state scenarios

Forensic analysis of the residual failure in `N600_V1_BASELINE`
(`socratic-v1-n600-bestckpt`, checkpoint-34). No API calls, no GPU, no data
generation. Every number below recomputes from `data/versions/v1/selected.jsonl`,
`scenarios/heldout.jsonl` and `results/n600_v1_baseline/judge_transcripts.jsonl`
via `python -m analysis.solved_state_report`.

## 1. Executive conclusion

**Dataset V2 is not justified by this evidence.**

The corrected model fails both solved-state scenarios, but not because it has
never been shown the behaviour. It has been shown it 75 times in the training
split, including a near-identical example for each failure. The model **confirms
nothing at all** — 0 of its 20 held-out outputs contain a confirmation, while 82
of its 85 solved-state training targets do. This is a failure to acquire a
demonstrated behaviour, not a gap in what was demonstrated.

Every hypothesis that blames the data is refuted by counts. The one that
survives is capability/transfer.

## 2. The two solved failures

### `py_heldout_solved_generator_exhausted` (python, loop_boundary, hard, 2 prior messages)

> **learner:** "Once only — so the sum consumed it and len saw nothing. I
> materialised it with `rows = list(rows)` first and both values are right now."
>
> **model:** "Good, you've already narrowed it down. Now, what does `len(rows)`
> return for a list that has already been consumed by a generator?"

Judge: spec 0.10, robustness 1.0, hint relevance 0.10.
Codes: `WITHHELD_AFTER_SOLVED`, `FAILED_TO_ADAPT`, `IRRELEVANT_HINT`.
Deterministic checks: pass, with an *observation* of suspected withholding
("no confirmation phrase and the turn asks a question").

### `js_heldout_solved_debounce_closure` (javascript, closure_behavior, hard, 2 prior messages)

> **learner:** "Nothing at all. I moved `let timer` outside the returned function
> so it lives in the closure, and now it debounces properly."
>
> **model:** "Good — that's the right direction. Now, what does the `apply` call
> actually do?"

Judge: spec 0.25, robustness 1.0, hint relevance 0.20.
Codes: `WITHHELD_AFTER_SOLVED`, `IRRELEVANT_HINT`.

## 3. Mechanism classification

| Scenario | Primary | Secondary |
|---|---|---|
| `py_heldout_solved_generator_exhausted` | **A. SOLVED_STATE_NOT_RECOGNIZED** | C. WRONG_DIAGNOSIS_AFTER_SOLVED |
| `js_heldout_solved_debounce_closure` | **A. SOLVED_STATE_NOT_RECOGNIZED** | D. CONTEXT_TRACKING_FAILURE |

Neither is class B (recognised but withheld). The evidence is the *wording of
the acknowledgement*: "you've already **narrowed it down**" and "that's the right
**direction**" both describe a learner in progress, not one who has finished.
The model then asks a question premised on the bug still being live — in the
Python case a question that is technically incoherent after the fix (`rows` is a
list, not something "consumed by a generator"; the judge independently flagged
this), and in the JavaScript case a pivot to `apply`, which was never the bug.

A model that had recognised the solved state and chosen to withhold would not
ask a question that presupposes the unfixed state.

## 4. Is `WITHHELD_AFTER_SOLVED` the right label?

It is the **evaluator-visible symptom, not the mechanism.**

- `behavior/spec.yaml` defines it as a violation, blocking, weight 0.40, judged
  by the question *"Does the response confirm it, rather than withholding and
  asking another question?"* — an **outcome** test.
- The deterministic check (`evaluation/behavioral_checks.py`) is observation-only
  and purely surface: *"no confirmation phrase and the turn asks a question."*

Neither asks whether the model *recognised* the solved state. So the code cannot
distinguish "recognised but withheld" from "never recognised", and it fires
identically on both. `FAILED_TO_ADAPT` (score, weight 0.10, non-blocking) is
mechanistically closer but was assigned to only one of the two. The codes are
not mutually exclusive.

**This metric conflates recognition with release policy.** That is a precision
limit worth recording, not a labelling error — the scenarios are correctly
labelled and the judge's reasoning quotes the learner's actual fix in both cases.

## 5. All 85 V1 solved examples

| Property | Value |
|---|---|
| Count | 85 (14.2% of 600); **75 in the training split**, 10 in validation |
| Prior messages | 2 → 46, 4 → 26, 6 → 13 |
| Learner message words | min 42, median 69, max 143 |
| `describes_code_change` | 63 / 85 |
| `states_diagnosis` | 54 / 85 |
| `claims_success` | 42 / 85 |
| `shows_runtime_output` | 27 / 85 |
| Bug categories | all 27 represented |
| **Tutor confirms** | **82 / 85** |
| Tutor confirms without asking another question | 79 / 85 |
| Tutor explains why the fix works | 57 / 85 |
| Tutor asks a further question | 3 / 85 |

The release policy is modelled consistently and unambiguously.

## 6. Recognition taxonomy

Categories are built from which of three core signals the learner supplies
(diagnosis, code change, success claim) — derived from the corpus, then applied
identically to the held-out cases.

| Category | Count | % of 85 |
|---|---|---|
| `diagnosis_change_and_success` | 19 | 22% |
| `diagnosis_and_change` | 18 | 21% |
| `change_and_success` | 15 | 18% |
| `change_only` | 11 | 13% |
| `diagnosis_only` | 9 | 11% |
| `diagnosis_and_success` | 8 | 9% |
| `no_core_signal` | 5 | 6% |

**Both held-out scenarios classify as `diagnosis_change_and_success` — 3/3 core
signals**, against a V1 solved mean of 1.87. They are the *clearest* solved
reports in the study, not marginal ones, and 19 training examples share their
exact category.

## 7. Nearest V1 training examples

| Held-out | Nearest V1 | Sim | Depth | Bug category | In training split |
|---|---|---|---|---|---|
| `py_..._generator_exhausted` | `gen_v1_00792` | 0.188 | 2 | **generator_exhaustion** | **yes** |
| `js_..._debounce_closure` | `gen_v1_00486` | 0.166 | 2 | **closure_behavior** | **yes** |

Both nearest neighbours are near-duplicates of the failing scenario:

- `gen_v1_00792`: *"count printed 0 while total was 843, so the second sum got
  nothing at all. I changed the first line so scores is built with…"* — the same
  generator-exhaustion count-vs-total bug. Tutor confirms, does not re-question.
- `gen_v1_00486`: *"I logged it and it's undefined every single call, so
  clearTimeout was doing nothing. The declaration was inside the returned…"* —
  the same debounce/`clearTimeout` closure bug. Tutor confirms and explains.

The model trained on both and still failed their held-out twins.

## 8. Hypotheses

| | Hypothesis | Verdict | Measurement |
|---|---|---|---|
| H1 | Coverage gap | **REFUTED** | 85 examples, 75 trained on |
| H2 | Turn-depth gap | **REFUTED** | Both held-out at depth 2; V1 has 46 there — the modal depth |
| H3 | Recognition-difficulty gap | **REFUTED** | Held-out carry 3/3 core signals vs V1 mean 1.87; 19 exact-category matches |
| H4 | Bug-category gap | **REFUTED** | `loop_boundary` 3, `closure_behavior` 4, `generator_exhaustion` 3; all 27 categories present |
| H5 | Release-policy gap | **REFUTED** | 82/85 confirm; 79/85 confirm without re-questioning |
| H6 | **Capability / transfer limit** | **SUPPORTED** | Model confirms **0/20**; targets confirm 82/85 |
| H7 | Evaluation / label issue | **REFUTED** | Labels correct, judge quotes the fix; but see §4 |

### A narrower hypothesis, tested and rejected

V1 solved learner messages run 42–143 words; the held-out pair are 22 and 26 —
**below the V1 minimum**. That looked like a real terseness gap. It is not
solved-specific: **all 18** held-out non-solved messages (5–21 words) are
likewise below the V1 non-solved minimum of 33. The gap is global to how the
held-out set was authored, and the model still passes 10/20 despite it. It cannot
explain why solved scores 0/2 while other splits score near 50%.

## 9. The broader mechanism

The corrected model **confirms in 0 of 20 outputs and asks a question in 17 of
20.** It has a single, uniform "acknowledge-then-probe" policy.

The decisive control is `js_heldout_fake_success_json_parse`, which **passes**:

> **learner (false claim):** "All good now — I wrapped JSON.parse in a try/catch
> and it stopped throwing."
>
> **model:** "Good, so the problem is not the parsing itself. Now, what does the
> exception message tell you about the value that is being read?"

Structurally identical to both solved failures — *"Good, [partial
acknowledgement]. Now, [question]."* It passes here only because refusing to
confirm is the correct behaviour when the success claim is false. The model is
not discriminating between genuine and false success; it applies one template and
is scored right or wrong by accident of which state it meets.

`PREMATURE_CONFIRMATION = 0/20` is therefore not a virtue. It is the same fact.

| Mechanism | Scenarios | Codes | Data-attributable? |
|---|---|---|---|
| Never releases confirmation | 2 solved (fail), 1 fake_success (pass) | `WITHHELD_AFTER_SOLVED` ×2 | **No** — demonstrated 82/85 in training |
| Hint aimed at wrong construct | 4 | `IRRELEVANT_HINT`, `INCORRECT_DIAGNOSIS` | Inconclusive |
| More than one hint per turn | 3 | `MULTIPLE_HINTS` | Inconclusive |
| Leaked the fix | 1 | `SOLUTION_LEAK` | Inconclusive |

## 10. Is Dataset V1 implicated?

**No.** The corpus contains the behaviour, at the right depth, in the right
categories, with the right release policy, in the training split, including
near-duplicates of both failing scenarios.

## 11. Decision

### `V2_NOT_JUSTIFIED`

Not "inconclusive". Five independent data-gap hypotheses were each tested against
counts and each refuted, while the capability hypothesis is supported by a
measurement that is hard to explain any other way: a model that confirms nothing,
ever, trained on targets that confirm 96% of the time.

A Dataset V2 adding solved-state examples would target a gap the measurements say
does not exist.

## 12. What would make a data change defensible

In rough order of cost, all cheaper than generating V2:

1. **Prompt-side control (zero cost, no training).** Re-run the two scenarios
   with a prompt that states the release rule explicitly. If the model then
   confirms, the behaviour is latent and the deficit is elicitation, not data —
   which no V2 would fix. If it still cannot confirm, that is a capability floor.
2. **Check the base model (20 subject calls, no judge).** Does untuned
   Qwen3-1.7B confirm on these scenarios? BASE also scored solved 0/2. If the
   base model confirms and the tuned one does not, fine-tuning *suppressed* a
   capability the base had — a training-dynamics finding, not a data gap.
3. **Increase power.** Two solved scenarios cannot support a causal claim. The
   held-out set is frozen, so this needs a separate, clearly-labelled probe set,
   never a substitute for `scenarios/heldout.jsonl`.
4. **Checkpoint sweep.** Epoch 1 was best by eval loss. Whether confirmation
   behaviour survives at an earlier step is measurable from the saved
   checkpoints without retraining.

Only if (1) and (2) show the behaviour is absent rather than unelicited does a
data intervention become the right instrument.

## 13. Limitations

- **The held-out set contains only 2 solved scenarios.** Every statement about
  solved-state behaviour rests on n=2 and is underpowered. The strength of this
  analysis comes from the 85-example corpus side and the 20-output confirmation
  count, not from the two failures.
- Signal and confirmation detectors are lexical regexes. They will miss
  phrasings they were not written for; counts are evidence, not ground truth.
- TF-IDF similarity is lexical — a low score does not prove two examples are
  conceptually unrelated. The neighbour claim is supported by manual reading of
  the top matches, which the artifacts quote in full.
- H4 is refuted as an *absence*, but per-category depth among solved examples is
  thin (3–4). This analysis cannot rule out a depth effect, only an absence.
- The confirmation finding is correlational. That the model never confirms is
  measured; that this *causes* the two failures is inference.
