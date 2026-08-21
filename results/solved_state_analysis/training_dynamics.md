# Why solved-state confirmation is retained inconsistently

Forensic reading of the actual training recipe behind `N600_V1_BASELINE`, and
the smallest controlled ablation it justifies. No API calls, no GPU, no training,
no data generation. Every count recomputes via
`python -m analysis.training_signal`.

Observed result being explained (**not reinterpreted**):

```
training examples : gen_v1_00486 CONFIRMS | gen_v1_00792 no | gen_v1_00008 no   -> 1/3
held-out solved   : 0/2 zero_shot, 0/2 with explicit release rule
base Qwen3-1.7B   : 1/2 zero_shot, 2/2 with explicit release rule
VERDICT           : CASE_3 - mixed / unstable retention
```

---

## Step 1 — the actual recipe

All from `training/configs/qlora_qwen3_1_7b_t4_bestckpt.yaml` and
`training/train.py` unless noted.

| # | Item | Value | Source |
|---|---|---|---|
| 1 | Learning rate | `2.0e-4` | config:67 |
| 2 | Scheduler | `cosine` | config:68 |
| 3 | Warmup | `warmup_ratio: 0.03` → 3 steps of 102 | config:69; train.py:318 |
| 4 | Epochs | `3` | config:64 |
| 5 | Optimizer | `paged_adamw_8bit` | config:72 |
| 6 | Effective batch | **16** (2 × 8) | config:65-66 |
| 7 | Grad accumulation | `8` | config:66 |
| 8 | LoRA r / α / dropout | `16 / 32 / 0.05` | config:49-51 |
| 9 | LoRA targets | q,k,v,o,gate,up,down_proj | config:53-59 |
| 10 | Weight decay | `0.0` | config:70 |
| 11 | Max sequence length | `2048` | config:38 |
| 12 | Packing | **not set** → TRL default `False` | train.py:367-390 |
| 13 | Gradient clipping | `max_grad_norm: 0.3` | config:71 |
| 14 | Seed | `42` | config:97 |
| 15 | bf16 / fp16 | both `false` (T4 constraint) | config:94-95 |
| 16 | LoRA params kept fp32 | **yes**, forced | `cast_trainable_parameters_to_fp32`, train.py:244 |
| 17 | Save / eval cadence | both `epoch`, `save_total_limit: 3` | config:74-78 |
| 18 | `load_best_model_at_end` | `true`, `eval_loss`, `greater_is_better: false` | config:79-81 |

### Loss masking — the answer is B

**Loss applies to system + user + assistant tokens.** Nothing is masked to
`-100`.

Proven, not assumed:

1. `training/train.py:367-390` — `requested_trainer_arguments()` never sets
   `assistant_only_loss` or `completion_only_loss`, so TRL's defaults apply.
2. `trl/trainer/sft_config.py:272` — `assistant_only_loss: bool = False`, whose
   own help text reads *"If `False`, loss is computed on the entire sequence."*
3. `trl/trainer/sft_config.py:260` — `completion_only_loss: bool | None = None`;
   completion-only applies **only to prompt-completion datasets**. Ours is a
   conversational `messages` dataset with no `prompt` key.
4. `trl/trainer/sft_trainer.py:1510-1519` — the language-modeling branch calls
   `_tokenize(..., return_assistant_tokens_mask=assistant_only_loss)` and keeps
   `{k for k in ("input_ids", "assistant_masks") if k in processed}`. With the
   flag `False`, **only `input_ids` survives**; no `completion_mask` is built
   (that happens only in the prompt-completion branch at line 1506).
5. `trl/trainer/sft_trainer.py:465` — the collator:
   `labels = [example.get("labels", example["input_ids"]) for example in examples]`.
   No `labels` key exists, so **labels are the input ids** and every token
   contributes.

TRL 1.10.0 was installed locally at the exact Colab version to read this.

---

## Step 2 — where the optimization signal actually goes

Measured with the run's own tokenizer at pinned revision `70d244cc`, over the
540-row train split.

**Example counts**

| | Count |
|---|---|
| Train total | 540 |
| Solved | **75** |
| Unsolved | **465** |
| Ratio | **6.2 : 1** |
| Solved share | **13.89%** |

**Target behaviour** (shared confirmation detector)

| State | confirm, no question | confirm + question | neither | question only |
|---|---|---|---|---|
| solved (75) | **69** | 3 | 3 | 0 |
| unsolved (465) | 2 | 25 | 77 | **361** |

96% of solved targets confirm without re-questioning; 78% of unsolved ask a
question. **The two regimes are cleanly separated in the data** — the model is
not being taught something ambiguous.

**Token-weighted, under full-sequence loss**

| Segment | Tokens | Share of loss |
|---|---|---|
| **All loss-bearing** | **272,647** | 100% |
| System prompt | 17,820 | **6.54%** |
| Learner + context | 198,925 | **72.96%** |
| Tutor target | 55,902 | **20.50%** |

**How much signal teaches RELEASE**

| Measure | Value |
|---|---|
| Solved target tokens | 20,291 |
| As share of tutor targets | **36.30%** |
| As share of **all** loss tokens | **7.44%** |

Two things follow, and they are the core of this report.

**Only 20.5% of the gradient signal is the behaviour being taught.** The other
79.5% trains the model to reproduce learner prose and an invariant system
prompt. Under assistant-only loss, that 20.5% would be 100%.

**Release behaviour is 36.3% of tutor-target tokens but 7.44% of total loss.**
Full-sequence masking dilutes the release signal roughly five-fold, and pushes
it *below* its 13.89% example share rather than above it — even though solved
targets are individually ~3.5× longer than unsolved ones.

---

## Step 3 — system-prompt invariance

| Measurement | Result |
|---|---|
| Unique system prompts in the train split | **1** |
| Every example uses the identical prompt | **yes** (540/540) |
| Prompt contains the solved-state release rule | **no** — it is the weak zero-shot prompt |
| System tokens receive training loss | **yes** — 17,820 tokens, 6.54% of all loss |
| Any example conditioning behaviour on a different system instruction | **none exist** |

**Classification: SUPPORTED.**

The evidence is stronger than "the prompts happen to be identical". Because loss
is full-sequence, the model was *actively trained* to reproduce that exact
invariant string 540 times — 6.54% of the entire gradient budget spent
memorising a constant. A token sequence that never varies and always appears
carries zero conditional information, and the model is optimised to predict it
regardless of context.

That is a concrete mechanism for the probe's central observation: an explicit
release rule in the system prompt moved the base model to 2/2 and the adapter to
0/2. The adapter has been trained to treat that region as fixed.

Honest limit: this explains *insensitivity to system-prompt instructions*. It
does not by itself explain the loss of the behaviour, since the held-out
scenarios also fail under the plain prompt.

---

## Steps 4 & 5 — checkpoint and base matrix (NOT YET RUN)

`scripts/probe_checkpoint_retention.py` is written and tested, but requires a
GPU. **It has not been executed and no result is reported here.**

It runs the same three training examples, prompts, generation parameters and
detector across:

```
base            ?/3     <- what the capability looked like before fine-tuning
checkpoint-34   ?/3     <- the exported adapter (epoch 1, best eval_loss)
checkpoint-68   ?/3     <- epoch 2
checkpoint-102  ?/3     <- epoch 3
```

`describe_shape()` classifies the outcome mechanically:

- `BASE_LACKS_IT` — the untuned model does not confirm on these inputs either,
  so fine-tuning cannot have removed it here. **This would refute H1/H2/H3
  outright** and is the single most important thing this probe can find.
- `ERODED_BY_EPOCH_1` — the loss happens inside the first epoch, which points at
  learning rate rather than epoch count.
- `ERODED_EARLY_THEN_FLAT` — most loss present at epoch 1, later epochs do not
  recover it.
- `RETAINED_AT_EPOCH_1` — loss happens after epoch 1, pointing at epoch count.

The base row is the load-bearing one. Every training hypothesis below assumes the
base model *has* the capability on these specific inputs, and that assumption is
currently supported only by held-out data (base 1/2 and 2/2), not by these three.

---

## Step 6 — what the evidence supports

| | Hypothesis | Verdict | Measurement |
|---|---|---|---|
| **H1** | Excessive LR / destructive update | **INSUFFICIENT EVIDENCE** | 2e-4 at rank 16 is conventional for QLoRA. Nothing measured here isolates LR from epochs. Needs Step 4 |
| **H2** | Too many epochs | **PLAUSIBLE** | Validation loss rose 1.967 → 2.819 → 2.840 and entropy 2.99 → 5.24, so epochs 2–3 clearly degraded. But the *exported* adapter is epoch 1, and it already fails 2/3 — so epoch count cannot be the whole story |
| **H3** | Solved/unsolved optimization imbalance | **SUPPORTED** | 6.2:1 by example; release is **7.44%** of loss tokens. Measured, not inferred |
| **H4** | System-prompt invariance / instruction insensitivity | **SUPPORTED** | 1 unique prompt, 540/540 identical, **receiving 6.54% of loss**. Base moves 1/2 → 2/2 under an explicit rule; adapter 0/2 → 0/2 |
| **H5** | Loss masking causes undesirable optimization | **SUPPORTED** | Full-sequence loss proven from TRL source. 79.5% of gradient trains non-target text; release diluted ~5× |
| **H6** | LoRA capacity / configuration | **INSUFFICIENT EVIDENCE** | r=16, α=32 on all 7 projections is ample for one behaviour, and 17.4M trainable params fit 75 demonstrations easily. Nothing measured isolates capacity |
| **H7** | Dataset V1 content deficiency | **REFUTED** | 75 solved in train, 96% confirm cleanly, regimes cleanly separated, near-duplicates of both held-out failures present. Five prior hypotheses already refuted in `report.md` §8 |
| **H8** | Small-model capability limit | **REFUTED as a full explanation** | The base model confirms 2/2 with an explicit rule and 1/2 without. The capability exists at 1.7B. Whether it can be *retained* through fine-tuning is a different question and remains open |

Three hypotheses are supported and they are not independent: **H5 causes H3 and
H4.** Full-sequence loss is why release is 7.44% rather than 36.3% of the signal,
and why 6.54% of the gradient goes into memorising an invariant prompt. That
makes H5 the upstream variable and the one an ablation should move first.

---

## Step 7 — the smallest controlled ablation

Dataset V1 unchanged. Same frozen train/validation split, same seed 42, same base
model and revision, same LoRA architecture, same eval scenarios, same generation
settings. **Four arms, one variable each.**

| Arm | Change | Tests | Why supported |
|---|---|---|---|
| **CONTROL** | none — current corrected recipe | reproduces `N600_V1_BASELINE` | Without it the other arms have no reference |
| **A — assistant-only loss** | `assistant_only_loss: true` | **H5** | Moves release from 7.44% → 36.3% of the signal and stops training on the invariant prompt. The largest measured lever |
| **B — one epoch** | `num_train_epochs: 1` | **H2** | Validation loss and entropy both degraded after epoch 1. Isolates epoch count from LR |
| **C — lower LR** | `learning_rate: 5e-5` | **H1** | Only interpretable *after* Step 4 shows whether erosion happens inside epoch 1. **Run last, or drop if Step 4 says `RETAINED_AT_EPOCH_1`** |

Deliberately excluded: LoRA rank (H6 has no supporting measurement), any dataset
change (H7 refuted), and system-prompt variation — which is tempting for H4 but
is *not* a pure recipe change, since it edits what every training example
contains. Arm A already removes the system prompt from the loss, which tests the
same mechanism without touching the data.

Cost: 3 additional T4 runs at ~38 min each. Arm A alone is the minimum viable
experiment if budget is tight.

---

## Step 8 — success criteria, fixed before training

**The primary metric is capability retention, not held-out pass rate.**

| # | Measurement | Baseline |
|---|---|---|
| 1 | Confirmation on the 3 solved **training** examples | **1/3** |
| 2 | Confirmation on the 2 frozen held-out solved scenarios | **0/2** |
| 3 | Premature confirmation / fake-success behaviour | 0.00 |
| 4 | Solution leakage | 1/20 = 0.05 |
| 5 | Frozen held-out pass rate | 10/20 = 0.50 |

**An arm is not better merely because it confirms more.** It must recover
confirmation *while preserving discrimination*:

- genuine solved → **confirm / release**
- fake success or still-wrong → **withhold + one diagnostic question**

`js_heldout_fake_success_json_parse` is the discriminating control: an arm that
confirms there has traded one failure for a worse one.

**Guardrails, not to be weakened after seeing results:**

```
solution leak rate        <= 0.05
overall pass rate         >= 0.50
premature confirmation    <= 0.10
spec adherence            >= 0.631 - 0.05 = 0.581
infrastructure errors     == 0
```

---

## Step 9 — V2 decision gate

**A. Is unstable retention primarily explainable by training dynamics?**
**Partly, and more than by anything else.** Three supported hypotheses (H5 → H3,
H4) are all recipe properties, measured rather than argued: full-sequence loss,
a 7.44% release share, and 6.54% of gradient spent on an invariant prompt. But
"primarily" is not yet established — Step 4 has not run, and the base model's
behaviour on these three inputs is unmeasured.

**B. Is there evidence a recipe-only change can preserve confirmation while
keeping non-leak behaviour?** **Not yet — this is the open question.** The
supporting evidence is indirect: the base model has the capability, the data
teaches it cleanly, and the loss configuration demonstrably dilutes it ~5×. No
recipe change has been trained, so nothing shows the trade is achievable. Arm A
is the test.

**C. Is Dataset V2 justified now?** **No.** H7 is refuted on counts, and a data
change cannot fix a 7.44% signal share caused by masking configuration. Building
V2 before Arm A risks attributing a masking artifact to data content — the same
error as the MVP checkpoint defect, one layer down.

**D. If justified, what should change?** N/A at this gate.

**E. What evidence is missing?**

1. **Step 4/5 matrix** — free, GPU-only. Whether the base model confirms on these
   three inputs, and whether erosion happens inside epoch 1.
2. **Arm A** — whether assistant-only loss recovers retention without losing the
   non-leak behaviour.
3. **Discrimination under any recovered arm** — the fake-success control must
   still withhold.

Only after (1)–(3) does the contrastive minimal-pair V2 become answerable. Its
premise is that the model can learn and hold a solved/unsolved distinction, and
the retention gate shows it does not currently hold the *simpler* behaviour from
examples it saw directly. If Arm A restores retention, that premise is repaired
and V2 becomes designable. If it does not, V2 would be built on the same broken
premise.
