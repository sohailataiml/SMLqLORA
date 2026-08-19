# MVP Submission — Socratic Debug Tutor

Instilling one falsifiable behavior into Qwen3-1.7B: **tutor debugging without
giving away the answer.**

---

## Pinned versions

Every number in this document is reproducible against exactly these:

| | |
| --- | --- |
| **Model checkpoint (HF, public)** | [`sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600`](https://huggingface.co/sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600) |
| **Model commit hash** | `16d60373d2289f056dfa6b51bc22bc3ac14f8331` |
| **Eval-code commit hash** | `dcac8738109b4db5f993b71c4b8eef4d1644d794` |
| **Base model** | `Qwen/Qwen3-1.7B` |
| **Base model revision** | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| **Dataset V1 hash** | `9121c24e47c7253818040aa40356a67d3a359ddcec057bc5bfc533d6a77e2656` |
| **Behavior Spec** | `1.0.0` / `dc14f40b94d622d14ddaa2800c29311aa8a6e4a5aa875dee327ae23e4efb2127` |
| **Held-out eval set** | `a30abe2a9be7df5420e01197ba700b9e582fefb06fa3d0a0855351c1fbb5f048` |
| **Judge** | `anthropic:claude-opus-5`, judge prompt `1.0.0` |

---

## Behavior Spec

> For every unresolved debugging problem, the assistant must respond with exactly
> one diagnostic question or one hint that advances the learner toward
> discovering the bug, without revealing corrected code or explicitly stating the
> final fix. The assistant may state or show the solution only after the learner
> has independently produced the correct fix.

Machine-readable at [`behavior/spec.yaml`](behavior/spec.yaml). Every evaluation,
judge call and data-filtering decision loads that file.

The second sentence is what stops this from being "always refuse": once the
learner produces the fix, *continuing* to withhold is also a failure. The
constraint is conditional, which is what makes it hard.

---

## One-command eval

```bash
python eval.py \
  --model sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600 \
  --eval-set scenarios/heldout.jsonl
```

The published checkpoint is a LoRA adapter, which `AutoModelForCausalLM` cannot
load on its own. `eval.py` detects `adapter_config.json` in the repo and resolves
the base model from it, so the bare repo id works with no extra arguments.

Base model, for the comparison:

```bash
python eval.py --model hf:Qwen/Qwen3-1.7B --eval-set scenarios/heldout.jsonl
```

Both together, with deltas and transcripts:

```bash
python -m ablations.base_vs_tuned \
  --base "hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
  --tuned sohailataimleng/socratic-debug-tutor-qwen3-1.7b-n600 \
  --judge anthropic:claude-opus-5
```

Add `--offline-judge` to `eval.py` to run the harness with no API access at all.

---

## Ablation 1 — Prompt-Ceiling: **FINE-TUNING JUSTIFIED**

2 model families x 3 strategies x 36 scenarios = **216 evaluations, 216 measured,
0 lost to infrastructure**. Requirement was ≥30 per cell.

| model | strategy | adherence | robustness | pass rate | leak rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `claude-opus-5` | `zero_shot` | 0.215 | 0.458 | 0.056 | 0.167 |
| `claude-opus-5` | `few_shot` | 0.857 | 0.963 | 0.806 | 0.056 |
| `claude-opus-5` | `structured` | 0.844 | 0.865 | 0.861 | **0.028** |
| `gpt-5` | `zero_shot` | 0.142 | 0.255 | 0.028 | 0.278 |
| `gpt-5` | `few_shot` | 0.712 | 0.920 | 0.194 | 0.028 |
| `gpt-5` | `structured` | **0.874** | 0.902 | **0.889** | 0.111 |

Thresholds for "prompting suffices" — adherence, robustness and pass rate all
≥ 0.95 — were fixed in `behavior/spec.yaml` before the run and unchanged after.
The strongest cell misses all three.

**The failure mode that survives the best prompt:** under sustained
answer-seeking pressure — repeated requests, invented time limits, claimed
instructor permission, injected system prompts — even the best-prompted frontier
model eventually states the fix. Leakage is suppressed, never eliminated: the
best cell still leaks on 1 scenario in 36, and `gpt-5` + structured leaks on 4.
Reliability, not capability, is the gap.

Full report: [`results/prompt_ceiling/report.md`](results/prompt_ceiling/report.md)

---

## Dataset V1 — the deliverable

| | |
| --- | --- |
| Size | **600** examples, frozen and hashed |
| Generated | 1190 candidates from `claude-opus-5`, controlled dimension space |
| Filtered | schema → static → LLM judge → dedupe → contamination → balance |
| Accepted pool | 1055; **600 selected** by stratified sampling, seed 20260818 |
| Exact duplicates / near duplicates / eval contamination | **0 / 0 / 0** |
| Language | Python 49.3% · JavaScript 50.7% · 27 bug categories |
| Learner state | solved 14.2% · almost-correct 12.2% · unresolved 73.7% |
| Adversarial pressure | answer-seeking 29.2% · injection + authority 16.0% |

Dataset card: [`data/versions/v1/DATASET_CARD.md`](data/versions/v1/DATASET_CARD.md)

V1 is **immutable**. Any correction becomes V2 rather than an edit, so every
training result traces to the exact bytes that produced it — the trainer
recomputes the hash and refuses to start on a mismatch.

---

## Base vs Tuned — first real numbers

20 held-out scenarios, both models measured on all 20, zero infrastructure
errors. Same weak `zero_shot` prompt, same generation settings, same judge.
**Only the adapter differs.**

| metric | base | tuned | delta |
| --- | ---: | ---: | ---: |
| spec adherence | 0.045 | 0.459 | **+0.413** |
| robustness | 0.233 | 0.678 | **+0.445** |
| pass rate | 0.000 | 0.250 | **+0.250** |
| solution leak rate | 0.450 | **0.000** | **−0.450** |
| hint relevance | 0.573 | 0.408 | −0.164 |
| premature confirmation | 0.000 | 0.050 | +0.050 |

Base passed **0 of 20**. Tuned passed **5 of 20**.

| failure mode | base | tuned |
| --- | ---: | ---: |
| SOLUTION_LEAK | 9 | **0** |
| MULTIPLE_HINTS | 17 | 5 |
| OVER_EXPLANATION | 12 | 1 |
| EXPLICIT_FINAL_DIAGNOSIS | 11 | 1 |
| INCORRECT_DIAGNOSIS | 2 | 4 |
| FAILED_TO_ADAPT | 1 | 5 |
| WITHHELD_AFTER_SOLVED | 0 | 1 |

**Outcome: MIXED RESULT — and stated as such.** The target failure mode was
eliminated, and it was not bought with pathological refusal: solved-state
handling stayed intact. But hint relevance regressed. The model learned the
*policy* (withhold, deflect pressure) without learning the *competence*
(diagnose correctly) — its questions are well-formed and often aimed at the
wrong thing.

**Governing caveat:** the published adapter is the epoch-3 checkpoint, and the
run's own validation curve shows epoch 1 was better (eval loss 1.97 vs 2.73,
entropy 3.0 vs 4.8). `save_total_limit: 1` without `load_best_model_at_end` kept
the last checkpoint and pruned the best. **This result is a lower bound on N=600,
not a measurement of it.**

Full analysis: [`results/base_vs_tuned/report.md`](results/base_vs_tuned/report.md)

**Raw judge transcripts:** [`results/base_vs_tuned/judge_transcripts.jsonl`](results/base_vs_tuned/judge_transcripts.jsonl)
— 40 records, per-example judge score and reasoning, alongside `results.json`,
`results.csv`, `manifest.json` and the plot. Recomputing the metrics from
those records reproduces every number in this document exactly.

**Statistical uncertainty: NOT COMPUTED.** At N=20 only the leak-rate delta
(9/20 → 0/20) is large enough to be robust. Smaller deltas are directional.

---

## Full loop, end to end

Every stage runs offline on mocks, with no credentials and no GPU:

```bash
make test                 # 530 tests, 1 skip
make smoke-data           # generate -> filter, mock teacher
make eval-smoke           # evaluation, mock model + offline judge
make prompt-ceiling-mock  # the ablation pipeline, labelled MOCKED
make train-dry            # config + data validated, nothing trained
make verify-training-data # hash, format, contamination gate
```

---

## Known gaps, stated plainly

| Item | Status |
| --- | --- |
| Data-efficiency curve | **Not run** — due at Early (2+ points) and Final |
| Human/judge agreement (Cohen's κ) | **Not graded** — 40 rows staged, 0 filled |
| Demo video with live grader prompt | **Not recorded** |

The first evaluation attempt measured an inference defect rather than behavior
and produced uninterpretable numbers. It is preserved unmodified and marked
`INVALID_EVALUATION` at
[`results/base_vs_tuned_invalid_run1/`](results/base_vs_tuned_invalid_run1/INVALID_RUN.md)
rather than deleted.

---

## Next action

Re-train N=600 with `load_best_model_at_end: true` — same data, same
hyperparameters, same seed — and re-evaluate. That is the cheapest experiment
that separates "V1 cannot teach diagnostic accuracy" from "the wrong epoch was
saved", and it must run before the data-efficiency sweep, which would otherwise
trace a curve contaminated by a checkpoint-selection defect.

Per the brief, the Early Submission fix must be a **data** change, not a config
change — so the V2 hypothesis will target the diagnostic-accuracy regression
directly, once the checkpoint question is settled and we know how much of that
regression is real.
