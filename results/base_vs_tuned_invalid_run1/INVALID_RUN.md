# Run 1 — INVALID_EVALUATION / INFRASTRUCTURE_OR_INFERENCE_DEFECT

**Status: INVALID. Do not cite any number from this run.**

This directory preserves the first N=600 base-vs-tuned attempt. It is kept for
provenance and must not be overwritten. The repaired rerun writes to
`results/base_vs_tuned_run2/`.

## What this run measured

Not behavior. An inference/output-handling defect.

| Observation | |
| --- | --- |
| Reported held-out scenarios | 19 (base's denominator, printed as if it covered both models) |
| BASE `EMPTY_RESPONSE` | 18 of 19 measured |
| TUNED `EMPTY_RESPONSE` | ~20 of 20 measured |
| TUNED passes | 0 |
| `solution_leak_rate` | 0 — only because almost nothing was said |

A model that emits nothing has not "failed to hold the behavior"; it has not
been measured. Reading a fine-tuning conclusion out of these numbers would be
reading an OOM as a personality trait.

## Correct experiment status

```
EVALUATION INVALID — FINE-TUNING EFFECT UNKNOWN
```

Explicitly **not**: regression, Dataset V1 failure, or a reason to retrain,
change hyperparameters, or build Dataset V2.

## Diagnosed causes

Found by reading the inference path; all three fixed in the same commit that
created this file. See `tests/test_inference_path.py`, which fails against the
old code.

1. **`torch_dtype` was silently dropped.** `models/local_hf.py` had
   `if self.dtype != "auto": kwargs["torch_dtype"] = self.dtype`, so the default
   `"auto"` meant the argument was never passed and transformers fell back to
   **float32** — ~8.1 GiB for Qwen3-1.7B instead of ~2.0 GiB.
2. **Two threads shared one GPU model.** `base_vs_tuned` defaulted to
   `max_workers=2`, doubling peak activation memory on an already-full T4 and
   racing the per-call `torch.manual_seed`.
3. **A crashed generation was scored as behavior.** On exception the adapter
   returned `text=""` with an error set; the deterministic checks then found
   `EMPTY_RESPONSE` in that empty string, and `classify_error()` recognised no
   CUDA/OOM markers — so the record stayed in the behavioral denominator instead
   of being excluded as infrastructure.

Together these predict the exact symptom, including the single lucky base
response before memory pressure accumulated.

## The 19-vs-20 discrepancy

Not a miscount. `report.md` printed `base.scenario_count` as though it described
both models. Infrastructure failures are excluded per cell, so base (1 such
failure) had 19 and tuned (0) had 20. The report now prints both denominators
and states that failure codes are multi-label.

## Dropping the original artifacts here

The Colab session's `results.json`, `results.csv`, `report.md`,
`judge_transcripts.jsonl` and `manifest.json` belong in this directory. They are
evidence — in particular `judge_transcripts.jsonl` carries `usage.output_tokens`
per record, which settles whether the models generated zero tokens or generated
tokens that post-processing discarded.

## What is preserved and untouched

* Dataset V1 — frozen, hash `9121c24e…`
* The N=600 training run, its adapter, manifest and logs
* This invalid evaluation
