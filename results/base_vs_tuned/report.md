# Base vs Tuned

> **STATUS: EVALUATION INVALID — FINE-TUNING EFFECT UNKNOWN.**

The first N=600 comparison ran, but it did not measure behavior. Both models
returned `EMPTY_RESPONSE` on nearly every scenario, which is an
inference/output-handling defect rather than a result. That run is preserved,
unmodified and clearly marked, at
[`results/base_vs_tuned_invalid_run1/`](../base_vs_tuned_invalid_run1/INVALID_RUN.md).

**No fine-tuning conclusion exists.** Not a regression, not a Dataset V1 failure
— an unmeasured experiment.

## Diagnosis

Three defects in the shared inference path, all fixed:

1. `torch_dtype` was never passed to the loader, so `"auto"` silently meant
   float32 — 4× the memory of Qwen3's bfloat16 checkpoint.
2. `base_vs_tuned` ran two threads through one GPU model, doubling peak
   activation memory on a card already near its limit.
3. A crashed generation was scored as a behavioral failure: the empty text
   produced `EMPTY_RESPONSE`, and CUDA/OOM errors were not recognised as
   infrastructure, so the record stayed in the denominator.

That both *base* and *tuned* failed identically is what pointed at the shared
path rather than the adapter.

Regression tests reproducing all three: `tests/test_inference_path.py`.

## Repair procedure

[`notebooks/diagnose_inference.ipynb`](../../notebooks/diagnose_inference.ipynb)
runs on the same T4, reuses the **same N=600 checkpoint**, and works up from raw
token IDs:

1. Validate the checkpoint — adapter config, target modules, non-zero LoRA weights
2. Direct generation, base and tuned, printing token IDs and both decodings
3. Think-tag and `skip_special_tokens` comparison
4. Prompt-rendering parity between the two models
5. One held-out scenario, no judge
6. Three-scenario smoke, no judge, then judged
7. Full rerun into `results/base_vs_tuned_run2/` — gated on the smoke passing

## Not done, deliberately

No retraining, no Dataset V2, no hyperparameter changes, no data-efficiency
sweep, and no blind re-spend on judge calls. The checkpoint is presumed valid
until the notebook's checkpoint validation says otherwise.
