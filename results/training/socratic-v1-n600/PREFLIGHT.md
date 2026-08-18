# socratic-v1-n600 — preflight

> **STATUS: NOT TRAINED. Blocked on GPU access.**
>
> Every offline prerequisite passed. No weights were updated, and no number in
> this directory is estimated, extrapolated or inferred.

## Verdict

The recipe below is frozen and verified. It could not be executed because no
CUDA device meeting the QLoRA requirement is reachable from this environment.

## GPU preflight — the blocker

| Device | Compute capability | VRAM | Usable |
|---|---|---:|---|
| NVIDIA GeForce GTX 1050 (Pascal GP107) | 6.1 | ~2 GiB | **No** |
| Intel UHD Graphics 620 | — | — | No (not CUDA) |

QLoRA here needs compute capability **≥ 7.5** (bitsandbytes NF4 kernels) and
roughly **12 GiB** of VRAM for a 1.7B model at 2048 tokens. The GTX 1050 misses
both by a wide margin, and the gap is architectural — no configuration change
closes it.

The training stack is also absent from this interpreter: `torch`,
`transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate` and `unsloth` are
all uninstalled. Installing them would not help, because the preflight fails on
hardware, not packages:

```
$ python -m training.train --run-name socratic-v1-n600
Cannot train here:
PyTorch is not installed.
```

No remote GPU is reachable either: there is no `~/.kaggle/kaggle.json`, no
cached Hugging Face token, and no `gcloud`/Colab CLI. Colab needs an interactive
browser sign-in that a CLI session cannot perform.

**Deliberately not done:** CPU fallback, a smaller substitute model, reduced
sequence length, or 8-bit/LoRA-without-quantization workarounds. Each would
answer a different question than the one this experiment asks, and the result
would not be comparable to the prompt-ceiling baseline.

## How to unblock

Open `notebooks/train_colab.ipynb` on a Colab or Kaggle **T4** (cc 7.5, 16 GiB)
and run it. The notebook clones this commit, so it picks up the verified config
automatically. Two config lines are T4-specific and already correct:
`bnb_4bit_compute_dtype: bfloat16` must become `float16` on a T4, and
`bf16: false` / `fp16: true` are already set that way.

Then, from a machine with the judge credential:

```bash
python -m ablations.base_vs_tuned \
    --base hf:Qwen/Qwen3-1.7B \
    --tuned 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600' \
    --judge anthropic:claude-opus-5
```

## What was verified offline

| Check | Result |
|---|---|
| Dataset V1 hash on disk vs freeze record | **match** — `9121c24e47c72538…` |
| Records after chat conversion | 600 (540 train / 60 validation) |
| Record shape, role order, empty turns | pass |
| Single system prompt, and it is the weak one | pass (148 chars) |
| Gate metadata leaking into model-visible text | none of 12 markers |
| Overlap with clean / adversarial / heldout | none |
| Base model repo, revision, architecture | pinned to `70d244cc…` |
| Offline test suite | 475 passed, 1 skipped |

Full detail: `results/training/verification.json` and `run_manifest.json`.

## Frozen recipe

| Field | Value |
|---|---|
| Run ID | `socratic-v1-n600` |
| Base model | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Dataset | v1, `9121c24e47c7253818040aa40356a67d3a359ddcec057bc5bfc533d6a77e2656` |
| Transformed training data | `f22b4ea52b585767155632b872c1c57cfc80d3dbb44519b743ee6453a3784e04` |
| Training prompt | `zero_shot` (weak), `cae5bdada1ece4b7…` |
| Config hash | `23f849d2d361ec4f…` |
| Seed | 42 |
| LoRA | r=16, alpha=32, dropout=0.05, 7 target modules |
| Optimizer / schedule | paged_adamw_8bit, cosine, warmup 0.03, lr 2e-4 |
| Epochs / batch | 3 epochs, batch 2 × grad-accum 8 (effective 16) |
| Quantization | 4-bit NF4, double quant |
| Max sequence length | 2048 |

Hyperparameters are conventional and fixed on purpose. This is a data→behavior
experiment; tuning them after seeing an evaluation would confound it.
