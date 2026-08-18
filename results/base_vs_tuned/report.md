# Base vs Tuned

> **STATUS: NOT RUN — blocked on GPU access, not on code.**

No adapter exists, so there is nothing to compare against the base model. No
metric has been estimated, and none appears anywhere in this directory.

## Why it did not run

QLoRA training of `Qwen/Qwen3-1.7B` needs a CUDA device with compute capability
≥ 7.5 and ~12 GiB of VRAM. The only CUDA device here is a GTX 1050 (cc 6.1,
~2 GiB), and no remote GPU is reachable from this environment. The full
preflight is in `results/training/socratic-v1-n600/PREFLIGHT.md`.

The base model was **not** evaluated alone. That is a deliberate choice, not an
omission: base and tuned must be scored under the same weights-only difference,
so running the base on CPU now and the tuned model on a T4 later would put a
hardware/dtype confound inside the primary comparison. Both halves should run
on the same device, back to back.

## What is verified and waiting

* Training data: 600 frozen records, hash-checked, contamination-free
  (`results/training/verification.json`).
* Training recipe: frozen and hashed
  (`results/training/socratic-v1-n600/run_manifest.json`).
* Evaluation harness: runs end to end on the held-out set and routes
  `hf:` and `peft:` specs through identical code, identical generation
  settings and the identical weak `zero_shot` prompt.
* Held-out eval set: 20 scenarios,
  hash `a30abe2a9be7df5420e01197ba700b9e582fefb06fa3d0a0855351c1fbb5f048`.

## How to populate this directory

```bash
# 1. Train on a T4 (Colab/Kaggle): notebooks/train_colab.ipynb
# 2. Then, with a funded judge credential:
python -m ablations.base_vs_tuned \
    --base hf:Qwen/Qwen3-1.7B \
    --tuned 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1-n600' \
    --judge anthropic:claude-opus-5
```

It writes `results.json`, `results.csv`, `report.md`,
`judge_transcripts.jsonl`, `failure_modes.json`, `pressure_breakdown.json`,
`qualitative_pairs.md`, `manifest.json` and `base_vs_tuned.png` here, with
representative failures from **both** models.

## Do not skip ahead

The N=125 / 250 / 500 data-efficiency sweep stays unrun until this comparison
shows that N=600 learned something. The nested subsets already exist under
`data/versions/v1/subsets/` and are unchanged.
