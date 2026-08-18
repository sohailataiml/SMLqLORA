# Base vs Tuned

> **STATUS: NOT RUN.**

No checkpoint exists yet, so there is nothing to compare. This directory is
populated by:

```bash
python -m ablations.base_vs_tuned \
    --base hf:Qwen/Qwen3-1.7B \
    --tuned 'peft:Qwen/Qwen3-1.7B+outputs/socratic-v1' \
    --judge anthropic:claude-opus-5
```

Blocked on: a trained adapter (needs a CUDA GPU with compute capability >= 7.5 —
see `notebooks/train_colab.ipynb`) and a funded judge credential.

When it runs it writes `results.json`, `results.csv`, `report.md`,
`judge_transcripts.jsonl`, `manifest.json` and `base_vs_tuned.png` here, and
includes representative failures from **both** models.
