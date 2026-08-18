# Data Efficiency

> **STATUS: NOT RUN.**

**Minimum Viable Dataset Size: NOT DETERMINED** — no checkpoint has been trained
and evaluated, so it cannot be derived. It is never interpolated or guessed.

This directory is populated by:

```bash
python -m ablations.data_efficiency --plan                 # see the sizes
python -m ablations.data_efficiency --train                # needs a GPU
python -m ablations.data_efficiency --evaluate --judge anthropic:claude-opus-5
```

Blocked on: an accepted dataset (`data/accepted/v1.jsonl`), a CUDA GPU, and a
funded judge credential.
