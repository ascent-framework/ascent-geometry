# Dependency Notes

`requirements.txt` is the baseline package set for ASCENT-G scripts.

## Why these pins exist

- `numpy`, `accelerate`, `datasets`, `peft`, `transformers`, and `trl` should
  stay aligned across local and Kaggle workflows to reduce behavior drift.
- `torch` and `bitsandbytes` are pinned as the baseline expectation, but they
  are also the packages most likely to require environment-specific changes.

## Practical rule

Use `requirements.txt` as the default. If your local machine needs a different
`torch` wheel for CUDA, CPU-only, or Apple-specific acceleration, treat that as
an environment override and record it in the run notes.

## Recommended recording

For every real run, log:

- Python version
- Torch version
- CUDA / backend details
- Whether the environment matched `requirements.txt` exactly
