# analysis

SVD-based geometry analysis of task update vectors.

## Scope

Primary registered analysis for H1a and H1b operates on the normalized
multi-task update matrix after at least 10 task vectors exist.

Phase 0 single-task SVD is allowed only as a pipeline diagnostic proving that
the pilot artifact can be consumed by analysis code. It is not evidence for the
registered hypotheses.

## Registered analyses (`v1.3`)

- H1a: compute SVD on the normalized task matrix and report `r_90`
- H1a: compute `rho = r_90 / min(N, sqrt(D_eff))`
- H1a: bootstrap 95% confidence intervals over resampled tasks
- H1b: project onto the shared subspace and compute pairwise cosine similarity

## Inputs

- Registered primary update vectors: `concat(ΔW_A, ΔW_B)`
- Normalized copies for direction analysis
- Unnormalized norms for magnitude analysis

## Outputs

- SVD spectrum for the task matrix
- `r_90`, `rho`, and bootstrap confidence intervals
- H1b pairwise cosine summary after projection
- Plots and summary tables

## Current entry point

- `pilot_svd_diagnostic.py`: dense effective-delta SVD diagnostic for Phase 0
  pipeline validation
- `h1a_h1b_task_matrix.py`: registered multi-task H1a/H1b analysis on
  normalized `concat(ΔW_A, ΔW_B)` vectors

## Example usage

```bash
python analysis/h1a_h1b_task_matrix.py \
  --vector GSM8K=/path/to/gsm8k_update_vector.npy \
  --vector MATH=/path/to/math_update_vector.npy \
  --vector AIME=/path/to/aime_update_vector.npy \
  --vector AMC=/path/to/amc_update_vector.npy \
  --vector MATH500=/path/to/math500_update_vector.npy \
  --vector HumanEval=/path/to/humaneval_update_vector.npy \
  --vector MBPP=/path/to/mbpp_update_vector.npy \
  --vector CommonsenseQA=/path/to/commonsenseqa_update_vector.npy \
  --vector HellaSwag=/path/to/hellaswag_update_vector.npy \
  --vector ARC-Challenge=/path/to/arc_challenge_update_vector.npy \
  --output-path runs/phase1-h1a-h1b-report.json
```
