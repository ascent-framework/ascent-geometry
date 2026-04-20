# extraction

Scripts for extracting task update vectors from trained adapters.

## Scope

Load a trained LoRA adapter and export update vectors for downstream analysis.

## Registered primary object (`v1.3`)

The registered update vector is:

`u_i = concat(ΔW_A, ΔW_B)` over all LoRA adapter layers

This object is the one used for the registered H1a, H1b, and magnitude
analyses.

## Exploratory / pilot-only objects

Dense effective deltas such as `scaling * (B @ A)` may still be useful for
single-layer diagnostics or exploratory analysis, but they should be saved and
labeled separately from the registered primary object.

## Outputs

- Registered update vector file (`.npy` or similar) per task/model combination
- Extraction metadata: layer coverage, vector shape, checksum, and object type
- Provenance JSON suitable for reproducible reload and audit

## Current entry point

- `extract_registered_update_vector.py`: extracts the registered
  `concat(ΔW_A, ΔW_B)` object from a saved LoRA adapter
