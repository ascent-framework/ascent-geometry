# extraction

Scripts for extracting task update vectors from trained adapters.

## Scope

Load a trained LoRA adapter, compute the weight delta relative to the base
model, and export a flattened/concatenated update vector for downstream analysis.

## Outputs

- Update vector file (`.npy` or similar) per task/model combination
- Extraction metadata (layer coverage, vector shape, checksum)
