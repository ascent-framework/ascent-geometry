# Phase 0 Run Note

## Summary

- Date: 2026-04-22 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `main`
- Commit: `cc2ee064294a02781e78c8c56a1c6160a3b231d9`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `GSM8K`
- Method: `GRPO`

## Environment

- Python version: not captured in Kaggle run report
- Torch version: `2.10.0+cu128`
- GPU model: `Tesla T4`
- VRAM: `15.6 GB`
- Precision: `bf16`
- Environment matched `requirements.txt` exactly: not verified in imported report

## Outputs

- Training report: `training_report.json`
- Extraction report: `extraction_report.json`
- Analysis report: `analysis_report.json`
- Analysis summary: `analysis_summary.md`
- Run manifest: `run_manifest.json`
- Adapter artifact location: `/tmp/kaggle-kernel-output/gsm8k-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/tmp/kaggle-kernel-output/gsm8k-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- Kaggle T4 environment completed the full `train -> extract -> SVD` pilot pipeline.
- Adapter save and reload succeeded without dtype-path failure.
- Registered target modules were all covered, and the extracted update vector was non-zero.

## What Failed

- The imported Kaggle report still contains the placeholder notes field and does not record operator-written run commentary.
- Durable artifact URIs were not recorded; current references point to downloaded `/tmp` paths.

## What To Fix Next

- Publish the adapter and vector artifacts to a durable Kaggle Dataset or Drive location and replace the temporary paths in this run record.
- Record Python/package exact versions and any notebook-side pip installs in the next run note.
- Start Phase 1 task collection now that the Phase 0 pipeline has been validated end-to-end.
- Promote the pilot SVD summary into future phase-level notes only after multi-task vectors exist.

## Notes

- Source kernel: `chson0316/ascent-g-phase-0-pilot-gsm8k-qwen2-5-1-5b`
- Source report timestamp: `2026-04-21T15:47:24.149191Z` (`2026-04-22 00:47:24 KST`)
- Update vector SHA-256: `8bb6b927c8b0c39f46016f9d6aaa5eead753faf8a0805e5beff6252970a78420`
