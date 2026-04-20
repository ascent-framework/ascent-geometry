# Python Setup

Reproducible local work for `ascent-geometry` should use a dedicated Python
virtual environment.

## Recommended version

- Python `3.11`

The current scripts were written and checked against a Python `3.11` workflow.

## 1. Create a virtual environment

From the repository root:

```bash
python3.11 -m venv .venv
```

If `python3.11` is not available on your machine, install it first and verify:

```bash
python3.11 --version
```

## 2. Activate the environment

```bash
source .venv/bin/activate
```

After activation, verify that `python` and `pip` point into `.venv`:

```bash
which python
which pip
python --version
```

Expected pattern:

- `python` → `<repo>/.venv/bin/python`
- `pip` → `<repo>/.venv/bin/pip`

## 3. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

If `torch` or `bitsandbytes` need a platform-specific build on your machine,
install the matching `torch` build first and then rerun:

```bash
python -m pip install -r requirements.txt
```

Or use the bootstrap helper from the repository root:

```bash
bash scripts/bootstrap_venv.sh
```

## 5. Verify the core experiment stack

```bash
python scripts/verify_env.py
```

If this exits successfully, the minimum local runtime is in place.

## 6. Optional: freeze exact versions

For stronger reproducibility after a known-good install:

```bash
python -m pip freeze > requirements.lock.txt
```

This is useful when you want to rerun the same environment later without
depending on upstream package changes.

## 7. Example commands

Training:

```bash
python training/phase0_gsm8k_grpo.py --help
```

Extraction:

```bash
python extraction/extract_registered_update_vector.py --help
```

Analysis:

```bash
python analysis/pilot_svd_diagnostic.py --help
python analysis/h1a_h1b_task_matrix.py --help
```

Pipeline orchestration:

```bash
python runs/run_phase0_pipeline.py --help
```

## Notes

- Large artifacts should still be written outside the repository working tree
  when running real experiments.
- Kaggle runs may install packages inside the notebook session instead of using
  a local `.venv`, but the package set should match `requirements.txt`.
- `torch` and `bitsandbytes` are the most likely packages to vary across local
  GPU setups. If local training fails on install, keep training on Kaggle and
  use the local environment for extraction and analysis.
- If GPU-backed packages fail to install locally, keep local work limited to
  report generation and analysis, and run training on Kaggle or another GPU
  environment.
