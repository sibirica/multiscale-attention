# Repository Guidelines

This repository hosts **BCAT**, a Block Causal Transformer PDE foundation model for fluid dynamics. The codebase is PyTorch-based and configured through Hydra.

## Project Structure

- `src/` — Core Python package (installed as `bcat`).
  - `main.py`, `trainer.py`, `evaluate.py` — entrypoint, training loop, evaluation loop.
  - `dataset.py`, `data_utils/` — dataset wrappers, collation, and conversion scripts (see `src/data_utils/README.md`).
  - `models/` — all model cases, including `bcat.py`, `prose.py`, `transformer.py`, baselines (`fno`, `unet`, `vit`, `deeponet`, `diffusion`, ...).
  - `symbol_utils/`, `utils/`, `vq_utils.py` — symbolic encoders, logging/metrics helpers, VQ utilities.
- `configs/` — Hydra configs: `main.yaml` plus `data/`, `model/`, `optim/` groups.
- `scripts/` — Shell scripts for training and evaluation runs, and archives of past runs.
- `notebooks/` — Jupyter notebooks for visualization and result analysis.
- `checkpoint/` — Run artifacts (gitignored). Model checkpoints, run logs, and intermediate visualizations.
- `docs/skills/` — Project guidelines and conventions

## Build, Test, and Development Commands

Environment setup:
```bash
conda create -n bcat python=3.11 && conda activate bcat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]  # install package + dev tools (ruff, jupyter)
```

Example full training script on 4 GPUs and all datasets:
```bash
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=bcat_baseline batch_size=32 data=fluids_sample compile=1 model.flex_attn=1
```

Example full inference script on 4 GPUs and all datasets:
```bash
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat_test/bcat_baseline log_eval_plots=-1 exp_id=bcat_baseline batch_size_eval=64 model.flex_attn=1
```

Previous training scripts can be found in `scripts/repo_archive.sh` and `scripts/archive.sh`.

## Coding Style & Naming Conventions

Follow these rules for all code changes in this repository:

- Python 3.11, 4-space indentation, `snake_case` for functions/variables, `CamelCase` for classes, lower_snake for modules and config files.
- Format and lint with **ruff** (`line-length = 120`, config in `pyproject.toml`):
  ```bash
  ruff format . && ruff check .
  ```
- Keep new modules under the matching `src/<area>/` subpackage; add a Hydra config in the corresponding `configs/<group>/` folder when introducing new datasets, models, or optimizers.
- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless
  it significantly improves code readability.
- Match existing code style and architectural patterns.

If uncertain, choose the simpler, more concise implementation.

## Testing Guidelines

For quick checks, use `dryrun=1` in `configs/main.yaml`. This runs 5 batches of training and 1 batch of evaluation. Results for dryrun are saved with experiment name `debug`.

## Commit & Pull Request Guidelines

- Run `ruff format` and `ruff check` before committing.
- PRs should describe motivation, list config/script changes, link related issues, and include wandb run links or loss/error plots when modifying training behavior.
