# Repository Guidelines

This repository hosts **BCAT**, a Block Causal Transformer PDE foundation model for fluid dynamics. The codebase is PyTorch-based.

## Project Structure

- `src/` — Core Python package (installed as `bcat`).
  - `main.py`, `trainer.py`, `evaluate.py` — entrypoint, training loop, evaluation loop.
  - `dataset.py`, `data_utils/` — dataset wrappers, collation, and conversion scripts (see `src/data_utils/README.md`).
  - `models/` — all model cases, including `bcat.py`, `prose.py`, `transformer.py`, and baselines.
  - `symbol_utils/`, `utils/`, `vq_utils.py` — symbolic encoders, logging/metrics helpers, VQ utilities.
- `configs/` — Hydra configs: `main.yaml` plus `data/`, `model/`, `optim/` groups.
- `scripts/` — Shell scripts for training and evaluation, and archives of past runs.
- `notebooks/` — Jupyter notebooks for visualization and result analysis.
- `checkpoint/` — Run artifacts (gitignored). Model checkpoints, run logs, and intermediate visualizations.
- `docs/skills/` — Project guidelines and other skills.

## Build, Test, and Development Commands

Environment setup:

```bash
conda create -n bcat python=3.11 && conda activate bcat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
```

Example full training script on 4 GPUs and all datasets:
```bash
train_args=(
    exp_id=bcat_baseline
    data=fluids_sample
    compile=1
    model.flex_attn=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}"
```

Example full inference script on 4 GPUs and all datasets (after training):
```bash
test_args=(
    eval_only=1
    use_wandb=0
    log_eval_plots=-1
    exp_name=eval
    exp_id=bcat_baseline
    reload_model=checkpoint/bcat_test/bcat_baseline
    batch_size_eval=64
    model.flex_attn=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}"
```

Previous training scripts can be found in `scripts/archive/paper_repro.sh` and `scripts/archive.sh`.

For quick checks, use `dryrun=X` in `configs/main.yaml`. This runs X batches of training and 1 batch of evaluation. Results for dryrun are saved with experiment name `debug`.

## Coding Style & Naming Conventions

Follow these rules for all code changes in this repository:

- Keep new modules under the matching `src/<area>/` subpackage; add a Hydra config in the corresponding `configs/<group>/` folder when introducing new datasets, models, or optimizers.
- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless it significantly improves code readability.
- Match existing code style and architectural patterns.

If uncertain, choose the simpler, more concise implementation.

## Commit & Pull Request Guidelines

- Run `ruff format` and `ruff check` before committing.
- PRs should describe motivation, list config/script changes, link related issues, and include wandb run links or loss/error plots when modifying training behavior.

## Progress Tracking

You are equipped with a project management skill (`progress-tracking`) to maintain context across sessions. Use this skill to manage the `docs/progress.md` state file.