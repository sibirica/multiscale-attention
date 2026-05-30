# BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics

This repository contains code for the paper [BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://www.arxiv.org/abs/2501.18972). Pretrained weights are available at: https://huggingface.co/felix-lyx/bcat.

The code is based on the repository [PROSE](https://github.com/felix-lyx/prose).

## Install dependencies

```bash
conda create -n bcat python=3.11 && conda activate bcat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # or other compatible versions
pip install -e .[dev]
```

## Run the model

Scripts for reproducing the results in the paper are in `scripts/archive/paper_repro.sh`. All default arguments can be found in the `configs` folder, and are managed using [Hydra](https://hydra.cc/). Distributed training is available via PyTorch Distributed Data Parallel (DDP).

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
    reload_model=checkpoint/bcat/bcat_baseline
    batch_size_eval=64
    model.flex_attn=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}"
```

## Data

The dataset we used are collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena), and [CFDBench](https://github.com/luo-yining/CFDBench). More details about data preprocessing are included in ```src/data_utils/README.md```.

## Citation

If you find our paper and code useful, please consider citing:

[BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://www.arxiv.org/abs/2501.18972)

```
@article{liu2025bcat,
  title={{BCAT}: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics},
  author={Yuxuan Liu and Jingmin Sun and Hayden Schaeffer},
  journal={arXiv preprint arXiv:2501.18972},
  year={2025}
}
```