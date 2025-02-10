### Main BCAT model in the paper

expid=bcat
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=wsd &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64


### Main Ablation Studies

# next token prediction variant

expid=bcat_next_token
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=40 batch_size_eval=40 eval_size=50 data=fluids_arena compile=1 optim=wsd model=bcat_next_token &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=fluids_arena model=bcat_next_token &&

# Time-then-Space variant

expid=st
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=24 data=fluids_sample compile=1 optim=wsd model=time_then_space &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=time_then_space

# Patch Size

expid=bcat_patch_size_16
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=80 data=fluids_sample compile=1 optim=wsd model.patch_num=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=8

expid=bcat_patch_size_32
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=80 data=fluids_sample compile=1 optim=wsd model.patch_num=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=4

## Baselines

# FNO

expid=fno
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=64 model=fno amp=0 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=fno amp=0

# UNet

expid=unet
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=160 model=unet &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 model=unet

# DeepONet

expid=don
CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=128 model=deeponet &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=deeponet

# ViT

expid=vit_1 # patch size 8
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=60 compile=1 model=vit &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=vit

expid=vit_2 # patch size 16
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_id=${expid} batch_size=64 compile=1 model=vit model.patch_num=8 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=vit model.patch_num=8