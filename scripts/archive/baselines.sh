GPU=0
GPUs=0,1

# FNO

expid=fno
CUDA_VISIBLE_DEVICES=$GPU python src/main.py exp_id=${expid} batch_size=64 model=fno amp=0 &&
CUDA_VISIBLE_DEVICES=$GPU python src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=fno amp=0

for ds in "sw" "cns" "arena_ns" "arena_ns_cond" "ins" "cfdbench"
do
    expid="fno_${ds}"
    torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.pyt exp_id=${expid} batch_size=24 model=fno amp=0 max_epoch=5 data=${ds} &&
    torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=32 model=fno amp=0 data=${ds}
done

# UNet

expid=unet
CUDA_VISIBLE_DEVICES=$GPU python src/main.py exp_id=${expid} batch_size=160 model=unet &&
CUDA_VISIBLE_DEVICES=$GPU python src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=256 model=unet

for ds in "sw" "cns" "arena_ns" "arena_ns_cond" "ins" "cfdbench"
do
    expid="unet_${ds}"
    CUDA_VISIBLE_DEVICES=$GPU python src/main.py exp_id=${expid} batch_size=64 model=unet max_epoch=5 data=${ds} &&
    CUDA_VISIBLE_DEVICES=$GPU python src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=unet data=${ds}
done

# DeepONet

expid=don
CUDA_VISIBLE_DEVICES=$GPU python src/main.py exp_id=${expid} batch_size=128 model=deeponet &&
CUDA_VISIBLE_DEVICES=$GPU python src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=deeponet

for ds in "sw" "cns" "arena_ns" "arena_ns_cond" "ins" "cfdbench"
do
    expid="don_${ds}"
    CUDA_VISIBLE_DEVICES=$GPU python src/main.py exp_id=${expid} batch_size=128 model=deeponet max_epoch=5 data=${ds} &&
    CUDA_VISIBLE_DEVICES=$GPU python src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=deeponet data=${ds}
done

# ViT

expid=vit_1 # patch size 8
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_id=${expid} batch_size=60 compile=1 model=vit &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=vit

expid=vit_2 # patch size 16
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_id=${expid} batch_size=64 compile=1 model=vit model.patch_num=8 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=eval reload_model=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=vit model.patch_num=8