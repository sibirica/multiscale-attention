GPU=1
GPUs=0,1

expid=bcat_7
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_name=bcat exp_id=${expid} compile=1 max_epoch=20 save_periodic=12 model.flex_attn=1 base_seed=570593307 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=bcat_eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=bcat_eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 overfit_test=1