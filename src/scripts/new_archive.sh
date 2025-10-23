expid=bcat_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=bcat exp_id=${expid} compile=1 save_periodic=8 model.flex_attn=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=bcat_eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 &&