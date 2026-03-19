# baseline

expid=bcat_baseline
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=${expid} batch_size=32 data=fluids_sample compile=1 model.flex_attn=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 overfit_test=1