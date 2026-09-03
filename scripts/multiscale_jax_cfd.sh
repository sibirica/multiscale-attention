# python src/main.py exp_id=${expid} batch_size=16 data=jax_cfd compile=1 optim=adamw model=multiscale_bcat model.flex_attn=1 dryrun=1

dataset=jax_cfd
# multiscale bcat
expid=multiscale_jax_cfd_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=${expid} batch_size=16 data=${dataset} compile=1 optim=muon model=multiscale_bcat model.flex_attn=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/multiscale/${expid} log_eval_plots=-1 exp_id=${expid} data=${dataset} batch_size_eval=64 model=multiscale_bcat model.flex_attn=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/multiscale/${expid} log_eval_plots=-1 exp_id=${expid} data=${dataset} batch_size_eval=64 model=multiscale_bcat model.flex_attn=1 overfit_test=1
