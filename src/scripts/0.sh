GPU=0
GPUs=0,1

expid=diff_3
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 max_epoch=20 n_steps_per_epoch=12000 model=diffusion eval_size=100 normalize=range &&
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion normalize=range
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion normalize=range overfit_test=1 

torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=320 data=arena_u data.t_num=14 input_len=1 model=diffusion normalize=range rollout=1