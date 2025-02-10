# bcat

expid=bcat_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena