GPU=1
GPUs=0,1

expid=fno_test_3
# CUDA_VISIBLE_DEVICES=$GPU python main.py exp_id=${expid} batch_size=64 model=fno amp=0 max_epoch=20 &&
CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=fno amp=0