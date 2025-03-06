GPU=0
GPUs=0,1

expid=bcat_muon_4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=28 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 model=bcat_resnet &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet overfit_test=1