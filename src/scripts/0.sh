GPU=0
GPUs=0,1

expid=bcat_muon_all_24 # flex, rms qk norm, no bias
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 model.bias=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 model.bias=0
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 model.bias=0 overfit_test=1
