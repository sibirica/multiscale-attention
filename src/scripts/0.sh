GPU=0
GPUs=0,1

expid=bcat_muon_all_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon optim.decay=0.5 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 overfit_test=1
