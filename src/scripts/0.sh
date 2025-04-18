GPU=0
GPUs=0,1

expid=bcat_muon_all_9
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 batch_size_eval=40 eval_size=50 data=arena_u compile=1 optim=muon model=bcat_next_token &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u model=bcat_next_token &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u model=bcat_next_token overfit_test=1

sleep 60

expid=bcat_muon_all_10
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=20 data=fluids_sample compile=1 optim=muon model=time_then_space model.n_layer=9 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=time_then_space model.n_layer=9 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=time_then_space model.n_layer=9 overfit_test=1




