GPU=2
GPUs=2,3

expid=prose_6
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.warmup=0.1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 overfit_test=1