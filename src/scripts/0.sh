GPU=0
GPUs=0,1

expid=prose_8 # muon, prose, qk norm, more warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.4 save_periodic=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 overfit_test=1



