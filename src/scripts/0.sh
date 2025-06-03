GPU=0
GPUs=0,1

expid=prose_ft_8
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py use_wandb=0 exp_name=prose_fd exp_id=${expid} batch_size=16 data=turb compile=1 optim=adamw optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 reload_model=checkpoint/prose_fd/prose_20/best-_l2_error.pth n_steps_per_epoch=2000 model=prose_2to1 symbol.symbol_input=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb model=prose_2to1 symbol.symbol_input=1 
