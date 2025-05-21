GPU=0
GPUs=0,1

expid=prose_24 # 1to1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_1to1 model.qk_norm=1 save_periodic=8 optim.lr=7e-4 model.data_encoder.n_layer=6 model.data_decoder.n_layer=12 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_1to1 model.qk_norm=1 model.data_encoder.n_layer=6 model.data_decoder.n_layer=12
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_1to1 model.qk_norm=1 model.data_encoder.n_layer=6 model.data_decoder.n_layer=12 overfit_test=1

expid=prose_25 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 data.t_num=11 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 rollout=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 rollout=1 overfit_test=1



