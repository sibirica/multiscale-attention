GPU=0
GPUs=0,1


# python src/main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20
# python src/main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat
# python src/main.py dryrun=1 batch_size=16 data=arena compile=1 max_epoch=20 model=vq_bcat model.name=vqvae train_vq=1 clip_grad_norm=0
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py dryrun=1 batch_size=28 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_resnet
# python src/main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=20 model=diffusion ema.enable=1
# python src/main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=40 model=diffusion ema.enable=1 eval_size=100 model.prediction_type=sample


# CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_name=fluids_test dryrun=1 batch_size=32 data=fno compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=5 reload_model=checkpoint/fluids_test/bcat_muon_all_1/best-_l2_error.pth

# python src/main.py dryrun=1 batch_size=32 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1

# CUDA_VISIBLE_DEVICES=1 python src/main.py dryrun=1 batch_size=8 data=arena compile=1 optim=muon model.patch_num=8

# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py dryrun=1 batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.4 save_periodic=8

expid=bcat_muon_all_11 # main exp, 10% warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=debug eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64

expid=prose_20 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4


expid=prose_ft_1
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_name=prose_fd exp_id=${expid} batch_size=32 data=turb compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 reload_model=checkpoint/prose_fd/prose_20/best-_l2_error.pth n_steps_per_epoch=500 model=prose_2to1 symbol.symbol_input=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb model=prose_2to1 symbol.symbol_input=1 

expid=prose_ft_2
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py exp_name=prose_fd exp_id=${expid} batch_size=32 data=turb compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 n_steps_per_epoch=500 model=prose_2to1 symbol.symbol_input=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb model=prose_2to1 symbol.symbol_input=1 

expid=prose_20 # muon, prose, qk norm
python src/main.py eval_only=1 use_wandb=0 exp_name=save eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=50 model=prose_2to1 symbol.symbol_input=1 save_outputs=1 eval_size=1