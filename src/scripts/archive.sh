## bcat

expid=bcat_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_2
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=28 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_resnet &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet &&

expid=bcat_3
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=28 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_resnet loss_weight=l2 denormalize_for_loss=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet loss_weight=l2 denormalize_for_loss=1 &&

expid=bcat_4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_reg &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_reg &&

expid=bcat_5
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd optim.lr=5e-4 max_epoch=20 model=bcat_reg &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_reg &&

expid=bcat_6
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 compile=1 optim=wsd max_epoch=20 data=arena_u data.t_num=2 input_len=1 && 
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=64 data=arena_u data.t_num=14 input_len=1 rollout=1 model.kv_cache=0

expid=bcat_7
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 compile=1 optim=wsd max_epoch=20 data=arena_u data.t_num=2 input_len=1 normalize=range && 
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 normalize=range

## vqvae

expid=vqvae_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=16 data=arena compile=1 max_epoch=40 model=vq_bcat model.name=vqvae train_vq=1 clip_grad_norm=0 &&
python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat model.name=vqvae train_vq=1 &&

expid=vqvae_2
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=16 data=arena compile=1 max_epoch=40 model=vq_bcat_L model.name=vqvae train_vq=1 clip_grad_norm=0 optim.warmup=0.01 &&
python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat_L model.name=vqvae train_vq=1 &&

expid=vqvae_3
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=16 data=arena compile=1 max_epoch=40 model=vq_bcat_3 model.name=vqvae train_vq=1 clip_grad_norm=0 optim.warmup=0.01 &&
python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat_3 model.name=vqvae train_vq=1 &&

## vq_bcat

expid=vqbcat_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat &&

expid=vqbcat_2
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat &&

## diffusion

expid=diff_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=40 model=diffusion ema.enable=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion &&

expid=diff_2
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 max_epoch=20 n_steps_per_epoch=12000 model=diffusion ema.enable=1 model.prediction_type=sample eval_size=100 clip_grad_norm=200 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion model.prediction_type=sample &&

expid=diff_3
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 max_epoch=20 n_steps_per_epoch=12000 model=diffusion eval_size=100 normalize=range &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion normalize=range
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=320 data=arena_u data.t_num=14 input_len=1 model=diffusion normalize=range rollout=1

expid=diff_4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 max_epoch=20 n_steps_per_epoch=12000 model=diffusion eval_size=100 normalize=range ema.enable=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u data.t_num=2 input_len=1 model=diffusion normalize=range ema.enable=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=320 data=arena_u data.t_num=14 input_len=1 model=diffusion normalize=range ema.enable=1 rollout=1


# test muon
expid=bcat_muon_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_2
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_3
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 optim.lr=5e-4 model=bcat_reg &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_reg &&

expid=bcat_muon_4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=28 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 model=bcat_resnet &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet &&

expid=bcat_muon_5
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_6
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_7
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 optim.weight_decay=1e-1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_8
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 optim.weight_decay=1e-3 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_9
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 optim.weight_decay=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_10
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=28 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=12 model.dropout=0.05 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model.dropout=0.05 &&

expid=bcat_muon_11
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=12 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_12
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 reload_checkpoint=checkpoint/ts/bcat_muon_11/periodic-11.pth optim.decay=0.2 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_13
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 reload_checkpoint=checkpoint/ts/bcat_muon_12/periodic-15.pth optim.decay=0.1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_14
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=12 model.norm=dyt &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model.norm=dyt &&

expid=bcat_muon_15
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.scheduler_type=cosine save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_16
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.5 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_17
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=nadam max_epoch=20 optim.decay=0.5 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_18
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_19
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.scheduler_type=cosine save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_20
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.scheduler_type=cosine optim.lr=2e-3 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena &&

expid=bcat_muon_21
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.scheduler_type=cosine model.qk_norm=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model.qk_norm=0 &&





## train bcat + muon on all fluid datasets
expid=bcat_muon_all_1 # main exp
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon optim.decay=0.5 save_periodic=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 &&
python main.py eval_only=1 use_wandb=0 exp_name=save eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=50 save_outputs=1 eval_size=1

expid=bcat_muon_all_2 # ablation qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon optim.decay=0.5 save_periodic=4 model.qk_norm=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.qk_norm=0 &&

expid=bcat_muon_all_4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon optim.scheduler_type=cosine save_periodic=16 model.qk_norm=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.qk_norm=0 &&

expid=bcat_muon_all_5 # ablation swiglu
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.activation=gelu model.dim_ffn=4096 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.activation=gelu model.dim_ffn=4096 &&

expid=bcat_muon_all_7 # compare time-then-space
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=20 data=fluids_sample compile=1 optim=muon model=time_then_space &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=time_then_space

expid=bcat_muon_all_8 # compare next token
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 batch_size_eval=40 eval_size=50 data=arena compile=1 optim=muon model=bcat_next_token &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_next_token &&

expid=bcat_muon_all_9 # compare next token
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=40 batch_size_eval=40 eval_size=50 data=arena_u compile=1 optim=muon model=bcat_next_token &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena_u model=bcat_next_token &&

expid=bcat_muon_all_10 # compare time-then-space 9 layers
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=20 data=fluids_sample compile=1 optim=muon model=time_then_space model.n_layer=9 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=time_then_space model.n_layer=9 &&

expid=bcat_muon_all_11 # main exp, 10% warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon optim.warmup=0.1 optim.decay=0.5 save_periodic=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 &&

expid=bcat_muon_all_12 # ablation, causal mask
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.kv_cache=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.kv_cache=0 &&

expid=bcat_muon_all_13 # patch size 16
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.patch_num=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=8 &&

expid=bcat_muon_all_14 # patch size 32
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.patch_num=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=4 &&

expid=bcat_muon_all_15 # patch size 16
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon optim.warmup=0.15 model.patch_num=8 save_periodic=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=8 &&

expid=bcat_muon_all_16 # patch size 32
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.patch_num=4 save_periodic=8 optim.scheduler_type=cosine &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=4 &&

expid=bcat_muon_all_17 # patch size 16
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model.patch_num=8 save_periodic=8 optim.scheduler_type=cosine &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=8 &&

expid=bcat_muon_all_19 # patch size 16
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=64 data=fluids_sample compile=1 optim=muon model.patch_num=8 save_periodic=8 optim.scheduler_type=cosine max_epoch=20 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.patch_num=8 &&

expid=bcat_muon_all_20 # main exp, 10% warmup, flex
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 &&

expid=bcat_muon_all_21 # flex, rms qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1

expid=bcat_muon_all_22 # flex, rms qk norm, no bias
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 model.bias=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 model.bias=0

expid=bcat_muon_all_23 # flex, no bias
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 model.bias=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 model.bias=0

expid=bcat_muon_all_24 # flex, rms qk norm, no bias
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon save_periodic=8 model.flex_attn=1 model.bias=0 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1 model.bias=0


## ft bcat on turbulence.

expid=bcat_muon_ft_1
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fno compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=5 reload_model=checkpoint/fluids_test/bcat_muon_all_1/best-_l2_error.pth &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=fno &&

expid=bcat_muon_ft_2
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=fno compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=5 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=fno &&

expid=bcat_muon_ft_9
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=turb compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 reload_model=checkpoint/fluids_test/bcat_muon_all_1/best-_l2_error.pth n_steps_per_epoch=500 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb &&

expid=bcat_muon_ft_10
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=turb compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 n_steps_per_epoch=500 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb &&











## Vit
expid=vit_1
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=64 compile=1 model=vit model.patch_num=8 model.encoder.norm=layer &&
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=vit model.patch_num=8 model.encoder.norm=layer



## prose-fd
expid=prose_1 # prose + adam
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=wsd model=prose_2to1 symbol.symbol_input=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 symbol.symbol_input=1 &&

expid=prose_2 # prose + muon
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 symbol.symbol_input=1 &&

expid=prose_3 # prose + muon + qk norm
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 &&

expid=prose_4
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=72 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 &&

expid=prose_5 # adam, prose, qk norm
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=wsd model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1

expid=prose_6 # muon, prose, qk norm, more warmup
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=prose_fd exp_id=${expid} batch_size=76 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.warmup=0.1 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_7 # muon, prose, qk norm, more warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 model.dropout=0.1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 model.dropout=0.1 symbol.symbol_input=1

expid=prose_8 # muon, prose, qk norm, more warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.4 save_periodic=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_9 # muon, prose, qk norm, more warmup
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.4 save_periodic=8 reload_checkpoint=checkpoint/prose_fd/prose_8/periodic-31.pth optim.weight_decay=5e-4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_10 # muon, prose, qk norm, more warmup, smaller lr
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.5 save_periodic=8 optim.lr=7e-4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_11 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 save_periodic=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_12 # smaller lr, noise
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.5 save_periodic=8 optim.lr=7e-4 noise=1e-4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_13 # smaller lr, noise
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.5 save_periodic=8 optim.lr=7e-4 noise=5e-3 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_14 # smaller lr, noise
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.5 save_periodic=8 optim.lr=7e-4 noise=0.02 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_15 # smaller lr, flip
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.decay=0.5 save_periodic=8 optim.lr=7e-4 flip=1 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1

expid=prose_16 # 1to1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_1to1 model.qk_norm=1 save_periodic=8 optim.lr=7e-4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_1to1 model.qk_norm=1

expid=prose_17 # 1to1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_1to1 model.qk_norm=1 save_periodic=8 optim.lr=7e-4 model.data_encoder.n_layer=10 model.data_decoder.n_layer=8 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_1to1 model.qk_norm=1 model.data_encoder.n_layer=10 model.data_decoder.n_layer=8

expid=prose_18 # 2to1, more encoder
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=6 model.fusion.n_layer=6 model.data_decoder.n_layer=6 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=6 model.fusion.n_layer=6 model.data_decoder.n_layer=6

expid=prose_19 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=6 model.fusion.n_layer=8 model.data_decoder.n_layer=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=6 model.fusion.n_layer=8 model.data_decoder.n_layer=4

expid=prose_20 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid}_rollout batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 rollout=1

expid=prose_21 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=2 model.fusion.n_layer=12 model.data_decoder.n_layer=4 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=2 model.fusion.n_layer=12 model.data_decoder.n_layer=4

expid=prose_22 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=2 model.fusion.n_layer=10 model.data_decoder.n_layer=6 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=2 model.fusion.n_layer=10 model.data_decoder.n_layer=6

expid=prose_23 # muon, prose, qk norm
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=4 model.fusion.n_layer=12 model.data_decoder.n_layer=2 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=12 model.data_decoder.n_layer=2

expid=prose_24 # 1to1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_1to1 model.qk_norm=1 save_periodic=8 optim.lr=7e-4 model.data_encoder.n_layer=6 model.data_decoder.n_layer=12 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_1to1 model.qk_norm=1 model.data_encoder.n_layer=6 model.data_decoder.n_layer=12

expid=prose_25 # rollout
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_name=prose_fd exp_id=${expid} batch_size=40 data=fluids_sample compile=1 optim=muon model=prose_2to1 symbol.symbol_input=1 model.qk_norm=1 optim.lr=7e-4 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 data.t_num=11 &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=prose_fd_eval eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model=prose_2to1 model.qk_norm=1 symbol.symbol_input=1 model.data_encoder.n_layer=4 model.fusion.n_layer=10 model.data_decoder.n_layer=4 rollout=1