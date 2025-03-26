GPU=0
GPUs=0,1


# python main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20
# python main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat
# python main.py dryrun=1 batch_size=16 data=arena compile=1 max_epoch=20 model=vq_bcat model.name=vqvae train_vq=1 clip_grad_norm=0
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py dryrun=1 batch_size=28 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_resnet
# python main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=20 model=diffusion ema.enable=1
# python main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=40 model=diffusion ema.enable=1 eval_size=100 model.prediction_type=sample

# expid=bcat_8
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=12 model.norm=dyt &&
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model.norm=dyt &&
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model.norm=dyt overfit_test=1


# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py dryrun=1 batch_size=32 data=arena compile=1 optim=muon max_epoch=20 optim.decay=0.4 save_periodic=4


# expid=bcat_muon_19
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=debug eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena

ds="sw"

CUDA_VISIBLE_DEVICES=$GPU python main.py exp_name=fluids_test dryrun=1 batch_size=64 model=unet max_epoch=5 data=${ds}
