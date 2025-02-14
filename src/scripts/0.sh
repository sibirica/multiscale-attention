GPU=0
GPUs=0,1

expid=vqvae_1
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=16 data=arena compile=1 max_epoch=20 model=vq_bcat model.name=vqvae train_vq=1 clip_grad_norm=0 &&
python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat model.name=vqvae train_vq=1 &&
python main.py eval_only=1 use_wandb=0 exp_name=ts_eval eval_from_exp=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=vq_bcat model.name=vqvae train_vq=1 overfit_test=1
