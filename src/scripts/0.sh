GPU=0
GPUs=0,1

expid=bcat_muon_ft_9
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=32 data=turb compile=1 optim=muon optim.scheduler_type=cosine optim.lr=1e-4 save_periodic=4 max_epoch=1 reload_model=checkpoint/fluids_test/bcat_muon_all_1/best-_l2_error.pth n_steps_per_epoch=500 &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb &&
CUDA_VISIBLE_DEVICES=$GPUs torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb overfit_test=1


