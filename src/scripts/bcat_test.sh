# run only BCAT without ablation tests
expid=bcat_baseline
python main.py exp_id=${expid} batch_size=16 data=fluids_arena compile=1 optim=adamw model.flex_attn=1 #dryrun=1
#python main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=16 model.flex_attn=1
#python main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=16 model.flex_attn=1 overfit_test=1
#torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py exp_id=${expid} batch_size=32 data=fluids_sample compile=1 optim=muon model.flex_attn=1
#torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 model.flex_attn=1