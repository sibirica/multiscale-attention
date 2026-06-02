# python src/main.py exp_id=${expid} batch_size=16 data=arena compile=1 optim=adamw model=multiscale_bcat model.flex_attn=1 dryrun=1

# multiscale bcat
# expid=multiscale_bcat_46
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=${expid} batch_size=16 data=fluids_bench compile=1 optim=muon model=multiscale_bcat model.flex_attn=1
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=multiscale_bcat model.flex_attn=0
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=multiscale_bcat model.flex_attn=0 overfit_test=1

# vanilla bcat
# expid=bcat_baseline_6
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=${expid} batch_size=16 data=fluids_bench compile=1 optim=muon model=bcat model.flex_attn=1
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=bcat model.flex_attn=0
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=bcat model.flex_attn=0 overfit_test=1

# windowed bcat
expid=bcat_windowed
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=${expid} batch_size=16 data=fluids_bench compile=1 optim=muon model=bcat_windowed model.flex_attn=1
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=bcat_windowed model.flex_attn=0
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py eval_only=1 use_wandb=0 exp_name=eval eval_from_exp=checkpoint/bcat/${expid} log_eval_plots=-1 exp_id=${expid} data=fluids_bench batch_size_eval=64 model=bcat_windowed model.flex_attn=0 overfit_test=1

# use data = arena to switch to PDEArena cond_ns