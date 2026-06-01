## save outputs for visual

# bcat
expid=bcat_1
python src/main.py eval_only=1 use_wandb=0 exp_name=save reload_model=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena save_outputs=1 eval_size=1 data.max_output_dimension=4

# vae + bcat
expid=bcat_2
python src/main.py eval_only=1 use_wandb=0 exp_name=save reload_model=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_resnet save_outputs=1 eval_size=1

# bcat + register tokens
expid=bcat_5
python src/main.py eval_only=1 use_wandb=0 exp_name=save reload_model=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena model=bcat_reg save_outputs=1 eval_size=1

# vanilla bcat + Muon optimizer
expid=bcat_muon_6
python src/main.py eval_only=1 use_wandb=0 exp_name=save reload_model=checkpoint/ts/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=arena save_outputs=1 eval_size=1


