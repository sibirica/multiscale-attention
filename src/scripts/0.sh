GPU=0
GPUs=0,1

# expid=prose_20 # muon, prose, qk norm
# python main.py eval_only=1 use_wandb=0 exp_name=debug eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=50 model=prose_2to1 symbol.symbol_input=1 data=turb data.com_ns.type=all



# expid=bcat_muon_all_11 # main exp, 10% warmup
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py eval_only=1 use_wandb=0 exp_name=debug eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=64 data=turb data.com_ns.type=all

expid=prose_5 # adam, prose, qk norm
python main.py eval_only=1 use_wandb=0 exp_name=save eval_from_exp=checkpoint/prose_fd/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=50 model=prose_2to1 symbol.symbol_input=1 save_outputs=1 eval_size=1
