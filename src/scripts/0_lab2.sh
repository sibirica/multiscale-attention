GPU=0

# "sw" "cns" "arena_ns" "arena_ns_cond" "ins" "cfdbench"

for ds in "sw" "cns" "arena_ns" "arena_ns_cond" "ins"
do
    expid="don_${ds}"
    CUDA_VISIBLE_DEVICES=$GPU python main.py exp_name=fluids_test exp_id=${expid} batch_size=128 model=deeponet max_epoch=5 data=${ds} &&
    CUDA_VISIBLE_DEVICES=$GPU python main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=128 model=deeponet data=${ds}
done