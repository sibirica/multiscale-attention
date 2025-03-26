GPU=0

# "sw" "cns" "arena_ns" "arena_ns_cond" "ins" "cfdbench"

for ds in "sw" "cns" "arena_ns" "arena_ns_cond"
do
    expid="fno_${ds}"
    torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py exp_name=fluids_test exp_id=${expid} batch_size=24 model=fno amp=0 max_epoch=5 data=${ds} &&
    torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py eval_only=1 use_wandb=0 exp_name=fluids_eval eval_from_exp=checkpoint/fluids_test/${expid} log_eval_plots=-1 exp_id=${expid} batch_size_eval=32 model=fno amp=0 data=${ds}
done