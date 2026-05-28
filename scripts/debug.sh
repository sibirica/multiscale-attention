train_args=(
    dryrun=5
    data=fluids_sample
    compile=1
    model.flex_attn=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py "${train_args[@]}"