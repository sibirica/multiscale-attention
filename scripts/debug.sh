export CUDA_VISIBLE_DEVICES=0,1 

train_args=(
    # exp_id=debug1
    dryrun=20
    data=arena
    compile=1
    model.flex_attn=1
)

torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py "${train_args[@]}"
# python src/main.py "${train_args[@]}"