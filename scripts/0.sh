GPU=1
GPUs=0,1

export CUDA_VISIBLE_DEVICES="${GPUs}"

expid=bcat_baseline
train_args=(
    exp_id=${expid}
    data=arena
    max_epoch=20
    compile=1
    model.flex_attn=1
)
test_args=(
    eval_only=1
    use_wandb=0
    log_eval_plots=-1
    exp_name=eval
    exp_id=${expid}
    data=arena
    reload_model=checkpoint/bcat/${expid}
    batch_size_eval=64
    model.flex_attn=1
    compile=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py "${train_args[@]}" &&
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py "${test_args[@]}" &&   
torchrun --standalone --nnodes 1 --nproc_per_node 2 src/main.py "${test_args[@]}" overfit_test=1
