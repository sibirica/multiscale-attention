# train example

expid=bcat_baseline
train_args=(
    exp_id=${expid}
    data=fluids_sample
    compile=1
    model.flex_attn=1
)
test_args=(
    eval_only=1
    use_wandb=0
    log_eval_plots=-1
    exp_name=eval
    exp_id=${expid}
    reload_model=checkpoint/bcat/${expid}
    batch_size_eval=64
    model.flex_attn=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1

# logs

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
    reload_model=checkpoint/bcat/${expid}
    batch_size_eval=64
    model.flex_attn=1
    compile=1
)