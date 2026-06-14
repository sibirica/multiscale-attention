GPU=0
GPUs=0,1

# export CUDA_VISIBLE_DEVICES="${GPUs}"

expid=bcat_vae2_8_muon
train_args=(
    exp_id=${expid}
    data=arena
    max_epoch=20
    compile=1
    batch_size=16
    model.flex_attn=1
    model.n_layer=12
    model.embedder.compression_ratio=8
    model.embedder.num_res_blocks=2
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
    compile=1
    model.flex_attn=1
    model.n_layer=12
    model.embedder.compression_ratio=8
    model.embedder.num_res_blocks=2
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1

sleep 120

expid=bcat_vae2_16_muon
train_args=(
    exp_id=${expid}
    data=arena
    max_epoch=20
    compile=1
    batch_size=16
    model.flex_attn=1
    model.n_layer=12
    model.embedder.compression_ratio=16
    model.embedder.num_res_blocks=2
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
    compile=1
    model.flex_attn=1
    model.n_layer=12
    model.embedder.compression_ratio=16
    model.embedder.num_res_blocks=2
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1

# sleep 120

# expid=bcat_vae2_32
# train_args=(
#     exp_id=${expid}
#     data=arena
#     max_epoch=20
#     compile=1
#     batch_size=16
#     model.flex_attn=1
#     model.n_layer=12
#     model.embedder.compression_ratio=32
#     model.embedder.num_res_blocks=2
# )
# test_args=(
#     eval_only=1
#     use_wandb=0
#     log_eval_plots=-1
#     exp_name=eval
#     exp_id=${expid}
#     data=arena
#     reload_model=checkpoint/bcat/${expid}
#     batch_size_eval=64
#     compile=1
#     model.flex_attn=1
#     model.n_layer=12
#     model.embedder.compression_ratio=32
#     model.embedder.num_res_blocks=2
# )
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1
