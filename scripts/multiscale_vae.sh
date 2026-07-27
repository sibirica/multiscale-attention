GPU=0
GPUs=0,1

# export CUDA_VISIBLE_DEVICES="${GPUs}"

# expid=multiscale_vae2_8_muon_1
# train_args=(
#     exp_id=${expid}
#     max_epoch=20
#     compile=1
#     batch_size=16
#     data=fluids_bench
#     model=multiscale_bcat
#     model.flex_attn=1
#     model.n_layer=6
#     model.embedder.compression_ratio=8
#     model.embedder.num_res_blocks=2
# )
# test_args=(
#     eval_only=1
#     use_wandb=0
#     log_eval_plots=-1
#     exp_name=eval
#     exp_id=${expid}
#     reload_model=checkpoint/multiscale/${expid}
#     batch_size_eval=64
#     compile=1
#     data=fluids_bench
#     model=multiscale_bcat
#     model.flex_attn=1
#     model.n_layer=6
#     model.embedder.compression_ratio=8
#     model.embedder.num_res_blocks=2
# )
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
# torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1

# sleep 120

expid=multiscale_vae2_16_muon_1
train_args=(
    exp_id=${expid}
    max_epoch=40
    compile=1
    batch_size=8
    data=fluids_bench
    model=multiscale_bcat
    model.flex_attn=1
    model.n_layer=6
    model.embedder.compression_ratio=16
    model.embedder.num_res_blocks=1
)
test_args=(
    eval_only=1
    use_wandb=0
    log_eval_plots=-1
    exp_name=eval
    exp_id=${expid}
    reload_model=checkpoint/multiscale/${expid}
    batch_size_eval=64
    compile=1
    data=fluids_bench
    model=multiscale_bcat
    model.flex_attn=1
    model.n_layer=6
    model.embedder.compression_ratio=16
    model.embedder.num_res_blocks=1
)
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${train_args[@]}" &&
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" &&   
torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py "${test_args[@]}" overfit_test=1