GPU=1
GPUs=0,1

torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_name=bcat compile=1 max_epoch=20 save_periodic=12 model.flex_attn=1 dryrun=1