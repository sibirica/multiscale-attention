GPU=0
GPUs=0,1


# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20
python main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat