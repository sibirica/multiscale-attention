GPU=0
GPUs=0,1


# python main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20
# python main.py dryrun=1 batch_size=32 data=arena compile=1 optim=wsd max_epoch=20 model=vq_bcat
# python main.py dryrun=1 batch_size=16 data=arena compile=1 max_epoch=20 model=vq_bcat model.name=vqvae train_vq=1 clip_grad_norm=0
# torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py dryrun=1 batch_size=28 data=arena compile=1 optim=wsd max_epoch=20 model=bcat_resnet
# python main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=20 model=diffusion ema.enable=1
# python main.py dryrun=1 batch_size=64 data=arena_u data.t_num=2 input_len=1 compile=1 optim=wsd max_epoch=40 model=diffusion ema.enable=1 eval_size=100 model.prediction_type=sample