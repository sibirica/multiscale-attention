model=multiscale_bcat
python src/models/tests/test_mixed_precision_probe.py --model=${model} --steps=5 --device=cpu
