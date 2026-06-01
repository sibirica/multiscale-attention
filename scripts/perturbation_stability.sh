checkpoint=checkpoint/bcat/multiscale_bcat_34/best-_l2_error.pth
python src/models/tests/test_perturbation_stability.py --checkpoint="${checkpoint}" --samples=10 --repeats=10 --eps=1e-4 --device=cpu
