import logging
from .logger import create_logger

import re
import sys
import json
import random

import torch
import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path


DUMP_PATH = "checkpoint"
CUDA = True


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


def load_json(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


def initialize_exp(params, write_dump_path=True):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    if write_dump_path:
        get_dump_path(params)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append(f"{x}")
            else:
                command.append(f"'{x}'")
    command = " ".join(command)
    params.command = command + f' --exp_id "{params.exp_id}"'

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # prepare random seed
    if params.base_seed < 0:
        params.base_seed = np.random.randint(0, 1000000000)
    if params.test_seed < 0:
        params.test_seed = np.random.randint(0, 1000000000)

    OmegaConf.save(params, Path(params.dump_path) / "config.yaml")
    params.dump_path = Path(params.dump_path)

    # create a logger
    logger = create_logger(str(params.dump_path / "train.log"), rank=getattr(params, "global_rank", 0))
    logger.info("============ Initialized logger ============")
    logger.info(f"The experiment will be stored in {str(params.dump_path)}\n")
    logger.info(f"Running command: {command}")
    logger.info("")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    if not params.dump_path:
        dump_path = Path(DUMP_PATH)
    else:
        dump_path = Path(params.dump_path)

    # create the sweep path if it does not exist
    sweep_path = dump_path / params.exp_name
    sweep_path.mkdir(parents=True, exist_ok=True)

    # create an ID for the job if it is not given in the parameters.
    if not params.exp_id:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not (sweep_path / exp_id).exists():
                break

        params.exp_id = exp_id

    # create the dump folder / update parameters
    dump_path = sweep_path / params.exp_id
    dump_path.mkdir(parents=True, exist_ok=True)
    params.dump_path = str(dump_path)


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        if len(args) == 1:
            return args[0]
        else:
            return args
    if len(args) == 1:
        return None if args[0] is None else args[0].cuda(non_blocking=True)
    else:
        return [None if x is None else x.cuda(non_blocking=True) for x in args]


def sync_tensor(t):
    """
    Synchronize a tensor across processes
    """
    device = t.device
    t_sync = t.cuda()

    dist.barrier()
    dist.all_reduce(t_sync, op=dist.ReduceOp.SUM)

    return t_sync.to(device)
