import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Seeds Python, NumPy, and PyTorch for reproducibility.
    Also makes CuDNN deterministic if you want fully consistent runs
    (at the cost of performance).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    DataLoader worker_init_fn to ensure each worker uses a reproducible seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)