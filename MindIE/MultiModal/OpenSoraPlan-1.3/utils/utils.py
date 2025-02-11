
import importlib
import random
import torch
import numpy as np


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed