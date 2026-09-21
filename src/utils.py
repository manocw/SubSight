"""Shared helpers: seeding, paths. Full version at Stage 4."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix Python/NumPy/PyTorch seeds for reproducible splits + training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
