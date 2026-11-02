"""Random seeds and deterministic kernels.

A run is reproducible only if every source of randomness is fixed before it is
used: the weight initialization (when the model is built), the order of the
training mini-batches (when the loader is iterated), dropout, and the sampler of
a hyperparameter search. ``set_seed`` fixes the global generators of Python,
NumPy and PyTorch; ``loader_generator`` gives a training loader its own
generator, so that its shuffling does not depend on what else consumed the
global one.

Fixing seeds is not sufficient on an accelerator: several CUDA reductions are
nondeterministic, and MPS offers no deterministic mode. ``deterministic=True``
asks PyTorch for deterministic kernels where they exist (CUDA, CPU) and warns
where they do not. Bitwise reproducibility is guaranteed on CPU only.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch, optionally with deterministic kernels.

    Call it before building the model: the weights are drawn at construction,
    so a seed set afterwards (for instance by ``Trainer(seed=...)``) only fixes
    the batch order and dropout.

    Parameters
    ----------
    seed : int
        Seed for every global generator.
    deterministic : bool, default=False
        If True, request deterministic algorithms from PyTorch
        (``torch.use_deterministic_algorithms(True, warn_only=True)``), disable
        cuDNN autotuning and set ``CUBLAS_WORKSPACE_CONFIG``. Some operators are
        slower in this mode. These settings are process-wide and persist: a
        later call with ``deterministic=False`` does not undo them. cuBLAS reads
        ``CUBLAS_WORKSPACE_CONFIG`` when CUDA starts, so on GPU call this before
        anything touches CUDA.

    Notes
    -----
    ``PYTHONHASHSEED`` is exported for subprocesses only; the hash seed of the
    running interpreter is fixed at startup.

    Returns
    -------
    int
        The seed, for logging.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def loader_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """A CPU generator for a shuffling loader, or None when no seed is given."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
