"""Deterministic seeding across all RNG sources.

Every entrypoint MUST call `set_all_seeds(seed)` before any model/data work.
This is the single mechanism for reproducibility; do not seed RNGs anywhere else.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (incl. CUDA) for reproducible runs.

    Args:
        seed: integer seed. We standardize on seeds [0, 1, 2, 3, 4] for the
            5-run mean +/- std reported in the paper.
        deterministic_torch: if True, force deterministic cuDNN algorithms.
            This can slow training slightly but is required for the
            reproducibility claims in CLAUDE.md section 2.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyG scatter ops: opt into deterministic where available.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # older torch
                pass
    except ImportError:
        # torch not installed yet (e.g., during Gate 0 environment setup)
        pass


SEEDS = (0, 1, 2, 3, 4)
"""Canonical seed set. All reported metrics are mean +/- std over these."""
