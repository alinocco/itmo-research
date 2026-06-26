"""Shared utilities: logging, seeding, device resolution."""
from __future__ import annotations

import logging
import os
import random

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "textvec") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across random / numpy / torch (if present)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to 'cuda'/'mps'/'cpu'; validate explicit device requests."""
    logger = get_logger("textvec.utils")
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if device == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if device == "cuda":
        if not cuda_ok:
            cuda_built = torch.version.cuda or "none"
            logger.warning(
                "CUDA requested but unavailable (torch CUDA build: %s). "
                "Reinstall GPU PyTorch: pip install --force-reinstall torch torchvision "
                "torchaudio --index-url https://download.pytorch.org/whl/cu124 — using CPU.",
                cuda_built,
            )
            return "cpu"
        return "cuda"

    if device == "mps":
        if not mps_ok:
            logger.warning("MPS requested but unavailable — using CPU.")
            return "cpu"
        return "mps"

    return device
