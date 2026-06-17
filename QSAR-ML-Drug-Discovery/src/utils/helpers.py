"""
helpers.py
==========
Shared utility functions: logging setup, reproducibility seeding,
and convenience I/O wrappers used across the QSAR-ML pipeline.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure root logger with a consistent format across the project.

    Parameters
    ----------
    level : int
        Logging level (default logging.INFO).
    log_file : str, optional
        If provided, also write logs to this file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds across numpy, random, and (if available) torch
    for full reproducibility.

    Parameters
    ----------
    seed : int
        Seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str) -> Path:
    """
    Create a directory (and parents) if it doesn't already exist.

    Parameters
    ----------
    path : str
        Directory path.

    Returns
    -------
    Path
        The created/existing Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df, path: str, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=index)


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
