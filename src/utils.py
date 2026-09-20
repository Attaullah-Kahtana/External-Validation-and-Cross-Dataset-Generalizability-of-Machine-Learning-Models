
from pathlib import Path
import logging
import os
import random

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found:\n{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative_path):
    return PROJECT_ROOT / relative_path


def get_config_path(config, key):
    return project_path(config["paths"][key])


def ensure_project_directories(config):

    directories = [
        get_config_path(config, "raw_dir"),
        get_config_path(config, "processed_dir"),
        get_config_path(config, "models_dir"),
        get_config_path(config, "results_dir"),
        get_config_path(config, "figures_dir"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name):

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

    return logger