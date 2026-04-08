import glob
import os
from typing import Optional

import yaml


def _root_slug(root_path: str) -> str:
    return os.path.basename(os.path.normpath(root_path)) or "analysis"


def _output_dir(root_path: str) -> str:
    out_dir = os.path.join("media", _root_slug(root_path))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _data_artifact_path(root_path: str, filename: str) -> str:
    return os.path.join(_output_dir(root_path), f"{_root_slug(root_path)}_{filename}")


def _plot_artifact_path(root_path: str, filename: str) -> str:
    return os.path.join(_output_dir(root_path), f"{_root_slug(root_path)}_{filename}")


def get_p_c_micro(root_path: str) -> Optional[float]:
    """p_c_micro = 1 / (n_neurons - 1)."""
    for exp_dir in sorted(d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d)):
        meta_path = os.path.join(exp_dir, "metadata.yaml")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            n = meta.get("n_neurons")
            if n is not None and int(n) > 1:
                return 1.0 / (int(n) - 1)
    return None
