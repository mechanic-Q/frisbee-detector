"""Centralized path configuration.

Reads configs/paths.yaml and resolves environment variable patterns.
Usage:
    from configs.paths import RESEARCH_ROOT, PROJECT_ROOT, DATASETS, VIDEOS, MODELS

Environment variables used:
    RESEARCH_ROOT — root of external research data (default: /mnt/e/firsbee)
    PROJECT_ROOT  — root of this project (default: /mnt/e/frisbee-detector)
"""

import os
import re
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/mnt/e/frisbee-detector"))
_RESEARCH_ROOT = Path(os.environ.get("RESEARCH_ROOT", "/mnt/e/firsbee"))


def _load_paths_yaml() -> dict[str, str]:
    """Load paths.yaml and resolve env-var placeholders."""
    yaml_path = _THIS_DIR / "paths.yaml"
    if not yaml_path.exists():
        return {}

    raw: dict[str, str] = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            def _expand(m: re.Match) -> str:
                var = m.group(1)
                default = ""
                if ":-" in var:
                    var, default = var.split(":-", 1)
                return os.environ.get(var, default)

            value = re.sub(r"\$\{([^}]+)\}", _expand, value)
            raw[key] = value
    return raw


_yaml_paths = _load_paths_yaml()

_POOL_ROOT = Path(os.environ.get("POOL_ROOT", "/mnt/e/frisbee-pool"))

# Convenience accessors
RESEARCH_ROOT = Path(_RESEARCH_ROOT)
PROJECT_ROOT = Path(_PROJECT_ROOT)
POOL_ROOT = Path(_POOL_ROOT)


class Paths:
    """Typed access to dataset/model/video paths."""

    ultimateml_dir = Path(_yaml_paths.get("ultimateml_dir", str(RESEARCH_ROOT / "03_datasets/UltimateML")))
    ultimate_analytics_dir = Path(_yaml_paths.get("ultimate_analytics_dir", str(RESEARCH_ROOT / "03_datasets/ultimate_analytics")))
    frisbee_dataset_dir = Path(_yaml_paths.get("frisbee_dataset_dir", str(RESEARCH_ROOT / "03_datasets/frisbee_dataset")))
    frisbee_tracking_dir = Path(_yaml_paths.get("frisbee_tracking_dir", str(RESEARCH_ROOT / "03_datasets/frisbee-tracking")))
    frisbee_vision_dir = Path(_yaml_paths.get("frisbee_vision_dir", str(RESEARCH_ROOT / "03_datasets/frisbee-vision-project")))

    raw_videos_dir = Path(_yaml_paths.get("raw_videos_dir", str(PROJECT_ROOT / "data/raw")))
    frames_dir = Path(_yaml_paths.get("frames_dir", str(PROJECT_ROOT / "data/frames")))
    datasets_dir = Path(_yaml_paths.get("datasets_dir", str(PROJECT_ROOT / "data/datasets")))


# For backwards-compatible direct access
EXTERNAL = Paths()
