"""跨平台路径探测：统一 frisbee-detector 项目根定位（Windows / WSL 通用）。
用法: from tools.platform_paths import PROJECT_ROOT, DATASETS, MOVIE
"""
import os
from pathlib import Path


def find_project_root() -> Path:
    env = os.environ.get("FRISBEE_PROJECT_ROOT")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        Path("E:/frisbee-detector"),          # Windows 原生
        Path("/mnt/e/frisbee-detector"),      # WSL
        Path.home() / "frisbee-detector",
    ]
    for c in candidates:
        if (c / "configs").exists() and (c / "tools").exists():
            return c
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data"
DATASETS = DATA / "datasets"
MOVIE = DATA / "movie" if (DATA / "movie").exists() else PROJECT_ROOT / "movie"
RESULTS = PROJECT_ROOT / "results"
