"""Centralized model path configuration."""

from configs.paths import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

# Current model (v2)
V2_MODEL = RUNS_DIR / "frisbee_det_s_v2" / "weights" / "best.pt"

# Legacy model (v1)
V1_MODEL = RUNS_DIR / "frisbee_det_s" / "weights" / "best.pt"

# v3 model (cleaned data, box=5)
V3_MODEL = RUNS_DIR / "frisbee_det_s_v3" / "weights" / "best.pt"

DEFAULT_MODEL = V3_MODEL
DEFAULT_MODEL_SIZE = "s"
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.35
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 2
SEED = 42
