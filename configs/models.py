"""Centralized model path configuration."""

from configs.paths import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

# v2 model (data leak inflated, high precision)
V2_MODEL = RUNS_DIR / "frisbee_det_s_v2" / "weights" / "best.pt"

# Legacy model (v1)
V1_MODEL = RUNS_DIR / "frisbee_det_s" / "weights" / "best.pt"

# v3 model (cleaned data, box=5)
V3_MODEL = RUNS_DIR / "frisbee_det_s_v3" / "weights" / "best.pt"

# v7 model (unified pool 5150 images, box=5, 18.9% bg)
V7_MODEL = RUNS_DIR / "frisbee_det_s_v7" / "weights" / "best.pt"

DEFAULT_MODEL = V7_MODEL
DEFAULT_MODEL_SIZE = "s"
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.35
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 2
SEED = 42
