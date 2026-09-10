"""worker_paths.build_worker_argv 单测（纯标准库，任何 python 环境可跑）。"""

import os

from gui.worker_paths import (
    PROJECT_ROOT_GUI,
    build_worker_argv,
    python_exe,
)


def test_argv_wraps_gpu_queue_lock():
    argv = build_worker_argv("E:/x/a.mp4", "E:/runs/out")
    assert argv[:3] == [python_exe(), str(PROJECT_ROOT_GUI / "tools" / "gpu_run.py"),
                        "gui-analysis"]
    assert argv[3:] == [python_exe(), "-m", "frisbee_analyzer.pipeline",
                        "--video", "E:/x/a.mp4", "--output-dir", "E:/runs/out"]
    assert "wsl" not in " ".join(argv).lower()
    assert "/mnt/" not in " ".join(argv)


def test_argv_weights_and_max_frames():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", weights_win="E:/frisbee-detector/yolo26x.pt",
                             max_frames=300)
    assert argv[argv.index("--weights") + 1] == "E:/frisbee-detector/yolo26x.pt"
    assert argv[argv.index("--max-frames") + 1] == "300"


def test_argv_without_queue():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", use_gpu_queue=False)
    assert "gpu_run" not in " ".join(argv)
    assert argv[:3] == [python_exe(), "-m", "frisbee_analyzer.pipeline"]


def test_team_only_mode():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", team_only="E:/out/tracks.json")
    assert "--team-only" in argv
    assert "--video" not in argv


def test_python_exe_env_override():
    old = os.environ.get("FRISBEE_PYTHON")
    os.environ["FRISBEE_PYTHON"] = "py-custom-311"
    try:
        assert python_exe() == "py-custom-311"
        assert build_worker_argv("E:/x/a.mp4", "E:/out", use_gpu_queue=False)[0] == "py-custom-311"
    finally:
        if old is None:
            os.environ.pop("FRISBEE_PYTHON", None)
        else:
            os.environ["FRISBEE_PYTHON"] = old


def test_gpu_run_py_exists():
    assert (PROJECT_ROOT_GUI / "tools" / "gpu_run.py").exists()
    assert (PROJECT_ROOT_GUI / "frisbee_analyzer" / "pipeline.py").exists()
