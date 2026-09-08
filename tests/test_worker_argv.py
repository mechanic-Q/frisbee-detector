"""worker_paths.build_worker_argv 单测（纯标准库，任何 python 环境可跑）。"""

from gui.worker_paths import build_worker_argv, gpu_queue_script, main_checkout_root


def test_argv_wraps_gpu_queue_lock():
    argv = build_worker_argv("E:/x/a.mp4", "E:/runs/out")
    assert argv[0] == "--cd"
    # bash + 主 checkout 的 gpu_run.sh + 任务名 + 原命令
    i = argv.index("bash")
    assert argv[i + 1].endswith("tools/gpu_run.sh")
    assert "frisbee-detector" in argv[i + 1]
    assert argv[i + 2] == "gui-analysis"
    assert argv[i + 3:] == ["python3", "-m", "frisbee_analyzer.pipeline",
                            "--video", "E:/x/a.mp4", "--output-dir", "E:/runs/out"]


def test_argv_weights_and_max_frames():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", weights_win="E:/frisbee-detector/yolo26x.pt",
                             max_frames=300)
    assert "--weights" in argv
    assert argv[argv.index("--weights") + 1] == "/mnt/e/frisbee-detector/yolo26x.pt"
    assert argv[argv.index("--max-frames") + 1] == "300"


def test_argv_without_queue():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", use_gpu_queue=False)
    assert "gpu_run.sh" not in " ".join(argv)
    assert "python3" in argv


def test_team_only_mode():
    argv = build_worker_argv("E:/x/a.mp4", "E:/out", team_only="E:/out/tracks.json")
    assert "--team-only" in argv
    assert "--video" not in argv


def test_paths():
    root = main_checkout_root()
    assert (root / "tools" / "gpu_run.sh").exists()
    assert gpu_queue_script().startswith("/mnt/")
