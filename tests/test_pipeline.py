"""pipeline.run 的错误路径单测（无 GPU 依赖）。"""

import argparse

from frisbee_analyzer import pipeline


def _args(**kw):
    base = dict(video="X:/definitely_missing.mp4", output_dir=None, weights="w.pt",
                conf=0.25, imgsz=1280, max_frames=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_missing_video_returns_1():
    assert pipeline.run(_args()) == 1


def test_main_rejects_missing_required_arg():
    try:
        pipeline.main(["--novideo"])
    except SystemExit:
        pass  # argparse exits 2 on usage error
    else:
        raise AssertionError("argparse should exit on missing --video")
