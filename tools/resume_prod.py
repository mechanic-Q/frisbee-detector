"""prod 断点恢复入口（避免 CLI 引号嵌套问题）。"""
from ultralytics import YOLO

YOLO(r"runs/detect/bili_prod_v8sp2/weights/last.pt").train(resume=True)
