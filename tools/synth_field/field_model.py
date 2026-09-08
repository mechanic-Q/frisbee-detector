"""场地模型（世界坐标，米）：纯数据与几何定义，无渲染依赖。"""
import numpy as np

FIELD_W, FIELD_H = 100.0, 37.0
EZ_A, EZ_B = 18.0, 82.0
LINE_WIDTH_M = (0.05, 0.12)
BRICK_XS = (EZ_A + 18.0, EZ_B - 18.0)


def line_segments():
    """边线 4 条 + 得分区线 2 条 → [((x,y),(x,y)), ...]"""
    c = [(0, 0), (100, 0), (100, 37), (0, 37)]
    segs = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    segs += [((EZ_A, 0), (EZ_A, 37)), ((EZ_B, 0), (EZ_B, 37))]
    return segs


def keypoints():
    """8 个训练关键点：4 角 + 2 得分区线端点(各 2 个)"""
    return [(0, 0), (100, 0), (100, 37), (0, 37),
            (EZ_A, 0), (EZ_A, 37), (EZ_B, 0), (EZ_B, 37)]


def brick_marks():
    """4 个砖标 (x, y)"""
    return [(x, y) for x in BRICK_XS for y in (0, 37)]


def corners():
    return np.array([(0, 0), (FIELD_W, 0), (FIELD_W, FIELD_H), (0, FIELD_H)], dtype=np.float64)
