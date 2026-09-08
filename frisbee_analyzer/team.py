"""分队模块：SigLIP 上半身裁剪 embedding → UMAP(3) → KMeans(2)，半场 x 先验定队号。

蓝本 roboflow/sports team.py。team_id 编码与 tools/auto_label_roles.py 一致：
0/1 = 两队（0=平均 x 较小的一侧），None = 未分配（勿用 -1，events.py 里 -1 是裁判）。
纯函数（聚类/定队号/多数投票）与视频 I/O 分离，前者可无 GPU 单测。
embedding 模型用 HF 缓存的 google/siglip-so400m-patch14-384（transformers）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .tracking import WorkerCancelled

TEAM_SAMPLE_INTERVAL = 60   # 每 N 帧采样一次用于聚类
MAX_CROPS_PER_TRACK = 40    # 单 track 采样裁剪上限，防长 track 主导
TEAM_MAX_CROPS = 1500       # 聚类裁剪总量上限（等距抽样），约束 embedding/UMAP 耗时
MIN_CROPS_FOR_CLUSTER = 16  # 少于此数不聚类（开球前/远景段）
UMAP_DIM = 3
USE_UMAP_MIN_SAMPLES = 100  # UMAP 在小样本上近邻图会塌缩（实测 16 个点两簇被搅混）；小样本直接用 SigLIP 原始空间


class SiglipEmbedder:
    """transformers SigLIP 图像 embedding。惰性加载；BGR ndarray 列表 → L2 归一化矩阵。"""

    MODEL_ID = "google/siglip-so400m-patch14-384"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import SiglipImageProcessor, SiglipModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:  # 优先本地缓存，避免离线机器上 HF HEAD 探测重试卡住启动
            self.model = SiglipModel.from_pretrained(self.MODEL_ID, local_files_only=True)
            self.processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID, local_files_only=True)
        except Exception:
            self.model = SiglipModel.from_pretrained(self.MODEL_ID)
            self.processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device).eval()

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """(N, H, W, 3) BGR uint8 → (N, D) float32，L2 归一化。分批前向防大样本撑爆显存。"""
        import torch
        from PIL import Image

        feats = []
        for i in range(0, len(crops_bgr), batch_size):
            chunk = crops_bgr[i:i + batch_size]
            images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in chunk]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
            if not isinstance(out, torch.Tensor):
                # transformers 5.x 起返回 ModelOutput；pooled 特征即判别性图像表示
                out = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                    else out.last_hidden_state.mean(dim=1)
            feats.append(out.detach().cpu().numpy().astype(np.float32))
        feats = np.concatenate(feats, axis=0)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-8, None)


def crop_upper_body(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """取检测框上半身（上部 45%），球衣颜色主要分布区。太小则返回 None。"""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    y_mid = y1 + int((y2 - y1) * 0.45)
    crop = frame[y1:max(y_mid, y1 + 8), x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return crop


def collect_crops(
    frames: dict,
    video_path: str,
    sample_interval: int = TEAM_SAMPLE_INTERVAL,
    cancel_check=None,
) -> tuple[list[int], list[np.ndarray], list[float]]:
    """按采样间隔读视频帧，产出 (track_ids, 裁剪图, bbox 中心 x)。单 track 有采样上限。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for team sampling: {video_path}")

    per_track_count: Counter[int] = Counter()
    track_ids: list[int] = []
    crops: list[np.ndarray] = []
    xs: list[float] = []
    cur = -1

    for frame_key in sorted(frames, key=int):
        idx = int(frame_key)
        if idx != 0 and idx % sample_interval:
            continue
        if cancel_check is not None and cancel_check():
            cap.release()
            raise WorkerCancelled(f"cancelled sampling at frame {idx}")
        if idx != cur + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        cur = idx
        for det in frames[frame_key]:
            tid = det["track_id"]
            if per_track_count[tid] >= MAX_CROPS_PER_TRACK:
                continue
            crop = crop_upper_body(frame, det["bbox"])
            if crop is None:
                continue
            per_track_count[tid] += 1
            track_ids.append(tid)
            crops.append(crop)
            xs.append((det["bbox"][0] + det["bbox"][2]) / 2)
    cap.release()

    if len(crops) > TEAM_MAX_CROPS:  # 等距抽样，三列表同步
        stride = len(crops) / TEAM_MAX_CROPS
        keep = [int(i * stride) for i in range(TEAM_MAX_CROPS)]
        track_ids = [track_ids[i] for i in keep]
        crops = [crops[i] for i in keep]
        xs = [xs[i] for i in keep]
    return track_ids, crops, xs


def cluster_team_embeddings(embeddings: np.ndarray, n_teams: int = 2) -> np.ndarray | None:
    """≥100 样本：UMAP(UMAP_DIM) 降维 → KMeans(n_teams)；更少：SigLIP 原始空间直接 KMeans。
    样本过少返回 None。umap-learn 缺失时自动降级为不降维。
    """
    if len(embeddings) < MIN_CROPS_FOR_CLUSTER:
        return None
    from sklearn.cluster import KMeans

    feats = embeddings
    if len(embeddings) >= USE_UMAP_MIN_SAMPLES:
        try:
            import umap

            feats = umap.UMAP(n_components=UMAP_DIM, random_state=42).fit_transform(embeddings)
        except ImportError:
            feats = embeddings
    km = KMeans(n_clusters=n_teams, n_init=10, random_state=42).fit(feats)
    return km.labels_


def team_ids_by_x(labels: np.ndarray, xs: np.ndarray) -> dict[int, int]:
    """半场 x 先验：簇平均 x 较小 → team 0，较大 → team 1。跨重拟合保持稳定映射。"""
    means = {}
    for c in np.unique(labels):
        means[int(c)] = float(np.mean(xs[labels == c]))
    ordered = sorted(means, key=lambda c: means[c])
    return {c: i for i, c in enumerate(ordered)}


def majority_vote(per_track_labels: dict[int, list[int]], cluster_to_team: dict[int, int]) -> dict[int, int]:
    """每 track 按出现次数最多的簇号投票，再映射为 team_id。"""
    out: dict[int, int] = {}
    for tid, labels in per_track_labels.items():
        if not labels:
            continue
        cluster = Counter(labels).most_common(1)[0][0]
        if cluster in cluster_to_team:
            out[tid] = cluster_to_team[cluster]
    return out


def label_from_crops(
    track_ids: list[int],
    crops: list[np.ndarray],
    xs: list[float],
    embedder,
    n_teams: int = 2,
) -> dict[int, int]:
    """(track_id, crop, x) 三元组 → {track_id: team_id}。embedding 可注入以便单测。"""
    if not crops:
        return {}
    embeddings = np.asarray(embedder.embed(crops), dtype=np.float32)
    labels = cluster_team_embeddings(embeddings, n_teams=n_teams)
    if labels is None:
        return {}
    mapping = team_ids_by_x(labels, np.asarray(xs, dtype=np.float64))
    per_track: defaultdict[int, list[int]] = defaultdict(list)
    for tid, lab in zip(track_ids, labels):
        per_track[int(tid)].append(int(lab))
    return majority_vote(per_track, mapping)


def assign_teams(
    frames: dict,
    video_path: str,
    cancel_check=None,
    log=print,
    embedder=None,
    sample_interval: int = TEAM_SAMPLE_INTERVAL,
) -> None:
    """就地填充 frames[...][i]["team_id"]（未分配为 None）。失败降级为"不分队"，不拖垮 worker。"""
    try:
        track_ids, crops, xs = collect_crops(frames, video_path, sample_interval, cancel_check)
        if not crops:
            log("team: no crops collected, skip")
            return
        log(f"team: collected {len(crops)} crops / {len(set(track_ids))} tracks")
        if embedder is None:
            embedder = SiglipEmbedder()
        mapping = label_from_crops(track_ids, crops, xs, embedder)
        if not mapping:
            log("team: too few crops to cluster, frames left unassigned")
            return
        for dets in frames.values():
            for det in dets:
                det["team_id"] = mapping.get(det["track_id"])
        log(f"team: assigned {len(mapping)} tracks into teams {sorted(set(mapping.values()))}")
    except WorkerCancelled:
        raise
    except Exception as e:  # noqa: BLE001 —— 分队失败不拖垮整个 worker
        log(f"team: assignment failed ({type(e).__name__}: {e}); frames left unassigned")


def write_team_overrides(doc: dict, track_id: int, team_id: int, doc_path: str | Path) -> dict:
    """GUI 人工改判：更新 team_overrides 并原子写回 JSON（决策即落盘）。"""
    doc.setdefault("team_overrides", {})[str(int(track_id))] = int(team_id)
    p = Path(doc_path)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return doc
