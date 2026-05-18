# 追踪 + 场地坐标映射 Pipeline 设计

> **目标：** 在 v3 检测输出上跑 ByteTrack 追踪，用 Homography 将像素坐标映射到飞盘场地真实坐标（100m × 37m），输出标注视频 + 结构化轨迹数据，为后续物理过滤 pipeline 提供基础。
>
> **前置条件：** Homography 标定工具已完成（标定 JSON 存储在 `configs/homography/`）。固定机位视频，只需标定一次。

## 硬件与约束

- **视频源：** 1280×720, 25fps, 固定机位
- **模型：** `runs/detect/frisbee_det_s_v3/weights/best.pt`，conf=0.35
- **追踪器：** YOLOv8 内置 ByteTrack
- **标定文件：** `configs/homography/<video_name>.json`
- **输出：** 标注视频 + CSV 轨迹数据

## 架构

```
v3 模型 (.pt) ──┐
                 ├──→ YOLO.track() → ByteTrack → 每帧检测+track ID
Homography JSON ─┘                          ↓
                                      pixel_to_world()
                                      bbox 底边中点 → (wx, wy)
                                      ↓
                            ┌──→ 输出视频（标注框 + ID + 坐标）
                            │
                            └──→ 输出 CSV（frame, track_id, px, py, wx, wy, conf）
```

## 组件设计

### 1. 追踪引擎（inference/predict_track.py）

主入口脚本，包含：

**参数：**
- `--video`：输入视频路径（必填）
- `--model`：模型路径（默认 v3）
- `--calibration`：标定 JSON 路径（可选，自动查找规则见下文）
- `--conf`：置信度阈值（默认 0.35）
- `--output-dir`：输出目录（默认 `runs/track/<video_name>/`）
- `--no-visualize`：是否跳过生成标注视频（默认生成）

**标定文件自动查找规则：**
1. 如果传了 `--calibration`，直接使用
2. 否则按优先级尝试：
   a. `configs/homography/<video_stem>.json`（精确匹配）
   b. `configs/homography/<video_stem_up_to_first_underscore>.json`（如 `25866279684-1-192_55-56min` → `25866279684-1-192.json`）
3. 找不到则打印警告 `⚠️ No calibration found, output will be pixel-only`，CSV 中 `wx, wy` 留空，**不退出**。

**流程：**
```python
model = YOLO(str(model_path))
calib = load_calibration(cal_path)
matrix = calib["matrix"]

results = model.track(
    source=str(video_path),
    conf=conf_threshold,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    verbose=False,
)

for frame_idx, r in enumerate(results):
    if r.boxes is None or r.boxes.id is None:
        continue
    for box, tid, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
        cx = (box[0] + box[2]) / 2
        by = box[3]  # bbox bottom
        wx, wy = pixel_to_world(matrix, float(cx), float(by))
        # draw + accumulate
```

**关键细节：**
- `stream=True`：逐帧产生结果，避免内存爆炸
- `persist=True`：帧间保持追踪状态
- bbox 底边中点 → 场地坐标（飞盘与地面接触点）

### 2. 视频标注渲染

同脚本内实现，用 OpenCV 在每帧上绘制：
- 每个检测框（绿色矩形）
- 左上角 track ID 编号（如 `#3`）
- 右下角场地坐标（如 `(45.2, 18.5)m`）
- 帧号和检测总数在画面顶部

### 3. CSV 轨迹导出

每帧收集数据，在脚本结束时写入 CSV：

```
frame, track_id, px, py, wx, wy, conf
0, 1, 640.0, 500.0, 50.0, 0.0, 0.52
0, 2, 320.0, 200.0, 25.0, 25.0, 0.38
...
```

### 4. 错误处理

| 场景 | 处理 |
|------|------|
| 标定 JSON 不存在 | 警告并继续，仅输出像素坐标，CSV 中 `wx, wy` 留空 |
| 某帧追踪无结果 | 跳过该帧，不中断 |
| bbox 坐标越界 | clamp 到图像边界后在标注 |
| CSV 写入失败 | print 错误消息，不覆盖可能已存在的同名 CSV |
| Homography 矩阵空 | 跳过坐标映射，只输出像素坐标 |

### 5. 使用方式

```bash
# 基本用法（自动查找标定文件，弱匹配）
python3 inference/predict_track.py --video movie/25866279684-1-192_55-56min.mp4

# 指定标定文件（精确匹配）
python3 inference/predict_track.py \
  --video movie/25866279684-1-192_55-56min.mp4 \
  --calibration configs/homography/25866279684-1-192.json

# 只输出数据不生成视频
python3 inference/predict_track.py \
  --video movie/25866279684-1-192_55-56min.mp4 \
  --no-visualize
```

## 文件变更

| 文件 | 操作 | 职责 |
|------|------|------|
| `inference/predict_track.py` | 新建 | 追踪 pipeline 主入口（约 200 行） |

只有一个文件。坐标映射复用 `utils/homography.py` 的 `pixel_to_world()` 和 `load_calibration()`。

## 不在此设计中的范围

- 物理过滤逻辑（速度/加速度/轨迹连续性 —— 下个阶段）
- 多机位标定（单机位）
- 其他追踪器（只用 ByteTrack）
- 自动标定（需手动标定）
- 追踪器的单元测试（追踪逻辑嵌入在 cv2 循环中，适合集成验证而非单元测试）

## 验证方式

1. 在 55-56min 视频上运行
2. 检查输出视频：框是否跟随飞盘、track ID 是否稳定、场地坐标是否合理
3. 检查 CSV：字段完整、坐标值在 [0,100]×[0,37] 范围内
4. 在 20-23min 视频上重复验证
