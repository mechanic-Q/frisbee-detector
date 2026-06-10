# Homography 场地标定工具设计

> **目标：** 创建 Streamlit 工具，在飞盘比赛视频帧上手动标定场地控制点，计算 Homography 变换矩阵，将像素坐标映射到真实场地坐标（100m × 37m），为后续追踪 + 物理过滤 pipeline 提供基础。
>
> **背景：** 行业方案采用 检测→追踪→轨迹验证 策略，物理过滤需要场地坐标。固定机位 + 清晰标线已验证可行（14帧 GLM-4V-Flash 评分 7-8/10）。

## 硬件与约束

- **场地尺寸：** 标准飞盘场地 100m × 37m
- **视频源：** 1280×720, 25fps, 固定机位，全程未切换
- **标定频率：** 只需一次（固定机位，整段视频共用同一个 Homography 矩阵）。**2026-05-15 通过 GLM-4V-Flash 验证（14帧 评分 7-8/10），确认可行。**
- **坐标系：** 左下角 (0, 0) 为原点，X 向右（长边 100m），Y 向上（短边 37m）

## 组件设计

### 1. 标定 UI（tools/calibrate_field.py）

Streamlit 页面功能：

- `streamlit run tools/calibrate_field.py -- --video <path>` 启动
- 自动加载视频，提取第一帧作为标定基准帧
- 用户在图像上点击 4-8 个控制点（场地标线交叉点）
- 每个点弹出输入框，填写对应真实坐标 (world_x, world_y)
- 图像上实时标注已选点位置和编号
- 提供"撤销上一点"和"清空所有点"操作
- 点选完成后点击"计算 Homography"按钮
- 计算后显示：矩阵、重投影误差、各点偏差

交互流程：

```
加载视频 → 显示第一帧 → 用户点击标线交叉点
  → 输入真实坐标 → 实时显示点位
  → 点击"计算 Homography"
  → 显示矩阵 + 误差 + 网格线叠加 + 鸟瞰图
  → 点击"保存标定" → 写入 JSON
```

### 2. Homography 计算（utils/homography.py）

```
函数：
  pixel_to_world(matrix, px, py)    → (wx, wy)
  world_to_pixel(matrix, wx, wy)    → (px, py)
  compute_homography(points)        → matrix, error
  draw_field_overlay(image, matrix) → image（网格线叠加）
  warp_to_birdseye(image, matrix)   → image（鸟瞰图）
```

`compute_homography` 实现：
- 输入：`[(px, py, wx, wy), ...]`
- 4 点 → `cv2.getPerspectiveTransform()` 精确解
- 5+ 点 → `cv2.findHomography(method=cv2.RANSAC, ransacReprojThreshold=3.0)` 过约束
- 输出：3×3 矩阵 + 均方根重投影误差（单位：像素）

### 3. 可视化验证

**网格线叠加：**
- 在原始帧上绘制标准场地线（四条边线 + 两条得分线 + 中线）
- 每条线均匀采样 100 个世界坐标点，用 `world_to_pixel` 投影到图像
- 用绿色绘制，半透明覆盖，与实际标线对比

**鸟瞰图：**
- 目标尺寸：1000×370 px（每米 10 像素）
- 使用 `cv2.warpPerspective()` 将整帧 warp 到俯视角度
- 输出应该呈现长方形场地，边线水平/垂直

**误差报告：**
- 每个控制点的像素偏差表格（实际图像点 vs 矩阵反投影点）
- 均方根误差
- 判断标定质量：误差 < 3px → 优秀，3-8px → 可接受，> 8px → 建议重标

### 4. 数据持久化

存储目录：`configs/homography/`

文件命名：`<video_basename>.json`

结构：
```json
{
  "video": "25866279684-1-192.mp4",
  "image_size": [1280, 720],
  "field_size_m": [100, 37],
  "calibration_frame": 0,
  "points": [
    {"pixel": [x, y], "world": [x, y], "error_px": 0.5},
    ...
  ],
  "matrix": [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1.0]],
  "reprojection_error_px": 0.8
}
```

载入时验证 JSON schema：检查必要字段、矩阵形状、非空 point 列表。

## 文件结构

```
tools/
  calibrate_field.py      # Streamlit 应用（主入口）
utils/
  homography.py           # Homography 计算 + 可视化函数
tests/
  test_homography.py      # 单元测试
configs/homography/       # 标定结果 JSON 存储目录
```

## 错误处理

| 场景 | 处理 |
|------|------|
| 不满 4 个点点击计算 | Streamlit 弹窗提示"至少需要 4 个点" |
| 点共线导致矩阵奇异 | catch OpenCV 异常，提示"控制点近似共线，请调整" |
| JSON 文件无法解析 | 提示文件损坏，建议重新标定 |
| 视频路径不存在 | 启动时 Argparse 验证路径，报错退出 |
| 鸟瞰图尺寸不正确 | 固定 1000×370 输出，不依赖输入 |

## 测试

- `tests/test_homography.py` — 3 个测试
  1. 已知变换的点和矩阵，验证正反变换互逆
  2. 4 个共线点 → 预期抛出错误
  3. 5+ 个随机点 RANSAC → 验证误差小于阈值

## 后续使用方式

```python
from utils.homography import load_homography, pixel_to_world

matrix = load_homography("configs/homography/25866279684-1-192.json")
# 对追踪轨迹每条检测：
world_x, world_y = pixel_to_world(matrix, bbox_center_x, bbox_center_y)
# 计算速度、加速度、位置连续性 → 物理过滤
```

## 不在此设计中的范围

- 追踪器实现（ByteTrack/BoT-SORT）
- 物理过滤逻辑（速度/轨迹连续性检查）
- 自动标线检测（完全手动标定）
- 多机位标定（单机位）
