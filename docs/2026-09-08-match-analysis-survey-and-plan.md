# 极限飞盘赛事识别软件 — 穷举式调研报告与技术方案

日期：2026-09-08
状态：调研完成并进入多轮实测执行（最后更新 2026-09-11，最新结论见 §13；§11.4 的 RF-DETR 选型建议已被 §13.1 的 D-FINE 结果取代）
范围：5 路并行穷举式开源调研（球员检测/跟踪/分队、飞盘追踪、单应性/测速、事件统计/VLM、GUI/同类软件）+ 本地项目现状盘点。star 数为 2026-09 抓取值。

---

## 0. 结论速览（TL;DR）

1. **极限飞盘视觉分析在开源世界是空白市场。** 没有任何可用的"飞盘球员检测/跟踪/分队/比分"开源项目；最完整的同类项目是一个 0 star 的 [ccheever/frisbee-tracker](https://github.com/ccheever/frisbee-tracker)（MIT，2026-04 新建，架构可参考）和几个学生项目。本项目的飞盘检测器在这个细分领域已是领先水平。
2. **技术路线不需要推倒重来，但需要大幅扩展。** 现有资产（YOLOv8s-P2 飞盘检测 + SAHI + 手动单应性标定 + 单目标 Kalman + Streamlit 复核工具 + GLM-4V 硬负样本流水线）全部保留复用；缺的是球员侧（检测/跟踪/分队）、事件统计层、和真正的产品 GUI。
3. **核心选型**（均有 MIT/Apache 可商用实现，全部有落地先例）：
   - 球员检测：**roboflow/sports 的 DFL 足球权重零样本试跑 → 自标 500–3000 帧微调**
   - 球员跟踪：**ultralytics 内置 BoT-SORT**（固定机位关 GMC、不开 ReID）
   - 队伍分类：**SigLIP 嵌入 + UMAP + KMeans**（照搬 roboflow/sports `sports/common/team.py`）+ 颜色 KMeans 兜底 + 开球半场位置先验校正
   - 飞盘轨迹：**保留 YOLO+SAHI 主路，候选第二路 TrackNetV3（MIT，遮挡修复），离线两遍式轨迹重建 + 抛体物理外推**
   - 速度/距离：**静态单应性（固定机位）+ 世界坐标平滑 + 低空段限定 + 误差条**（单目诚实精度 ±10–20%）
   - 交换手/攻守转换：**几何规则主判定**（Tryolabs 持球算法骨架：最近邻 + 连续 N 帧惯性）+ clip 分类器过滤 + **人工复核队列**
   - 比分：**几何谓词**（接住点 ∈ 对方得分区多边形）+ 每次得分人工一键确认（记分牌 OCR 对业余比赛不适用）
   - GUI：**PySide6 桌面应用**（QVideoSink 逐帧管线 + QGraphicsScene 叠加），分析任务用 QProcess 独立进程 + JSON-lines 进度协议；Streamlit 退居内部工具
   - VLM（可选后置）：**Qwen2.5-VL-7B / GLM-4.1V-9B（量化，16GB 可跑）**，只做镜头切换/丢盘后的场景重理解，不做细粒度判定
4. **一个重要纠错**：此前计划引用的 "UltimateTracker"（CVPRW 2023 极限飞盘跟踪基准）**不存在**——repo 404、无存档、CVSports 2023 论文目录无此文。勿再引用。

---

## 1. 现状盘点（本地项目）

### 1.1 可复用资产

| 资产 | 位置 | 状态 |
|---|---|---|
| 飞盘检测器（10+ 版本，默认 `frisbee_det_p2_shadow_v1`） | `runs/detect/`, `models/train.py` | 可用，P2 小目标架构 |
| SAHI 切片推理 | `inference/predict_video.py` | 可用，但与追踪管线割裂 |
| 单应性标定（4 点 getPerspectiveTransform + 鸟瞰图 + JSON 持久化） | `utils/homography.py`, `tools/calibrate_field.py`, `configs/homography/*.json` | 可用，需升级 RANSAC 多点 |
| 单目标 Kalman 追踪 + 候选评分 + 世界坐标速度拒绝阈值（>25 m/s） | `inference/predict_track.py`, `utils/tracker_utils.py` | 可用，仅单目标 4 状态 |
| 标注复核工具（Streamlit×2 + yololabeler 导出 + GLM-4V 硬负样本） | `tools/review_*.py`, `tools/collect_hard_negatives.py` | 可复用为球员标注底座 |
| 数据规范（`exclude_ranges`、naming-glossary、防泄露规则） | `docs/conventions/` | 继续严格执行 |
| 训练管线（P2 变体、tmux 流程、防 OOM 参数） | `models/train.py` | 直接复用 |

### 1.2 需要修的问题

- `predict_video.py`（SAHI）与 `predict_track.py`（Kalman）是两条割裂管线，追踪不走 SAHI——小目标漏检会断轨迹。
- 文档与代码不一致：rd-journey 称 Kalman 已升 6 状态，实际 `tracker_utils.py` 仍是 `cv2.KalmanFilter(4, 2)`。
- `vlm_review_crops.py`、`collect_hard_negatives.py` **硬编码了 API key**，必须移到环境变量。
- `WIKI.md`、`QMR-web-*`、`package.json`（camofox-browser）是混入仓库的无关 AI 工具链资产，建议移出。
- `AGENTS.md` 部分内容已过时（仍称 v3 为最佳模型；实际默认已是 P2 shadow_v1）。
- `configs/paths.py` 默认 WSL 路径（/mnt/e/...），Windows 原生运行需 env 覆盖。

### 1.3 与目标的差距

| 目标需求 | 差距 |
|---|---|
| GUI | 无产品级 GUI（只有内部复核工具）；Streamlit 做不了视频逐帧叠加（`st.video` 拿不到帧号、无 overlay、后台线程受限） |
| 队伍区分 | 模型只有 frisbee 单类；无球员检测、无 MOT、无分队 |
| 速度/距离 | 有像素→米映射与单点速度，无显式统计输出；无高度误差处理；标定只覆盖 1 个视频且不抗机位移动 |
| 交换手统计 | 完全没有（无持盘人判定） |
| 比分统计 | 完全没有（无得分区事件检测） |

---

## 2. 五路调研核心结论

### 2.1 球员检测 / 多目标跟踪 / 队伍分类

**关键仓库：**

| 仓库 | Star | License | 要点 |
|---|---|---|---|
| [roboflow/sports](https://github.com/roboflow/sports) | 5.3k | MIT | **架构蓝本**。球员/球/守门员/裁判 4 类 YOLO 检测（DFL 德甲数据）、SigLIP 分队、BallTracker+InferenceSlicer 球跟踪、32 关键点单应性、雷达图 |
| [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis) | 1.0k | 无 | 教程级完整管线；两层 KMeans 球衣分队（上半身裁剪→分离背景→全帧 2 簇），可作 fallback |
| [roboflow/trackers](https://github.com/roboflow/trackers) | 3.7k | Apache-2.0 | ByteTrack 干净重实现；官方基准显示 **ByteTrack 在 SportsMOT(73.0)/SoccerNet(84.0) 是最好跟踪器** |
| [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) | 8.3k | **AGPL-3.0** | SOTA MOT 集合 + ReID；AGPL 有传染性，仅在接受开源义务或后置采用 |
| [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | 4.9k | MIT | OSNet ReID；行人权重与体育场景存在显著 domain gap，只作弱线索 |
| [MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 226 | 数据 CC BY-4.0 | 240 序列/160 万框球员跟踪基准，可用于跟踪器评测 |
| [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | 449 | **GPL-3.0** | 最完整学术参考管线（检测+ReID+跟踪+每帧标定+号码 OCR）；**只学思路不可抄码** |
| [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) | — | — | 球衣号 OCR（曲棍球 91.4%）；业余飞盘常无号码，后置 |

**结论：**

- **跟踪**：直接用 ultralytics `model.track(tracker="botsort.yaml")`。固定机位：`gmc_method: none`、`with_reid: False`；`track_buffer≈50`（25fps 下 2 秒遮挡容忍）。球员出画再入场 ID 跳变成为痛点时，再开 ReID（appearance_thresh≥0.85）或换 boxmot。
- **分队**（推荐三层递进）：
  1. 主方案：SigLIP（`google/siglip-base-patch16-224`）嵌入 + UMAP(3) + KMeans(2)，每 60 帧 fit 一次——roboflow/sports 已验证，对宽松队服/花哨图案比裸颜色鲁棒，零标注；
  2. 兜底：上半身颜色 KMeans（abdullahtarek 式），速度快可交叉验证；
  3. 校正：开球 pull 时两队各占半场 → 用 track 平均 x 坐标先验自动翻转 KMeans 标签（SoccerNet GSR 思路）。裁判由检测类直接排除。
- **姿态**：MVP 不需要。二期用 `yolo11s-pose.pt` 零样本试跑，服务持盘判定（手腕关键点距离）和倒地判定。
- **数据量**：先零样本试跑 DFL 足球权重；微调 500–1000 帧可达可用，生产级 1500–3000 帧（含 100–200 个裁判框），用现有 review 工具半自动标注（约 2–5 人日）。

### 2.2 飞盘定位与轨迹重建

**关键仓库：**

| 仓库 | Star | License | 要点 |
|---|---|---|---|
| [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) | 302 | MIT | **候选第二路**。8 帧热力图回归 + InpaintNet 遮挡轨迹修复，羽毛球 Acc 97.5%、25 FPS，活跃维护，预训练权重可迁移 |
| [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) | 190 | MIT | SoccerNet 2023 球跟踪冠军，5 种运动验证泛化，时序一致性推理离线友好 |
| [roboflow/trackers](https://github.com/roboflow/trackers) | 3.7k | Apache-2.0 | 关联器底座（ByteTrack 思想：低分框二次关联，正对半遮挡飞盘） |
| [tmcclintock/FrisPy](https://github.com/tmcclintock/FrisPy) | 43 | MIT | **飞盘飞行动力学模拟器**（Hummel 2003 升力/阻力/力矩模型）——物理外推先验 + 合成轨迹增广 |
| [obss/sahi](https://github.com/obss/sahi) | 5.5k | MIT | 已集成；SAHI+ByteTrack 组合在无人机小目标场景有多个成熟先例 |
| [jhwang7628/monotrack](https://github.com/jhwang7628/monotrack) | 56 | — | 羽毛球 2D/3D 轨迹重建全管线，研究代码 |

**结论：**

- 飞盘社区无成熟方案（UltimateTracker 不存在；其余全是 0–2 star 学生项目）。
- **推荐离线两遍式架构**（我们是赛后分析，可双向利用未来帧——在线方案做不到的遮挡修复能力）：
  - Pass 1：全视频逐帧候选 = YOLO+SAHI（近景高精度框）⊕ TrackNetV3 热图（远景/模糊/遮挡回忆）双路融合；
  - Pass 2：全局轨迹重建 = tracklet 链接（图匹配）+ 双向 Kalman/RTS 平滑 + 抛体物理外推补洞 + 插值。
- 分阶段运动模型：持盘段（挂接持盘人速度）→ 飞行段（重力+气动外推）→ 落地段（强阻尼）→ re-find 判据（外推邻域 + 外观 + 时间连续性）。
- **决策门**：先用羽毛球 ckpt + 自标 5000–10000 帧点标注（1–2 秒/帧，可用现有检测器预标）快速验证 TrackNetV3 迁移收益，再决定是否深度投入；近景大目标 YOLO 路已够，TrackNet 的收益集中在远景小目标与遮挡。
- 训练帧严禁来自测试视频（`exclude_ranges` 规则照旧）。

### 2.3 场地坐标映射与速度/距离

**关键仓库：**

| 仓库 | Star | License | 要点 |
|---|---|---|---|
| [roboflow/sports](https://github.com/roboflow/sports) | 5.3k | MIT | 关键点检测（YOLOv8x-pose 32 点）→ `findHomography(RANSAC)` → ViewTransformer 批量投影 |
| [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib) | 105 | **GPL-2.0** | 点+线联合优化场地配准（WACV 2024），精度高但 license 传染 |
| [NikolasEnt/soccernet-calibration-sportlight](https://github.com/NikolasEnt/soccernet-calibration-sportlight) | 63 | 无 | SoccerNet 2023 冠军 RayMeshField，精度天花板但重 |
| [lood339/two_point_calib](https://github.com/lood339/two_point_calib) | 56 | — | PTZ 相机两点标定法，摇镜头维持标定的轻量参考 |
| [cemunds/awesome-sports-camera-calibration](https://github.com/cemunds/awesome-sports-camera-calibration) | — | — | 本领域最全文献/代码索引 |
| [syncom/psfv](https://github.com/syncom/psfv) | 1 | MIT | 棒球测速：**速度=已知真实距离÷帧数/帧率**，对单应误差最鲁棒的"平均速度"范式 |
| [captainfffsama/LabelHomography](https://github.com/captainfffsama/LabelHomography) | 13 | BSD-3 | 点击两点图标注单应导出 json 的 Qt 工具，交互设计参考 |

**结论：**

- **固定机位：静态单应性一劳永逸**（一次手动标定 + 定期把场地线反投画面目检）。转播摇镜头：镜头切换检测 + 每 N 帧重估（起步甚至可以"切换后人工快标/复用已标定机位库"）；深度单应性（HomographyNet 系）只做帧间平滑，绝对标定必须靠场线锚定，**不引入**。
- 飞盘场只有 6 条直线 + 2 砖标（WFDF：100×37m，得分区 18m 深；美国 USAU 25 码——**场地模板做成可配置**）。转播局部画面自动场线检测欠定，比足球难；**手动 4–8 点标定是当前最优**（全程 <1 分钟），自动关键点检测（照搬 roboflow 模式，8–10 点）二期再说。
- `utils/homography.py` 升级：`getPerspectiveTransform` → `findHomography+RANSAC`，支持 6–8 点（4 角 + 得分区线端点 + 砖标），冗余点给重投影残差自检。
- **速度/距离的天空高度问题（核心局限）**：地面单应只对地面点成立，径向误差 ≈ d·h/(H−h)。例：相机高 20m、飞盘高 3m、水平距 30m → 投影偏差 5.3m；1.5 秒飞行足以制造几十 km/h 假速度。策略：
  1. **只在低空段（h≲2m）报地面速度**（滚地盘、低平传、出手/接盘瞬间）；
  2. 出手点→接盘点"已知距离÷时间"的全程平均速度（psfv 范式）；
  3. 抛体拟合 / 飞盘标准直径 27.5cm 视半径测距反演高度（二期）；
  4. 输出必须带误差条，诚实定位单目精度 ±10–20%（商用 Hawk-Eye 多目才 ±2.5km/h）。

### 2.4 交换手 / 攻守转换 / 比分 / VLM

**关键仓库/证据：**

| 仓库 | Star | License | 要点 |
|---|---|---|---|
| [tryolabs/soccer-video-analytics](https://github.com/tryolabs/soccer-video-analytics) | 305 | MIT | **持球判定算法模板**：持球人=双脚离球最近球员 + 距离阈值；队级惯性（对方须连续 N 帧控球才切换）；传球=球在同队球员间转移 |
| [lRomul/ball-action-spotting](https://github.com/lRomul/ball-action-spotting) | 137 | MIT | SoccerNet 2023 冠军：滑窗 clip 分类 + 峰值检测，工程细节最完整 |
| [arturxe2/t-deed](https://github.com/arturxe2/t-deed) | ≈100 | — | SoccerNet 2024 冠军，单帧级事件定位（为"视觉上极相似的相邻帧"设计，与接住/触地瞬间同构） |
| 羽毛球回合检测文献（斯坦福 CVPR'22 轨迹重建等） | — | — | 共识：**用球的轨迹断点（速度突变/落地）定义回合边界，比识别动作可靠** |
| [avishah3/AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker) | ≈1k | MIT | 球-篮圈几何关系判进球——"几何谓词判得分"的先例 |
| [royshil/scoresight](https://github.com/royshil/scoresight) | 139 | MIT | 记分牌 OCR（已停更）；业余比赛无记分牌，**不作为比分主路线** |
| [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) / [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) | — | Apache-2.0 / MIT | 16GB 量化可跑的本地 VLM |
| [MMVP "Eyes Wide Shut"](https://arxiv.org/abs/2401.06209) | — | — | **关键证据**：SOTA VLM 在空间关系等基础视觉模式上系统性失明 → "盘是否在得分区内被接住"不能交给 VLM |
| [UltiAnalytics](https://www.ultianalytics.com/) | — | 闭源 | 飞盘社区现役统计范式：pass/drop/D/throwaway/goal 事件体系 + 人工事后改判习惯——本产品即其自动化 |

**结论（四层混合架构，每层有证据支撑）：**

- **第一层（主判定，几何+轨迹规则）**：持盘人 = 盘中心落入球员框手部区域（世界坐标距离阈值）+ 盘速低于阈值 + 连续 N 帧 + 惯性。飞盘"持盘"是离散布尔态（比足球模糊的"控球"更容易判准）。**交换手事件 = 持盘人 ID 变化**；**turnover = 变化且跨队**；盘触地/出界由轨迹形态学判；pull（守方端区长传开盘）触发进攻方向重置。
- **第二层（候选过滤）**：对几何引擎候选窗口（±2s）跑 16 帧 clip 二分类（接住/掉盘），VideoMAE-Kinetics 微调每类 50–200 clip 即可用（20–30 场比赛的事件时刻点击标注即可凑齐）。
- **第三层（低频兜底，可选 VLM）**：仅三种时刻调用本地 VLM——镜头切换后重理解控盘方、盘长时间丢失、长遮挡结束。输出只作为候选事件+证据帧进复核队列，**不进判定链**。
- **第四层（人工复核队列，产品核心）**：低置信度事件按时间轴确认/改判/删除（Label Studio 预标-确认协议的交互范式），修正回写统计。**比分每次得分候选弹一键确认**——比分是记账型数据，错误代价最高，人工确认是产品信任的锚点。

### 2.5 GUI 与整体软件形态

**关键仓库：**

| 仓库 | Star | License | 要点 |
|---|---|---|---|
| [Kinovea/Kinovea](https://github.com/Kinovea/Kinovea) | 491 | **GPL-2.0** | 最接近的成熟开源品：播放器+标绘+跟踪+标定测速桌面软件（C#）。架构教材，代码不可抄 |
| [CVHub520/X-AnyLabeling](https://github.com/cvhub520/x-anylabeling) | ≈7.5k | GPL-3.0 | Python/Qt 视频标注交互最佳参考（播放器+帧标注+AI 辅助） |
| [cvat-ai/cvat](https://github.com/cvat-ai/cvat) | 16.7k | MIT | Web 形态天花板，交互范式可参考 |
| [roboflow/supervision](https://github.com/roboflow/supervision) | 49.9k | MIT | 检测结构/标注器/区域分析工具箱 |
| [PyAV](https://github.com/PyAV-Org/PyAV) | — | — | 帧精确 seek；OpenCV 顺序读够推理用 |
| pyqtgraph | — | MIT | Qt 原生实时图表（速度曲线/时间线） |

**结论：**

- **形态定为 PySide6 桌面单应用（本地优先）**。理由：核心界面是"视频播放+逐帧叠加+修正标注"，这是 Qt 系工具（X-AnyLabeling/Kinovea/LabelImg）验证十年的场景；Streamlit 的 `st.video` 拿不到帧号、无 overlay、后台线程拿不到 ScriptRunContext——三大硬伤正好全踩；FastAPI+Vue 要养两套技术栈；纯 Python 与现有 13 个脚本零摩擦；PySide6 LGPL 无商用风险。
- 视频渲染：**QVideoSink → QImage → QPainter 逐帧管线**（AraViQ6 封装可起步），叠加层用 QGraphicsScene（轨迹、速度标签、比分板、标定交互）。
- **分析任务用 QProcess 拉独立 worker 进程**，JSON-lines 协议报进度/取消——相当于 GUI 内建版 tmux，worker 崩溃不拖垮 UI，天然规避 GIL/GPU 冲突；不引入 Celery/Redis（单机桌面过重）。
- 视频导出：画帧 → rawvideo 管道喂 ffmpeg（libx264 veryfast 或 `h264_nvenc`），比 `cv2.VideoWriter` 快约 2 倍。
- 存储：SQLite（项目/视频/事件/复核状态）+ 每视频 JSON（COCO-video 风格轨迹 + 自定义事件层），可导出 MOT 格式；事件字段沿用现有 naming-glossary（`review_status`、`reviewer_decision` 等）。

---

## 3. 总体架构设计

```
┌────────────────────────── PySide6 桌面应用 ──────────────────────────┐
│  播放器+叠加层    时间线事件轨    复核对话框    仪表盘(pyqtgraph)      │
│  (QVideoSink+    (得分/turnover/  (低置信度     (比分板/速度曲线/     │
│   QGraphicsScene  交换手标注)      事件队列)     场地热图)            │
└───────────────┬──────────────────────────────────────────────────────┘
                │ QProcess + JSON-lines（进度/取消/结果）
┌───────────────▼──────────── 分析 worker 进程 ───────────────────────┐
│ video_io:  解码(OpenCV顺序/PyAV seek) → 逐帧 → ffmpeg pipe 导出叠加视频│
│ engine:    ①飞盘检测(YOLO-P2+SAHI) ②球员检测(YOLO 4类)               │
│            ③BoT-SORT球员跟踪 ④SigLIP分队 ⑤轨迹重建(离线两遍+物理外推) │
│            ⑥单应性投影(utils/homography升级版)                       │
│ events:    持盘人判定 → 交换手/turnover → 得分谓词 → pull重置        │
│            → clip分类器过滤 → 置信度分级                             │
│ storage:   SQLite(项目/事件/复核) + JSON(轨迹/事件) → 统计视图        │
└──────────────────────────────────────────────────────────────────────┘
          可选：本地 VLM 服务(Qwen2.5-VL-7B/GLM-4.1V-9B) 只吃候选片段
```

**与现有仓库的关系**：不另起炉灶。新增 `frisbee_analyzer/` 包（video_io/engine/events/tasks/storage/ui 模块），复用 `utils/homography.py`（升级）、`models/train.py`（扩类训练）、`tools/review_*`（内部标注工具）、现有飞盘检测权重。GUI 与推理引擎解耦，引擎各步骤可独立用 CLI 跑（保持脚本文化）。

---

## 4. 技术选型决策表

| # | 需求 | 选型 | 备选（触发条件） | 关键理由 |
|---|---|---|---|---|
| 1 | GUI | PySide6 + QVideoSink 逐帧 + QGraphicsScene + QProcess worker | FastAPI+Vue（未来多人协作） | 视频+叠加是 Qt 验证场景；Streamlit 有硬伤；LGPL 安全 |
| 2a | 球员检测 | roboflow DFL 权重零样本 → 自标微调（COCO person 预训练起步） | RF-DETR | 500–3000 帧微调即可，现有训练管线复用 |
| 2b | 球员跟踪 | ultralytics BoT-SORT（内置） | boxmot StrongSORT+OSNet（出画重入痛点时）；roboflow/trackers（要 Apache 干净依赖时） | 零集成成本；SportsMOT 基准支持 ByteTrack 系 |
| 2c | 队伍分类 | SigLIP+UMAP+KMeans（60 帧 fit）+ 颜色 KMeans 兜底 + 开球位置先验校正 | CLIP 零样本 prompt | 无监督零标注；对花哨队服鲁棒；roboflow 代码可搬（2–3 人日） |
| 3a | 飞盘定位 | 现有 YOLO-P2 + SAHI（主路） | +TrackNetV3 第二路（5k 帧验证门通过后） | 已验证资产；TrackNet 补远景/遮挡 |
| 3b | 轨迹重建 | 离线两遍式：tracklet 链接 + 双向 Kalman/RTS + 抛体外推 + 插值 | FrisPy 全气动模型（外推精度不够时） | 赛后分析独享双向信息；FrisPy 方程可作先验 |
| 3c | 速度/距离 | 静态单应（固定机位）+ 低空段地面速度 + 已知距离平均速度 + 误差条 | 视半径测距/抛体拟合反演高度（精度不够时）；关键点自动标定（转播摇镜头） | 高度误差 ≈d·h/(H−h) 必须显式处理；诚实精度 ±10–20% |
| 4a | 交换手/turnover | 几何规则（手部区域+低速+N 帧惯性）主判定 + clip 分类器过滤 | VLM 兜底（仅候选发现） | Tryolabs 模板 + 轨迹断点共识；飞盘持盘是布尔态更易判 |
| 4b | 比分 | 几何谓词（接住点∈对方得分区）+ pull 后解锁 + 人工一键确认 | 记分牌 OCR（业余场景不适用，弃） | 得分定义本身是几何谓词；人工确认是信任锚点 |
| 5 | VLM（可选后置） | Qwen2.5-VL-7B（Apache）或 GLM-4.1V-9B（MIT），vLLM 量化部署 | MiniCPM-V 4.5（高帧率扫描） | 只做镜头切换/丢盘重理解；MMVP 证据禁止其做细粒度判定 |

**License 红线**：可自由集成——roboflow/sports、supervision、trackers、ByteTrack/BoT-SORT、torchreid、MMPose、TrackNetV3、WASB-SBDT、FrisPy、PySide6（均 MIT/Apache/BSD/LGPL）。**注意**——ultralytics 为 **AGPL-3.0**（闭源商用需授权，或评估 RF-DETR/其他框架重训）；boxmot、sn-gamestate、PnLCalib、Kinovea、X-AnyLabeling 为 GPL/AGPL（只学不抄）；VideoMAE 权重 CC-BY-NC（商用选 SlowFast/TimeSformer）。

---

## 5. 实施路线图

> 原则：每个阶段以"在 1–2 个真实视频上的可验收门"结束；先跑通全管线（精度可以差），再逐段提精度；所有阶段复用 tmux 训练流程与 exclude_ranges 防泄露规则。

### Phase 0 — 地基与验证实验（约 1 周）

- [ ] 仓库清理：API key 移出源码进环境变量；无关资产（QMR-web、WIKI.md、package.json）移出或归档；AGENTS.md 更新至真实状态（默认模型、Kalman 状态数）
- [ ] `utils/homography.py` 升级：findHomography+RANSAC、6–8 点、重投影残差、可配置场地模板（WFDF/USAU）
- [ ] **实验 E1（关键决策点）**：roboflow DFL 足球球员检测权重在本项目比赛视频上零样本试跑，统计 P/R（用 review_web 抽 100 帧人工核）
- [ ] **实验 E2**：现有飞盘检测器 + BoT-SORT（ultralytics model.track）在测试视频跑通，观察 ID 稳定性
- [ ] 产出决策：球员检测走"直接微调"还是"重标数据"；机位是固定还是摇镜，决定标定策略

### Phase 1 — 球员管线 + GUI 骨架（约 2–3 周）

- [ ] 球员检测器微调（frisbee+player+referee 多类或独立模型，按 E1 结果定），500–1000 帧起步
- [ ] BoT-SORT 球员跟踪接入（固定机位配置），输出 (track_id, bbox) 序列
- [ ] SigLIP 分队模块（移植 roboflow/sports `team.py`）+ 开球位置先验校正 + 人工在 GUI 改判入口
- [ ] **PySide6 GUI v0**：打开视频 → QProcess 跑分析（进度条/取消）→ 播放器叠加球员框+队伍色 → 场地 4/6 点标定交互（升级 calibrate_field 逻辑进桌面端）→ 保存标定
- [ ] 验收门：一段 10 分钟视频端到端，分队准确率人工抽检 ≥90%，跟踪 ID 跳变可视评估

### Phase 2 — 飞盘轨迹与速度/距离（约 2–3 周）

- [ ] 统一管线：SAHI 候选 → 飞盘轨迹重建（离线两遍式：tracklet 链接 + 双向 Kalman + 抛体外推补洞）
- [ ] 速度/距离统计输出：世界坐标平滑、低空段地面速度、出手→接盘平均速度、**误差条**；数据结构按 MOT/COCO-video 风格落 JSON
- [ ] GUI：轨迹叠加回放、速度标签、轨迹长度/瞬时速度曲线（pyqtgraph）
- [ ] **实验 E3（TrackNet 决策门）**：预标 5k–10k 帧点标注，TrackNetV3 羽毛球 ckpt 微调，对比双路 vs 单路召回——通过则并入主路，否则搁置
- [ ] 验收门：标注 3 段 ground-truth 传盘，速度误差与轨迹完整率量化报告

### Phase 3 — 事件统计（交换手/比分）+ 复核闭环（约 2–3 周）

- [ ] 持盘人判定引擎（手部区域+低速+N 帧惯性+世界坐标阈值）→ 交换手/turnover 事件流
- [ ] 得分谓词（接住点 ∈ 对方得分区多边形 + 进攻方向）+ pull 检测重置 + 得分后锁定
- [ ] clip 分类器（VideoMAE-Kinetics 或 T-DEED）过滤候选事件（20–30 场事件点击标注，50–200 clip/类）
- [ ] GUI：时间线事件轨、低置信度复核队列（确认/改判/删除/回写统计）、比分板、交换手统计表
- [ ] 导出：比赛统计报告（JSON/CSV）+ 叠加标注视频（ffmpeg NVENC）
- [ ] 验收门：1 场完整比赛 vs 人工记录（UltiAnalytics 式记法），交换手计数误差、比分 100% 正确（经人工确认后）

### Phase 4 — 鲁棒性增强（可选/按需）

- [ ] 本地 VLM（Qwen2.5-VL-7B/GLM-4.1V-9B 量化，vLLM）：镜头切换重理解、丢盘候选发现，仅产复核建议
- [ ] 转播摇镜头：镜头切换检测 + 每关键帧重估单应性（关键帧标定 + LK 光流传播 + 定期重锚）；二期飞盘场关键点检测模型（8–10 点，照搬 roboflow 模式）
- [ ] 姿态估计（yolo11s-pose 零样本→微调）：持盘判定加入手腕关键点、倒地判别
- [ ] ReID（boxmot/OSNet 弱线索）：球员出画重入 ID 保持
- [ ] FrisPy 气动模型替代简化抛体外推；视半径测距（盘径 27.5cm）反演高度提升测速精度

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| 飞盘高度导致的测速误差被用户误解 | 高 | UI 显式标注"低空速度/平均速度"+误差条；文档写明单目精度边界 |
| 转播摇镜头使静态标定失效 | 高 | Phase 0 先确认素材类型；摇镜头素材走"镜头切换+重标/机位库"路线，关键点模型后置 |
| 球员跟踪 ID 跳变污染分队与持盘统计 | 中 | 队级惯性规则吸收短时跳变；队属按 track 聚类而非逐帧；复核队列兜底 |
| 遮挡导致飞盘长丢、交换手漏计 | 中 | 离线两遍式重建 + 物理外推 + clip 分类器找疑似交接；复核队列把"疑似"呈给用户 |
| ultralytics AGPL 对闭源商用的限制 | 中 | 当前非商业阶段无影响；商业化前评估买授权或迁移 RF-DETR |
| 业余比赛画质差/无号码/无记分牌 | 中 | 架构已按"无记分牌"设计；号码/OCR 全部后置为可选 |
| 训练数据泄露测试视频（历史踩坑） | 中 | 沿用 exclude_ranges + 提交前 find 校验，规则写进新模块的 checklist |

---

## 7. 附录：调研覆盖的主要仓库清单

**综合/足球分析**：roboflow/sports（5.3k）、abdullahtarek/football_analysis（1k）、tryolabs/soccer-video-analytics（305）、SoccerNet/sn-gamestate（449, GPL）、FootballAnalysis/footballanalysis（223）、Gsak3l/Sports-Analysis-Software（53, PySide6+DeepSort）

**跟踪/MOT**：roboflow/trackers（3.7k）、FoundationVision/ByteTrack（6.7k）、NirAharon/BoT-SORT（1.5k）、mikel-brostrom/boxmot（8.3k, AGPL）、MCG-NJU/SportsMOT（226）、KaiyangZhou/deep-person-reid（4.9k）、shallowlearn/sportsreid（26）

**球/飞盘追踪**：qaz812345/TrackNetV3（302）、TrackNetV4（96）、nttcom/WASB-SBDT（190）、jhwang7628/monotrack（56）、tmcclintock/FrisPy（43）、obss/sahi（5.5k）、asigatchov/fast-volleyball-tracking-inference（67）、OrcustD/RacketVision（88）

**标定/测速**：mguti97/PnLCalib（105, GPL）、NikolasEnt/soccernet-calibration-sportlight（63）、MM4SPA/tvcalib（52）、Spiideo/soccersegcal（42）、shaheedmalik/tlNet（50）、lood339/two_point_calib（56）、cemunds/awesome-sports-camera-calibration、syncom/psfv、BaseballCV（13）、vcg-uvic/sportsfield_release（77）

**事件/VLM**：lRomul/ball-action-spotting（137）、SoccerNet/sn-spotting（≈600）、arturxe2/t-deed（≈100）、avishah3/AI-Basketball-Shot-Detection-Tracker（≈1k）、royshil/scoresight（139）、Qwen2.5-VL-7B / Qwen3-VL / InternVL3-8B / MiniCPM-V 4.5 / GLM-4.1V-9B、simula/SoccerChat、happyharrycn/actionformer_release（≈1k）、facebookresearch/SlowFast、TimeSformer

**GUI/标注**：Kinovea（491, GPL）、CVHub520/X-AnyLabeling（≈7.5k, GPL）、cvat-ai/cvat（16.7k）、HumanSignal/label-studio（≈19k）、roboflow/supervision（49.9k）、captainfffsama/LabelHomography（13）、jaseg/python-mpv（≈4.5k）、AraViQ6、pyqtgraph

**飞盘垂直**（全部极早期）：ccheever/frisbee-tracker（0, MIT, 架构最像）、shunsuke-iwashita/VTCS（2, UltimateTrack 数据集待发布）、phuang1024/Discam（0, AGPL）、saundesh/frisbee-tracking（2）、scoolerh/Ultimetrics、Stanford CS231n "Ultimate Vision"、bbwieland/fRisbee（9, R 语言统计包）、UltiAnalytics（商业 app，事件体系参考）

> 注：此前提及的 "UltimateTracker"（CVPRW 2023）经核实不存在（repo 404、无 Wayback 存档、CVSports 2023 论文集无此文），已从所有规划中移除。

---

## 8. 附录：2026-09-08 首轮实测（B 站决赛视频）

测试素材：[2024城市飞盘俱乐部锦标赛决赛 北京大哥 VS 上海沪蛙](https://www.bilibili.com/video/BV1hTtpeyEzb/)（47min13s，852×480@30fps，匿名可下载的最高清晰度；1080P 需登录 cookies）。工具链：**BBDown**（yt-dlp 被 B 站 playurl API 412 风控拦截，浏览器 cookies 解密失败）。测试工件：`data/bili_final_test/`。

### 8.1 素材勘察结论

- 边线单机位、会**摇移跟随比赛**（非全固定）→ 单应性按镜头段处理，证实现有方案风险项。
- 红队 vs 蓝队队服颜色分明 → 颜色聚类分队可行。
- 有转播记分牌 overlay。**用户明确：记分牌只做可选交叉验证，所有统计主判定必须算法自洽（几何/轨迹），因为不是所有比赛都有记分牌。**
- 场上 14 人 + 边线大量观众/替补同框 → 球员 vs 观众需按场地多边形区域过滤。

### 8.2 E1 球员检测零样本对比（2 帧人工计数 ~25 / ~14）

| 方案 | probe_60s | probe_1200s | 速度 | 结论 |
|---|---|---|---|---|
| DFL 足球权重 全帧 conf0.10 | 15 | 7 | ~25ms | **不可用**：漏检近半，前景大目标也漏 |
| DFL + SAHI(2x) | 11 | 1 | ~600ms | 更差 |
| yolov8x COCO person 全帧 | 28 | 17 | ~25ms | 覆盖良好 |
| yolov8x + SAHI(2x) | 31 | 21 | ~250ms | 覆盖最好之一 |
| yolo11x 全帧 / SAHI | 30 / 31 | 20 / 18 | ~25 / ~260ms | 同 v8x |
| yolo26x 全帧 / SAHI | 28 / 32 | 20 / 19 | ~26 / ~250ms | 同 v8x |

**结论：**
1. roboflow DFL 足球权重不适用本素材（其训练域是转播特写，我们是全景边线机位）——E1 原假设反转，**改走 COCO person 预训练 + 自标微调**（分类别：player/referee）。
2. **YOLO 版本问题（用户提问）**：零样本 person 层面 v8x/11x/26x 检出数几乎一致（28-32），架构差异在此不明显；"x" 来自预训练权重底座而非选型偏好。**架构选型推迟到微调基准测试**（frisbee 重训时 v8s-P2 vs yolo11/26-p2 正面对比），注意 ultralytics 全系 AGPL-3.0，闭源商用备选 RF-DETR（Apache-2.0）。
3. 观众同框被正确检出为 person（无害，由场地多边形过滤解决）。

### 8.3 飞盘检测 480P 表现（现有 p2_shadow_v1 + SAHI，60s 片段 900 帧）

- 16.7% 帧有检测（150 帧/152 个检测），但**置信度 top10 的检测放大后全部是草纹误检**（最高 0.90 也是 FP）。
- 原因：训练域为 1080P 级画面，480P 下盘仅 ~10px 且草纹理成为混淆源。
- **行动项**：a) 从本视频收集 bbox 级硬负样本（草纹）入训练池；b) 训练加降分辨率/低质量增广；c) 微调基准中加入 480P 域抽帧标注数据；d) 尽可能获取 1080P 源（登录 cookies 或其他高清来源）。

### 8.4 E 盘资料库盘点结论

| 位置 | 内容 | 处置 |
|---|---|---|
| `E:\frisbee-pool`（5150 图+标注+manifest） | 当前训练池镜像（ultimateml 1001/kaggle 851/coco 2268/coconeg 445/hardneg 202/negatives 80/pseudo 103/game1080 200，19% 背景） | 已在用，作为权威副本 |
| `E:\firsbee\03_datasets\UltimateML`、`kaggle_frisbee` | 已入池的原始来源 | 保留 |
| `E:\firsbee\03_datasets\openimages_frisbee`（1.3GB parquet+全量标签 CSV+frisbee id 清单） | 只挖出过 5 张；parquet 里还有可挖掘的 Flying-disc 图 | **待挖掘**：扩充训练域多样性 |
| `E:\firsbee\03_datasets\ultimate_analytics` | **最有价值先例**：YOLOv8+KMeans 分队（7v7 每队上限先验）+鸟瞰战术板；Supervisely 精标数据集未随库发布（仅 4 样帧 person 框） | 借鉴 7 人上限先验与工程结构 |
| `E:\firsbee\03_datasets\frisbee-vision-project` | COSC428 课程项目：CSRT/KCF 传统跟踪+近景投掷素材+报告 | 近景测试素材与传统方法参考 |
| `E:\firsbee\03_datasets\SoccerSynth-Field` 等 | 场线合成图（4 张）/零散 | 边际参考 |

### 8.5 环境事实

- GPU 推理/训练用 **WSL Ubuntu**（torch 2.11.0+cu128）；Windows 侧 anaconda 是 CPU torch。
- B 站下载：BBDown 匿名 480P 上限；yt-dlp 被 412 拦截；1080P 需登录 cookies。

---

## 9. 附录：2026-09-08 凌晨自主执行记录（03:24–08:50 窗口）

### 9.1 高清源打通

- 从发布工具（WSL `~/social-auto-upload`，sau CLI）的 `cookies/bilibili_diyi.json` 提取 B 站登录 cookie，BBDown `-c` 传参解锁 **1080P**（943MB AVC，1920×1080@30fps）。工具脚本：`data/bili_final_test/make_cookie_str.py`。

### 9.2 1080P 飞盘检测复测（关键发现）

| 指标 | 480P | 1080P |
|---|---|---|
| 帧检出率（60-120s，900 帧） | 16.7% | **53.1%** |
| 置信度中位数 | ~0.42 | **0.62** |
| 高置信度 top10 放大目检 | 全部草纹误检 | **仍是同一批草纹误检（0.82-0.91），位置与 480P 一致** |

- 结论：分辨率提升大幅改善召回，但**草纹误检是域缺口问题而非分辨率问题**——硬负样本路线被证实为正确方向。f382 帧检出真飞盘（鱼跃接盘），证明模型有判真能力。

### 9.3 硬负样本与数据集 v2

- 全 480P 视频 1fps 扫描 → 703 个检测裁剪候选 → conf≥0.45 过滤剩 559 → **GLM-4V-Flash 逐个确认 494 个为非飞盘**（65 个确实含真盘被排除，VLM 过滤有效）。
- **`frisbee_merged_v2`**：原 merged（4135 train）+ 494 个 bbox 级确认负样本 = 4629 train / 1246 背景（27%）。`configs/frisbee_merged_v2.yaml`。脚本：`data/bili_final_test/build_v2_dataset.py`。
- 数据卫生：这些负样本来自决赛视频（bbox 裁剪级，符合 AGENTS 规则）；**该视频今后作为域内开发/验证视频，最终评测需另用未见过的视频**。

### 9.4 球员预标注包（人工复核就绪）

- 1080P 全视频每 2 秒抽帧 **1417 帧** → yolo26x @conf0.30 预标 **34072 个 person 框**（`data/bili_final_test/player_labels/`）。
- yololabeler 复核包已导出：`yololabeler data/bili_final_test/player_frames` 即可开始人工修正（1417 图 / 37758 预测框）。目检质量良好（含前景观众全覆盖）。

### 9.5 架构基准训练（进行中）

- 对照：**yolov8s-p2 vs yolo26s-p2**（ultralytics 8.4.46 无 11 系 P2 yaml；两代 P2 头、同级参数量，变量干净）。
- 同超参：merged_v2 数据、box=5、epochs=30、patience=8、close_mosaic=5、imgsz=1280、batch=2、workers=2、seed=42。
- tmux 会话 `bench`，脚本 `data/bili_final_test/run_bench.sh`，日志 `train_bili_bench_*.log`；epoch 实测 ~2.5-3 分钟（含 val）。
- 训后评测（`eval_bench.sh`）：固定标尺 = 原 `frisbee_merged` test 集 mAP；域检查 = 同条件 1080P 片段 SAHI 复测草纹误检是否消退。

### 9.6 架构基准与硬负样本效果（07:25 更新）

**同标尺评测**（frisbee_merged test 集，476 图，imgsz1280）：

| 模型 | 数据 | epochs | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|
| p2_shadow_v1（现役基线） | merged | 100 | **0.855** | 0.481 | 0.923 | 0.763 |
| bili_bench_v8sp2 | merged_v2(+494负) | 30 | 0.701 | 0.361 | 0.775 | 0.643 |
| bili_bench_26sp2 | merged_v2(+494负) | 30 | 0.581 | 0.307 | 0.687 | 0.537 |

结论：
1. **架构对比（同数据同 30 epochs）：v8s-p2 完胜 yolo26s-p2**（+12 mAP50）——"更新一代 YOLO"在此 P2 小目标任务上不占优，现役架构选型保持 v8s-p2。YAML 可用性：ultralytics 8.4.46 无 11 系 P2，v8/26 有。
2. bench 绝对值低于基线主因是**训练时长（30 vs 100 epochs 欠训练）**——对照实验已证实（08:34）：v8s-p2 @ 原始 merged、同 30 epochs → val mAP50=**0.687**（P 0.739 / R 0.649），与加负样本的 0.699（P 0.805 / R 0.600）相当且精度更高。**负样本不伤标准 mAP 还提升精度，策略完全验证**；下一步生产级训练 = merged_v2 + 100 epochs（约 4h），预期在保持域内 FP 压制的同时追平基线 0.855。
3. **硬负样本对域内误检的效果（已验证）**：同条件 1080P 片段复测，重训后检测数 527→236（−55%）、置信度峰值 0.91→0.75、≥0.70 检测 47→3——高置信度草纹误检被显著压制，但未根除（f380 帧真盘正常检出）。需多轮负样本迭代（更多帧、更多场地/光照、1080P 源负样本）。
4. 工程注意：ultralytics 全局 settings 的 runs_dir 指向 `~/comfy/ComfyUI/runs`，训练实际输出在 `ComfyUI/runs/detect/runs/<name>`，训后需复制回项目 `runs/detect/<name>`；后台任务必须用 **tmux** 保活（nohup 会随 WSL 会话退出被杀）。

### 9.7 待办（08:50 停止点后由用户安排）

1. 训练与评测结果判读（bench_status.log / eval_status.log）
2. 球员预标注人工复核（yololabeler 包）→ 训练 player/referee 检测器
3. 换更清晰源或更多场次视频扩硬负样本；单应性标定（按镜头段）
4. 交换手/比分几何事件引擎开发（方案 §4 第 4 行）

---

## 10. 计划审核（17:30–19:00 第二轮）：去人工化修订与并行开发体系

背景：用户对原计划提出两个关键挑战——"人工成分不能再减少吗？有再互联网中穷举式寻找方案了吗？"与"任务分 worktree 并行、GPU 排队"。据此做了三路专项调研并重构执行体系。

### 10.1 假设验证总表（实测 vs 原假设，含反转）

| # | 原假设 | 实测结果 | 结论 |
|---|---|---|---|
| H1 | roboflow DFL 足球权重可零样本用于球员检测 | 漏检近半（15/25），前景大目标也漏 | **反转**：改 COCO person 预训练（yolo26x 覆盖 28-32/帧）|
| H2 | 误检主因是 480P 分辨率 | 1080P 检出率 16.7%→53.1% 但高置信误检仍是同一批草纹 | **部分反转**：召回受分辨率影响，误检是域缺口 |
| H3 | 更新一代 YOLO（yolo26）更好 | 同数据同 30ep：v8s-p2 0.701 vs 26s-p2 0.581 | **反转**：现役 v8s-p2 架构保持 |
| H4 | 事件统计走"几何主判定+人工复核队列" | 维持（三路调研证据支持：MMVP、Tryolabs、羽毛球轨迹断点共识）| 维持 |
| H5 | TrackNetV3 第二路提升遮挡召回 | 未验证 | 维持候选（决策门保留）|
| H6 | 硬负样本会轻微伤标准 mAP | 对照实验：0.699 vs 0.687，P 反升（0.805 vs 0.739）| **反转（正面）**：无代价纯收益 |
| H7 | 球员标注需人工 2-5 人日 | E4' 全自动管线 0 人工完成 34072 框 | **反转**：见 10.2 |

### 10.2 三路"去人工化"调研结论与落地实测

**E4' 球员角色全自动标注**（替代 2-5 人日人工复核）：person 域是开放词汇/自动标注官方验证"接近人工"的域；队伍分类走无监督通道（调研：SAM2 抠 torso+KMeans / SigLIP 投票 / GLM-4V 仲裁）。落地实测（零人工）：yolo26x 预标 34072 框 → HSV 红蓝/黑黄条纹规则 + GLM-4V 仲裁 400 次 → 红 10899 / 蓝 10613 / 裁判候选 3521（初版误判 7888，收紧后修复）/ ignore 9039；抽检图确认场上球员全对，残余误差集中在场边观众（标定后场地多边形可滤）。产出：`tools/auto_label_roles.py` + role_labels/ + 抽检图。

**E6 自动事件 GT**（替代 30-60 分钟人工看场）：调研关键证据 IJCNLP 2025（arXiv 2506.17144）纯解说文本 mAP 64.5 追平视频 SOTA。落地实测：AI 字幕（659 条）→ LLM 抽取 52 事件；记分牌 OCR（帧差触发+GLM-4V）追出 15 次比分跳变（0:0→5:7 与真实一致）；FunASR 转写对照路；三路融合钳位去重后 **14 得分 + 1 攻守转换**。残余人工：GUI 阶段抽检 10-15 分钟。

**E5-v0 场地自动标定**（免训练档）：诚实负结果——业余素材上白线磨损/白衣干扰/杂物多，Hough+矩形模板匹配 10 帧仅 1 帧"解算"且为假阳性（画面根本无场线）。结论：免训练经典路线不可行；可行路线 = ①合成数据训练档（调研有 SCCvSD/SoccerSynth 先例，飞盘模板仅 6 线+2 点，本地 2D 渲染器一天可产 10 万帧零人工标注）②人工 4-8 点 fallback（每机位段 <1min，`tools/calibrate_field.py` 已有）。

### 10.3 人工必要性验证表（修订版）

| 人工点 | 原估计 | 现估计 | 依据 |
|---|---|---|---|
| 球员标注 | 2-5 人日 | **0（E4' 自动）**，抽检 ~15min | E4' 实测 |
| 场地标定 | 每段手动点击 | E5-v0 证伪免训练路线 → **合成数据训练档（零人工标注）为主，人工 4-8 点为 fallback** | E5-v0 负结果 |
| 事件 GT | 30-60 min | **10-15 min 抽检** | E6 实测（三路融合）|
| 运行时确认 | 每次得分确认 | **可选修正接口**（三路旁证高置信自动记账）| 用户需求 + E6 |
| GUI 验收 | 30 min | 30 min（不可消除：用户验收自己产品）| — |
| cookie 续期/push 凭据 | — | 一次性 | — |

### 10.4 并行开发体系（用户要求的 worktree + GPU 排队）

- **三个任务 worktree**（代码隔离，数据经绝对路径共享主树）：
  - `.worktrees/auto-label` → `feat/auto-label-roles`（E4'）
  - `.worktrees/auto-gt` → `feat/auto-gt-events`（E6）
  - `.worktrees/auto-calib` → `feat/auto-calibration`（E5-v0）
- **GPU 排队**：`tools/gpu_run.sh`（flock 锁 /tmp/frisbee_gpu.lock，跨 worktree 串行）+ tmux 长驻。任务顺序示例：OCR → ASR → 生产级重训。
- **集成分支** `feat/match-analysis`：三路已合并 + GUI v0 已并回；基准补测队列（11s/26s/v8s）在 makeup 会话排队，完成后自动恢复生产级重训（断点 last.pt.deferred 已保留）。
- 训练输出路径陷阱（runs_dir 指向 ~/comfy/ComfyUI/runs）：训后需从 `ComfyUI/runs/detect/runs/<name>` 复制回项目。

### 10.5 更新后路线图

- **Phase 1（本轮已完成大半）**：E4'/E6/E5-v0 三产物 + 生产级重训（跑完看数）+ 事件引擎骨架（106 测试全绿）
- **Phase 2**：GUI v0（PySide6，用户已开独立会话在 .worktrees 相应目录开发）→ 集成 E4' 标签与 E6 GT → 交换手/比分统计首秀
- **Phase 3**：标定合成数据训练档（零人工）→ 事件引擎接真实管线用 E6 GT 验证精度 → autodistill 式自蒸馏球员检测器
- **Phase 4**：TrackNetV3 决策门、VLM 兜底、光流传播标定、ReID

### 10.6 补充评估（09-09 凌晨，用户提交的三仓库与四视频）

**三仓库**（子代理逐仓核实 README/目录/源码）：

| 仓库 | 等级 | 结论 |
|---|---|---|
| ultralytics/skills | **实质可用** | 官方 AI 代理技能包（AGPL，纯 Markdown），已克隆至 `.agent-references/ultralytics-skills/`。要点：①基准协议修正——YOLO26 用 MuSGD 优化器且端到端 val 的 iou 参数无效，报告架构对比时须注明训练器差异；RT-DETR 无 s 尺寸（不影响我们用 RF-DETR）；②四个增量能力入口：知识蒸馏（`distill_model`）、Ray Tune、`yolo26s-reid.pt`（球员跟踪 ReID）、TensorRT 导出——排入 backlog |
| longjingcha/yolov8skill | 部分参考（弱） | 其 pipeline（qwen 预标+难度分流+复核+难例回流）是我们已有工具链的简化版；**无 LICENSE 文件+源码泄露 API key**，代码不可碰。唯一可借鉴概念：几何启发式"风险评分分流"复核排序（可自行实现到 review_labels，半天）|
| positive666/yolo_research | 不相关 | GPL-3.0 的 v5/v7 fork，实停更于 2023-06，全部 trick 绑定过时代码结构，与 ultralytics 包技术栈和同起跑线基准方法论均不兼容 |

**四视频**（子代理下载元信息+AI字幕+抽帧分析，产物在 `data/video_analysis/`）：全部为 YOLO 工具教程，**无飞盘比赛素材**（1080P 场次素材缺口未解决）。唯一有价值项：BV1j6b56VEE4 的 YOLO26-depth 单目米制深度评测——结论"20m 距离误差 4-5m、只能量级正确"，**反向确认精确测速必须走场地几何标定路线**（我们现有方案）；可选低成本实验：yolo26n-depth.pt 在现有 movie/ 片段交叉验证（一个下午，GPU 空闲后）。

**技术选型影响：无改变，有补强**——三仓库均未提供新架构证据，4 架构基准照做；ultralytics/skills 提供基准公平性注意事项（YOLO26 MuSGD 优化器与 val 行为差异须在报告注明）与四项 backlog 能力入口。

---

## 11. 检测器架构基准最终结果与选型建议（09-09）

### 11.1 基准协议

- **同起跑线**：全部官方 COCO 预训练默认权重（v8s/11s/26s 官方发布版；RF-DETR-small 官方 COCO 权重），历史自训版本一律不参与
- **同配方**：frisbee_merged_v2（4629 训练图，含 494 VLM 确认硬负样本）、box=5、30 epochs、patience=8、imgsz=1280（RF-DETR 672——其 32 倍数约束与官方推荐档位）、seed=42
- **同标尺**：frisbee_merged test split（476 图），conf=0.001 标准口径；FPS 为 1080P 直推
- **注意事项**（ultralytics/skills 提示）：YOLO26 训练器为 MuSGD 且端到端 val 行为不同，其对比数字含训练器差异

### 11.2 四维对比表（test 集）

| 模型 | License | mAP50 | mAP50-95 | P | R | FPS@1080P | 参数量 |
|---|---|---|---|---|---|---|---|
| **RF-DETR-small** | **Apache-2.0（纯开源）** | **0.7432** | **0.5714** | — | — | **39.8**（@672） | 31.8M |
| YOLOv8s（标准头） | AGPL-3.0 | 0.6386 | 0.3378 | 0.732 | 0.584 | 13.6 | 11.1M |
| YOLO26s（标准头） | AGPL-3.0 | 0.5697 | 0.2955 | 0.699 | 0.534 | 14.1 | 9.5M |
| YOLO11s（标准头） | AGPL-3.0 | 重训中（此前 val 0.773，权重遗失后补测） | — | — | — | — | — |
| v8s-**P2**（参考线，昨日） | AGPL-3.0 | 0.699(val)/0.701(test) | 0.361 | 0.775 | 0.643 | ~13 | 10.6M |
| **prod v8s-P2**（merged_v2+100ep） | AGPL-3.0 | **0.8092(test)** | 0.4332 | 0.875 | 0.716 | — | 10.6M |

### 11.3 结论

1. **RF-DETR-small 意外夺魁且优势显著**：mAP50 0.743 领先第二名（v8s-P2 0.701）4 分、领先 v8s 标准头 10 分；**mAP50-95 0.5714 断层领先**（第二名 0.36-0.44）——框回归质量高一个档次；且 39.8 FPS 的推理速度反而是最快的（672 分辨率下）。**纯开源 Apache-2.0，无 AGPL 传染**。
2. **P2 头的价值再次被确认**：同架构 v8s，P2 头（0.699）比标准头（0.639）高 6 分——小目标头对我们的 10-25px 目标是实打实的增益。
3. **架构代际排序（标准头）**：v8s(0.639) > 26s(0.570)，与 P2 对照实验结论一致——更新≠更好。
4. **生产级重训成功**：v8s-P2 + merged_v2 + 100ep 达到 **val mAP50 0.811 / P 0.845**（对比旧基线 shadow_v1 的 0.855 略低但训练数据含 494 硬负样本，域内误检压制能力是 shadow_v1 没有的）。
5. **YOLO11s 事故记录**：权重在训练成功后遗失（疑似被后续清理逻辑误删，日志有保存声明但文件消失），已用相同配方+cache 提速补测中。

### 11.4 选型建议

> **09-11 更新**：本节成文时 D-FINE 尚未入列基准。此后 D-FINE-S 以 test 0.8655/0.6470 断层夺魁（RF-DETR 退居第二），随后 100 帧域外金标准验证又查明"自动判据 GT 失效、全部 mAP 作废"——最新结论与教训见 §13.1/§13.2，本节 B 轨的"RF-DETR 转正"表述作废。

- **飞盘检测器主架构（推荐）**：**双轨并行**
  - **A 轨（当前生产）**：v8s-P2 prod 权重（0.811 val，已含误检压制数据）——立即可用、与现有 SAHI/追踪管线零适配
  - **B 轨（升级候选）**：**RF-DETR-small 微调版**——把它的 0.743@30ep 视为未充分调优的下限（30ep 对 DETR 系偏少，DETR 通常 50+ep 收敛），值得加训 20-30ep 观察；其 mAP50-95 优势意味着定位更准（对测速的 bbox 中心精度直接有利）；Apache-2.0 消除 AGPL 商用障碍
- **训练方案**：数据侧继续（硬负样本迭代已证明正收益）；若 RF-DETR 加训后 mAP50 过 0.78，B 轨转正，A 轨退为对照
- **纯开源备选清单**（后续可测）：D-FINE（ICLR 2025, Apache-2.0）、RT-DETR-Paddle（Apache-2.0，注意无 s 尺寸）
- **AGPL 红线备忘**：当前研究/内部使用无影响；对外分发闭源版本前必须完成 RF-DETR 转正或购买 Ultralytics 授权

### 11.4b 场线关键点模型（E5 phase-3 合成数据档）——验证成功（09-09 23:40）

- 训练数据：合成场渲染器生成 **5,000 训练 + 500 验证帧**（零人工标注，含遮挡/模糊/噪声增广）
- 模型：yolov8s-pose，8 关键点回归，60 epochs，~1.7min/epoch
- **验证集 pose mAP50-95 = 0.472 / mAP50 = 0.624**（合成 held-out）
- 意义：E5-v0 证伪的"免训练 Hough 路线"的替代路线（合成数据训练档）**首次闭环成功**——
  纯合成 → 场地关键点检测能力成立。下一步：真实帧泛化目检 + 与自动标定管线（auto_calibrate.py）
  的集成（用关键点替换 Hough 线匹配）。

### 11.5 事故与修复记录（队列可靠性）

连环崩溃最终根因链：mlflow 新版 file-store 门禁杀训练启动回调（修复：`MLFLOW_ALLOW_FILE_STORE=true`）→ WSL2 GPU-PV 多进程 CUDA 并发不稳定（修复：严格串行+GPU 空闲守卫+全局 flock）→ watcher 日志匹配误触发（教训：完成标记用文件不用 grep）→ GUI 会话未走锁占用 GPU（修复：协调文件+其 worker 已自动套锁）。prod 恢复的 yolo CLI GreenSocket 崩溃（修复：python API resume）。11s 权重遗失（修复：同配方补测中）。**系统性教训：多会话共享 GPU 必须全部走 gpu_run.sh 锁；完成信号用标记文件。**


---

## 12. 多会话分工盘点（09-09 晚）：GUI 会话 7 轮迭代成果与主会话去重

GUI 会话（.worktrees/gui-v0）在 09-09 自治窗口完成 **7 轮带验收门的迭代**（报告：docs/superpowers/plans/2026-09-09-iteration-report.md）。本节为两会话产出的去重结论。

### 12.1 GUI 会话已定案的（主会话直接用，不重复造）

| 能力 | 定案 | 关键数字 |
|---|---|---|
| 球员分队生产路径 | 零样本 person+BoT-SORT+HSV 三级门聚类（跨素材验证+诚实标色 dark+稳定性自洽门） | 一致性 97.77%/97.35%，跨素材红蓝与"一彩一暗"双向验证 |
| 检测器补召回 | 2×2 切片推理+跨片 NMS+补框近邻继承队伍（GUI 默认开） | 检出 +13%、队伍覆盖 68%→96%、轨迹一致性 100% |
| 零人工检测器上限 | 3 轮迭代负结果定案：E4' 标签训检测器收敛 ~0.78 档一致性，低于聚类 0.978——**检测器补召回、聚类保一致性，互补而非替代** | v1 0.739→v2 0.778→v3 0.783 |
| BoT-SORT 阈值 bug | track_high_thresh=0.25 会吞掉 --conf——按 conf 动态覆写 tracker 配置 | --conf 此前完全失效已修 |
| 渲染 | 边线 5+得分线 2+得分区 2 随标定 JSON 自动渲染；暗色毛玻璃主题 | — |

### 12.2 主会话独有产出（GUI 直接消费）

| 能力 | 状态 | GUI 消费方式 |
|---|---|---|
| 双模型飞盘联合跟踪（iter_player_and_disc_tracks） | ✅ 已合并进集成分支（冲突已解：与切片并存） | worker --disc-weights 直接用 |
| events_runner 事件统计适配层（标定驱动+场外过滤） | ✅ | tracks.json+标定 json → 事件表 |
| E6 自动 GT（三路融合 15 事件）+ 对齐评测器 | ✅ | 飞盘轨迹就绪后跑 P/R/F1 |
| 合成场渲染器+关键点训练脚手架 | ✅ pitch_kp 训练中 | 训成即解锁自动场线标定 |
| 4 架构基准（RF-DETR 夺魁） | ✅ | 飞盘检测器选型依据 |

### 12.3 踩坑同步（两会话共同教训）

- **WSL2 GPU-PV 多进程并发不稳定**（主会话：双/三车道 4 次崩溃；GUI：切片与 track 共用 YOLO 实例死锁 38 分钟）——**单进程串行+锁**是唯一可靠形态，多实例必须用独立 YOLO 对象
- **Windows 原生 --cache 死锁**（4629 张 1280px 缓存进 RAM 卡死）——Windows 侧不用 --cache
- **完成信号用标记文件**，不用 grep 日志（历史残留会误触发）
- **验证集与训练标签同源时 mAP 高 ≠ 真值准确**（GUI 迭代 2 教训：0.937 mAP 但行为一致性仅 0.739）——行为级独立通道抽检必须做

### 12.4 待用户决策（GUI 会话遗留）

零人工检测器要突破 0.78 上限需引入"人工抽检校准的金标准小集"（约 30 帧人工核对）作第三通道仲裁——超出零人工边界，交由用户决策。

---

## 13. 附录：2026-09-10–09-11 收尾盘点：D-FINE 夺魁、金标准方法论修正与生产配置定案

> 覆盖 §12 之后的提交（4c3a5cd → 557b3dc）。方法论结论此后以 `wiki/`（llm-wiki 规则）为权威沉淀处，本节为与主计划的对接摘要。

### 13.1 检测器架构基准 v2 终局：D-FINE-S 断层夺魁（09-10）

11s 补测完成（0.6122，Windows native 30ep，OOM 重试后），`results/bench_summary.json` 四模型无缺；prod v8s-P2 权重归位。D-FINE-S（官方配方 72ep@640，Apache-2.0）加入后格局改写：

| 模型 | License | Epochs@分辨率 | mAP50 | mAP50-95 | 备注 |
|---|---|---|---|---|---|
| **D-FINE-S** | **Apache-2.0** | 72@640 | **0.8655** | **0.6470** | 断层第一 |
| RF-DETR-small | Apache-2.0 | 30@672 | 0.7432 | 0.5714 | 退居第二 |
| prod v8s-P2 | AGPL | 100@1280 | 0.8092 | 0.4332 | 已含硬负样本 |
| shadow_v1 基线 | AGPL | 100@1280 | 0.8547* | 0.4810 | *旧切分，仅量级参考 |
| v8s / 11s / 26s 标准头 | AGPL | 30@1280 | 0.639 / 0.612 / 0.570 | — | 更新≠更好 |

- **速度代价**：D-FINE-S @640 单帧 52.8 FPS vs YOLOv8s @640 167.9 FPS（YOLO 快 3.2 倍）——我们是离线分析，可接受。
- **纯开源双保险**：D-FINE/RF-DETR 均 Apache-2.0 且性能覆盖 AGPL 系，商业化授权障碍实际消除。
- **遗留实验**（wiki 基准页）：D-FINE-S @1280 重训（位置编码绑定，640 权重不能直推 1280；预期小目标召回再涨）、D-FINE + 硬负样本迭代（YOLO 线已证正收益）、TensorRT INT8 导出复测。

### 13.2 金标准 100 帧域外验证：自动判据失败与修正（09-10，重要方法论教训）

- **设计**：1080P 决赛视频抽 100 帧（50 全局均匀 + 50 检测导向；全片 1fps 扫描 2833 帧、1991 帧有检出），9 个模型低阈值推理离线扫阈值（`tools/golden_*.py`，产物 `results/golden_inference/`、`results/golden_metrics.json`）。
- **初报数字已作废**：初看 D-FINE 0.659 领先扩大 / prod 0.638 / RF-DETR 跌至第 8（0.397）/ 26s 垫底（0.302）——但随后人工巡检 210 个高共识框裁剪，**真盘仅占 40-45%**，GT 本身被误检污染，全部 mAP 不可采信。"9 模型全共识=自动采标信号"的初判同样作废。
- **三层自动判据全部失效的原因**：
  1. **多模型共识（≥3 模型 & conf≥0.3）**：8 个 YOLO 系模型共享训练数据（含同一批 494 硬负样本），在相同物体上犯相同错误——同源共识只反映共同偏差；
  2. **场地投影过滤（真盘在场内）**：标定 json 只在标定帧附近有效，机位摇移后其他帧全部失准；
  3. **外观判据（白+圆）**：白帽子同样白色圆形，无法区分。
- **真正的有效产出：域外误检类型清单**（按频次）：白帽子（~40%）、黄色场地锥/标记、记分牌绿色装饰、橙色运动鞋、白色车身/帐篷——模型不是"没学好飞盘"，而是没人教过它们这些不该看的东西。**负样本扩展方向由清单直接给出，预计比换架构更提升真实域表现**。
- **方法论教训（已入 wiki）**：同源模型共识≠真值；test 集排名不可外推（RF-DETR 过拟合信号）；无独立验证通道时宁可给定性发现也不给建立在污染 GT 上的假 mAP。
- **修正后路线**：选型暂沿用 test 结论（D-FINE 领先）；金标准的正确构建方式=人工标注 100 帧（含负帧，`tools/review_desktop.py`/yololabeler 现成）——这是当前唯一剩余的人工决策点；事件统计的瓶颈确认在上游飞盘检测，**先解决检测再谈事件**。

### 13.3 GUI 会话迭代终评：分队/球员管线生产配置定案（09-10，回应 §12.4）

- **默认生产配置 = v3 三类权重 + grid3 切片 + 自动标定多边形**：21.9 框/帧（召回达标），HSV 一致性 0.929 框级 / 0.957 轨迹级。
- **裁判可见配置 = players_e4_v4ref（过采样裁判）**：红蓝合计 15.7/帧 ≈ 真实 14 球员（召回达标）+ 裁判 5.3/帧（过度激发，约 2-3/帧为误报），一致性 0.854/0.864。两条配置 `--weights` 一键切换：召回优先 vs 一致性优先，默认 v3。
- **自动标定 v2（players-extent 钳位优化，`tools/auto_calibrate_v2.py`）**：内点率 0.934，多边形语义=球员可达范围，过滤观众达标；场线渲染仍需 pitch_kp 或手动标定。
- **跨素材验证发现真 bug 并修复（迭代 4-5）**：红衫 vs 深绿/黑衫素材上旧 HSV 门把"亮度差异簇"误判为两队同色——修复为三级门（分离度≥0.15 + 信号≥0.08 + 对比度≥2x）+ 无色信号簇诚实标 `dark`（不硬凑红蓝）+ `stability_check` 自洽门（两次采样一致率<0.85 时拒绝判定）。**"拒绝优于错误"**——139 测试全绿。
- **§12.4 决策落定**：零人工路径（HSV 三级门 + SigLIP 兜底 + 跨素材互验）覆盖红蓝衫与"一彩一暗"两类素材；剩余风险面（双方皆非红蓝且互相相似）由系统拒绝 + GUI 人工改判入口兜底——不再需要前置人工金标准。

### 13.4 事件引擎现状（诚实账）

- **对齐冒烟（`results/engine_vs_gt.json`）**：引擎 15 事件 vs E6 GT 15 事件全部匹配（P/R/F1=1.0 @90s 容差）——但每个匹配 dt 恒为 20.0s（片段时间轴原点偏移），这是时间轴对齐验证，**不是精度证明**。
- **引擎实跑（`results/engine_validation.json`）**：accept_10min（18004 帧/897 轨迹）仅产出 71 个 disc_on_ground 骨架事件，score_state 全 0——得分谓词依赖飞盘轨迹+标定输入，尚未真实运转。
- 双模型飞盘联合跟踪已合并进集成分支（`--disc-weights`），事件引擎接真实轨迹 + 消除时间轴偏移后，才到用 E6 GT 跑真实 P/R/F1 的时点。

### 13.5 其他落地

- worker 新增 `--no-field-filter`（c72c145）：保存未过滤原始检出——自动标定需要全部 person 检出，场内过滤会截断标定输入。
- **分支/worktree 收编（09-11，用户要求）**：全部内容已并入 `main`（`dev` 同步同点）。4 个 worktree（auto-label/auto-gt/auto-calib/gui-v0）及其分支删除——遗留工件存 `results/worktree_artifacts/`，gui_analysis 15 个分析 run 迁至主仓 `runs/gui_analysis/`；4 个 5-6 月旧实验分支（v5/v6-Mahalanobis/siglip/v7-classifier）以 `archive/*` tag 保留后删除（其中 v6 的 6 状态 Kalman 从未合入主线，恢复实验从 tag 取）。远端仅剩 `main`/`dev`。
- 项目 wiki 按 llm-wiki 规则初始化（index/log + comparisons/concepts/entities/queries/synthesis，已沉淀基准对比、金标准修正、硬负样本、P2 头、GPU 可靠性、单目测速等页）。
- Windows native 训练通路跑通（11s 30ep + 3ep 速度探针 `results/speed_probe_win_run/`），作为 WSL GPU-PV 队列之外的备份通道。

### 13.6 更新后的行动优先级（接替 §12 收尾与 §9.7 待办）

1. **检测优先**：按 §13.2 误检清单扩展域外硬负样本（白帽/黄锥/记分牌/橙鞋/白车）+ D-FINE-S 加训与 @1280 重训——上游检测精度是事件统计的前置瓶颈。
2. **人工金标准 100 帧标注**（~1-2 小时量级，唯一剩余人工决策点）：作为检测器选型域外复核 + 后续评测的持久标尺。
3. **pitch_kp 真实帧泛化目检** + 接入 auto_calibrate_v2：打通场线渲染/标定自动化闭环（§11.4b 后续）。
4. **事件引擎接真实飞盘轨迹**，消除时间轴偏移后用 E6 GT 跑真实 P/R/F1（§13.4）。
5. Phase 4 不变：TrackNetV3 决策门、VLM 兜底、光流传播标定、ReID。

### 13.7 v6 分支遗产合入与马氏门控复测（09-11，用户要求"进主线"）背景：分支清理时确认 `archive/improve-precision-v6-mahalanobis`（2026-06-08）的"6 态 Kalman + 马氏门控"从未合入主线。调其 4 份实验文档复盘：**该实验当时没有做完**——v1/v2 两版门控因伤基线被回退，v3 改为"事后验证"设计后只做了 d² 数据采集（门控阈值停在 500=不拦截），端到端收益从未测量。

**合入内容**（commit 1016d0c，`feat/mahalanobis-gating` → main）：`init_kalman(nstate=4|6)` 参数化（**默认仍 4 态基线**，6 态走 `--kalman-6state`）；`mahalanobis_gate()`（H 随 state 维度自适应）+ v3 事后门控（`--mahalanobis-gate --gate-threshold`，默认 chi²₀.₉₉₉=13.8155）；CSV 恒增 `mahalanobis_d2` 列；`tools/analyze_d2.py`；4 份实验文档归档入库；+6 测试（全套 159 绿）。

**GPU 复测**（videoplayback_first60s，3598 帧，v3@conf0.35，pixel-only）：

| 配置 | 行数 | d² 中位 | d² p95 | >13.8 占比 |
|---|---|---|---|---|
| A：4 态基线 | 1068 | 545 | 6.9×10⁵ | 91.0% |
| B：6 态 | 1071 | **143（−74%）** | 3.2×10⁵（−53%） | 78.5% |

- **"6 态使 d² 收窄 60-80%"的旧结论复现成立**（p50 −74% / p90 −56% / p95 −53% / p99 −69%）——6 态对飞盘运动的预测拟合确实更好。
- **但门控在当前噪声参数下不可用**：`measurementNoiseCov=0.1`（σ≈0.3px）对检测器真实噪声严重过信，d² 绝对值膨胀 2 个数量级，任何 chi² 量级阈值都会拒掉 78-91% 的**正确**检测——这正是 6 月 v1/v2 门控"伤基线"的根因，也是实验停摆的根因。按 B 的 p99.5（2.06×10⁶）跑门控（C 配置）：仅 4/1071（0.37%）候选被拒，召回无损但增益可忽略。
- **结论与下一步**：6 态/门控代码已主线可用（开关默认关，行为零变化）；门控要真正起效需先做 **v4 噪声校准实验**——把 `measurementNoiseCov` 标定到检测器真实定位噪声（σ≈3-5px 量级）使 d² 回到 chi² 分布，再重选阈值。在此之前，FP 治理仍以 §13.6 的负样本扩展为主线。

### 13.8 金标准 v2：零人工双通道重评与选型反转（09-11）

§13.2 宣布金标准 v1 的 mAP 作废后，本轮以**零人工**方式重建 GT 并重评 9 模型。方法：**ZCode 内生判读 × 模型共识 双通道**（用户授权窗口内，由会话本体 glm-5.3-flash 直接读编号拼图判读，不写外部 API 脚本、零成本；工具 `tools/golden_sheet_build.py`，747 候选 → 21 张 36 格拼图逐格判读）。

- **判读统计**：747 候选 = 38 yes（5.1%）/ 12 abstain（1.6%）/ 697 no（93.3%）——与 §13.2 误检清单吻合（白帽/纸巾/记分牌/黄锥/白车占绝对多数）。
- **GT 合成规则**：yes+共识≥2 → 正框；yes+单模型 → ignore；abstain → ignore；no+强共识(≥3) → ignore（通道冲突）；no+弱共识 → FP 区。`tools/golden_metrics.py --gt-v2` 新增 ignore 机制，**旧 GT 路径回归通过**（9 模型数字与 v1 完全一致）。
- **GT v2 规模**：30 正框 / 29 帧（共识度 9 模型×8、7×5、5×4…；尺寸 18-79px），ignore 197 框，FP 区 520 框。弃权率 1.6% ≪ 30% 门槛 → 榜单有效。

**域外金标准榜单（v2）**：

| 排名 | 模型 | mAP50 | mAP50-95 | bestF1 工作点 |
|---|---|---|---|---|
| 1 | **shadow_v1（旧基线，未加硬负样本）** | **0.900** | 0.573 | **F1 0.893 @conf0.15（tp25/fp1/fn5，P 0.962/R 0.833）** |
| 2 | prod v8s-P2（现役 A 轨） | 0.723 | 0.483 | 0.737 @0.3（tp21/fp6/fn9） |
| 3 | D-FINE-S | 0.693 | **0.553** | 0.760 @0.7（tp19/fp1/fn11） |
| 4 | RF-DETR-small | 0.656 | 0.417 | 0.704 @0.5 |
| 5-9 | prod@640 / v8s / 26s / ctrl / 11s | 0.52-0.30 | — | — |

**重大发现：选型反转**。test 集第 1 的 D-FINE 与现役 prod 都落后于旧基线 shadow_v1——它未加硬负样本、检出极保守，在 100 帧真实转播帧上 bestF1 点仅 1 个 FP。这是 §13.2"test 集排名不可外推"的第二个实证（第一个是 RF-DETR 在 v1 假 GT 上跌到第 8，本次 v2 上回升到第 4）。D-FINE 需要把 conf 提到 0.7 才达到同档精度，代价是召回掉到 0.633。

**诚实边界**：GT v2 是保守双通道集（30 框，双通道都认可才入选，偏大而明显的盘）——度量的是"**高置信可靠度**"而非完整召回能力；高召回模型的合法检出若落在 GT 外会被计 FP，系统性不利于低阈值火力全开的模型（如 D-FINE@0.05 有 4173 检出）。

**选型决策（接替 §13.1/§13.4）**：
1. **事件统计高置信通道换血**：比分/交换手主判定的检测输入优先 shadow_v1@conf0.15（P 0.962）——现役 prod@0.35（P 0.778@best 点）并非域外最优；D-FINE@0.7 可作等效备选。
2. **补召回通道不变**：低置信补召回仍可用 D-FINE/prod，但其额外检出必须过独立校验（不得直进判定链）。
3. **"硬负样本+保守头"组合值得验证**：shadow_v1 旧数据+保守行为 vs prod 新数据+硬负样本——把 §13.6 负样本扩展后的重训与 shadow_v1 风格对照，确认能否两全。
4. **GT 扩充候选**：8 个 yes-单模型框与 12 个弃权框值得二次复核升级为 GT/FP 区；D-FINE@1280 与负样本扩展重训的模型入评时复用本管线（`golden_sheet_build.py` + `--gt-v2`）。

### 13.9 F1 多维融合落地 + YOLO26/D-FINE 官方用法排查（09-11 晚）

**背景**：用户判断"单维度识别上限不高→多维才是最快路径"（§13.8 金标准反转+负样本伤校准均支持），批准 F1 融合主线（零训练零识图）+ obj365 底座测试（后置）。

**两模型官方用法排查**（用户质疑"26 新却差 12 分是否用错了"，两案并查）：
- **YOLO26**：官方文档确认训练接口与 v8 完全兼容、无专用配置警告——用法无错。但 **26s-P2 输 12 分含隐性不公平**：yolo26-p2.yaml 只有架构无 COCO 预训练权重（v8s-P2 有成熟权重生态）。官方卖点在推理侧（CPU +43%、NMS-free），精度仅对标 YOLO11 +1~2 分；社区（GitHub issues 全量）无训练劣化报告、零体育场景经验。公平复测需 26-p2+官方 recipe（box=9.83 等），6h 级，挂起待批。
- **D-FINE**：官方微调路线 = obj365 底座+类别映射+大 epoch——我们用的 COCO 底座非其上限（官方数据 obj365 底座 +2.2 AP）。**许可警示**：obj365 系 checkpoint 商用受 Objects365 条款约束（README 明示）；COCO 权重干净（现用）。已知坑 #247 空标注死锁、#82 分辨率绑定。修正表述："D-FINE 断层夺魁"→"非官方微调路线下已领先，官方路线可能更高但有许可代价"。
- 两案共同指向：**检测器层继续卷收益递减，多维融合性价比最高**（用户判断成立）。

**F1 融合落地**（`frisbee_analyzer/disc_fusion.py` + `--disc-fusion` 开关默认关，commit 0722965）：
- Gate-A 8/8：合成轨迹关联 100%、野值门控恢复、速度边界、遮挡重关联、零 cv2/torch 纯度（AST 级测试钉死）。**修出真 bug**：is_tracking 原定义使门控降级 1 帧后轨迹永久哑火。
- Gate-B 4/4 + T7 真实冒烟：60 帧 41 检出→gated 15/speed 9→32 帧带状态（tracking/predicting/gated/rejected）。
- graphify 知识图谱建图（1288 节点/2265 边，`graphify-out/` + `.mcp.json` stdio server，重启会话后 MCP 生效）。

**Phase C 3 格矩阵**（300 帧，55-56min 片段；格④⑤ D-FINE/负样本模型待 GPU 窗口）：

| 格 | 盘检出帧 | 最长连续轨迹 | 场外率 |
|---|---|---|---|
| ①prod 基线 | 0 | — | — |
| ②shadow_v1@0.15 裸 | 198（全 raw 无状态） | 0（无轨迹概念） | 0.884 |
| ③+融合 | 104（tracking49/predicting55） | **10f=0.4s** | 0.99* |

- **抽检（G3）**：tracking 帧 8 格目检——f0-f5 **6/6 全是真盘**（清晰椭圆草地移动）；f26/f27 白衣误检 2 帧混入（低阈值补召回预期内，status 字段可分层）。
- ***场外率不可用作本素材判据**：抽检证明 tracking 帧是真盘但投影全"场外"→ **55-56min 段标定失效**（标定帧在原片 79min 处，机位摇移；§13.2 已知限制再实证）。G1 改用轨迹质量标尺。
- **Gate-C 判定**：G2 通过（0→10f 增量真实）、G3 通过（6/6≥80%）、G1 标尺失效改判、G4 事件对照待真盘轨迹更长的素材、G5 检测器裁决待格④⑤。**融合层机制验证通过；完整 Gate-C 需 ①段内标定（auto_calibrate_v2）或 ②格④⑤补齐**。

**段内标定实验（09-11 深夜，矛盾解开）**：auto_calibrate_v2 段内重标（内点率 0.975 过门）后球员脚点场内率 4.1%→94.9%——旧标定确认失效、新标定健康。**但盘检出场外率仍 0.99**，画框目检（results/f1_matrix/frame1_annotated.jpg）解开全部矛盾：① 黄框处是真盘在边线附近飞行（检测无误）；② 该"55-56min"片段实为 **日中国际赛转播（JPN 9:4 CHN）**，非城市俱乐部决赛——素材身份认知错误；③ 盘检出 wy≈40.5（边线外 3-4m）是**盘沿边线飞行的合法位置**+自动标定语义=球员可达范围（near 边收紧）——**"场内约束"对盘这个会出界/高飞的对象先天不适用**。G1 判据最终定版：场外率弃用，以"轨迹连续性+抽检真盘率"为准（G2/G3 已通过）。

**obj365 零样本测试（09-12，用户批准"拿来测一下，效果不好就删"）**：零样本金标准 mAP50 仅 **0.093**（vs COCO 底座微调后 0.693）——Objects365 的 Frisbee 类（玩具盘/特写为主）与真实转播远景小盘域差距巨大。**结论：微调必要性被正名（+60 分来自域内微调）；obj365 路线取消，权重按许可纪律删除，记录结论**。G5 裁决更新：格④⑤（D-FINE+融合 longest 19f 最长；负样本模型 9f）已完成——**格④ D-FINE+融合为高召回通道最优组合**，检测器矩阵 5 格闭环。

**下一步**：① 段内标定已入 pipeline（--auto-calibrate），格①-⑤矩阵可随跑随评；② G4 终判待含持盘回合素材（60-120s）；③ TrackNetV3 第二路（远景小盘召回，Phase 4 决策门）。

### 13.9 F1 多维融合落地 + YOLO26/D-FINE 官方用法排查（09-11 晚）

**背景**：用户判断"单维度识别上限不高→多维才是最快路径"（§13.8 的金标准反转+负样本伤校准均支持此判断），批准 F1 融合主线（零训练零识图）+ obj365 底座测试（后置）。

**两模型官方用法排查**（用户质疑"26 新却差 12 分是否用错了"，两案并查）：
- **YOLO26**：官方文档确认训练接口与 v8 完全兼容、无专用配置警告——用法无错。但 **26s-P2 输 12 分含隐性不公平**：yolo26-p2.yaml 只有架构无 COCO 预训练权重（v8s-P2 有成熟权重生态）。官方卖点在推理侧（CPU +43%、NMS-free），精度仅对标 YOLO11 +1~2 分；社区（GitHub issues 全量）无训练劣化报告、零体育场景经验。公平复测需 26-p2+官方 recipe（box=9.83 等），6h 级，挂起待批。
- **D-FINE**：官方微调路线 = obj365 底座+类别映射+大 epoch——我们用的 COCO 底座非其上限（官方数据 obj365 底座 +2.2 AP）。**许可警示**：obj365 系 checkpoint 商用受 Objects365 条款约束（README 明示）；COCO 权重干净（现用）。已知坑 #247 空标注死锁、#82 分辨率绑定。修正表述：~~"D-FINE 断层夺魁"~~→"非官方微调路线下已领先，官方路线可能更高但有许可代价"。
- 两案共同指向：**检测器层继续卷收益递减，多维融合性价比最高**（用户判断成立）。

**F1 融合落地**（`frisbee_analyzer/disc_fusion.py` + `--disc-fusion` 开关默认关，commit 0722965）：
- Gate-A 8/8：合成轨迹关联 100%、野值门控恢复、速度边界（scale 设计修正）、遮挡重关联、零 cv2/torch 纯度。**修出真 bug**：is_tracking 原定义使门控降级 1 帧后轨迹永久哑火。
- Gate-B 4/4 + T7 真实冒烟：60 帧 41 检出→gated 15/speed 9→32 帧带状态（tracking/predicting/gated/rejected）。
- graphify 知识图谱建图（1288 节点/2265 边，`graphify-out/` + `.mcp.json` stdio server）。

**Phase C 3 格矩阵**（300 帧，55-56min 片段；格④⑤ D-FINE/负样本模型待 GPU 窗口）：

| 格 | 盘检出帧 | 最长连续轨迹 | 场外率 |
|---|---|---|---|
| ①prod 基线 | 0 | — | — |
| ②shadow_v1@0.15 裸 | 198（全 raw 无状态） | 0（无轨迹概念） | 0.884 |
| ③+融合 | 104（tracking49/predicting55） | **10f=0.4s** | 0.99* |

- **抽检（G3 代运行）**：tracking 帧 8 格目检——f0-f5 **6/6 全是真盘**（清晰椭圆草地移动）；f26/f27 白衣误检 2 帧混入（低阈值补召回预期内，status 可分层）。
- ***场外率不可用作本素材判据**：抽检证明 tracking 帧是真盘但投影全"场外"→ **55-56min 段标定失效**（标定帧在原片 79min，机位摇移；§13.2 已知限制再实证）。G1 改用轨迹质量标尺。
- **Gate-C 判定**：G2 通过（0→10f 增量真实）、G3 通过（6/6 真盘≥80%）、G1 标尺失效改判、G4 事件对照待真盘轨迹更长的素材、G5 检测器裁决待格④⑤。**融合层机制验证通过；完整 Gate-C 需 ①标定有效的素材（auto_calibrate_v2 按段标定）或 ②D-FINE 格补齐**。

**下一步**（优先级序）：① 用 auto_calibrate_v2 给 55-56min 段做段内标定→重跑矩阵使 G1/G4 可判；② 格④⑤ GPU 补跑；③ obj365 测试（已批，Part 2）；④ 负样本 bbox 实验 v3.5（挂起）。

### 13.10 v3.5 bbox 形态实验收官——负样本配方定版（09-12）

**三点对照**（域外金标准 v2，同 119 个靶向目标、同 100ep 预算）：

| 数据 | 负样本形态 | 域内 val mAP50 | 域外 mAP50 | bestF1 | P/R @best |
|---|---|---|---|---|---|
| prod（v2） | 494 张 480P 裁剪（草纹） | 0.809* | 0.723 | 0.737 @0.3 | 0.778/0.700 |
| race_v8sp2_v3 | +114 张**整帧**（白帽靶向） | 0.855 | **0.539（崩）** | 0.591 | 校准塌 |
| **race_v8sp2_v35** | +119 张 **bbox 裁剪**（白帽/布堆靶向） | 0.834 | **0.801** | **0.800 @0.35** | **0.880/0.733** |
| shadow_v1（参照王座） | 旧数据无负样本 | 0.858* | 0.900 | 0.893 @0.15 | 0.962/0.833 |

**结论定版**：
1. **形态是决定性变量**——同 119 个靶向目标，整帧崩（0.539）、裁剪升（0.801 超 prod），AGENTS.md bbox-level 规则实验正名：整帧=场景级抑制有毒，裁剪=外观级判别纯收益；
2. **负样本配方定版**：裁剪形态+靶向清单+判读确认（本 pipeline：`hardneg_sheet_build`→判读→`race_v35` 式重训），可按此循环迭代；
3. 域外王座仍是 shadow_v1（0.900）——保守基线+多样数据仍是泛化之王；v35 证明了"硬负样本收益与域外可靠度可以兼得"（P 0.880 优于 prod 的 0.778，域外仅次于 shadow_v1）；
4. obj365 路线已取消（零样本 0.093，微调必要性被正名）。

### 13.11 G4 终判：端区持盘规则补上漏掉的得分（09-12）

**实验**：testclip_60_120s.mp4（60s 片段，含完整得分回合）全量 1800 帧 pipeline（--disc-fusion --auto-calibrate），事件引擎 + 新增**端区持盘规则**（`events_runner.detect_endzone_carries`：盘在端区(±1m)内慢速(<2.5m/s)持续≥15 帧 → score 候选进复核队列，同端区 30s 内碎片合并）。

**结果**：
- 融合层：1021 帧检出 → tracking 605 / predicting 257 / gated 347，**最长连续轨迹 196f = 6.5s**（此前所有素材最长 0.4s——连续帧素材上融合层价值充分显现）；段内自动标定内点率 0.9348 过门。
- 事件引擎（状态机）：0 持盘/0 得分——**持盘三条件（盘-手部点<1.2m、盘速<6m/s、连续 5 帧）在全片仅 20 帧近距且零散**，量化诊断证实非 bug。
- **端区持盘规则命中**：9 个碎片候选合并为 1 个得分候选（clip t=33.3-53.5s 右端区，253 帧持盘走段），与 GT score@118s（原片时间）差 21s < 90s 容差 → **matched**。
- **窗口公平终评**：片段窗口内 GT 仅 1 个事件（score@118s），引擎 match 级事件 1 个且命中 → **P 1.0 / R 1.0（窗口内）**；GT 全场 15 事件口径下 R=1/15 是分母假象（引擎只看了 60 秒）。

**多维互证的直接体现**：这个得分事件，持盘状态机看不到（持盘时盘贴身遮挡、手部点模型不匹配），端区持盘几何规则补上了——检测×轨迹×标定×几何四维各出一票。**规则发现的"得分后持盘走到底线"行为正是极限飞盘的标准流程**（比 GT 字幕时间戳更接近真实动作时刻）。

**遗留缺口（下一维度的明确靶子）**：持盘中的盘（贴身遮挡）检测器看不见——① 手部 ROI 高分辨率复检（ crop 手部区域二次检测，CPU 可行）；② 持盘行为分类器；③ 检测器加"持盘状态"标注微调。均为后续候选，待用户排优先级。

### 13.12 Stage 1 手部 ROI 复检：负结果定案 + conf0.35 解锁状态机得分（09-12）

**实现了什么**：`frisbee_analyzer/hand_roi.py`（F1 第二检测通道，`--hand-roi` 隐含 `--disc-fusion`，默认关）——融合态非 tracking 帧取离最后观测锚点最近的 K=2 个球员，手部区（躯干横带+双侧扩展 0.6×框宽）crop 放大 imgsz640 复检，检出映射回原帧坐标（`source:"hand_roi"`）并入序列二次融合；健康 tracking 帧不复检。12 个纯几何单测。产物 `tools/summarize_events.py`（事件汇总复用工具）。

**主假设证伪（诚实账）**：持盘中的盘在 crop 放大后依然物理不可见——
- 480p 诊断：持盘窗口 183 复检帧 / ~440 ROI，conf≥0.05 检出仅 17 个，max 0.32 且目检为远处草丛误检；
- 480p 全量：1082 复检帧 / 1948 ROI → 13 检出（5 去重），tracking 579→581，longest 193f 不变；
- 1080p 全量（testclip1080_60_120s.mp4）：1106 复检帧 / 1962 ROI → 89 检出（64 去重），tracking 507→515，longest 140f 不变。
**结论**：贴身遮挡是分辨率无关的物理盲区，图像级复检救不回持盘盘；通道保留为 `--hand-roi` 零回退可选项（对地滚盘/出手瞬间有小样本补充价值），②持盘行为分类器升级为主攻方向。

**意外正发现（配对实验证实）**：状态机得分由 **disc-conf 阈值效应**解锁，与 ROI 通道无关——
| run | conf | ROI | 融合 tracking/pred/gated（longest） | 状态机持盘/得分 |
|---|---|---|---|---|
| g4_repro015（精确复现 G4：605/257/347, 196f, 1021→862 逐项一致） | 0.15 | 关 | 同 G4 | **0 / 0**（复现 G4 账面） |
| g4_base480 | 0.35 | 关 | 579/345/222（193f） | **2 / 1（team=1 右端区）** |
| g4_handroi480 | 0.35 | 开 | 同上+3 ROI 条 | 2 / 1 |
| g4_handroi1080 | 0.35 | 开 | 507/326/467（140f）+8 ROI 条 | 1 / 1 |
因果链完整：剥离 ROI 条目重算事件不变 + 无 ROI 配对基线同样得分 → conf 0.35 高置信检出让融合轨迹在端区持盘站立段足够干净，持盘三条件（盘-手<1.2m、<6m/s、连续 5 帧）首次在真实素材上满足 → **持盘状态机从 0 产出变为真实命中得分**（f836 目检：右端区持盘弯腰建立轴心脚，真盘绿框锁定）。窗口内仍恰 1 GT 得分，状态机+端区规则双维命中同一目标，无假阳性 → **P 1.0 / R 1.0 保持**。
生产含义：**盘通道推荐 conf 0.35（生产默认）而非矩阵实验的 0.15**——低 conf 的 347 次 gated 毛刺让轨迹在持盘段抖动，永不满足连续 5 帧条件；高 conf 以量换质反而解锁状态机。

**审核关**：`g4_handroi480_overlay.mp4` 叠加视频目检 f830-848 通过；`summarize_events.py` 对 g4_final 复现 15 落地+1 端区候选，链路对齐。

**遗留**：① 状态机 f836 同时打出的 turnover(ground_recovery) 标签是同瞬产物（低持盘被误判落地后拾起），候选级无害、记分不重复；② 1080p 融合 gated 467 次（480p 222）——高分辨率像素速度翻倍加剧紧 Kalman 门拒，v4 噪声校准（§13.7）优先级上调；③ 手部 ROI 在新机位/更高清素材上的复测并入 Stage 2。

### 13.13 Stage 2 十分钟 1080P 终判：P0.75/R0.75 + 两个结构性发现（09-12）

**实验**：accept_10min.mp4（决赛原片 [300s,900s]，1920×1080×30fps，18004 帧，窗口内 GT 得分 4 个：@321 ASR 单票存疑、@464/@644/@782 OCR 实锤）。生产配置（yolo26x+2×2 切片+HSV 分队）+ shadow_v1@conf0.35 + fusion + hand-roi + auto-calibrate。整段跑一遍 + 切 5×2min 分块各跑（整段 auto-calibrate FAIL 内点率 0.826，分块后 4/5 PASS——**段内标定必须分段用**，跨摇镜头撑不过 10 分钟）。

**结果（分块聚合，`tools/s2_aggregate.py`）**：
- **提案级 P 0.750 / R 0.750**：4 个 novel 提案命中 3 个 OCR 实锤 GT（dt 4.8/19.2/9.3s），6 个伪候选被 90s 新颖性链标为 followup；唯一 MISS = @321（chunk0 标定 FAIL 结构性缺口 + GT 本身 ASR 单票存疑）。
- G4 60s 片段回归检查通过（状态机得分+端区候选完好，score 1:0 不变）。

**迭代自纠实录（三轮）**：
1. 直跑：贪心匹配 P0.30/R0.75——伪候选两类：拉盘者端区等待/得分后走回（between-points）、活球僵持持盘被误投影（下条）；
2. attended 硬过滤（盘 2.5m 内有球员 ≥70% 帧才保留）：**全灭**——实测真得分段盘-最近球员世界距离 6.7-9.9m，**单应性地平面视差（持盘离地 ~1m 随相机距离放大）+ players-extent 标定端区区误差，世界坐标近距判别噪声 ~7m**，持有/落地在任何阈值下不可分（attended=0% 时帧里明明有人持盘）。降级为软信号恒上报（attended_frac 进 detail），G4 回归清零；
3. 90s 新颖性链（得分后 90s 内候选=followup 不计新提案）+ 全局最小 dt 匹配（替代按引擎时间贪心，避免大 dt 抢占）→ P/R 双 0.75。

**剩余唯一伪候选的机制**（t=267.9 左端区 x[1->1]，目检）：活球 1-1 僵持，蓝队持盘者在边线附近 stall 计数等待，被不完美标定投影进左端区边缘。**这正是复核队列（candidate=true 永不自动记分）设计要接的残余类别**；根治靠 pitch_kp 场线标定（几何）或持盘行为分类器（外观），非规则本身可解。

**决策门裁定**：
1. **端区规则不转自动记分**——维持 candidate+复核队列，新增 novelty/attended 标注辅助人工；R 0.75（3/4）且唯一 miss 是标定结构问题非检测问题，转正前置条件 = chunk0 类标定恢复（分块自动标定已解决大半）+ 攻守方向与队伍绑定（当前 team→端区映射靠 G4 单例校准）。
2. **盘通道生产配置定版：shadow_v1 @ conf0.35 + fusion**（§13.12 conf 发现的 10 分钟级验证，184f 最长轨迹，ROI 通道 tracking +198 帧增益显著——10min 素材上地滚/松散盘场景多，ROI 通道有正贡献：tracking 2362→2560，241 条 ROI 源检出）。
3. **TrackNetV3 维持 parked**：本场考试 miss 根因是标定与 GT 结构问题，远景小盘召回不是当前瓶颈。

**工具沉淀**：`tools/eval_window_gt.py`（窗口 GT 对齐）、`tools/s2_aggregate.py`（分块聚合+新颖性链+全局匹配）、`tools/summarize_events.py`（事件汇总）。

**遗留**：① chunk0 标定恢复（子窗重标定/pitch_kp）可把 R 提到 1.0 待验证；② 攻守方向自动判定（队伍↔端区绑定）；③ 伪候选根治 = pitch_kp 标定升级 或 Stage 3 持盘分类器；④ 半场换向未处理（窗口内单半场安全）。
