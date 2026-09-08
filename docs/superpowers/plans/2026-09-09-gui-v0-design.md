# GUI v0 设计文档 — 窗口/层级/解耦/排查/流程

日期：2026-09-09。配套：执行计划 `2026-09-09-gui-v0.md`、验收报告 `2026-09-09-gui-v0-acceptance-report.md`。
截图样张：`runs/gui_analysis/screenshots/01–04*.png`（由 `tools/gui_screenshots.py` 离屏渲染生成，可随时重生成）。

## 1. 窗口布局与层级功能

```
┌─ MainWindow ────────────────────────────────────────────────────────────────┐
│ 工具栏: 打开视频 | 打开分析结果 | 启动分析 | 取消分析 | 场地标定 | 队伍改判      │
│ 快捷键: Ctrl+O 打开 / Space 播放暂停 / ←→ 逐帧                                │
├──────────────────────────────────────────────┬──────────────────────────────┤
│ 中央 VideoPlayerWidget（播放画布）             │ 分析日志 dock（QPlainTextEdit）│
│  QMediaPlayer → QVideoSink → QImage → QPainter│  worker stdout 逐行原文       │
│  第1层 原始视频帧（letterbox 居中）            │  [gpu-queue] 锁状态           │
│  第2层 场地线叠加（已标定时，绿色投影）         │  {"type":"progress"...}       │
│  第3层 检测叠加（球员框+队色+ID+置信度）        │  异常栈尾部 → 排查第一入口     │
│  第4层 标定点（标定模式，绿圈+序号）            │                              │
├──────────────────────────────────────────────┴──────────────────────────────┤
│ 播放控制 dock: ◀帧 | 播放/暂停 | 帧▶ | 时间轴滑杆                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 状态栏: 运行消息 | 分析进度条(0..total_frames) | 当前帧号 f9000                │
└──────────────────────────────────────────────┬──────────────────────────────┘
                                               │ QProcess + JSON-lines（UTF-8）
┌──────────────────────────────────────────────┴──────────────────────────────┐
│ wsl.exe --cd <worktree> -e bash tools/gpu_run.sh gui-analysis \              │
│   python3 -m frisbee_analyzer.pipeline --video … --output-dir …              │
│  protocol.py  消息 schema + Windows↔WSL 路径转换（两侧共用的契约）             │
│  tracking.py  YOLO(零样本/微调权重) + BoT-SORT 逐帧轨迹                       │
│  team.py      HSV 球衣色特征 KMeans 分队（SigLIP 兜底）→ team_id + team_colors│
│  events.py    事件引擎骨架（Phase 3 接入，吃世界坐标+track+team）             │
└─────────────────────────────────────────────────────────────────────────────┘
```

对话框（模态层级）：
- **CalibrateDialog**：下拉选世界坐标（WFDF 100×37m 的 9 个角/中点）→ 点画面 → ≥4 点 →
  计算单应性（RANSAC + RMSE 显示 + 场地线叠加预览）→ 保存 `configs/homography/<视频名>.json`。
- **TeamOverrideDialog**：全部 track 的表格（帧数/平均置信度/当前队），下拉改判即时
  写回 tracks.json（`team_overrides` + 原子替换），内存同步刷新叠加。

## 2. 解耦设计评估

| 边界 | 机制 | 收益 |
|---|---|---|
| GUI 进程 ⟂ worker 进程 | QProcess + JSON-lines，崩溃互不拖垮；worker 可独立 CLI 运行 | GUI 崩不影响分析；分析可无 GUI 复现 |
| Windows ⟂ WSL | protocol.win_to_wsl/wsl_to_win 集中转换；stdout 强制 UTF-8 | 换 GPU 宿主（如 Windows 原生 CUDA）只改启动命令 |
| 算法 ⟂ I/O | team.py 纯函数（label_from_color/majority_vote）与 collect_crops 分离 | 无 GPU/视频即可单测 |
| UI ⟂ 路径/启动参数 | worker_paths.py 纯 stdlib 构造 argv | 参数逻辑可在 conda/WSL 任意环境测试 |
| 队伍语义 ⟂ 颜色假设 | doc["team_colors"] 记录实际红蓝，UI 按 team_colors 着色 | 换素材队色变化不改 UI 代码 |
| 坐标换算单点 | overlay.letterbox/widget_to_video 全 UI 共用 | 标定点击与框绘制同一数学，不会各算各的 |

已知的妥协（有意为之，v0 可接受）：TEAM_NAMES 中文名在无 team_colors 时按 0=红/1=蓝
显示（颜色主路下恰好成立）；QMediaPlayer 难以无头单测（用 offscreen 烟雾 + 截图脚本代替）；
标定世界坐标预设硬编码 WFDF 模板（多模板配置留待需要时）。

## 3. 排查机制

排查链：**GUI 分析日志 dock（第一现场）→ tracks.json / team_color_check.json（产物侧证据）
→ pytest 分层复现（协议/分队/参数构造均可无 GPU 复现）→ worker 单独 CLI 重跑**。

| 症状 | 首查 | 依据 |
|---|---|---|
| 分析启动即失败 | 日志 dock 里 `[worker error]` / 退出码 | 退出码 0/1/2 = 成功/出错/取消；顶层异常必回传 error 消息 |
| 进度条卡住 | 日志里 `[gpu-queue] 排队等待` | 任务在锁队列里，不是挂了 |
| 播放有画面无叠加 | 状态栏帧号 vs tracks.json 的 frames 键 | fps 不匹配或该帧无结果（短命 track 为 None 属预期） |
| 队色反了 | doc["team_colors"] | 改判入口即时修正并落盘；或重跑 --team-only |
| tracks.json 损坏 | 不可能半截——原子写（tmp+replace） | 写入中断只会丢整个新文件，旧文件完好 |
| 非 JSON 输出炸解析 | parse_line 降级为 log | 库告警不会打断 GUI 读取循环 |

## 4. 设计检查点

- **里程碑门**：M1 协议+跟踪 → M2 分队 → M3 GUI 壳 → M4 标定/改判 → M5 验收，每门有完成标志（见执行计划）。
- **自动化检查点**（每次改动后必须全绿）：
  - `pytest tests/ -q`：131 项（协议 schema/路径转换、分队纯函数、pipeline 错误路径、worker argv、events 引擎）；
  - `--smoke` 离屏烟雾：窗口构造 + 真实数据加载 + 完整渲染一遍，输出 SMOKE PASS；
  - `tools/gui_screenshots.py`：四层级截图可重现（本文件的样张即出自它）；
  - 验收量化门：`tools/team_color_check.py` 抽检一致性 ≥0.90（当前 0.9777/0.9735）。
- **数据检查点**：tracks.json 原子写；改判每步落盘（决策即保存，沿用 review 工具惯例）。

## 5. 端到端流程

```
打开视频 ──► 自动加载同名 tracks.json / 标定 JSON（存在则跳过对应步骤）
   │
   ├─► [启动分析] ─► QProcess: gpu_run.sh 排队 ─► worker:
   │      人员检测(yolo26x 零样本/微调) → BoT-SORT 轨迹 → HSV 分队
   │      ├ stdout: meta/progress/log ──► 进度条 + 日志 dock
   │      └ 完成: tracks.json（原子写）──► 自动加载 ──► 叠加回放
   │           （取消: terminate→kill，旧产物不受影响）
   ├─► [场地标定] ─► 点 ≥4 点 → RANSAC 单应性 → 预览 → 存 homography/<视频名>.json
   ├─► [队伍改判] ─► 表格改队 → 内存+磁盘即时同步
   └─► 回放：position(ms)→帧号→该帧 dets → QPainter 叠加
```
