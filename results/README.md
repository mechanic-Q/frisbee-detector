# E4' 球员角色全自动标注 — 结果说明

管线脚本：`tools/auto_label_roles.py`（零人工，纯 CPU 可复现）。

## 输入 → 三通道 → 输出

**输入**（主树 `data/bili_final_test/`）：
- `player_frames/`：1417 张 1080P 帧（`f_*.jpg`）
- `player_labels/`：yolo26x person 预标（YOLO 格式，与帧同名）

**三通道设计**（`auto_label_roles.py` docstring）：
1. **HSV 颜色 + 黑黄条纹规则**（主通道，实际生效）：取上半身裁剪做 HSV 统计。
   裁判需同时满足条纹交替 >0.12、黄色占比 >0.04、暗部 >0.15 且红蓝均 <0.15；
   红衣/蓝衣按颜色占比阈值（>0.10 且互为 1.5 倍以上）判定，弱信号（>0.06）低置信保留；
   白衣/灰衣/远景小框归入 ignore。
2. **SigLIP 零样本投票**（交叉通道）：仅见于设计描述，最终脚本中未实现/未启用，
   实际运行路径为 ① + ③。
3. **GLM-4V-Flash 仲裁**（实际生效）：低置信框（conf<0.5）与**全部裁判候选**强制送
   GLM-4V 复核，每图最多 8 个裁剪拼图问一次，预算 400 次调用。

**输出**：
- `data/bili_final_test/role_labels/*.txt`：1416 个 YOLO 格式标签
  （`0=player-red  1=player-blue  2=referee  3=ignore`）
- `results/role_stats.json`：统计
- `results/role_spotcheck.jpg`：人工抽检图板

## 类别数量（质检复核值，与 role_stats.json 一致）

| class_id | 类别 | 框数 |
|---|---|---|
| 0 | player-red | 10899 |
| 1 | player-blue | 10613 |
| 2 | referee | 3521 |
| 3 | ignore | 9039 |
| 合计 | | 34072 |

帧数：1416（1417 帧中 1 帧无 person 框被跳过）；VLM 仲裁调用：400 次（预算用满）。

## 质检结论（2026-09-08）

- 文件数 1416，符合预期 ≈1416。
- 随机抽检 200 个文件（4845 行）：每行 5 列、class_id ∈ {0,1,2,3}、坐标均于 [0,1]，零错误。
- 全量 34072 行扫描：0 列数异常、0 非法类别、0 坐标越界，无空文件。
- 全量 class 分布与 `role_stats.json` 的 `by_class` 完全一致。

## 已知残余误差

- **观众误标为裁判候选**：看台上穿深色/黄黑花纹服装的观众可能触发 HSV 条纹规则被判为
  referee。标签本身没有场地位置过滤，建议下游使用前做**场地多边形过滤**
  （仅保留落在场地多边形内的 referee 框）。
- ignore 类是"非红非蓝非裁判"的兜底（含白衣球员、观众、远景小框），语义混杂。
- VLM 预算（400 次）按文件名顺序耗尽，靠后的帧中低置信框仅由 HSV 规则判定、未经仲裁。

## 如何用

这些标签可直接作为 **player-red / player-blue / referee 三类检测器**的训练数据：
导出训练集时丢弃 class_id=3（ignore）的行即可。使用前请按仓库数据泄漏规则确认
相关帧不落在 eval 视频的 `exclude_ranges` 内。
