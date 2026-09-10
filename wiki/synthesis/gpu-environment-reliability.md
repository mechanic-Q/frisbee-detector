---
type: synthesis
title: WSL2/Windows 双环境 GPU 可靠性结论
tags: [wsl2, gpu, queue, reliability]
related: [detector-architecture-benchmark-v2]
created: 2026-09-10
updated: 2026-09-10
---

# WSL2/Windows 双环境 GPU 可靠性结论

## 核心结论（多次实测验证）

1. **WSL2 GPU-PV 多进程 CUDA 并发不稳定**：双车道/三车道实验 4 次连环崩溃（pin_memory 线程异常、AcceleratorError）；单进程从未崩。**WSL 侧只允许一个 CUDA 训练进程**（flock 锁强制串行）。
2. **Windows 原生训练稳定且快**（多进程问题不存在）；但 **`--cache` 死锁**（4629 张 1280px 图缓存进 RAM 时进程冻结，CPU 时间不增长）——Windows 侧禁用 --cache，用 workers=4。
3. **环境差异**：WSL2 跨 /mnt/e 读盘（9P 协议）+ Windows 原生 NTFS 的训练速度差异，被 dataloader 解码瓶颈掩盖——不是主矛盾；**多进程并发稳定性才是**。
4. **WSL tmux 会话随实例回收全灭**：长任务用 tmux 仍可能丢失（实例重启），关键产物要及时搬运；完成信号用**标记文件**而非 grep 日志（历史残留会误触发）。

## 环境级坑与修复

| 坑 | 症状 | 修复 |
|---|---|---|
| mlflow 新版 file-store 门禁 | 训练启动即崩（MlflowException） | `MLFLOW_ALLOW_FILE_STORE=true` |
| yolo CLI resume 的 GreenSocket 错误 | eventlet 猴子补丁冲突 | python API：`YOLO(last.pt).train(resume=True)` |
| ultralytics 全局 runs_dir 指向 ComfyUI | 训练产物落在 `~/comfy/ComfyUI/runs/detect/runs/` | 训后从该路径复制回项目 |
| Windows nvidia-smi 判忙误判 | explorer/chrome 被列为 compute-apps，守卫永等 | 用进程命令行判据（train.py 存在=忙） |
| shell 脚本 CRLF | `exit 127\r` 秒崩 | `.gitattributes` 强制 `*.sh text eol=lf` |
| OMP libiomp5md 双拷贝 | Windows 推理崩溃 | `KMP_DUPLICATE_LIB_OK=TRUE`（测试用） |

## GPU 队列正确形态

- 单条串行队列 + `gpu_run.sh`（flock 全局锁）+ GPU 空闲守卫（进程命令行判据）
- **绝不叠多层编排器**（09-10 事故：6 个互相竞争的队列导致重复训练/互杀/状态丢失）
- 长任务用 `nohup`/独立后台进程（WSL tmux 不可靠）
