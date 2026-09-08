"""WFDF 飞盘场合成数据渲染包（E5 phase-3：合成→真实训练场线/单应性检测模型）。

分层（解耦原则，每层可独立测试替换）：
- field_model.py  纯几何：场地常量/线段/关键点（无渲染依赖）
- camera.py       相机：参数采样/内外参/投影
- raster.py       光栅化：草地/线段/遮挡物/后期（不依赖相机模型）
- render.py       编排：相机+场地 → (图像, 真值 H/关键点)
- render_synth_field.py  CLI
"""
