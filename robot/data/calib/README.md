# 标定文件

此目录存放每台机器不同的标定结果 (不提交git)。

## 文件列表
- `intrinsics.yaml` — 相机内参 (K矩阵 + 畸变系数)
- `hand_eye.yaml` — 手眼标定外参 (T_base_cam)
- `metric_align.yaml` — DA3-Small度量对齐参数 (alpha, beta)

## 生成方式
```bash
python scripts/calibrate_camera.py      # 相机内参
python scripts/calibrate_hand_eye.py    # 手眼标定
```
