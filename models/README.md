# 模型权重下载与放置

本仓库默认不提交大型模型权重。Task319 GitHub 交付版只提交源码、仿真资产、文档、PPT 和压缩演示视频；模型权重按 `docs/task319_delivery/README.md` 的“模型下载位置”小节下载或从交付模型附件复制。

## 公开模型来源
- Grounding DINO (base): https://huggingface.co/IDEA-Research/grounding-dino-base
- SAM (vit_b): https://huggingface.co/facebook/sam-vit-base
- DINOv2 (vit_b): https://huggingface.co/facebook/dinov2-base
- DA3-Small: https://huggingface.co/depth-anything/DA3-Small
- YOLOv8n: 代码自动下载

项目自训练/整理权重需要由交付模型附件提供：

- `viss/models/yolo11s-seg-best.pt`
- `viss/models/best_seg.pt`
- `viss/models/yolo11s-seg.pt`

## 放置位置
```
models/
├── grounding-dino-base/
├── sam-vit-b/
├── dinov2-base/
└── da3-small/
viss/models/
└── yolo11s-seg-best.pt
```
