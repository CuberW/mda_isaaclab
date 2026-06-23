import torch
import ultralytics
from pathlib import Path
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("ultralytics:", ultralytics.__version__)
for p in [
    Path.home() / "trashbot_ws/models/yolo11s-seg-best.pt",
    Path.home() / "trashbot_ws/models/best_seg.pt",
    Path.home() / "trashbot_ws/models/yolo11s-seg.pt",
]:
    print(p, p.exists(), p.stat().st_size if p.exists() else None)
