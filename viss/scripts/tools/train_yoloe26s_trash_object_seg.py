import sys
import os

# Remove local workspace paths to force importing from environment site-packages
for path in list(sys.path):
    if 'ultralytics-main' in path or path == '':
        sys.path.remove(path)

import torch

# Register safe globals for PyTorch 2.6+ compatibility
try:
    from ultralytics.nn.tasks import YOLOESegModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([YOLOESegModel])
        print("Registered YOLOESegModel as safe global.")
except Exception as e:
    print("Could not register safe globals:", e)

from ultralytics import YOLO

def main():
    workspace = os.environ.get("TRASHBOT_WS", os.getcwd())
    # Load model from the local weights path
    weights_path = os.environ.get(
        "YOLOE_INIT_WEIGHTS",
        os.path.join(workspace, "viss/models/yoloe-26s-seg.pt"),
    )
    print(f"Loading initial YOLOE model weights from {weights_path}...")
    model = YOLO(weights_path)
    
    # Train parameters
    train_args = {
        'data': os.environ.get(
            "TRASH_YOLO_DATA",
            os.path.join(workspace, "datasets/trash_object_yoloe_seg/data.yaml"),
        ),
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,     # RTX 4090 has 24GB VRAM, batch 16 is extremely safe and stable
        'device': 0,
        'workers': 4,
        'patience': 20,
        'cache': False,
        'amp': False,    # Disable AMP to avoid c10::Half != float mismatch
        'project': os.environ.get("YOLO_RUNS_DIR", os.path.join(workspace, "viss/runs")),
        'name': 'yoloe_trash_object_seg',
        'exist_ok': True
    }
    
    # Check if dry-run argument is passed
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        print("Running 1-epoch dry-run test...")
        train_args['epochs'] = 1
        
    print("Starting training with the following arguments:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
        
    model.train(**train_args)
    print("Training process finished.")

if __name__ == '__main__':
    main()
