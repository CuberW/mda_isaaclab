import sys
import os
import glob
import random
import torch

# Remove local workspace paths to force importing from environment site-packages
for path in list(sys.path):
    if 'ultralytics-main' in path or path == '':
        sys.path.remove(path)

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
    model_path = '/home/robot/trashbot_ws/runs/yoloe_trash_object_seg/weights/best.pt'
    data_yaml = '/home/robot/trashbot_ws/datasets/trash_object_yoloe_seg/data.yaml'
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please wait for training to finish.")
        return
        
    print(f"Loading trained YOLOE model from {model_path}...")
    model = YOLO(model_path)
    
    print("\n" + "="*40)
    print("RUNNING VAL EVALUATION")
    print("="*40)
    metrics = model.val(data=data_yaml)
    
    # Print results
    print("\nmAP Results Summary:")
    print(f"  Box mAP50: {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95: {metrics.box.map:.4f}")
    print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")
    
    print("\n" + "="*40)
    print("RUNNING INFERENCE PREDICTION ON SAMPLE IMAGES")
    print("="*40)
    
    # Look for images inside /home/robot/trashbot_ws/data/images recursively
    image_pattern = "/home/robot/trashbot_ws/data/images/**/*.jpg"
    images = glob.glob(image_pattern, recursive=True)
    if not images:
        # Fallback to check if png images exist
        image_pattern = "/home/robot/trashbot_ws/data/images/**/*.png"
        images = glob.glob(image_pattern, recursive=True)
        
    if not images:
        print("Warning: No sample images found under /home/robot/trashbot_ws/data/images/ recursively.")
        print("Skipping visualization prediction step.")
        return
        
    print(f"Found {len(images)} sample images. Selecting 5 at random for prediction overlay...")
    sampled_imgs = random.sample(images, min(5, len(images)))
    
    output_dir = '/home/robot/trashbot_ws/data/logs/val_predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving overlay images to: {output_dir}")
    for idx, img_path in enumerate(sampled_imgs):
        filename = os.path.basename(img_path)
        print(f"  Predicting on: {filename}...")
        results = model.predict(source=img_path, conf=0.25, imgsz=640, device=0)
        
        # Save results (visualized plots)
        for r in results:
            save_path = os.path.join(output_dir, f"prediction_{idx}_{filename}")
            r.save(filename=save_path)
            print(f"    Saved overlay: {save_path}")
            
    print("\nValidation and test predictions completed successfully.")

if __name__ == '__main__':
    main()
