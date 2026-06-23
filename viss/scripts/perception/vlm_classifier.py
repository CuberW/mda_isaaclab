import os
import cv2
import numpy as np

def crop_object_by_mask(img_path, polygon_pts, bbox_xyxy, pad=10):
    """
    Crops the target object from the original image based on mask polygon or bounding box.
    
    Args:
        img_path (str): Path to the source image file
        polygon_pts (list): List of [x, y] coordinates representing the mask boundary
        bbox_xyxy (list): [x1, y1, x2, y2] bounding box coordinates
        pad (int): Extra padding pixels around the crop
    
    Returns:
        np.ndarray: Cropped object image, or None if image reading fails
    """
    if not os.path.exists(img_path):
        print(f"Error: Image path {img_path} does not exist.")
        return None
        
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    H, W = img.shape[:2]
    
    # 1. Determine bounding box for crop (using polygon if available, else bbox)
    if polygon_pts and len(polygon_pts) >= 3:
        pts = np.array(polygon_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        
    # 2. Add padding and clamp to image boundaries
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    
    # 3. Crop and return
    crop = img[y1:y2, x1:x2]
    return crop


def classify_crop_with_vlm(crop_img):
    """
    Placeholder interface for Qwen3-VL or other Multimodal Large Models.
    
    Args:
        crop_img (np.ndarray): Cropped image array of the single garbage object
        
    Returns:
        dict: Classification results containing category labels and target bin mapping
    """
    # =========================================================================
    # TODO: Integrate your Qwen3-VL API call here.
    # Below is a pseudocode example:
    # 
    # encoded_image = encode_image_to_base64(crop_img)
    # response = qwen_client.chat(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"image": encoded_image},
    #                 {"text": "Determine the category of this garbage item. Return exactly one of: 'recyclable', 'kitchen', 'hazardous', 'other'."}
    #             ]
    #         }
    #     ]
    # )
    # ans = response.output.choices[0].message.content.strip().lower()
    # =========================================================================
    
    # Placeholder implementation
    vlm_result = {
        "raw_garbage_category": "unknown",
        "category": "unknown",         # recyclable, kitchen, hazardous, other
        "target_bin": "unknown",      # bin_recyclable_blue, bin_kitchen_green, etc.
        "confidence": 1.0,
    }
    
    return vlm_result
