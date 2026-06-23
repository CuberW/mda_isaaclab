import cv2
from pathlib import Path
import numpy as np

root = Path.home() / "trashbot_ws" / "data" / "images"
sessions = sorted([p for p in root.iterdir() if p.is_dir()])

if not sessions:
    raise RuntimeError("No image session found.")

session = sessions[-1]
images = sorted(list(session.glob("*.jpg")) + list(session.glob("*.png")))[:9]

if not images:
    raise RuntimeError("No image files found.")

thumbs = []
for p in images:
    img = cv2.imread(str(p))
    if img is None:
        continue
    img = cv2.resize(img, (320, 180))
    thumbs.append(img)

while len(thumbs) < 9:
    thumbs.append(np.zeros((180, 320, 3), dtype=np.uint8))

preview = np.vstack([
    np.hstack(thumbs[0:3]),
    np.hstack(thumbs[3:6]),
    np.hstack(thumbs[6:9]),
])

out = session / "dataset_preview.jpg"
cv2.imwrite(str(out), preview)
print(f"Preview saved to: {out}")
