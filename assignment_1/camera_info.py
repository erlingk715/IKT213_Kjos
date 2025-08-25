import cv2, os
from pathlib import Path

out_path = Path.home() / "IKT213_Kjos"  / "IKT213_Kjos" / "assignment_1" / "solutions" / "camera_outputs.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    fps, w, h = 0, 0, 0
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"height: {h}\n")
    f.write(f"width: {w}\n")

print(f"Skrev til: {out_path}")
