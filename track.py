"""Step 1 (automated): detect and track people in the clip with YOLOv8 + ByteTrack.

Outputs:
  tracks.csv      one row per (frame, track_id) with box, confidence, motion score
  track_ids.jpg   contact sheet showing each track ID so you can map IDs -> P1/S1 in annotations.json

Usage: python track.py clip.mp4 --start 15 --end 25 --out work/
"""
import argparse, os, csv
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default="work")
ap.add_argument("--model", default="yolov8m.pt")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
end_f = int(a.end * fps) if a.end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

model = YOLO(a.model)
rows, crops = [], {}
f = int(a.start * fps)
prev = {}
while f < end_f:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, classes=[0], conf=a.conf,
                    tracker="bytetrack.yaml", imgsz=a.imgsz, verbose=False)[0]
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid, c in zip(r.boxes.xyxy.cpu().numpy(),
                                            r.boxes.id.cpu().numpy().astype(int),
                                            r.boxes.conf.cpu().numpy()):
            cx, cy, h = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1)
            # motion = centre displacement per second, normalised by body height
            # (camera shake inflates this, so it is a hint, not ground truth)
            motion = 0.0
            if tid in prev:
                pcx, pcy, pf = prev[tid]
                motion = np.hypot(cx - pcx, cy - pcy) / h * fps / max(f - pf, 1)
            prev[tid] = (cx, cy, f)
            rows.append([f, round(f / fps, 3), tid, *map(lambda v: round(float(v), 1), (x1, y1, x2, y2)),
                         round(float(c), 3), round(float(motion), 3)])
            area = (x2 - x1) * (y2 - y1)
            if tid not in crops or area > crops[tid][0]:
                crop = frame[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)].copy()
                crops[tid] = (area, crop, f / fps)
    f += 1

with open(os.path.join(a.out, "tracks.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "motion"])
    w.writerows(rows)

# contact sheet of track IDs
tiles = []
for tid, (_, crop, t) in sorted(crops.items()):
    n = sum(1 for r in rows if r[2] == tid)
    if crop.size == 0:
        continue
    tile = cv2.resize(crop, (160, int(160 * crop.shape[0] / max(crop.shape[1], 1))))
    tile = cv2.resize(tile, (160, 240)) if tile.shape[0] > 240 else cv2.copyMakeBorder(
        tile, 0, 240 - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    cv2.rectangle(tile, (0, 0), (160, 34), (0, 0, 0), -1)
    cv2.putText(tile, f"id {tid}  ({n}f)", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(tile)
if tiles:
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(a.out, "track_ids.jpg"), sheet)
print(f"{len(rows)} detections, {len(crops)} track ids, fps={fps:.2f}")
