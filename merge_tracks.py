"""Merge verified tracks from two detector runs into one tracks.csv.

For each person, frames covered by the primary run (e.g. YOLOv8x @1280) are used first;
frames it missed are filled from the secondary run (e.g. YOLOv8m @640).
Tracker IDs were linked to people after reviewing every box; frames listed under
"exclude" were rejected in review (e.g. a box that landed on the officer's own hand).

Usage: python merge_tracks.py track_merge.json merged_tracks.csv
"""
import csv, json, sys

cfg = json.load(open(sys.argv[1]))
offset = cfg.get("offset", 0.0)
fps = cfg.get("fps", 30.0)
out_rows = {}
person_id = {pid: i + 1 for i, pid in enumerate(cfg["people"])}

def load(path):
    return list(csv.DictReader(open(path)))

for rank, run in enumerate(cfg["runs"]):            # runs in priority order
    for r in load(run["tracks"]):
        tid, fr = int(r["track_id"]), int(r["frame"])
        t = fr / fps + offset
        pid = None
        for p, spec in cfg["people"].items():
            for ent in spec.get(run["name"], []):
                ids, t0, t1 = (ent, -1e9, 1e9) if isinstance(ent, int) else (ent["id"], ent.get("from", -1e9), ent.get("to", 1e9))
                if tid == ids and t0 <= t <= t1:
                    pid = p
        if pid is None:
            continue
        if any(a <= t <= b for a, b in cfg["people"][pid].get("exclude", [])):
            continue
        key = (pid, fr)
        if key in out_rows:          # already filled by a higher-priority run
            continue
        r = dict(r); r["track_id"] = person_id[pid]; r["source"] = run["name"]
        out_rows[key] = r

rows = sorted(out_rows.values(), key=lambda r: (int(r["frame"]), int(r["track_id"])))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "motion", "source"])
    w.writeheader(); w.writerows(rows)
for p, i in person_id.items():
    n = sum(1 for r in rows if int(r["track_id"]) == i)
    print(f"{p} -> track_id {i}: {n} frames")
