# Smart Body-Camera Demo: conflict-escalation visualization

This project turns a police body-camera clip and analyst-coded annotations into a polished demo video. The video shows:

- **People tracked across the footage**, with role, action and weapon labels that follow them as the camera moves
- **An animated 0–5 risk dial**, the reasons behind each score, and gesture/facial/voice signal bars
- **Escalation warnings**: a banner and red edge glow at risk 4–5, plus time-stamped event pop-ups and a running event log
- **Freeze-frame key moments** that circle and zoom in on the weapon (presentation version)
- **A risk timeline** with event markers, plus a title card and an auto-computed incident summary card

## Pipeline
| Step | Script | Automated? |
|---|---|---|
| 1. Cut and crop the clip | `ffmpeg` | Yes |
| 2. Detect and track people | `track.py` (YOLOv8x @1280 and YOLOv8m @640, ByteTrack) | Yes |
| 3. Review the tracks and link them to people | `track_merge.json` → `merge_tracks.py` | Reviewed by an analyst, merged automatically |
| 4. Code behaviors, risk and events | `annotations.json` (starting point: `excel_to_annotations.py` from the team's coding sheet) | Manual (analyst) |
| 5. Render videos and cards | `render.py`, `title_card.py`, `summary_card.py`, `make_demo.sh` | Yes |

## Run it
```bash
pip install -r requirements.txt     # plus ffmpeg (macOS: brew install ffmpeg)
# put the clip next to these files as clip.mp4 (1920x1080, 30 fps, starting at video time 00:14.6)
./make_demo.sh                      # -> out/demo_present.mp4 (half speed + key moments) and out/demo_realtime.mp4
```
`data/` holds the detector outputs, so the videos rebuild in a few minutes without re-running YOLO.
To re-detect, see the commands at the top of `make_demo.sh`. The weights download automatically.

## annotations.json
- `clip.offset`: the original-video time of the clip file's first frame. All other times are original-video times.
- `people`: ID → role (`police` / `suspect`), label, merged track ID, and optional `hide` windows (person hidden from view).
- `segments`: time windows with overall `risk` (0–5); `channels` for gesture, facial and voice (0–5, or `null` when not observable); per-person `activity`, `weapon`, `gesture` and `sudden`; plus `triggers`, `headline` and `speech`.
- `events`: time-stamped moments shown as pop-ups, log entries and timeline markers.
- `spotlights`: freeze-frame key moments (`t`, weapon `box` in pixels, title, text, hold seconds).
- `manual_boxes`: analyst-placed keyframes `[t, x1, y1, x2, y2]` for stretches the detector misses. These are drawn dashed and labeled "MANUAL BOX".

## Automated vs. manual
- **Automated:** person detection, tracking, gap bridging (≤4 frames), smoothing, all rendering, and the summary statistics.
- **Manual (analyst):** reviewing every box and linking tracker IDs to people (one wrong frame excluded), roles, actions, weapon, gestures, escalation scores, speech text, event times, weapon boxes for key moments, and boxes for 00:23.6–00:25.

Fonts: Inter (SIL Open Font License).
