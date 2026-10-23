#!/usr/bin/env bash
# Build both demo videos from clip.mp4 + annotations.json.
# Re-run detection first only if you want fresh tracks (slow on CPU):
#   python track.py clip.mp4 --out work_m                                   # YOLOv8m @640
#   python track.py clip.mp4 --out work_x --model yolov8x.pt --imgsz 1280   # YOLOv8x @1280
set -euo pipefail
mkdir -p out
python merge_tracks.py track_merge.json data/tracks_merged.csv
python title_card.py out/title_present.png "Clip 00001  ·  Video time 00:15 – 00:25  ·  Half speed with key moments"
python title_card.py out/title_realtime.png "Clip 00001  ·  Video time 00:15 – 00:25  ·  Real-time"
python summary_card.py annotations.json out/summary.png
python render.py clip.mp4 data/tracks_merged.csv annotations.json --out out/body_present_src.mp4 --spotlights --tag "0.5x SPEED"
ffmpeg -v error -y -i out/body_present_src.mp4 -vf "setpts=2.0*PTS,fps=30" -c:v libx264 -crf 17 -pix_fmt yuv420p out/body_present.mp4
python render.py clip.mp4 data/tracks_merged.csv annotations.json --out out/body_realtime.mp4
for v in present realtime; do
  ffmpeg -v error -y -loop 1 -t 3.5 -framerate 30 -i out/title_$v.png -i out/body_$v.mp4 -loop 1 -t 6 -framerate 30 -i out/summary.png \
    -filter_complex "[0:v]format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st=3.1:d=0.4[a];[1:v]format=yuv420p,fade=t=in:st=0:d=0.3[b];[2:v]format=yuv420p,fade=t=in:st=0:d=0.5,fade=t=out:st=5.5:d=0.5[c];[a][b][c]concat=n=3:v=1:a=0[v]" \
    -map "[v]" -c:v libx264 -crf 17 -pix_fmt yuv420p -movflags +faststart out/demo_$v.mp4
done
echo "Done: out/demo_present.mp4 and out/demo_realtime.mp4"
