"""Closing 'incident summary' card, computed from annotations.json so every number matches the demo.
Usage: python summary_card.py annotations.json out.png
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda w, s: ImageFont.truetype(os.path.join(HERE, "fonts", f"Inter-{w}.otf"), s)
BG, PANEL, LINE = (12, 15, 21), (20, 24, 32), (46, 53, 66)
TEXT, MUTED, FAINT = (234, 238, 244), (142, 152, 168), (96, 106, 122)
RISK_COL = {0: (100, 116, 139), 1: (34, 197, 94), 2: (132, 204, 22), 3: (234, 179, 8), 4: (249, 115, 22), 5: (239, 68, 68)}
RISK_NAME = {0: "None", 1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very high"}
KIND = {"info": (59, 130, 246), "warn": (234, 179, 8), "danger": (239, 68, 68)}

ann = json.load(open(sys.argv[1])); out = sys.argv[2]
clip, segs, events = ann["clip"], sorted(ann["segments"], key=lambda s: s["start"]), sorted(ann["events"], key=lambda e: e["t"])
start, end = clip["start"], clip["end"]
dur = end - start
peak = max(s["risk"] for s in segs)
t_peak = min(s["start"] for s in segs if s["risk"] == peak)
high = sum(min(s["end"], end) - max(s["start"], start) for s in segs if s["risk"] >= 4)
weapon_t = next(e["t"] for e in events if "weapon" in e["title"].lower())
first_person = min(s["start"] for s in segs if s.get("people"))
fmt = lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}"

im = Image.new("RGB", (1920, 1080), BG); d = ImageDraw.Draw(im)
X = 120
d.text((X, 84), "INCIDENT SUMMARY", font=f("SemiBold", 26), fill=MUTED)
d.text((X, 120), f"{clip['id']}  ·  {clip['location']}  ·  {clip['type']}  ·  video time {fmt(start)[:5]} – {fmt(end)[:5]}",
       font=f("Bold", 48), fill=TEXT)

tiles = [
    ("Peak escalation risk", f"{peak} / 5", RISK_NAME[peak] + f" from {fmt(t_peak)}", RISK_COL[peak]),
    ("Weapon first visible", fmt(weapon_t), f"{weapon_t - first_person:.1f} s after the suspect appears", RISK_COL[5]),
    ("Time at high risk (4–5)", f"{high:.1f} s", f"{high / dur * 100:.0f}% of the {dur:.0f}-second clip", RISK_COL[4]),
    ("People tracked", f"{len(ann['people'])}", "S1 suspect and P2 officer", (59, 130, 246)),
]
tw_ = (1920 - 2 * X - 3 * 32) / 4
for i, (lab, big, sub, col) in enumerate(tiles):
    x0 = X + i * (tw_ + 32); y0 = 230
    d.rounded_rectangle([x0, y0, x0 + tw_, y0 + 200], 16, fill=PANEL)
    d.rectangle([x0, y0 + 24, x0 + 5, y0 + 176], fill=col)
    d.text((x0 + 32, y0 + 26), lab.upper(), font=f("SemiBold", 18), fill=MUTED)
    d.text((x0 + 32, y0 + 58), big, font=f("Bold", 64), fill=col)
    d.text((x0 + 32, y0 + 148), sub, font=f("Medium", 19), fill=TEXT)

# risk step chart with event markers
cx0, cx1, cy0, cy1 = X, 1920 - X, 520, 760
d.text((cx0, 476), "OVERALL ESCALATION RISK OVER TIME", font=f("SemiBold", 18), fill=MUTED)
for v in range(0, 6):
    yv = cy1 - v / 5 * (cy1 - cy0)
    d.line([(cx0 + 40, yv), (cx1, yv)], fill=LINE if v else FAINT, width=1)
    d.text((cx0, yv - 10), str(v), font=f("Medium", 16), fill=FAINT)
tx = lambda t: cx0 + 40 + (t - start) / dur * (cx1 - cx0 - 40)
im2 = im.convert("RGBA"); ov = Image.new("RGBA", im2.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
for s in segs:
    yv = cy1 - s["risk"] / 5 * (cy1 - cy0)
    do.rectangle([tx(s["start"]) + 1, yv, tx(s["end"]) - 1, cy1], fill=RISK_COL[s["risk"]] + (70,))
    do.rectangle([tx(s["start"]) + 1, yv, tx(s["end"]) - 1, yv + 5], fill=RISK_COL[s["risk"]] + (255,))
im = Image.alpha_composite(im2, ov).convert("RGB"); d = ImageDraw.Draw(im)
for ts in range(int(start), int(end) + 1):
    lab = f"00:{ts:02d}"
    d.text((tx(ts) - d.textlength(lab, font=f("Medium", 16)) / 2, cy1 + 10), lab, font=f("Medium", 16), fill=FAINT)
for e in events:
    ex = tx(e["t"]); col = KIND.get(e["kind"], TEXT)
    d.line([(ex, cy0 - 6), (ex, cy1)], fill=col, width=2)
    d.ellipse([ex - 6, cy0 - 12, ex + 6, cy0], fill=col)

# key moments list
d.text((X, 806), "KEY MOMENTS", font=f("SemiBold", 18), fill=MUTED)
colw = (1920 - 2 * X) / len(events)
for i, e in enumerate(events):
    x0 = X + i * colw; col = KIND.get(e["kind"], TEXT)
    d.rectangle([x0, 842, x0 + 4, 916], fill=col)
    d.text((x0 + 18, 840), fmt(e["t"]), font=f("Bold", 22), fill=col)
    d.text((x0 + 18, 870), e["title"], font=f("SemiBold", 20), fill=TEXT)
    d.text((x0 + 18, 896), e["text"], font=f("Medium", 16), fill=MUTED)

d.line([(X, 962), (1920 - X, 962)], fill=LINE)
d.text((X, 980), "Automated: person detection and tracking (YOLOv8 + ByteTrack), visualization.   "
                 "Analyst-coded: roles, actions, weapon, escalation scores, speech (research coding sheet).",
       font=f("Medium", 18), fill=MUTED)
d.text((X, 1010), "Built with Python, OpenCV, Pillow and FFmpeg  ·  Prepared by Namratha Indla, UMass Boston College of Management",
       font=f("Medium", 18), fill=FAINT)
im.save(out)
print(f"peak={peak} at {t_peak}, weapon={weapon_t}, high={high:.1f}s, people={len(ann['people'])}")
