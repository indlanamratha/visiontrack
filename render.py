"""Render the annotated Smart Body-Camera demo.

Combines automated person tracking (tracks.csv from track.py) with analyst-coded
annotations (annotations.json, following the research team's coding sheet):

  on the video   tracked people with role / action / weapon labels, a body-cam HUD,
                 escalation banner + red edge glow, event pop-ups, subtitles,
                 optional freeze-frame "key moments" that zoom in on the weapon
  right panel    animated risk dial, reasons ("why this score"), escalation signals,
                 people in scene, running event log, provenance
  bottom strip   risk timeline (overall / gesture / facial / voice) with event markers

Usage:
  python render.py clip.mp4 work/tracks.csv annotations.json --out demo.mp4 [--spotlights] [--tag "0.5x SPEED"]
"""
import argparse, csv, json, math, os, subprocess, tempfile
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
def font(weight, size):
    return ImageFont.truetype(os.path.join(HERE, "fonts", f"Inter-{weight}.otf"), size)

# ---------------- layout + palette ----------------
W, H = 1920, 1080
VW, VH = 1440, 810
PX = VW
BG, PANEL, CARD, LINE = (12, 15, 21), (20, 24, 32), (30, 36, 47), (46, 53, 66)
TEXT, MUTED, FAINT = (234, 238, 244), (142, 152, 168), (96, 106, 122)
ROLE = {"police": (59, 130, 246), "suspect": (245, 158, 11), "bystander": (148, 163, 184)}
DANGER = (239, 68, 68)
RISK_COL = {0: (100, 116, 139), 1: (34, 197, 94), 2: (132, 204, 22), 3: (234, 179, 8), 4: (249, 115, 22), 5: (239, 68, 68)}
RISK_NAME = {0: "NONE", 1: "VERY LOW", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "VERY HIGH"}
KIND_COL = {"info": ROLE["police"], "warn": RISK_COL[3], "danger": DANGER}
NOT_THREAT = ("none", "", None, "holstered")

F = {k: font(w, s) for k, (w, s) in {
    "hud": ("SemiBold", 22), "tag": ("SemiBold", 14), "chip": ("Bold", 19), "chip2": ("Medium", 17),
    "sec": ("Bold", 13), "meta": ("Medium", 14), "body": ("Medium", 17), "small": ("Medium", 14), "tiny": ("Medium", 12),
    "dial": ("Bold", 76), "dial2": ("SemiBold", 22), "level": ("Bold", 22),
    "banner": ("Bold", 28), "banner2": ("Medium", 19), "toast1": ("Bold", 17), "toast2": ("Medium", 16),
    "sub": ("SemiBold", 30), "subwho": ("Bold", 15), "mark": ("SemiBold", 13),
    "spot1": ("Bold", 26), "spot2": ("Medium", 18), "log_t": ("SemiBold", 14), "log": ("Medium", 15)}.items()}

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("tracks"); ap.add_argument("annotations")
ap.add_argument("--out", default="demo.mp4")
ap.add_argument("--tag", default="", help="small tag under the REC clock, e.g. '0.5x SPEED'")
ap.add_argument("--spotlights", action="store_true", help="insert freeze-frame key moments")
ap.add_argument("--show-unmapped", action="store_true")
a = ap.parse_args()

ann = json.load(open(a.annotations))
clip = ann["clip"]
start, end = clip["start"], clip["end"]          # original-video time
OFFSET = clip.get("offset", 0.0)                 # original-video time of the clip file's first frame
segs = sorted(ann["segments"], key=lambda s: s["start"])
events = sorted(ann.get("events", []), key=lambda e: e["t"])
spots = sorted(ann.get("spotlights", []), key=lambda e: e["t"]) if a.spotlights else []
people = ann["people"]
track_to_person = {t: pid for pid, p in people.items() for t in p["tracks"]}
first_seen = {}
for sg in segs:
    for pid in sg.get("people", {}):
        first_seen.setdefault(pid, sg["start"])

def seg_at(t):
    for sg in segs:
        if sg["start"] <= t < sg["end"]:
            return sg
    return segs[-1] if t >= segs[-1]["end"] else None

def alert_since(t):
    """Start time of the continuous run of risk>=4 segments containing t."""
    idx = next((i for i, sg in enumerate(segs) if sg["start"] <= t < sg["end"]), len(segs) - 1)
    while idx > 0 and segs[idx - 1].get("risk", 0) >= 4 and abs(segs[idx - 1]["end"] - segs[idx]["start"]) < 1e-6:
        idx -= 1
    return segs[idx]["start"]

def channel(sg, k):
    return (sg or {}).get("channels", {}).get(k, 0)

def is_armed(st):
    return bool(st) and st.get("weapon", "none") not in NOT_THREAT

def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"

# ---------------- tracks: merge IDs per person, bridge short gaps, smooth ----------------
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
f0, f1 = int(round((start - OFFSET) * fps)), int(round((end - OFFSET) * fps))
raw = defaultdict(dict)
for r in csv.DictReader(open(a.tracks)):
    tid, fr = int(r["track_id"]), int(r["frame"])
    key = track_to_person.get(tid, f"_{tid}")
    box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
    if fr in raw[key]:
        ob = raw[key][fr]
        if (ob[2] - ob[0]) * (ob[3] - ob[1]) >= (box[2] - box[0]) * (box[3] - box[1]):
            continue
    raw[key][fr] = box

MAX_GAP = int(ann.get("max_gap_frames", 4))
boxes = {}
for key, d in raw.items():
    frames = sorted(d)
    filled = {}
    for i, fr in enumerate(frames):
        filled[fr] = d[fr]
        if i + 1 < len(frames):
            nxt = frames[i + 1]
            if 1 < nxt - fr <= MAX_GAP:          # bridge only brief detector dropouts
                for g in range(fr + 1, nxt):
                    w = (g - fr) / (nxt - fr)
                    filled[g] = d[fr] * (1 - w) + d[nxt] * w
    sm, prev = {}, None
    for fr in sorted(filled):
        b = filled[fr]
        prev = b if prev is None or fr - 1 not in sm else 0.75 * b + 0.25 * prev
        sm[fr] = prev
    boxes[key] = sm

# analyst-rejected windows (e.g. person hidden behind the officer's hand)
for pid, p in people.items():
    for a0, a1 in p.get("hide", []):
        for fr in list(boxes.get(pid, {})):
            if a0 <= fr / fps + OFFSET <= a1:
                del boxes[pid][fr]

manual = defaultdict(set)
for pid, kfs in ann.get("manual_boxes", {}).items():
    kfs = sorted(kfs)
    boxes.setdefault(pid, {})
    for (ta, *ba), (tb, *bb) in zip(kfs, kfs[1:]):
        for fr in range(int(round((ta - OFFSET) * fps)), int(round((tb - OFFSET) * fps)) + 1):
            if fr in boxes[pid] and fr not in manual[pid]:
                continue
            w = (fr / fps + OFFSET - ta) / (tb - ta)
            boxes[pid][fr] = np.array(ba, float) * (1 - w) + np.array(bb, float) * w
            manual[pid].add(fr)

# ---------------- geometry ----------------
src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale = min(VW / src_w, VH / src_h)
dw, dh = int(src_w * scale), int(src_h * scale)
ox, oy = (VW - dw) // 2, (VH - dh) // 2

yy_, xx_ = np.mgrid[0:dh, 0:dw]
edge = np.minimum.reduce([xx_, yy_, dw - 1 - xx_, dh - 1 - yy_]).astype(float)
GLOW = (np.clip(1 - edge / 90.0, 0, 1) ** 1.6 * 255).astype(np.uint8)   # red edge glow ramp

def rr(d, xy, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(xy, r, fill=fill, outline=outline, width=width)

def tw(d, s, f):
    return d.textlength(s, font=f)

def wrap(d, s, f, maxw):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if tw(d, t, f) <= maxw: cur = t
        else: lines.append(cur); cur = w_
    if cur: lines.append(cur)
    return lines

def ease(x):
    x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3

def warn_icon(d, x, y, s, fg, bg):
    d.polygon([(x + s / 2, y), (x + s, y + s * 0.9), (x, y + s * 0.9)], fill=fg)
    d.rectangle([x + s / 2 - 1.5, y + s * 0.3, x + s / 2 + 1.5, y + s * 0.62], fill=bg)
    d.rectangle([x + s / 2 - 1.5, y + s * 0.7, x + s / 2 + 1.5, y + s * 0.78], fill=bg)

# ---------------- frame drawing ----------------
state = {"risk": None}

def draw_frame(frame_bgr, fr, t, spot=None, sp=0.0):
    sg = seg_at(t)
    risk = sg.get("risk", 0) if sg else 0
    if spot is None:
        state["risk"] = risk if state["risk"] is None else state["risk"] + (risk - state["risk"]) * 0.18
    shown = state["risk"]
    alert = risk >= 4
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.4) if spot is None else 0.6

    canvas = Image.new("RGB", (W, H), BG)
    vid = Image.fromarray(cv2.cvtColor(cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    canvas.paste(vid, (ox, oy))
    canvas = canvas.convert("RGBA")

    if alert and spot is None:
        glow = Image.new("RGBA", (dw, dh), DANGER + (0,))
        glow.putalpha(Image.fromarray((GLOW * (0.35 + 0.45 * pulse)).astype(np.uint8)))
        canvas.alpha_composite(glow, (ox, oy))

    ui = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)        # banner, pop-up, subtitles, HUD
    lbox = Image.new("RGBA", (W, H), (0, 0, 0, 0)); db = ImageDraw.Draw(lbox)   # person boxes
    lchip = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dc = ImageDraw.Draw(lchip) # person labels
    placed = [[ox + 10, oy + 10, ox + 262, oy + (92 if a.tag or spot else 62)]]

    # ---- banner ----
    if alert and spot is None:
        msg, sub = "ESCALATION WARNING", sg.get("headline", "")
        bwid = max(tw(d, msg, F["banner"]) + 76, tw(d, sub, F["banner2"]) + 44)
        bx0 = ox + (dw - bwid) / 2
        fade = ease((t - alert_since(t)) / 0.25)     # fade in once per continuous high-risk stretch
        by0 = oy + 16 - 14 * (1 - fade)
        rr(d, [bx0, by0, bx0 + bwid, by0 + (88 if sub else 56)], 10, fill=(176, 24, 24, int(238 * fade)))
        warn_icon(d, bx0 + 20, by0 + 13, 30, (255, 255, 255, int(255 * fade)), (176, 24, 24, 255))
        d.text((bx0 + 60, by0 + 12), msg, font=F["banner"], fill=(255, 255, 255, int(255 * fade)))
        if sub:
            d.text((bx0 + 22, by0 + 54), sub, font=F["banner2"], fill=(254, 226, 226, int(255 * fade)))
        placed.append([bx0 - 6, by0 - 4, bx0 + bwid + 6, by0 + 96])

    # ---- event pop-up (bottom-left of video) ----
    toast = None
    if spot is None:
        for ev in events:
            if 0 <= t - ev["t"] < 1.6:
                toast = ev
    if toast:
        age = t - toast["t"]
        k = ease(age / 0.25) * (1 - ease((age - 1.3) / 0.3))
        col = KIND_COL.get(toast.get("kind", "info"), ROLE["police"])
        title = f"{fmt(toast['t'])}  ·  {toast['title'].upper()}"
        wdt = max(tw(d, title, F["toast1"]), tw(d, toast["text"], F["toast2"])) + 44
        tx0 = ox + 20 - 40 * (1 - k); ty0 = oy + dh - 92
        rr(d, [tx0, ty0, tx0 + wdt, ty0 + 66], 9, fill=(10, 12, 16, int(225 * k)))
        d.rectangle([tx0, ty0 + 8, tx0 + 5, ty0 + 58], fill=col + (int(255 * k),))
        d.text((tx0 + 20, ty0 + 10), title, font=F["toast1"], fill=col + (int(255 * k),))
        d.text((tx0 + 20, ty0 + 36), toast["text"], font=F["toast2"], fill=TEXT + (int(255 * k),))
        placed.append([tx0 - 4, ty0 - 4, tx0 + wdt + 4, ty0 + 70])

    # ---- subtitles (coded speech) ----
    speech = (sg or {}).get("speech")
    if speech and spot is None:
        who, txt = speech.get("who", ""), f"“{speech['text']}”"
        sw = tw(d, txt, F["sub"]) + (tw(d, who, F["subwho"]) + 18 if who else 0) + 40
        sx0 = ox + (dw - sw) / 2; sy0 = oy + dh - 150
        rr(d, [sx0, sy0, sx0 + sw, sy0 + 52], 10, fill=(0, 0, 0, 200))
        xq = sx0 + 20
        if who:
            d.text((xq, sy0 + 18), who, font=F["subwho"], fill=(250, 204, 21, 255)); xq += tw(d, who, F["subwho"]) + 18
        d.text((xq, sy0 + 9), txt, font=F["sub"], fill=(255, 255, 255, 255))
        placed.append([sx0 - 4, sy0 - 4, sx0 + sw + 4, sy0 + 56])

    # ---- people ----
    ymin = oy + (96 if a.tag or spot else 66)
    def overlaps(rc):
        return any(not (rc[2] < q[0] or rc[0] > q[2] or rc[3] < q[1] or rc[1] > q[3]) for q in placed)
    for key, bx in boxes.items():
        if fr not in bx: continue
        unmapped = key.startswith("_")
        if unmapped and not a.show_unmapped: continue
        x1, y1, x2, y2 = bx[fr] * scale + np.array([ox, oy, ox, oy])
        x1, y1 = max(x1, ox + 2), max(y1, oy + 2); x2, y2 = min(x2, ox + dw - 2), min(y2, oy + dh - 2)
        if x2 - x1 < 8 or y2 - y1 < 8: continue
        if unmapped:
            db.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 110), width=2); continue
        p = people[key]
        st = (sg or {}).get("people", {}).get(key, {})
        armed = is_armed(st)
        col = DANGER if armed else ROLE.get(p["role"], ROLE["bystander"])
        dim = 0.55 if spot else 1.0
        wd = 4 if armed or st.get("sudden") else 3
        is_manual = fr in manual.get(key, ())
        db.rectangle([x1, y1, x2, y2], fill=col + (int(22 * dim),))
        if is_manual:
            for xq in range(int(x1), int(x2), 16):
                db.line([(xq, y1), (min(xq + 8, x2), y1)], fill=col + (int(220 * dim),), width=2)
                db.line([(xq, y2), (min(xq + 8, x2), y2)], fill=col + (int(220 * dim),), width=2)
            for yq in range(int(y1), int(y2), 16):
                db.line([(x1, yq), (x1, min(yq + 8, y2))], fill=col + (int(220 * dim),), width=2)
                db.line([(x2, yq), (x2, min(yq + 8, y2))], fill=col + (int(220 * dim),), width=2)
        else:
            L = max(16, min(x2 - x1, y2 - y1) * 0.22)
            for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                db.line([(cx, cy), (cx + sx * L, cy)], fill=col + (int(255 * dim),), width=wd)
                db.line([(cx, cy), (cx, cy + sy * L)], fill=col + (int(255 * dim),), width=wd)
            db.rectangle([x1, y1, x2, y2], outline=col + (int(80 * dim),), width=1)
        if spot: continue
        line1 = f"{key} · {p.get('label', p['role']).upper()}" + ("  · MANUAL BOX" if is_manual else "")
        bits = [st.get("activity", "")]
        if armed: bits.append(st["weapon"].upper())
        if st.get("gesture"): bits.append(st["gesture"])
        line2 = " · ".join(b for b in bits if b)
        cw = max(tw(d, line1, F["chip"]), tw(d, line2, F["chip2"]) if line2 else 0) + 22
        ch = 58 if line2 else 34
        cx0 = min(max(x1, ox + 8), ox + dw - cw - 8)
        cands = [y1 - ch - 6, y1 + 6, y2 + 6] + [y1 + 6 + k * (ch + 4) for k in range(1, 8)]
        cy0 = None
        for c in cands:
            if c < ymin or c + ch > oy + dh - 6: continue
            if not overlaps([cx0, c, cx0 + cw, c + ch]): cy0 = c; break
        if cy0 is None: cy0 = max(ymin, min(y1 + 6, oy + dh - ch - 6))
        placed.append([cx0, cy0, cx0 + cw, cy0 + ch])
        rr(dc, [cx0, cy0, cx0 + cw, cy0 + ch], 7, fill=(10, 12, 16, 222), outline=col + (255,), width=2)
        dc.text((cx0 + 11, cy0 + 6), line1, font=F["chip"], fill=col + (255,))
        if line2:
            dc.text((cx0 + 11, cy0 + 32), line2, font=F["chip2"], fill=TEXT + (255,))

    # ---- body-cam HUD ----
    rr(d, [ox + 16, oy + 16, ox + 256, oy + 56], 7, fill=(0, 0, 0, 160))
    d.ellipse([ox + 30, oy + 30, ox + 42, oy + 42], fill=(239, 68, 68, int(110 + 145 * pulse)))
    d.text((ox + 52, oy + 22), f"REC  {int(t // 60):02d}:{t % 60:05.2f}", font=F["hud"], fill=(255, 255, 255, 255))
    tag = "KEY MOMENT · PAUSED" if spot else a.tag
    if tag:
        w2 = tw(d, tag, F["tag"]) + 20
        rr(d, [ox + 16, oy + 62, ox + 16 + w2, oy + 88], 6, fill=((250, 204, 21, 235) if spot else (255, 255, 255, 46)))
        d.text((ox + 26, oy + 66), tag, font=F["tag"], fill=((20, 20, 20, 255) if spot else (255, 255, 255, 235)))

    canvas.alpha_composite(lbox); canvas.alpha_composite(lchip)
    if spot is not None:
        canvas = draw_spotlight(canvas, frame_bgr, spot, sp)
    canvas.alpha_composite(ui)

    pan = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(pan)
    draw_panel(d, t, sg, risk, shown)
    draw_timeline(d, t)
    canvas.alpha_composite(pan)
    return canvas.convert("RGB")

def draw_spotlight(canvas, frame_bgr, spot, sp):
    k = ease(sp / 0.3)
    x1, y1, x2, y2 = np.array(spot["box"], float) * scale + np.array([ox, oy, ox, oy])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rad = max(x2 - x1, y2 - y1) * 0.7 + 16
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([ox, oy, ox + dw, oy + dh], fill=int(165 * k))
    md.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(10))
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 255)); dark.putalpha(mask)
    canvas.alpha_composite(dark)
    o = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(o)
    rr_ = rad * (1.3 - 0.3 * k)
    a_ = int(245 * k)
    d.ellipse([cx - rr_, cy - rr_, cx + rr_, cy + rr_], outline=DANGER + (a_,), width=4)
    d.ellipse([cx - rr_ - 9, cy - rr_ - 9, cx + rr_ + 9, cy + rr_ + 9], outline=DANGER + (int(90 * k),), width=2)
    # zoom inset of the weapon region
    sx1, sy1, sx2, sy2 = spot["box"]
    pad = max(sx2 - sx1, sy2 - sy1) * 0.2
    crop = frame_bgr[int(max(sy1 - pad, 0)):int(min(sy2 + pad, src_h)), int(max(sx1 - pad, 0)):int(min(sx2 + pad, src_w))]
    fit = min(320 / crop.shape[1], 240 / crop.shape[0])          # keep aspect ratio
    iw, ih = int(crop.shape[1] * fit), int(crop.shape[0] * fit)
    crop_r = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_CUBIC)
    zoom = fit / scale                                            # magnification vs. the video on screen
    cwid = 360
    title, text = spot["title"].upper(), spot["text"]
    lines = wrap(d, text, F["spot2"], cwid - 40)
    chgt = 60 + 26 * len(lines) + 14 + ih + 20
    right = cx < ox + dw / 2
    cx0 = (cx + rad + 70) if right else (cx - rad - 70 - cwid)
    cx0 = min(max(cx0, ox + 20), ox + dw - cwid - 20)
    cy0 = min(max(cy - chgt / 2, oy + 100), oy + dh - chgt - 20)
    ax = cx + (rr_ if right else -rr_); ay = cy
    bx = cx0 if right else cx0 + cwid; by = min(max(cy, cy0 + 24), cy0 + chgt - 24)
    d.line([(ax, ay), (bx, by)], fill=DANGER + (a_,), width=3)
    rr(d, [cx0, cy0, cx0 + cwid, cy0 + chgt], 12, fill=(12, 14, 18, a_), outline=DANGER + (a_,), width=2)
    warn_icon(d, cx0 + 20, cy0 + 20, 24, DANGER + (a_,), (12, 14, 18, 255))
    d.text((cx0 + 54, cy0 + 18), title, font=F["spot1"], fill=(255, 255, 255, a_))
    for i, ln in enumerate(lines):
        d.text((cx0 + 20, cy0 + 60 + 26 * i), ln, font=F["spot2"], fill=TEXT + (a_,))
    iy = int(cy0 + 60 + 26 * len(lines) + 14)
    canvas.alpha_composite(o)
    if k > 0.02:
        inset = Image.fromarray(cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)).convert("RGBA")
        inset.putalpha(int(255 * k))
        ix = int(cx0 + (cwid - iw) / 2)
        canvas.alpha_composite(inset, (ix, iy))
        o2 = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d2 = ImageDraw.Draw(o2)
        d2.rectangle([ix, iy, ix + iw, iy + ih], outline=DANGER + (a_,), width=2)
        zl = f"DETAIL · {zoom:.1f}× ZOOM" if zoom >= 1.4 else "DETAIL"
        rr(d2, [ix + 6, iy + ih - 28, ix + 22 + tw(d2, zl, F["tiny"]), iy + ih - 8], 4, fill=(0, 0, 0, int(190 * k)))
        d2.text((ix + 14, iy + ih - 26), zl, font=F["tiny"], fill=(255, 255, 255, a_))
        canvas.alpha_composite(o2)
    return canvas

def section(d, x, y, title):
    d.text((x, y), title, font=F["sec"], fill=MUTED + (255,))

def draw_panel(d, t, sg, risk, shown):
    d.rectangle([PX, 0, W, H], fill=PANEL + (255,))
    d.line([(PX, 0), (PX, H)], fill=LINE + (255,), width=1)
    x, xr = PX + 30, W - 30
    d.text((x, 24), "SMART BODY-CAM ANALYTICS", font=F["sec"], fill=TEXT + (255,))
    d.text((x, 44), f"{clip.get('id', '')}  ·  {clip.get('location', '')}  ·  {clip.get('type', '')}", font=F["meta"], fill=MUTED + (255,))
    d.line([(x, 74), (xr, 74)], fill=LINE + (255,))

    # risk dial
    section(d, x, 88, "CONFLICT ESCALATION RISK")
    cx, cy, R, th = PX + 240, 262, 118, 20
    for i in range(5):
        a0 = 180 + i * 36 + 1.5; a1 = 180 + (i + 1) * 36 - 1.5
        d.arc([cx - R, cy - R, cx + R, cy + R], a0, a1, fill=LINE + (255,), width=th)
        frac = max(0.0, min(1.0, shown - i))
        if frac > 0.01:
            d.arc([cx - R, cy - R, cx + R, cy + R], a0, a0 + (a1 - a0) * frac, fill=RISK_COL[i + 1] + (255,), width=th)
    ang = math.radians(180 + max(0.0, min(5.0, shown)) / 5 * 180)
    mx, my = cx + (R - th / 2) * math.cos(ang), cy + (R - th / 2) * math.sin(ang)
    d.ellipse([mx - 9, my - 9, mx + 9, my + 9], fill=(255, 255, 255, 255), outline=BG + (255,), width=3)
    rc = RISK_COL[risk]
    num = str(risk)
    nw = tw(d, num, F["dial"]); sw_ = tw(d, "/5", F["dial2"])
    nx = cx - (nw + 6 + sw_) / 2
    d.text((nx, cy - 96), num, font=F["dial"], fill=rc + (255,))
    d.text((nx + nw + 6, cy - 46), "/5", font=F["dial2"], fill=MUTED + (255,))
    lv = RISK_NAME[risk]
    d.text((cx - tw(d, lv, F["level"]) / 2, cy + 4), lv, font=F["level"], fill=rc + (255,))
    d.text((cx - R - 4, cy + 10), "0", font=F["tiny"], fill=FAINT + (255,))
    d.text((cx + R - 4, cy + 10), "5", font=F["tiny"], fill=FAINT + (255,))

    # why
    y = 312
    d.line([(x, y), (xr, y)], fill=LINE + (255,)); y += 14
    section(d, x, y, "WHY THIS SCORE"); y += 24
    trig = (sg or {}).get("triggers", []) or ["No escalation indicators observed"]
    for tr in trig[:3]:
        for j, ln in enumerate(wrap(d, tr, F["body"], xr - x - 22)[:2]):
            if j == 0: d.ellipse([x + 1, y + 7, x + 9, y + 15], fill=rc + (255,))
            d.text((x + 20, y), ln, font=F["body"], fill=TEXT + (255,)); y += 23
        y += 5

    # signals
    y = 448
    d.line([(x, y), (xr, y)], fill=LINE + (255,)); y += 14
    section(d, x, y, "ESCALATION SIGNALS  (0–5)"); y += 26
    for name, k in (("Gesture / body language", "gesture"), ("Facial expression", "facial"), ("Voice / speech", "voice")):
        v = channel(sg, k)
        d.text((x, y), name, font=F["small"], fill=TEXT + (255,))
        vs = ("face not visible" if k == "facial" else "no audio") if v is None else str(v)
        d.text((xr - tw(d, vs, F["small"]), y), vs, font=F["small"], fill=(TEXT if v is not None else FAINT) + (255,))
        rr(d, [x, y + 22, xr, y + 30], 4, fill=LINE + (255,))
        if v: rr(d, [x, y + 22, x + (xr - x) * v / 5, y + 30], 4, fill=RISK_COL[v] + (255,))
        y += 42

    # people
    y = 626
    d.line([(x, y), (xr, y)], fill=LINE + (255,)); y += 14
    section(d, x, y, "PEOPLE IN SCENE"); y += 26
    shown_any = False
    fr_now = int(round((t - OFFSET) * fps))
    for pid, p in people.items():
        if first_seen.get(pid, 1e9) > t: continue
        shown_any = True
        st = (sg or {}).get("people", {}).get(pid)
        visible = pid in boxes and fr_now in boxes[pid]
        col = DANGER if is_armed(st) else ROLE.get(p["role"], ROLE["bystander"])
        al = 255 if visible else 120
        rr(d, [x, y, x + 46, y + 24], 5, fill=col + (al,))
        d.text((x + 23 - tw(d, pid, F["sec"]) / 2, y + 4), pid, font=F["sec"], fill=(255, 255, 255, al))
        desc = p.get("label", p["role"])
        if st and st.get("activity"): desc += " · " + st["activity"]
        if not visible: desc += " · out of frame"
        d.text((x + 58, y + 3), wrap(d, desc, F["small"], xr - x - 60)[0], font=F["small"], fill=TEXT + (al,))
        y += 34
    if not shown_any:
        d.text((x, y + 2), "Camera officer only. No one else in view yet", font=F["small"], fill=FAINT + (255,))

    # event log
    y = 742
    d.line([(x, y), (xr, y)], fill=LINE + (255,)); y += 14
    section(d, x, y, "EVENT LOG"); y += 26
    past = [e for e in events if e["t"] <= t][::-1][:4]
    if not past:
        d.text((x, y + 2), "Monitoring…", font=F["small"], fill=FAINT + (255,))
    for i, ev in enumerate(past):
        col = KIND_COL.get(ev.get("kind", "info"), ROLE["police"])
        if i == 0 and (t - ev["t"]) < 1.2:
            rr(d, [x - 8, y - 4, xr + 8, y + 40], 6, fill=CARD + (255,))
        d.rectangle([x - 2, y, x + 1, y + 34], fill=col + (255,))
        d.text((x + 10, y), fmt(ev["t"]), font=F["log_t"], fill=col + (255,))
        d.text((x + 72, y), ev["title"], font=F["log_t"], fill=TEXT + (255,))
        d.text((x + 10, y + 18), wrap(d, ev["text"], F["log"], xr - x - 14)[0], font=F["log"], fill=MUTED + (255,))
        y += 46

    # provenance
    y = H - 76
    d.line([(x, y - 10), (xr, y - 10)], fill=LINE + (255,))
    d.ellipse([x, y + 4, x + 8, y + 12], fill=(56, 189, 248, 255))
    d.text((x + 16, y), "Automated: person detection + tracking (YOLOv8, ByteTrack)", font=F["tiny"], fill=MUTED + (255,))
    d.ellipse([x, y + 24, x + 8, y + 32], fill=(250, 204, 21, 255))
    d.text((x + 16, y + 20), "Analyst-coded: roles, actions, weapon, escalation scores,", font=F["tiny"], fill=MUTED + (255,))
    d.text((x + 16, y + 36), "speech (research coding sheet)", font=F["tiny"], fill=MUTED + (255,))

def draw_timeline(d, t):
    by = VH
    d.rectangle([0, by, VW, H], fill=BG + (255,))
    x0, x1 = 170, VW - 44
    def tx(tt): return x0 + (tt - start) / (end - start) * (x1 - x0)
    d.text((28, by + 18), "RISK TIMELINE", font=F["sec"], fill=MUTED + (255,))
    ly0 = by + 56
    last_end = -1e9
    for ev in events:
        ex = tx(ev["t"]); col = KIND_COL.get(ev.get("kind", "info"), ROLE["police"])
        done = ev["t"] <= t
        lab = ev.get("short", ev["title"])
        lw = tw(d, lab, F["mark"])
        lx = min(max(ex - lw / 2, x0 - 10), x1 - lw)
        if lx < last_end + 10:              # events close together: push label right, marker stays at true time
            lx = last_end + 10
            d.line([(ex, by + 34), (lx + 4, by + 30)], fill=(col if done else FAINT) + (160,), width=1)
        last_end = lx + lw
        d.text((lx, by + 16), lab, font=F["mark"], fill=(col if done else FAINT) + (255,))
        d.polygon([(ex - 6, by + 37), (ex + 6, by + 37), (ex, by + 45)], fill=(col if done else FAINT) + (255,))
        d.line([(ex, by + 47), (ex, ly0 + 4 * 34 - 12)], fill=(col if done else LINE) + (110,), width=1)
    lanes = [("Overall risk", None), ("Gesture / body", "gesture"), ("Facial", "facial"), ("Voice", "voice")]
    for i, (nm, k) in enumerate(lanes):
        yy = ly0 + i * 34
        d.text((28, yy + 3), nm, font=F["small"], fill=(TEXT if k is None else MUTED) + (255,))
        rr(d, [x0, yy, x1, yy + 22], 5, fill=PANEL + (255,))
        for sg in segs:
            v = sg.get("risk", 0) if k is None else channel(sg, k)
            sx0, sx1 = tx(max(sg["start"], start)) + 1, tx(min(sg["end"], end)) - 1
            if v is None:
                for hx in range(int(sx0), int(sx1) - 2, 9):
                    d.line([(hx, yy + 19), (min(hx + 8, sx1), yy + 3)], fill=LINE + (255,), width=1)
                continue
            if v:
                if sg["start"] <= t < sg["end"]:
                    d.rectangle([sx0, yy + 2, max(sx0, tx(t)), yy + 20], fill=RISK_COL[v] + (255,))
                    d.rectangle([max(sx0, tx(t)), yy + 2, sx1, yy + 20], fill=RISK_COL[v] + (70,))
                else:
                    d.rectangle([sx0, yy + 2, sx1, yy + 20], fill=RISK_COL[v] + (255 if sg["start"] <= t else 70,))
    for s_ in range(int(math.ceil(start)), int(end) + 1):
        lab = f"{s_ // 60:02d}:{s_ % 60:02d}"
        d.line([(tx(s_), H - 42), (tx(s_), H - 37)], fill=FAINT + (255,))
        d.text((tx(s_) - tw(d, lab, F["tiny"]) / 2, H - 34), lab, font=F["tiny"], fill=FAINT + (255,))
    ph = tx(t)
    d.line([(ph, by + 47), (ph, ly0 + 4 * 34 - 6)], fill=(255, 255, 255, 255), width=2)

# ---------------- main loop ----------------
tmp = tempfile.mkdtemp()
silent = os.path.join(tmp, "v.mp4")
vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
spot_frames = {int(round((s_["t"] - OFFSET) * fps)): s_ for s_ in spots}
n_out = 0
for fr in range(f0, f1):
    ok, frame = cap.read()
    if not ok: break
    t = fr / fps + OFFSET
    img = draw_frame(frame, fr, t)
    vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)); n_out += 1
    if fr in spot_frames:
        s_ = spot_frames[fr]
        for i in range(int(s_.get("hold", 1.5) * fps)):
            img = draw_frame(frame, fr, t, spot=s_, sp=i / fps)
            vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)); n_out += 1
vw.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
print(f"wrote {a.out}: {n_out} frames ({n_out / fps:.1f}s)")
