"""Make a 1920x1080 title card PNG matching the demo style.
Usage: python title_card.py out.png "subtitle line"
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda w, s: ImageFont.truetype(os.path.join(HERE, "fonts", f"Inter-{w}.otf"), s)
out, sub = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""

im = Image.new("RGB", (1920, 1080), (14, 17, 23))
d = ImageDraw.Draw(im)
x = 160
d.rectangle([x, 300, x + 8, 640], fill=(239, 68, 68))
d.text((x + 48, 300), "SMART BODY-CAMERA ANALYTICS", font=f("SemiBold", 30), fill=(140, 150, 166))
d.text((x + 44, 350), "Conflict-Escalation Demo", font=f("Bold", 92), fill=(232, 236, 242))
d.text((x + 48, 480), sub, font=f("Medium", 34), fill=(200, 208, 220))
d.text((x + 48, 540), "Person tracking  ·  Action & weapon labels  ·  Live risk score  ·  Escalation warnings",
       font=f("Medium", 28), fill=(140, 150, 166))
d.text((x + 48, 596), "Prepared by Namratha Indla  ·  UMass Boston, College of Management", font=f("Medium", 26), fill=(140, 150, 166))

legend = [((59, 130, 246), "Police"), ((245, 158, 11), "Suspect (unarmed)"), ((239, 68, 68), "Weapon visible / high risk")]
lx = x + 48
for col, name in legend:
    d.rounded_rectangle([lx, 700, lx + 26, 726], 5, fill=col)
    d.text((lx + 38, 699), name, font=f("Medium", 24), fill=(232, 236, 242))
    lx += 38 + d.textlength(name, font=f("Medium", 24)) + 48

d.text((x + 48, 900), "Contains real police body-camera footage (public release). Research demonstration only.",
       font=f("Medium", 22), fill=(110, 120, 136))
im.save(out)
