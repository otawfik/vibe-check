"""Generate screenshots/demo.svg (lightweight hand-built mock) and data/sample.txt."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vibe import VibeChecker

SAMPLES = [
    "lets gooooo we won, this is incredible, best night ever!!",
    "quiet morning, coffee and a good book, feeling calm and cozy",
    "i hate this broken app, worst update ever, so angry",
]

SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">
<rect width="900" height="420" rx="18" fill="#0f1220"/>
<text x="36" y="52" font-family="sans-serif" font-size="30" fill="#ffffff" font-weight="700">Vibe Check</text>
<text x="36" y="82" font-family="sans-serif" font-size="15" fill="#aab">sentiment + emotion + vibe meter for short social text</text>
{rows}
</svg>
"""

ROW = """<g>
<rect x="36" y="{y}" width="828" height="86" rx="12" fill="#1a1f36"/>
<text x="56" y="{y1}" font-family="sans-serif" font-size="15" fill="#eef">{text}</text>
<text x="56" y="{y2}" font-family="sans-serif" font-size="14" fill="#3ddc84">vibe: {vibe}  score: {score}  emotion: {emotion}  conf: {conf}</text>
</g>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")[:72]


def main():
    checker = VibeChecker.trained_default()
    rows = []
    for i, text in enumerate(SAMPLES):
        r = checker.check(text)
        y = 104 + i * 100
        rows.append(ROW.format(y=y, y1=y + 32, y2=y + 60, text=esc(text),
                                 vibe=r["vibe"], score=r["vibe_score"],
                                 emotion=r["top_emotion"], conf=r["confidence"]))
    out = os.path.join(os.path.dirname(__file__), "..", "screenshots", "demo.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(SVG_TEMPLATE.format(rows="\n".join(rows)))
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.txt")
    with open(sample_path, "w") as f:
        f.write("\n".join(SAMPLES) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
