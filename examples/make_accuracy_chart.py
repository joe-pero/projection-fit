#!/usr/bin/env python3
"""Measure corner accuracy on five synthetic scenes, before and after edge refinement, and chart it.

    python3 examples/make_accuracy_chart.py      # writes docs/images/accuracy-{light,dark}.png

"Before" is the surface outline from the brightness map alone; "after" adds the edge
refinement that uses only pixels known to be white on the marker card. The right answer is
known exactly for every scene, so the error is real, not estimated.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from projfit import fit as F, synth  # noqa: E402
from projfit.layout import Layout  # noqa: E402

SURFACE = np.array([[620, 140], [1300, 160], [1290, 960], [640, 940]], float)
PHOTO = (2400, 1800)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", before="#eb6834", after="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", before="#d95926", after="#3987e5"),
}


def measure():
    lay = Layout(1920, 1080)
    scenes = [
        ("Angled, left", [[300, 330], [2150, 180], [2200, 1640], [260, 1450]]),
        ("Angled, right", [[150, 250], [2300, 420], [2250, 1500], [180, 1700]]),
        ("Phone camera,\n28° right", synth.camera_quad_for(lay, PHOTO, 26, 28).tolist()),
        ("Phone camera,\n35° left", synth.camera_quad_for(lay, PHOTO, 26, -35).tolist()),
        ("Phone camera,\nhead-on", synth.camera_quad_for(lay, PHOTO, 26, 0).tolist()),
    ]
    out = []
    for i, (name, cam) in enumerate(scenes):
        img, _ = synth.scene(lay, lay.image(), SURFACE, np.array(cam, float), PHOTO, seed=i)
        m = F.on_plane(F.detect(img, lay))
        mask, info = F.surface_mask(img, m, lay)
        rough, _ = F.quad_from_mask(mask)
        sharp = F.refine_quad(rough, info)
        err = lambda q: float(np.linalg.norm(np.array([F.to_projector(p, m) for p in q]) - SURFACE, axis=1).max())
        out.append((name, err(rough), err(sharp)))
    return out


def chart(rows, mode, t):
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(rows))
    b = [r[1] for r in rows]
    a = [r[2] for r in rows]
    ax.bar(x - 0.17, b, width=0.32, color=t["before"], label="Outline from brightness only")
    ax.bar(x + 0.17, a, width=0.32, color=t["after"], label="With edge refinement (shipped)")
    for xi, v in zip(x - 0.17, b):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", color=t["ink2"], fontsize=8.5)
    for xi, v in zip(x + 0.17, a):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", color=t["ink"], fontsize=8.5, fontweight="bold")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("worst corner error (projector px)", color=t["muted"], fontsize=9)
    ax.set_ylim(0, max(b) * 1.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=t["ink2"], ncol=2, handlelength=1)
    fig.text(0.012, 0.965, "Corner accuracy on scenes with a known answer", color=t["ink"], fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.012, 0.885, f"Worst corner error fell from about {max(b):.0f} pixels to {max(a):.1f} or less "
             "(on a 1920 × 1080 projector picture)", color=t["ink2"], fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.15)
    fig.savefig(os.path.join(ROOT, "docs", "images", f"accuracy-{mode}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    rows = measure()
    for name, b, a in rows:
        print(f"{name.replace(chr(10), ' '):<26} before {b:5.1f} px   after {a:4.1f} px")
    for mode, t in THEMES.items():
        chart(rows, mode, t)
    print("wrote docs/images/accuracy-{light,dark}.png")
