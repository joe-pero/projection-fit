#!/usr/bin/env python3
"""Build the README pictures from a synthetic scene (no real photos involved).

    python3 examples/make_demo.py      # writes docs/images/*.jpg
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from projfit import render, synth  # noqa: E402
from projfit.fit import fit_photo, quad_aspect  # noqa: E402
from projfit.layout import Layout  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")
SURFACE = np.array([[620, 140], [1300, 160], [1290, 960], [640, 940]], float)   # the "door", projector px
CAM = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
PHOTO = (2400, 1800)


def save(name, img, width=1200):
    s = width / img.shape[1]
    cv2.imwrite(os.path.join(OUT, name), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def content(w, h):
    """Stand-in 'show video' frame: a moody gradient with a moon and a title."""
    y, x = np.mgrid[0:h, 0:w]
    im = np.dstack([40 + 60 * y / h, 10 + 20 * x / w, 60 + 40 * (1 - y / h)]).astype(np.uint8)
    cv2.circle(im, (w // 2, int(h * 0.35)), int(h * 0.17), (200, 235, 245), -1, cv2.LINE_AA)
    for i, word in enumerate(("YOUR", "VIDEO")):
        fs, th = h / 420, max(2, h // 160)
        (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        cv2.putText(im, word, ((w - tw) // 2, int(h * (0.68 + 0.14 * i))), cv2.FONT_HERSHEY_DUPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    lay = Layout(1920, 1080)
    save("1_marker_card.jpg", lay.image())

    photo, _ = synth.scene(lay, lay.image(), SURFACE, CAM, PHOTO)
    path = os.path.join(OUT, "_photo.jpg")
    synth.save_jpeg(path, photo)
    save("2_photo.jpg", photo)

    fit, dbg = fit_photo(path, lay, size=(quad_aspect(SURFACE), 1))
    os.remove(path)
    save("3_read.jpg", dbg)
    err = np.linalg.norm(fit.quad - SURFACE, axis=1)
    print("corner error vs ground truth (projector px):", err.round(2))

    ver = render.verify(fit, lay)
    save("4_verify.jpg", ver)
    lit, _ = synth.scene(lay, ver, SURFACE, CAM, PHOTO, seed=1)
    save("5_verify_on_surface.jpg", lit)

    fitted = render.warp_image(content(1920, 1080), fit, lay)
    shot, _ = synth.scene(lay, fitted, SURFACE, CAM, PHOTO, seed=2)
    save("6_result.jpg", shot)

    before, _ = synth.scene(lay, content(1920, 1080), SURFACE, CAM, PHOTO, seed=3)
    save("0_before.jpg", before)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
