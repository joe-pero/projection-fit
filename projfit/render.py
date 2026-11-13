"""Pictures for the person in the loop, and the final video warp.

* grid:    a labelled coordinate grid (the by-eye method; still handy as a sanity check)
* verify:  a test pattern pushed through exactly the same crop + warp + mask the video will get.
           Circles should look round, squares square, the white border should sit on the edges.
* warp:    ffmpeg pipeline that fits a video onto the surface, black everywhere else.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np

from .fit import NAMES, Fit
from .layout import Layout

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRANSPOSE = {0: "", 90: ",transpose=2", 180: ",transpose=2,transpose=2", 270: ",transpose=1"}


def crop_size(aspect: float, w: int, h: int) -> tuple[int, int]:
    """Largest centred crop of a w x h frame with the given width/height ratio (even sizes)."""
    if w / h > aspect:
        cw, ch = int(round(h * aspect / 2)) * 2, h
    else:
        cw, ch = w, int(round(w / aspect / 2)) * 2
    return max(2, cw), max(2, ch)


# --------------------------------------------------------------------------- grid
def grid(layout: Layout, fit: Fit | None = None, major: int = 100) -> np.ndarray:
    W, H = layout.width, layout.height
    im = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W, major // 2):
        cv2.line(im, (x, 0), (x, H), (90, 90, 90) if x % major else (0, 200, 255), 1 if x % major else 2)
    for y in range(0, H, major // 2):
        cv2.line(im, (0, y), (W, y), (90, 90, 90) if y % major else (0, 200, 255), 1 if y % major else 2)
    for x in range(0, W, major):
        for y in range(0, H, major):
            cv2.putText(im, "%d,%d" % (x, y), (x + 4, y + 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (W - 1, H - 1), (255, 255, 255), 6)
    _top_arrow(im)
    if fit is not None:
        cv2.polylines(im, [fit.quad.astype(np.int32)], True, (0, 255, 0), 4)
    return im


def _top_arrow(im):
    H, W = im.shape[:2]
    cx, y = W // 2, max(60, H // 12)
    s = max(20, min(W, H) // 30)
    cv2.fillConvexPoly(im, np.array([[cx, y - s], [cx - s, y + s // 2], [cx + s, y + s // 2]], np.int32), (0, 255, 0))
    cv2.putText(im, "TOP", (cx - s, y + 2 * s), FONT, s / 25, (0, 255, 0), max(2, s // 10), cv2.LINE_AA)


# --------------------------------------------------------------------------- verify
def check_pattern(w: int, h: int) -> np.ndarray:
    """Upright test content: checkerboard (squares), circles (roundness), tenth ticks on each edge."""
    im = np.zeros((h, w, 3), np.uint8)
    sq = max(20, min(w, h) // 9)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if (x // sq + y // sq) % 2:
                im[y:y + sq, x:x + sq] = (70, 55, 55)
    r = int(min(w, h) * 0.4)
    cv2.circle(im, (w // 2, h // 2), r, (255, 255, 255), max(3, r // 40), cv2.LINE_AA)
    cv2.circle(im, (w // 2, h // 2), r // 3, (255, 255, 255), max(3, r // 60), cv2.LINE_AA)
    t = max(2, min(w, h) // 120)
    for i in range(1, 10):
        x, y = w * i // 10, h * i // 10
        for p in ((x, 0), (x, h), (0, y), (w, y)):
            cv2.circle(im, p, 3 * t, (0, 220, 255), -1)
    cv2.rectangle(im, (0, 0), (w - 1, h - 1), (255, 255, 255), max(6, min(w, h) // 60))
    _top_arrow(im)
    return im


def warp_image(content: np.ndarray, fit: Fit, layout: Layout) -> np.ndarray:
    """Same maths as the video warp: centre-crop to the surface aspect, map the frame onto the corners."""
    h, w = content.shape[:2]
    cw, ch = crop_size(fit.aspect, w, h)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = content[y0:y0 + ch, x0:x0 + cw]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), fit.quad.astype(np.float32))
    return cv2.warpPerspective(crop, Hm, (layout.width, layout.height), flags=cv2.INTER_LANCZOS4)


def verify(fit: Fit, layout: Layout) -> np.ndarray:
    W, H = layout.width, layout.height
    out = warp_image(check_pattern(W, H), fit, layout)
    q = fit.quad
    P = lambda a: tuple(int(round(v)) for v in a)
    cv2.polylines(out, [q.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(out, P(q[0]), P(q[2]), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.line(out, P(q[1]), P(q[3]), (255, 255, 0), 1, cv2.LINE_AA)
    fs = max(0.5, min(W, H) / 1400)
    for i, side in enumerate(("top", "right", "bottom", "left")):
        a, b = q[i], q[(i + 1) % 4]
        ang = np.degrees(np.arctan2(*(b - a)[::-1])) % 180
        mid = (a + b) / 2
        mid = mid + (q.mean(0) - mid) * 0.12
        cv2.putText(out, "%s %.1f deg" % (side, ang), P(mid - [60 * fs, 0]), FONT, fs, (0, 255, 0), 2, cv2.LINE_AA)
    for n, p in zip(NAMES, q):
        c = np.clip(p, [10, 30], [W - 220 * fs, H - 10])
        cv2.putText(out, "%s %d,%d" % (n, p[0], p[1]), P(c), FONT, fs, (0, 255, 0), 2, cv2.LINE_AA)
    status = "APPROVED by %s" % fit.approved_by if fit.approved else "NOT APPROVED - check on the surface, then run: projfit approve"
    cv2.putText(out, status, (20, H - 20), FONT, fs, (0, 255, 0) if fit.approved else (0, 165, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- video
def still_video(png: str, out: str, seconds: int = 60) -> None:
    _need_ffmpeg()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-i", png,
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(seconds),
                    "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest", out], check=True)


def mask_png(fit: Fit, layout: Layout, path: str) -> None:
    a = np.zeros((layout.height, layout.width, 4), np.uint8)
    a[..., 3] = 255
    cv2.fillPoly(a, [np.round(fit.quad).astype(np.int32)], (0, 0, 0, 0), cv2.LINE_AA)
    cv2.imwrite(path, a)


def probe_size(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w or not h:
        raise RuntimeError("could not read the video size of %s" % path)
    return w, h


def warp_video(src: str, dst: str, fit: Fit, layout: Layout, crf: int = 19) -> list[str]:
    """Fit a video onto the surface. Input is upright content; output is in the player's file orientation."""
    _need_ffmpeg()
    w, h = probe_size(src)
    cw, ch = crop_size(fit.aspect, w, h)
    q = np.round(fit.quad).astype(int)
    tmp_mask = dst + ".mask.png"
    mask_png(fit, layout, tmp_mask)
    fc = ("[0:v]crop=%d:%d,scale=%d:%d:flags=lanczos,setsar=1,"
          "perspective=x0=%d:y0=%d:x1=%d:y1=%d:x2=%d:y2=%d:x3=%d:y3=%d:sense=destination[w];"
          "[w][1:v]overlay=0:0:format=auto:shortest=1%s,format=yuv420p[o]"
          % (cw, ch, layout.width, layout.height, q[0][0], q[0][1], q[1][0], q[1][1], q[3][0], q[3][1],
             q[2][0], q[2][1], TRANSPOSE[layout.rotate]))
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-loop", "1", "-i", tmp_mask, "-filter_complex", fc,
           "-map", "[o]", "-map", "0:a?", "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(tmp_mask):
            os.remove(tmp_mask)
    return cmd


def _need_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (needed for video output)")
