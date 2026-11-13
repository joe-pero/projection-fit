"""Synthetic test scenes: a projector throwing the marker card onto a light rectangle on a
dark wall, photographed at an angle. Known ground truth, no real photos needed.

Used by the tests and to make the README pictures.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .layout import Layout


def scene(layout: Layout, proj_img: np.ndarray, surface_proj, cam_quad, photo_size=(2400, 1800),
          surface_real=(1.0, 2.2), seed: int = 0, blur: float = 1.2, noise: float = 4.0):
    """Render one photo.

    surface_proj: true surface corners in projector px (TL TR BR BL). The answer the fit should find.
    cam_quad:     where the projector picture's 4 corners land in the photo (sets the camera angle).
    Returns (photo BGR, surface corners in the photo).
    """
    rng = np.random.default_rng(seed)
    W, H = photo_size
    src = np.float32([[0, 0], [layout.width, 0], [layout.width, layout.height], [0, layout.height]])
    Hpc = cv2.getPerspectiveTransform(src, np.float32(cam_quad))      # projector px -> photo px
    surf_photo = cv2.perspectiveTransform(np.float32(surface_proj).reshape(-1, 1, 2), Hpc).reshape(-1, 2)

    # albedo: dark textured wall, light surface with a thin dark frame
    wall = (rng.normal(70, 18, (H, W)).clip(20, 140)).astype(np.float32)
    wall = cv2.GaussianBlur(wall, (0, 0), 3)
    alb = np.dstack([wall * 0.9, wall * 0.75, wall * 0.65]) / 255.0
    frame = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(frame, surf_photo.astype(np.int32), 255)
    big = cv2.dilate(frame, np.ones((15, 15), np.uint8))
    alb[(big > 0) & (frame == 0)] = 0.08
    alb[frame > 0] = (0.92, 0.93, 0.95)

    light = cv2.warpPerspective(proj_img, Hpc, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    ambient = 0.06
    img = alb * (light * 0.94 + ambient)
    img = cv2.GaussianBlur(img, (0, 0), blur) if blur else img
    img = (img * 255 + rng.normal(0, noise, img.shape)).clip(0, 255).astype(np.uint8)
    return img, surf_photo


def save_jpeg(path: str, bgr: np.ndarray, focal35: float | None = None) -> None:
    """Save as JPEG, optionally with a 35 mm-equivalent focal length in EXIF (as a phone would)."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    if focal35:
        exif[0x8769] = {41989: int(round(focal35))}      # Exif sub-IFD: FocalLengthIn35mmFilm
    pil.save(path, quality=92, exif=exif)


def camera_quad_for(layout: Layout, photo_size, f35: float, yaw_deg: float, fill: float = 0.8):
    """Project the picture plane through a real pinhole camera (so EXIF-based aspect is testable).

    The picture is a plane of width 1 (height by aspect), the camera yawed by yaw_deg and placed so
    the picture spans about `fill` of the photo width (as a person framing the shot would).
    """
    W, H = photo_size
    f = f35 / 36.0 * max(W, H)
    dist = f / (fill * W) + 0.5 * abs(np.sin(np.radians(yaw_deg)))
    aspect = layout.width / layout.height
    pw, ph = 1.0, 1.0 / aspect
    pts3 = np.array([[-pw / 2, -ph / 2, 0], [pw / 2, -ph / 2, 0], [pw / 2, ph / 2, 0], [-pw / 2, ph / 2, 0]])
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    cam = (R @ pts3.T).T + [0, 0, dist]
    return np.array([[f * x / z + W / 2, f * y / z + H / 2] for x, y, z in cam])
