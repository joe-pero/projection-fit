"""Read one photo of the projected marker card and work out where the target surface is,
in projector (view) pixels.

Steps
  1. Detect markers -> exact photo <-> projector point pairs.
  2. RANSAC drops markers that landed off the target's plane (a wall beside the door, a step).
  3. Find the surface's 4 corners in the photo: automatically (the bright surface the markers
     sit on) or given by hand with --corners.
  4. Map each corner into projector pixels with a homography fitted to the markers NEAREST that
     corner. Local fits cope with a surface that isn't perfectly flat (cloth, old wood).
  5. Work out the surface's true width/height ratio, so video is cropped, never stretched:
     from a measured size if given, else from the photo's EXIF focal length, else from the
     projector-pixel shape.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from .layout import Layout

NAMES = ("TL", "TR", "BR", "BL")


# --------------------------------------------------------------------------- photo
def load_photo(path: str):
    """Photo as BGR (EXIF rotation applied) + 35 mm-equivalent focal length if the camera recorded it."""
    pil = Image.open(path)
    f35 = None
    try:
        f35 = pil.getexif().get_ifd(0x8769).get(41989)  # FocalLengthIn35mmFilm
    except Exception:
        pass
    im = cv2.cvtColor(np.array(ImageOps.exif_transpose(pil).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im, (float(f35) if f35 else None)


# --------------------------------------------------------------------------- markers
@dataclass
class Matches:
    photo: np.ndarray      # (N, 4, 2) marker corners in the photo
    proj: np.ndarray       # (N, 4, 2) the same corners in projector view px
    ids: np.ndarray        # (N,)

    @property
    def centres(self) -> np.ndarray:
        return self.photo.mean(1)

    def __len__(self):
        return len(self.ids)


def detect(im: np.ndarray, layout: Layout) -> Matches:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(layout.aruco_dict(), p)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    if ids is None:
        raise RuntimeError("no markers found in the photo (is the marker card on, and in focus?)")
    table = layout.corners()
    ph, pj, keep = [], [], []
    for q, i in zip(corners, ids.ravel()):
        if int(i) in table:
            ph.append(q[0]); pj.append(table[int(i)]); keep.append(int(i))
    if len(keep) < 4:
        raise RuntimeError("only %d usable markers; need at least 4" % len(keep))
    return Matches(np.array(ph, float), np.array(pj, float), np.array(keep))


def on_plane(m: Matches, tol: float = 8.0) -> Matches:
    """Keep markers consistent with one plane (drops markers on a side wall / step)."""
    _, inl = cv2.findHomography(m.centres.astype(np.float32), m.proj.mean(1).astype(np.float32), cv2.RANSAC, tol)
    k = inl.ravel() > 0
    return Matches(m.photo[k], m.proj[k], m.ids[k])


# --------------------------------------------------------------------------- surface
def marker_brightness(v: np.ndarray, m: Matches) -> np.ndarray:
    """Brightness of each marker's white quiet zone = how light the surface under it is.

    The projector lights every marker with the same white, so this tracks the surface colour:
    high on a light target, low on a dark wall around it.
    """
    vb = cv2.GaussianBlur(v, (0, 0), 2)
    out = []
    for q in m.photo:
        c0 = q.mean(0)
        pts = c0 + (q - c0) * 1.18          # just outside each corner, in the white gap
        xs = pts[:, 0].astype(int).clip(0, v.shape[1] - 1)
        ys = pts[:, 1].astype(int).clip(0, v.shape[0] - 1)
        out.append(np.median(vb[ys, xs]))
    return np.array(out, float)


def split_light_dark(b: np.ndarray, min_ratio: float = 1.6):
    """Otsu split of marker brightness. Returns a mask of 'on the light surface' markers.

    If the markers don't fall into two clearly different groups, they are all on the surface.
    """
    if len(b) < 8:
        return np.ones(len(b), bool)
    vals = np.sort(b)
    best, cut = -1.0, None
    for i in range(2, len(vals) - 1):
        lo, hi = vals[:i], vals[i:]
        between = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if between > best:
            best, cut = between, (vals[i - 1] + vals[i]) / 2
    light = b > cut
    if light.sum() < 4 or b[light].mean() < min_ratio * max(1.0, b[~light].mean()):
        return np.ones(len(b), bool)
    return light


def surface_mask(im: np.ndarray, m: Matches, layout: Layout, bright_frac: float = 0.33) -> np.ndarray:
    """The light-coloured surface under the projection, as a filled blob in the photo.

    The tool knows exactly which pixels of the marker card are white. Mapped into the photo, only
    those pixels say how light the surface is (the black marker squares carry no information), so:
      1. albedo = brightness at white-card pixels, filled across the black squares by normalised
         blurring (no notches where a marker straddles the surface edge);
      2. markers are split into 'light surface' and 'dark surround' by the brightness of their
         white gaps; the cut is the midpoint (or `bright_frac` of the white level if there is no
         dark surround in view);
      3. the blob holding the most light markers wins; holes are filled.
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    h, w = v.shape
    H, _ = cv2.findHomography(m.proj.reshape(-1, 2), m.photo.reshape(-1, 2), cv2.RANSAC, 3.0)
    card = (cv2.cvtColor(layout.image(), cv2.COLOR_BGR2GRAY) > 128).astype(np.float32)
    card[:6, :] = card[-6:, :] = 0
    card[:, :6] = card[:, -6:] = 0
    side = float(np.median(np.linalg.norm(m.photo[:, 1] - m.photo[:, 0], axis=1)))
    valid = cv2.warpPerspective(card, H, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.erode((valid > 0.95).astype(np.float32), np.ones((3, 3), np.uint8))   # stay off module edges
    covered = cv2.warpPerspective(np.ones_like(card), H, (w, h)) > 0.5
    s = max(1.5, side * 0.09)
    num = cv2.GaussianBlur(v * valid, (0, 0), s)
    den = cv2.GaussianBlur(valid, (0, 0), s)
    alb = np.where(den > 0.02, num / np.maximum(den, 1e-6), 0)
    alb[~covered] = 0

    b = marker_brightness(hsv[..., 2], m)
    light = split_light_dark(b)
    if light.all():
        thr = bright_frac * np.percentile(b, 90)
    else:
        thr = (np.median(b[light]) + np.median(b[~light])) / 2
    white = (alb > thr).astype(np.uint8) * 255
    k = max(3, int(side * 0.1))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab = cv2.connectedComponents(white)
    ctr = m.centres[light]
    ys = ctr[:, 1].astype(int).clip(0, h - 1)
    xs = ctr[:, 0].astype(int).clip(0, w - 1)
    votes = np.bincount(lab[ys, xs], minlength=n)
    votes[0] = 0
    blob = (lab == votes.argmax()).astype(np.uint8) * 255
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(blob)
    cv2.drawContours(out, cs, -1, 255, -1)
    return out, (v, valid, thr, side)


def refine_quad(qp: np.ndarray, info, samples: int = 160) -> np.ndarray:
    """Sharpen each side using only pixels known to be white on the card.

    Walk straight across the side at many points. Wherever a trusted (white-card) pixel sits just
    inside AND just outside the edge, the light-to-dark crossing between them is an exact edge
    point. Fit each side to those points; keep the first estimate where too few are found.
    """
    v, valid, thr, side = info
    h, w = v.shape
    R = 0.6 * side
    d = np.arange(-R, R, 0.5, dtype=np.float32)
    lines = []
    for i in range(4):
        a, b = qp[i], qp[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        u = ab / L
        nrm = np.array([u[1], -u[0]])                          # outward (clockwise TL->TR->BR->BL)
        ts = np.linspace(0.06, 0.94, samples)
        P = a[None, None] + (ts[:, None, None] * ab[None, None]) + d[None, :, None] * nrm[None, None]
        mx, my = P[..., 0].astype(np.float32), P[..., 1].astype(np.float32)
        vv = cv2.remap(v, mx, my, cv2.INTER_LINEAR, borderValue=0)
        ok = cv2.remap(valid, mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.95
        pts = []
        for k in range(samples):
            idx = np.flatnonzero(ok[k])
            if len(idx) < 4:
                continue
            bright = vv[k, idx] > thr
            for j in range(len(idx) - 1):                     # first trusted bright -> dark step going out
                if bright[j] and not bright[j + 1] and d[idx[j + 1]] - d[idx[j]] <= 2.0:
                    v0, v1 = vv[k, idx[j]], vv[k, idx[j + 1]]
                    f = (v0 - thr) / max(v0 - v1, 1e-6)
                    dd = d[idx[j]] + f * (d[idx[j + 1]] - d[idx[j]])
                    pts.append(a + ts[k] * ab + dd * nrm)
                    break
        if len(pts) >= 8:
            pts = np.array(pts, np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        else:
            lines.append((a, u))
    out = []
    for (p1, d1), (p2, d2) in zip(lines[-1:] + lines[:-1], lines):
        t_ = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        out.append(p1 + t_[0] * d1)
    return np.array(out)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Any 4 points -> TL, TR, BR, BL."""
    pts = np.asarray(pts, float)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])


def quad_from_mask(mask: np.ndarray):
    """4 corners (TL, TR, BR, BL) of the blob: a line fitted to each side's OUTER edge, intersected.

    Unread markers at the edge only notch the outline inward, so each side keeps the outermost
    outline point per slice before fitting.
    """
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise RuntimeError("no surface found; try --corners")
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    c32 = c.astype(np.float32)
    rough = cv2.approxPolyDP(c32, 0.02 * cv2.arcLength(c32, True), True).reshape(-1, 2)
    if len(rough) < 4:
        raise RuntimeError("surface outline is not 4-sided; try --corners")
    rc = list(order_quad(cv2.convexHull(rough.astype(np.float32)).reshape(-1, 2)))
    lines = []
    for a, b in zip(rc, rc[1:] + rc[:1]):
        ab = b - a
        L2 = ab @ ab
        t = ((c - a) @ ab) / L2
        dist = np.abs(ab[0] * (c[:, 1] - a[1]) - ab[1] * (c[:, 0] - a[0])) / np.sqrt(L2)
        sel = (t > 0.1) & (t < 0.9) & (dist < 0.08 * np.sqrt(L2))
        nrm = np.array([ab[1], -ab[0]]) / np.sqrt(L2)   # outward for a clockwise TL->TR->BR->BL walk
        off = (c - a) @ nrm
        pts = []
        for b0 in np.linspace(0.1, 0.9, 41)[:-1]:
            k = sel & (t >= b0) & (t < b0 + 0.02)
            if k.any():
                pts.append(c[k][off[k].argmax()])
        if len(pts) < 2:
            raise RuntimeError("could not trace a side of the surface; try --corners")
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for (p1, d1), (p2, d2) in zip(lines[-1:] + lines[:-1], lines):
        t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        out.append(p1 + t[0] * d1)
    return np.array(out), c


# --------------------------------------------------------------------------- mapping
def to_projector(pt: np.ndarray, m: Matches, k: int = 12) -> np.ndarray:
    """Photo point -> projector px, using the k markers nearest the point."""
    near = np.argsort(np.linalg.norm(m.centres - pt, axis=1))[:max(4, k)]
    H, _ = cv2.findHomography(m.photo[near].reshape(-1, 2), m.proj[near].reshape(-1, 2), cv2.RANSAC, 3.0)
    if H is None:
        H, _ = cv2.findHomography(m.photo.reshape(-1, 2), m.proj.reshape(-1, 2), cv2.RANSAC, 3.0)
    return cv2.perspectiveTransform(np.array([[pt]], np.float32), H)[0, 0].astype(float)


def residuals(m: Matches) -> np.ndarray:
    """How well one flat-plane model explains every marker corner, in projector px."""
    H, _ = cv2.findHomography(m.photo.reshape(-1, 2), m.proj.reshape(-1, 2), cv2.RANSAC, 3.0)
    got = cv2.perspectiveTransform(m.photo.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    return np.linalg.norm(got - m.proj.reshape(-1, 2), axis=1)


def aspect_from_photo(q: np.ndarray, w: int, h: int, f35: float | None) -> float | None:
    """True width/height of the real rectangle seen as quad q (TL TR BR BL).

    Single-view rectangle method (Zhang & He, "Whiteboard scanning and image enhancement"),
    pinhole camera at the image centre with focal length from the 35 mm-equivalent EXIF value.
    """
    if not f35:
        return None
    f = f35 / 36.0 * max(w, h)
    u0, v0 = w / 2, h / 2
    m1, m2, m3, m4 = [np.array([p[0], p[1], 1.0]) for p in (q[3], q[2], q[0], q[1])]
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
    Ai = np.linalg.inv(A)
    B = Ai.T @ Ai
    r = (n2 @ B @ n2) / (n3 @ B @ n3)
    return float(np.sqrt(r)) if r > 0 else None


def quad_aspect(q: np.ndarray) -> float:
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w / h)


# --------------------------------------------------------------------------- result
@dataclass
class Fit:
    corners: list                      # [[x, y] x4] TL, TR, BR, BL, projector view px
    aspect: float                      # true width / height of the surface
    aspect_source: str                 # "measured" | "photo-exif" | "projector-px"
    layout: dict
    photo: str
    markers_found: int
    markers_total: int
    residual_median_px: float
    residual_p90_px: float
    warnings: list = field(default_factory=list)
    approved: bool = False             # set only by `projfit approve`, after a person checks the verify image
    approved_by: str | None = None
    approved_at: str | None = None
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Fit":
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def quad(self) -> np.ndarray:
        return np.array(self.corners, float)


def sanity_check(qj: np.ndarray, layout: Layout) -> None:
    """Refuse results that can't be right, instead of handing back a confident wrong answer."""
    W, H = layout.width, layout.height
    area = abs(cv2.contourArea(qj.astype(np.float32)))
    if area < 0.01 * W * H:
        raise RuntimeError("the surface found is tiny (%.1f%% of the picture); check photo_read.jpg or use --corners"
                           % (100 * area / (W * H)))
    if np.any(np.abs(qj[:, 0] - W / 2) > 1.5 * W) or np.any(np.abs(qj[:, 1] - H / 2) > 1.5 * H):
        raise RuntimeError("a corner landed far outside the projector picture; the photo probably has too few "
                           "readable markers. Move closer or use --corners")
    if not cv2.isContourConvex(qj.astype(np.float32).reshape(-1, 1, 2)):
        raise RuntimeError("the corners found don't make a convex 4-sided shape; use --corners")


def fit_photo(photo_path: str, layout: Layout, corners_photo=None, size=None, inset: float = 0.0,
              bright_frac: float = 0.33):
    """Run the whole fit. Returns (Fit, debug overlay image)."""
    im, f35 = load_photo(photo_path)
    m = on_plane(detect(im, layout))
    contour = None
    if corners_photo is not None:
        qp = order_quad(np.array(corners_photo, float).reshape(4, 2))
    else:
        mask, info = surface_mask(im, m, layout, bright_frac)
        qp, contour = quad_from_mask(mask)
        qp = refine_quad(qp, info)
    qj = np.array([to_projector(p, m) for p in qp])
    sanity_check(qj, layout)

    warnings = []
    if size:
        aspect, src = float(size[0]) / float(size[1]), "measured"
    else:
        a = aspect_from_photo(qp, im.shape[1], im.shape[0], f35)
        if a and 0.05 < a < 20:
            aspect, src = a, "photo-exif"
        else:
            aspect, src = quad_aspect(qj), "projector-px"
            warnings.append("no measured size or usable EXIF focal length: aspect taken from the projector "
                            "picture, which is only right if the projector faces the surface squarely. "
                            "Pass --size W H for a better result.")

    if inset:
        cen = qj.mean(0)
        qj = qj + (cen - qj) / np.linalg.norm(cen - qj, axis=1)[:, None] * inset

    for n, j in zip(NAMES, qj):
        if not (0 <= j[0] <= layout.width and 0 <= j[1] <= layout.height):
            warnings.append("%s corner (%.0f, %.0f) is outside the projector picture: the projector does not "
                            "fully cover the surface there." % (n, j[0], j[1]))
    res = residuals(m)
    total = layout.cols * layout.rows
    if len(m) < 0.25 * total:
        warnings.append("only %d of %d markers read; move closer, refocus, or darken the scene." % (len(m), total))
    if np.percentile(res, 90) > 6:
        warnings.append("markers disagree by up to %.1f px: the surface may be curved or the photo blurred."
                        % np.percentile(res, 90))

    fit = Fit(
        corners=[[round(float(x), 1), round(float(y), 1)] for x, y in qj],
        aspect=round(aspect, 4), aspect_source=src, layout=asdict(layout), photo=photo_path,
        markers_found=int(len(m)), markers_total=int(total),
        residual_median_px=round(float(np.median(res)), 2), residual_p90_px=round(float(np.percentile(res, 90)), 2),
        warnings=warnings,
    )
    return fit, overlay(im, m, qp, contour)


def overlay(im, m: Matches, qp, contour=None):
    dbg = im.copy()
    s = max(1, int(round(max(im.shape[:2]) / 1500)))
    for q in m.photo:
        cv2.polylines(dbg, [q.astype(np.int32)], True, (255, 160, 0), 2 * s)
    if contour is not None:
        cv2.drawContours(dbg, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 255), 2 * s)
    cv2.polylines(dbg, [qp.astype(np.int32)], True, (0, 0, 255), 3 * s)
    for n, p in zip(NAMES, qp):
        cv2.circle(dbg, tuple(p.astype(int)), 9 * s, (0, 0, 255), -1)
        cv2.putText(dbg, n, tuple((p + 12 * s).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (0, 0, 255), 3 * s)
    return dbg
