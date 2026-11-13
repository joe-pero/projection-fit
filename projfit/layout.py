"""The marker card: a grid of ArUco markers that tiles the projector picture.

Every marker id is one grid slot, so each marker corner found in a photo is an exact
(photo pixel <-> projector pixel) pair. All coordinates here are in VIEW space: the
picture as the audience sees it, upright. `rotate` describes how a view-space image is
turned into the file the player actually plays (projector mounted on its side, etc.).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

DICTS = {
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
}


@dataclass
class Layout:
    width: int                 # view-space picture size, px
    height: int
    marker: int = 56           # marker side, px
    pitch: int = 72            # marker + white quiet zone, px
    dictionary: str = "4x4_1000"
    rotate: int = 0            # degrees counter-clockwise, view -> file (0, 90, 180, 270)

    def __post_init__(self):
        if self.rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be 0, 90, 180 or 270")
        if self.pitch <= self.marker:
            raise ValueError("pitch must be larger than marker (needs a white quiet zone)")
        if self.dictionary not in DICTS:
            raise ValueError("dictionary must be one of %s" % ", ".join(DICTS))
        if self.cols * self.rows > 1000:
            raise ValueError("%d markers > 1000 ids; use a bigger --marker/--pitch" % (self.cols * self.rows))

    # ---- geometry -------------------------------------------------------------
    @property
    def cols(self) -> int:
        return self.width // self.pitch

    @property
    def rows(self) -> int:
        return self.height // self.pitch

    @property
    def origin(self) -> tuple[int, int]:
        x0 = (self.width - self.cols * self.pitch) // 2 + (self.pitch - self.marker) // 2
        y0 = (self.height - self.rows * self.pitch) // 2 + (self.pitch - self.marker) // 2
        return x0, y0

    def corners(self) -> dict[int, np.ndarray]:
        """marker id -> its 4 corners (TL, TR, BR, BL) in view px, the order ArUco reports."""
        x0, y0 = self.origin
        m = self.marker
        out = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = x0 + c * self.pitch, y0 + r * self.pitch
                out[r * self.cols + c] = np.array([[x, y], [x + m, y], [x + m, y + m], [x, y + m]], float)
        return out

    def aruco_dict(self):
        return cv2.aruco.getPredefinedDictionary(DICTS[self.dictionary])

    # ---- images ---------------------------------------------------------------
    def image(self) -> np.ndarray:
        """View-space marker card, BGR. White background, thin red border marks the picture edge."""
        d = self.aruco_dict()
        im = np.full((self.height, self.width), 255, np.uint8)
        for i, q in self.corners().items():
            x, y = q[0].astype(int)
            im[y:y + self.marker, x:x + self.marker] = cv2.aruco.generateImageMarker(d, i, self.marker)
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(bgr, (0, 0), (self.width - 1, self.height - 1), (0, 0, 255), 4)
        return bgr

    def to_file(self, view_img: np.ndarray) -> np.ndarray:
        """Rotate a view-space image into the orientation the player plays."""
        return np.ascontiguousarray(np.rot90(view_img, k=self.rotate // 90))

    # ---- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Layout":
        with open(path) as f:
            return cls(**json.load(f))
