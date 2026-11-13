"""projfit: fit projected video onto a real surface from one photo.

    projfit markers  --size 1920 1080 [--rotate 90] -o card/      # 1. project this
    projfit fit      photo.jpg --layout card/layout.json -o fit/   # 2. photo -> corners
    projfit verify   fit/fit.json -o fit/                          # 3. project + look
    projfit approve  fit/fit.json --by "Your Name"                 # 4. a person signs off
    projfit warp     fit/fit.json clip.mp4 clip_fitted.mp4         # 5. fit your video
    projfit grid     --layout card/layout.json -o card/            # optional by-eye grid
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import cv2

from . import __version__
from .fit import NAMES, Fit, fit_photo
from .layout import Layout
from . import render


def _write(path, img, layout=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, layout.to_file(img) if layout else img)
    return path


def cmd_markers(a):
    lay = Layout(a.size[0], a.size[1], marker=a.marker, pitch=a.pitch, dictionary=a.dict, rotate=a.rotate)
    os.makedirs(a.out, exist_ok=True)
    lay.save(os.path.join(a.out, "layout.json"))
    view = lay.image()
    _write(os.path.join(a.out, "markers_view.png"), view)
    png = _write(os.path.join(a.out, "markers.png"), view, lay)
    print("%d markers (%d x %d) on a %d x %d picture, rotate %d" % (lay.cols * lay.rows, lay.cols, lay.rows,
                                                                   lay.width, lay.height, lay.rotate))
    print("wrote", os.path.join(a.out, "layout.json"), "and", png)
    if a.video:
        mp4 = os.path.join(a.out, "markers.mp4")
        render.still_video(png, mp4, a.seconds)
        print("wrote", mp4)
    print("next: project markers.png on the surface, take one photo from the audience's spot, then run "
          "`projfit fit PHOTO --layout %s`" % os.path.join(a.out, "layout.json"))


def cmd_fit(a):
    lay = Layout.load(a.layout)
    corners = None
    if a.corners:
        if len(a.corners) != 8:
            sys.exit("--corners needs 8 numbers: x1 y1 x2 y2 x3 y3 x4 y4 (photo pixels, any order)")
        corners = a.corners
    fit, dbg = fit_photo(a.photo, lay, corners_photo=corners, size=a.size_real, inset=a.inset,
                         bright_frac=a.bright)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "fit.json")
    fit.save(path)
    _write(os.path.join(a.out, "photo_read.jpg"), cv2.resize(dbg, None, fx=0.5, fy=0.5))
    print("markers read: %d of %d; plane fit median %.1f px, 90%% %.1f px"
          % (fit.markers_found, fit.markers_total, fit.residual_median_px, fit.residual_p90_px))
    for n, (x, y) in zip(NAMES, fit.corners):
        print("  %s -> projector (%.0f, %.0f)" % (n, x, y))
    print("aspect (width/height) %.3f from %s" % (fit.aspect, fit.aspect_source))
    for w in fit.warnings:
        print("WARNING:", w)
    print("wrote", path, "and", os.path.join(a.out, "photo_read.jpg"), "(check the red outline sits on the surface)")
    print("next: `projfit verify %s`" % path)


def cmd_verify(a):
    fit = Fit.load(a.fit)
    lay = Layout(**fit.layout)
    out = a.out or os.path.dirname(os.path.abspath(a.fit))
    img = render.verify(fit, lay)
    _write(os.path.join(out, "verify_view.png"), img)
    png = _write(os.path.join(out, "verify.png"), img, lay)
    print("wrote", png)
    if a.video:
        mp4 = os.path.join(out, "verify.mp4")
        render.still_video(png, mp4, a.seconds)
        print("wrote", mp4)
    print("project it and look from where the audience stands: circles round, squares square, white border on "
          "the edges. If it's right: `projfit approve %s --by NAME`" % a.fit)


def cmd_approve(a):
    fit = Fit.load(a.fit)
    if fit.warnings and not a.accept_warnings:
        print("This fit has warnings:")
        for w in fit.warnings:
            print("  -", w)
        sys.exit("re-run with --accept-warnings if you checked the verify picture on the surface and it looks right")
    fit.approved, fit.approved_by = True, a.by
    fit.approved_at = dt.datetime.now().isoformat(timespec="seconds")
    fit.save(a.fit)
    print("approved by %s at %s" % (fit.approved_by, fit.approved_at))


def cmd_warp(a):
    fit = Fit.load(a.fit)
    if not fit.approved and not a.force:
        sys.exit("this fit has not been approved. Project `projfit verify` output, check it, then "
                 "`projfit approve %s --by NAME` (or pass --force for a test render)" % a.fit)
    lay = Layout(**fit.layout)
    render.warp_video(a.src, a.dst, fit, lay, crf=a.crf)
    print("wrote", a.dst)


def cmd_grid(a):
    lay = Layout.load(a.layout)
    fit = Fit.load(a.fit) if a.fit else None
    os.makedirs(a.out, exist_ok=True)
    img = render.grid(lay, fit)
    _write(os.path.join(a.out, "grid_view.png"), img)
    png = _write(os.path.join(a.out, "grid.png"), img, lay)
    print("wrote", png)
    if a.video:
        mp4 = os.path.join(a.out, "grid.mp4")
        render.still_video(png, mp4, a.seconds)
        print("wrote", mp4)


def main(argv=None):
    p = argparse.ArgumentParser(prog="projfit", description="Fit projected video onto a real surface from one photo.")
    p.add_argument("--version", action="version", version="projfit " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("markers", help="make the marker card to project")
    s.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"),
                   help="picture size as the audience sees it (e.g. 1920 1080, or 1080 1920 for a projector on its side)")
    s.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                   help="degrees counter-clockwise to turn the picture into the file the player plays")
    s.add_argument("--marker", type=int, default=56, help="marker size, px (bigger = readable from further away)")
    s.add_argument("--pitch", type=int, default=72, help="marker spacing, px (must be > --marker)")
    s.add_argument("--dict", default="4x4_1000", choices=("4x4_1000", "5x5_1000", "6x6_1000"))
    s.add_argument("--video", action="store_true", help="also write a looping .mp4 for media players")
    s.add_argument("--seconds", type=int, default=60)
    s.add_argument("-o", "--out", default="card")
    s.set_defaults(func=cmd_markers)

    s = sub.add_parser("fit", help="read a photo of the projected marker card")
    s.add_argument("photo")
    s.add_argument("--layout", required=True, help="layout.json written by `projfit markers`")
    s.add_argument("--size-real", type=float, nargs=2, metavar=("W", "H"),
                   help="measured surface size in any unit (e.g. 36 80); best source for the aspect ratio")
    s.add_argument("--corners", type=float, nargs="+", metavar="XY",
                   help="surface corners in PHOTO pixels if auto-detection struggles: x1 y1 ... x4 y4")
    s.add_argument("--inset", type=float, default=0, help="pull each corner in by this many projector px")
    s.add_argument("--bright", type=float, default=0.33,
                   help="surface brightness cut as a fraction of its white level (auto mode)")
    s.add_argument("-o", "--out", default="fit")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("verify", help="make the check picture to project before approving")
    s.add_argument("fit")
    s.add_argument("-o", "--out")
    s.add_argument("--video", action="store_true")
    s.add_argument("--seconds", type=int, default=60)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("approve", help="record that a person checked the fit on the real surface")
    s.add_argument("fit")
    s.add_argument("--by", required=True, help="who checked it")
    s.add_argument("--accept-warnings", action="store_true")
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser("warp", help="fit a video onto the surface (needs an approved fit)")
    s.add_argument("fit")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--crf", type=int, default=19)
    s.add_argument("--force", action="store_true", help="render without approval (testing only)")
    s.set_defaults(func=cmd_warp)

    s = sub.add_parser("grid", help="labelled coordinate grid for checking by eye")
    s.add_argument("--layout", required=True)
    s.add_argument("--fit", help="draw this fit's outline on the grid")
    s.add_argument("--video", action="store_true")
    s.add_argument("--seconds", type=int, default=60)
    s.add_argument("-o", "--out", default="card")
    s.set_defaults(func=cmd_grid)

    a = p.parse_args(argv)
    try:
        a.func(a)
    except (RuntimeError, ValueError) as e:
        sys.exit("error: %s" % e)


if __name__ == "__main__":
    main()
