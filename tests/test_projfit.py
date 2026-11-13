"""End-to-end tests on synthetic photos with known answers. Run: python3 -m unittest -v"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from projfit import render, synth  # noqa: E402
from projfit.cli import main as cli  # noqa: E402
from projfit.fit import Fit, fit_photo, quad_aspect  # noqa: E402
from projfit.layout import Layout  # noqa: E402

SURFACE = np.array([[620, 140], [1300, 160], [1290, 960], [640, 940]], float)   # true answer, projector px
PHOTO = (2400, 1800)


def make_photo(tmp, layout, cam_quad, f35=None, seed=0, name="photo.jpg"):
    img, surf_photo = synth.scene(layout, layout.image(), SURFACE, cam_quad, PHOTO, seed=seed)
    path = os.path.join(tmp, name)
    synth.save_jpeg(path, img, f35)
    return path, surf_photo


class FitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="projfit_test_")
        self.lay = Layout(1920, 1080)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_auto_corners_angled_photo(self):
        cam = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
        path, _ = make_photo(self.tmp, self.lay, cam)
        fit, _ = fit_photo(path, self.lay)
        err = np.linalg.norm(fit.quad - SURFACE, axis=1)
        self.assertLess(err.max(), 3.0, "corner error %s" % err.round(1))
        self.assertGreater(fit.markers_found, 50)
        self.assertLess(fit.residual_median_px, 1.5)

    def test_manual_corners(self):
        cam = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
        path, surf_photo = make_photo(self.tmp, self.lay, cam, seed=1)
        shuffled = surf_photo[[2, 0, 3, 1]].ravel()          # order must not matter
        fit, _ = fit_photo(path, self.lay, corners_photo=shuffled)
        self.assertLess(np.linalg.norm(fit.quad - SURFACE, axis=1).max(), 2.0)

    def test_aspect_from_exif_through_a_real_camera(self):
        f35 = 26.0                                           # a typical phone main camera
        cam = synth.camera_quad_for(self.lay, PHOTO, f35, yaw_deg=28)
        path, _ = make_photo(self.tmp, self.lay, cam, f35=f35, seed=2)
        fit, _ = fit_photo(path, self.lay)
        self.assertEqual(fit.aspect_source, "photo-exif")
        self.assertAlmostEqual(fit.aspect, quad_aspect(SURFACE), delta=0.03)

    def test_measured_size_wins(self):
        cam = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
        path, _ = make_photo(self.tmp, self.lay, cam, f35=26, seed=3)
        fit, _ = fit_photo(path, self.lay, size=(36, 80))
        self.assertEqual(fit.aspect_source, "measured")
        self.assertAlmostEqual(fit.aspect, 0.45, places=3)

    def test_no_exif_warns(self):
        cam = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
        path, _ = make_photo(self.tmp, self.lay, cam, seed=4)
        fit, _ = fit_photo(path, self.lay)
        self.assertEqual(fit.aspect_source, "projector-px")
        self.assertTrue(any("--size" in w for w in fit.warnings))

    def test_layout_rotation_round_trip(self):
        lay = Layout(1080, 1920, rotate=90)
        f = lay.to_file(lay.image())
        self.assertEqual(f.shape[:2], (1080, 1920))          # file is landscape for a projector on its side
        self.assertTrue(np.array_equal(np.rot90(f, k=-1), lay.image()))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="projfit_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_full_workflow_and_approval_gate(self):
        card, out = os.path.join(self.tmp, "card"), os.path.join(self.tmp, "fit")
        cli(["markers", "--size", "1920", "1080", "-o", card])
        lay = Layout.load(os.path.join(card, "layout.json"))
        cam = np.array([[300, 330], [2150, 180], [2200, 1640], [260, 1450]], float)
        photo, _ = make_photo(self.tmp, lay, cam)
        cli(["fit", photo, "--layout", os.path.join(card, "layout.json"), "--size-real", "34", "40", "-o", out])
        fj = os.path.join(out, "fit.json")
        cli(["verify", fj])
        self.assertTrue(os.path.exists(os.path.join(out, "verify.png")))

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        clip = os.path.join(self.tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                        "-t", "1", "-pix_fmt", "yuv420p", clip], check=True)
        dst = os.path.join(self.tmp, "fitted.mp4")
        with self.assertRaises(SystemExit):                  # not approved yet -> refused
            cli(["warp", fj, clip, dst])
        self.assertFalse(os.path.exists(dst))
        cli(["approve", fj, "--by", "Test Person"])
        self.assertTrue(json.load(open(fj))["approved"])
        cli(["warp", fj, clip, dst])
        self.assertEqual(render.probe_size(dst), (1920, 1080))

        # outside the surface is black, inside is lit
        cap = cv2.VideoCapture(dst)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertLess(frame[50, 50].mean(), 10)
        self.assertGreater(frame[550, 960].mean(), 20)

    def test_rotated_output_orientation(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        lay = Layout(1080, 1920, rotate=90)
        fit = Fit(corners=[[100, 200], [980, 200], [980, 1700], [100, 1700]], aspect=0.5,
                  aspect_source="measured", layout=lay.__dict__, photo="-", markers_found=1, markers_total=1,
                  residual_median_px=0, residual_p90_px=0, approved=True)
        clip, dst = os.path.join(self.tmp, "c.mp4"), os.path.join(self.tmp, "d.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=30",
                        "-t", "1", "-pix_fmt", "yuv420p", clip], check=True)
        render.warp_video(clip, dst, fit, lay)
        self.assertEqual(render.probe_size(dst), (1920, 1080))   # landscape file for a sideways projector


if __name__ == "__main__":
    unittest.main()
