# Changelog

## 1.0.0 (2026-11-12)

First stable release.

- `projfit markers`: ArUco marker card for any projector resolution, including projectors
  mounted on their side (`--rotate`).
- `projfit fit`: finds the surface and its four corners from one photo; maps each corner into
  projector pixels using the nearest markers; works out the surface's true shape from a measured
  size or the photo's focal length. Corners land within 1.5 projector pixels on every test scene.
- `projfit verify` and `projfit approve`: a check pattern to project, and a recorded human
  sign-off. `projfit warp` refuses to render a fit nobody has approved.
- `projfit warp`: crops (never stretches) and warps a video onto the surface with ffmpeg; black
  everywhere else.
- `projfit grid`: labelled coordinate grid for checking by eye.
- Refuses results that can't be right (tiny surface, corners far outside the picture, a
  non-convex outline) instead of guessing.
- Tests on synthetic scenes with known answers, run on every push (Python 3.10 and 3.12).
