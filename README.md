# rcsa

Two things live here:

- **3D field pipeline** — turn AMIGA rig footage into an interactive 3D field of
  per-plant bounding boxes. `scripts/run_field3d.py`
- **VLM crop-staging baseline** — the earlier experiment comparing three
  vision-language models on crop/stage identification. `scripts/run_vlm.py`,
  results in `results/`

---

## 3D field pipeline

```bash
.venv-3d/bin/python scripts/run_field3d.py cabbage
open results/field/cabbage_cam3.html
```

The argument is a folder name under `original_footage/`: `cabbage`, `pepper` or
`broccoli`. Each holds three synchronized clips from the robot's camera rig.
Every stage caches, so a re-run only redoes what is missing; `--force` redoes
everything.

### What the footage actually is

Three cameras, but **not** a stereo rig. Cam1 and cam2 sit at ground level
pointing roughly opposite ways; cam3 looks straight down. A cabbage seen from the
side and the same cabbage seen from above share no matchable features, so the
usual multi-view approach — VGGT, DUSt3R, MASt3R fusing several views of one
instant — has nothing to work with.

Depth comes from **motion** instead. The robot drives, so each camera on its own
is a monocular sequence with real temporal baseline. Phase A reconstructs the
nadir camera alone, which is the one that sees the field as a field.

Other things measured from the clips, which the parameters are built around:

- Motion is **bursty stop-and-go**. Some one-second gaps are pixel-identical,
  others change the frame almost completely. Fixed-fps sampling is wrong in both
  directions, so keyframes are chosen on measured drift.
- The **rig is in every frame** and, being bolted to the camera, looks like static
  world geometry to a reconstruction solver. It has to be masked.
- Everything is **unitless**. Monocular reconstruction fixes shape, not size. Set
  `SCALE` in `scripts/lift_boxes.py` once you have one real measurement — the
  nadir camera's height above the soil is the easiest.

### Stages

| Stage | Script | Output |
|---|---|---|
| Keyframe selection | `keyframes.py` | `keyframes/*.jpg`, `keyframes.csv` |
| Rig mask | `rig_mask.py` | `rig_mask.png`, `rig_mask_overlay.jpg` |
| Reconstruction | `reconstruct.py` | `cloud.ply`, `frames.npz` |
| Detection | `detect.py` | `detections.json` |
| 3D boxes | `lift_boxes.py` | `plants.json` |
| Viewer | `viewer.py` | `results/field/<crop>_cam<n>.html` |

All intermediates live in `work/<crop>/cam<n>/` (git-ignored, as is
`original_footage/`).

### Parameters that were measured, not guessed

Keyframe spacing is the one that matters most. VGGT solves jointly across every
frame in a window, so it needs high mutual co-visibility; once consecutive frames
stop overlapping heavily the reconstruction turns from a soil plane with
separable cabbage heads into an undifferentiated curtain.

| drift between keyframes | ground-plane inliers | trajectory straightness | depth confidence |
|---|---|---|---|
| 2.5% of frame width | 46% | 9.7 | 3.1 |
| 5% | 31% | 5.2 | 1.1 |
| 10% | 32% | 4.6 | 1.0 |

Window size trades co-visibility against coverage — 16 frames beat both 8 and 24.
Keeping only the top fifth of points by confidence lifted plane inliers from 46%
to 55%.

A caution worth keeping: a **parked** camera produces a reconstruction that looks
far cleaner than a moving one, because with no baseline VGGT falls back on its
monocular depth prior. It is not triangulated and cannot be stitched. That is why
stationary stretches are dropped rather than sampled.

### Runtime

On an M4 / 16 GB, VGGT runs locally on MPS at roughly 26s per 16-frame window.
The default `--limit 360` keyframes takes about 15 minutes; `--limit 0` does the
whole clip and takes well over an hour. Benchmark your own machine with
`scripts/bench_reconstruct.py`.

### Known limits

- Reconstruction quality is moderate. 640×480, a low camera, and a self-similar
  canopy are all working against it — the same repeated-texture problem the NSF
  snap-bean work describes.
- Detection is open-vocabulary with no training, so per-frame recall is partial.
  Clustering across many keyframes covers most of the gap.
- Cam1/cam2 registration (Phase B) is not built yet. The plan is to exploit the
  rigid rig: all three trajectories are one curve up to a fixed transform, which
  can be solved without any visual overlap.

### Environment

`.venv-3d` (torch, open3d, ultralytics, plotly, vggt) is separate from `.venv`
(mlx-vlm) so the VLM baseline stays reproducible.

```bash
python3 -m venv .venv-3d
.venv-3d/bin/pip install torch torchvision open3d ultralytics plotly opencv-python scipy transformers
.venv-3d/bin/pip install git+https://github.com/facebookresearch/vggt.git
.venv-3d/bin/pip install "numpy>=2.0,<2.8"   # vggt pins numpy<2; the pin is conservative
```
