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
.venv-3d/bin/python scripts/run_field3d.py cabbage           # nadir camera only
.venv-3d/bin/python scripts/run_field3d.py cabbage --fuse    # all three cameras
open results/field/cabbage_fused.html
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
| Top-down figure | `render_topdown.py` | `field_topdown.png` |

Phase B adds four stages in front of the last three, which then run once on the
merged result rather than per camera:

| Stage | Script | Output |
|---|---|---|
| Clip synchronisation | `clip_sync.py` | `clip_sync.json` |
| Rig transform | `rig_solve.py` | `rig.json` |
| Merge | `fuse.py` | `fused/cloud.ply` |
| Everything after | `flatten.py --fused`, `lift_boxes.py --fused`, `viewer.py --fused` | `results/field/<crop>_fused.html` |

`render_topdown.py` is a side tool rather than a pipeline stage. It collapses the
cloud to a plan view by averaging the video colours of every point that lands in
each output pixel — useful for checking at a glance whether the rows came out
straight and how many plants detection missed.

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

### Phase B: registering the ground cameras

The two ground cameras share no field of view with the nadir camera and no
amount of feature matching will invent one. What all three share is the robot:
they are bolted to one chassis, so the three trajectories are one curve in three
coordinate frames, and the constant transform between those frames falls out of
the motion alone. That is the hand-eye problem, `rig_solve.py`, with two
wrinkles this footage adds -- each camera's monocular scale is arbitrary so only
the ratio is recoverable, and the three clips were started by hand so the time
offset is a continuous unknown.

Two results are worth stating because they are not what you would guess.

**The cheaper reconstruction is the better one.** What conditions the rig solve
is how long a span of time a camera tracks without breaking, not how many points
it produces. cam1's whole-clip run broke track and kept a 780-keyframe segment
covering 52 s; its 360-keyframe run tracked 150 s unbroken. The second is the one
that solves. `--ground-limit` defaults accordingly.

**Hand-eye's weak unknowns are fixed geometrically, not statistically.** The
rotation's two tilt components and the baseline both ride on `(I - R_A)`, which
is near-singular at one degree of rotation. Rather than trust them, both are
re-derived from the one surface all three cameras genuinely share despite having
no common view: the soil. Making the cameras' ground planes coincide pins the
tilts and the baseline's normal component, and leaves hand-eye only the yaw and
the scale, which it does estimate well. Scale is then refined again by aligning
the trajectories directly, which is a four-unknown linear fit over hundreds of
samples instead of a near-singular three-by-three.

The checks that say it worked are physical ones the solver never saw. cam1 and
cam2's recovered optical axes come out 175 degrees apart -- the two ground
cameras do face opposite ways -- and each sits near perpendicular to the nadir
camera, at 107 and 75 degrees. Nothing in the solve was told the rig's layout.

What this buys, measured: plants seen by a ground camera have a median
height-to-footprint ratio of 0.77, against 0.38 for plants only the nadir camera
saw. An overhead view systematically under-measures how tall a plant is, which
is the whole reason for having cameras at ground level.

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
- Cross-camera plant correspondence is below the registration accuracy, and the
  numbers say so plainly. The rig transform places the ground cameras within
  about 0.11-0.14 of the nadir camera, which is roughly one plant wide -- but
  plants in this bed sit 0.09-0.12 apart, so a nearest-neighbour match between a
  cam1 plant and a cam3 plant is about as likely to pick the neighbour as the
  plant itself. The clustering threshold is therefore left where single-camera
  evidence puts it and plants are reported per detecting camera, with the dozen
  that do merge shown as their own legend group. Raising the threshold to make
  the merge counts look better would be exactly the confident nonsense
  `rig_solve.py` spends its validation gate trying to avoid. Closing this needs a
  better reconstruction, not a looser threshold.
- The ground cameras reach only as far as their clips track. cam3 covers the
  whole 303 s row; cam1 and cam2 contribute over the first 150 s and 136 s, so
  roughly the first 40% of the strip is three-camera and the rest is nadir only.

### Environment

`.venv-3d` (torch, open3d, ultralytics, plotly, vggt) is separate from `.venv`
(mlx-vlm) so the VLM baseline stays reproducible.

```bash
python3 -m venv .venv-3d
.venv-3d/bin/pip install torch torchvision open3d ultralytics plotly opencv-python scipy transformers
.venv-3d/bin/pip install git+https://github.com/facebookresearch/vggt.git
.venv-3d/bin/pip install "numpy>=2.0,<2.8"   # vggt pins numpy<2; the pin is conservative
```
