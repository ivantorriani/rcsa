#!/usr/bin/env python3
"""Find the robot's own hardware in frame so reconstruction can ignore it.

The rails, wheels and the sibling cameras are bolted to the same chassis as the
lens, so they never move relative to it. To a structure-from-motion solver that
makes them look like static world geometry while the field streams past -- a
contradiction it resolves by corrupting the camera track.

Two obvious detectors both fail on this footage, and it is worth recording why:

  brightness stability   Measured over 120 samples of cabbage cam1 and cam3,
                         0.0% and 0.1% of pixels held still. The aluminium is
                         specular, so its value swings with the sun even though
                         it never moves.
  optical flow           Flow near zero should mark the rig, but Farneback
                         returns noise on smooth metal rather than zero -- no
                         texture, no gradient, no usable flow. Measured: no
                         pixel read as still in more than 38% of frame pairs.

What works is the temporal *median* image. Across hundreds of keyframes the
field slides past and averages into featureless mush, while the rig -- identical
in every single frame -- survives perfectly sharp. So the rig is exactly the part
of the median image that still has structure in it.

Two signals, multiplied:

  sharpness   Laplacian energy of the median image. High where the median kept
              real structure, i.e. the hardware.
  stability   inverse temporal standard deviation. The rig changes least.

Each frame is photometrically normalised first, which is what rescues the
stability term from the specularity problem above.

The robot's own shadow is also fixed in frame, but it is a soft-edged gradient,
so it scores low on sharpness and is left alone. That is deliberate -- shadowed
plants are still real plants, and masking them would throw away live data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

# How many keyframes feed the median. More frames mean the field averages flatter
# and the rig stands out further; past a few hundred it stops paying.
SAMPLE_FRAMES = 250
# Keep the top slice of the score. Swept against the cabbage nadir clip: at the
# 95th percentile the rail and camera mount survive unmasked, at the 92nd the
# whole bottom-left assembly and the right wheel are covered and no plant is
# touched. Masking a little soil is free; leaving hardware in is not.
SCORE_PERCENTILE = 92.0
# A blob must be this fraction of the frame to count as hardware.
MIN_BLOB_FRACTION = 0.002
# Sanity rail: the rig intrudes from an edge, it never fills the view.
MAX_MASK_FRACTION = 0.45


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def rig_score(frames: list[Path], downscale: int) -> np.ndarray:
    """Per-pixel likelihood that a pixel belongs to the chassis, in [0, 1]."""
    idx = np.linspace(0, len(frames) - 1, min(SAMPLE_FRAMES, len(frames))).astype(int)

    stack = []
    for i in idx:
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if downscale > 1:
            g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale))
        stack.append(g.astype(np.float32))

    if len(stack) < 20:
        raise SystemExit("need at least 20 keyframes to separate rig from field")

    a = np.stack(stack)
    # Per-frame photometric normalisation: without this the specular glare on the
    # rails reads as change and the stability term collapses.
    z = (a - a.mean(axis=(1, 2), keepdims=True)) / (a.std(axis=(1, 2), keepdims=True) + 1e-6)

    median = np.median(z, 0)
    sharp = cv2.GaussianBlur(np.abs(cv2.Laplacian(median, cv2.CV_32F)), (0, 0), 3)
    stable = 1.0 / (cv2.GaussianBlur(z.std(0), (0, 0), 3) + 1e-3)

    return cv2.GaussianBlur(_norm(sharp) * _norm(stable), (0, 0), 4)


def clean(mask: np.ndarray) -> np.ndarray:
    """Close gaps, drop specks, keep only blobs anchored to the frame edge.

    The chassis always intrudes from outside the image, so a high-scoring blob
    floating in the middle of the frame is a coincidence in the soil, not metal.
    """
    m = (mask.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    h, w = m.shape
    keep = np.zeros(m.shape, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_edge = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if touches_edge and area > MIN_BLOB_FRACTION * h * w:
            keep |= labels == i
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cams", default="3")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--percentile", type=float, default=SCORE_PERCENTILE,
                    help="score cutoff; lower masks more")
    args = ap.parse_args()

    for cam in [c.strip() for c in args.cams.split(",") if c.strip()]:
        cam_dir = WORK / args.crop / f"cam{cam}"
        manifest = cam_dir / "keyframes.csv"
        if not manifest.exists():
            raise SystemExit(f"run scripts/keyframes.py {args.crop} --cams {cam} first")

        rows = list(csv.DictReader(manifest.open()))
        frames = [cam_dir / "keyframes" / r["keyframe"] for r in rows]

        score = rig_score(frames, args.downscale)
        mask = clean(score > np.percentile(score, args.percentile))

        frac = float(mask.mean())
        if frac > MAX_MASK_FRACTION:
            print(f"  cam{cam}: WARNING mask covers {100*frac:.0f}% of frame; "
                  f"review the overlay before trusting it")

        # Mid-clip frame for the overlay: the opening frames are often blown out
        # by auto-exposure settling, which makes the review image useless.
        full = cv2.imread(str(frames[len(frames) // 2]))
        out = cv2.resize(mask.astype(np.uint8) * 255,
                         (full.shape[1], full.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(cam_dir / "rig_mask.png"), out)

        # Overlay to eyeball: red is what reconstruction will throw away.
        tint = full.copy()
        tint[out > 0] = (0, 0, 255)
        cv2.imwrite(str(cam_dir / "rig_mask_overlay.jpg"),
                    cv2.addWeighted(full, 0.55, tint, 0.45, 0))

        print(f"  cam{cam}: masked {100*frac:.1f}% of frame "
              f"-> {(cam_dir / 'rig_mask.png').relative_to(REPO)}")


if __name__ == "__main__":
    main()
