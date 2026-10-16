#!/usr/bin/env python3
"""Pick keyframes from a rig clip by measured overlap, not by a fixed frame rate.

The AMIGA drives in bursts. Measured on cabbage_3.avi: consecutive frames differ
by ~4-8/255, but some 1-second gaps jump past 45, and t=60 and t=61 are pixel
identical. A fixed `fps=N` sample therefore does the worst of both things -- it
burns frames on stationary stretches that carry no parallax at all, and it steps
straight over the fast traverses where the geometry actually changes.

So we walk the clip and keep a frame only when it has drifted far enough from the
last keyframe to add baseline, but not so far that feature matching will fail:

  too little motion -> no parallax, nothing to triangulate, and a stationary run
                       would otherwise flood the set with near duplicates
  too much motion   -> not enough overlap for the reconstructor to connect them

Displacement is measured with sparse optical flow on corner features, which
tracks real pixel motion. A raw frame difference cannot tell motion apart from a
cloud passing over the sun, and dense flow is far too slow to run over 4500
frames.

Blurred frames are dropped outright: the fast bursts smear this rolling-shutter
webcam footage badly, and a blurred keyframe poisons the window it lands in.

Writes keyframes/*.jpg plus keyframes.csv, which is the source of truth for every
later stage (the same manifest-as-contract convention extract_frames.py uses).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FOOTAGE = REPO / "original_footage"
WORK = REPO / "work"

# Target drift between consecutive keyframes, as a fraction of image width.
#
# Measured, not guessed. Sweeping this against VGGT on a genuinely moving stretch
# of cabbage_3.avi, median depth confidence collapses from 3.1 to 1.1 somewhere
# between 2.5% and 5% of frame width, and the reconstruction turns from a soil
# plane with separable cabbage heads into an undifferentiated curtain:
#
#   drift  2.5% of width -> plane 46%, straightness 9.7, confidence 3.1
#   drift  5.0%          -> plane 31%, straightness 5.2, confidence 1.1
#   drift 10.0%          -> plane 32%, straightness 4.6, confidence 1.0
#
# VGGT solves jointly across every frame in a window, so it wants high mutual
# co-visibility; once consecutive frames stop overlapping heavily it degrades
# fast. 2.5% is the knee.
TARGET_DRIFT = 0.025
# Below this the rig is effectively parked. Worth being strict: frames taken
# with no baseline do not reconstruct, they just return VGGT's monocular depth
# prior, which looks deceptively clean and cannot be triangulated or stitched.
STATIONARY_DRIFT = 0.004
# Hard ceiling. Past this the tracker has lost the plot (or the robot lurched);
# take the frame anyway so we never silently drop a whole segment of the row.
MAX_DRIFT = 0.08

# Variance of Laplacian. Low means smeared. Tuned against the fast bursts in
# this footage rather than taken from a textbook -- see --blur-report.
BLUR_FLOOR = 45.0

TRACK_PARAMS = dict(maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian; higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def drift(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Median feature displacement between two frames, in pixels.

    Median rather than mean so a few plants waving in the wind, or a tracker
    that latches onto the rig, cannot drag the estimate around.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, **TRACK_PARAMS)
    if pts is None or len(pts) < 12:
        # Featureless frame (sun flare, soil close-up). Report a large drift so
        # the caller keeps it rather than stalling on a frame it cannot measure.
        return float("inf")

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
    if nxt is None:
        return float("inf")

    ok = status.ravel() == 1
    if ok.sum() < 12:
        return float("inf")

    delta = (nxt[ok] - pts[ok]).reshape(-1, 2)
    return float(np.median(np.linalg.norm(delta, axis=1)))


def select(video: Path, out_dir: Path, downscale: int = 2) -> list[dict]:
    """Walk the clip once, keeping frames that have drifted far enough."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Thresholds are declared as a fraction of width; convert once. The tracker
    # runs on a downscaled copy for speed, so scale the pixel budget to match.
    target = TARGET_DRIFT * width / downscale
    stationary = STATIONARY_DRIFT * width / downscale
    ceiling = MAX_DRIFT * width / downscale

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    kept: list[dict] = []
    prev_small = None      # last keyframe, downscaled, for drift measurement
    accumulated = 0.0      # drift since the last keyframe
    last_small = None      # previous frame, to accumulate step by step
    idx = -1
    skipped_blur = 0
    skipped_still = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        small = cv2.cvtColor(
            cv2.resize(frame, (width // downscale, frame.shape[0] // downscale)),
            cv2.COLOR_BGR2GRAY,
        )

        if prev_small is None:
            # Seed on the first sharp frame.
            sharp = blur_score(small)
            if sharp < BLUR_FLOOR:
                skipped_blur += 1
                continue
            prev_small = last_small = small
            kept.append(_record(frame, idx, idx / fps, sharp, 0.0, out_dir, len(kept)))
            continue

        # Accumulate drift frame to frame. Summing small steps survives the
        # stop-and-go far better than comparing directly against the last
        # keyframe, which loses tracking once the view has moved on.
        step = drift(last_small, small)
        last_small = small
        if np.isfinite(step):
            accumulated += step
        else:
            accumulated = ceiling  # force a keep; we cannot measure this one

        if accumulated < stationary:
            skipped_still += 1
            continue
        if accumulated < target:
            continue

        sharp = blur_score(small)
        if sharp < BLUR_FLOOR and accumulated < ceiling:
            # Blurred but not yet desperate -- wait for a sharper frame.
            skipped_blur += 1
            continue

        kept.append(_record(frame, idx, idx / fps, sharp, accumulated, out_dir, len(kept)))
        prev_small = small
        accumulated = 0.0

    cap.release()
    print(f"  {video.name}: {len(kept)} keyframes from {total} frames "
          f"(skipped {skipped_still} stationary, {skipped_blur} blurred)")
    return kept


def _record(frame, idx: int, ts: float, sharp: float, drift_px: float,
            out_dir: Path, n: int) -> dict:
    """Write the full-resolution keyframe and return its manifest row."""
    name = f"kf_{n:04d}.jpg"
    cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {
        "keyframe": name,
        "frame_idx": idx,
        "timestamp_s": round(ts, 3),
        "blur": round(sharp, 1),
        "drift_px": round(drift_px, 2) if np.isfinite(drift_px) else "inf",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop", help="folder under original_footage/ (e.g. cabbage)")
    ap.add_argument("--cams", default="3",
                    help="comma-separated camera numbers; default 3 (nadir) "
                         "since that is the Phase A field frame")
    args = ap.parse_args()

    crop_dir = FOOTAGE / args.crop
    if not crop_dir.is_dir():
        have = sorted(p.name for p in FOOTAGE.iterdir() if p.is_dir())
        sys.exit(f"no footage folder '{args.crop}'. Available: {', '.join(have)}")

    for cam in [c.strip() for c in args.cams.split(",") if c.strip()]:
        video = crop_dir / f"{args.crop}_{cam}.avi"
        if not video.exists():
            sys.exit(f"missing {video}")

        out_dir = WORK / args.crop / f"cam{cam}" / "keyframes"
        rows = select(video, out_dir)

        manifest = out_dir.parent / "keyframes.csv"
        with manifest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {manifest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
