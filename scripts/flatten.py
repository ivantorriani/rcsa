#!/usr/bin/env python3
"""Take the bend out of the reconstruction and give the axes real meaning.

Reconstruction chains ~35 windows, each aligned to the previous one. Every
alignment carries a fraction of a degree of rotation error, and composing them
makes the strip curve -- the "banana" every drift-free-claiming SLAM paper is
quietly fighting. Measured on the cabbage run, the local ground plane rotates
from 5.6 deg to 26.3 deg relative to the global fit, and the soil climbs from
0.00 to +0.29 between one end of the row and the other. Nothing closes the loop,
because the robot drives one pass and never revisits a plant.

Three symptoms, one cause: the arc; the two crop rows smearing into one band;
and boxes stretching into the air at the far end, where a plant sitting on
drift-raised ground got its floor pinned to a global zero that no longer matched
the soil beneath it.

We cannot re-solve the poses without re-running VGGT, but we do know something
the solver did not: **the bed is flat and straight**. That prior is enough.
Walking along the row in bins, we fit the local soil plane -- offset and sideways
tilt both -- and the lateral centre of the bed, smooth those profiles, and
subtract them. The bend comes out, the soil lands on z = 0 everywhere, and the
rows stop wandering across the frame.

The axes get fixed too. lift_boxes.py previously built its ground frame from an
arbitrary seed vector, so "across the row" pointed wherever the cross product
happened to land. Here the long axis is taken from the cloud's own dominant
direction, which is the direction the robot drove.

This is a deformation, not a rigid transform: it trades global metric fidelity
for a scene whose local geometry and axes are actually true. For counting plants
and reading field layout that is the better trade. It is recorded in
ground_model.json so every later stage applies exactly the same correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

BINS = 60            # along-row bins for the local fits
SOIL_PCT = 30.0      # within a bin, points below this height percentile are soil
SMOOTH = 7           # bins in the moving average that smooths the profiles
MIN_BIN_POINTS = 200


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average that tolerates NaN gaps from sparse bins."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan_to_num(x)
    # fill gaps by interpolation first so the window never straddles a hole
    idx = np.arange(len(x))
    x = np.interp(idx, idx[ok], x[ok])
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(x)]


def build_model(points: np.ndarray) -> dict:
    """Fit the provisional frame and the along-row correction profiles."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3,
                                       num_iterations=3000)
    n = np.asarray(plane[:3], float)
    n /= np.linalg.norm(n)
    origin = points[inliers].mean(0)
    if np.dot(points.mean(0) - origin, n) < 0:
        n = -n

    # Long axis from the data rather than an arbitrary seed: project onto the
    # plane and take the dominant direction, which is the way the robot drove.
    flat = points - np.outer((points - origin) @ n, n)
    centred = flat - flat.mean(0)
    u = np.linalg.svd(centred, full_matrices=False)[2][0]
    u -= np.dot(u, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    basis = np.stack([u, v, n], axis=1)
    g = (points - origin) @ basis

    along = g[:, 0]
    edges = np.linspace(along.min(), along.max(), BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    offset = np.full(BINS, np.nan)    # soil height at the middle of the bin
    roll = np.full(BINS, np.nan)      # d(height)/d(across): sideways tilt
    lateral = np.full(BINS, np.nan)   # centre of the bed

    which = np.clip(np.digitize(along, edges) - 1, 0, BINS - 1)
    for b in range(BINS):
        m = which == b
        if m.sum() < MIN_BIN_POINTS:
            continue
        gb = g[m]
        soil = gb[gb[:, 2] <= np.percentile(gb[:, 2], SOIL_PCT)]
        if len(soil) < 50:
            continue
        # height ~ a + b * across, fitted on soil only
        A = np.stack([np.ones(len(soil)), soil[:, 1]], axis=1)
        coef, *_ = np.linalg.lstsq(A, soil[:, 2], rcond=None)
        offset[b], roll[b] = coef[0], coef[1]
        lateral[b] = np.median(gb[:, 1])

    return {
        "origin": origin.tolist(),
        "basis": basis.tolist(),
        "bin_centres": centres.tolist(),
        "offset": _smooth(offset, SMOOTH).tolist(),
        "roll": _smooth(roll, SMOOTH).tolist(),
        "lateral": _smooth(lateral, SMOOTH).tolist(),
    }


def apply_model(points: np.ndarray, model: dict) -> np.ndarray:
    """World points -> straightened ground frame (along, across, height)."""
    origin = np.asarray(model["origin"])
    basis = np.asarray(model["basis"])
    g = (points - origin) @ basis

    c = np.asarray(model["bin_centres"])
    along = g[:, 0]
    lateral = np.interp(along, c, model["lateral"])
    offset = np.interp(along, c, model["offset"])
    roll = np.interp(along, c, model["roll"])

    across = g[:, 1] - lateral
    # Remove the local soil plane: its height at this point is the bin offset
    # plus the sideways tilt over the (re-centred) across distance.
    height = g[:, 2] - (offset + roll * g[:, 1])
    return np.stack([along, across, height], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    args = ap.parse_args()

    import open3d as o3d

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    cloud_path = cam_dir / "cloud.ply"
    if not cloud_path.exists():
        raise SystemExit(f"missing {cloud_path}; run scripts/reconstruct.py first")

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(cloud.points)

    model = build_model(pts)
    straight = apply_model(pts, model)

    # Report what the correction removed, as a check that it did anything.
    g_before = (pts - np.asarray(model["origin"])) @ np.asarray(model["basis"])
    edges = np.linspace(straight[:, 0].min(), straight[:, 0].max(), 9)
    print("soil height along the row (should flatten to ~0):")
    for i in range(8):
        m = (straight[:, 0] >= edges[i]) & (straight[:, 0] < edges[i + 1])
        if m.sum() < 1000:
            continue
        before = np.percentile(g_before[m, 2], 20)
        after = np.percentile(straight[m, 2], 20)
        print(f"  segment {i}: {before:+.3f} -> {after:+.3f}")

    flat = o3d.geometry.PointCloud()
    flat.points = o3d.utility.Vector3dVector(straight)
    flat.colors = cloud.colors
    o3d.io.write_point_cloud(str(cam_dir / "cloud_flat.ply"), flat)
    (cam_dir / "ground_model.json").write_text(json.dumps(model, indent=1))

    print(f"\nlong axis taken from the data (robot's travel direction)")
    print(f"-> {(cam_dir / 'cloud_flat.ply').relative_to(REPO)}")
    print(f"-> {(cam_dir / 'ground_model.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
