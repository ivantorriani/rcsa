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


def camera_centres(cam_dir: Path) -> np.ndarray | None:
    """Reconstructed camera positions in the same frame as cloud.ply.

    frames.npz stores extrinsic as world->cam in the window's own local frame and
    T_window as the similarity carrying that window into the global frame, so the
    centre is -R^T t pushed through T_window. The scale baked into T_window's
    rotation block is left in deliberately: cloud.ply carries it too, and these
    centres only ever get compared against that cloud.
    """
    npz = cam_dir / "frames.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    E = d["extrinsic"].astype(np.float64)
    T = d["T_window"].astype(np.float64)
    R, t = E[:, :3, :3], E[:, :3, 3]
    local = -np.einsum("nji,nj->ni", R, t)          # R^T t per keyframe
    return np.einsum("nij,nj->ni", T[:, :3, :3], local) + T[:, :3, 3]


def fused_camera_centres(fused_dir: Path) -> np.ndarray | None:
    """Camera centres of every camera in the fused cloud's frame.

    Each fused frames_cam<n>.npz carries T_fused, which is T_window already
    composed with the rig transform, so the same -R^T t construction lands
    directly in the shared frame.
    """
    out = []
    for npz in sorted(fused_dir.glob("frames_cam*.npz")):
        d = np.load(npz)
        if "T_fused" not in d.files:
            continue
        E = d["extrinsic"].astype(np.float64)
        T = d["T_fused"].astype(np.float64)
        R, t = E[:, :3, :3], E[:, :3, 3]
        local = -np.einsum("nji,nj->ni", R, t)
        out.append(np.einsum("nij,nj->ni", T[:, :3, :3], local) + T[:, :3, 3])
    return np.concatenate(out) if out else None


def build_model(points: np.ndarray, cam_centres: np.ndarray | None = None) -> dict:
    """Fit the provisional frame and the along-row correction profiles.

    `cam_centres` are the reconstructed camera positions. They are what decides
    which way is up, and the reason is worth stating because the obvious
    alternative is wrong.

    RANSAC returns a plane normal with an arbitrary sign, so something has to
    orient it. The natural-looking test -- point the normal from the soil plane
    toward the centroid of the whole cloud, on the theory that there is more
    scene above the soil than below it -- is false for this data, and not
    marginally so. Monocular depth error extends *away* from the camera, so the
    haze behind the soil surface outweighs the plants in front of it: measured on
    cabbage cam3 the cloud centroid sits +0.186 on the wrong side, 86% of the
    cloud's own thickness. The test fired confidently and fired backwards, which
    put the entire ground frame upside down -- the nadir camera ended up at
    height -0.815 in a frame whose z was supposed to mean "above soil".

    A camera, by contrast, cannot be under the ground it is looking at. The same
    measurement against the median camera centre gives -0.744, which is 3.4x the
    cloud thickness, and all 360 keyframes agree on the sign. That is not a
    heuristic, it is an observation.

    The fallback is kept for callers with no poses to hand, but it warns, because
    silently guessing which way is up is how this went wrong the first time.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3,
                                       num_iterations=3000)
    n = np.asarray(plane[:3], float)
    n /= np.linalg.norm(n)
    origin = points[inliers].mean(0)

    if cam_centres is not None and len(cam_centres):
        offsets = (np.asarray(cam_centres, float) - origin) @ n
        if np.median(offsets) < 0:
            n = -n
            offsets = -offsets
        agree = float((offsets > 0).mean())
        if agree < 0.95:
            print(f"  warning: only {agree:.0%} of camera centres agree on which "
                  f"side of the soil plane they are. The ground frame may be "
                  f"wrong; check cloud_flat.ply before trusting any height.")
    else:
        if np.dot(points.mean(0) - origin, n) < 0:
            n = -n
        print("  warning: no camera centres supplied, so 'up' is being guessed "
              "from the cloud centroid. That test is known to invert on nadir "
              "reconstructions -- pass cam_centres if you have frames.npz.")

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
    ap.add_argument("--fused", action="store_true",
                    help="operate on work/<crop>/fused/ -- the merged three-camera "
                         "cloud. Run this ONCE after fuse.py; flattening the "
                         "cameras separately makes them permanently unmergeable.")
    ap.add_argument("--fit-cam", default=None,
                    help="with --fused, build the soil model from this camera's "
                         "points only and apply it to all. The nadir camera is "
                         "usually the only one that reliably sees bare soil.")
    args = ap.parse_args()

    import open3d as o3d

    if args.fused:
        cam_dir = WORK / args.crop / "fused"
    else:
        cam_dir = WORK / args.crop / f"cam{args.cam}"
    cloud_path = cam_dir / "cloud.ply"
    if not cloud_path.exists():
        raise SystemExit(f"missing {cloud_path}; run scripts/reconstruct.py "
                         f"(or scripts/fuse.py) first")

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(cloud.points)

    cams = fused_camera_centres(cam_dir) if args.fused else camera_centres(cam_dir)

    # Which points define where the soil is. On the fused cloud this matters:
    # a ground-level camera immersed in canopy may never see bare soil at all
    # (broccoli cam1 is a wall of leaves), and feeding it to a percentile-based
    # soil fit poisons every bin it touches. The nadir camera is the one that
    # actually observes the ground, so let it define the ground.
    fit_pts = pts
    if args.fused and args.fit_cam:
        src_path = cam_dir / "source_cam.npy"
        if not src_path.exists():
            raise SystemExit("--fit-cam needs fused/source_cam.npy from fuse.py")
        # source_cam.npy indexes the pre-downsample cloud, so match by nearest
        # neighbour onto the cloud we actually have.
        src = np.load(src_path)
        if len(src) == len(pts):
            fit_pts = pts[src == int(args.fit_cam)]
        else:
            ref_cloud = o3d.io.read_point_cloud(
                str(WORK / args.crop / f"cam{args.fit_cam}" / "cloud.ply"))
            tree = o3d.geometry.KDTreeFlann(cloud)
            keep = np.zeros(len(pts), bool)
            for q in np.asarray(ref_cloud.points)[::7]:
                _, idx, _ = tree.search_knn_vector_3d(q, 1)
                keep[idx[0]] = True
            fit_pts = pts[keep]
        print(f"fitting the soil model on cam{args.fit_cam} only "
              f"({len(fit_pts):,} of {len(pts):,} points)")

    model = build_model(fit_pts, cams)
    straight = apply_model(pts, model)

    # The check that the frame is the right way up. A nadir camera is above the
    # soil it photographs and looks down at it; if either of these comes out
    # backwards, every height downstream is measuring the wrong side of the dirt.
    if cams is not None:
        g_cam = apply_model(cams, model)
        if np.median(g_cam[:, 2]) <= 0:
            raise SystemExit(
                f"ground frame is inverted: camera sits at height "
                f"{np.median(g_cam[:, 2]):+.3f} in a frame where positive means "
                f"above soil. Refusing to write it.")
        print(f"camera height above fitted soil: {np.median(g_cam[:, 2]):+.3f} "
              f"(positive, as it must be)")

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
