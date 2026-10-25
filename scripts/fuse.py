#!/usr/bin/env python3
"""Put all three cameras into one coordinate frame.

Takes the rig transform from rig_solve.py and carries cam1 and cam2 into cam3's
frame, producing one cloud and one set of per-keyframe records that every later
stage can treat exactly as it treats a single camera today.

**Where this sits in the pipeline is the load-bearing decision.** flatten.py is a
non-rigid deformation -- sixty along-row bins, each with its own offset, roll and
lateral correction, interpolated per point. Two independent warps of the same
physical bed are not relatable by any transform, rigid or otherwise. So if each
camera were flattened separately the three could never be reconciled afterwards,
no matter how good the rig transform was. The merge therefore happens here, while
the data is still rigid, and flatten.py runs exactly once on the result:

    reconstruct (per camera) -> fuse.py -> flatten.py (once) -> lift_boxes.py

Scale is handled by rig_solve's recovered ratio: each camera's reconstruction has
its own arbitrary monocular scale, and the ratio is what makes them commensurable.

The baseline translation gets refined here rather than trusted from hand-eye. The
rig transform's rotation and scale are well determined, but t_X is not -- (I - R)
is near-singular when the rotations are about a degree, which is the case on this
footage, and the self-test measures ~16% error on the baseline against <2% on
scale. 16% of a half-metre rig is comparable to a cabbage, so instead of accepting
it we use the one surface all three cameras genuinely share despite having no
overlapping view: the soil. Fitting each camera's ground plane and making the
planes coincide pins the component of the baseline perpendicular to the bed, which
is the one that matters for height.

Writes work/<crop>/fused/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

VOXEL = 0.004        # match reconstruct.py so the fused cloud has one density
PLANE_DIST = 0.02    # match flatten.py's RANSAC band


def ground_plane(points: np.ndarray, cam_centres: np.ndarray):
    """Soil plane as (normal, origin), oriented so the normal points at the camera.

    Same orientation rule as flatten.py and for the same reason: the cloud
    centroid cannot decide which way is up on this data, because monocular depth
    error piles up behind the soil surface and outweighs the plants in front of
    it. A camera is never underneath the ground it is photographing.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane, inliers = pcd.segment_plane(distance_threshold=PLANE_DIST, ransac_n=3,
                                       num_iterations=3000)
    n = np.asarray(plane[:3], float)
    n /= np.linalg.norm(n)
    origin = points[inliers].mean(0)
    if np.median((cam_centres - origin) @ n) < 0:
        n = -n
    return n, origin


def align_normals(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation carrying unit vector `a` onto unit vector `b`."""
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate by pi about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s ** 2)


def refine_against_soil(t_X, R_X, scale, src_pts, src_cams, dst_pts, dst_cams,
                        correct_tilt: bool = True):
    """Correct the rig transform against the one surface all three cameras share.

    Hand-eye gives t_X weakly, and on this footage it gives the *rotation* less
    well than the conditioning numbers suggest: the recovered optical axes come
    out 15-17 degrees off the perpendicular the rig geometry requires, and the
    same error shows up as the cameras' two soil planes disagreeing by about 21
    degrees. That is far more than a plant is wide, so it is the reason a plant
    seen from above and the same plant seen from the side do not land on top of
    each other and never merge.

    The soil does not care that the cameras share no field of view: each one
    reconstructs a plane, and after the rig transform those planes must be the
    same plane. That constraint pins two of the three rotation degrees of freedom
    -- the two tilts -- and says nothing about yaw around the normal, which is
    left as hand-eye estimated it. Hand-eye is good at yaw; it is the tilts it
    gets wrong, because tilt is what a near-singular (I - R_A) fails to see.

    The tilt correction is applied about the destination soil origin rather than
    about the coordinate origin, so fixing the orientation does not also throw
    the cloud across the field.

    Returns (R_X, t_X, info).
    """
    n_dst, o_dst = ground_plane(dst_pts, dst_cams)
    n_src, o_src = ground_plane(src_pts, src_cams)

    # Carry the source plane through the rig transform and see where it lands.
    def carried(R):
        n = R @ n_src
        return -n if np.dot(n, n_dst) < 0 else n

    tilt_before = float(np.degrees(np.arccos(
        np.clip(np.dot(carried(R_X), n_dst), -1, 1))))

    if correct_tilt:
        R_corr = align_normals(carried(R_X), n_dst)
        R_X = R_corr @ R_X
        # Rotate about the soil origin so the correction is a pivot, not a shove.
        t_X = R_corr @ (t_X - o_dst) + o_dst

    tilt_after = float(np.degrees(np.arccos(
        np.clip(np.dot(carried(R_X), n_dst), -1, 1))))

    o_src_in_dst = scale * (R_X @ o_src) + t_X
    gap = float(np.dot(o_src_in_dst - o_dst, n_dst))
    return R_X, t_X - gap * n_dst, {"plane_tilt_deg": tilt_before,
                                    "plane_tilt_after_deg": tilt_after,
                                    "plane_gap": gap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop", nargs="?", default="cabbage")
    ap.add_argument("--no-refine", action="store_true",
                    help="trust hand-eye's baseline instead of the ground-plane fit")
    ap.add_argument("--no-tilt", action="store_true",
                    help="refine the baseline but leave hand-eye's rotation alone")
    args = ap.parse_args()

    import open3d as o3d

    crop_work = WORK / args.crop
    rig_path = crop_work / "rig.json"
    if not rig_path.exists():
        raise SystemExit(f"run scripts/rig_solve.py {args.crop} first "
                         f"(and it must pass its validation gate)")
    rig = json.loads(rig_path.read_text())
    if rig.get("status") == "fail":
        raise SystemExit("rig.json records a failed validation; refusing to fuse")
    ref = rig["ref_cam"]

    out_dir = crop_work / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference camera: already in the target frame, so it defines it.
    ref_dir = crop_work / f"cam{ref}"
    ref_cloud = o3d.io.read_point_cloud(str(ref_dir / "cloud.ply"))
    ref_pts = np.asarray(ref_cloud.points)
    ref_cols = np.asarray(ref_cloud.colors)

    sys.path.insert(0, str(REPO / "scripts"))
    from flatten import camera_centres
    ref_cams = camera_centres(ref_dir)

    xyz = [ref_pts]
    rgb = [ref_cols]
    src = [np.full(len(ref_pts), int(ref), dtype=np.int32)]
    report = {}

    shutil.copy(ref_dir / "frames.npz", out_dir / f"frames_cam{ref}.npz")

    for key, p in rig["pairs"].items():
        cam = key.split("-")[1]
        cam_dir = crop_work / f"cam{cam}"
        if not (cam_dir / "cloud.ply").exists():
            print(f"skipping cam{cam}: no cloud.ply")
            continue

        X = np.array(p["X"], float)
        R_X, t_X, scale = X[:3, :3], X[:3, 3], float(p["scale"])

        cloud = o3d.io.read_point_cloud(str(cam_dir / "cloud.ply"))
        pts = np.asarray(cloud.points)
        cams = camera_centres(cam_dir)

        if not args.no_refine:
            R_X, t_X, info = refine_against_soil(
                t_X, R_X, scale, pts, cams, ref_pts, ref_cams,
                correct_tilt=not args.no_tilt)
            report[key] = info
            print(f"cam{cam}: ground planes tilt {info['plane_tilt_deg']:.1f} deg "
                  f"apart -> {info['plane_tilt_after_deg']:.1f} deg, "
                  f"gap {info['plane_gap']:+.4f} -> corrected")
            if info["plane_tilt_after_deg"] > 25:
                print(f"  warning: the two soil planes disagree by "
                      f"{info['plane_tilt_deg']:.0f} deg after the rig rotation. "
                      f"Either the rotation is wrong or cam{cam} never saw enough "
                      f"soil to fit a plane. Treat the fused height with suspicion.")

        moved = scale * (pts @ R_X.T) + t_X
        xyz.append(moved)
        rgb.append(np.asarray(cloud.colors))
        src.append(np.full(len(moved), int(cam), dtype=np.int32))
        print(f"cam{cam}: {len(moved):,} points -> fused frame "
              f"(scale {scale:.3f}, |t_X| {np.linalg.norm(t_X):.3f})")

        # Per-keyframe records with T_window composed into the fused frame, so
        # lift_boxes.py needs only to read T_fused where it read T_window.
        d = dict(np.load(cam_dir / "frames.npz"))
        G = np.eye(4)
        G[:3, :3] = scale * R_X
        G[:3, 3] = t_X
        d["T_fused"] = np.einsum("ij,njk->nik", G, d["T_window"]).astype(np.float32)
        d["source_cam"] = np.full(len(d["T_window"]), int(cam), dtype=np.int32)
        np.savez_compressed(out_dir / f"frames_cam{cam}.npz", **d)

    # The reference camera's T_fused is just its T_window.
    dref = dict(np.load(out_dir / f"frames_cam{ref}.npz"))
    dref["T_fused"] = dref["T_window"].astype(np.float32)
    dref["source_cam"] = np.full(len(dref["T_window"]), int(ref), dtype=np.int32)
    np.savez_compressed(out_dir / f"frames_cam{ref}.npz", **dref)

    allxyz = np.concatenate(xyz)
    allrgb = np.clip(np.concatenate(rgb), 0, 1)
    allsrc = np.concatenate(src)
    print(f"\nfused cloud before downsample: {len(allxyz):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(allxyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(allrgb.astype(np.float64))
    pcd = pcd.voxel_down_sample(VOXEL)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), pcd)
    print(f"after voxel+outlier filter: {len(pcd.points):,} points")

    np.save(out_dir / "source_cam.npy", allsrc)
    (out_dir / "manifest.json").write_text(json.dumps({
        "crop": args.crop, "ref_cam": ref,
        "cameras": sorted({int(s) for s in np.unique(allsrc)}),
        "points_per_camera": {str(c): int((allsrc == c).sum())
                              for c in np.unique(allsrc)},
        "baseline_refinement": report,
        "rig_status": rig.get("status"),
        "note": "merged in the rigid pre-flatten frame; run flatten.py on this",
    }, indent=1))

    print(f"\n-> {(out_dir / 'cloud.ply').relative_to(REPO)}")
    print(f"next: scripts/flatten.py {args.crop} --fused")


if __name__ == "__main__":
    main()
