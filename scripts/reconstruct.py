#!/usr/bin/env python3
"""Reconstruct one camera's run into a single point cloud and camera track.

VGGT can only hold a couple of dozen frames at once -- benchmarked on this M4,
24 frames costs 48s and 13.2 GB of the 16 GB available -- so a 400-keyframe clip
has to be processed in overlapping windows and stitched afterwards.

Each window comes back in its own coordinate frame, with its own arbitrary scale.
Consecutive windows deliberately share several keyframes, and those shared frames
are what lets us line the windows up: the same pixel of the same keyframe gets a
3D position from both windows, which is a dense set of exact correspondences.
A similarity transform (rotation, translation, uniform scale) fitted to those
correspondences carries one window into the previous one's frame, and composing
those transforms walks the whole clip into a single space.

Aligning on the shared *camera centres* instead would be the obvious shortcut and
it does not work here: the AMIGA drives in a near-straight line, so the centres
are close to collinear and leave rotation about the direction of travel
unconstrained. The dense point correspondences span the whole bed and do not have
that degeneracy.

Scale is recovered only up to a single global factor -- monocular reconstruction
cannot do better -- so distances are self-consistent but unitless until a real
measurement is supplied. See scale in the config.

Writes cloud.ply plus frames.npz, which holds the per-keyframe depth, confidence
and camera parameters that lift_boxes.py needs to turn 2D masks into 3D boxes.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

# Window size is a quality choice, not a memory one. 24 frames fits in 16 GB, but
# measured against a moving stretch, quality peaks lower -- fewer frames means
# more mutual co-visibility, which is what VGGT's joint solve wants:
#
#   8 frames  -> straightness 5.1
#   12        -> 8.4
#   16        -> 9.7   <- best
#   24        -> 6.3
WINDOW = 16
OVERLAP = 6          # shared frames per window boundary, for the alignment fit
# Keeping only the top fifth of points by confidence lifted ground-plane inliers
# from 46% to 55% on the same window. The discarded points are mostly the
# self-similar canopy interior, where depth is guesswork.
CONF_PERCENTILE = 80.0
POINTS_PER_WINDOW = 60_000
VOXEL = 0.004        # final downsample, in the reconstruction's own units


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src onto dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, sigma, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:          # guard against a reflection
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (s_c ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(sigma) @ W) / var) if var > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def robust_umeyama(src: np.ndarray, dst: np.ndarray, rounds: int = 3):
    """Umeyama with outlier rejection.

    A handful of correspondences always land on a badly-estimated depth; fitting
    straight through them tilts the whole window. Two refit passes against the
    residual distribution is enough to shake them off.
    """
    keep = np.ones(len(src), dtype=bool)
    scale, R, t = umeyama(src, dst)
    for _ in range(rounds):
        resid = np.linalg.norm((scale * (R @ src.T).T + t) - dst, axis=1)
        cutoff = np.median(resid) + 2.5 * (np.median(np.abs(resid - np.median(resid))) + 1e-9)
        new_keep = resid < cutoff
        if new_keep.sum() < 50 or new_keep.sum() == keep.sum():
            break
        keep = new_keep
        scale, R, t = umeyama(src[keep], dst[keep])
    resid = np.linalg.norm((scale * (R @ src.T).T + t) - dst, axis=1)
    return scale, R, t, float(np.median(resid)), int(keep.sum())


def windows_of(n: int, size: int, overlap: int) -> list[list[int]]:
    step = size - overlap
    out, start = [], 0
    while start < n:
        idx = list(range(start, min(start + size, n)))
        if len(idx) < 4 and out:          # tail too short to reconstruct; fold back
            out[-1] = sorted(set(out[-1] + idx))
            break
        out.append(idx)
        if start + size >= n:
            break
        start += step
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    # Dense keyframes are what make the geometry work, but they also make a full
    # clip a ~70 minute job. A bounded span still covers a long strip of row with
    # plenty of plants, so that is the default; 0 means the whole clip.
    ap.add_argument("--limit", type=int, default=360,
                    help="keyframes to reconstruct (0 = whole clip, much slower)")
    args = ap.parse_args()

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    manifest = cam_dir / "keyframes.csv"
    if not manifest.exists():
        raise SystemExit(f"run scripts/keyframes.py {args.crop} --cams {args.cam} first")

    rows = list(csv.DictReader(manifest.open()))
    available = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < available:
        span = float(rows[-1]["timestamp_s"]) - float(rows[0]["timestamp_s"])
        print(f"reconstructing {len(rows)} of {available} keyframes "
              f"({span:.0f}s of footage). Pass --limit 0 for the whole clip.")
    paths = [cam_dir / "keyframes" / r["keyframe"] for r in rows]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.bfloat16

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    print(f"loading VGGT on {device.type}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    wins = windows_of(len(paths), args.window, args.overlap)
    print(f"{len(paths)} keyframes -> {len(wins)} windows of {args.window} "
          f"(overlap {args.overlap})")

    # Rig mask, resized to VGGT's working resolution once we know it.
    mask_path = cam_dir / "rig_mask.png"
    rig = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if rig is None:
        print("  note: no rig_mask.png; reconstructing with chassis included")

    cloud_xyz: list[np.ndarray] = []
    cloud_rgb: list[np.ndarray] = []
    frames: dict[int, dict] = {}          # keyframe index -> pose/depth record
    prev_shared: dict[int, np.ndarray] = {}   # keyframe index -> world pts (prev window)
    T_global = np.eye(4)                  # window-local -> global
    s_global = 1.0

    t_start = time.perf_counter()
    for wi, idx in enumerate(wins):
        images = load_and_preprocess_images([str(paths[i]) for i in idx]).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            out = model(images)

        H, W = images.shape[-2:]
        extri, intri = pose_encoding_to_extri_intri(out["pose_enc"], (H, W))
        extri = extri[0].float().cpu().numpy()      # (S,3,4) world->cam
        intri = intri[0].float().cpu().numpy()      # (S,3,3)
        depth = out["depth"][0, ..., 0].float().cpu().numpy()        # (S,H,W)
        conf = out["depth_conf"][0].float().cpu().numpy()            # (S,H,W)
        rgb = out["images"][0].float().cpu().numpy().transpose(0, 2, 3, 1)  # (S,H,W,3)

        del out
        if device.type == "mps":
            torch.mps.empty_cache()

        pts = unproject_depth_map_to_point_map(depth[..., None], extri, intri)  # (S,H,W,3)

        if rig is not None:
            keep_px = cv2.resize(rig, (W, H), interpolation=cv2.INTER_NEAREST) == 0
        else:
            keep_px = np.ones((H, W), dtype=bool)

        # --- line this window up with the previous one -------------------------
        if wi > 0:
            shared = [i for i in idx if i in prev_shared]
            src, dst = [], []
            for i in shared:
                j = idx.index(i)
                good = (conf[j] > np.percentile(conf[j], 70)) & keep_px
                sel = np.argwhere(good)
                if len(sel) > 4000:                 # cap the fit; it is already dense
                    sel = sel[np.random.choice(len(sel), 4000, replace=False)]
                src.append(pts[j][sel[:, 0], sel[:, 1]])
                dst.append(prev_shared[i][sel[:, 0], sel[:, 1]])
            if src:
                src = np.concatenate(src)
                dst = np.concatenate(dst)
                ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
                scale, R, t, resid, used = robust_umeyama(src[ok], dst[ok])
                step = np.eye(4)
                step[:3, :3] = scale * R
                step[:3, 3] = t
                T_global = T_global @ step
                s_global *= scale
                print(f"  window {wi:>3}: aligned on {len(shared)} shared frames, "
                      f"{used} inliers, residual {resid:.4f}")
            else:
                print(f"  window {wi:>3}: NO shared frames -- track breaks here")

        # --- carry into the global frame ---------------------------------------
        A, b = T_global[:3, :3], T_global[:3, 3]
        flat = pts.reshape(-1, 3)
        world = (A @ flat.T).T + b
        world = world.reshape(pts.shape)

        keep_next = set(idx[-args.overlap:])
        prev_shared = {i: pts[idx.index(i)] for i in keep_next}

        # --- accumulate the cloud ----------------------------------------------
        thresh = np.percentile(conf, CONF_PERCENTILE)
        good = (conf > thresh) & keep_px[None] & np.isfinite(world).all(-1)
        sel = np.argwhere(good)
        if len(sel) > POINTS_PER_WINDOW:
            sel = sel[np.random.choice(len(sel), POINTS_PER_WINDOW, replace=False)]
        cloud_xyz.append(world[sel[:, 0], sel[:, 1], sel[:, 2]])
        cloud_rgb.append(rgb[sel[:, 0], sel[:, 1], sel[:, 2]])

        # --- per-keyframe record for the box-lifting stage ----------------------
        for j, i in enumerate(idx):
            if i in frames:
                continue                   # first window to see it wins
            E = np.eye(4)
            E[:3, :4] = extri[j]
            frames[i] = {
                "extrinsic": E.astype(np.float32),
                "intrinsic": intri[j].astype(np.float32),
                "depth": depth[j].astype(np.float16),
                "conf": conf[j].astype(np.float16),
                "T_window": T_global.astype(np.float32),
            }

        done = time.perf_counter() - t_start
        print(f"  window {wi + 1}/{len(wins)} done ({done / (wi + 1):.0f}s/window, "
              f"{done / 60:.1f} min elapsed)")

    xyz = np.concatenate(cloud_xyz)
    rgb = np.clip(np.concatenate(cloud_rgb), 0, 1)
    print(f"\nraw cloud: {len(xyz):,} points")

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    pcd = pcd.voxel_down_sample(VOXEL)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"after voxel+outlier filter: {len(pcd.points):,} points")

    out_ply = cam_dir / "cloud.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd)

    np.savez_compressed(
        cam_dir / "frames.npz",
        keyframe_idx=np.array(sorted(frames)),
        extrinsic=np.stack([frames[i]["extrinsic"] for i in sorted(frames)]),
        intrinsic=np.stack([frames[i]["intrinsic"] for i in sorted(frames)]),
        depth=np.stack([frames[i]["depth"] for i in sorted(frames)]),
        conf=np.stack([frames[i]["conf"] for i in sorted(frames)]),
        T_window=np.stack([frames[i]["T_window"] for i in sorted(frames)]),
    )
    print(f"-> {out_ply.relative_to(REPO)}")
    print(f"-> {(cam_dir / 'frames.npz').relative_to(REPO)}")
    print(f"total {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
