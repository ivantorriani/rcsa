#!/usr/bin/env python3
"""Turn 2D crop masks into one 3D box per plant.

Each detection is lifted by reading the reconstruction's depth map wherever the
mask covers, unprojecting those pixels into the global frame, and taking their
extent. The mask matters here: a box drawn around a cabbage seen from overhead is
mostly soil in the corners, and soil depths dragged into the extent would inflate
every plant.

The harder half is that a plant is not detected once. The robot creeps forward,
so the same cabbage appears in dozens of consecutive keyframes and produces
dozens of overlapping 3D blobs. Clustering their centroids collapses those back
into one plant, and the number of times a plant was seen becomes a useful
confidence signal in its own right -- a box found in thirty keyframes is real, a
box found once is probably a shadow.

Boxes are axis-aligned in the *ground* frame rather than free-floating oriented
boxes. A free OBB fitted to a noisy partial point cloud tends to tilt to chase
the noise, which looks obviously wrong for plants growing out of flat soil. So we
fit the soil plane once, build a frame on it, and let the boxes keep their feet
on the ground with a single shared yaw taken from the direction of the row.

Everything is unitless -- monocular reconstruction fixes shape but not size.
SCALE below is the one place to apply a real measurement later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

# VGGT's working resolution for a 640x480 source: width is set to 518 and height
# follows the aspect ratio rounded to a multiple of 14. No crop, no padding, so
# detections map across by a plain scale factor.
SRC_W, SRC_H = 640, 480

SCALE = 1.0              # multiply every distance by this once a real measurement exists
MIN_POINTS = 40          # a detection with fewer lifted points has no reliable extent
CLUSTER_EPS = 0.035      # centroid spacing that still counts as the same plant
MIN_OBSERVATIONS = 3     # a plant seen in fewer keyframes than this is noise
EXTENT_PCT = (5, 95)     # robust extent: ignore the tails rather than the true max


# The ground frame now comes from flatten.py, which both removes the chaining
# drift and takes the long axis from the robot's actual travel direction. The
# previous version fitted one global plane and picked its in-plane axes from an
# arbitrary seed vector, which is why "across the row" pointed nowhere in
# particular and the two crop rows smeared into a single band.


def lift_detection(poly: np.ndarray, depth: np.ndarray, conf: np.ndarray,
                   K: np.ndarray, E: np.ndarray, T: np.ndarray,
                   conf_floor: float) -> np.ndarray | None:
    """Unproject the pixels inside one mask into global coordinates."""
    import cv2
    h, w = depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    scaled = poly * np.array([w / SRC_W, h / SRC_H])
    cv2.fillPoly(mask, [scaled.astype(np.int32)], 1)

    sel = (mask > 0) & (conf > conf_floor) & np.isfinite(depth) & (depth > 0)
    if sel.sum() < MIN_POINTS:
        return None

    ys, xs = np.nonzero(sel)
    z = depth[ys, xs].astype(np.float64)
    # pixel -> camera ray -> camera-space point
    cam = np.stack([(xs - K[0, 2]) * z / K[0, 0],
                    (ys - K[1, 2]) * z / K[1, 1],
                    z], axis=1)
    # camera -> window-local world (E is world->cam)
    R, t = E[:3, :3], E[:3, 3]
    local = (cam - t) @ R
    # window-local -> global
    return local @ T[:3, :3].T + T[:3, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--conf-floor", type=float, default=1.5,
                    help="VGGT depth confidence below which pixels are ignored")
    args = ap.parse_args()

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    for required in ("frames.npz", "detections.json", "cloud.ply", "ground_model.json"):
        if not (cam_dir / required).exists():
            raise SystemExit(f"missing {cam_dir / required}; run the earlier stages first")

    import open3d as o3d

    data = np.load(cam_dir / "frames.npz")
    # Pull each array out once. Indexing `data["depth"][s]` inside the loop would
    # re-decompress the whole stack on every single detection.
    depths, confs = data["depth"], data["conf"]
    intrinsics, extrinsics, transforms = (data["intrinsic"], data["extrinsic"],
                                          data["T_window"])
    kf_idx = list(data["keyframe_idx"])
    rows = list(csv.DictReader((cam_dir / "keyframes.csv").open()))
    # detections are keyed by keyframe filename; frames.npz by row position
    name_of = {i: rows[i]["keyframe"] for i in kf_idx}
    slot = {i: s for s, i in enumerate(kf_idx)}

    dets_doc = json.loads((cam_dir / "detections.json").read_text())
    frames = dets_doc["frames"]

    # --- lift every detection independently ---------------------------------
    blobs = []          # (centroid, points, score, keyframe index)
    skipped = 0
    for i in kf_idx:
        dets = frames.get(name_of[i]) or []
        if not dets:
            continue
        s = slot[i]
        depth = depths[s].astype(np.float32)
        conf = confs[s].astype(np.float32)
        K, E, T = intrinsics[s], extrinsics[s], transforms[s]

        for det in dets:
            poly = det.get("polygon")
            if poly is None:
                x1, y1, x2, y2 = det["bbox"]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            pts = lift_detection(np.asarray(poly, dtype=float), depth, conf,
                                 K, E, T, args.conf_floor)
            if pts is None:
                skipped += 1
                continue
            blobs.append((pts.mean(0), pts, float(det["score"]), i))

    if not blobs:
        raise SystemExit("no detections could be lifted -- check --conf-floor")
    print(f"lifted {len(blobs)} detections ({skipped} had too few valid depths)")

    # --- straightened ground frame from flatten.py --------------------------
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flatten import apply_model

    model = json.loads((cam_dir / "ground_model.json").read_text())

    # Cluster on the *corrected* centroids. Clustering in the bent frame would
    # merge plants that drift happened to push together and split ones it pulled
    # apart.
    blobs = [(apply_model(b[0][None], model)[0], b[1], b[2], b[3]) for b in blobs]
    centroids = np.stack([b[0] for b in blobs])
    cp = o3d.geometry.PointCloud()
    cp.points = o3d.utility.Vector3dVector(centroids.astype(np.float64))
    labels = np.asarray(cp.cluster_dbscan(eps=CLUSTER_EPS,
                                          min_points=MIN_OBSERVATIONS,
                                          print_progress=False))
    n_clusters = labels.max() + 1 if len(labels) else 0
    print(f"{len(blobs)} sightings -> {n_clusters} plants "
          f"({int((labels < 0).sum())} unclustered, dropped)")

    plants = []
    for c in range(n_clusters):
        members = [blobs[j] for j in np.nonzero(labels == c)[0]]
        pts = apply_model(np.concatenate([m[1] for m in members]), model)

        lo = np.percentile(pts, EXTENT_PCT[0], axis=0)
        hi = np.percentile(pts, EXTENT_PCT[1], axis=0)
        # Plants grow up from the soil, so the box floor sits at ground level.
        # This is only safe now that flatten.py puts the soil at z=0 everywhere:
        # against the old bent frame, a plant on drift-raised ground got pinned
        # to a zero far below it and the box stretched into the air.
        lo[2] = max(lo[2], 0.0)
        if hi[2] <= lo[2]:
            continue

        centre_local = (lo + hi) / 2
        plants.append({
            "id": len(plants),
            "centre_ground": centre_local.tolist(),
            "extent": ((hi - lo) * SCALE).tolist(),
            "observations": len(members),
            "score": float(np.mean([m[2] for m in members])),
            "points": int(len(pts)),
        })

    out = {
        "crop": args.crop,
        "cam": args.cam,
        "scale": SCALE,
        "unitless": SCALE == 1.0,
        "frame": "straightened ground frame from ground_model.json",
        "plants": plants,
    }
    path = cam_dir / "plants.json"
    path.write_text(json.dumps(out, indent=1))

    if plants:
        ext = np.array([p["extent"] for p in plants])
        obs = np.array([p["observations"] for p in plants])
        print(f"\n{len(plants)} plants")
        print(f"  extent  median {np.round(np.median(ext, axis=0), 3)} "
              f"(along, across, height; unitless)")
        print(f"  seen in median {int(np.median(obs))} keyframes "
              f"(min {obs.min()}, max {obs.max()})")
    print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
