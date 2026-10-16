#!/usr/bin/env python3
"""Flatten the reconstructed field into a single top-down image.

The interactive viewer is the real deliverable, but a plan view is what you
actually want when the question is "did the rows come out straight" or "how many
plants did it miss". Collapsing to 2D removes the perspective that makes an
orbiting 3D point cloud hard to read, and the planting rows jump straight out.

Nothing here is generated or interpolated. Each point already carries the colour
of the video pixel it came from, so this walks the cloud, works out which output
pixel each point lands in, and averages the colours that land together. A pixel
is dark because nothing was reconstructed there, not because anything was
painted in.

Height is discarded outright, which is what makes it a plan view: boxes that
overlap here are plants whose canopies overlap seen from above, not duplicate
detections.

Run it after flatten.py and lift_boxes.py, since it reads both of their outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

WIDTH = 1800          # the strip is long and thin, so width carries the detail
BOX_BGR = (230, 110, 60)   # OpenCV is BGR; this is the blue the boxes render as
BOX_THICKNESS = 2
MIN_HEIGHT, MAX_HEIGHT = 200, 1400


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--no-boxes", action="store_true",
                    help="render the bare field, without plant boxes")
    ap.add_argument("--out", default=None, help="output path (default: work dir)")
    args = ap.parse_args()

    import cv2
    import open3d as o3d

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    cloud_path = cam_dir / "cloud_flat.ply"
    if not cloud_path.exists():
        raise SystemExit(f"missing {cloud_path}; run scripts/flatten.py first")

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(cloud.points)
    cols = np.asarray(cloud.colors)
    if len(cols) != len(pts):
        raise SystemExit("cloud has no colours; re-run reconstruct.py")

    along, across = pts[:, 0], pts[:, 1]
    span_a, span_c = np.ptp(along), np.ptp(across)

    # Keep the aspect ratio honest: one output pixel covers the same distance
    # in both directions, so the picture is not secretly stretched.
    height = int(np.clip(round(args.width * span_c / max(span_a, 1e-9)),
                         MIN_HEIGHT, MAX_HEIGHT))

    def to_px(a: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (a - along.min()) / (span_a + 1e-9) * (args.width - 1)
        y = (c - across.min()) / (span_c + 1e-9) * (height - 1)
        return x.astype(int), y.astype(int)

    xi, yi = to_px(along, across)

    # Sum the colours landing in each pixel, then divide by how many landed
    # there. Averaging rather than last-write-wins stops a single stray point
    # from speaking for a whole pixel.
    acc = np.zeros((height, args.width, 3), np.float32)
    hits = np.zeros((height, args.width), np.float32)
    np.add.at(acc, (yi, xi), cols)
    np.add.at(hits, (yi, xi), 1)
    img = (acc / np.maximum(hits, 1)[..., None] * 255).astype(np.uint8)

    n_boxes = 0
    plants_path = cam_dir / "plants.json"
    if not args.no_boxes and plants_path.exists():
        plants = json.loads(plants_path.read_text())["plants"]
        for p in plants:
            centre = np.asarray(p["centre_ground"])
            extent = np.asarray(p["extent"])
            lo = centre[:2] - extent[:2] / 2
            hi = centre[:2] + extent[:2] / 2
            (x0, x1), (y0, y1) = to_px(np.array([lo[0], hi[0]]),
                                       np.array([lo[1], hi[1]]))
            cv2.rectangle(img, (x0, y0), (x1, y1), BOX_BGR, BOX_THICKNESS)
            n_boxes += 1

    # Flip vertically so increasing "across" runs up the image rather than down,
    # and BGR -> RGB on the way out.
    out_path = Path(args.out) if args.out else cam_dir / "field_topdown.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img[::-1, :, ::-1])

    coverage = 100 * (hits > 0).mean()
    print(f"{args.width}x{height} px, {len(pts):,} points, {n_boxes} boxes")
    print(f"{coverage:.0f}% of the image has reconstructed data behind it")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
