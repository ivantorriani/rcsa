#!/usr/bin/env python3
"""Find crops in each keyframe using text prompts rather than a fixed class list.

A stock YOLO knows 80 everyday objects. That list happens to include broccoli,
because it came from food photography, but there is nothing for cabbage and
nothing for a pepper plant -- so an off-the-shelf detector finds two thirds of
nothing on this footage.

Open-vocabulary detection takes the prompt instead of a fixed vocabulary, which
covers all three crops with no annotation work.

Segmentation masks, not boxes. A box drawn around a cabbage from directly
overhead is mostly soil at the corners, and every one of those soil pixels
carries a depth that would be averaged into the plant's 3D extent. The mask
keeps the lifted points on the plant.

Writes detections.json: one entry per keyframe, each holding polygon outlines in
pixel coordinates, so lift_boxes.py can index the depth map directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

# Prompts are per crop. Phrasing matters more than it should with open-vocab
# models, so these are deliberately concrete nouns rather than plant names:
# "cabbage head" beats "cabbage", which drifts toward whole-plant foliage.
PROMPTS = {
    "cabbage": ["cabbage head", "green leafy cabbage plant"],
    "broccoli": ["broccoli head", "broccoli plant"],
    # The nadir camera shows the fruit clearly, which is also what disambiguates
    # this crop -- the ground-level view reads as tomato to every VLM we ran.
    "pepper": ["red chili pepper", "green chili pepper", "pepper plant"],
}

CONF = 0.10          # open-vocab scores run low; filtered properly at the 3D stage
MIN_AREA_FRAC = 0.0008   # ignore specks that cannot carry a usable depth patch


def load_model(prefer_seg: bool = True):
    """YOLOE segmentation if available, else YOLO-World boxes.

    YOLOE gives masks from a text prompt, which is what we actually want.
    YOLO-World is the fallback and only produces boxes -- lift_boxes.py handles
    both, it just gets noisier 3D extents from boxes.
    """
    from ultralytics import YOLO

    if prefer_seg:
        try:
            model = YOLO("yoloe-11l-seg.pt")
            return model, True
        except Exception as exc:
            print(f"  YOLOE unavailable ({type(exc).__name__}), falling back to YOLO-World")

    from ultralytics import YOLOWorld
    return YOLOWorld("yolov8x-worldv2.pt"), False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--limit", type=int, default=0,
                    help="match the reconstruct --limit so the two stages agree")
    ap.add_argument("--boxes-only", action="store_true",
                    help="skip YOLOE and use YOLO-World box prompts")
    args = ap.parse_args()

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    manifest = cam_dir / "keyframes.csv"
    if not manifest.exists():
        raise SystemExit(f"run scripts/keyframes.py {args.crop} --cams {args.cam} first")

    rows = list(csv.DictReader(manifest.open()))
    if args.limit:
        rows = rows[: args.limit]

    prompts = PROMPTS.get(args.crop)
    if prompts is None:
        raise SystemExit(f"no prompts defined for '{args.crop}'; add them to PROMPTS")

    model, has_masks = load_model(prefer_seg=not args.boxes_only)
    if has_masks:
        model.set_classes(prompts, model.get_text_pe(prompts))
    else:
        model.set_classes(prompts)
    print(f"prompts: {prompts}  (masks={has_masks})")

    out: dict[str, list[dict]] = {}
    n_det = 0
    for i, row in enumerate(rows):
        img = cam_dir / "keyframes" / row["keyframe"]
        res = model.predict(str(img), conf=args.conf, verbose=False)[0]

        h, w = res.orig_shape
        min_area = MIN_AREA_FRAC * h * w
        dets = []

        boxes = res.boxes
        polys = res.masks.xy if (has_masks and res.masks is not None) else None

        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[k].tolist()
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            det = {
                "label": prompts[int(boxes.cls[k])],
                "score": float(boxes.conf[k]),
                "bbox": [x1, y1, x2, y2],
            }
            if polys is not None and k < len(polys) and len(polys[k]) >= 3:
                det["polygon"] = np.asarray(polys[k], dtype=float).round(1).tolist()
            dets.append(det)

        out[row["keyframe"]] = dets
        n_det += len(dets)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} keyframes, {n_det} detections so far")

    path = cam_dir / "detections.json"
    path.write_text(json.dumps({
        "crop": args.crop,
        "prompts": prompts,
        "has_masks": has_masks,
        "conf": args.conf,
        "frames": out,
    }, indent=1))

    per_frame = n_det / max(1, len(rows))
    print(f"\n{n_det} detections over {len(rows)} keyframes ({per_frame:.1f}/frame)")
    print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
