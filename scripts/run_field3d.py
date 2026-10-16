#!/usr/bin/env python3
"""Run the whole pipeline for one crop: `run_field3d.py cabbage`.

Takes the name of a folder under original_footage/ and produces an interactive
3D field in results/field/.

Every stage caches its output, so a re-run picks up where the last one stopped
and costs nothing for work already done. Reconstruction is the expensive step
(tens of minutes); the rest are seconds to a couple of minutes. Use --force to
redo a stage you have changed the parameters of.

Phase A is the nadir camera alone. It is the camera that actually sees the field
as a field -- the two ground-level cameras are immersed in the canopy and face
opposite directions, and they get registered to this one later rather than being
reconstructed alongside it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
WORK = REPO / "work"
FOOTAGE = REPO / "original_footage"
VENV_PY = REPO / ".venv-3d" / "bin" / "python"


def run(script: str, cam_args: list[str], label: str) -> None:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    cmd = [py, str(SCRIPTS / script), *cam_args]
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed ({result.returncode})")
    print(f"    {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop", help="folder under original_footage/ (cabbage, pepper, broccoli)")
    ap.add_argument("--cam", default="3", help="camera to reconstruct; 3 is the nadir view")
    ap.add_argument("--limit", type=int, default=360,
                    help="keyframes to reconstruct (0 = whole clip, much slower)")
    ap.add_argument("--force", action="store_true", help="redo stages that already have output")
    args = ap.parse_args()

    crop_dir = FOOTAGE / args.crop
    if not crop_dir.is_dir():
        have = sorted(p.name for p in FOOTAGE.iterdir() if p.is_dir())
        raise SystemExit(f"no footage folder '{args.crop}'. Available: {', '.join(have)}")

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    crop, cam = args.crop, args.cam

    stages = [
        ("keyframes.py",  [crop, "--cams", cam],                 cam_dir / "keyframes.csv",   "keyframe selection"),
        ("rig_mask.py",   [crop, "--cams", cam],                 cam_dir / "rig_mask.png",    "rig mask"),
        ("reconstruct.py", [crop, "--cam", cam, "--limit", str(args.limit)],
                                                                  cam_dir / "cloud.ply",       "3D reconstruction (slow)"),
        ("flatten.py",    [crop, "--cam", cam],                   cam_dir / "ground_model.json", "drift correction"),
        ("detect.py",     [crop, "--cam", cam, "--limit", str(args.limit)],
                                                                  cam_dir / "detections.json", "crop detection"),
        ("lift_boxes.py", [crop, "--cam", cam],                   cam_dir / "plants.json",     "3D boxes"),
        ("viewer.py",     [crop, "--cam", cam],
                          REPO / "results" / "field" / f"{crop}_cam{cam}.html",                "interactive field"),
    ]

    t0 = time.perf_counter()
    for script, cargs, output, label in stages:
        if output.exists() and not args.force:
            print(f"\n=== {label} === cached ({output.relative_to(REPO)})")
            continue
        run(script, cargs, label)

    html = REPO / "results" / "field" / f"{crop}_cam{cam}.html"
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"open {html.relative_to(REPO)}")


if __name__ == "__main__":
    main()
