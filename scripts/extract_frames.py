#!/usr/bin/env python3
"""Sample frames from the field clips into a flat frame set + manifest.

Each source clip holds exactly one crop, so the filename stem gives us free
ground truth for crop type. That lands in the manifest as `crop_gt` and is the
only correctness signal available in this first pass.

The manifest is the single source of truth for the run: every model reads the
same rows, so results are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELD_DIR = REPO / "assets" / "field"
FRAMES_DIR = REPO / "data" / "frames"
MANIFEST = FRAMES_DIR / "manifest.csv"

# Crop label comes from the video filename stem.
CROPS = ("pepper", "broccoli", "cabbage")


def find_ffmpeg() -> str:
    for cand in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if found:
            return found
    sys.exit("ffmpeg not found. Install with: brew install ffmpeg")


def extract(ffmpeg: str, video: Path, crop: str, fps: float, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale frames so a re-run at a different fps can't leave a mixed set.
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    pattern = str(out_dir / f"{crop}_%03d.jpg")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high-quality JPEG; these frames are the model input
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames sampled per second of video (default: 2.0)")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
    rows = []

    for crop in CROPS:
        video = FIELD_DIR / f"{crop}.mp4"
        if not video.is_file():
            sys.exit(f"missing footage: {video}")

        frames = extract(ffmpeg, video, crop, args.fps, FRAMES_DIR / crop)
        print(f"{crop:9s} {len(frames):3d} frames")

        for i, path in enumerate(frames):
            rows.append({
                "frame_id": path.stem,
                "crop_gt": crop,
                "source_video": video.name,
                # ffmpeg's fps filter emits frames on an even 1/fps grid.
                "timestamp_s": round(i / args.fps, 3),
                "path": str(path.relative_to(REPO)),
            })

    with MANIFEST.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["frame_id", "crop_gt", "source_video", "timestamp_s", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} frames total -> {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
