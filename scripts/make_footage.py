#!/usr/bin/env python3
"""Render annotated experiment footage from the recorded VLM results.

Produces, under results/footage/:
  <model>_<crop>.mp4          - one model's verdict burned onto each frame
  compare_<variant>_<crop>.mp4 - the models of one prompt variant, side by side

Runs made with a different prompt are a different experiment, so every panel
carries a badge naming its variant (v1 baseline vs v2 prompted) and the 3-up
only ever compares models that saw the same prompt.

The frames were sampled every 0.5s, so the default 2 fps replays at real speed.
Drop to --fps 1 if you want time to actually read the overlays.

Nothing here re-runs a model. It reads results/*/results.jsonl only, so it is
cheap to re-render after tweaking the layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_DIR = RESULTS / "footage"

BANNER_H = 112          # strip under the frame holding the verdict text
PAD = 12

# Frames are 640x480; anything else gets letterboxed to match so the 3-up
# panels line up even if a clip was captured at a different size.
FRAME_W, FRAME_H = 640, 480

BG = (18, 18, 20)
FG = (238, 238, 238)
DIM = (150, 150, 155)
GOOD = (86, 196, 120)
BAD = (232, 92, 92)
WARN = (226, 170, 72)

STAGE_COLOR = {"early": (120, 170, 235), "middle": (226, 170, 72),
               "harvest": (86, 196, 120)}

# Prompt variants. The badge colour is the fastest way to tell, mid-playback,
# which experiment you are looking at.
VARIANTS = {
    "v1": {"label": "V1  BASELINE", "accent": (150, 150, 158)},
    "v2": {"label": "V2  PROMPTED", "accent": (120, 170, 235)},
}


def variant_of(rec: dict) -> str:
    """Which prompt produced this record."""
    prompt = (rec.get("prompt") or "").lower()
    if "v2" in prompt or "__v2" in str(rec.get("model", "")):
        return "v2"
    return "v1"


def display_model(label: str) -> str:
    """Strip the results-dir tag; the badge already says which variant it is."""
    return label.split("__", 1)[0]


def font(size: int, bold: bool = False):
    """Menlo if present (fixed width keeps the columns from jittering)."""
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                 if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


F_MODEL = font(19, bold=True)
F_BODY = font(15)
F_SMALL = font(13)
F_BADGE = font(14, bold=True)


def load_results() -> dict[str, dict[str, dict]]:
    """{model: {frame_id: record}} from every results.jsonl on disk."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for jsonl in sorted(RESULTS.glob("*/results.jsonl")):
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["model"]][r["frame_id"]] = r
    return dict(out)


def panel(rec: dict, n: int, total: int, disputed: bool) -> Image.Image:
    """One frame plus its verdict banner."""
    img = Image.new("RGB", (FRAME_W, FRAME_H + BANNER_H), BG)

    src = REPO / "data" / "frames" / rec["frame_id"].rsplit("_", 1)[0] / f"{rec['frame_id']}.jpg"
    if src.is_file():
        photo = Image.open(src).convert("RGB")
        if photo.size != (FRAME_W, FRAME_H):
            photo.thumbnail((FRAME_W, FRAME_H))
        img.paste(photo, ((FRAME_W - photo.width) // 2, (FRAME_H - photo.height) // 2))

    d = ImageDraw.Draw(img)

    # variant badge, top-right over the photo, so it is readable at a glance
    v = VARIANTS[variant_of(rec)]
    btxt = v["label"]
    bw = d.textlength(btxt, font=F_BADGE) + 18
    bx, by, bh = FRAME_W - PAD - bw, PAD, 26
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5,
                        fill=(12, 12, 14), outline=v["accent"], width=2)
    d.text((bx + 9, by + 5), btxt, font=F_BADGE, fill=v["accent"])

    y = FRAME_H + 8
    d.line([(0, FRAME_H), (FRAME_W, FRAME_H)], fill=(60, 60, 66), width=2)

    d.text((PAD, y), display_model(rec["model"]), font=F_MODEL, fill=FG)
    # frame counter, right-aligned
    counter = f"{rec['frame_id']}   {n}/{total}"
    d.text((FRAME_W - PAD - d.textlength(counter, font=F_SMALL), y + 4),
           counter, font=F_SMALL, fill=DIM)

    y += 27
    crop, gt = rec.get("crop_type"), rec.get("crop_gt")
    hit = crop == gt
    d.text((PAD, y), "crop ", font=F_BODY, fill=DIM)
    x = PAD + d.textlength("crop ", font=F_BODY)
    d.text((x, y), str(crop), font=F_BODY, fill=GOOD if hit else BAD)
    x += d.textlength(str(crop), font=F_BODY)
    if not hit:
        d.text((x, y), f"   (gt {gt})", font=F_BODY, fill=DIM)

    y += 22
    stage = rec.get("stage")
    d.text((PAD, y), "stage ", font=F_BODY, fill=DIM)
    x = PAD + d.textlength("stage ", font=F_BODY)
    d.text((x, y), str(stage), font=F_BODY, fill=STAGE_COLOR.get(stage, WARN))
    x += d.textlength(str(stage), font=F_BODY)
    if disputed:
        d.text((x + 10, y), "* models disagree", font=F_SMALL, fill=WARN)

    y += 22
    conf = rec.get("confidence")
    meta = f"conf {conf:.2f}" if isinstance(conf, (int, float)) else "conf n/a"
    hv = rec.get("head_visible")
    if hv is not None:
        meta += f"    head {'yes' if hv else 'no'}"
    meta += f"    {rec.get('latency_s', 0):.2f}s"
    if not rec.get("schema_ok"):
        meta += "    SCHEMA FAIL"
    if rec.get("error"):
        meta += "    ERROR"
    d.text((PAD, y), meta, font=F_SMALL, fill=DIM)
    return img


def encode(frames: list[Image.Image], dest: Path, fps: float) -> None:
    """PNG sequence -> H.264. yuv420p + even dims so QuickTime will play it."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, im in enumerate(frames):
            if im.width % 2 or im.height % 2:
                im = im.crop((0, 0, im.width - im.width % 2, im.height - im.height % 2))
            im.save(tmp / f"{i:04d}.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(tmp / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", str(dest)],
            check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=float, default=2.0,
                    help="2 = real time (frames were sampled every 0.5s)")
    ap.add_argument("--crops", nargs="+", default=None, help="limit to these crops")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found - brew install ffmpeg")

    res = load_results()
    if not res:
        sys.exit("no results found - run scripts/run_vlm.py first")

    models = sorted(res)
    # {variant: [model labels]} - comparing across prompts would be apples to
    # oranges, so each variant gets its own 3-up.
    by_variant: dict[str, list[str]] = defaultdict(list)
    for m in models:
        by_variant[variant_of(next(iter(res[m].values())))].append(m)
    # group frame ids by clip, ordered by the numeric suffix
    clips: dict[str, list[str]] = defaultdict(list)
    for fid in {f for m in res.values() for f in m}:
        clips[fid.rsplit("_", 1)[0]].append(fid)
    for k in clips:
        clips[k].sort(key=lambda f: int(f.rsplit("_", 1)[1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    for clip, fids in sorted(clips.items()):
        if args.crops and clip not in args.crops:
            continue

        # a frame is "disputed" when models on the SAME prompt disagree on stage
        disputed = {}
        for f in fids:
            disputed[f] = any(
                len({res[m][f].get("stage") for m in peers if f in res[m]}) > 1
                for peers in by_variant.values()
            )

        for m in models:
            have = [f for f in fids if f in res[m]]
            if not have:
                continue
            frames = [panel(res[m][f], i, len(have), disputed[f])
                      for i, f in enumerate(have, 1)]
            dest = OUT_DIR / f"{m}_{clip}.mp4"
            encode(frames, dest, args.fps)
            made.append(dest)
            print(f"  {dest.relative_to(REPO)}  ({len(frames)} frames)")

        # side-by-side: same frame, one column per model, one video per variant
        for variant, peers in sorted(by_variant.items()):
            shared = [f for f in fids if all(f in res[m] for m in peers)]
            if not shared or len(peers) < 2:
                continue
            frames = []
            for i, f in enumerate(shared, 1):
                cols = [panel(res[m][f], i, len(shared), disputed[f]) for m in peers]
                w = sum(c.width for c in cols)
                strip = Image.new("RGB", (w, cols[0].height), BG)
                x = 0
                for c in cols:
                    strip.paste(c, (x, 0))
                    x += c.width
                d = ImageDraw.Draw(strip)
                for gx in range(cols[0].width, w, cols[0].width):
                    d.line([(gx, 0), (gx, strip.height)], fill=(60, 60, 66), width=2)
                frames.append(strip)
            dest = OUT_DIR / f"compare_{variant}_{clip}.mp4"
            encode(frames, dest, args.fps)
            made.append(dest)
            print(f"  {dest.relative_to(REPO)}  ({len(frames)} frames, {len(peers)}-up)")

    print(f"\n{len(made)} file(s) -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
