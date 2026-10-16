#!/usr/bin/env python3
"""Measure whether VGGT is usable on this Mac before the pipeline depends on it.

There is no NVIDIA GPU here, so VGGT runs on Apple's MPS backend. Two things can
go wrong and neither shows up as an error: MPS silently falls back to the CPU for
operations it does not implement, which can cost 10x, and the 16 GB is shared with
the rest of the system, so a long window can thrash instead of failing cleanly.

So we measure rather than assume. Reports wall time and peak memory for one
window, then extrapolates to a full clip at the observed keyframe count. That
number is the go/no-go on staying local.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"


def pick_device() -> tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available():
        # float16 rather than bfloat16: MPS bf16 coverage is patchy and a silent
        # CPU fallback is exactly the failure this benchmark exists to catch.
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--frames", type=int, default=24, help="window size to test")
    args = ap.parse_args()

    cam_dir = WORK / args.crop / f"cam{args.cam}"
    manifest = cam_dir / "keyframes.csv"
    if not manifest.exists():
        raise SystemExit(f"run scripts/keyframes.py {args.crop} first ({manifest} missing)")

    rows = list(csv.DictReader(manifest.open()))
    total_kf = len(rows)
    # Sample across the clip, not the first N: the opening frames are often a
    # parked stretch and would flatter the benchmark.
    step = max(1, total_kf // args.frames)
    picked = [cam_dir / "keyframes" / r["keyframe"] for r in rows[::step]][: args.frames]

    device, dtype = pick_device()
    print(f"device={device.type} dtype={dtype} frames={len(picked)} "
          f"(of {total_kf} keyframes in clip)")

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    t_load = time.perf_counter() - t0
    print(f"model load: {t_load:.1f}s")

    images = load_and_preprocess_images([str(p) for p in picked]).to(device)
    print(f"input tensor: {tuple(images.shape)}")

    if device.type == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(images)
    if device.type == "mps":
        torch.mps.synchronize()
    t_infer = time.perf_counter() - t0

    peak = (torch.mps.driver_allocated_memory() / 1e9) if device.type == "mps" else 0.0
    print(f"\ninference: {t_infer:.1f}s for {len(picked)} frames "
          f"({t_infer / len(picked):.2f}s/frame)")
    print(f"peak driver memory: {peak:.1f} GB")
    print("outputs:", {k: tuple(v.shape) for k, v in out.items()
                       if hasattr(v, "shape")})

    # Extrapolate: windows overlap by a third so we pay for ~1.5x the keyframes.
    stride = max(1, int(args.frames * 2 / 3))
    windows = max(1, -(-total_kf // stride))
    projected = windows * t_infer / 60
    print(f"\nprojected full clip: {windows} windows x {t_infer:.0f}s "
          f"= {projected:.1f} min")
    print("GATE: " + ("PASS (under 30 min)" if projected < 30 else
                      "FAIL (over 30 min) -- report before proceeding"))


if __name__ == "__main__":
    main()
