#!/usr/bin/env python3
"""Recover the time offset between the three clips of one crop.

The three cameras were not hardware-triggered. Frame counts differ within every
crop -- cabbage is 4578 / 4571 / 4561, broccoli 4963 / 4962 / 4953, pepper
4042 / 4041 / 4030 -- and cam3 is always the shortest while cam1 is always the
longest, which is what three webcams started and stopped by hand look like.
Nothing in the footage records the offset, so it has to be measured.

The cameras share no field of view at all: cam1 and cam2 point roughly opposite
ways and cam3 looks straight down, so there is no image content to match. What
they *do* share is the robot. The AMIGA drives in bursts, and a burst is a burst
in all three clips at the same instant. Per-frame optical flow magnitude is a
one-dimensional signature of that motion, and it survives the fact that the three
cameras are looking at completely different things.

Flow magnitude is not comparable across cameras in absolute terms -- the ground
cameras sit 20 cm from the canopy and see several times the pixel motion the
nadir camera does for the same ground speed -- so each profile is z-scored before
correlating. Only the *shape* of the stop-and-go carries across.

Measured on cabbage, correlating at full frame resolution:

    cam3-cam2   peak 0.686 @   0 frames   sharp (1 frame to -10%)
    cam1-cam2   peak 0.822 @  +7 frames   sharp (4 frames)
    cam3-cam1   peak 0.647 @ -17 frames   FLAT -- lag 0 scores 0.641

Those three lags do not close the triangle: 0 and +7 imply cam3-cam1 is -7, not
-17. The two ground cameras agree with each other because they share flow
statistics; nadir against ground is weak and ambiguous. So a per-pair argmax is
not good enough and this script does not use one.

Instead the three pairwise correlation curves are fitted jointly for the two free
offsets that a consistent triangle actually has (cam3 is the reference, so there
are two unknowns, not three), each pair weighted by how prominent its peak is
above its own background. That lets the sharp cam1-cam2 pair carry the flat
cam3-cam1 pair instead of being outvoted by it.

The result is an *initialiser*. It is good to a frame or two, which is not good
enough to fuse geometry on; rig_solve.py refines it as a continuous parameter
against the trajectory residual, where the signal is far more discriminative.

Writes work/<crop>/clip_sync.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FOOTAGE = REPO / "original_footage"
WORK = REPO / "work"

# Search half-width. The observed spread of clip durations within a crop is under
# 20 frames (1.3 s); +-45 frames (3 s) leaves generous room without letting the
# quasi-periodic stop-and-go alias onto a wrong burst.
MAX_LAG = 45

# Tracking width. The profile only needs the *shape* of the speed curve, and 160
# px is enough for that while keeping a 4500-frame pass to a couple of minutes.
PROFILE_WIDTH = 160

# Same Shi-Tomasi + Lucas-Kanade pairing keyframes.py uses, at a lower budget --
# this is a per-frame scalar, not a keyframe decision, so it can be cheaper.
TRACK_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=6, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# A peak must beat its own background by this much to be trusted on its own.
# Measured: cam1-cam2 clears it comfortably, cam3-cam1 does not.
PROMINENCE_FLOOR = 0.02
# Neighbourhood excluded when measuring "background" -- the correlation curve is
# smooth, so lags adjacent to the peak are not independent evidence.
PEAK_EXCLUSION = 8


def speed_profile(video: Path, width: int = PROFILE_WIDTH) -> np.ndarray:
    """Per-frame median feature displacement: a 1-D signature of robot motion."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")

    out: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = int(width * frame.shape[0] / frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            out.append(_flow(prev, gray))
        prev = gray
    cap.release()

    a = np.array(out, dtype=np.float64)
    if not np.isfinite(a).any():
        sys.exit(f"{video.name}: no measurable motion anywhere in the clip")
    # Unmeasurable frames (sun flare, buried lens) become the clip median rather
    # than a hole, so they neither invent motion nor punch a notch in the curve.
    return np.where(np.isfinite(a), a, np.nanmedian(a))


def _flow(prev: np.ndarray, gray: np.ndarray) -> float:
    pts = cv2.goodFeaturesToTrack(prev, **TRACK_PARAMS)
    if pts is None or len(pts) < 10:
        return float("nan")
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **LK_PARAMS)
    if nxt is None:
        return float("nan")
    ok = status.ravel() == 1
    if ok.sum() < 10:
        return float("nan")
    delta = (nxt[ok] - pts[ok]).reshape(-1, 2)
    return float(np.median(np.linalg.norm(delta, axis=1)))


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def correlate(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG):
    """Normalised cross-correlation of two speed profiles over +-max_lag frames.

    Returns (lags, corr). A positive lag means b must be shifted later than a to
    line them up, i.e. b's clip started earlier.
    """
    n = min(len(a), len(b))
    x, y = _z(a[:n]), _z(b[:n])
    full = np.correlate(x, y, "full") / n
    lags = np.arange(-n + 1, n)
    keep = np.abs(lags) <= max_lag
    return lags[keep], full[keep]


def peak_stats(lags: np.ndarray, corr: np.ndarray) -> dict:
    """Peak lag plus the two numbers that say whether to believe it."""
    k = int(np.argmax(corr))
    far = np.abs(lags - lags[k]) > PEAK_EXCLUSION
    background = float(np.max(corr[far])) if far.any() else 0.0
    # How far from the peak before the curve drops 10%: a sharp peak is localised.
    below = np.where(corr < corr[k] * 0.9)[0]
    width = int(abs(below[np.argmin(np.abs(below - k))] - k)) if len(below) else len(corr)
    return {
        "lag": int(lags[k]),
        "corr": float(corr[k]),
        "background": background,
        "prominence": float(corr[k] - background),
        "width": width,
        "trusted": bool(corr[k] - background >= PROMINENCE_FLOOR),
    }


def solve_offsets(curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
                  stats: dict[str, dict], max_lag: int) -> dict[int, float]:
    """Fit two triangle-consistent offsets to all three correlation curves at once.

    Convention: offset[c] is added to camera c's own timestamps to reach the common
    clock, so an event at common time T lands at t_c = T - offset[c]. The lag a
    correlation peak reports for pair (a, b) -- how much later the same event shows
    up in a than in b -- is therefore offset[b] - offset[a].

    cam3 is the reference at 0, so the unknowns are just offset[1] and offset[2].
    For a candidate pair of offsets every camera pair has an implied lag, and the
    score is the sum of each pair's correlation at that lag, weighted by that
    pair's prominence. Maximising the total forces one consistent answer and lets
    a sharp pair outvote a flat one -- which is the whole point, given cam3-cam1
    is the flat one and cam1-cam2 is the sharp one.
    """
    grid = np.arange(-max_lag, max_lag + 1)
    best, best_score = (0.0, 0.0), -np.inf
    for t1 in grid:
        for t2 in grid:
            offs = {3: 0.0, 1: float(t1), 2: float(t2)}
            score = 0.0
            for pair in ((3, 1), (3, 2), (1, 2)):
                lag = offs[pair[1]] - offs[pair[0]]
                if abs(lag) > max_lag:
                    score = -np.inf
                    break
                lags, corr = curves[pair]
                w = max(stats[f"cam{pair[0]}-cam{pair[1]}"]["prominence"], 0.0)
                score += w * float(corr[int(np.searchsorted(lags, lag))])
            if score > best_score:
                best_score, best = score, (float(t1), float(t2))
    return {3: 0.0, 1: best[0], 2: best[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop", help="folder under original_footage/ (cabbage, pepper, broccoli)")
    ap.add_argument("--max-lag", type=int, default=MAX_LAG,
                    help=f"search half-width in frames (default {MAX_LAG})")
    ap.add_argument("--width", type=int, default=PROFILE_WIDTH,
                    help="tracking width in pixels")
    args = ap.parse_args()

    crop_dir = FOOTAGE / args.crop
    if not crop_dir.is_dir():
        have = sorted(p.name for p in FOOTAGE.iterdir() if p.is_dir())
        sys.exit(f"no footage folder '{args.crop}'. Available: {', '.join(have)}")

    fps = None
    profiles: dict[int, np.ndarray] = {}
    for cam in (1, 2, 3):
        video = crop_dir / f"{args.crop}_{cam}.avi"
        if not video.exists():
            sys.exit(f"missing {video}")
        if fps is None:
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            cap.release()
        print(f"profiling cam{cam}...", flush=True)
        profiles[cam] = speed_profile(video, args.width)
        print(f"  {len(profiles[cam])} samples, median flow "
              f"{np.median(profiles[cam]):.2f} px")

    curves, stats = {}, {}
    print()
    for a, b in ((3, 1), (3, 2), (1, 2)):
        lags, corr = correlate(profiles[a], profiles[b], args.max_lag)
        curves[(a, b)] = (lags, corr)
        s = peak_stats(lags, corr)
        stats[f"cam{a}-cam{b}"] = s
        flag = "" if s["trusted"] else "   <- flat, not trusted alone"
        print(f"cam{a}-cam{b}  peak {s['corr']:.3f} @ {s['lag']:+d} frames "
              f"({s['lag'] / fps:+.3f} s)  prominence {s['prominence']:.3f} "
              f"width {s['width']}{flag}")

    offsets = solve_offsets(curves, stats, args.max_lag)

    # Closure error of the raw per-pair argmaxes. Consistency requires
    # lag(1,2) == lag(3,2) - lag(3,1); this is how far off that was, and so how
    # much work the joint fit had to do.
    raw = {k: v["lag"] for k, v in stats.items()}
    closure = raw["cam1-cam2"] - (raw["cam3-cam2"] - raw["cam3-cam1"])
    fitted = {f"cam{a}-cam{b}": offsets[b] - offsets[a] for a, b in ((3, 1), (3, 2), (1, 2))}

    print(f"\ntriangle closure of raw peaks: {closure:+d} frames "
          f"({'consistent' if abs(closure) <= 2 else 'INCONSISTENT -- joint fit required'})")
    print("\nfitted offsets (add to that camera's timestamps; cam3 = 0):")
    for cam in (1, 2, 3):
        print(f"  cam{cam}: {offsets[cam]:+.1f} frames ({offsets[cam] / fps:+.3f} s)")
    print("\nfitted pairwise lags vs raw argmax:")
    for key, lag in fitted.items():
        print(f"  {key}: fitted {lag:+.1f}  raw {raw[key]:+d}  "
              f"residual {abs(lag - raw[key]):.1f} frames")

    out_dir = WORK / args.crop
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clip_sync.json"
    out_path.write_text(json.dumps({
        "crop": args.crop,
        "fps": fps,
        "reference_cam": 3,
        "offset_frames": {str(c): offsets[c] for c in (1, 2, 3)},
        "offset_seconds": {str(c): offsets[c] / fps for c in (1, 2, 3)},
        "pairs": stats,
        "raw_closure_frames": int(closure),
        "note": "initialiser only; rig_solve.py refines tau against trajectory residual",
    }, indent=1))
    print(f"\n-> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
