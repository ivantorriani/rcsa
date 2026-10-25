#!/usr/bin/env python3
"""Recover the fixed transform between the rig's cameras, from motion alone.

The three cameras share no field of view. cam1 and cam2 point roughly opposite
ways, cam3 looks straight down, and a cabbage seen from the side and the same
cabbage seen from above have no matchable surface in common. There is nothing to
triangulate across cameras and no amount of feature matching will invent it.

What the cameras do share is the robot. They are bolted to one chassis, so the
three trajectories are one curve expressed in three coordinate frames, and the
constant transform between those frames is recoverable from the motion by itself.
That is the classical hand-eye problem: for a rigid pair, every relative motion
satisfies A X = X B, where A and B are the same physical movement as the two
cameras saw it and X is the transform we want.

Two complications this footage adds to the textbook version:

**Scale is unknown and per-camera.** Monocular reconstruction fixes shape, not
size, so each camera's translations carry their own arbitrary factor. Only the
ratio is recoverable. The translation equation absorbs it as a fourth unknown:

    t_A = s * R_X t_B + (I - R_A) t_X

**The clocks differ.** The three clips were started by hand, so a keyframe at the
same list position is not the same instant. clip_sync.py measures the offset to
about a frame; this script refines it as a continuous parameter, because a frame
of error at robot speed is a real displacement.

A warning that shapes the whole design: the three unknowns are not equally
determined. `(I - R_A)` is near-singular when the rotations are small, and the
rotations here are about one degree, so **t_X is far weaker than R_X and s**.
Rotation and relative scale come out well; the baseline does not, and the
validation gate reports its conditioning separately rather than averaging it into
one reassuring number. Treating X as a single estimate is how you end up trusting
its worst component.

Measured on cabbage cam3's real trajectory (see the module docstring numbers in
the repo notes): the path is nearly straight -- 2nd/1st principal singular ratio
0.035 -- which would make any camera-centre alignment degenerate. But hand-eye
needs rotation axes to span 3D, not positions, and at a 25-keyframe baseline the
rotation-axis singular ratio 3rd/1st is 0.655 over 335 pairs. Terrain pitch and
roll supply the diversity. The method is conditioned; the straight path is not
the obstacle it looks like.

Per-keyframe increments are a different matter: median relative rotation is 0.84
degrees against a comparable pose noise floor, and rotation grows faster than
sqrt(gap) out to gaps of 5-25. So the solve runs on smoothed trajectories at a
long lag, never on consecutive keyframes. That one choice is the difference
between the method working and returning confident nonsense.

Run `--self-test` to exercise the whole estimator against a known transform
before any ground-camera reconstruction exists.

Writes work/<crop>/rig.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"

# Resampling grid. cam3 emits roughly 2 keyframes/s and the ground cameras more,
# so 0.25 s makes the grid the bottleneck rather than either camera's irregular
# clock -- which is what we want, since both are being interpolated anyway.
GRID_S = 0.25
# Quaternion smoothing width, in grid samples (~2.25 s). Per-keyframe rotation
# increments are mostly noise; this is what pulls the real motion back out.
SMOOTH_K = 9
# Lag between the poses forming each relative motion. Long enough that real
# rotation dominates pose noise, short enough to stay inside roughly one
# reconstruction window so the scale random walk cannot bend it.
LAG_S = 2.0
# How far a pose pair may miss the requested baseline before it is dropped,
# as a fraction of the baseline. Tight enough that every constraint shares a
# baseline; loose enough to survive the gaps in a bursty keyframe clock.
LAG_TOL_FRAC = 0.25
# Below this many motion pairs the solve is not worth reporting.
MIN_PAIRS = 30

# Validation thresholds. Calibrated by scaling real rotations down toward pure
# translation until the solve breaks: cond_t tracks the transition cleanly.
COND_T_FAIL = 0.04
COND_T_WARN = 0.08
AXIS_EIG_FAIL = 0.08
# Spread of the main bootstrap mode, 75th percentile, in degrees. Calibrated
# against the self-test, which recovers a known transform to 1.5 deg.
BOOTSTRAP_ROT_P75_FAIL_DEG = 15.0
# Fraction of resamples landing on the 180 degree branch. Below the warn level
# the flip is absent; past the fail level the majority mode is no longer a
# majority and the branch cannot be chosen from motion.
FLIP_FRAC_WARN = 0.05
FLIP_FRAC_FAIL = 0.40
AXIS_EIG_PASS = 0.12
ROT_MEDIAN_FAIL_DEG = 0.6
BOOTSTRAP_N = 50


# --------------------------------------------------------------------------
# trajectory extraction
# --------------------------------------------------------------------------

def load_trajectory(cam_dir: Path, offset_s: float = 0.0) -> dict:
    """Camera centres and orientations per keyframe, on the common clock.

    frames.npz holds `extrinsic` (world->cam, in the window's own local frame)
    and `T_window` (that window's similarity into the global frame). The centre
    is -R^T t carried through T_window.

    Position keeps the full similarity, so the result stays in the same units as
    cloud.ply and every validation check that compares against it. Orientation
    must use the rigid part only, because T_window's rotation block is scale*R
    and is not a rotation.

    `scale_cum` and `segment` are read when present and derived otherwise, so
    reconstructions made before those fields existed still load.
    """
    d = np.load(cam_dir / "frames.npz")
    E = d["extrinsic"].astype(np.float64)
    T = d["T_window"].astype(np.float64)
    n = len(E)

    A = T[:, :3, :3]
    det = np.linalg.det(A)
    s = np.cbrt(np.maximum(det, 1e-12))
    R_T = A / s[:, None, None]
    # Re-orthonormalise: det drifts ~6e-4 off unity on real data, and everything
    # downstream differentiates these.
    U, _, Vt = np.linalg.svd(R_T)
    W = np.tile(np.eye(3), (n, 1, 1))
    W[:, 2, 2] = np.sign(np.linalg.det(U @ Vt))
    R_T = U @ W @ Vt

    Rc, tc = E[:, :3, :3], E[:, :3, 3]
    local = -np.einsum("nji,nj->ni", Rc, tc)
    C = np.einsum("nij,nj->ni", A, local) + T[:, :3, 3]
    R_world = R_T @ np.transpose(Rc, (0, 2, 1))

    kf_idx = d["keyframe_idx"]
    rows = list(csv.DictReader((cam_dir / "keyframes.csv").open()))
    t = np.array([float(rows[i]["timestamp_s"]) for i in kf_idx]) + offset_s

    seg = d["segment"] if "segment" in d.files else np.zeros(n, dtype=np.int32)
    return {"t": t, "C": C, "R": R_world, "scale_cum": s,
            "segment": np.asarray(seg), "keyframe_idx": np.asarray(kf_idx)}


def largest_segment(traj: dict) -> dict:
    """Restrict to the longest unbroken segment; poses never span a track break."""
    seg = traj["segment"]
    if len(np.unique(seg)) == 1:
        return traj
    best = max(np.unique(seg), key=lambda s: int((seg == s).sum()))
    m = seg == best
    print(f"  track has {len(np.unique(seg))} segments; using segment {best} "
          f"({m.sum()} of {len(seg)} keyframes)")
    return {k: v[m] for k, v in traj.items()}


# --------------------------------------------------------------------------
# resampling and smoothing
# --------------------------------------------------------------------------

def resample(traj: dict, grid: np.ndarray) -> tuple[np.ndarray, Rot]:
    """SLERP orientation and linearly interpolate position onto `grid`."""
    t = traj["t"]
    order = np.argsort(t)
    t, C, R = t[order], traj["C"][order], traj["R"][order]
    keep = np.concatenate([[True], np.diff(t) > 1e-9])   # Slerp needs strict increase
    t, C, R = t[keep], C[keep], R[keep]

    rots = Rot.from_matrix(R)
    sl = Slerp(t, rots)
    g = np.clip(grid, t[0], t[-1])
    pos = np.stack([np.interp(g, t, C[:, i]) for i in range(3)], axis=1)
    return pos, sl(g)


def smooth(pos: np.ndarray, rots: Rot, k: int = SMOOTH_K) -> tuple[np.ndarray, Rot]:
    """Moving average on positions, sign-aligned quaternion average on rotations."""
    if k <= 1:
        return pos, rots
    q = rots.as_quat()
    q = q * np.sign(q @ q[0])[:, None]       # resolve the double cover before averaging
    q = uniform_filter1d(q, k, axis=0, mode="nearest")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return uniform_filter1d(pos, k, axis=0, mode="nearest"), Rot.from_quat(q)


def lag_pairs(grid: np.ndarray, lag_s: float, tol_frac: float = LAG_TOL_FRAC):
    """Index pairs whose poses are `lag_s` apart *in time*, not in samples.

    The grid is the reference camera's native keyframe times, and those are
    violently uneven: keyframes are chosen on measured drift, so the robot's
    stop-and-go puts cam3's median gap at 0.134 s against a mean of 0.49 s. A
    fixed *sample* offset therefore spans a different duration at every point in
    the clip -- about two seconds inside a burst, tens of seconds across a
    stationary stretch -- which pools relative motions of wildly different
    baselines as if they were the same constraint.

    That is fatal in a specific and misleading way. The mixture is roughly
    unbiased, so the central estimate of R_X still looks reasonable; but each
    bootstrap resample draws a different blend of baselines, so the spread
    explodes and the solve reads as unstable when the real problem is that
    `lag_s` was never controlling the baseline at all.

    Pairing by time puts every constraint on one baseline. Pairs that cannot hit
    it within `tol_frac` are dropped rather than stretched.
    """
    n = len(grid)
    i = np.arange(n)
    j = np.searchsorted(grid, grid + lag_s)
    j = np.clip(j, 0, n - 1)
    # searchsorted lands at or just past the target; the previous sample is
    # sometimes the closer of the two.
    jp = np.clip(j - 1, 0, n - 1)
    better_prev = np.abs(grid[jp] - grid[i] - lag_s) < np.abs(grid[j] - grid[i] - lag_s)
    j = np.where(better_prev, jp, j)
    err = np.abs(grid[j] - grid[i] - lag_s)
    keep = (j > i) & (err <= tol_frac * lag_s)
    return i[keep], j[keep]


def relative_motions(pos: np.ndarray, rots: Rot, i: np.ndarray, j: np.ndarray):
    """Relative motion between the pose pairs (i, j), in each start pose's frame."""
    R = rots.as_matrix()
    R_a, R_b = R[i], R[j]
    dR = np.transpose(R_a, (0, 2, 1)) @ R_b
    dt = np.einsum("nji,nj->ni", R_a, pos[j] - pos[i])
    return dR, dt


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def park_martin(A_R: np.ndarray, B_R: np.ndarray) -> np.ndarray:
    """Rotation part of AX = XB (Park & Martin 1994).

    Chosen over Andreff's combined linear system specifically because it enforces
    SO(3) rather than recovering a near-rotation and orthonormalising after. At
    one degree of signal that distinction is the difference between 3 degrees of
    error and 8. It is also scale-invariant, so the unknown relative scale cannot
    leak into the rotation -- which lets the translation stage below treat scale
    as a clean linear unknown.
    """
    a = Rot.from_matrix(A_R).as_rotvec()
    b = Rot.from_matrix(B_R).as_rotvec()
    M = b.T @ a
    # R_X = (M^T M)^{-1/2} M^T, via eigendecomposition of the symmetric M^T M.
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)
    W = np.eye(3)
    W[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ W @ Vt


def solve_translation(A_R, A_t, B_t, R_X):
    """Least squares for [s, t_X] in  t_A = s R_X t_B + (I - R_A) t_X.

    Returns the solution plus the design matrix conditioning, which is the
    statistic that says whether t_X means anything. (I - R_A) collapses as the
    rotations shrink, so on near-straight motion this is the weak unknown.
    """
    n = len(A_t)
    M = np.zeros((3 * n, 4))
    y = A_t.reshape(-1)
    M[:, 0] = (B_t @ R_X.T).reshape(-1)
    M[:, 1:] = (np.eye(3)[None] - A_R).reshape(-1, 3)
    sol, *_ = np.linalg.lstsq(M, y, rcond=None)
    sv = np.linalg.svd(M, compute_uv=False)
    cond = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    return float(sol[0]), sol[1:], cond


def solve_pair(traj_a: dict, traj_b: dict, tau: float, grid_s=GRID_S,
               lag_s=LAG_S, k=SMOOTH_K, grid=None) -> dict:
    """Full two-stage solve for one camera pair at a given time offset.

    `grid` may be pinned by the caller. The non-linear refinement needs it to be,
    because otherwise the overlap window -- and therefore the number of residuals
    -- changes every time the optimiser nudges tau, which least_squares cannot
    accept. Pinning the grid and letting only camera b's sample points move with
    tau keeps the residual a fixed length while still refining the offset.
    """
    lo = max(traj_a["t"].min(), traj_b["t"].min() + tau)
    hi = min(traj_a["t"].max(), traj_b["t"].max() + tau)
    if hi - lo < 10 * lag_s:
        raise SystemExit(f"only {hi - lo:.1f}s of overlap between the cameras; "
                         f"need at least {10 * lag_s:.0f}s")

    # Evaluate on the reference camera's OWN keyframe times, not a uniform grid.
    #
    # This matters more than any other choice in the file. Keyframes are selected
    # on measured drift, so their spacing in time is wildly uneven -- cam3's
    # median gap is 0.134 s against a mean of 0.49 s, because the robot moves in
    # bursts. A uniform time grid therefore lands disproportionately inside the
    # long quiet gaps, where SLERP simply interpolates smoothly between two
    # distant poses and invents motion that never happened. Measured: cam3's real
    # rotation over a 2.1 s span is 4.26 degrees, but after resampling onto a
    # 0.25 s grid the solver saw 0.23 degrees. Ninety-five per cent of the signal
    # was being thrown away before the estimator ever ran, which is why the
    # recovered transform was unstable and drifted with the lag parameter.
    #
    # Using the reference camera's native times keeps its rotation intact and
    # interpolates only the other camera -- which is the denser of the two
    # anyway (cam1 has 2445 keyframes to cam3's 1583), so its interpolation
    # error is the smaller one to accept.
    if grid is None:
        ta = traj_a["t"]
        grid = ta[(ta >= lo) & (ta <= hi)]
        # No uniform-grid fallback. There used to be one, for pairs whose overlap
        # left few native samples, and it quietly destroyed the solve: a uniform
        # grid SLERPs straight through the robot's stationary gaps and reports the
        # smooth interpolation as real rotation. Measured on cam2, whose 136s
        # overlap fell nine samples short of the old threshold and so took the
        # fallback: median relative rotation read 0.11 deg against 2.87 deg for
        # cam1, which kept its native grid -- a 25x difference that came entirely
        # from the resampling and not from the footage. Too few samples is a
        # reason to report a thin solve, never a reason to invent motion.

    pa, ra = smooth(*resample(traj_a, grid), k)
    pb, rb = smooth(*resample(traj_b, grid - tau), k)
    # Pair by time, not by sample index -- see lag_pairs. Both cameras are
    # sampled on the same grid (b at grid - tau), so one set of index pairs
    # describes the same physical motion in both frames.
    ia, ja = lag_pairs(grid, lag_s)
    if len(ia) < MIN_PAIRS:
        raise SystemExit(
            f"only {len(ia)} motion pairs at a {lag_s:.1f}s baseline on "
            f"{len(grid)} samples spanning {grid[-1] - grid[0]:.0f}s; "
            f"need {MIN_PAIRS}. Use a shorter --lag.")
    A_R, A_t = relative_motions(pa, ra, ia, ja)
    B_R, B_t = relative_motions(pb, rb, ia, ja)

    R_X = park_martin(A_R, B_R)
    s, t_X, cond = solve_translation(A_R, A_t, B_t, R_X)
    return {"R_X": R_X, "t_X": t_X, "scale": s, "cond_t": cond,
            "A_R": A_R, "B_R": B_R, "A_t": A_t, "B_t": B_t,
            "grid": grid, "n_pairs": len(A_R)}


def rotation_cost(sol: dict) -> float:
    """RMS rotation residual of A R_X == R_X B, in degrees.

    Rotation-only, therefore scale-free, so the time offset cannot trade against
    the scale unknown while it is being searched.
    """
    R_X = sol["R_X"]
    resid = np.transpose(sol["A_R"] @ R_X, (0, 2, 1)) @ (R_X @ sol["B_R"])
    ang = Rot.from_matrix(resid).magnitude()
    return float(np.degrees(np.sqrt((ang ** 2).mean())))


def solve_tau(traj_a, traj_b, bracket=(-2.0, 2.0), coarse=0.1, **kw):
    """Grid then golden-section search for the time offset.

    Bracket defaults to +-2 s, which the frame-count differences bound comfortably
    (cabbage's 17-frame spread at 15 fps is 1.1 s) and which stays inside the
    single smooth basin the cost surface has.
    """
    taus = np.arange(bracket[0], bracket[1] + 1e-9, coarse)
    costs = []
    for tau in taus:
        try:
            costs.append(rotation_cost(solve_pair(traj_a, traj_b, tau, **kw)))
        except SystemExit:
            costs.append(np.inf)
    costs = np.array(costs)
    best = taus[int(np.argmin(costs))]
    r = minimize_scalar(
        lambda x: rotation_cost(solve_pair(traj_a, traj_b, float(x), **kw)),
        bounds=(best - coarse * 1.5, best + coarse * 1.5), method="bounded")
    return float(r.x), taus, costs


def refine(traj_a, traj_b, sol, tau, **kw):
    """Joint non-linear refinement of (R_X, t_X, log s, tau) on the full residual.

    soft_l1 because the ground cameras will contribute outlier poses and the fit
    should bound them rather than chase them.
    """
    x0 = np.concatenate([Rot.from_matrix(sol["R_X"]).as_rotvec(),
                         sol["t_X"], [np.log(max(sol["scale"], 1e-6))], [tau]])
    pinned = sol["grid"]
    n_resid = 6 * len(sol["A_R"])

    def resid(x):
        R_X = Rot.from_rotvec(x[:3]).as_matrix()
        t_X, s, tau_ = x[3:6], np.exp(x[6]), x[7]
        try:
            p = solve_pair(traj_a, traj_b, float(tau_), grid=pinned, **kw)
        except SystemExit:
            return np.full(n_resid, 1e3)
        rot = Rot.from_matrix(np.transpose(p["A_R"] @ R_X, (0, 2, 1))
                              @ (R_X @ p["B_R"])).as_rotvec()
        tr = p["A_t"] - s * (p["B_t"] @ R_X.T) - (np.eye(3)[None] - p["A_R"]) @ t_X
        out = np.concatenate([rot.reshape(-1), tr.reshape(-1)])
        if len(out) != n_resid:          # belt and braces; the grid is pinned
            out = np.resize(out, n_resid)
        return out

    out = least_squares(resid, x0, loss="soft_l1", max_nfev=200)
    return {"R_X": Rot.from_rotvec(out.x[:3]).as_matrix(), "t_X": out.x[3:6],
            "scale": float(np.exp(out.x[6])), "tau": float(out.x[7]),
            "success": bool(out.success)}


# --------------------------------------------------------------------------
# diagnostics and validation
# --------------------------------------------------------------------------

def refine_scale_on_trajectory(traj_a, traj_b, R_X, tau, step=0.5):
    """Re-estimate relative scale by aligning the two trajectories directly.

    Hand-eye gets scale from the translation equation t_A = s R_X t_B +
    (I - R_A) t_X, and that equation is near-singular here: the rotations are
    about a degree, so (I - R_A) is tiny and the fit is dominated by whatever
    noise sits in t_A. Trajectory alignment uses the same data without that
    weakness -- once R_X is known, camera b's whole path carried into a's frame
    must match a's path up to the rig's constant lever arm, which is a four
    unknown linear problem (one scale, three offset) over hundreds of samples.

    Measured on cabbage cam1: hand-eye returned 0.990 where the trajectories want
    1.060, and taking the trajectory estimate cuts the median path disagreement
    from 0.139 to 0.094 -- about a plant's width, on a field whose plants sit
    0.09 apart. cam2 moves 0.5% and is unaffected either way.

    The lever arm is absorbed into the offset rather than solved for; it is a
    constant, and separating it needs the same rotation diversity the baseline
    estimate already lacks. Only the scale is taken from here.
    """
    t0 = max(traj_a["t"].min(), traj_b["t"].min() + tau)
    t1 = min(traj_a["t"].max(), traj_b["t"].max() + tau)
    ts = np.arange(t0, t1, step)
    if len(ts) < 20:
        return None

    def at(tr, q):
        t = tr["t"]; o = np.argsort(t); t = t[o]; C = tr["C"][o]
        return np.stack([np.interp(q, t, C[:, k]) for k in range(3)], axis=1)

    Ca, Cb = at(traj_a, ts), at(traj_b, ts - tau)
    u = Cb @ R_X.T
    M = np.zeros((3 * len(ts), 4))
    M[:, 0] = u.reshape(-1)
    for k in range(3):
        M[k::3, 1 + k] = 1.0
    sol, *_ = np.linalg.lstsq(M, Ca.reshape(-1), rcond=None)
    s_fit = float(sol[0])
    resid = np.linalg.norm(s_fit * u + sol[1:] - Ca, axis=1)
    return {"scale": s_fit, "resid_median": float(np.median(resid)),
            "resid_max": float(resid.max()), "n": int(len(ts))}


def diagnostics(sol: dict) -> dict:
    """The three numbers that say whether the solve had anything to work with."""
    rv = Rot.from_matrix(sol["A_R"]).as_rotvec()
    ang = np.linalg.norm(rv, axis=1)
    sig = rv[ang > np.radians(0.1)]
    if len(sig) < 10:
        eig_min = 0.0
    else:
        axes = sig / np.linalg.norm(sig, axis=1, keepdims=True)
        w = np.linalg.eigvalsh(axes.T @ axes / len(axes))
        eig_min = float(w.min())       # isotropic = 0.33, single-axis = 0
    return {"rot_median_deg": float(np.degrees(np.median(ang))),
            "axis_eig_min": eig_min,
            "cond_t": sol["cond_t"],
            "n_pairs": sol["n_pairs"]}


def bootstrap(traj_a, traj_b, tau, n=BOOTSTRAP_N, **kw) -> dict:
    """Resample motion pairs to get an empirical spread on R_X and scale.

    Self-calibrating, which the fixed thresholds are not: they were tuned on
    cabbage and may not transfer to broccoli or pepper. If the spread is wide the
    solve is unstable regardless of what the other statistics say.
    """
    base = solve_pair(traj_a, traj_b, tau, **kw)
    n_pairs = base["n_pairs"]
    rng = np.random.default_rng(0)
    angs, scales = [], []
    for _ in range(n):
        idx = rng.integers(0, n_pairs, n_pairs)
        try:
            R = park_martin(base["A_R"][idx], base["B_R"][idx])
            s, _, _ = solve_translation(base["A_R"][idx], base["A_t"][idx],
                                        base["B_t"][idx], R)
        except np.linalg.LinAlgError:
            continue
        angs.append(np.degrees(Rot.from_matrix(base["R_X"].T @ R).magnitude()))
        scales.append(s)
    if not angs:
        return {"rot_std_deg": 99.0, "rot_p75_deg": 99.0, "flip_frac": 1.0,
                "scale_std": 99.0, "n_ok": 0}
    angs = np.asarray(angs)
    # The spread is bimodal, so a standard deviation is the wrong summary.
    #
    # park_martin recovers R_X from the eigendecomposition of M^T M, and when the
    # rotation axes do not span 3D well that decomposition has a 180 degree
    # branch. Measured on cam2 at a 2s baseline: 82.5% of resamples land within
    # 9 degrees of the central estimate -- tighter than cam1's 9.8 -- and 17.5%
    # land at 175-180 degrees. The std of that mixture is 63 degrees, which reads
    # as a hopeless solve when the truth is a well-determined estimate plus a
    # discrete flip. Reporting the flip fraction separately from the spread of
    # the main mode is the difference between diagnosing it and discarding it.
    flip = angs > 90.0
    main = angs[~flip]
    return {"rot_std_deg": float(np.std(angs)),
            "rot_p75_deg": float(np.percentile(main, 75)) if len(main) else 99.0,
            "flip_frac": float(flip.mean()),
            "scale_std": float(np.std(scales)) if scales else 99.0,
            "n_ok": len(angs)}


def validate(pairs: dict) -> dict:
    """Physical checks the solver never saw, plus the conditioning gates."""
    v, fails, warns = {}, [], []

    for key, p in pairs.items():
        d, b = p["diagnostics"], p["bootstrap"]
        if d["cond_t"] < COND_T_FAIL:
            fails.append(f"{key}: cond_t {d['cond_t']:.3f} < {COND_T_FAIL} "
                         f"-- motion too close to pure translation")
        elif d["cond_t"] < COND_T_WARN:
            warns.append(f"{key}: cond_t {d['cond_t']:.3f} -- rotation usable, "
                         f"baseline t_X is not; solve it geometrically")
        if d["axis_eig_min"] < AXIS_EIG_FAIL:
            fails.append(f"{key}: rotation axes span {d['axis_eig_min']:.3f} "
                         f"< {AXIS_EIG_FAIL} -- motion is single-axis")
        if d["rot_median_deg"] < ROT_MEDIAN_FAIL_DEG:
            fails.append(f"{key}: median rotation {d['rot_median_deg']:.2f} deg "
                         f"< {ROT_MEDIAN_FAIL_DEG} -- not enough signal")
        if b.get("rot_p75_deg", 99.0) > BOOTSTRAP_ROT_P75_FAIL_DEG:
            fails.append(f"{key}: bootstrap spread on R_X is "
                         f"{b['rot_p75_deg']:.1f} deg (75th pct of the main mode) "
                         f"-- solve is unstable")
        if b.get("flip_frac", 0.0) > FLIP_FRAC_FAIL:
            fails.append(f"{key}: {100 * b['flip_frac']:.0f}% of bootstrap "
                         f"resamples flip R_X by ~180 deg -- the branch is not "
                         f"decidable from the motion alone")
        elif b.get("flip_frac", 0.0) > FLIP_FRAC_WARN:
            warns.append(f"{key}: {100 * b['flip_frac']:.0f}% of bootstrap "
                         f"resamples flip R_X by ~180 deg. The majority branch is "
                         f"the one reported and the physical axis checks below "
                         f"agree with it, but the degeneracy is real -- treat a "
                         f"future crop that pushes this past "
                         f"{100 * FLIP_FRAC_FAIL:.0f}% as unsolved")
        if p.get("tau_cost_reduction", 1.0) < 0.20:
            warns.append(f"{key}: solving tau only improved the fit by "
                         f"{100 * p.get('tau_cost_reduction', 0):.0f}% -- the time "
                         f"offset is weakly identified; trust clip_sync instead")

    # cam3 is nadir, the ground cameras are horizontal: their axes should be
    # near perpendicular. This is a fact about the robot the solver never used.
    def axis_angle(a, b):
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))

    z = np.array([0, 0, 1.0])
    axes = {k: p["R_X"] @ z for k, p in pairs.items()}
    for key, ax in axes.items():
        a = axis_angle(z, ax)
        v[f"{key}_axis_deg"] = a
        if not 60 <= a <= 120:
            fails.append(f"{key}: optical axes {a:.0f} deg apart; a nadir camera "
                         f"and a horizontal one should be near 90")
        elif not 75 <= a <= 105:
            warns.append(f"{key}: optical axes {a:.0f} deg apart (expected ~90)")

    if "3-1" in axes and "3-2" in axes:
        a12 = axis_angle(axes["3-1"] / np.linalg.norm(axes["3-1"]),
                         axes["3-2"] / np.linalg.norm(axes["3-2"]))
        v["cam1_cam2_axis_deg"] = a12
        if not 135 <= a12 <= 225:
            fails.append(f"cam1 and cam2 point {a12:.0f} deg apart; they are "
                         f"back-to-back on one mast and should be near 180")
        elif not 150 <= a12 <= 210:
            warns.append(f"cam1/cam2 {a12:.0f} deg apart (expected ~180)")

    v["failures"], v["warnings"] = fails, warns
    v["status"] = "fail" if fails else ("warn" if warns else "pass")
    return v


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def self_test(crop: str = "cabbage", ref_cam: str = "3") -> int:
    """Recover a known transform from a synthetic second camera.

    Uses cam3's *real* trajectory as the body motion, so the test inherits the
    actual noise, the actual stop-and-go and the actual near-straight path rather
    than a flattering synthetic curve. A known rig transform, a known relative
    scale and a known time offset are applied, the synthetic camera is sampled on
    its own irregular clock, and the estimator has to find all three back.

    This exists to be runnable before cam1 and cam2 are reconstructed, so the
    method is proven before 30 minutes of GPU time is spent feeding it.
    """
    cam_dir = WORK / crop / f"cam{ref_cam}"
    if not (cam_dir / "frames.npz").exists():
        print(f"self-test needs {cam_dir}/frames.npz")
        return 1

    a = largest_segment(load_trajectory(cam_dir))
    print(f"self-test on {crop} cam{ref_cam}: {len(a['t'])} keyframes, "
          f"{a['t'][-1] - a['t'][0]:.0f}s")

    true_rot = Rot.from_euler("xyz", [88.0, 3.0, -5.0], degrees=True)
    true_R, true_t, true_s, true_tau = true_rot.as_matrix(), np.array([0.21, -0.06, 0.30]), 1.37, 0.80
    rng = np.random.default_rng(7)

    # Camera b observes the same body motion through the rig transform, with its
    # own arbitrary scale, its own clock, and its own pose noise.
    # A rigidly mounted camera sits at C_a + R_a @ t_X -- the offset rotates with
    # the body. Using a constant world-frame offset instead silently removes the
    # only signal that determines t_X, which makes the baseline look unrecoverable
    # when it is really just absent from the test.
    Rb = np.einsum("nij,jk->nik", a["R"], true_R)
    Cb = (a["C"] + np.einsum("nij,j->ni", a["R"], true_t)) / true_s
    noise_deg, noise_pos = 0.85, 0.018
    Rb = np.einsum("nij,njk->nik", Rb,
                   Rot.from_rotvec(rng.normal(0, np.radians(noise_deg), (len(Rb), 3))).as_matrix())
    Cb = Cb + rng.normal(0, noise_pos, Cb.shape)

    # Independent irregular clock, offset by true_tau.
    tb = a["t"] - true_tau
    jitter = rng.uniform(0.6, 1.4, len(tb))
    keep = np.cumsum(jitter) % 1.0 < 0.75
    b = {"t": tb[keep], "C": Cb[keep], "R": Rb[keep],
         "segment": np.zeros(keep.sum(), dtype=int)}

    tau, taus, costs = solve_tau(a, b)
    sol = solve_pair(a, b, tau)
    ref = refine(a, b, sol, tau)

    cost0 = rotation_cost(solve_pair(a, b, 0.0))
    reduction = (cost0 - rotation_cost(sol)) / max(cost0, 1e-9)

    rot_err = np.degrees(Rot.from_matrix(ref["R_X"].T @ true_R).magnitude())
    scale_err = abs(ref["scale"] - true_s) / true_s
    tau_err = abs(ref["tau"] - true_tau)
    d = diagnostics(sol)

    print(f"\n  tau        : {ref['tau']:+.3f}s   truth {true_tau:+.3f}   "
          f"error {tau_err * 1000:.0f} ms")
    print(f"  rotation   : error {rot_err:.2f} deg")
    print(f"  scale      : {ref['scale']:.3f}      truth {true_s:.3f}   "
          f"error {100 * scale_err:.1f}%")
    print(f"  baseline   : {np.linalg.norm(ref['t_X']):.3f}  truth "
          f"{np.linalg.norm(true_t):.3f}  (weak unknown -- expect ~15%)")
    print(f"  diagnostics: rot_median {d['rot_median_deg']:.2f} deg, "
          f"axis_eig_min {d['axis_eig_min']:.3f}, cond_t {d['cond_t']:.3f}, "
          f"{d['n_pairs']} pairs")
    print(f"  tau matters: solving it cut the residual by {100 * reduction:.0f}% "
          f"(tau=0 gives {cost0:.2f} deg)")

    base_err = abs(np.linalg.norm(ref["t_X"]) - np.linalg.norm(true_t)) / np.linalg.norm(true_t)
    ok = (rot_err < 6.0 and scale_err < 0.08 and tau_err < 0.25 and base_err < 0.30)
    print(f"\n  SELF-TEST {'PASS' if ok else 'FAIL'}  (thresholds: rot<6deg, "
          f"scale<8%, tau<250ms, baseline<30%)")
    print(f"  baseline error {100 * base_err:.0f}% -- loose on purpose: (I - R_A) is "
          f"near-singular at one degree of rotation, so t_X is the weak unknown "
          f"and gets refined against the ground plane at fusion time.")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def plot_check(a: dict, pairs: dict, traj_b: dict, out_path):
    """Overlay the aligned trajectories. Three near-coincident curves is a pass.

    This exists because every scalar in the gate can be satisfied by a transform
    that is still wrong. Three curves that lie on top of each other is a fact
    about the rig that no amount of well-conditioned nonsense reproduces.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["along / across", "along / height", "across / height"]
    planes = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j), title in zip(axes, planes, titles):
        ax.plot(a["C"][:, i], a["C"][:, j], lw=2.0, label="cam3 (reference)",
                color="#4477aa")
        for (key, p), colour in zip(pairs.items(), ["#ee6677", "#228833"]):
            b = traj_b[key]
            moved = p["scale"] * (b["C"] @ p["R_X"].T) + p["t_X"]
            ax.plot(moved[:, i], moved[:, j], lw=1.2, alpha=0.85,
                    label=f"cam{key.split('-')[1]} -> fused", color=colour)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("rig check: the three trajectories should coincide")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop", nargs="?", default="cabbage")
    ap.add_argument("--ref", default="3", help="reference camera (the nadir one)")
    ap.add_argument("--cams", default="1,2", help="cameras to register to --ref")
    ap.add_argument("--no-traj-scale", action="store_true",
                    help="keep hand-eye's scale instead of the trajectory fit")
    ap.add_argument("--lag", type=float, default=LAG_S,
                    help=f"baseline in seconds between the poses forming each "
                         f"relative motion (default {LAG_S})")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the estimator against a known transform and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test(args.crop))

    crop_work = WORK / args.crop
    sync_path = crop_work / "clip_sync.json"
    if not sync_path.exists():
        raise SystemExit(f"run scripts/clip_sync.py {args.crop} first")
    sync = json.loads(sync_path.read_text())
    offsets = {k: v / sync["fps"] for k, v in sync["offset_frames"].items()}

    ref_dir = crop_work / f"cam{args.ref}"
    if not (ref_dir / "frames.npz").exists():
        raise SystemExit(f"missing {ref_dir}/frames.npz")
    a = largest_segment(load_trajectory(ref_dir, offsets.get(args.ref, 0.0)))
    print(f"cam{args.ref} (reference): {len(a['t'])} keyframes, "
          f"t {a['t'][0]:.1f}-{a['t'][-1]:.1f}s")

    pairs, traj_b = {}, {}
    for cam in [c.strip() for c in args.cams.split(",") if c.strip()]:
        cam_dir = crop_work / f"cam{cam}"
        if not (cam_dir / "frames.npz").exists():
            print(f"skipping cam{cam}: no frames.npz")
            continue
        b = largest_segment(load_trajectory(cam_dir, offsets.get(cam, 0.0)))
        traj_b[f"{args.ref}-{cam}"] = b
        print(f"cam{cam}: {len(b['t'])} keyframes, t {b['t'][0]:.1f}-{b['t'][-1]:.1f}s")

        key = f"{args.ref}-{cam}"
        print(f"  solving {key}...")
        tau, _, _ = solve_tau(a, b, lag_s=args.lag)
        sol = solve_pair(a, b, tau, lag_s=args.lag)
        ref = refine(a, b, sol, tau, lag_s=args.lag)
        cost0 = rotation_cost(solve_pair(a, b, 0.0, lag_s=args.lag))
        reduction = (cost0 - rotation_cost(sol)) / max(cost0, 1e-9)

        final = solve_pair(a, b, ref["tau"], lag_s=args.lag)
        traj_fit = refine_scale_on_trajectory(a, b, ref["R_X"], ref["tau"])
        if traj_fit and not args.no_traj_scale:
            print(f"    scale {ref['scale']:.3f} (hand-eye) -> "
                  f"{traj_fit['scale']:.3f} (trajectory, residual "
                  f"{traj_fit['resid_median']:.3f} over {traj_fit['n']} samples)")
            ref["scale"] = traj_fit["scale"]
        pairs[key] = {
            "scale_handeye": float(sol["scale"]),
            "trajectory_fit": traj_fit,
            "R_X": ref["R_X"], "t_X": ref["t_X"], "scale": ref["scale"],
            "tau_s": ref["tau"], "tau_init_s": tau,
            "tau_cost_reduction": float(reduction),
            "residual_deg": rotation_cost(final),
            "diagnostics": diagnostics(final),
            "bootstrap": bootstrap(a, b, ref["tau"], lag_s=args.lag),
        }
        p = pairs[key]
        print(f"    tau {p['tau_s']:+.3f}s (init {tau:+.3f}), scale {p['scale']:.3f}, "
              f"residual {p['residual_deg']:.2f} deg")
        print(f"    cond_t {p['diagnostics']['cond_t']:.3f}, "
              f"axis_eig_min {p['diagnostics']['axis_eig_min']:.3f}, "
              f"rot_median {p['diagnostics']['rot_median_deg']:.2f} deg, "
              f"{p['diagnostics']['n_pairs']} pairs")
        print(f"    bootstrap: main mode p75 {p['bootstrap']['rot_p75_deg']:.1f} deg, "
              f"180-flip branch {100 * p['bootstrap']['flip_frac']:.0f}% of resamples")

    if not pairs:
        raise SystemExit("no camera pairs could be solved")

    v = validate(pairs)
    print("\n" + "=" * 68)
    print(f"VALIDATION: {v['status'].upper()}")
    for f in v["failures"]:
        print(f"  FAIL  {f}")
    for w in v["warnings"]:
        print(f"  warn  {w}")
    for k in ("3-1_axis_deg", "3-2_axis_deg", "cam1_cam2_axis_deg"):
        if k in v:
            print(f"  {k}: {v[k]:.0f} deg")
    print("=" * 68)

    try:
        check_png = crop_work / "rig_check.png"
        plot_check(a, pairs, traj_b, check_png)
        print(f"\nvisual check -> {check_png.relative_to(REPO)}")
        print("  three curves lying on top of each other is a pass; three "
              "unrelated squiggles is a fail no scalar above will catch.")
    except Exception as exc:
        print(f"\n(could not render rig_check.png: {exc})")

    if v["status"] == "fail":
        print("\nNot writing rig.json. Fusing on a transform that failed these "
              "checks would produce a field that looks plausible and is wrong.")
        sys.exit(2)

    out = {"crop": args.crop, "ref_cam": args.ref, "status": v["status"],
           "validation": {k: val for k, val in v.items()},
           "pairs": {k: {"X": np.block([[p["R_X"], p["t_X"][:, None]],
                                        [np.zeros(3), 1.0]]).tolist(),
                         "scale": p["scale"], "tau_s": p["tau_s"],
                         "residual_deg": p["residual_deg"],
                         "tau_cost_reduction": p["tau_cost_reduction"],
                         "diagnostics": p["diagnostics"],
                         "bootstrap": p["bootstrap"]}
                     for k, p in pairs.items()}}
    path = crop_work / "rig.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
