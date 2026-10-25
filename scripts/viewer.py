#!/usr/bin/env python3
"""Build the interactive 3D field: point cloud plus one box per plant.

Everything is drawn in the ground frame rather than the reconstruction's own
axes, so Z really is height above the soil and the camera presets mean what
their names say.

Plotly rather than Open3D for the viewer itself: the output is a single HTML
file that opens in a browser with orbit, zoom and hover already working, and it
can be sent to someone who has none of this installed. Open3D is still doing the
geometry upstream, it just is not a good way to hand someone a scene.

Plotly.js is inlined, so the page works with no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"
RESULTS = REPO / "results" / "field"

MAX_POINTS = 120_000     # browsers start to crawl past this
BOX_COLOR = "#ff7043"
# One colour per detecting camera, so the fused view shows at a glance which
# part of the field each camera contributed. cam3 is the nadir camera and
# covers the whole strip; the two ground cameras reach only as far as their
# clips do.
CAM_COLOR = {"3": "#4fc3f7", "1": "#ff7043", "2": "#ffd54f"}
CAM_LABEL = {"3": "cam3 (nadir)", "1": "cam1 (ground)", "2": "cam2 (ground)"}


def hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def box_corners(centre: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Eight corners of an axis-aligned box in the ground frame."""
    h = extent / 2
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    return centre + signs * h


# Triangles for a cuboid, and the 12 edges as vertex-index pairs.
BOX_TRIS = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
            (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4)]
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crop")
    ap.add_argument("--cam", default="3")
    ap.add_argument("--fused", action="store_true",
                    help="render the merged three-camera field")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    ap.add_argument("--no-cloud", action="store_true",
                    help="plant boxes only -- the cloud is noisy enough that the "
                         "field layout reads more clearly without it")
    args = ap.parse_args()

    import open3d as o3d
    import plotly.graph_objects as go

    cam_dir = (WORK / args.crop / "fused") if args.fused else (WORK / args.crop / f"cam{args.cam}")
    plants_doc = json.loads((cam_dir / "plants.json").read_text())
    plants = plants_doc["plants"]

    traces = []
    if args.no_cloud:
        # Extent still has to come from somewhere for the axis ranges; the boxes
        # themselves define it when the cloud is not drawn.
        ground = np.array([p["centre_ground"] for p in plants])
    else:
        # cloud_flat.ply is already in the straightened ground frame, the same
        # one the boxes live in, so the two cannot disagree.
        cloud = o3d.io.read_point_cloud(str(cam_dir / "cloud_flat.ply"))
        ground = np.asarray(cloud.points)
        rgb = np.asarray(cloud.colors)

        # Which camera each point came from, when fuse.py recorded it. Without
        # this the fused cloud is one undifferentiated trace and looks exactly
        # like the single-camera one with extra clutter at one end -- there is
        # no way to see what the ground cameras contributed, which is the only
        # thing the fused view has to say.
        # flatten.py trims the height band, so prefer the labels it wrote for
        # the cloud that survived over the pre-trim ones.
        src_path = cam_dir / "source_cam_flat.npy"
        if not src_path.exists():
            src_path = cam_dir / "source_cam.npy"
        src = None
        if args.fused and src_path.exists():
            s_all = np.load(src_path)
            if len(s_all) == len(ground):
                src = s_all
            else:
                print(f"  note: source_cam.npy has {len(s_all):,} labels for "
                      f"{len(ground):,} points; re-run fuse.py to colour the "
                      f"cloud by camera")

        if len(ground) > args.max_points:
            pick = np.random.choice(len(ground), args.max_points, replace=False)
            ground, rgb = ground[pick], rgb[pick]
            if src is not None:
                src = src[pick]

        def cloud_trace(xyz, colours, name, visible=True, size=1.2):
            # A per-point colour list is what makes these traces big -- roughly
            # 18 characters a point. When every point shares a colour, pass the
            # one string instead; that is what keeps the flat-coloured copies
            # from doubling the page.
            colour = (colours if isinstance(colours, str) else
                      [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                       for r, g, b in colours])
            return go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                marker=dict(size=size, color=colour, opacity=0.85),
                name=name, hoverinfo="skip", visible=True if visible else "legendonly")

        if src is None:
            traces.append(cloud_trace(ground, rgb, "point cloud"))
        else:
            # True colour, split by camera so each can be switched off. A second
            # flat-coloured copy of each camera, hidden until clicked, answers
            # the question true colour cannot: which camera put that point there.
            for c in sorted(np.unique(src)):
                m = src == c
                label = CAM_LABEL.get(str(c), f"cam{c}")
                traces.append(cloud_trace(ground[m], rgb[m],
                                          f"{label} cloud — {int(m.sum()):,} pts"))
            # Schematic copies: one flat colour per camera, hidden until asked
            # for. Thinned hard, because their job is to show where each camera
            # put points, not to render the canopy.
            thin = max(1, len(ground) // max(1, args.max_points // 3))
            for c in sorted(np.unique(src)):
                m = np.where(src == c)[0][::thin]
                label = CAM_LABEL.get(str(c), f"cam{c}")
                traces.append(cloud_trace(ground[m],
                                          CAM_COLOR.get(str(c), BOX_COLOR),
                                          f"{label} cloud, flat colour",
                                          visible=False, size=1.6))

    # One mesh for every box, merged into a single trace so the legend toggle
    # turns all of them on and off at once.
    def box_traces(subset, colour, label):
        """Boxes, edges and hover points for one group of plants, as 3 traces."""
        vx, vy, vz, ti, tj, tk = [], [], [], [], [], []
        ex, ey, ez = [], [], []
        hx, hy, hz, htext = [], [], [], []
        for p in subset:
            c = np.array(p["centre_ground"])
            e = np.array(p["extent"])
            corners = box_corners(c, e)
            base = len(vx)
            vx.extend(corners[:, 0]); vy.extend(corners[:, 1]); vz.extend(corners[:, 2])
            for a, b, d in BOX_TRIS:
                ti.append(base + a); tj.append(base + b); tk.append(base + d)
            for a, b in BOX_EDGES:
                ex.extend([corners[a, 0], corners[b, 0], None])
                ey.extend([corners[a, 1], corners[b, 1], None])
                ez.extend([corners[a, 2], corners[b, 2], None])
            hx.append(c[0]); hy.append(c[1]); hz.append(c[2] + e[2] / 2)
            seen = ", ".join(f"cam{c}" for c in sorted(map(str, p.get("cameras", []))))
            fp = (e[0] + e[1]) / 2
            htext.append(
                f"plant {p['id']}<br>"
                f"extent {e[0]:.3f} x {e[1]:.3f} x {e[2]:.3f}<br>"
                f"height/footprint {e[2] / fp:.2f}<br>"
                f"seen by {seen or 'n/a'}<br>"
                f"seen in {p['observations']} keyframes<br>"
                f"score {p['score']:.2f}"
            )
        if not vx:
            return []
        # With the cloud hidden there is nothing to see through, so the boxes
        # carry the whole picture and can afford to be solid.
        return [
            go.Mesh3d(x=vx, y=vy, z=vz, i=ti, j=tj, k=tk, color=colour,
                      opacity=0.55 if args.no_cloud else 0.18,
                      name=label, legendgroup=label, showlegend=True,
                      hoverinfo="skip"),
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                         line=dict(color=colour, width=3), name=label,
                         legendgroup=label, showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=hx, y=hy, z=hz, mode="markers",
                         marker=dict(size=4, color=colour), text=htext,
                         hoverinfo="text", name=label, legendgroup=label,
                         showlegend=False),
        ]

    if plants:
        if args.fused:
            # Group by detecting camera. A plant the rig registered well enough
            # to be seen twice gets its own group rather than being filed under
            # either camera -- on this footage that is rare, and saying so in the
            # legend is more honest than hiding it.
            groups: dict[str, list] = {}
            for p in plants:
                cams = sorted(map(str, p.get("cameras", [])))
                key = "+".join(cams) if len(cams) > 1 else (cams[0] if cams else "?")
                groups.setdefault(key, []).append(p)
            for key in sorted(groups, key=lambda k: (len(k), k)):
                subset = groups[key]
                colour = (CAM_COLOR.get(key) if len(key) == 1 else "#b388ff")
                label = (CAM_LABEL.get(key, f"cam{key}") if len(key) == 1
                         else f"seen by {key.replace('+', ' + cam').replace(key[0], 'cam' + key[0], 1)}")
                traces.extend(box_traces(subset, colour or BOX_COLOR,
                                         f"{label} — {len(subset)} plants"))
        else:
            traces.extend(box_traces(plants, BOX_COLOR,
                                     f"plant boxes — {len(plants)}"))

    # The strip is roughly 10 long by 1.7 across by 0.9 tall. Drawn to true
    # proportions it renders as an unreadable sliver, so the two short axes are
    # stretched. Not physically honest, but the alternative is a scene you
    # cannot see; the axis labels still carry the real numbers.
    span = np.ptp(ground, axis=0)
    aspect = dict(x=1.0,
                  y=float(np.clip(span[1] / span[0] * 3, 0.25, 0.7)),
                  z=float(np.clip(span[2] / span[0] * 4, 0.15, 0.5)))

    # Camera presets. The nadir camera only ever saw the bed from above, so
    # "along the row" and "across the row" are synthetic viewpoints -- that is
    # the entire point of reconstructing it.
    views = {
        "Top (nadir)":   dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0)),
        "Along the row": dict(eye=dict(x=2.2, y=0, z=0.5), up=dict(x=0, y=0, z=1)),
        "Across the row": dict(eye=dict(x=0, y=2.2, z=0.5), up=dict(x=0, y=0, z=1)),
        "Oblique":       dict(eye=dict(x=1.5, y=1.5, z=1.2), up=dict(x=0, y=0, z=1)),
        "Ground level":  dict(eye=dict(x=1.8, y=0.4, z=0.12), up=dict(x=0, y=0, z=1)),
    }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f"{args.crop} — {'all three cameras' if args.fused else f'cam{args.cam}'} — {len(plants)} plants "
               f"({'unitless' if plants_doc['unitless'] else 'scaled'})"),
        template="plotly_dark",
        scene=dict(
            xaxis_title="along the row", yaxis_title="across the row",
            zaxis_title="height above soil",
            aspectmode="manual", aspectratio=aspect,
            # Open on the view the first button names, so the highlighted button
            # and what you are actually looking at agree.
            camera=views["Top (nadir)"],
        ),
        margin=dict(l=0, r=0, t=46, b=0),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.02, y=0.98,
            bgcolor="#222", font=dict(color="#eee"),
            buttons=[dict(label=name, method="relayout",
                          args=[{"scene.camera": cam}])
                     for name, cam in views.items()],
        )],
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_boxes" if args.no_cloud else ""
    label = "fused" if args.fused else f"cam{args.cam}"
    out = RESULTS / f"{args.crop}_{label}{suffix}.html"
    fig.write_html(str(out), include_plotlyjs="inline")
    size_mb = out.stat().st_size / 1e6
    drawn = "boxes only" if args.no_cloud else f"{len(ground):,} points"
    print(f"{len(plants)} plants, {drawn} -> {out.relative_to(REPO)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
