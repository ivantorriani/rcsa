#!/usr/bin/env python3
"""Consolidate per-model JSONL results into a CSV plus a readable summary.

Two signals are available without any hand labeling:

  1. Crop-ID accuracy. Each source clip holds one crop, so the clip is ground
     truth. One correction applies: assets/field/pepper.mp4 in fact shows a
     tomato crop -- round orange fruit on a sprawling vine, not blocky peppers.
     All three models independently called it tomato, so crop_gt was relabelled
     to "tomato" after visual inspection. Frame ids still read pepper_NNN
     because the filenames were left alone; only the label moved.

  2. Cross-model agreement on stage. There are no stage labels at all, so
     agreement between independently-trained models is the best proxy we have.
     Frames where models disagree are the ones worth labeling first.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CSV_OUT = RESULTS / "all_results.csv"
MD_OUT = RESULTS / "SUMMARY.md"

# crop_gt values, and the clip each came from. pepper.mp4 -> tomato: see above.
CROPS = ("tomato", "broccoli", "cabbage")
SOURCE_VIDEO = {"tomato": "pepper.mp4", "broccoli": "broccoli.mp4",
                "cabbage": "cabbage.mp4"}

FIELDS = ["model", "model_repo", "frame_id", "crop_gt", "timestamp_s",
          "crop_type", "stage", "confidence", "rationale",
          "parse_ok", "schema_ok", "latency_s", "error", "raw_output"]


def load_all() -> list[dict]:
    rows = []
    for jsonl in sorted(RESULTS.glob("*/results.jsonl")):
        with jsonl.open() as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def dist(values) -> str:
    c = Counter(v for v in values if v is not None)
    return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"


def main() -> None:
    rows = load_all()
    if not rows:
        raise SystemExit("no results found - run scripts/run_vlm.py first")

    RESULTS.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out = ["# VLM baseline results", ""]
    out.append(f"Frames per model: {len(next(iter(by_model.values())))} "
               f"| Models: {len(by_model)} | Total calls: {len(rows)}")
    out.append("")
    out.append("All models saw an identical frame set and an identical prompt.")
    out.append("")

    # ---- per-model table
    out.append("## Per-model")
    out.append("")
    out.append("| Model | Frames | Parse OK | Schema OK | Errors | Mean lat | p95 lat | Total |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m, rs in sorted(by_model.items()):
        lats = sorted(r["latency_s"] for r in rs)
        p95 = lats[min(int(0.95 * len(lats)), len(lats) - 1)]
        out.append(
            f"| `{m}` | {len(rs)} | {pct(sum(r['parse_ok'] for r in rs), len(rs))} "
            f"| {pct(sum(r['schema_ok'] for r in rs), len(rs))} "
            f"| {sum(1 for r in rs if r['error'])} "
            f"| {mean(lats):.2f}s | {p95:.2f}s | {sum(lats) / 60:.1f} min |")
    out.append("")

    # ---- crop identification vs filename
    out.append("## Crop identification vs. filename")
    out.append("")
    out.append("| Model | Overall | " + " | ".join(
        f"`{SOURCE_VIDEO[c]}` says" for c in CROPS) + " |")
    out.append("|---|---|---|---|---|")
    for m, rs in sorted(by_model.items()):
        hit = sum(1 for r in rs if r["crop_type"] == r["crop_gt"])
        cells = []
        for crop in CROPS:
            cells.append(dist(r["crop_type"] for r in rs if r["crop_gt"] == crop))
        out.append(f"| `{m}` | {pct(hit, len(rs))} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- stage distribution
    out.append("## Stage distribution per crop")
    out.append("")
    out.append("| Model | " + " | ".join(CROPS) + " |")
    out.append("|---|---|---|---|")
    for m, rs in sorted(by_model.items()):
        cells = [dist(r["stage"] for r in rs if r["crop_gt"] == c)
                 for c in CROPS]
        out.append(f"| `{m}` | " + " | ".join(cells) + " |")
    out.append("")

    # ---- agreement
    out.append("## Cross-model agreement")
    out.append("")
    out.append("No stage ground truth exists, so agreement between independently")
    out.append("trained models is the strongest available signal. Disagreement marks")
    out.append("the frames worth labeling first.")
    out.append("")
    idx = {m: {r["frame_id"]: r for r in rs} for m, rs in by_model.items()}
    out.append("| Pair | Crop agreement | Stage agreement |")
    out.append("|---|---|---|")
    for a, b in combinations(sorted(idx), 2):
        shared = set(idx[a]) & set(idx[b])
        if not shared:
            continue
        ca = sum(1 for f in shared if idx[a][f]["crop_type"] == idx[b][f]["crop_type"])
        sa = sum(1 for f in shared if idx[a][f]["stage"] == idx[b][f]["stage"])
        out.append(f"| `{a}` vs `{b}` | {pct(ca, len(shared))} | {pct(sa, len(shared))} |")
    out.append("")

    # unanimous stage
    if len(idx) > 1:
        frames = set.intersection(*(set(v) for v in idx.values()))
        unanimous = sum(1 for f in frames
                        if len({idx[m][f]["stage"] for m in idx}) == 1)
        out.append(f"All {len(idx)} models agree on stage for "
                   f"**{unanimous}/{len(frames)}** frames ({pct(unanimous, len(frames))}).")
        out.append("")

    MD_OUT.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {CSV_OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
