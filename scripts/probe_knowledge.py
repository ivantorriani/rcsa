#!/usr/bin/env python3
"""Probe whether a model *knows* broccoli vs cabbage, separately from whether it *sees* it.

Two failure modes look identical in the benchmark but need opposite fixes:

  A. The model lacks the concept. It cannot describe how broccoli differs from
     cabbage even with no image in front of it. Prompting will not rescue this;
     it needs reference images or a fine-tune.

  B. The model has the concept but does not apply it to an overhead field shot.
     It can recite the difference, yet still answers "cabbage" on a broccoli
     frame. This is a grounding failure, and a prompt carrying explicit visual
     discriminators is exactly the right lever.

So we ask three things:
  knowledge  - no image: how do you tell them apart from above?
  describe   - a real broccoli frame, but forbidden from naming the crop, so it
               has to commit to observable features before it can hide behind a label
  forced     - the same frame, name the crop, and say which feature decided it
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "probes.jsonl"

MODELS = {
    "qwen3.5-4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "gemma4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "molmo2-8b":  "mlx-community/Molmo2-8B-4bit",
}

KNOWLEDGE = (
    "In an overhead photograph of a vegetable field, how do you tell a broccoli "
    "plant from a cabbage plant when neither has formed a harvestable head yet? "
    "Answer only about what is visible: leaf shape, leaf margin, leaf posture, "
    "colour, and how the leaves are arranged around the centre. Be specific and "
    "concise."
)

DESCRIBE = (
    "Describe the main plant in this overhead photograph using only observable "
    "detail: leaf shape, leaf margin (smooth or wavy), leaf posture (upright, "
    "spreading, or cupping inward), colour, midrib visibility, and whether any "
    "dense head has formed at the centre. "
    "Do NOT name the crop. Do not guess what vegetable it is. Two or three sentences."
)

FORCED = (
    "Is the main plant in this overhead photograph broccoli or cabbage? "
    "Name it, then state the single visual feature that decided your answer. "
    "One sentence."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--frames", nargs="+",
                    default=["broccoli_011", "cabbage_011"],
                    help="frames used for the describe/forced probes")
    ap.add_argument("--max-tokens", type=int, default=260)
    args = ap.parse_args()

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a")

    for key in args.models:
        repo = MODELS[key]
        print(f"\n{'=' * 72}\n{key}\n{'=' * 72}", flush=True)
        model, processor = load(repo)
        config = model.config

        def ask(prompt: str, images: list[str], label: str, frame: str | None):
            t = time.time()
            try:
                formatted = apply_chat_template(processor, config, prompt,
                                                num_images=len(images))
                res = generate(model, processor, formatted, images,
                               max_tokens=args.max_tokens, temperature=0.0,
                               verbose=False)
                text = (res.text if hasattr(res, "text") else str(res)).strip()
                err = None
            except Exception as exc:
                text, err = "", f"{type(exc).__name__}: {exc}"
            rec = {"model": key, "probe": label, "frame": frame,
                   "latency_s": round(time.time() - t, 2),
                   "error": err, "response": text}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"\n--- [{label}] {frame or 'no image'}  ({rec['latency_s']}s)")
            print(text if text else f"ERROR: {err}", flush=True)

        # A: latent knowledge, no image at all
        ask(KNOWLEDGE, [], "knowledge", None)

        # B/C: grounded, on real frames
        for fid in args.frames:
            img = str(REPO / "data" / "frames" / fid.rsplit("_", 1)[0] / f"{fid}.jpg")
            ask(DESCRIBE, [img], "describe", fid)
            ask(FORCED, [img], "forced", fid)

        del model, processor

    fh.close()
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
