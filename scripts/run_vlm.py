#!/usr/bin/env python3
"""Run vision-language models over the sampled field frames and log structured results.

Every model sees the identical frame set (from data/frames/manifest.csv) and the
identical prompt (prompts/stage_prompt.txt), so results are directly comparable.

Design notes:
  - Results are appended to JSONL as each frame completes. A crash or Ctrl-C
    partway through a slow MPS run must not lose finished work; re-running skips
    frames already recorded for that model.
  - A model that returns unparseable text is recorded with parse_ok=false rather
    than aborting the sweep. Parse rate is itself one of the things we're measuring.
  - Models are loaded one at a time. 16 GB unified memory will not hold two.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "frames" / "manifest.csv"
PROMPT_FILE = REPO / "prompts" / "stage_prompt.txt"
RESULTS = REPO / "results"

MODELS = {
    "qwen3.5-4b":  "mlx-community/Qwen3.5-4B-MLX-4bit",
    "gemma4-e4b":  "mlx-community/gemma-4-e4b-it-4bit",
    "molmo2-8b":   "mlx-community/Molmo2-8B-4bit",
}

VALID_CROP = {"pepper", "tomato", "broccoli", "cabbage", "other", "none"}
VALID_STAGE = {"early", "middle", "harvest"}


# ---------------------------------------------------------------- parsing

def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Models drift from the requested format constantly: markdown fences, a
    sentence of preamble, reasoning before the answer. Rather than demanding
    clean output we scan for the outermost balanced {...} and try that.
    """
    if not text:
        return None

    # Strip ``` fences, keeping the contents.
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # malformed; try the next '{'
        start = text.find("{", start + 1)
    return None


def normalize(obj: dict | None) -> dict:
    """Coerce a parsed reply into our schema, flagging anything off-vocabulary."""
    out = {"crop_type": None, "stage": None, "confidence": None,
           "rationale": None, "head_visible": None,
           "parse_ok": False, "schema_ok": False}
    if not obj:
        return out

    out["parse_ok"] = True
    crop = str(obj.get("crop_type", "")).strip().lower()
    stage = str(obj.get("stage", "")).strip().lower()

    out["crop_type"] = crop if crop in VALID_CROP else f"?{crop}"
    out["stage"] = stage if stage in VALID_STAGE else f"?{stage}"

    try:
        c = float(obj.get("confidence"))
        out["confidence"] = min(max(c, 0.0), 1.0)
    except (TypeError, ValueError):
        out["confidence"] = None

    # v2 asks the model to commit to whether a head is really there, so we can
    # measure how often it invents one. v1 has no such field; leave it None.
    hv = obj.get("head_visible")
    if isinstance(hv, str):
        hv = {"true": True, "false": False}.get(hv.strip().lower())
    out["head_visible"] = hv if isinstance(hv, bool) else None

    rat = obj.get("rationale")
    out["rationale"] = str(rat)[:300] if rat is not None else None
    out["schema_ok"] = crop in VALID_CROP and stage in VALID_STAGE
    return out


# ---------------------------------------------------------------- runner

def load_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        sys.exit(f"missing manifest: {MANIFEST}\nRun scripts/extract_frames.py first.")
    with MANIFEST.open() as fh:
        return list(csv.DictReader(fh))


def already_done(path: Path) -> set[str]:
    """Frame ids already recorded, so a resumed run doesn't redo them."""
    if not path.is_file():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["frame_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a hard kill
    return done


def run_model(key: str, repo: str, rows: list[dict], prompt: str,
              max_tokens: int, limit: int | None, resume: bool,
              tag: str = "", prompt_name: str = "") -> None:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # A tagged run is a different experiment, so it gets its own directory and
    # its own model label - otherwise summarize/footage would merge the variants.
    label = f"{key}{tag}"
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    done = already_done(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    todo = [r for r in rows if r["frame_id"] not in done]
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f"[{label}] nothing to do ({len(done)} already recorded)")
        return

    print(f"[{label}] loading {repo} ...", flush=True)
    t0 = time.time()
    model, processor = load(repo)
    config = model.config
    print(f"[{label}] loaded in {time.time() - t0:.1f}s; {len(todo)} frames to go", flush=True)

    with out_path.open("a") as fh:
        for n, row in enumerate(todo, 1):
            img = str(REPO / row["path"])
            rec = {
                "model": label, "model_key": key, "prompt": prompt_name,
                "model_repo": repo,
                "frame_id": row["frame_id"], "crop_gt": row["crop_gt"],
                "timestamp_s": float(row["timestamp_s"]),
            }
            t = time.time()
            try:
                formatted = apply_chat_template(processor, config, prompt, num_images=1)
                result = generate(model, processor, formatted, [img],
                                  max_tokens=max_tokens, temperature=0.0, verbose=False)
                raw = result.text if hasattr(result, "text") else str(result)
                rec["error"] = None
            except Exception as exc:
                raw = ""
                rec["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

            rec["latency_s"] = round(time.time() - t, 3)
            rec.update(normalize(extract_json(raw)))
            rec["raw_output"] = raw[:2000]

            fh.write(json.dumps(rec) + "\n")
            fh.flush()  # survive a hard kill

            flag = "ok " if rec["schema_ok"] else ("parse" if rec["parse_ok"] else "FAIL")
            print(f"[{label}] {n:3d}/{len(todo)} {row['frame_id']:14s} "
                  f"{flag} {str(rec['crop_type']):10s} {str(rec['stage']):8s} "
                  f"{rec['latency_s']:6.2f}s", flush=True)

    print(f"[{label}] done -> {out_path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS), help="which models to run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N frames (for a dry run)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt", type=Path, default=PROMPT_FILE,
                    help="prompt file to use (default: prompts/stage_prompt.txt)")
    ap.add_argument("--tag", default="",
                    help="suffix for the results dir, e.g. __v2, to keep "
                         "prompt variants side by side")
    ap.add_argument("--no-resume", action="store_true",
                    help="discard existing results and start over")
    args = ap.parse_args()

    rows = load_manifest()
    prompt = args.prompt.read_text()
    print(f"{len(rows)} frames in manifest | prompt: {args.prompt.name}\n")

    for key in args.models:
        try:
            run_model(key, MODELS[key], rows, prompt,
                      args.max_tokens, args.limit, not args.no_resume,
                      args.tag, args.prompt.name)
        except Exception:
            # One broken model must not stop the other two.
            print(f"[{key}] ABORTED", file=sys.stderr)
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
