"""Render high-res frames of ladder rungs for the paper figure (GPU: MJWarp renderer).

Writes <out>/<rung>/{top,left,right}.png at the requested resolution, plus meta.json with
the prompt. Camera intrinsics are the eval's (fovy 58 top cam), only the pixel count grows.

    PYTHONPATH=. ABC_SCENE_XML=assets/put_bottles/put_bottle_4light.xml \
      .venv/bin/python scripts/ladder_figure_render.py --out outputs/ladder_figure --height 672 --width 896
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from abc_minimal.config import LadderSimConfig
from abc_minimal.eval_policy import SimEnv
from abc_minimal.ladder import get_task

CAMERAS = ("top", "left", "right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", required=True, help="comma-separated rung[:seed] entries")
    ap.add_argument("--out", default="outputs/ladder_figure")
    ap.add_argument("--height", type=int, default=672)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--settle", type=int, default=20, help="held control steps before the shot")
    args = ap.parse_args()
    out = Path(args.out)
    scene = LadderSimConfig()
    hold = np.asarray(scene.init_q, dtype=np.float32)
    for entry in args.rungs.split(","):
        name, _, seed = entry.partition(":")
        seed = int(seed) if seed else 20260511
        task = get_task(name)
        env = SimEnv(height=args.height, width=args.width, camera_keys=CAMERAS, prompt=task.prompt,
                     scene=scene, task="ladder", ladder_task=name)
        try:
            obs = env.reset(seed=seed)
            for _ in range(args.settle):
                env.step_one(hold)
            images = env.render_cameras()
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            for cam in CAMERAS:
                imageio.imwrite(d / f"{cam}.png", images[cam].transpose(1, 2, 0))
            (d / "meta.json").write_text(json.dumps({"rung": name, "seed": seed, "prompt": obs["prompt"],
                                                     "goal": task.goal, "n_objects": len(task.objects)}, indent=1))
            print(f"[render] {name:28s} seed={seed} prompt={obs['prompt']!r}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
