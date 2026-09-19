"""Render frame-0 observations for a scene XML, with no policy server.

Reuses eval_policy's SimEnv so the pixels are exactly what a rollout would feed the
model: same MJWarp ray tracer, same cameras, same per-world randomization draw.
"""
import argparse, os, sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260511, 20260513, 20260516])
    ap.add_argument("--table-bounds", type=float, nargs=4, default=None)
    ap.add_argument("--bin-xy-range", type=float, nargs=4, default=None)
    ap.add_argument("--bottle-count", type=int, default=1)
    args = ap.parse_args()

    # Must precede the import: SCENE_XML is resolved at module load.
    os.environ["ABC_SCENE_XML"] = args.scene_xml
    from abc_minimal.config import SimEvalConfig
    from abc_minimal.eval_policy import SimEnv
    from PIL import Image

    cfg = SimEvalConfig()
    scene = cfg.scene
    scene.bottle_count = args.bottle_count
    if args.table_bounds:
        scene.table_bounds = tuple(args.table_bounds)
    if args.bin_xy_range:
        scene.bin_xy_range = tuple(args.bin_xy_range)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = SimEnv(height=cfg.camera_height, width=cfg.camera_width,
                 camera_keys=("top", "left", "right"), prompt=cfg.prompt, scene=scene, gpu_id=0)
    try:
        for seed in args.seeds:
            env.reset(seed)
            imgs = env.render_cameras()
            for name, chw in imgs.items():
                hwc = np.transpose(chw, (1, 2, 0))
                Image.fromarray(hwc.astype(np.uint8)).save(out / f"seed{seed}_{name}.png")
                print(f"seed={seed} cam={name:22s} mean={hwc.mean():6.1f} "
                      f"p05={np.percentile(hwc,5):6.1f} p95={np.percentile(hwc,95):6.1f} "
                      f"frac_dark={(hwc.mean(2)<25).mean():.2f}")
            b = env.randomization.bottle_states
            pos = list(b.values())[0]["pos"] if b else None
            if pos:
                print(f"  bottle=({pos[0]:.3f},{pos[1]:.3f})")
    finally:
        env.close()
    print("wrote ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
