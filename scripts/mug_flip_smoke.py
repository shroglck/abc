"""GPU smoke test for the mug-flip task: MJWarp compile, render, metric.

Runs no policy -- the arms hold the home pose -- so it answers only "does the new
scene build, render, and score under the real eval stack?". Needs a GPU (MJWarp).

    CUDA_VISIBLE_DEVICES=0 uv run scripts/mug_flip_smoke.py --out-dir outputs/mug_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from abc_minimal.config import MugFlipSimConfig
from abc_minimal.eval_policy import SimEnv, video_frame

CAMERAS = ("top", "left", "right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/mug_flip_smoke")
    ap.add_argument("--mug-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = MugFlipSimConfig(mug_count=args.mug_count)
    env = SimEnv(
        height=168,
        width=224,
        camera_keys=CAMERAS,
        prompt="turn the mug right side up",
        scene=scene,
        task="mug_flip",
    )
    try:
        obs = env.reset(seed=args.seed)
        print("reset ok; images", {k: v.shape for k, v in obs["images"].items()})
        print("randomization:", env.randomization)

        import imageio.v2 as imageio

        writer = imageio.get_writer(str(out_dir / "smoke.mp4"), fps=30, macro_block_size=1)
        writer.append_data(video_frame(obs["images"], CAMERAS))

        # Hold the home pose: the mug must stay put and must NOT score.
        hold = np.asarray(scene.init_q, dtype=np.float32)
        for _ in range(args.steps):
            env.step_one(hold)
            metric = env.evaluate()
            writer.append_data(video_frame(env.render_cameras(), CAMERAS))
        print(
            "held home:",
            {k: metric[k] for k in ("num_solved", "num_active", "success", "mug_up_z", "mug_height_err")},
        )
        assert metric["num_solved"] == 0, "an untouched upside-down mug scored"

        # Oracle: place every mug upright and confirm the metric fires through the
        # same MJWarp qpos path the rollout reads.
        for idx, scale in enumerate(env.mug_scales):
            env._set_freejoint(
                f"mug_{idx}_joint",
                [
                    float(env.randomization.mug_states[f"mug_{idx}_joint"]["pos"][0]),
                    float(env.randomization.mug_states[f"mug_{idx}_joint"]["pos"][1]),
                    float(scene.table_z + scene.mug_half_height * float(scale)),
                ],
                [1.0, 0.0, 0.0, 0.0],
            )
        env.sim.load_state()
        env.sim.forward()
        for _ in range(scene.upright_hold_evals * 2):
            env.step_one(hold)
            metric = env.evaluate()
            writer.append_data(video_frame(env.render_cameras(), CAMERAS))
        writer.close()
        print(
            "oracle upright:",
            {k: metric[k] for k in ("num_solved", "num_active", "success", "ever_success", "mug_up_z")},
        )
        assert metric["ever_success"], "upright mugs did not score"
        print(f"OK -- wrote {out_dir / 'smoke.mp4'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
