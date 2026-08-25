"""Scripted per-hand command probe (no LAP model / no server).

Drives the put-bottles sim through the SAME language-action -> bridge IK path the
AR head uses (``parse_bimanual_langact`` -> ``LAPSimBridge.joint_chunk_from_eef``
-> interpolated joint targets), but with HAND-AUTHORED per-hand commands instead
of model output. Purpose: verify the RIGHT arm is actually controllable through
our pipeline (mechanics/IK), independent of whether the policy chooses to move it.

Each "chunk" applies a distinct command to each arm and logs how far each arm's
six joints actually moved, so an immobile right arm shows up as ~0 joint delta.

Run inside an srun GPU allocation (MuJoCo-Warp needs a GPU):

    python -m abc_minimal.scripted_arms_probe --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import PutBottlesEnv, scene_xml, video_frame
from abc_minimal.lap_bridge import LAPSimBridge
from abc_minimal.lap_langact import parse_bimanual_langact


# Distinct, deliberately OPPOSITE commands per arm so left/right motion can't be
# confused. Gripper named on both arms each step. ~moderate magnitudes.
COMMANDS = [
    ("Left arm: move up 10 cm, open gripper.  Right arm: move down 8 cm, close gripper", ),
    ("Left arm: move right 10 cm, close gripper. Right arm: move left 10 cm, open gripper", ),
    ("Left arm: move forward 8 cm, tilt left 20 degrees, open gripper. "
     "Right arm: move back 8 cm, tilt right 20 degrees, close gripper", ),
    ("Left arm: move down 8 cm, rotate clockwise 30 degrees, close gripper. "
     "Right arm: move up 10 cm, rotate counterclockwise 30 degrees, open gripper", ),
]


def interp_targets(start14: np.ndarray, target14: np.ndarray, horizon: int) -> np.ndarray:
    return np.linspace(start14, target14, horizon + 1)[1:].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--execute-chunk-dim", type=int, default=15)
    ap.add_argument("--repeat", type=int, default=2, help="times to cycle the command list")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimEvalConfig()

    bridge_model = mujoco.MjModel.from_xml_string(
        scene_xml(cfg.scene, np.ones(cfg.scene.bottle_count), 1.0)
    )
    bridge = LAPSimBridge(
        bridge_model,
        gripper_ctrl_max=cfg.scene.gripper_ctrl_max,
        rotation_delta_in_base=cfg.lap_rotation_delta_in_base,
        align_eef_rot=cfg.lap_align_eef_rot,
    )
    env = PutBottlesEnv(
        height=cfg.camera_height,
        width=cfg.camera_width,
        camera_keys=cfg.model.camera_keys,
        prompt=cfg.prompt,
        scene=cfg.scene,
        gpu_id=args.gpu_id,
    )

    obs = env.reset(seed=args.seed)
    import imageio.v2 as imageio

    video_path = out / "scripted_arms.mp4"
    video = imageio.get_writer(str(video_path), fps=cfg.video_fps, macro_block_size=1)
    video.append_data(video_frame(obs["images"], cfg.model.camera_keys))

    horizon = max(int(args.execute_chunk_dim), 1)
    print(f"out={out}  horizon={horizon}  repeat={args.repeat}", flush=True)

    cmds = list(COMMANDS) * args.repeat
    for ci, (reasoning,) in enumerate(cmds):
        state14 = np.asarray(obs["state"], dtype=np.float32)
        state20 = bridge.state20_from_joints(state14)
        delta14 = parse_bimanual_langact(reasoning, state20)
        target14 = bridge.joint_chunk_from_eef(delta14[None], state14)[0]
        traj = interp_targets(state14, target14, horizon)

        for action in traj:
            env.step_one(action)
            video.append_data(video_frame(env.render_cameras(), cfg.model.camera_keys))

        end14 = np.asarray(env.obs()["state"], dtype=np.float32)
        # Per-arm joint movement (6 joints each), gripper excluded.
        l_move = np.abs(end14[0:6] - state14[0:6])
        r_move = np.abs(end14[7:13] - state14[7:13])
        print(
            f"chunk {ci:02d}: L_joint|Δ|sum={l_move.sum():.4f} max={l_move.max():.4f} | "
            f"R_joint|Δ|sum={r_move.sum():.4f} max={r_move.max():.4f}",
            flush=True,
        )
        print(f"    cmd: {reasoning}", flush=True)
        print(
            f"    L_eef_delta={np.round(delta14[0:7], 3).tolist()}\n"
            f"    R_eef_delta={np.round(delta14[7:14], 3).tolist()}",
            flush=True,
        )
        obs = env.obs()

    video.close()
    print(f"DONE video={video_path}", flush=True)


if __name__ == "__main__":
    main()
