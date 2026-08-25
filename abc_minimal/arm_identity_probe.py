"""Arm-identity / camera-identity probe for the LAP bridge (GPU, no server).

Verifies suspect #3 (arm or camera transposition) directly, at the mechanics
level, without a policy server:

  * commands a PURE left-arm EEF delta (left slot nonzero, right slot zeroed with
    the right gripper held) through ``LAPSimBridge.joint_chunk_from_eef``, steps
    the sim, and confirms ONLY the sim ``left_arm`` (y=+0.31) joints move; then
    the mirror for the right arm;
  * dumps one frame from each task camera (``top``/``left``/``right``) so the
    wrist-vs-base camera identity can be eyeballed (left wrist should track the
    left arm, right wrist the right arm, top = fixed base view).

This isolates the bridge's action-slot -> physical-arm mapping (and the camera
role mapping) from the policy and from language parsing.

Run directly on a GPU box (MuJoCo-Warp needs a GPU); use the venv interpreter,
NOT ``uv run`` (which re-syncs the env and hangs):

    .venv/bin/python -m abc_minimal.arm_identity_probe --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import PutBottlesEnv, scene_xml, video_frame
from abc_minimal.lap_bridge import LAPSimBridge


def _pure_arm_action(state20: np.ndarray, arm: str, dxyz: np.ndarray) -> np.ndarray:
    """14-D EEF action moving only ``arm`` by ``dxyz`` (rot 0), grippers held."""
    a = np.zeros(14, dtype=np.float32)
    lgrip, rgrip = float(state20[9]), float(state20[19])
    a[6], a[13] = lgrip, rgrip  # hold both grippers absolute
    if arm == "left":
        a[0:3] = dxyz
    else:
        a[7:10] = dxyz
    return a


def _drive(env, bridge, obs, action14, horizon):
    """Decode one EEF action to joint targets, interpolate, step; return end obs."""
    state14 = np.asarray(obs["state"], dtype=np.float32)
    target14 = bridge.joint_chunk_from_eef(action14[None], state14)[0]
    traj = np.linspace(state14, target14, horizon + 1)[1:].astype(np.float32)
    for a in traj:
        env.step_one(a)
    return env.obs(), state14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--dist", type=float, default=0.08, help="EEF move distance (m)")
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

    # Dump one frame per camera at reset for camera-identity inspection.
    for cam in cfg.model.camera_keys:
        img = np.asarray(obs["images"][cam])
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(str(out / f"cam_{cam}.png"), img.astype(np.uint8))
    print(f"cameras dumped: {list(cfg.model.camera_keys)}", flush=True)

    dxyz = np.array([0.0, 0.0, args.dist], dtype=np.float32)  # move up
    for arm in ("left", "right"):
        obs = env.reset(seed=args.seed)
        state20 = bridge.state20_from_joints(np.asarray(obs["state"], dtype=np.float32))
        action = _pure_arm_action(state20, arm, dxyz)
        end_obs, start14 = _drive(env, bridge, obs, action, args.horizon)
        end14 = np.asarray(end_obs["state"], dtype=np.float32)
        l_move = np.abs(end14[0:6] - start14[0:6])
        r_move = np.abs(end14[7:13] - start14[7:13])
        moved = "left" if l_move.sum() > r_move.sum() else "right"
        ok = "OK" if moved == arm else "*** MISMATCH ***"
        print(
            f"commanded={arm:5s} -> moved={moved:5s} {ok} | "
            f"L|Δ|sum={l_move.sum():.4f} R|Δ|sum={r_move.sum():.4f}",
            flush=True,
        )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
