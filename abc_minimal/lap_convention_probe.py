"""Offline round-trip arbiter for the LAP state/action conventions (no server, CPU).

Numerically proves that ``LAPSimBridge``'s state encode and action decode invert
ego-lap's *exact* training encode, and — decisively — which
``rotation_delta_in_base`` value is correct for the served ``ckeckpoint_bimanip``.

The test is self-contained: it fabricates a short bimanual joint trajectory
(H+1 frames, a small random walk within joint limits), runs it through a faithful
replica of ego-lap's encode to produce the (H, 14) EEF action chunk the server
would emit for that trajectory, decodes that chunk back with
``bridge.joint_chunk_from_eef``, and checks that the decoded joint targets
reproduce the original grasp-site poses (pos < 1e-3 m, rot < 1e-2 rad).

Encode replica (mirrors ego-lap ``add_ee_pose_to_trajectory`` + ``chunk_actions``):
  * per frame, FK grasp_site -> raw base-frame rot ``rot_b`` -> user frame
    ``R_user = R_af^T @ R_cb^T @ rot_b @ R_af`` -> extrinsic-xyz euler + xyz + grip;
  * action[t] (from chunk start, t=1..H): per arm
    ``[pos[t]-pos[0], euler_diff(euler[t], euler[0]), grip[t]]`` where
    ``euler_diff(et,e0) = mat_to_rpy(R(e0)^T @ R(et))`` — a BODY-LOCAL relative
    rotation (post-multiply), matching training. This is what decides
    ``rotation_delta_in_base``.

The chunk is decoded under BOTH ``rotation_delta_in_base ∈ {False, True}``.
Expected: ``False`` round-trips (~0 error); ``True`` fails on rotation — i.e. the
eval must use ``False`` (the current hardcoded ``True`` is the wrong-motion bug).

CPU only (no GPU / MuJoCo-Warp, no policy server):

    python -m abc_minimal.lap_convention_probe
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import scene_xml
from abc_minimal.lap_bridge import LAPSimBridge, rpy_to_mat
from abc_minimal.lap_langact import mat_to_rpy

H = 16  # action horizon (chunk length); trajectory has H+1 frames.


def _fk_user_pose(bridge: LAPSimBridge, frame14: np.ndarray, arm: str):
    """FK one 14-D joint frame -> (pos_b, rot_b_raw, R_user) for ``arm``."""
    bridge.data.qpos[:] = 0.0
    bridge._set_arm_joints("left", frame14[0:6])
    bridge._set_arm_joints("right", frame14[7:13])
    mujoco.mj_kinematics(bridge.model, bridge.data)
    pos_b, rot_b = bridge._site_pose_in_base(arm)
    R_user = bridge.R_af.T @ bridge.R_cb.T @ rot_b @ bridge.R_af
    return pos_b, rot_b, R_user


def _encode_chunk(bridge: LAPSimBridge, traj: np.ndarray) -> np.ndarray:
    """Replicate ego-lap's from-chunk-start encode -> (H, 14) EEF action chunk.

    ``traj`` is (H+1, 14) joint frames; frame 0 is the chunk start.
    """
    n = traj.shape[0] - 1
    chunk = np.zeros((n, 14), dtype=np.float64)
    # Chunk-start (frame 0) user pose + euler, per arm.
    start = {}
    for arm in ("left", "right"):
        pos0, _rot0, R_user0 = _fk_user_pose(bridge, traj[0], arm)
        start[arm] = (pos0, mat_to_rpy(R_user0))
    for t in range(1, n + 1):
        for arm, (aslot, gsrc) in (
            ("left", (slice(0, 7), 6)),
            ("right", (slice(7, 14), 13)),
        ):
            pos_t, _rot_t, R_user_t = _fk_user_pose(bridge, traj[t], arm)
            pos0, euler0 = start[arm]
            euler_t = mat_to_rpy(R_user_t)
            # Body-local relative rotation (post-multiply): R(e0)^T @ R(et).
            drpy = mat_to_rpy(rpy_to_mat(euler0).T @ rpy_to_mat(euler_t))
            grip = traj[t][gsrc]
            chunk[t - 1, aslot] = np.concatenate([pos_t - pos0, drpy, [grip]])
    return chunk


def _grasp_pose_world(bridge: LAPSimBridge, frame14: np.ndarray, arm: str):
    """FK a joint frame -> world grasp-site (pos, rotmat) for pose comparison."""
    bridge.data.qpos[:] = 0.0
    bridge._set_arm_joints("left", frame14[0:6])
    bridge._set_arm_joints("right", frame14[7:13])
    mujoco.mj_kinematics(bridge.model, bridge.data)
    ak = bridge.arms[arm]
    pos = bridge.data.site_xpos[ak.site_id].copy()
    mat = bridge.data.site_xmat[ak.site_id].reshape(3, 3).copy()
    return pos, mat


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _sample_traj(bridge: LAPSimBridge, rng: np.random.Generator) -> np.ndarray:
    """(H+1, 14) small random-walk joint trajectory within limits."""
    lo = np.zeros(14)
    hi = np.zeros(14)
    for arm, base in (("left", 0), ("right", 7)):
        ak = bridge.arms[arm]
        lo[base : base + 6] = ak.lo
        hi[base : base + 6] = ak.hi
    # Non-singular seed: 35% into each joint's range; grippers open (1.0).
    seed = lo + 0.35 * (hi - lo)
    seed[6] = seed[13] = 1.0
    traj = [seed.copy()]
    for _ in range(H):
        step = np.zeros(14)
        for base in (0, 7):
            step[base : base + 6] = rng.uniform(-0.04, 0.04, size=6)
        nxt = traj[-1] + step
        for arm, base in (("left", 0), ("right", 7)):
            ak = bridge.arms[arm]
            nxt[base : base + 6] = np.clip(nxt[base : base + 6], ak.lo, ak.hi)
        nxt[6] = nxt[13] = 1.0
        traj.append(nxt)
    return np.asarray(traj)


def _run(bridge: LAPSimBridge, traj: np.ndarray, rot_in_base: bool):
    """Encode -> decode(rot_in_base) -> return (max_pos_err, max_rot_err)."""
    bridge.rotation_delta_in_base = rot_in_base
    chunk = _encode_chunk(bridge, traj)
    joints = bridge.joint_chunk_from_eef(chunk, traj[0])  # (H, 14)
    max_pos = 0.0
    max_rot = 0.0
    for t in range(H):
        for arm in ("left", "right"):
            gt_pos, gt_mat = _grasp_pose_world(bridge, traj[t + 1], arm)
            dc_pos, dc_mat = _grasp_pose_world(bridge, joints[t], arm)
            max_pos = max(max_pos, float(np.linalg.norm(gt_pos - dc_pos)))
            max_rot = max(max_rot, _rot_angle(gt_mat, dc_mat))
    return max_pos, max_rot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--pos-tol", type=float, default=1e-3)
    ap.add_argument("--rot-tol", type=float, default=1e-2)
    args = ap.parse_args()

    cfg = SimEvalConfig()
    model = mujoco.MjModel.from_xml_string(
        scene_xml(cfg.scene, np.ones(cfg.scene.bottle_count), 1.0)
    )
    bridge = LAPSimBridge(
        model,
        gripper_ctrl_max=cfg.scene.gripper_ctrl_max,
        align_eef_rot=cfg.lap_align_eef_rot,
    )
    rng = np.random.default_rng(args.seed)
    traj = _sample_traj(bridge, rng)

    # --- state20 self-check: encode-implied state == bridge.state20_from_joints
    state20 = bridge.state20_from_joints(traj[0])
    st_err = 0.0
    for arm, off in (("left", 0), ("right", 10)):
        pos_b, _rot_b, R_user = _fk_user_pose(bridge, traj[0], arm)
        expect = np.concatenate([pos_b, R_user[:, 0], R_user[:, 1]])
        st_err = max(st_err, float(np.max(np.abs(state20[off : off + 9] - expect))))
    print(f"state20 rot6d/pos self-check: max|Δ| = {st_err:.2e}")

    print(f"\naction round-trip (H={H}, tol pos<{args.pos_tol}, rot<{args.rot_tol}):")
    results = {}
    for rib in (False, True):
        pos_err, rot_err = _run(bridge, traj, rib)
        results[rib] = (pos_err, rot_err)
        ok = pos_err < args.pos_tol and rot_err < args.rot_tol
        print(
            f"  rotation_delta_in_base={str(rib):5s}: "
            f"max_pos_err={pos_err:.2e} m  max_rot_err={rot_err:.2e} rad  "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

    false_ok = (
        results[False][0] < args.pos_tol and results[False][1] < args.rot_tol
    )
    true_ok = results[True][0] < args.pos_tol and results[True][1] < args.rot_tol
    print()
    if false_ok and not true_ok:
        print("VERDICT: rotation_delta_in_base=False is correct (confirms suspect #2).")
    elif false_ok and true_ok:
        print("VERDICT: both pass — trajectory rotations too small to discriminate; "
              "increase random-walk magnitude.")
    else:
        print("VERDICT: unexpected — False did not round-trip; revisit encode/euler order.")


if __name__ == "__main__":
    main()
