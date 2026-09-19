"""Real-shard controller oracle: does the bridge reproduce the DATASET's own targets?

The synthetic round-trip in `lap_convention_probe` only proves the bridge inverts its
own encode. This proves something stronger and more useful: that a chunk built exactly
as ego-lap builds it, from real abc_yam shard rows, decodes back to the poses the
dataset actually recorded. If this passes, a bad rollout is a policy or a data problem
and NOT a controller problem -- which is the distinction every previous debugging round
kept failing to make.

Pipeline under test, end to end:
  1. real window        action[t:t+H+1], joint_pos_action[t:t+H+1] from the TFDS shard
  2. training target    exactly `chunk_actions` (action_windows.py:190-215)
  3. normalize          bounds_q99 with the checkpoint's own actions_yam stats
  4. unnormalize        the inverse the serving output transform applies
  5. decode             bridge.joint_chunk_from_eef, anchored at joint_pos_action[t]
  6. FK                 every decoded joint target, in the arm base frame
  7. compare            against action[t+1:t+H+1]

Run from the ego-lap venv (needs both tensorflow and mujoco):
    .venv/bin/python -m abc_minimal.lap_shard_oracle
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PASS_POS_M = 1e-3      # 1 mm
PASS_ROT_RAD = 1e-2    # 0.01 rad


def _euler_diff(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """ego-lap's euler_diff: rel with R(a2) @ R(rel) = R(a1), i.e. rel = R2^T R1."""
    from scipy.spatial.transform import Rotation as R
    R1 = R.from_euler("xyz", a1).as_matrix()
    R2 = R.from_euler("xyz", a2).as_matrix()
    return R.from_matrix(np.swapaxes(R2, -1, -2) @ R1).as_euler("xyz")


def build_training_target(action: np.ndarray, H: int) -> np.ndarray:
    """action[t:t+H+1] (H+1, 14) -> the (H, 14) chunk ego-lap trains on."""
    w = action
    return np.concatenate(
        [
            w[1:, 0:3] - w[0:1, 0:3],
            _euler_diff(w[1:, 3:6], np.repeat(w[0:1, 3:6], H, axis=0)),
            w[:-1, 6:7],                      # ABSOLUTE gripper, index k not k+1
            w[1:, 7:10] - w[0:1, 7:10],
            _euler_diff(w[1:, 10:13], np.repeat(w[0:1, 10:13], H, axis=0)),
            w[:-1, 13:14],
        ],
        axis=-1,
    )


def norm_unnorm(chunk: np.ndarray, q01: np.ndarray, q99: np.ndarray):
    """bounds_q99 forward then inverse, exactly as transforms.py:511 and serving do.

    Also returns a per-ROW clipped mask. The forward pass clips to [-1, 1], so any row
    with a dimension outside [q01, q99] cannot survive the round trip -- 13.9% of real
    abc_yam rows, losing up to 1.0 in the worst dim. That loss is a property of the
    TRAINING REPRESENTATION (the model is fit on the clipped target too), not a decode
    error, so the controller verdict below is taken on unclipped rows only and the
    clipped ones are reported separately.
    """
    out_of_range = (chunk < q01) | (chunk > q99)
    scaled = 2.0 * (chunk - q01) / (q99 - q01 + 1e-8) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    recovered = (scaled + 1.0) / 2.0 * (q99 - q01 + 1e-8) + q01
    return recovered, out_of_range.any(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds-dir", default="/n/fs/prism-data/shresth/abc_sample_tfds")
    ap.add_argument("--norm-stats",
                    default="/n/fs/prism-data/shresth/ego-lap/checkpoint_abc_yam_29k/29000/assets/oxe/norm_stats.json")
    ap.add_argument("--scene", default="/n/fs/prism-data/shresth/abc/assets/put_bottles/put_bottle.xml")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import mujoco
    import tensorflow_datasets as tfds
    from scipy.spatial.transform import Rotation as R

    sys.path.insert(0, "/n/fs/prism-data/shresth/ego-lap/rlds_dataset_builder")
    from abc_yam_dataset.abc_yam_dataset_dataset_builder import AbcYamDataset
    from abc_minimal.lap_bridge import LAPSimBridge

    stats = json.load(open(args.norm_stats))["norm_stats"]["actions_yam"]
    q01, q99 = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
    print(f"norm stats: {args.norm_stats}")
    print(f"  actions_yam  num_trajectories={stats.get('num_trajectories')} "
          f"num_transitions={stats.get('num_transitions')}")

    model = mujoco.MjModel.from_xml_path(args.scene)
    # abc_yam's convention: raw grasp_site FK, no i2rt axis flip, post-multiplied
    # rotation delta, fixed-origin cumulative decoding, commanded-pose anchor.
    bridge = LAPSimBridge(model, gripper_ctrl_max=0.0475, eef_link="grasp_site",
                          align_eef_rot=False, rotation_delta_in_base=False,
                          binarize_gripper=False, anchor_on_command=True)

    b = AbcYamDataset(data_dir=args.tfds_dir)
    dec = {"steps": {"observation": {k: tfds.decode.SkipDecoding()
                                     for k in ("top_image", "left_image", "right_image")}}}
    H = args.horizon
    pos_errs, rot_errs, grip_errs, clip_flags, n_win = [], [], [], [], 0

    for ep in tfds.as_numpy(b.as_dataset(split="train", decoders=dec).take(args.episodes)):
        steps = list(ep["steps"])
        A = np.array([s["action"] for s in steps], np.float64)              # (T,14) absolute
        JA = np.array([s["joint_pos_action"] for s in steps], np.float64)   # (T,12) commanded joints
        T = len(A)
        if T < H + 2:
            continue
        for t in range(0, T - H - 1, max(1, (T - H - 1) // max(1, args.windows // args.episodes))):
            target = build_training_target(A[t:t + H + 1], H)
            decoded_chunk, clipped = norm_unnorm(target, q01, q99)

            anchor14 = np.concatenate([JA[t, :6], A[t, 6:7], JA[t, 6:12], A[t, 13:14]])
            bridge.reset()
            joints = bridge.joint_chunk_from_eef(decoded_chunk.astype(np.float32), anchor14)

            for k in range(H):
                q14 = np.concatenate([joints[k, :6], [0.0], joints[k, 7:13], [0.0]])
                bridge.data.qpos[:] = 0.0
                bridge._set_arm_joints("left", q14[0:6])
                bridge._set_arm_joints("right", q14[7:13])
                mujoco.mj_kinematics(bridge.model, bridge.data)
                for arm, sl_p, sl_r, gi in (("left", slice(0, 3), slice(3, 6), 6),
                                            ("right", slice(7, 10), slice(10, 13), 13)):
                    p, rot = bridge._site_pose_in_base(arm)
                    pos_errs.append(np.linalg.norm(p - A[t + 1 + k, sl_p]))
                    Rt = R.from_euler("xyz", A[t + 1 + k, sl_r]).as_matrix()
                    rot_errs.append(np.linalg.norm(R.from_matrix(rot @ Rt.T).as_rotvec()))
                    grip_errs.append(abs(float(joints[k, gi]) - float(A[t + k, gi])))
                    clip_flags.append(bool(clipped[k]))
            n_win += 1

    pos = np.asarray(pos_errs); rot = np.asarray(rot_errs); gr = np.asarray(grip_errs)
    cl = np.asarray(clip_flags)
    print(f"\nwindows={n_win}  comparisons={pos.size}  H={H}")

    def _block(label, m):
        if not m.any():
            print(f"  {label}: none"); return
        print(f"  {label} (n={m.sum()}, {100*m.mean():.1f}%)")
        print(f"    position  median {np.median(pos[m])*1000:9.4f} mm   q99 {np.quantile(pos[m],.99)*1000:9.3f} mm"
              f"   max {pos[m].max()*1000:9.3f} mm")
        print(f"    rotation  median {np.median(rot[m]):9.6f} rad  q99 {np.quantile(rot[m],.99):9.5f} rad"
              f"  max {rot[m].max():9.5f} rad")

    _block("UNCLIPPED rows -- the controller verdict", ~cl)
    _block("CLIPPED rows -- bounds_q99 loss, not a decode error", cl)
    print(f"  gripper row err  max {gr.max():.3e}   (row is passed through absolute)")
    bridge.report_ik_diagnostics()

    clean = ~cl
    ok = (pos[clean].max() < PASS_POS_M and rot[clean].max() < PASS_ROT_RAD
          and gr[clean].max() < 1e-4 and bridge.ik_stats["pos_over_audit"] == 0)
    print(f"\n{'PASS' if ok else 'FAIL'}: on unclipped rows, pos<{PASS_POS_M*1000:.0f}mm "
          f"rot<{PASS_ROT_RAD}rad, gripper exact, no IK audit violations")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
