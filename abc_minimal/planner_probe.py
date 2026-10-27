"""Geometry probe for the scripted planner (no policy server).

Run through the ladder launcher with NO_SERVER=1 EVAL_MODULE=abc_minimal.planner_probe so
the scene flags match the evals exactly. Prints, per world: arm base poses, the grasp-site
pose (base and world frames, rotation columns), gripper state, object positions and box
half-sizes; checks `eef_chunk_from_targets` against `joint_chunk_from_eef` + FK; and
executes one "down 5 cm, then +x 5 cm" chunk to compare commanded vs realized motion.
"""

from __future__ import annotations

import mujoco
import numpy as np

from abc_minimal.eval_policy import build_scene_xml
from abc_minimal.branching import privileged_state
from abc_minimal.eval_policy import SimEnv, active_scene, require_mjwarp
from abc_minimal.lap_bridge import LAPSimBridge, rpy_to_mat
from abc_minimal.rl_collect import RLCollectConfig

np.set_printoptions(precision=4, suppress=True)


def make_bridge(config: RLCollectConfig) -> LAPSimBridge:
    return LAPSimBridge(
        mujoco.MjModel.from_xml_string(build_scene_xml(config)),
        gripper_ctrl_max=active_scene(config).gripper_ctrl_max,
        rotation_delta_in_base=config.lap_rotation_delta_in_base,
        align_eef_rot=config.lap_align_eef_rot,
        anchor_on_command=config.lap_anchor_on_command,
        binarize_gripper=config.lap_binarize_gripper,
        eef_link=config.lap_eef_link,
    )


def body_half_sizes(model: mujoco.MjModel, root_body: int) -> list:
    out = []
    for g in range(model.ngeom):
        if int(model.body_rootid[model.geom_bodyid[g]]) == root_body:
            out.append((int(model.geom_type[g]), model.geom_size[g].round(4).tolist()))
    return out[:4]


def main(config: RLCollectConfig) -> None:
    require_mjwarp()
    bridge = make_bridge(config)
    env = SimEnv(height=config.camera_height, width=config.camera_width, camera_keys=config.model.camera_keys,
                 prompt=config.prompt, scene=active_scene(config), task=config.task,
                 ladder_task=config.ladder_task, gpu_id=config.gpu_id)
    print(f"[probe] rung={config.ladder_task} rotation_delta_in_base={bridge.rotation_delta_in_base} "
          f"R_af={np.diag(bridge.R_af)} execute_chunk_dim={config.execute_chunk_dim}")
    try:
        for w in range(min(config.num_worlds, 2)):
            seed = config.seed + w
            env.reset(seed=seed)
            bridge.reset()
            st = env.get_state_vanilla()
            print(f"\n== world {seed}: control_decimation={env.control_decimation} timestep={env.model.opt.timestep}"
                  f" state14 {st}")
            anchors = bridge.anchor_poses(st)
            for arm in ("left", "right"):
                bp, bR = bridge.base_pose_world(arm)
                p, R = anchors[arm]
                print(f"  {arm}: base world pos {bp} base R cols {bR.T.round(3).tolist()}")
                print(f"     eef(base) pos {p}  eef(world) pos {bR @ p + bp}")
                print(f"     eef world R columns x,y,z = {(bR @ R).T.round(3).tolist()}")
            ev = env.evaluator
            for i, name in enumerate(ev.names):
                adr = int(ev.addrs[i])
                jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
                root = int(env.model.body_rootid[env.model.jnt_bodyid[jid]])
                print(f"  object {name}: pos {env.data.qpos[adr:adr+3]} quat {env.data.qpos[adr+3:adr+7]} "
                      f"geoms {body_half_sizes(env.model, root)} target={i == ev.t}")
            priv = privileged_state(env)
            for k in ("dirt_center", "container_pos", "goal", "lift_m"):
                if k in priv:
                    print(f"  priv {k}: {priv[k]}")
            if getattr(ev, "container_adr", None) is not None:
                cj = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "container_joint")
                croot = int(env.model.body_rootid[env.model.jnt_bodyid[cj]])
                print(f"  container geoms {body_half_sizes(env.model, croot)}")

            # Inverse check: random targets -> delta chunk -> joints -> FK.
            rng = np.random.default_rng(seed)
            n = 10
            tg = {}
            for arm in ("left", "right"):
                p, R = anchors[arm]
                pos = p + rng.uniform(-0.05, 0.05, (n, 3))
                rot = np.stack([R @ rpy_to_mat(rng.uniform(-0.3, 0.3, 3)) for _ in range(n)])
                tg[arm] = (pos, rot, np.full(n, 0.5))
            chunk = bridge.eef_chunk_from_targets(tg, st)
            joints = bridge.joint_chunk_from_eef(chunk, st)
            pe, re_ = [], []
            for t in range(n):
                bridge.data.qpos[:] = 0.0
                bridge._set_arm_joints("left", joints[t, 0:6])  # noqa: SLF001
                bridge._set_arm_joints("right", joints[t, 7:13])  # noqa: SLF001
                mujoco.mj_kinematics(bridge.model, bridge.data)
                for arm in ("left", "right"):
                    p, R = bridge._site_pose_in_base(arm)  # noqa: SLF001
                    pe.append(np.linalg.norm(p - tg[arm][0][t]))
                    re_.append(np.linalg.norm(R - tg[arm][1][t]))
            print(f"  inverse check: pos err max {max(pe) * 1000:.3f} mm, rot (Frobenius) err max {max(re_):.4f}")

            # Execute: left arm 5 cm down then 5 cm +x (base frame), gripper unchanged.
            bridge.reset()
            p0, R0 = anchors["left"]
            n = 30
            s = np.linspace(0, 1, n)[:, None]
            path = np.where(s < 0.5, p0 + np.array([0, 0, -0.05]) * (2 * s),
                            p0 + np.array([0, 0, -0.05]) + np.array([0.05, 0, 0]) * (2 * s - 1))
            tg = {"left": (path, np.repeat(R0[None], n, 0), np.full(n, st[6])),
                  "right": (np.repeat(anchors["right"][0][None], n, 0), np.repeat(anchors["right"][1][None], n, 0),
                            np.full(n, st[13]))}
            chunk = bridge.eef_chunk_from_targets(tg, st)
            joints = bridge.joint_chunk_from_eef(chunk, st)
            g0 = privileged_state(env)["grasp_site"]["left"].copy()
            for t in range(n):
                env.step_one_vanilla(joints[t])
            g1 = privileged_state(env)["grasp_site"]["left"]
            bp, bR = bridge.base_pose_world("left")
            print(f"  executed left: commanded world delta {bR @ (path[-1] - p0)}, realized {g1 - g0}")
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(RLCollectConfig))
