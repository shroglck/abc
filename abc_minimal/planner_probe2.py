"""Gripper geometry probe: hold the left gripper top-down at a fixed point, open then
closed, and report fingertip depth below the grasp site and the jaw opening (no server)."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_minimal.eval_policy import SimEnv, active_scene, require_mjwarp
from abc_minimal.planner import topdown_rot
from abc_minimal.planner_probe import make_bridge
from abc_minimal.rl_collect import RLCollectConfig


def finger_geoms(model: mujoco.MjModel, arm: str) -> dict[str, list[int]]:
    out = {"left": [], "right": []}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        while b > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name in (f"{arm}_link_left_finger", f"{arm}_link_right_finger"):
                if model.geom_contype[g] or model.geom_conaffinity[g]:
                    out["left" if "left_finger" in name else "right"].append(g)
                break
            b = int(model.body_parentid[b])
    return out


def main(config: RLCollectConfig) -> None:
    require_mjwarp()
    bridge = make_bridge(config)
    env = SimEnv(height=config.camera_height, width=config.camera_width, camera_keys=config.model.camera_keys,
                 prompt=config.prompt, scene=active_scene(config), task=config.task,
                 ladder_task=config.ladder_task, gpu_id=config.gpu_id)
    env.reset(seed=config.seed)
    bridge.reset()
    m, d = env.model, env.data
    fg = finger_geoms(m, "left")
    print(f"finger collision geoms: left {len(fg['left'])} right {len(fg['right'])}")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    bp, bR = bridge.base_pose_world("left")
    target_w = np.array([0.55, 0.30, 0.95])
    for grip in (1.0, 0.5, 0.0):
        st = env.get_state_vanilla()
        anchors = bridge.anchor_poses(st)
        n = 60
        r_b = bR.T @ topdown_rot(np.pi / 2)
        tg = {"left": (np.repeat((bR.T @ (target_w - bp))[None], n, 0), np.repeat(r_b[None], n, 0), np.full(n, grip)),
              "right": (np.repeat(anchors["right"][0][None], n, 0), np.repeat(anchors["right"][1][None], n, 0),
                        np.full(n, st[13]))}
        joints = bridge.joint_chunk_from_eef(bridge.eef_chunk_from_targets(tg, st), st)
        for t in range(n):
            env.step_one_vanilla(joints[t])
        bridge.note_command(joints[-1])
        site = d.site_xpos[sid].copy()
        zmin = min(float(d.geom_xpos[g][2] - np.max(m.geom_size[g])) for g in fg["left"] + fg["right"])
        lc = np.mean([d.geom_xpos[g] for g in fg["left"]], axis=0)
        rc = np.mean([d.geom_xpos[g] for g in fg["right"]], axis=0)
        # Inner gap: smallest distance between any left and any right collision geom centres.
        gap = min(float(np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b])) for a in fg["left"] for b in fg["right"])
        R = d.site_xmat[sid].reshape(3, 3)
        for side in ("left", "right"):
            low = sorted(fg[side], key=lambda g: d.geom_xpos[g][2])[:2]
            for g in low:
                rel = R.T @ (d.geom_xpos[g] - site)
                print(f"   {side} finger low geom {g} type {int(m.geom_type[g])} size {m.geom_size[g].round(4)} "
                      f"world {d.geom_xpos[g].round(4)} in site frame {rel.round(4)}")
        print(f"grip={grip}: site {site.round(4)} (target {target_w}) state grip {env.get_state_vanilla()[6]:.3f} "
              f"| lowest finger geom extent z {zmin:.4f} -> {1000 * (site[2] - zmin):.1f} mm below site "
              f"| finger centres L {lc.round(3)} R {rc.round(3)} | min centre gap {1000 * gap:.1f} mm")
    env.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(RLCollectConfig))
