"""Branch-and-restore primitives for frozen-VLA selection studies.

A decision point is snapshotted, each of K policy candidates is executed from it
independently, and the chosen branch's end state is restored. Everything that a
rollout mutates has to round-trip, or branches leak into each other:

  - physics      MjData via mj_getState(mjSTATE_INTEGRATION): qpos, qvel, act, ctrl,
                 warmstart, mocap (the wipe rungs hide stains through mocap_pos), time.
  - evaluator    stateful "ever" metrics (spawn pose, max_lift, ever_picked, hold,
                 wipe strokes). Arrays/scalars are copied; model/data handles are not.
  - client       LAPClientPolicy carries prev langact text and the bridge's command
                 anchor (`_command_joints`), and `infer()` overwrites the anchor, so a
                 candidate that is sampled but not chosen would otherwise re-anchor the
                 next chunk.

VANILLA PHYSICS ONLY: the MJWarp path keeps its state on the GPU. Rendering after a
restore is safe because `SimEnv.render_cameras_vanilla_state` uploads MjData first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_SCALARS = (int, float, bool, str, type(None), np.generic)
_CONTAINERS = (np.ndarray, list, dict, tuple, set)
_CLIENT_ATTRS = ("_prev_language_actions", "_prev_langact", "reasoning_trace")


def _copy_attrs(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(obj).items():
        if isinstance(value, (mujoco.MjModel, mujoco.MjData)):
            continue
        if isinstance(value, _SCALARS):
            out[key] = value
        elif isinstance(value, _CONTAINERS):
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class SimSnapshot:
    physics: np.ndarray
    evaluator: dict[str, Any]
    client: dict[str, Any]
    command_joints: np.ndarray | None


def snapshot(env: Any, policy: Any) -> SimSnapshot:
    physics = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, physics, STATE_SPEC)
    client = {k: copy.deepcopy(getattr(policy, k)) for k in _CLIENT_ATTRS if hasattr(policy, k)}
    bridge = getattr(policy, "bridge", None)
    anchor = getattr(bridge, "_command_joints", None) if bridge is not None else None
    return SimSnapshot(
        physics=physics,
        evaluator=_copy_attrs(env.evaluator),
        client=client,
        command_joints=None if anchor is None else np.array(anchor, copy=True),
    )


def restore(env: Any, policy: Any, snap: SimSnapshot) -> None:
    mujoco.mj_setState(env.model, env.data, snap.physics, STATE_SPEC)
    mujoco.mj_forward(env.model, env.data)
    env.evaluator.__dict__.update(copy.deepcopy(snap.evaluator))
    for key, value in snap.client.items():
        setattr(policy, key, copy.deepcopy(value))
    bridge = getattr(policy, "bridge", None)
    if bridge is not None:
        bridge._command_joints = None if snap.command_joints is None else snap.command_joints.copy()


# Privileged state (ladder rungs).


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def _arm_of_body(model: mujoco.MjModel, body_id: int) -> str | None:
    name = _body_name(model, body_id)
    for arm in ("left", "right"):
        if name.startswith(f"{arm}_"):
            return arm
    return None


def privileged_state(env: Any) -> dict[str, Any]:
    """Ground-truth scene facts a deployed policy cannot observe directly.

    `grasped[arm]` is a pinch test: the target touches >= 2 distinct bodies of that
    arm's kinematic chain (both fingers, or finger + link). One contact is a nudge.
    """
    model, data, ev = env.model, env.data, env.evaluator
    target = ev.names[ev.t]
    adr = int(ev.addrs[ev.t])
    target_pos = np.asarray(data.qpos[adr : adr + 3], dtype=np.float64).copy()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{target}_joint")
    target_root = int(model.body_rootid[model.jnt_bodyid[jid]])

    grasp_site = {}
    for arm in ("left", "right"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site")
        grasp_site[arm] = np.asarray(data.site_xpos[sid], dtype=np.float64).copy()

    touching: dict[str, set[int]] = {"left": set(), "right": set()}
    # Arm bodies touching the container (plate/bin): tells a real one-arm hold from a
    # plate that was tipped or carried on top of something.
    c_touch: dict[str, set[int]] = {"left": set(), "right": set()}
    c_root = None
    if getattr(ev, "container_adr", None) is not None:
        cj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "container_joint")
        if cj >= 0:
            c_root = int(model.body_rootid[model.jnt_bodyid[cj]])
    arm_world_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        for a, b in ((b1, b2), (b2, b1)):
            arm = _arm_of_body(model, b)
            if arm is None:
                continue
            if int(model.body_rootid[a]) == target_root:
                touching[arm].add(b)
            elif c_root is not None and int(model.body_rootid[a]) == c_root:
                c_touch[arm].add(b)
            elif _arm_of_body(model, a) is None and int(model.body_rootid[a]) == 0:
                arm_world_contacts += 1  # arm against static scenery (table, walls)

    container_pos = None
    if getattr(ev, "container_adr", None) is not None:
        c = int(ev.container_adr)
        container_pos = np.asarray(data.qpos[c : c + 3], dtype=np.float64).copy()
    spawn = None if ev.spawn is None else np.asarray(ev.spawn[ev.t], dtype=np.float64)
    goal = ev.task.goal
    dirt_center = None
    if goal == "wipe" and getattr(ev, "dirt_ref", None) is not None:
        dirt_center = np.asarray(ev._dirt_world(np.asarray(data.qpos, dtype=np.float64)), dtype=np.float64).mean(axis=0)
    support_pos = None
    if goal == "stack" and getattr(ev, "s", None) is not None:
        sa = int(ev.addrs[ev.s])
        support_pos = np.asarray(data.qpos[sa : sa + 3], dtype=np.float64).copy()
    return {
        "goal": goal,
        "dirt_center": dirt_center,
        "support_pos": support_pos,
        "wipe_min_total_stroke": float(getattr(ev.task, "wipe_min_total_stroke", 0.0) or 0.0),
        "hold_lift": float(getattr(ev.task, "wipe_hold_lift", 0.0) or 0.0),
        "container_lift_m": (float(data.qpos[int(ev.container_adr) + 2]) - float(ev.container_z0))
        if getattr(ev, "container_z0", None) is not None and getattr(ev, "container_adr", None) is not None else 0.0,
        "target": target,
        "target_pos": target_pos,
        "lift_m": float(target_pos[2] - spawn[2]) if spawn is not None else 0.0,
        "grasp_site": grasp_site,
        "reach_m": {arm: float(np.linalg.norm(p - target_pos)) for arm, p in grasp_site.items()},
        "grasped": {arm: len(bodies) >= 2 for arm, bodies in touching.items()},
        "touching": {arm: bool(bodies) for arm, bodies in touching.items()},
        "arm_world_contacts": arm_world_contacts,
        "container_touching": {arm: bool(bodies) for arm, bodies in c_touch.items()},
        "container_grasped": {arm: len(bodies) >= 2 for arm, bodies in c_touch.items()},
        # 3D: a sponge hovering over a held plate is NOT close (the xy-only term missed this).
        "target_to_dirt_m": float(np.linalg.norm(target_pos - dirt_center)) if dirt_center is not None else None,
        "container_pos": container_pos,
    }


def progress_score(priv: dict[str, Any], ev_out: dict[str, Any], lift_m: float) -> float:
    """Dense privileged progress, higher is better. A HAND-WRITTEN heuristic, not
    ground truth: it ranks a candidate by the state its prefix reaches, not by whether
    the episode would go on to succeed.

    Shared phases: reach (metres from the nearest grasp site to the target) < contact <
    pinch grasp < lift (to `lift_m`), then a goal-specific term once the target is up:
      place/on  xy distance to the container
      stack     xy distance to the named support object, +1 while resting on it
      wipe      pad xy distance to the stain centroid, +contact, +scrub distance
                (vs the rung's minimum total stroke), +4 x fraction of stains cleaned;
                hold rungs add +2 x (container lift / required lift) at any phase
    """
    if ev_out.get("ever_success"):
        return 100.0
    score = -min(priv["reach_m"].values())
    if any(priv["touching"].values()):
        score += 0.5
    if any(priv["grasped"].values()):
        score += 1.0
    lift = float(np.clip(priv["lift_m"], 0.0, lift_m))
    goal = priv.get("goal")
    # The pad is pressed DOWN onto the dirt to wipe, so lifting it is only a step.
    score += (1.0 if goal == "wipe" else 3.0) * lift / lift_m
    if priv.get("hold_lift"):
        # Bimanual hold rungs: holding the container up is its own phase, in either order.
        # Only a plate an arm is actually touching counts fully: a knocked/tipped plate
        # (no arm contact) gets a quarter, so search does not learn to flip plates.
        lift_frac = float(np.clip(priv["container_lift_m"] / priv["hold_lift"], 0.0, 1.0))
        touched = any(priv.get("container_touching", {}).values())
        score += (2.0 if touched else 0.5) * lift_frac
    carried = bool(ev_out.get("picked")) or lift > 0.5 * lift_m
    if not carried:
        return float(score)

    def closeness(a: np.ndarray, b: np.ndarray, scale: float) -> float:
        return 1.0 - min(float(np.linalg.norm(a[:2] - b[:2])), scale) / scale

    if goal == "wipe" and priv.get("dirt_center") is not None:
        if priv.get("hold_lift") and priv["container_lift_m"] >= priv["hold_lift"]:
            # Plate held up: what is left is bringing the pad DOWN onto it, so score the 3D
            # gap (the sweep's bottleneck: sponge held 10+ cm above a held plate), plus a
            # bonus when the plate and the sponge are held by DIFFERENT arms.
            score += 3.0 * (1.0 - min(float(priv["target_to_dirt_m"]), 0.3) / 0.3)
            sponge_arms = {a for a, g in priv["grasped"].items() if g}
            plate_arms = {a for a, t in priv.get("container_touching", {}).items() if t}
            if sponge_arms and plate_arms - sponge_arms:
                score += 1.0
        else:
            score += 2.0 * closeness(priv["target_pos"], priv["dirt_center"], 0.5)
        score += 1.0 if ev_out.get("wipe_in_contact") else 0.0
        need = priv.get("wipe_min_total_stroke") or 0.10
        score += 2.0 * min(float(ev_out.get("wipe_total_stroke_m", 0.0)) / need, 1.0)
        score += 4.0 * float(ev_out.get("wipe_cleaned_frac", 0.0))
    elif goal == "stack" and priv.get("support_pos") is not None:
        score += 3.0 * closeness(priv["target_pos"], priv["support_pos"], 0.3)
        score += 1.0 if ev_out.get("on_top_now") else 0.0
    elif priv["container_pos"] is not None:
        score += 3.0 * closeness(priv["target_pos"], priv["container_pos"], 0.5)
    return float(score)
