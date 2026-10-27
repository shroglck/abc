"""Scripted privileged planner for SFT demonstrations (2026-10-27).

Produces the SAME thing the LAP flow head produces -- a 30-step bimanual EEF delta chunk
(per arm: dxyz, drpy, grip; deltas from the bridge's anchor, i.e. the FK of the last
commanded joints) -- so the chunk runs through the unchanged client path (bridge IK ->
joint targets -> sim) and doubles as a training label. Row i targets the planned pose
at step t+1+i (LAP chunks are lags 1..H from the anchor).

Tasks:
  w02b_wipe_plate_sponge  grasp the sponge top-down, carry it over the stains, press it
                          onto the plate and rub back and forth across the stain disc.
  o04_align_blocks        keep one lego, place the other two on either side of it in a
                          row (two pick-and-place subtasks).

The planner reads privileged state (object poses, stain positions) and replans when a
subtask fails (missed grasp, dropped object, misplaced lego). Diversity per episode:
speed, approach height, grasp offset and yaw, stroke pattern, row direction, spacing,
lego order. With `noise` > 0, DART-style perturbations are added to the EXECUTED chunk
only; the label stays the clean plan, which is re-anchored on the perturbed command next
chunk, so the labels include the corrections.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from abc_minimal.branching import _arm_of_body

DT = 0.034  # one action row = control_decimation (17) x 2 ms
HORIZON = 30
BLEND = 12  # steps over which a gap between the commanded pose and the plan is closed
# The fingers point along the grasp site's x axis; their tips sit 4.4 cm out along it and
# 3.8 mm along site z (measured, planner_probe2). Grasps put the tips TIP_CLEAR above the
# object's bottom (the table).
TIP_OFFSET = np.array([0.044, 0.0, 0.0038])
TIP_CLEAR = 0.006
TILTS = (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9)  # finger tilt about the jaw axis (rad), tried in order
LEGO_REST_Z = 0.767  # lego centre resting on the table
HALF_HEIGHT = {"sponge": 0.015, "lego_red": 0.012, "lego_blue": 0.012, "lego_yellow": 0.012}
PLANNER_TASKS = ("w02b_wipe_plate_sponge", "o04_align_blocks")


def topdown_rot(theta: float) -> np.ndarray:
    """World rotation of the grasp site with the fingers (site x) pointing down and the
    jaw (site y) along angle theta in the table plane. theta = pi/2 is the rest pose."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(theta), math.sin(theta), 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=1)


def tilt_about_jaw(r: np.ndarray, alpha: float) -> np.ndarray:
    c, s_ = math.cos(alpha), math.sin(alpha)
    return r @ np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m, dtype=np.float64).reshape(9))
    return q


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = _quat(r0), _quat(r1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        a = math.acos(d)
        q = (math.sin((1 - s) * a) * q0 + math.sin(s * a) * q1) / math.sin(a)
    q = q / np.linalg.norm(q)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def jaw_angle(obj_yaw: float, short_axis: str, symmetry: float, prefer: float) -> float:
    """Jaw direction across the object's short side, the equivalent (mod `symmetry`)
    closest to `prefer` so the wrist turns as little as possible."""
    base = obj_yaw + (math.pi / 2 if short_axis == "y" else 0.0)
    k = round((prefer - base) / symmetry)
    return base + k * symmetry


@dataclass
class Waypoint:
    pos: np.ndarray  # world
    rot: np.ndarray  # world
    grip: float      # 1 open, 0 closed
    dur: float       # seconds to reach it from the previous waypoint
    tag: str = ""


@dataclass
class ArmPlan:
    arm: str
    start_pos: np.ndarray
    start_rot: np.ndarray
    start_grip: float
    wps: list[Waypoint] = field(default_factory=list)

    def total_steps(self) -> int:
        return sum(max(1, round(w.dur / DT)) for w in self.wps)

    def at(self, k: int) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Planned (pos, rot, grip, tag) at step k after the plan started (clamped)."""
        p, r, g = self.start_pos, self.start_rot, self.start_grip
        for w in self.wps:
            n = max(1, round(w.dur / DT))
            if k <= n:
                s = k / n
                s = s * s * (3 - 2 * s)  # smoothstep: zero velocity at both ends
                return p + s * (w.pos - p), slerp(r, w.rot, s), g + s * (w.grip - g), w.tag
            k -= n
            p, r, g = w.pos, w.rot, w.grip
        return p, r, g, "done"


class ScriptedPlanner:
    def __init__(self, env: Any, bridge: Any, task_name: str, noise: float = 1.0):
        if task_name not in PLANNER_TASKS:
            raise ValueError(f"no scripted planner for {task_name}; have {PLANNER_TASKS}")
        self.env, self.bridge, self.task_name, self.noise = env, bridge, task_name, float(noise)
        self.rng = np.random.default_rng(0)
        self.base = {arm: bridge.base_pose_world(arm) for arm in ("left", "right")}

    # -- frames ------------------------------------------------------------------------
    def _world(self, arm: str, pos_b: np.ndarray, rot_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bp, bR = self.base[arm]
        return bR @ pos_b + bp, bR @ rot_b

    def _to_base(self, arm: str, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bp, bR = self.base[arm]
        return bR.T @ (pos - bp), bR.T @ rot

    def _grasped(self, arm: str, name: str) -> bool:
        """Pinch test as in branching.privileged_state, for any object: >= 2 distinct
        bodies of the arm touch it."""
        m, d = self.env.model, self.env.data
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        root = int(m.body_rootid[m.jnt_bodyid[jid]])
        bodies = set()
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if int(m.body_rootid[a]) == root and _arm_of_body(m, b) == arm:
                    bodies.add(b)
        return len(bodies) >= 2

    def _objects(self) -> dict[str, tuple[np.ndarray, float]]:
        ev, q = self.env.evaluator, self.env.data.qpos
        return {n: (np.asarray(q[int(a):int(a) + 3], float).copy(), yaw_of(q[int(a) + 3:int(a) + 7]))
                for n, a in zip(ev.names, ev.addrs)}

    # -- episode -------------------------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
        self.p = {  # per-episode style
            "speed": u(0.07, 0.12),          # m/s free-space translation (matched to the model's own chunks)
            "hover": u(0.06, 0.11),          # approach height above the object (m)
            "grasp_dz": u(-0.002, 0.004),    # grasp height offset
            "grasp_dxy": rng.normal(0, 0.004 * self.noise, 2) if self.noise else np.zeros(2),
            "yaw_noise": float(rng.normal(0, math.radians(4) * self.noise)) if self.noise else 0.0,
            "press": u(0.006, 0.014),        # how far below the stain surface the pad is pushed
            "stroke_amp": u(0.025, 0.045),   # half-length of a rub (m)
            "stroke_speed": u(0.05, 0.09),
            "stroke_axis": u(-0.6, 0.6),     # rub direction (rad, relative to world x)
            "gap": u(0.06, 0.08),            # lego centre spacing in the row
            "row_angle": u(-0.4, 0.4),       # row direction (rad from world x)
        }
        self.k = 0  # steps since the current plan started
        self.plan: dict[str, ArmPlan] = {}
        self.carrying: tuple[str, str] | None = None  # (arm, object) the current plan picks
        self.retries = 0
        self.anchor_lego: str | None = None
        self.slots: dict[str, np.ndarray] = {}
        self._replan(fresh=True)

    def _arm_pose_now(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        state14 = self.env.get_state_vanilla()
        p_b, r_b = self.bridge.anchor_poses(state14)[arm]
        return self._world(arm, p_b, r_b)

    def _hold(self, arm: str) -> ArmPlan:
        p, r = self._arm_pose_now(arm)
        g = float(self.env.get_state_vanilla()[6 if arm == "left" else 13])
        return ArmPlan(arm, p, r, g)

    def _move(self, plan: ArmPlan, pos: np.ndarray, rot: np.ndarray, grip: float, tag: str,
              min_dur: float = 0.4, speed: float | None = None) -> None:
        last = plan.wps[-1] if plan.wps else Waypoint(plan.start_pos, plan.start_rot, plan.start_grip, 0)
        dist = float(np.linalg.norm(pos - last.pos))
        ang = float(np.linalg.norm(_rotvec(last.rot.T @ rot)))
        dur = max(min_dur, dist / (speed or self.p["speed"]), ang / 0.8)
        plan.wps.append(Waypoint(np.asarray(pos, float), rot, grip, dur, tag))

    def _reachable(self, arm: str, pos: np.ndarray, rot: np.ndarray) -> bool:
        """IK reaches this world pose (from the current or the rest configuration)."""
        stats = {k: (list(v) if isinstance(v, list) else v) for k, v in self.bridge.ik_stats.items()}
        pb, rb = self._to_base(arm, pos, rot)
        cur = self.env.get_state_vanilla()[0:6] if arm == "left" else self.env.get_state_vanilla()[7:13]
        ok = False
        for seed in (cur, np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])):
            _, diag = self.bridge._ik_arm(arm, pb, rb, seed)  # noqa: SLF001
            if diag["position_error"] < 0.004 and diag["rotation_error"] < 0.03:
                ok = True
                break
        self.bridge.ik_stats = stats
        return ok

    def _grasp_pose(self, arm: str, obj_pos: np.ndarray, bottom_z: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site pose with the fingertips straddling the object TIP_CLEAR above its
        bottom; the fingers tilt about the jaw axis only as far as IK needs."""
        tip = np.array([*(obj_pos[:2] + self.p["grasp_dxy"]), bottom_z + TIP_CLEAR])
        r0 = topdown_rot(theta + self.p["yaw_noise"])
        for alpha in TILTS:
            r = tilt_about_jaw(r0, alpha)
            site = tip - r @ TIP_OFFSET
            if self._reachable(arm, site, r) and self._reachable(arm, site + [0, 0, self.p["hover"]], r):
                return site, r
        return tip - r0 @ TIP_OFFSET, r0

    def _pick(self, plan: ArmPlan, site: np.ndarray, r: np.ndarray) -> None:
        above = site + np.array([0, 0, self.p["hover"]])
        self._move(plan, above, r, 1.0, "approach", min_dur=1.0)
        self._move(plan, site, r, 1.0, "descend", speed=0.06)
        self._move(plan, site, r, 0.0, "close", min_dur=0.5)
        self._move(plan, above, r, 0.0, "lift", speed=0.08)

    def _replan(self, fresh: bool = False) -> None:
        """Plan the next leg from the current scene. A pick plan ends after the lift; the
        next replan then plans the carry/place (or press/rub) from the object's measured
        offset in the hand, so heights stay right whatever the grasp did."""
        objs = self._objects()
        prefer = math.pi / 2
        plans = {arm: self._hold(arm) for arm in ("left", "right")}
        self.carrying = None
        if self.task_name == "w02b_wipe_plate_sponge":
            sp, syaw = objs["sponge"]
            held = [a for a in ("left", "right") if self._grasped(a, "sponge")]
            arm = held[0] if held else ("left" if sp[1] > 0 else "right")
            plan = plans[arm]
            self.carrying = (arm, "sponge")
            spots = self.env.evaluator._dirt_world(self.env.data.qpos)  # noqa: SLF001
            c = spots.mean(axis=0)
            surf_z = float(np.median(spots[:, 2]))
            if not held:
                theta = jaw_angle(syaw, "y", math.pi, prefer)
                bottom = sp[2] - HALF_HEIGHT["sponge"] + self.p["grasp_dz"]
                self._pick(plan, *self._grasp_pose(arm, sp, bottom, theta))
            else:
                # Over the stains, down until the pad bottom is `press` below the stain
                # surface, then rub along stroke_axis across the disc.
                held_rot = plan.start_rot
                pad_off = sp[2] - plan.start_pos[2]  # pad centre relative to the commanded grasp site
                press_z = surf_z + HALF_HEIGHT["sponge"] - pad_off - self.p["press"]
                self._move(plan, np.array([c[0], c[1], press_z + 0.05]), held_rot, 0.0, "carry")
                self._move(plan, np.array([c[0], c[1], press_z]), held_rot, 0.0, "press", speed=0.05)
                a = self.p["stroke_axis"]
                d = np.array([math.cos(a), math.sin(a), 0.0])
                n = np.array([-d[1], d[0], 0.0])
                rel = spots[:, :2] - c[:2]
                half_w = float(np.abs(rel @ n[:2]).max()) if len(rel) else 0.0
                amp = max(self.p["stroke_amp"], float(np.abs(rel @ d[:2]).max()) if len(rel) else 0.0)
                rows = np.linspace(-half_w, half_w, max(2, int(math.ceil(2 * half_w / 0.03)) + 1))
                for i, off in enumerate(np.concatenate([rows, rows[::-1]])):
                    for sgn in ((1, -1) if i % 2 == 0 else (-1, 1)):
                        p = np.array([c[0], c[1], press_z]) + off * n + sgn * amp * d
                        self._move(plan, p, held_rot, 0.0, "rub", speed=self.p["stroke_speed"], min_dur=0.2)
        else:  # o04_align_blocks
            if fresh:
                names = list(objs)
                centre = np.array([0.57, 0.0])
                self.anchor_lego = min(names, key=lambda m: np.linalg.norm(objs[m][0][:2] - centre))
                ap = objs[self.anchor_lego][0]
                ang = self.p["row_angle"]
                d = np.array([math.cos(ang), math.sin(ang)])
                others = [m for m in names if m != self.anchor_lego]
                s1, s2 = ap[:2] + self.p["gap"] * d, ap[:2] - self.p["gap"] * d
                cost = lambda a, b: np.linalg.norm(objs[a][0][:2] - s1) + np.linalg.norm(objs[b][0][:2] - s2)  # noqa: E731
                o1, o2 = others if cost(*others) <= cost(*others[::-1]) else others[::-1]
                self.slots = {o1: s1, o2: s2}
                self.order = [o1, o2] if self.rng.random() < 0.5 else [o2, o1]
                self.row_dir = d
            todo = [m for m in self.order
                    if np.linalg.norm(objs[m][0][:2] - self.slots[m]) > 0.02 or objs[m][0][2] > 0.785]
            if todo and self.retries <= 6:
                m = todo[0]
                op, oyaw = objs[m]
                held = [a for a in ("left", "right") if self._grasped(a, m)]
                arm = held[0] if held else ("left" if op[1] > 0 else "right")
                plan = plans[arm]
                self.carrying = (arm, m)
                if not held:
                    theta = jaw_angle(oyaw, "y", math.pi / 2, prefer)
                    bottom = op[2] - HALF_HEIGHT[m] + self.p["grasp_dz"]
                    self._pick(plan, *self._grasp_pose(arm, op, bottom, theta))
                else:
                    # The row runs along x and the jaw stays near y, across the row, so the
                    # fingers never sit between neighbours; release opens only part way.
                    rd = math.atan2(self.row_dir[1], self.row_dir[0]) + math.pi / 2
                    cur = math.atan2(plan.start_rot[1, 1], plan.start_rot[0, 1])  # current jaw (site y) angle
                    pr = topdown_rot(jaw_angle(rd, "x", math.pi, cur))
                    # Lego offset in the hand, in the grasp-site frame (commanded pose), so the
                    # site target puts the LEGO on the slot whatever the grasp offset was.
                    off_local = plan.start_rot.T @ (op - plan.start_pos)
                    goal = np.array([*self.slots[m], LEGO_REST_Z + 0.003])
                    site_goal = goal - pr @ off_local
                    up = site_goal + np.array([0, 0, self.p["hover"]])
                    self._move(plan, up, pr, 0.0, "carry")
                    self._move(plan, site_goal, pr, 0.0, "lower", speed=0.05)
                    self._move(plan, site_goal, pr, 0.6, "release", min_dur=0.4)
                    self._move(plan, up, pr, 0.6, "retreat", speed=0.08)
        for arm, plan in plans.items():
            if not plan.wps and plan.start_grip < 0.99:
                self._move(plan, plan.start_pos, plan.start_rot, 1.0, "idle", min_dur=0.5)
        self.plan, self.k = plans, 0

    def _check(self) -> None:
        """Replan when the scene says the current plan cannot succeed."""
        if self.carrying is not None:
            arm, name = self.carrying
            plan = self.plan[arm]
            tag = plan.at(self.k)[3]
            if tag in ("lift", "carry", "press", "rub", "lower") and not self._grasped(arm, name) \
                    and plan.at(max(0, self.k - 20))[3] not in ("approach", "descend", "close"):
                self.retries += 1
                self._replan()
                return
        if self.task_name == "o04_align_blocks" and all(p.at(self.k)[3] == "done" for p in self.plan.values()):
            self._replan()
        elif self.task_name == "w02b_wipe_plate_sponge" and all(p.at(self.k)[3] == "done" for p in self.plan.values()):
            self._replan()  # keep rubbing with a fresh pattern until the episode succeeds

    # -- one chunk -----------------------------------------------------------------------
    def chunk(self, state14: np.ndarray, executed: int) -> tuple[np.ndarray, np.ndarray]:
        """(label, executed) 30x14 EEF delta chunks for the decision at the current state."""
        self._check()
        targets = {}
        anchors = self.bridge.anchor_poses(state14)
        for arm, plan in self.plan.items():
            # Where the arm was actually commanded to (after any perturbation) vs where the
            # plan is now: blend the gap out over BLEND steps, like a smooth correction.
            a_pos, a_rot = self._world(arm, *anchors[arm])
            off = a_pos - plan.at(self.k)[0]
            pos, rot, grip = [], [], []
            for i in range(1, HORIZON + 1):
                p, r, g, _ = plan.at(self.k + i)
                w = min(1.0, i / BLEND)
                w = w * w * (3 - 2 * w)
                p, r = p + (1 - w) * off, slerp(a_rot, r, w) if w < 1 else r
                pb, rb = self._to_base(arm, p, r)
                pos.append(pb); rot.append(rb); grip.append(g)
            targets[arm] = (np.stack(pos), np.stack(rot), np.asarray(grip))
        label = self.bridge.eef_chunk_from_targets(targets, state14)
        if os.environ.get("PLANNER_DEBUG"):
            arm, name = self.carrying or ("-", "-")
            tags = {a: p.at(self.k)[3] for a, p in self.plan.items()}
            obj = self._objects().get(name, (np.full(3, np.nan), 0))[0] if name != "-" else np.full(3, np.nan)
            site_err = ""
            if arm != "-":
                sid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site")
                act = self.env.data.site_xpos[sid]
                plan_p = self.plan[arm].at(self.k)[0]
                site_err = f"site={np.round(act, 3)} plan={np.round(plan_p, 3)} "
            print(f"[planner] k={self.k} tags={tags} carry={arm}:{name} {site_err}"
                  f"grasped={self._grasped(arm, name) if name != '-' else None} obj={obj.round(3)} "
                  f"retries={self.retries}", flush=True)
        execd = label.copy()
        if self.noise and self.rng.random() < 0.3:
            # Smooth offset ramping in over the executed rows (both translation and wrist yaw).
            ramp = np.clip(np.arange(1, HORIZON + 1) / executed, 0, 1)[:, None]
            for a0 in (0, 7):
                execd[:, a0:a0 + 3] += ramp * self.rng.normal(0, 0.012 * self.noise, 3)
                execd[:, a0 + 5] += ramp[:, 0] * self.rng.normal(0, math.radians(3) * self.noise)
        self.k += executed
        return label, execd


def _rotvec(m: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * math.sin(ang))
    return v * ang


class PlannerClient:
    """Stands in for the policy server: returns the planner's chunk as `actions` (executed)
    plus the clean label as `actions_label`."""

    def __init__(self, planner: ScriptedPlanner, execute_chunk_dim: int):
        self.planner, self.executed = planner, int(execute_chunk_dim)
        self.metadata: dict = {}

    def infer(self, request: dict) -> dict:
        label, execd = self.planner.chunk(self.planner.env.get_state_vanilla(), self.executed)
        return {"actions": execd, "actions_label": label}


class NoServer:
    """`_WebsocketClient` replacement for planner runs: never connects."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.metadata: dict = {}

    def infer(self, obs: dict) -> dict:
        raise RuntimeError("planner run: the policy server is not used")
