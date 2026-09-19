"""Smoke test for the pick/place ladder: scene build, spawn sanity, metric oracles.

Runs no policy -- the arms hold the home pose. Two passes:

  --no-gpu   host MuJoCo only: every task x seed compiles, and after 40 held control
             steps (mj_step) no object has moved > 1 cm (an overlapping or off-table
             spawn pops apart or falls) and nothing has scored.
  default    the same checks through the real eval stack (SimEnv, MJWarp), plus one
             PNG per camera per task/seed, a metric oracle per task (teleport the
             target to where success is), and the l01 == b1g layout check.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/ladder_smoke.py --out-dir outputs/ladder_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import math

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # cloth rungs render with OpenGL; must precede `import mujoco`

import mujoco
import numpy as np

from abc_minimal.config import LadderSimConfig, PutBottlesSimConfig
from abc_minimal.ladder import LADDER_TASK_NAMES, get_task, ladder_scene_xml, sample_draw

CAMERAS = ("top", "left", "right")
B1G_SEED = 20260511


def _object_pos(model: mujoco.MjModel, qpos: np.ndarray, name: str, ev=None) -> np.ndarray:
    if ev is not None and name in ev.cloth:
        return ev.cloth_vertices(np.asarray(qpos, dtype=np.float64), name).mean(axis=0)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
    adr = int(model.jnt_qposadr[jid])
    return np.asarray(qpos[adr : adr + 3], dtype=np.float64)


def _check_settle(task, draw, model, qpos0, qpos1, metric, ev=None) -> None:
    for name in draw.objects:
        p0, p1 = _object_pos(model, qpos0, name, ev), _object_pos(model, qpos1, name, ev)
        dxy, dz = float(np.linalg.norm(p1[:2] - p0[:2])), float(p1[2] - p0[2])
        assert dxy < 0.01 and abs(dz) < 0.01, (
            f"{task.name} seed={draw.seed}: {name} moved dxy={dxy:.4f} dz={dz:.4f} "
            f"from {np.round(p0, 3).tolist()} to {np.round(p1, 3).tolist()}"
        )
    assert not metric["ever_success"], f"{task.name} seed={draw.seed}: scored with no policy"


def _hold_home(model: mujoco.MjModel, data: mujoco.MjData, scene: LadderSimConfig) -> None:
    """SimEnv._set_state + action_to_ctrl for init_q, on host MuJoCo."""
    q = iter(scene.init_q)
    for robot in ("left", "right"):
        for joint, act in [(f"{robot}_joint{j}", f"{robot}_joint{j}") for j in range(1, 7)] + [
            (f"{robot}_left_finger", f"{robot}_gripper")
        ]:
            val = float(next(q)) * (scene.gripper_ctrl_max if act.endswith("gripper") else 1.0)
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = val
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)] = val


def cpu_pass(tasks: list[str], seeds: list[int], steps: int) -> None:
    from abc_minimal.ladder import LadderEvaluator

    scene = LadderSimConfig()
    for name in tasks:
        task = get_task(name)
        for seed in seeds:
            draw = sample_draw(np.random.default_rng(seed), scene, task, seed=seed)
            model = mujoco.MjModel.from_xml_string(ladder_scene_xml(scene, task, draw))
            model.opt.timestep = scene.timestep
            data = mujoco.MjData(model)
            # Bodies are written at their drawn pose in the XML, so qpos0 is the spawn;
            # the arms get init_q, commanded through their position actuators.
            data.qpos[:] = model.qpos0
            _hold_home(model, data, scene)
            mujoco.mj_forward(model, data)
            ev = LadderEvaluator(model, scene, task, draw)
            ev.evaluate(data.qpos)
            qpos0 = data.qpos.copy()
            # Hold the arms at home through the position actuators.
            for _ in range(steps * scene.control_decimation):
                mujoco.mj_step(model, data)
            metric = ev.evaluate(data.qpos)
            _check_settle(task, draw, model, qpos0, data.qpos, metric, ev)
        print(f"[cpu] {name:28s} ok  nq={model.nq} ngeom={model.ngeom} prompt={draw.prompt!r}", flush=True)


def _oracle(env, task) -> dict:
    """Teleport the target to where the task's predicate holds; the metric must fire."""
    draw = env.randomization
    t = draw.target
    hold = np.asarray(env.scene.init_q, dtype=np.float32)

    def put(pos):
        env._set_freejoint(f"{t}_joint", list(map(float, pos)), draw.objects[t].quat)
        env.data.qvel[:] = 0
        env.sim.load_state()
        env.sim.forward()

    if task.goal == "fold":
        return _oracle_fold(env, task)
    if task.goal == "wipe":
        # Lift the pad, then raster it over the dirt at contact height; wiped spots must hide.
        ev_ = env.evaluate if env.sim is not None else env.evaluate_vanilla
        o = draw.objects[t]
        spots = env.evaluator._dirt_world(env.data.qpos)
        put(np.asarray(o.pos) + [0.0, 0.0, 0.10])
        ev_()
        lo, hi = spots[:, :2].min(0), spots[:, :2].max(0)
        zc = float(np.median(spots[:, 2])) + float(env.evaluator.pad_half[2])
        for y in np.linspace(lo[1], hi[1], 6):
            for x in np.linspace(lo[0], hi[0], 20):
                put([x, y, zc])
                metric = ev_()
        hidden = int((env.data.mocap_pos[env.evaluator.dirt_mocap][:, 2] < -1).sum())
        assert hidden == metric["wipe_n_cleaned"], (hidden, metric["wipe_n_cleaned"])
        return metric
    if task.goal == "align":
        # Lift one object (the metric needs it), then lay all of them in an even row.
        draw = env.randomization
        names = list(draw.objects)
        first = names[0]
        env._set_freejoint(f"{first}_joint", (np.asarray(draw.objects[first].pos) + [0, 0, 0.10]).tolist(),
                           draw.objects[first].quat)
        env.data.qvel[:] = 0
        (env.sim.load_state(), env.sim.forward()) if env.sim is not None else mujoco.mj_forward(env.model, env.data)
        env.evaluate() if env.sim is not None else env.evaluate_vanilla()
        x0, y0 = 0.55, -0.12
        for i, n in enumerate(names):
            env._set_freejoint(f"{n}_joint", [x0, y0 + 0.12 * i, float(draw.objects[n].pos[2])], draw.objects[n].quat)
        env.data.qvel[:] = 0
        if env.sim is not None:
            env.sim.load_state(); env.sim.forward()
        else:
            mujoco.mj_forward(env.model, env.data)
        step = env.step_one if env.sim is not None else env.step_one_vanilla
        ev = env.evaluate if env.sim is not None else env.evaluate_vanilla
        for _ in range(env.scene.stack_hold_evals + 5):
            step(hold)
            metric = ev()
        return metric
    spawn = np.asarray(draw.objects[t].pos)
    put(spawn + [0.0, 0.0, 0.10])  # lifted: every goal requires it
    metric = env.evaluate()
    if task.goal == "pick":
        return metric
    if task.goal == "place":
        ev = env.evaluator
        c = draw.container
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(c.quat, dtype=np.float64))
        centre_local = 0.5 * (ev.c_lo + ev.c_hi)
        put(np.asarray(c.pos) + rot.reshape(3, 3) @ centre_local)
        return env.evaluate()
    if task.goal == "on":
        ev = env.evaluator
        c = draw.container
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(c.quat, dtype=np.float64))
        centre = 0.5 * (ev.c_lo + ev.c_hi)
        centre[ev.c_up] = ev.c_hi[ev.c_up] + ev.on_half_h + 0.002
        put(np.asarray(c.pos) + rot.reshape(3, 3) @ centre)
        for _ in range(env.scene.stack_hold_evals + 5):
            env.step_one(hold)
            metric = env.evaluate()
        return metric
    # stack: set the target on the support and let it rest there.
    support = draw.support or task.support  # drawn per world on language rungs
    put(np.asarray(draw.objects[support].pos) + [0.0, 0.0, env.evaluator.stack_dz + 0.002])
    for _ in range(env.scene.stack_hold_evals + 5):
        env.step_one(hold)
        metric = env.evaluate()
    return metric


def _oracle_fold(env, task) -> dict:
    """Lift one half of the towel, then lay it mirrored onto the other half; must score."""
    ev = env.evaluator
    name = env.randomization.target
    body_pos, adr, axes = ev.cloth[name]
    v0 = ev.cloth_vertices(np.asarray(env.data.qpos, dtype=np.float64), name)
    c = v0[:, :2].mean(axis=0)
    yaw = env.randomization.objects[name].yaw
    ax = np.array([math.cos(yaw), math.sin(yaw)])  # fold across the towel's long axis
    proj = (v0[:, :2] - c) @ ax
    # Crease ON the first vertex column past the centre, so mirrored vertices land on
    # existing columns and every edge keeps its rest length (the cloth is inextensible).
    crease = float(np.min(proj[proj > 1e-6]))
    proj = proj - crease
    half = proj > 1e-6
    def set_verts(target):
        disp = target - body_pos  # slide axes are world-aligned
        for i, a in enumerate(adr):
            env.data.qpos[a : a + 3] = disp[i]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
    hold = np.asarray(env.scene.init_q, dtype=np.float32)
    lifted = v0.copy(); lifted[half, 2] += 0.10
    set_verts(lifted); env.evaluate_vanilla()  # registers the lift
    folded = v0.copy()
    folded[half, :2] = v0[half, :2] - 2 * np.outer(proj[half], ax)  # mirror across the crease
    folded[half, 2] = v0[half, 2] + 2 * ev.specs[name].cloth_radius + 0.002
    set_verts(folded)
    for _ in range(env.scene.stack_hold_evals + 5):
        env.step_one_vanilla(hold)
        metric = env.evaluate_vanilla()
    return metric


def gpu_pass(tasks: list[str], seeds: list[int], steps: int, out_dir: Path) -> None:
    import imageio.v2 as imageio

    from abc_minimal.eval_policy import SimEnv

    scene = LadderSimConfig()
    hold = np.asarray(scene.init_q, dtype=np.float32)
    for name in tasks:
        task = get_task(name)
        env = SimEnv(height=168, width=224, camera_keys=CAMERAS, prompt=task.prompt,
                     scene=scene, task="ladder", ladder_task=name)
        try:
            for seed in seeds:
                obs = env.reset(seed=seed)
                vanilla = env.sim is None  # cloth scenes: host physics + GL renderer
                step_fn = env.step_one_vanilla if vanilla else env.step_one
                eval_fn = env.evaluate_vanilla if vanilla else env.evaluate
                qpos_fn = (lambda: np.asarray(env.data.qpos).copy()) if vanilla else env.sim.qpos
                (out_dir / name).mkdir(parents=True, exist_ok=True)
                for cam in CAMERAS:
                    imageio.imwrite(out_dir / name / f"seed{seed}_{cam}.png", obs["images"][cam].transpose(1, 2, 0))
                eval_fn()
                qpos0 = qpos_fn()
                for _ in range(steps):
                    step_fn(hold)
                    metric = eval_fn()
                _check_settle(task, env.randomization, env.model, qpos0, qpos_fn(), metric, env.evaluator)
                env.reset(seed=seed)
                eval_fn()
                metric = _oracle(env, task)
                assert metric["ever_success"], f"{name} seed={seed}: oracle did not score: {metric}"
        finally:
            env.close()
        print(f"[gpu] {name:28s} ok  prompt={obs['prompt']!r}", flush=True)


def b1g_check(n: int) -> None:
    """l01 must reproduce put-bottles' b1g layouts (same RNG stream, same acceptance)."""
    from abc_minimal.eval_policy import SimEnv

    b1g = PutBottlesSimConfig(table_bounds=(0.45, 0.65, -0.10, 0.10), bin_xy_range=(0.70, 0.78, 0.15, 0.25))
    env = SimEnv(height=168, width=224, camera_keys=CAMERAS, prompt="x", scene=b1g, task="put_bottles")
    ladder = LadderSimConfig()
    task = get_task("l01_bottle_in_bin_tight")
    try:
        for seed in range(B1G_SEED, B1G_SEED + n):
            env.reset(seed=seed)
            r = env.randomization
            d = sample_draw(np.random.default_rng(seed), ladder, task, seed=seed)
            np.testing.assert_allclose(r.bottle_states["bottle_1_joint"]["pos"], d.objects["bottle"].pos, atol=1e-9)
            np.testing.assert_allclose(r.bottle_states["bottle_1_joint"]["quat"], d.objects["bottle"].quat, atol=1e-9)
            np.testing.assert_allclose(r.bin_state["pos"], d.container.pos, atol=1e-9)
            np.testing.assert_allclose(r.bin_state["quat"], d.container.quat, atol=1e-9)
            assert abs(r.bottle_scales[0] - d.bottle_scale) < 1e-6 and abs(r.bin_scale - d.container.scale) < 1e-12
    finally:
        env.close()
    print(f"[b1g] l01 layouts == put_bottles b1g for {n} seeds", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=B1G_SEED)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--b1g-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/ladder_smoke")
    args = ap.parse_args()

    tasks = list(LADDER_TASK_NAMES) if args.tasks == "all" else args.tasks.split(",")
    seeds = list(range(args.seed, args.seed + args.seeds))
    cpu_pass(tasks, seeds, args.steps)
    if args.no_gpu:
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b1g_check(args.b1g_seeds)
    gpu_pass(tasks, seeds, args.steps, out_dir)
    print(f"OK -- PNGs under {out_dir}")


if __name__ == "__main__":
    main()
