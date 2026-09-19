"""Run MuJoCo-Warp put-bottles eval for ABC-DiT checkpoints.

Builds the scene, executes policy rollouts, and writes JSON/video outputs.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Headless OpenGL for the flex (cloth) rungs: MuJoCo picks its GL backend when the
# package is first imported, so this must precede `import mujoco` (GLFW fails on nodes).
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from abc_minimal.config import (
    FlowConfig,
    LadderSimConfig,
    MugFlipSimConfig,
    PutBottlesSimConfig,
    SimEvalConfig,
    validate_model_config,
)
from abc_minimal.mug_flip import (
    MugFlipEvaluator,
    mug_scene_xml,
    mug_spawn_z,
    sample_mug_xy,
    upside_down_quat,
)
from abc_minimal.ladder import (
    LadderDraw,
    LadderEvaluator,
    fill_prompt,
    get_task,
    ladder_scene_xml,
    sample_draw,
)
from abc_minimal.dit import (
    CLIPTextEmbedder,
    DiTPolicy,
    load_pretrained,
)
from abc_minimal.fast_inference import FastInferenceGraph, FastRTCInferenceGraph
from abc_minimal.preprocess import normalize, parse_norm_stats, resize_pad_normalize, unnormalize

torch.set_float32_matmul_precision("high")


# Config.

ROOT = Path(__file__).resolve().parents[1]
# Overridable per run: `reset()` re-reads this file for EVERY world, so two
# concurrent evals cannot share one mutable path -- editing it mid-run silently
# changes the scene partway through a rollout. ABC_SCENE_XML lets each job point at
# its own scene (e.g. the brightened vs the default-lit variant) instead.
SCENE_XML = Path(os.environ.get("ABC_SCENE_XML") or (ROOT / "assets" / "put_bottles" / "put_bottle.xml"))


# XML helpers.


def _fmt(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _flat_bottle_quat(yaw: float) -> np.ndarray:
    flat = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], dtype=np.float64)
    q = _quat_mul(_quat_yaw(yaw), flat)
    return q / np.linalg.norm(q)


def bottle_spawn_z(scene: PutBottlesSimConfig, index: int, scale: float = 1.0) -> float:
    return float(
        scene.table_z + scene.bottle_side_radii[index] * scale + scene.bottle_spawn_clearance
    )


def bottle_xy_footprint(
    scene: PutBottlesSimConfig,
    index: int,
    scale: float,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = float(scene.bottle_flat_lengths[index] * scale)
    half_width = float(scene.bottle_flat_half_widths[index] * scale)
    corners = np.array(
        [[0.0, -half_width], [0.0, half_width], [length, -half_width], [length, half_width]],
        dtype=np.float64,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    offsets = corners @ rot.T
    return offsets.min(axis=0), offsets.max(axis=0)


def sample_bottle_pose(
    rng: np.random.Generator,
    scene: PutBottlesSimConfig,
    index: int,
    scale: float,
    occupied: list[tuple[np.ndarray, float]],
) -> tuple[list[float], np.ndarray, np.ndarray, float]:
    candidate = None
    for _ in range(scene.bottle_sample_attempts):
        yaw = float(rng.uniform(-math.pi, math.pi))
        xy_min, xy_max = bottle_xy_footprint(scene, index, scale, yaw)
        table_x0, table_x1, table_y0, table_y1 = scene.table_bounds
        x_low, x_high = table_x0 - xy_min[0], table_x1 - xy_max[0]
        y_low, y_high = table_y0 - xy_min[1], table_y1 - xy_max[1]
        if x_low > x_high or y_low > y_high:
            continue
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        center = np.array([x, y], dtype=np.float64) + 0.5 * (xy_min + xy_max)
        radius = float(0.5 * np.linalg.norm(xy_max - xy_min))
        pos = [x, y, bottle_spawn_z(scene, index, scale)]
        quat = _flat_bottle_quat(yaw)
        candidate = (pos, quat, center, radius)
        if all(
            np.linalg.norm(center - c) > (radius + r + scene.bottle_collision_margin)
            for c, r in occupied
        ):
            return candidate
    if candidate is None:
        raise RuntimeError("Could not sample a bottle pose inside the table bounds")
    return candidate


def scene_xml(scene: PutBottlesSimConfig, bottle_scales: np.ndarray, bin_scale: float) -> str:
    root = ET.fromstring(SCENE_XML.read_text())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
        compiler.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        scale = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=np.float64)
        for idx in range(scene.bottle_count):
            if name.startswith(f"bottle_{idx}_"):
                mesh.set("scale", _fmt(scale * float(bottle_scales[idx])))
                break
        if name.startswith("water_bottle_"):
            mesh.set("scale", _fmt(scale * float(bin_scale)))

    if os.environ.get("ABC_SCENE_BRIGHT", "0") == "1":
        _brighten_surround(root)

    return ET.tostring(root, encoding="unicode")


def _brighten_surround(root: ET.Element) -> None:
    """Opt-in (ABC_SCENE_BRIGHT=1): render the surround light instead of near-black.

    The scene lights with `headlight diffuse="0 0 0"` plus one weak downward
    directional light, so under MuJoCo-Warp's ray tracer the `floor` plane comes out
    around RGB 27 despite its `rgba="0.75"`. (It is NOT a missed ray -- the miss path
    in mujoco_warp render.py only writes a pixel when render_skybox is set, and the
    value is unchanged with that flag on, so the ray is hitting the floor geom.)
    MuJoCo's OpenGL renderer shades the same XML far brighter, which is why the repo's
    sim eval looks white and these rollouts do not.

    This does not "fix" a bug -- it changes how the benchmark looks, so it must stay
    opt-in: every model ever scored in this scene was scored on the dark surround.
    Use it to measure how much the appearance gap matters, not as a new default.
    """
    for light in root.findall("./visual/headlight"):
        light.set("ambient", "0.80 0.80 0.80")
    for geom in root.findall(".//geom"):
        if geom.get("name") == "floor":
            geom.set("rgba", "1 1 1 1")


def active_scene(config: SimEvalConfig) -> PutBottlesSimConfig | MugFlipSimConfig | LadderSimConfig:
    """The task config the env, bridge, and metric all read."""
    return {"mug_flip": config.mug_scene, "ladder": config.ladder_scene}.get(config.task, config.scene)


def build_scene_xml(config: SimEvalConfig) -> str:
    """Un-randomized scene for the task, for consumers that only need the robot."""
    if config.task == "mug_flip":
        return mug_scene_xml(config.mug_scene, np.ones(config.mug_scene.mug_count))
    if config.task == "ladder":
        task = get_task(config.ladder_task)
        draw = sample_draw(np.random.default_rng(0), config.ladder_scene, task)
        return ladder_scene_xml(config.ladder_scene, task, draw)
    return scene_xml(config.scene, np.ones(config.scene.bottle_count), 1.0)


# Environment and metrics.


class PutBottlesEvaluator:
    def __init__(self, model: mujoco.MjModel, scene: PutBottlesSimConfig):
        self.model = model
        self.scene = scene
        self.bottle_names, self.bottle_qpos_addrs = self._bottle_addrs()
        self.bin_qpos_adr = self._joint_qpos_adr("bin_joint")
        self.bin_lo, self.bin_hi = self._bin_local_aabb()
        self.max_bottles = 0
        self.ever_success = False
        self.spawn_z: np.ndarray | None = None
        # Per-bottle running max lift. Kept per-bottle, not as one scalar, so PLACED
        # can ask "was THIS bottle raised" rather than "was any bottle raised".
        self.max_lift_per: np.ndarray | None = None
        self.ever_picked: np.ndarray | None = None

    def reset(self) -> None:
        self.max_bottles = 0
        self.ever_success = False
        # Cleared, not zeroed: the spawn height is captured on the first evaluate()
        # after a reset, which is the only point the bottle is known to be at rest.
        self.spawn_z = None
        self.max_lift_per = None
        self.ever_picked = None

    @property
    def max_lift(self) -> float:
        return 0.0 if self.max_lift_per is None else float(self.max_lift_per.max())

    def _joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _bin_local_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Extent of the bin's own geometry, in the bin body frame.

        Read off the model rather than hard-coded because each world rescales the bin
        mesh, so the numbers move per reset. In this frame the bin's axis is +y (the
        body carries a 90-deg x rotation), giving a footprint in x/z and the floor and
        rim as the y bounds -- which is why containment cannot be expressed as a
        radius about the joint origin the way `eval_bin_radius` tried to.
        """
        bin_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
        if bin_body < 0:
            raise ValueError("Body not found: bin_container")
        pts = []
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] != bin_body:
                continue
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = self.model.geom_dataid[geom]
            adr, num = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            verts = self.model.mesh_vert[adr : adr + num].reshape(-1, 3)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, self.model.geom_quat[geom])
            pts.append(verts @ rot.reshape(3, 3).T + self.model.geom_pos[geom])
        if not pts:
            raise ValueError("bin_container has no mesh geoms to measure")
        allv = np.vstack(pts)
        return allv.min(axis=0), allv.max(axis=0)

    def _bottle_addrs(self) -> tuple[list[str], np.ndarray]:
        entries = []
        for joint_id in range(self.model.njnt):
            name = self.model.jnt(joint_id).name
            match = re.fullmatch(r"bottle_(\d+)_joint", name or "")
            if match:
                idx = int(match.group(1))
                entries.append((idx, f"bottle_{idx}", int(self.model.jnt_qposadr[joint_id])))
        entries.sort()
        return [e[1] for e in entries], np.asarray([e[2] for e in entries], dtype=np.int32)

    def evaluate(self, qpos: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(qpos, dtype=np.float32)
        bin_pos = qpos[self.bin_qpos_adr : self.bin_qpos_adr + 3]
        bin_quat = qpos[self.bin_qpos_adr + 3 : self.bin_qpos_adr + 7]
        bottle_pos = np.stack([qpos[adr : adr + 3] for adr in self.bottle_qpos_addrs])
        rel = bottle_pos - bin_pos[None]

        # Containment is tested in the BIN's frame, not the world's: the bin is
        # yaw-randomized per world, so a world-axis box would clip a rotated bin's
        # corners and admit space outside its walls. `local` is the bottle center
        # expressed in the bin body frame, where `bin_lo/bin_hi` bound the geometry.
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(bin_quat, dtype=np.float64))
        local = rel @ rot.reshape(3, 3)

        inside_box = np.all((local >= self.bin_lo[None]) & (local <= self.bin_hi[None]), axis=1)
        # Kept for the summary/plots; these are diagnostics now, not the predicate.
        radial = np.linalg.norm(rel[:, :2], axis=1)
        radial_margin = self.scene.eval_bin_radius - radial
        lower_height_margin = rel[:, 2] - self.scene.eval_min_rel_z
        upper_height_margin = self.scene.eval_max_rel_z - rel[:, 2]

        if self.spawn_z is None:
            self.spawn_z = bottle_pos[:, 2].copy()
            self.max_lift_per = np.zeros(len(self.spawn_z), dtype=np.float64)
            self.ever_picked = np.zeros(len(self.spawn_z), dtype=bool)
        lift = bottle_pos[:, 2] - self.spawn_z
        self.max_lift_per = np.maximum(self.max_lift_per, lift)
        self.ever_picked |= self.max_lift_per >= self.scene.eval_pick_lift_m

        # PLACED is deliberately stricter than "inside the box the bin occupies": the
        # bottle center must be within the bin's own footprint AND still elevated,
        # which a bottle resting on the table beside the bin never is. Without the
        # lift term the radial+z window alone admits table-level bottles.
        # The lift term is EVER-lifted, not lift-right-now: the bin's interior floor
        # sits ~4 mm below the table top, so a bottle correctly resting in the bin is
        # at roughly its spawn height. Testing instantaneous lift would reject exactly
        # the successes we are trying to count. Requiring that it was raised at some
        # point is what separates "carried in" from "knocked or slid in".
        in_bin = inside_box & (self.max_lift_per >= self.scene.eval_place_min_lift_m)
        num = int(in_bin.sum())
        active = len(self.bottle_names)
        self.max_bottles = max(self.max_bottles, num)
        success = num == active
        self.ever_success = self.ever_success or success
        return {
            "reward": float(num / max(active, 1)),
            "success": bool(success),
            "ever_success": bool(self.ever_success),
            "num_bottles_in_bin": num,
            "num_active_bottles": active,
            "max_bottles_in_bin_so_far": int(self.max_bottles),
            # Task-agnostic aliases: the rollout loop and summary read these so a
            # second task only has to supply its own count, not its own logging.
            "num_solved": num,
            "num_active": active,
            "max_solved_so_far": int(self.max_bottles),
            "bottle_in_bin_mask": [bool(x) for x in in_bin.tolist()],
            "bottles_in_bin": [name for name, ok in zip(self.bottle_names, in_bin) if bool(ok)],
            "bottle_names": list(self.bottle_names),
            "success_count": active,
            "bin_local_pos": [[float(v) for v in row] for row in local.tolist()],
            "bin_local_lo": [float(v) for v in self.bin_lo.tolist()],
            "bin_local_hi": [float(v) for v in self.bin_hi.tolist()],
            "inside_bin_box": [bool(x) for x in inside_box.tolist()],
            "closest_radial_margin": float(radial_margin.max()),
            "closest_height_margin": float(lower_height_margin.max()),
            "closest_upper_height_margin": float(upper_height_margin.max()),
            "max_lift_m": float(self.max_lift),
            "max_lift_per_bottle_m": [float(x) for x in self.max_lift_per.tolist()],
            # PICKED: raised off the table at any point, regardless of where it ended.
            # PLACED: currently in the bin under the strict predicate above.
            # Every placed episode is also picked, so pick_rate is an upper bound on
            # place_rate and (picked & !placed) is the "lifted then dropped" bucket.
            "picked": bool(self.ever_picked.all()),
            "num_picked": int(self.ever_picked.sum()),
            "placed": bool(success),
            "num_placed": num,
            "lifted": bool(self.ever_picked.all()),
        }


@dataclass
class Randomization:
    seed: int
    bottle_states: dict[str, dict[str, list[float]]]
    bin_state: dict[str, list[float]]
    bottle_scales: list[float]
    bin_scale: float


@dataclass
class MugRandomization:
    seed: int
    mug_states: dict[str, dict[str, list[float]]]
    mug_scales: list[float]


def require_mjwarp() -> None:
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "eval_policy.py uses MJWarp only. Install/import `mujoco_warp` and `warp` "
            "in the eval environment before running sim eval."
        ) from exc


class MJWarpSim:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, height: int, width: int, gpu_id: int | None):
        require_mjwarp()
        import mujoco_warp as mjw
        import warp as wp

        self.mjw = mjw
        self.wp = wp
        self.model = model
        self.data = data
        self.height = height
        self.width = width
        self.nworld = 1
        if gpu_id is not None:
            wp.set_device(f"cuda:{gpu_id}")
        self.m_warp = mjw.put_model(model)
        self.d_warp = mjw.put_data(model, data, nworld=self.nworld, nconmax=model.nconmax, njmax=model.njmax)
        self.render_context = mjw.create_render_context(
            mjm=model,
            nworld=self.nworld,
            cam_res=(width, height),
            render_rgb=[True] * model.ncam,
            render_depth=[False] * model.ncam,
            use_textures=True,
            use_shadows=True,
            # MuJoCo-Warp's ray tracer defaults render_skybox=False, so every ray that
            # misses geometry returns the clear colour and the background comes out
            # BLACK -- even though put_bottle.xml defines a near-white skybox
            # (`builtin="gradient" rgb1="1 1 1" rgb2="0.98 0.98 0.98"`). The policy
            # then sees a black void where the scene specifies a bright surround,
            # which is both wrong w.r.t. the repo's intended render and further from
            # any real workspace the checkpoints trained on.
            render_skybox=True,
        )
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.wp.synchronize()
        except Exception:
            pass
        self.render_context = None
        self.m_warp = None
        self.d_warp = None
        self.model = None
        self.data = None
        self.closed = True

    def _copy(self, target: Any, values: np.ndarray, dtype: Any) -> None:
        self.wp.copy(target, self.wp.from_numpy(np.asarray(values), dtype=dtype))

    def load_state(self) -> None:
        self.mjw.reset_data(self.m_warp, self.d_warp)
        self._copy(self.d_warp.qpos, np.asarray(self.data.qpos, dtype=np.float32)[None], self.wp.float32)
        if self.model.nv > 0:
            self._copy(self.d_warp.qvel, np.asarray(self.data.qvel, dtype=np.float32)[None], self.wp.float32)
        if self.model.nu > 0:
            self._copy(self.d_warp.ctrl, np.asarray(self.data.ctrl, dtype=np.float32)[None], self.wp.float32)
        if self.model.na > 0 and hasattr(self.d_warp, "act"):
            self._copy(self.d_warp.act, np.asarray(self.data.act, dtype=np.float32)[None], self.wp.float32)
        if self.model.nmocap > 0:
            self._copy(self.d_warp.mocap_pos, np.asarray(self.data.mocap_pos, dtype=np.float32)[None], self.wp.vec3f)
            self._copy(self.d_warp.mocap_quat, np.asarray(self.data.mocap_quat, dtype=np.float32)[None], self.wp.quatf)
        if hasattr(self.d_warp, "time"):
            self._copy(self.d_warp.time, np.asarray([self.data.time], dtype=np.float32), self.wp.float32)

    def forward(self) -> None:
        self.mjw.forward(self.m_warp, self.d_warp)

    def qpos(self) -> np.ndarray:
        return self.d_warp.qpos.numpy()[0].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        if ctrl.shape != (self.model.nu,):
            raise ValueError(f"Expected ctrl shape {(self.model.nu,)}, got {ctrl.shape}")
        self._copy(self.d_warp.ctrl, ctrl[None], self.wp.float32)

    def step(self, nstep: int) -> None:
        for _ in range(nstep):
            self.mjw.step(self.m_warp, self.d_warp)

    def render(self) -> np.ndarray:
        self.mjw.refit_bvh(self.m_warp, self.d_warp, self.render_context)
        self.mjw.render(self.m_warp, self.d_warp, self.render_context)
        rgba = self.render_context.rgb_data.numpy().view(np.uint8).reshape(
            self.nworld,
            self.model.ncam,
            self.height,
            self.width,
            4,
        )
        return rgba[0, :, :, :, :3][..., ::-1].copy()


class SimEnv:
    """One ABC sim task on the shared YAM scene.

    `scene` is the ACTIVE task's config: both task configs carry the same robot
    fields (gripper stroke, init_q, timestep, control decimation), which is what
    every method outside `reset`/`_bind` reads, so only those two branch on `task`.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        camera_keys: tuple[str, ...],
        prompt: str,
        scene: PutBottlesSimConfig | MugFlipSimConfig | LadderSimConfig,
        task: str = "put_bottles",
        ladder_task: str = "",
        gpu_id: int | None = None,
    ):
        self.scene = scene
        self.task = task
        self.ladder_task = get_task(ladder_task) if task == "ladder" else None
        self._ladder_draw: LadderDraw | None = None
        self.mug_scales = np.ones(getattr(scene, "mug_count", 0), dtype=np.float32)
        self.height = height
        self.width = width
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        # The ladder fills a per-world prompt from this (p09's {color}).
        self.prompt_template = prompt
        self.control_decimation = scene.control_decimation
        self.gpu_id = gpu_id
        self.sim = None
        self.model = None
        self.data = None
        self.evaluator = None
        self.qpos_indices: list[int] = []
        self.ctrl_indices: list[int] = []
        self.gripper_state_indices: set[int] = set()
        self.randomization = None
        # Set in _bind when ABC_RENDER_GL=1; see render_cameras.
        self._gl_renderer = None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        if self._gl_renderer is not None:
            self._gl_renderer.close()
            self._gl_renderer = None

    def _bind(self, xml: str) -> None:
        self.close()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.scene.timestep
        self.data = mujoco.MjData(self.model)
        # Deformables (flex) exist only on the host: MuJoCo-Warp has no flex support, so a
        # cloth scene runs --vanilla-physics and renders with the OpenGL renderer. The
        # appearance differs from the ray-traced rigid rungs; summary.json records it.
        self.has_flex = int(self.model.nflex) > 0
        use_gl = os.environ.get("ABC_RENDER_GL", "0") == "1" or self.has_flex
        if self.has_flex:
            self.sim = None
        else:
            self.sim = MJWarpSim(self.model, self.data, height=self.height, width=self.width, gpu_id=self.gpu_id)
        # Rebuilt with the model: each reset randomizes bottle/bin scales, so the XML
        # (and therefore the MjModel this renderer is bound to) changes per world.
        if use_gl:
            self._gl_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        if self.task == "mug_flip":
            self.evaluator = MugFlipEvaluator(self.model, self.scene, self.mug_scales)
        elif self.task == "ladder":
            self.evaluator = LadderEvaluator(self.model, self.scene, self.ladder_task, self._ladder_draw, data=self.data)
        else:
            self.evaluator = PutBottlesEvaluator(self.model, self.scene)
        self.qpos_indices, self.ctrl_indices, self.gripper_state_indices = [], [], set()
        idx = 0
        for robot in ("left", "right"):
            for j in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_joint{j}")
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_joint{j}")
                self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                self.ctrl_indices.append(aid)
                idx += 1
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_left_finger")
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_gripper")
            self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
            self.ctrl_indices.append(aid)
            self.gripper_state_indices.add(idx)
            idx += 1

    def reset(self, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        if self.task == "mug_flip":
            return self._reset_mug_flip(rng, seed)
        if self.task == "ladder":
            return self._reset_ladder(rng, seed)
        scene = self.scene
        bottle_scales = rng.uniform(*scene.bottle_scale_range, size=scene.bottle_count).astype(np.float32)
        bin_scale = float(rng.uniform(*scene.bin_scale_range))
        self._bind(scene_xml(scene, bottle_scales, bin_scale))
        mujoco.mj_resetData(self.model, self.data)
        self._set_state(np.asarray(scene.init_q, dtype=np.float32))

        bin_yaw = float(rng.uniform(*scene.bin_yaw_range))
        bin_x0, bin_x1, bin_y0, bin_y1 = scene.bin_xy_range
        bin_pos = [
            float(rng.uniform(bin_x0, bin_x1)),
            float(rng.uniform(bin_y0, bin_y1)),
            float(scene.bin_z_scale * bin_scale),
        ]
        bin_quat = _quat_mul(_quat_yaw(bin_yaw), np.asarray(scene.bin_base_quat, dtype=np.float64))
        self._set_freejoint("bin_joint", bin_pos, bin_quat.tolist())

        occupied: list[tuple[np.ndarray, float]] = [
            (np.asarray(bin_pos[:2]), scene.bin_occupied_radius * bin_scale)
        ]
        bottle_states = {}
        # `put_bottle.xml` ships ONE bottle body. A larger --scene.bottle-count used
        # to write bottles 2..N onto a joint id of -1, i.e. over the last joint in
        # the model, so the extra bottles never existed and the surviving bottle got
        # the last sampled pose. Clamp to what the scene actually has, and say so.
        present = len(self.evaluator.bottle_names)
        count = min(scene.bottle_count, present)
        if count < scene.bottle_count:
            print(
                f"[scene] bottle_count={scene.bottle_count} but the scene has {present} "
                f"bottle body/bodies; spawning {count}.",
                flush=True,
            )
        for index in range(count):
            pos, q, center, radius = sample_bottle_pose(
                rng, scene, index, float(bottle_scales[index]), occupied
            )
            name = f"bottle_{index + 1}_joint"
            self._set_freejoint(name, pos, q.tolist())
            bottle_states[name] = {"pos": pos, "quat": q.tolist(), "scale": [float(bottle_scales[index])]}
            occupied.append((center, radius))

        mujoco.mj_forward(self.model, self.data)
        self.sim.load_state()
        self.sim.forward()
        self.evaluator.reset()
        self.randomization = Randomization(
            seed=seed,
            bottle_states=bottle_states,
            bin_state={"pos": bin_pos, "quat": bin_quat.tolist(), "yaw": [bin_yaw]},
            bottle_scales=bottle_scales.tolist(),
            bin_scale=bin_scale,
        )
        return self.obs()

    def _reset_mug_flip(self, rng: np.random.Generator, seed: int) -> dict[str, Any]:
        scene = self.scene
        self.mug_scales = rng.uniform(*scene.mug_scale_range, size=scene.mug_count).astype(
            np.float32
        )
        self._bind(mug_scene_xml(scene, self.mug_scales))
        mujoco.mj_resetData(self.model, self.data)
        self._set_state(np.asarray(scene.init_q, dtype=np.float32))

        occupied: list[np.ndarray] = []
        mug_states = {}
        for index in range(scene.mug_count):
            scale = float(self.mug_scales[index])
            xy = sample_mug_xy(rng, scene, occupied)
            occupied.append(xy)
            yaw = float(rng.uniform(-math.pi, math.pi))
            # Every mug starts upside down -- that is the task.
            quat = upside_down_quat(yaw)
            pos = [float(xy[0]), float(xy[1]), mug_spawn_z(scene, scale)]
            name = f"mug_{index}_joint"
            self._set_freejoint(name, pos, quat.tolist())
            mug_states[name] = {"pos": pos, "quat": quat.tolist(), "yaw": [yaw], "scale": [scale]}

        mujoco.mj_forward(self.model, self.data)
        self.sim.load_state()
        self.sim.forward()
        self.evaluator.reset()
        self.randomization = MugRandomization(
            seed=seed,
            mug_states=mug_states,
            mug_scales=[float(v) for v in self.mug_scales.tolist()],
        )
        return self.obs()

    def _reset_ladder(self, rng: np.random.Generator, seed: int) -> dict[str, Any]:
        scene, task = self.scene, self.ladder_task
        draw = sample_draw(rng, scene, task, seed=seed)
        self._ladder_draw = draw  # _bind hands it to the evaluator (target, sizes)
        self._bind(ladder_scene_xml(scene, task, draw))
        mujoco.mj_resetData(self.model, self.data)
        self._set_state(np.asarray(scene.init_q, dtype=np.float32))
        if draw.container is not None:
            self._set_freejoint("container_joint", draw.container.pos, draw.container.quat)
        for name, obj in draw.objects.items():
            if obj.kind == "cloth":
                continue  # a flexcomp's pose is baked into the XML; its vertices have no free joint
            self._set_freejoint(f"{name}_joint", obj.pos, obj.quat)

        mujoco.mj_forward(self.model, self.data)
        if task.goal == "wipe":
            self.evaluator.place_dirt(self.data)  # before the first render: the policy sees the dirt
        if self.sim is not None:
            self.sim.load_state()
            self.sim.forward()
        self.evaluator.reset()
        # Per-world prompt from the run's template (the table's, or --prompt):
        # p09 names a different colour each world.
        draw.prompt = fill_prompt(self.prompt_template, task, draw.target, draw.support)
        self.prompt = draw.prompt
        self.randomization = draw
        return self.obs()

    def _set_freejoint(self, name: str, pos: list[float], quat: list[float]) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            # numpy would index jnt_qposadr[-1] and silently write this pose over
            # the LAST joint in the model instead of erroring.
            raise ValueError(f"Free joint not found in the scene: {name}")
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat

    def _set_state(self, state: np.ndarray) -> None:
        for i, qpos_idx in enumerate(self.qpos_indices):
            val = float(state[i])
            if i in self.gripper_state_indices:
                val *= self.scene.gripper_ctrl_max
            self.data.qpos[qpos_idx] = val

    def get_state(self) -> np.ndarray:
        if self.sim is None:
            return self.get_state_vanilla()
        state = np.asarray(self.sim.qpos()[self.qpos_indices], dtype=np.float32)
        for i in self.gripper_state_indices:
            state[i] = np.clip(state[i] / self.scene.gripper_ctrl_max, 0.0, 1.0)
        return state

    def render_cameras(self) -> dict[str, np.ndarray]:
        if self._gl_renderer is not None:
            # MuJoCo's OpenGL renderer, off the HOST MjData -- the appearance the ABC
            # repo shows. MJWarp's ray tracer renders this scene ~3x darker (floor
            # corner 28 vs GL's 90) because it ignores the `headlight ambient` the
            # scene lights with, and never shows the white skybox at all.
            #
            # Renders only the cameras in use (3, not all 6) and needs no device->host
            # sync: with --vanilla-physics (the repo default) mj_step already advances
            # self.data on the host. An earlier version did get_data_into() per frame
            # over all cameras and stalled the rollout until the policy server dropped
            # the connection on a keepalive timeout.
            images = {}
            for name in self.camera_keys:
                self._gl_renderer.update_scene(self.data, camera=name)
                images[name] = self._gl_renderer.render().transpose(2, 0, 1).copy()
            return images

        rgb = self.sim.render()
        images = {}
        for name in self.camera_keys:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise ValueError(f"Camera not found: {name}")
            images[name] = rgb[cam_id].transpose(2, 0, 1).copy()
        return images

    def obs(self) -> dict[str, Any]:
        return {"state": self.get_state(), "images": self.render_cameras(), "prompt": self.prompt}

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.ctrl_indices):
            val = float(action[i])
            if i in self.gripper_state_indices:
                val *= self.scene.gripper_ctrl_max
            ctrl[act_id] = val
        return ctrl

    def step_one(self, action: np.ndarray) -> None:
        if self.sim is None:
            return self.step_one_vanilla(action)
        ctrl = self.action_to_ctrl(action)
        self.sim.set_ctrl(ctrl)
        self.sim.step(self.control_decimation)

    def evaluate(self) -> dict[str, Any]:
        if self.sim is None:
            return self.evaluate_vanilla()
        return self.evaluator.evaluate(self.sim.qpos())

    def step_one_vanilla(self, action: np.ndarray) -> None:
        self.data.ctrl[:] = self.action_to_ctrl(action)
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

    def evaluate_vanilla(self) -> dict[str, Any]:
        return self.evaluator.evaluate(np.asarray(self.data.qpos, dtype=np.float32))

    def get_state_vanilla(self) -> np.ndarray:
        state = np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float32)
        for i in self.gripper_state_indices:
            state[i] = np.clip(state[i] / self.scene.gripper_ctrl_max, 0.0, 1.0)
        return state

    def render_cameras_vanilla_state(self) -> dict[str, np.ndarray]:
        if self.sim is not None:
            self.sim.load_state()
            self.sim.forward()
        return self.render_cameras()

    def obs_vanilla_state(self) -> dict[str, Any]:
        return {
            "state": self.get_state_vanilla(),
            "images": self.render_cameras_vanilla_state(),
            "prompt": self.prompt,
        }


#: Back-compat alias: the env was put-bottles-only before the mug task.
PutBottlesEnv = SimEnv


# Policy adapter.


def load_json_or_s3(path: str) -> dict[str, Any]:
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            subprocess.run(["aws", "s3", "cp", path, tmp.name], check=True)
            return json.loads(Path(tmp.name).read_text())
    return json.loads(Path(path).expanduser().read_text())


def local_checkpoint(path: str) -> Path:
    if not path.startswith("s3://"):
        return Path(path).expanduser().resolve()
    out_dir = ROOT / "checkpoints" / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(path).name
    if not out.exists():
        subprocess.run(["aws", "s3", "cp", path, str(out)], check=True)
    return out


def resolve_norm_stats(ckpt: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override:
        raw = load_json_or_s3(override)
    elif ckpt.get("norm_stats") is not None:
        raw = ckpt["norm_stats"]
    else:
        raise ValueError("No norm_stats in checkpoint; pass --norm-stats-path")
    return parse_norm_stats(raw)


class _RTCManager:
    def __init__(
        self,
        policy: "SimPolicy",
        *,
        prefix_length: int,
        inference_lead_steps: int,
        execute_chunk_dim: int,
    ):
        self.policy = policy
        self.prefix_length = prefix_length
        self.inference_lead_steps = inference_lead_steps
        self.execute_chunk_dim = execute_chunk_dim
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: concurrent.futures.Future[tuple[np.ndarray, float]] | None = None

    def _action_prefix(self, actions: np.ndarray) -> np.ndarray:
        m = self.policy.config.model
        prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        executed = np.asarray(actions[: self.execute_chunk_dim], dtype=np.float32)
        prefix[: self.prefix_length] = executed[-self.prefix_length :]
        return prefix

    def start(
        self,
        obs: dict[str, Any],
        current_actions: np.ndarray,
        noise: np.ndarray | None,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("RTC inference is already pending")
        action_prefix = self._action_prefix(current_actions)

        def _run() -> tuple[np.ndarray, float]:
            t0 = time.perf_counter()
            actions = self.policy.infer(
                obs,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=self.prefix_length,
            )
            return actions[self.prefix_length :], time.perf_counter() - t0

        self._pending = self._executor.submit(_run)

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def get(self) -> tuple[np.ndarray, float, bool]:
        if self._pending is None:
            raise RuntimeError("No RTC inference is pending")
        ready = self._pending.done()
        actions, infer_s = self._pending.result()
        self._pending = None
        return actions, infer_s, ready

    def close(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending = None
        self._executor.shutdown(wait=True)


class SimPolicy:
    def __init__(self, checkpoint: Path, config: SimEvalConfig, device: str):
        self.config = config
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.model = DiTPolicy(config.model).to(self.device)
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.eval()
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.embedder = CLIPTextEmbedder(config.clip, device=self.device)
        self.task_vec = self.embedder.encode([config.prompt]).to(self.device)
        self._fast_graph: FastInferenceGraph | None = None
        self._fast_rtc_graphs: dict[int, FastRTCInferenceGraph] = {}

    def enable_fast_inference(
        self,
        compile_mode: str = "max-autotune-no-cudagraphs",
        replay_warmups: int = 24,
        warmup_obs: dict[str, Any] | None = None,
        warmup_noise: np.ndarray | None = None,
    ) -> None:
        """Compile velocity prediction and capture sample_actions in a CUDA graph."""
        if self._fast_graph is not None:
            return
        if self.device.type != "cuda":
            raise RuntimeError("fast inference requires a CUDA device")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.model.to(torch.bfloat16)
        self.model.img_backbone.set_bfloat16(True)
        self.task_vec = self.task_vec.to(device=self.device, dtype=torch.bfloat16)

        compile_kwargs: dict[str, Any] = {"dynamic": False}
        if compile_mode:
            compile_kwargs["mode"] = compile_mode
        self.model.predict_velocity = torch.compile(
            self.model.predict_velocity, **compile_kwargs
        )

        m = self.config.model
        if warmup_obs is None:
            warmup_obs = {
                "state": np.zeros(m.state_dim, dtype=np.float32),
                "images": {
                    cam: np.zeros(
                        (3, self.config.camera_height, self.config.camera_width),
                        dtype=np.uint8,
                    )
                    for cam in m.camera_keys
                },
                "prompt": self.config.prompt,
            }
        if warmup_noise is None:
            warmup_noise = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        self._fast_graph = FastInferenceGraph(self)
        self._fast_graph.capture(warmup_obs, warmup_noise, replay_warmups)

    def normalized_action_prefix(
        self,
        action_prefix: np.ndarray,
        prefix_length: int,
    ) -> np.ndarray:
        m = self.config.model
        prefix = np.asarray(action_prefix, dtype=np.float32)
        if prefix.shape == (prefix_length, m.action_dim):
            full_prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
            full_prefix[:prefix_length] = prefix
            prefix = full_prefix
        if prefix.shape != (m.chunk_length, m.action_dim):
            raise ValueError(
                f"action_prefix must have shape {(m.chunk_length, m.action_dim)} "
                f"or {(prefix_length, m.action_dim)}, got {prefix.shape}"
            )
        return normalize(prefix, self.norm_stats["actions"]).astype(np.float32, copy=False)

    def warmup_rtc(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None,
        prefix_length: int,
    ) -> None:
        m = self.config.model
        action_prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        if self._fast_graph is not None and self.device.type == "cuda":
            graph = FastRTCInferenceGraph(self, prefix_length)
            graph.capture(obs, noise, action_prefix, replay_warmups=8)
            self._fast_rtc_graphs[prefix_length] = graph
        else:
            _ = self.infer(
                obs,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    @torch.no_grad()
    def infer(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int = 0,
    ) -> np.ndarray:
        if action_prefix is None and self._fast_graph is not None:
            return self._fast_graph.infer(obs, noise)
        if action_prefix is not None and prefix_length in self._fast_rtc_graphs:
            return self._fast_rtc_graphs[prefix_length].infer(obs, noise, action_prefix)
        state = normalize(np.asarray(obs["state"], dtype=np.float32), self.norm_stats["state"])
        batch = {
            "state": torch.from_numpy(state[None]).float().to(self.device),
            "actions": torch.zeros(
                1, self.config.model.chunk_length, self.config.model.action_dim, device=self.device
            ),
            "images": {
                cam: resize_pad_normalize(obs["images"][cam]).unsqueeze(0).to(self.device)
                for cam in self.config.model.camera_keys
            },
            "task_vec_clip": self.task_vec,
        }
        noise_t = None
        if noise is not None:
            noise_t = torch.from_numpy(noise[None].astype(np.float32)).to(self.device)
        if action_prefix is None:
            actions = self.model.sample_actions(batch, num_steps=self.diffusion_steps, noise=noise_t)
        else:
            prefix_t = torch.from_numpy(
                self.normalized_action_prefix(action_prefix, prefix_length)[None]
            ).to(device=self.device, dtype=batch["state"].dtype)
            actions = self.model.sample_actions_rtc(
                batch,
                prefix_t,
                prefix_length=prefix_length,
                num_steps=self.diffusion_steps,
                noise=noise_t,
            )
        actions_np = actions[0].float().detach().cpu().numpy()
        return unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)


# Rollout.


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def video_frame(images: dict[str, np.ndarray], camera_keys: tuple[str, ...]) -> np.ndarray:
    frames = []
    for name in camera_keys:
        frame = images[name].transpose(1, 2, 0)
        frames.append(np.ascontiguousarray(frame))
    return np.concatenate(frames, axis=1)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def validate_rtc_config(config: SimEvalConfig) -> list[str]:
    if config.rtc and config.policy_backend == "lap":
        return ["RTC eval is only supported for the dit policy backend"]
    if not config.rtc:
        return []
    errors = []
    trained_max_prefix = FlowConfig().max_action_prefix
    if not 0 < config.rtc_prefix_length <= trained_max_prefix:
        errors.append(
            f"rtc_prefix_length must be in [1, {trained_max_prefix}] for this checkpoint"
        )
    if config.rtc_prefix_length > config.rtc_inference_lead_steps:
        errors.append("rtc_prefix_length must be <= rtc_inference_lead_steps")
    if config.rtc_inference_lead_steps > config.execute_chunk_dim:
        errors.append("rtc_inference_lead_steps must be <= execute_chunk_dim")
    if config.rtc and config.save_video:
        errors.append("RTC eval does not support --save-video; video rendering hides the overlap")
    return errors


def run_eval(config: SimEvalConfig) -> dict[str, Any]:
    model_errors = validate_model_config(config.model)
    config_errors = model_errors + validate_rtc_config(config)
    if config_errors:
        raise ValueError("Invalid sim eval config:\n  - " + "\n  - ".join(config_errors))

    require_mjwarp()
    device = resolve_device(config.device)
    use_lap = config.policy_backend == "lap"
    if use_lap:
        if config.lap_coarse_prior:
            # Coarse-prior seeding reads the flow EEF chunk and needs the AR
            # `reasoning` carried forward -- incompatible with the "ar" head (which
            # returns before the carry) and with predict-task (different wire key).
            if config.lap_head != "flow":
                raise ValueError("lap_coarse_prior requires lap_head='flow'")
            if config.lap_predict_task:
                raise ValueError("lap_coarse_prior and lap_predict_task are mutually exclusive")
        from abc_minimal.lap_client import LAPClientPolicy

        bridge = None
        if config.lap_bridge:
            from abc_minimal.lap_bridge import LAPSimBridge

            bridge_model = mujoco.MjModel.from_xml_string(build_scene_xml(config))
            bridge = LAPSimBridge(
                bridge_model,
                gripper_ctrl_max=active_scene(config).gripper_ctrl_max,
                rotation_delta_in_base=config.lap_rotation_delta_in_base,
                align_eef_rot=config.lap_align_eef_rot,
                anchor_on_command=config.lap_anchor_on_command,
                binarize_gripper=config.lap_binarize_gripper,
                eef_link=config.lap_eef_link,
            )
            print(f"[bridge] eef_link={config.lap_eef_link} "
                  f"align_eef_rot={config.lap_align_eef_rot} "
                  f"anchor_on_command={config.lap_anchor_on_command} "
                  f"binarize_gripper={config.lap_binarize_gripper}", flush=True)
        ckpt_path: Path | str = f"lap://{config.lap_host}:{config.lap_port}"
        policy: SimPolicy | LAPClientPolicy = LAPClientPolicy(config, bridge=bridge)
    else:
        if not config.checkpoint:
            raise ValueError("--checkpoint is required for the dit policy backend")
        ckpt_path = local_checkpoint(config.checkpoint)
        policy = SimPolicy(ckpt_path, config, device)
    env = SimEnv(
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=config.model.camera_keys,
        prompt=config.prompt,
        scene=active_scene(config),
        task=config.task,
        ladder_task=config.ladder_task,
        gpu_id=config.gpu_id,
    )
    if config.task == "ladder" and any(o.kind == "cloth" for o in get_task(config.ladder_task).objects) and not config.vanilla_physics:
        raise ValueError("cloth rungs need --vanilla-physics (MuJoCo-Warp cannot simulate flex bodies)")
    if config.task == "ladder" and get_task(config.ladder_task).goal == "wipe" and not config.vanilla_physics:
        # The evaluator hides wiped spots by writing MjData.mocap_pos; only the vanilla
        # path copies MjData into the renderer each frame.
        raise ValueError("wipe rungs need --vanilla-physics")
    rng = np.random.default_rng(config.policy_seed)
    worlds = []
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fast_inference_ready = False
    action_shape = (config.model.chunk_length, config.model.action_dim)

    def sample_noise(generator: np.random.Generator) -> np.ndarray:
        return generator.standard_normal(action_shape, dtype=np.float32)

    try:
        for world_index in range(config.num_worlds):
            video = None
            video_path = None
            t0 = time.perf_counter()
            seed = int(config.seed + world_index)
            obs = env.reset(seed=seed)
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()
            if not use_lap and config.fast_inference and not fast_inference_ready:
                warmup_rng = np.random.default_rng(config.policy_seed)
                warmup_noise = sample_noise(warmup_rng)
                t_fast = time.perf_counter()
                policy.enable_fast_inference(
                    compile_mode=config.fast_compile_mode,
                    warmup_obs=obs,
                    warmup_noise=warmup_noise,
                )
                torch.cuda.synchronize()
                print(
                    f"fast inference ready in {time.perf_counter() - t_fast:.1f}s",
                    flush=True,
                )
                fast_inference_ready = True
            if config.save_video:
                import imageio.v2 as imageio

                video_path = out_dir / f"world_{world_index:03d}.mp4"
                video = imageio.get_writer(str(video_path), fps=config.video_fps, macro_block_size=1)
                video.append_data(video_frame(obs["images"], config.model.camera_keys))
            final_eval = env.evaluate_vanilla() if config.vanilla_physics else env.evaluate()
            steps = 0
            chunk_metrics = []
            rtc = None
            try:
                obs_fn = env.obs_vanilla_state if config.vanilla_physics else env.obs
                eval_fn = env.evaluate_vanilla if config.vanilla_physics else env.evaluate
                step_fn = env.step_one_vanilla if config.vanilla_physics else env.step_one
                render_fn = (
                    env.render_cameras_vanilla_state
                    if config.vanilla_physics
                    else env.render_cameras
                )
                noise = sample_noise(rng)
                t_infer = time.perf_counter()
                actions = policy.infer(obs, noise=noise)
                current_infer_s = time.perf_counter() - t_infer
                if config.rtc:
                    rtc_warmup_rng = np.random.default_rng(config.policy_seed)
                    warmup_noise = sample_noise(rtc_warmup_rng)
                    t_rtc_warm = time.perf_counter()
                    policy.warmup_rtc(obs, warmup_noise, config.rtc_prefix_length)
                    print(
                        f"rtc inference ready in {time.perf_counter() - t_rtc_warm:.1f}s",
                        flush=True,
                    )
                    rtc = _RTCManager(
                        policy,
                        prefix_length=config.rtc_prefix_length,
                        inference_lead_steps=config.rtc_inference_lead_steps,
                        execute_chunk_dim=config.execute_chunk_dim,
                    )
                for chunk in range(config.num_chunks):
                    t_chunk = time.perf_counter()
                    chunk_infer_s = current_infer_s
                    t_steps = time.perf_counter()
                    rtc_started = False
                    rtc_ready = None
                    rtc_infer_s = None
                    rtc_obs_s = 0.0
                    lead_index = config.execute_chunk_dim - config.rtc_inference_lead_steps
                    for action_index, action in enumerate(actions[: config.execute_chunk_dim]):
                        if (
                            rtc is not None
                            and chunk + 1 < config.num_chunks
                            and action_index == lead_index
                        ):
                            t_obs = time.perf_counter()
                            next_obs = obs_fn()
                            rtc_obs_s = time.perf_counter() - t_obs
                            next_noise = sample_noise(rng)
                            rtc.start(next_obs, actions, next_noise)
                            rtc_started = True
                        step_fn(action)
                        final_eval = eval_fn()
                        steps += 1
                        if video is not None and steps % config.video_every_n_actions == 0:
                            video.append_data(video_frame(render_fn(), config.model.camera_keys))
                        if final_eval["ever_success"]:
                            break
                    steps_s = time.perf_counter() - t_steps
                    if final_eval["ever_success"]:
                        break
                    if rtc is None and chunk + 1 < config.num_chunks:
                        t_obs = time.perf_counter()
                        obs = obs_fn()
                        rtc_obs_s = time.perf_counter() - t_obs
                        noise = sample_noise(rng)
                        t_infer = time.perf_counter()
                        actions = policy.infer(obs, noise=noise)
                        current_infer_s = time.perf_counter() - t_infer
                    elif rtc_started:
                        actions, rtc_infer_s, rtc_ready = rtc.get()
                        current_infer_s = rtc_infer_s
                    logged_infer_s = rtc_infer_s if rtc_infer_s is not None else chunk_infer_s
                    metric = {
                        "chunk": chunk,
                        "infer_s": float(logged_infer_s),
                        "current_chunk_infer_s": float(chunk_infer_s),
                        "rtc_next_infer_s": (
                            float(rtc_infer_s) if rtc_infer_s is not None else None
                        ),
                        "steps_s": float(steps_s),
                        "obs_render_s": float(rtc_obs_s),
                        "wall_s": float(time.perf_counter() - t_chunk),
                        "rtc_ready_at_chunk_end": rtc_ready,
                        "bottles": int(final_eval["num_solved"]),
                        "max_bottles": int(final_eval["max_solved_so_far"]),
                    }
                    chunk_metrics.append(metric)
                    if config.log_every_chunk:
                        rtc_text = (
                            f" rtc_ready_at_chunk_end={rtc_ready}"
                            if config.rtc
                            else ""
                        )
                        print(
                            f"world={world_index:03d} chunk={chunk:02d} "
                            f"infer={metric['infer_s'] * 1000:.0f}ms "
                            f"steps={metric['steps_s'] * 1000:.0f}ms "
                            f"render={metric['obs_render_s'] * 1000:.0f}ms "
                            f"solved={final_eval['num_solved']}/{final_eval['num_active']} "
                            f"success={final_eval['ever_success']}{rtc_text}",
                            flush=True,
                        )
            finally:
                if rtc is not None:
                    rtc.close()
                if video is not None:
                    video.close()

            world = {
                "world_index": world_index,
                "world_seed": seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "steps": steps,
                "wall_s": time.perf_counter() - t0,
                "chunk_metrics": chunk_metrics,
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "video_path": str(video_path) if video_path is not None else None,
                # Per-chunk decoded langact for this episode, when the server returns
                # one. Empty list for policies that never decode reasoning.
                "reasoning_trace": list(getattr(policy, "reasoning_trace", []) or []),
            }
            worlds.append(world)
            print(
                f"world={world_index:03d} done placed={world['success']} "
                f"picked={final_eval.get('picked')} "
                f"max_lift={final_eval.get('max_lift_m', float('nan')):.3f} "
                f"solved={final_eval['max_solved_so_far']}/{final_eval['num_active']} "
                f"steps={steps}",
                flush=True,
            )
    finally:
        env.close()

    success = np.asarray([w["success"] for w in worlds], dtype=bool)
    rewards = np.asarray([w["reward"] for w in worlds], dtype=np.float32)
    max_solved = np.asarray(
        [w["final_task_eval"]["max_solved_so_far"] for w in worlds],
        dtype=np.float32,
    )
    max_lift = np.asarray(
        [w["final_task_eval"].get("max_lift_m", float("nan")) for w in worlds],
        dtype=np.float32,
    )
    picked = np.asarray(
        [bool(w["final_task_eval"].get("picked", False)) for w in worlds], dtype=bool
    )
    placed = np.asarray(
        [bool(w["final_task_eval"].get("placed", False)) for w in worlds], dtype=bool
    )
    lifted = picked
    summary = {
        "format": f"abc_minimal_{config.task}_eval/v1",
        "task": config.task,
        # Ladder rung ("" for put_bottles/mug_flip). `prompt` below is the task's
        # template; p09's per-world prompt is in each world's randomization.
        "ladder_task": config.ladder_task,
        "goal": get_task(config.ladder_task).goal if config.task == "ladder" else None,
        "checkpoint": str(ckpt_path),
        # Which scene file (lighting variant) this ran on; ABC_SCENE_XML overrides it.
        "scene_xml": str(SCENE_XML),
        "renderer": "gl" if (env.has_flex or os.environ.get("ABC_RENDER_GL", "0") == "1") else "mjwarp",
        "prompt": config.prompt,
        "config": asdict(config),
        "resolved_device": device,
        "success_rate": float(success.mean()) if success.size else None,
        "num_success": int(success.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(rewards.mean()) if rewards.size else None,
        "mean_max_solved": float(max_solved.mean()) if max_solved.size else None,
        # Two separate metrics, not one score with a caveat:
        #   pick_rate  -- got the bottle off the table at all
        #   place_rate -- and finished with it inside the bin, still elevated
        # `success_rate` is an alias of place_rate, kept so existing readers of the
        # summary do not silently change meaning.
        "pick_rate": float(picked.mean()) if picked.size else None,
        "num_picked": int(picked.sum()),
        "place_rate": float(placed.mean()) if placed.size else None,
        "num_placed": int(placed.sum()),
        # Picked but not placed: the "lifted then dropped" failure mode.
        "num_picked_not_placed": int((picked & ~placed).sum()),
        "lift_rate": float(lifted.mean()) if lifted.size else None,
        "num_lifted": int(lifted.sum()),
        "mean_max_lift_m": float(max_lift.mean()) if max_lift.size else None,
        "worlds": worlds,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print(
        f"summary: success_rate={summary['success_rate']} "
        f"num_success={summary['num_success']}/{summary['num_worlds']} "
        f"mean_reward={summary['mean_reward']} "
        f"mean_max_solved={summary['mean_max_solved']} "
        f"pick={summary['num_picked']}/{summary['num_worlds']} "
        f"place={summary['num_placed']}/{summary['num_worlds']} "
        f"picked_not_placed={summary['num_picked_not_placed']} "
        f"mean_max_lift_m={summary['mean_max_lift_m']}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    # Whether the policy ever instructed a gripper close, and whether the
    # bridge passed that instruction through (see LAPClientPolicy._log_action_rows).
    report = getattr(policy, "report_gripper_summary", None)
    if callable(report):
        report()
    # IK health: a grossly failed solve must be attributable to the CONTROLLER, not
    # read as bad policy output.
    if bridge is not None:
        bridge.report_ik_diagnostics()
    return summary


def main(config: SimEvalConfig) -> None:
    run_eval(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(SimEvalConfig))
