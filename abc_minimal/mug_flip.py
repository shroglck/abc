"""Turn-the-mug-right-side-up sim task, built on ABC's put-bottles scene.

ABC ships exactly one sim scene (`assets/put_bottles/put_bottle.xml`). Everything
in it except the manipulated objects -- the two YAM arms, the play table, the four
lights, the six cameras, the actuators and the `home` keyframe -- is task-neutral,
and every appearance decision in it (the dark surround, the near-white skybox, the
camera intrinsics) is part of how policies are scored in this benchmark. So this
task does NOT fork the XML: it loads the same file, deletes the bottle/bin bodies
and their assets, and grafts mugs in. A robot or lighting change upstream reaches
both tasks with no second file to keep in step.

Real counterpart: ABC-130k's `turn the mug right side up` (673 episodes), which is
in the `abc_yam` training mixture, so the prompt below is the real task string
verbatim rather than a `sim ...` variant.

Object: YCB 025_mug, centered on its own origin with +z out of the opening. The
visual is the scanned mesh; the collision is a hollow shell -- a floor disk and a
ring of wall boxes -- plus a capsule handle. MuJoCo convexifies mesh collisions, so
the scan's own hull would fill the cup and leave the outer diameter as the only
graspable feature; the shell keeps the rim and the wall pinchable, which is how a
mug this size is actually turned over.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from abc_minimal.config import MugFlipSimConfig

ROOT = Path(__file__).resolve().parents[1]
# Same override as eval_policy.SCENE_XML (ABC_SCENE_XML env var): the two must agree, or a
# --scene-xml-file run silently evaluates mug_flip on the default lighting while the summary
# records the override (happened 2026-09-16, jobs 3890337-9).
SCENE_XML = Path(os.environ.get("ABC_SCENE_XML") or (ROOT / "assets" / "put_bottles" / "put_bottle.xml"))
MUG_DIR = ROOT / "assets" / "mug_flip" / "mug"

PROMPT = "turn the mug right side up"


def _fmt(values) -> str:
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


def upside_down_quat(yaw: float) -> np.ndarray:
    """Mug opening pointing at the table, rotated `yaw` about world +z."""
    flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # pi about +x
    q = _quat_mul(_quat_yaw(yaw), flip)
    return q / np.linalg.norm(q)


def mug_up_axis(quat: np.ndarray) -> np.ndarray:
    """World-frame image of the mug's local +z (its opening direction)."""
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float64,
    )


def mug_spawn_z(scene: MugFlipSimConfig, scale: float) -> float:
    return float(scene.table_z + scene.mug_half_height * scale + scene.mug_spawn_clearance)


def sample_mug_xy(
    rng: np.random.Generator,
    scene: MugFlipSimConfig,
    occupied: list[np.ndarray],
) -> np.ndarray:
    x0, x1, y0, y1 = scene.mug_xy_range
    candidate = None
    for _ in range(scene.mug_sample_attempts):
        candidate = np.array(
            [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))], dtype=np.float64
        )
        if all(np.linalg.norm(candidate - c) > scene.mug_min_separation for c in occupied):
            return candidate
    if candidate is None:
        raise RuntimeError("Could not sample a mug position inside the spawn box")
    return candidate


# Scene construction.


def _strip_bottles_and_bin(root: ET.Element) -> None:
    """Delete the put-bottles objects and every asset only they referenced."""
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for body in list(worldbody):
        name = body.get("name", "") if body.tag == "body" else ""
        if name == "bin_container" or re.fullmatch(r"bottle_\d+", name):
            worldbody.remove(body)

    asset = root.find("asset")
    assert asset is not None
    dead = (
        lambda n: n.startswith("bottle_")
        or n.startswith("water_bottle_")
        or n.startswith("bottle")
    )
    for element in list(asset):
        if element.tag in ("mesh", "material", "texture") and dead(element.get("name", "")):
            asset.remove(element)

    # The bottle/bin defaults now match nothing; leaving them is harmless, but the
    # class list is part of what a reader checks against the scene, so drop them.
    default = root.find("default")
    if default is not None:
        for element in list(default):
            if element.get("class", "") in (
                "bottle_visual",
                "bottle_collision",
                "water_bottle_bin_col",
            ):
                default.remove(element)


def _add_mug_assets(root: ET.Element, mug_scales: np.ndarray) -> None:
    asset = root.find("asset")
    assert asset is not None
    # Absolute paths: the compiler's meshdir/texturedir point at the put-bottles
    # asset tree, which these files deliberately live outside of.
    ET.SubElement(
        asset,
        "texture",
        {"name": "mug_tex", "type": "2d", "file": str((MUG_DIR / "texture_map.png").resolve())},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "mug_mat", "texture": "mug_tex", "specular": "0.3", "shininess": "0.3"},
    )
    for idx, scale in enumerate(mug_scales):
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": f"mug_{idx}_vis",
                "file": str((MUG_DIR / "textured.obj").resolve()),
                "scale": _fmt([scale] * 3),
            },
        )


def _add_mug_body(
    worldbody: ET.Element,
    scene: MugFlipSimConfig,
    idx: int,
    scale: float,
) -> None:
    body = ET.SubElement(worldbody, "body", {"name": f"mug_{idx}", "pos": "0 0 0"})
    ET.SubElement(body, "freejoint", {"name": f"mug_{idx}_joint"})
    mass = scene.mug_mass
    radius = scene.mug_radius * scale
    half_height = scene.mug_half_height * scale
    # Thin-walled cup, treated as a solid cylinder for inertia: close enough at
    # this mass, and MuJoCo only needs it to be positive-definite and plausible.
    ixx = mass * (3.0 * radius**2 + (2.0 * half_height) ** 2) / 12.0
    izz = 0.5 * mass * radius**2
    ET.SubElement(
        body,
        "inertial",
        {"pos": "0 0 0", "mass": f"{mass:.8g}", "diaginertia": _fmt([ixx, ixx, izz])},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"mug_{idx}_vis",
            "type": "mesh",
            "mesh": f"mug_{idx}_vis",
            "material": "mug_mat",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "density": "0",
        },
    )
    # Collision params copied from the bottles' collision meshes so contact with
    # the fingers behaves the way it does in the scored scene.
    collision = {
        "density": "0",
        "group": "3",
        "rgba": "0 0 0 0",
        "friction": "3.0 0.03 0.003",
        "condim": "6",
        "solref": "0.004 1",
        "solimp": "0.998 0.998 0.001",
        "priority": "1",
    }
    floor_half = 0.5 * scene.mug_floor_thickness * scale
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"mug_{idx}_col_floor",
            "type": "cylinder",
            "size": _fmt([radius, floor_half]),
            "pos": _fmt([0.0, 0.0, -half_height + floor_half]),
            **collision,
        },
    )
    wall = scene.mug_wall_thickness * scale
    segments = scene.mug_wall_segments
    # Boxes tangent to the inner circumference, overlapped slightly so the ring has
    # no seams for a fingertip to catch in.
    chord_half = radius * math.tan(math.pi / segments) * 1.05
    wall_half_z = half_height - floor_half
    for seg in range(segments):
        theta = 2.0 * math.pi * seg / segments
        centre = radius - 0.5 * wall
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"mug_{idx}_col_wall{seg}",
                "type": "box",
                "size": _fmt([0.5 * wall, chord_half, wall_half_z]),
                "pos": _fmt([centre * math.cos(theta), centre * math.sin(theta), floor_half]),
                "euler": _fmt([0.0, 0.0, theta]),
                **collision,
            },
        )
    hx = scene.mug_handle_x * scale
    hz = scene.mug_handle_half_height * scale
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"mug_{idx}_col_handle",
            "type": "capsule",
            "fromto": _fmt([hx, 0.0, -hz, hx, 0.0, hz]),
            "size": f"{scene.mug_handle_radius * scale:.8g}",
            **collision,
        },
    )


def mug_scene_xml(scene: MugFlipSimConfig, mug_scales: np.ndarray) -> str:
    root = ET.fromstring(SCENE_XML.read_text())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
        compiler.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))

    _strip_bottles_and_bin(root)
    _add_mug_assets(root, mug_scales)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for idx, scale in enumerate(mug_scales):
        _add_mug_body(worldbody, scene, idx, float(scale))

    if os.environ.get("ABC_SCENE_BRIGHT", "0") == "1":
        # Same opt-in appearance switch the bottles scene has; imported here (not at
        # module scope) because eval_policy imports this module.
        from abc_minimal.eval_policy import _brighten_surround

        _brighten_surround(root)

    return ET.tostring(root, encoding="unicode")


# Metric.


class MugFlipEvaluator:
    """A mug counts once it is upright, resting on the table, and stays that way.

    "Upright" alone would fire while the mug is still in the gripper on the way
    down, and the episode loop stops at the first success -- so the metric also
    requires the mug's height to match a mug standing on the table, and requires
    the whole predicate to hold for `upright_hold_evals` consecutive control steps.
    """

    def __init__(self, model: mujoco.MjModel, scene: MugFlipSimConfig, mug_scales: np.ndarray):
        self.model = model
        self.scene = scene
        self.mug_scales = np.asarray(mug_scales, dtype=np.float64)
        self.mug_names, self.mug_qpos_addrs = self._mug_addrs()
        self.hold = np.zeros(len(self.mug_names), dtype=np.int32)
        self.max_solved = 0
        self.ever_success = False

    def reset(self) -> None:
        self.hold[:] = 0
        self.max_solved = 0
        self.ever_success = False

    def _mug_addrs(self) -> tuple[list[str], np.ndarray]:
        entries = []
        for joint_id in range(self.model.njnt):
            name = self.model.jnt(joint_id).name
            match = re.fullmatch(r"mug_(\d+)_joint", name or "")
            if match:
                idx = int(match.group(1))
                entries.append((idx, f"mug_{idx}", int(self.model.jnt_qposadr[joint_id])))
        entries.sort()
        return [e[1] for e in entries], np.asarray([e[2] for e in entries], dtype=np.int32)

    def evaluate(self, qpos: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(qpos, dtype=np.float32)
        scene = self.scene
        cos_limit = math.cos(scene.upright_max_tilt_rad)

        up_z, height_err = [], []
        for i, adr in enumerate(self.mug_qpos_addrs):
            pos = qpos[adr : adr + 3]
            quat = qpos[adr + 3 : adr + 7]
            up_z.append(float(mug_up_axis(quat)[2]))
            resting_z = scene.table_z + scene.mug_half_height * float(self.mug_scales[i])
            height_err.append(abs(float(pos[2]) - resting_z))
        up_z = np.asarray(up_z, dtype=np.float64)
        height_err = np.asarray(height_err, dtype=np.float64)

        instant = (up_z >= cos_limit) & (height_err <= scene.rest_z_tol)
        self.hold = np.where(instant, self.hold + 1, 0)
        solved = self.hold >= scene.upright_hold_evals

        num = int(solved.sum())
        active = len(self.mug_names)
        self.max_solved = max(self.max_solved, num)
        success = num == active and active > 0
        self.ever_success = self.ever_success or success
        return {
            "reward": float(num / max(active, 1)),
            "success": bool(success),
            "ever_success": bool(self.ever_success),
            "num_solved": num,
            "num_active": active,
            "max_solved_so_far": int(self.max_solved),
            "solved_mask": [bool(v) for v in solved.tolist()],
            "upright_now_mask": [bool(v) for v in instant.tolist()],
            "mug_names": list(self.mug_names),
            "mug_up_z": [float(v) for v in up_z.tolist()],
            "mug_height_err": [float(v) for v in height_err.tolist()],
            "mug_hold_steps": [int(v) for v in self.hold.tolist()],
            "best_up_z": float(up_z.max()) if active else 0.0,
        }
