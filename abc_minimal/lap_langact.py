"""Decode LAP language actions into a 14-D bimanual EEF delta (client side).

LAP's AR head returns a chain-of-thought string like::

    Left arm: move forward 1 cm, move up 1 cm, close gripper.
    Right arm: move back 2 cm, tilt left 5 degrees, open gripper

LAP's own decoder (``LanguageActionFormat.parse_language_to_deltas``) regexes the
*whole* string in one pass, so for a bimanual sentence it sums both arms into a
single 7-D action and drops the per-arm split. Here we split on ``Right arm:``
and parse each arm separately, reproducing LAP's verbose grammar and its
end-effector->base-frame transform (``VERBOSE_EEF_WITH_ROTATION_FORMAT`` with the
DEFAULT frame convention), to recover a faithful 14-D bimanual delta
``[L: dx,dy,dz, droll,dpitch,dyaw, grip | R: ...]`` in the robot base frame.

Ported from ego-lap ``lang_action_formats.py`` + ``frame_transforms.py`` +
``frame_conventions.py`` (DEFAULT: pos_sign=(1,-1,-1), rot_extra_sign=(1,1,1),
rotation_applied=False -> the (1,-1,-1) unrotated-wrist sign). Validated against
the server's own decode (the per-arm sum matches LAP's single 7-D output).
"""

from __future__ import annotations

import re

import numpy as np

from abc_minimal.lap_bridge import rpy_to_mat

# DEFAULT FrameConvention (ego-lap frame_conventions.py).
_POS_SIGN = np.array([1.0, -1.0, -1.0])
_ROT_EXTRA_SIGN = np.array([1.0, 1.0, 1.0])
_UNROTATED_WRIST_SIGN = np.array([1.0, -1.0, -1.0])  # rotation_applied=False

_MOVE_RE = re.compile(
    r"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*cm)?",
    re.IGNORECASE,
)
_ROT_RE = re.compile(
    r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|"
    r"rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
    re.IGNORECASE,
)
_GRIP_RE = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """6-D rotation (first two columns) -> 3x3 matrix (Gram-Schmidt)."""
    a1 = np.asarray(rot6d[0:3], dtype=float)
    a2 = np.asarray(rot6d[3:6], dtype=float)
    b1 = a1 / np.linalg.norm(a1)
    a2o = a2 - np.dot(b1, a2) * b1
    b2 = a2o / np.linalg.norm(a2o)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rpy(mat: np.ndarray) -> np.ndarray:
    """Inverse of rpy_to_mat: extrinsic xyz Euler from a rotation matrix."""
    m = np.asarray(mat, dtype=float)
    b = np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2]))
    a = np.arctan2(m[2, 1], m[2, 2])
    c = np.arctan2(m[1, 0], m[0, 0])
    return np.array([a, b, c])


def _parse_arm_movement(text: str) -> tuple[np.ndarray, float | None]:
    """Parse one arm's verbose language into a semantic EEF delta + gripper."""
    text = text.replace("slightly", "1.5 cm").replace("moderately", "5 cm").replace("a lot", "10 cm")
    move = np.zeros(6, dtype=float)
    dx = dy = dz = 0.0
    for m in _MOVE_RE.finditer(text):
        d = m.group(1).lower()
        v = float(m.group(2)) if m.group(2) is not None else 0.0
        if d == "forward":
            dx += v
        elif d in ("backward", "back"):
            dx -= v
        elif d == "left":
            dy += v
        elif d == "right":
            dy -= v
        elif d == "up":
            dz += v
        elif d == "down":
            dz -= v
    move[:3] = np.array([dx, dy, dz]) / 100.0

    dr = dp = dyw = 0.0
    for m in _ROT_RE.finditer(text):
        rt = m.group(1).lower()
        v = float(m.group(2))
        if rt == "tilt left":
            dr += v
        elif rt == "tilt right":
            dr -= v
        elif rt in ("tilt down", "tilt back"):
            dp += v
        elif rt in ("tilt up", "tilt forward"):
            dp -= v
        elif rt == "rotate counterclockwise":
            dyw += v
        elif rt == "rotate clockwise":
            dyw -= v
    move[3:6] = np.array([dr, dp, dyw]) * np.pi / 180.0

    grip: float | None = None
    gm = _GRIP_RE.search(text)
    if "open gripper" in text.lower():
        grip = 1.0
    elif "close gripper" in text.lower():
        grip = 0.0
    elif gm:
        grip = float(gm.group(1))
    return move, grip


def _eef_to_base(movement: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """EEF-frame semantic delta -> base-frame delta (LAP decode path, DEFAULT conv)."""
    r_eef_to_base = rot6d_to_rotmat(rot6d)
    out = np.zeros(6, dtype=float)
    # Position: undo the semantic convention, then rotate eef->base.
    dpos_eef = movement[:3] / _POS_SIGN  # pos_perm is identity
    out[:3] = r_eef_to_base @ dpos_eef
    # Rotation: undo conventions, rotate the delta into the base frame.
    drot_eef = (movement[3:6] / _ROT_EXTRA_SIGN) * _UNROTATED_WRIST_SIGN
    r_delta = r_eef_to_base @ rpy_to_mat(drot_eef) @ r_eef_to_base.T
    out[3:6] = mat_to_rpy(r_delta)
    return out


def split_arms(reasoning: str) -> tuple[str, str]:
    """Split a bimanual reasoning string into (left_text, right_text)."""
    idx = reasoning.lower().find("right arm")
    if idx < 0:
        return reasoning, ""
    return reasoning[:idx], reasoning[idx:]


def apply_raw_action_axis_convention(actions: np.ndarray, dataset_name: str) -> np.ndarray:
    """Undo a dataset's raw rotation-axis convention (mirror of ego-lap's copy).

    MolmoAct2-YAM stores each arm's rotation triple as ``[pitch, roll, -yaw]``
    relative to the repo's ``[roll, pitch, yaw]`` language convention, and ego-lap's
    ActionProcessor applies that remap on the way *out* (raw tensor -> langact text).
    The map is an involution, so applying it again inverts it. Ported from ego-lap
    ``datasets/utils/frame_conventions.apply_raw_action_axis_convention``; keep the
    two in sync.
    """
    if "molmoact2_yam_dataset" not in (dataset_name or ""):
        return actions
    out = np.asarray(actions, dtype=float).copy()
    for offset in (3, 10) if out.shape[-1] >= 13 else (3,):
        rot = out[..., offset : offset + 3].copy()
        out[..., offset : offset + 3] = rot[..., [1, 0, 2]]
        out[..., offset + 2] *= -1.0
    return out


def parse_bimanual_langact(
    reasoning: str,
    state20: np.ndarray,
    *,
    eef_frame: bool = True,
    dataset_name: str = "",
) -> np.ndarray:
    """Reasoning string + 20-D EEF state -> 14-D base-frame bimanual EEF delta.

    Gripper defaults to the current state value when an arm names no gripper.

    ``eef_frame`` selects which training-time convention produced the text:

    * ``True``  -- the langact was written in the semantic EEF frame
      (``transform_actions_to_eef_frame``), so undo the sign convention and rotate
      eef->base. This is what ego-lap does for *single-arm* samples.
    * ``False`` -- the langact numerics stayed in the robot/arm-base frame. This is
      the **bimanual** path: ``choose_action_frame`` returns
      ``egocentric_action_view(apply_base_convention=True)`` with
      ``use_eef_transform=False``, so only the prompt *label* says "egocentric
      frame" while the numbers were never rotated. Rotating them here would inject
      a spurious frame change that training never applied.

    ``dataset_name`` additionally inverts a dataset's raw rotation-axis convention
    (MolmoAct2-YAM); empty means none.
    """
    state20 = np.asarray(state20, dtype=float)
    left_text, right_text = split_arms(reasoning)
    out = np.zeros(14, dtype=np.float32)
    for sl, text, st in (
        (slice(0, 7), left_text, state20[0:10]),
        (slice(7, 14), right_text, state20[10:20]),
    ):
        move, grip = _parse_arm_movement(text)
        if eef_frame:
            move = _eef_to_base(move, st[3:9])
        if grip is None:
            grip = float(st[9])
        out[sl] = np.concatenate([move, [grip]])
    return apply_raw_action_axis_convention(out, dataset_name).astype(np.float32)
