"""FK/IK bridge between the joint-space put-bottles sim and LAP's EEF action space.

LAP (ego-lap) consumes a 20-D bimanual EEF *state* and emits 14-D EEF *action*
chunks, while the put-bottles sim observes/commands 14-D joint space
``[L: joint1..6, gripper | R: joint1..6, gripper]``. This module converts
between the two using the sim's own YAM kinematics (MuJoCo), so the loop is
internally consistent:

* ``state20_from_joints`` -- forward kinematics: each arm's 6 joint angles ->
  grasp-site pose expressed in that arm's base frame ->
  ``[xyz(3), rot6d(6), grip(1)]`` (per arm), concatenated to 20-D.
* ``joint_chunk_from_eef`` -- for each predicted EEF action, add the delta to
  the arm's pose at chunk start, solve damped-least-squares IK for the 6 joint
  targets, and pass the gripper command through. Output matches the sim action
  layout ``[L: joint1..6, gripper | R: ...]``. "Chunk start" is the last
  *commanded* pose (``note_command``/``anchor_on_command``), matching how the
  training deltas were defined; see that method's docstring.

Convention notes (ported verbatim from the i2rt source the YAM data was made
with -- see ``CANONICAL_BASE_ROT``/``CANONICAL_AXIS_FLIP`` below):
- State rotation = the deployed YAM encoder's: conjugate the grasp_site FK
  rotation by the axis flip (and canonical base rot),
  ``R_user = R_af^T @ R_cb^T @ R_raw @ R_af``; rot6d = first two columns of
  ``R_user`` (== euler(extrinsic "xyz")->rot6d of the same matrix). Position is
  the raw arm-base FK xyz (the encoder rotates only orientation).
- Action deltas (i2rt ``_solve_ik_from_action``, ``control_mode="delta"``):
  position adds in the base frame; the user-frame euler delta is mapped to the
  robot frame as ``delta_rot = R_cb @ (R_af @ R(delta_rpy) @ R_af^T) @ R_cb^T``
  then applied to the raw grasp_site rotation. ``rotation_delta_in_base=True``
  pre-multiplies (i2rt ``rotation_frame="world"``); False post-multiplies
  ("local"). Euler order extrinsic "xyz".
- Gripper is treated as an absolute command in [0, 1] (1 = open), matching LAP.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = 6
PER_ARM_STATE = 10  # xyz(3) + rot6d(6) + grip(1)
PER_ARM_ACTION = 7  # dxyz(3) + drpy(3) + grip(1)

# Real-YAM EEF rotation convention, ported verbatim from the i2rt source the
# training data was produced with (shroglck/lihzha fork,
# ``i2rt/ee_pose_env_real.py``: ``_canonical_base_rot``/``_canonical_axis_flip``;
# ``scripts/real_robot/yam_env.py``: ``_arm_state``). The deployed encoder takes
# the grasp_site FK rotation ``R_raw`` (arm-base frame) and reports
#
#     R_user = CANONICAL_AXIS_FLIP^T @ CANONICAL_BASE_ROT^T @ R_raw @ CANONICAL_AXIS_FLIP
#
# then euler(extrinsic xyz)->rot6d. For YAM ``_canonical_base_rot`` is identity
# and ``_canonical_axis_flip`` is ``diag(-1, 1, -1)`` (symmetric/involutory), so
# this is a pure *conjugation* of R_raw by the axis flip -- NOT a single
# left-multiply. Position is left in the raw arm-base FK frame (the deployed
# encoder applies only ``base_transform`` to xyz, which defaults to identity).
# The abc sim's vendored YAM model shares i2rt's exact grasp_site/link chain, so
# the FK frames coincide and no further offset is needed (verified by sys-id).
CANONICAL_BASE_ROT = np.eye(3)
CANONICAL_AXIS_FLIP = np.diag([-1.0, 1.0, -1.0])


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: rot6d (first two columns) -> orthonormal rotation matrix."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c1 = rot6d[:3] / max(np.linalg.norm(rot6d[:3]), 1e-9)
    c2 = rot6d[3:] - np.dot(c1, rot6d[3:]) * c1
    c2 = c2 / max(np.linalg.norm(c2), 1e-9)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=1)


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler angles -> rotation matrix (R = Rz @ Ry @ Rx)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class _ArmKin:
    base_bid: int
    eef_id: int  # site id when eef_is_site, else body id
    eef_is_site: bool
    qposadr: np.ndarray  # (6,) int
    dofadr: np.ndarray  # (6,) int
    lo: np.ndarray  # (6,) joint lower limits
    hi: np.ndarray  # (6,) joint upper limits


class LAPSimBridge:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        gripper_ctrl_max: float,
        ik_iters: int = 100,
        ik_tol: float = 1e-4,
        ik_damping: float = 0.1,
        ik_step: float = 1.0,
        rotation_delta_in_base: bool = False,
        binarize_gripper: bool = False,
        align_eef_rot: bool = False,
        anchor_on_command: bool = True,
        eef_link: str = "grasp_site",
        canonical_base_rot: np.ndarray = CANONICAL_BASE_ROT,
        canonical_axis_flip: np.ndarray = CANONICAL_AXIS_FLIP,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.gripper_ctrl_max = gripper_ctrl_max
        self.ik_iters = ik_iters
        self.ik_tol = ik_tol
        self.ik_damping = ik_damping
        self.ik_step = ik_step
        self.rotation_delta_in_base = rotation_delta_in_base
        self.binarize_gripper = binarize_gripper
        self.anchor_on_command = anchor_on_command
        self.eef_link = eef_link
        # Last joint target actually commanded to the sim (set via `note_command`).
        # None until the first chunk of an episode, and after `reset`.
        self._command_joints: np.ndarray | None = None
        self.arms = {arm: self._resolve_arm(arm) for arm in ("left", "right")}
        # Real-YAM rotation convention (i2rt canonical-base + axis-flip), applied
        # to the grasp_site FK rotation as R_user = R_af^T @ R_cb^T @ R_raw @ R_af.
        # When align_eef_rot is False both reduce to identity (raw grasp_site
        # frame, for inspection only).
        if align_eef_rot:
            self.R_cb = np.asarray(canonical_base_rot, dtype=np.float64)
            self.R_af = np.asarray(canonical_axis_flip, dtype=np.float64)
        else:
            self.R_cb = np.eye(3)
            self.R_af = np.eye(3)

    def _resolve_arm(self, arm: str) -> _ArmKin:
        m = self.model
        qposadr, dofadr, lo, hi = [], [], [], []
        for j in range(1, ARM_JOINTS + 1):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{j}")
            if jid < 0:
                raise ValueError(f"joint not found: {arm}_joint{j}")
            qposadr.append(int(m.jnt_qposadr[jid]))
            dofadr.append(int(m.jnt_dofadr[jid]))
            lo.append(float(m.jnt_range[jid, 0]))
            hi.append(float(m.jnt_range[jid, 1]))
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm")
        if bid < 0:
            raise ValueError(f"missing base body for {arm}")
        # The end-effector frame is per-DATASET, not per-robot: abc_yam's converter
        # wrote FK at grasp_site, molmoact2_yam's sidecar wrote FK at link_6, and the
        # two are 13.47 cm / 90 deg apart (grasp_site hangs off link_6 at
        # pos="0 0 0.1347" quat="1 0 0 -1"). Serving a checkpoint against the wrong one
        # puts every state it sees off its training manifold. Site first, then body,
        # mirroring abc_pipeline.fk.YamFK.
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_{self.eef_link}")
        if sid >= 0:
            eef_id, is_site = sid, True
        else:
            eef_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{self.eef_link}")
            is_site = False
            if eef_id < 0:
                raise ValueError(f"no site or body named {arm}_{self.eef_link}")
        return _ArmKin(bid, eef_id, is_site, np.array(qposadr), np.array(dofadr),
                       np.array(lo), np.array(hi))

    def _eef_pos_mat(self, ak: _ArmKin) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        if ak.eef_is_site:
            return d.site_xpos[ak.eef_id], d.site_xmat[ak.eef_id].reshape(3, 3)
        return d.xpos[ak.eef_id], d.xmat[ak.eef_id].reshape(3, 3)

    # -- forward kinematics ---------------------------------------------------

    def _set_arm_joints(self, arm: str, joints: np.ndarray) -> None:
        ak = self.arms[arm]
        self.data.qpos[ak.qposadr] = joints

    def _site_pose_in_base(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        ak = self.arms[arm]
        d = self.data
        base_pos = d.xpos[ak.base_bid].copy()
        base_mat = d.xmat[ak.base_bid].reshape(3, 3).copy()
        eef_pos, eef_mat = self._eef_pos_mat(ak)
        pos_b = base_mat.T @ (eef_pos.copy() - base_pos)
        rot_b = base_mat.T @ eef_mat.copy()
        return pos_b, rot_b

    def state20_from_joints(self, state14: np.ndarray) -> np.ndarray:
        """[L j1..6, L grip, R j1..6, R grip] (14) -> 20-D EEF state."""
        state14 = np.asarray(state14, dtype=np.float64)
        self.data.qpos[:] = 0.0
        self._set_arm_joints("left", state14[0:6])
        self._set_arm_joints("right", state14[7:13])
        mujoco.mj_kinematics(self.model, self.data)
        out = []
        for arm, grip in (("left", state14[6]), ("right", state14[13])):
            pos_b, rot_b = self._site_pose_in_base(arm)
            # Real-YAM convention: conjugate the grasp_site rotation by the axis
            # flip (and canonical base rot). rot6d = first two columns of R_user,
            # which equals euler(extrinsic xyz)->rot6d of the same matrix.
            rot_user = self.R_af.T @ self.R_cb.T @ rot_b @ self.R_af
            rot6d = np.concatenate([rot_user[:, 0], rot_user[:, 1]])
            out.append(np.concatenate([pos_b, rot6d, [grip]]))
        return np.concatenate(out).astype(np.float32)

    # -- inverse kinematics ---------------------------------------------------

    def _ik_arm(self, arm: str, target_pos_b, target_rot_b, seed: np.ndarray) -> np.ndarray:
        ak = self.arms[arm]
        m, d = self.model, self.data
        d.qpos[ak.qposadr] = np.clip(seed, ak.lo, ak.hi)
        mujoco.mj_kinematics(m, d)
        base_pos = d.xpos[ak.base_bid].copy()
        base_mat = d.xmat[ak.base_bid].reshape(3, 3).copy()
        target_pos_w = base_pos + base_mat @ np.asarray(target_pos_b)
        target_mat_w = base_mat @ np.asarray(target_rot_b)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat_w.reshape(-1))

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cur_quat = np.zeros(4)
        neg_quat = np.zeros(4)
        err_quat = np.zeros(4)
        rot_err = np.zeros(3)
        cols = ak.dofadr
        for _ in range(self.ik_iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            eef_pos, eef_mat = self._eef_pos_mat(ak)
            pos_err = target_pos_w - eef_pos
            mujoco.mju_mat2Quat(cur_quat, eef_mat.reshape(-1))
            mujoco.mju_negQuat(neg_quat, cur_quat)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < self.ik_tol:
                break
            if ak.eef_is_site:
                mujoco.mj_jacSite(m, d, jacp, jacr, ak.eef_id)
            else:
                mujoco.mj_jacBody(m, d, jacp, jacr, ak.eef_id)
            jac = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.ik_damping * np.eye(6), err)
            d.qpos[ak.qposadr] = np.clip(d.qpos[ak.qposadr] + self.ik_step * dq, ak.lo, ak.hi)
        return d.qpos[ak.qposadr].copy()

    def _grip(self, g: float) -> float:
        g = float(np.clip(g, 0.0, 1.0))
        if self.binarize_gripper:
            g = 1.0 if g > 0.5 else 0.0
        return g

    def reset(self) -> None:
        """Drop the carried command anchor (call at the start of each episode)."""
        self._command_joints = None

    def note_command(self, joints14: np.ndarray) -> None:
        """Record the joint target actually sent to the sim for this chunk.

        The next chunk's deltas are anchored on the FK of this target rather than on
        the measured joints -- see ``joint_chunk_from_eef``.
        """
        self._command_joints = np.asarray(joints14, dtype=np.float64).copy()

    def joint_chunk_from_eef(self, eef_chunk: np.ndarray, state14: np.ndarray) -> np.ndarray:
        """(N,14) EEF action chunk -> (N,14) joint action chunk.

        Deltas are applied relative to the arm poses at chunk start. Which pose that
        is matters: LAP's action chunks are ``cmd[t+i] - cmd[t]`` where ``cmd`` is
        the *commanded* EEF target stream (MolmoAct2-YAM's sidecar
        ``*_action_eef_pose``, i.e. FK of the commanded joints -- the observed pose
        lives in ``state``/``language_action`` instead). Anchoring on the measured
        pose therefore discards the controller's tracking error every chunk, and
        because that error points along the direction of travel it shows up as
        systematic under-travel that compounds over a rollout.

        So with ``anchor_on_command`` (default) the anchor is the FK of the last
        commanded joint target, falling back to ``state14`` on the first chunk of an
        episode and whenever no command has been recorded. Anchoring on *joint*
        targets (not a free-floating EEF pose) keeps this bounded: the anchor is
        always a pose the arm was actually told to reach, so an unreachable request
        cannot let the command frame drift away from the robot indefinitely.

        Set ``anchor_on_command=False`` to restore measured-pose anchoring for A/B.
        """
        eef_chunk = np.asarray(eef_chunk, dtype=np.float64).reshape(-1, 2 * PER_ARM_ACTION)
        state14 = np.asarray(state14, dtype=np.float64)
        anchor14 = (
            self._command_joints
            if self.anchor_on_command and self._command_joints is not None
            else state14
        )

        # Pose at chunk start + seed joints, per arm.
        self.data.qpos[:] = 0.0
        self._set_arm_joints("left", anchor14[0:6])
        self._set_arm_joints("right", anchor14[7:13])
        mujoco.mj_kinematics(self.model, self.data)
        start = {}
        # IK still warm-starts from the anchor's joints: they are the configuration
        # the poses were computed from, so the first solve starts at zero error.
        seed = {"left": anchor14[0:6].copy(), "right": anchor14[7:13].copy()}
        for arm in ("left", "right"):
            start[arm] = self._site_pose_in_base(arm)

        out = np.zeros((eef_chunk.shape[0], 14), dtype=np.float32)
        for t, action in enumerate(eef_chunk):
            for arm, (a0, jslot, gslot) in (
                ("left", (0, slice(0, 6), 6)),
                ("right", (PER_ARM_ACTION, slice(7, 13), 13)),
            ):
                a = action[a0 : a0 + PER_ARM_ACTION]
                start_pos, start_rot = start[arm]
                target_pos = start_pos + a[0:3]
                # i2rt _solve_ik_from_action (control_mode="delta"): map the
                # policy's user-frame euler delta into the robot frame via the
                # canonical-base/axis-flip, then apply it to the raw grasp_site
                # rotation. rotation_frame "world" pre-multiplies (default here),
                # "local" post-multiplies.
                drot_user = rpy_to_mat(a[3:6])
                delta_rot = self.R_cb @ (self.R_af @ drot_user @ self.R_af.T) @ self.R_cb.T
                target_rot = (
                    delta_rot @ start_rot if self.rotation_delta_in_base else start_rot @ delta_rot
                )
                q = self._ik_arm(arm, target_pos, target_rot, seed[arm])
                seed[arm] = q  # warm-start the next step
                out[t, jslot] = q
                out[t, gslot] = self._grip(a[6])
        return out
