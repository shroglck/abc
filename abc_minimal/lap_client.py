"""Evaluate a remote LAP (ego-lap) policy in the put-bottles sim eval.

The ABC sim eval drives a policy through a single ``infer(obs) -> (chunk,
action_dim)`` call (see ``SimPolicy`` in ``eval_policy.py``). LAP runs in a
separate JAX/openpi process and is served over a websocket
(``ego-lap/scripts/serve_policy.py`` -> openpi ``WebsocketPolicyServer``).
``LAPClientPolicy`` mirrors ``SimPolicy``'s inference interface but forwards each
observation to that server and returns the action chunk it predicts, so the
rollout loop in ``eval_policy.py`` is reused as-is.

We vendor a tiny msgpack-numpy websocket client (the same wire protocol as
``openpi_client.websocket_client_policy``) so the ABC env only needs ``msgpack``
+ ``websockets`` and never imports the JAX-side ego-lap package.

Run the LAP server first (on a GPU box), e.g.::

    JAX_PLATFORMS=cuda uv run --group cuda scripts/serve_policy.py --env=LAP \\
        --policy.config=<cfg> --policy.dir=checkpoints/<ckpt>

then point the eval at it::

    uv run eval_policy.py --policy-backend lap --lap-host <ip> --lap-port 8000

Note (action semantics): LAP predicts *end-effector* deltas while the sim consumes
joint position targets ``[L joint1..6, L gripper, R joint1..6, R gripper]``, so an
FK/IK bridge (``abc_minimal.lap_bridge.LAPSimBridge``) converts in both directions:
the sim's 14-D joint state becomes LAP's 20-D EEF state on the way in, and LAP's
14-D EEF chunk becomes joint targets on the way out. Pass ``bridge=None`` only for
inspection -- the sim cannot execute raw EEF actions.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import os
import numpy as np
import websockets.sync.client

from abc_minimal.config import SimEvalConfig


# --- msgpack-numpy (matches openpi_client.msgpack_numpy on the wire) ---


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class _WebsocketClient:
    """Minimal synchronous client for an openpi ``WebsocketPolicyServer``."""

    def __init__(self, host: str, port: int):
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            uri = f"{uri}:{port}"
        self._packer = _Packer()
        # Disable keepalive pings: the first inference can trigger a multi-second
        # model compile on the server, which would otherwise trip the ping timeout
        # and drop the connection mid-request.
        self._ws = websockets.sync.client.connect(
            uri, compression=None, max_size=None, open_timeout=None, ping_interval=None
        )
        self.metadata = _unpackb(self._ws.recv())

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            # The server returns bytes on success and a string traceback on error.
            raise RuntimeError(f"Error in LAP inference server:\n{response}")
        return _unpackb(response)


# ABC sim camera keys -> LAP camera roles. LAP was trained with these roles
# (ego-lap scripts/real_robot/yam_main.py): top->base_0_rgb,
# left->left_wrist_0_rgb, right->right_wrist_0_rgb.
LAP_CAMERA_ROLES = {
    "top": "base_0_rgb",
    "left": "left_wrist_0_rgb",
    "right": "right_wrist_0_rgb",
}


class LAPClientPolicy:
    """Drop-in replacement for ``SimPolicy`` backed by a remote LAP server."""

    def __init__(self, config: SimEvalConfig, bridge: Any | None = None):
        self.config = config
        self.frame_description = config.lap_frame_description
        # Optional FK/IK bridge (abc_minimal.lap_bridge.LAPSimBridge): converts the
        # sim's 14-D joint state -> LAP's 20-D EEF state on the way in, and LAP's
        # 14-D EEF action chunk -> 14-D joint targets on the way out. When None,
        # state/actions pass through unchanged (raw joint state in, raw EEF out).
        self.bridge = bridge
        # "flow" reads the action expert's EEF chunk; "ar" decodes the language
        # action (chain-of-thought) into a bimanual EEF delta. AR requires the
        # server to be running with --env LAP_AR.
        self.head = config.lap_head
        # Predict-task serving: the model conditions each chunk on the PREVIOUS
        # chunk's language action (carried forward as ``prev_language_actions``),
        # and returns this chunk's summary under ``reasoning``. Serve from the
        # predict-task repo/branch with a ``predict_task=True`` config; the action
        # still arrives as a flow EEF chunk (lap_head must be "flow").
        self.predict_task = getattr(config, "lap_predict_task", False)
        self._prev_language_actions = "none"
        # Coarse-prior serving: the flow expert is seeded from the PREVIOUS chunk's
        # language action (--policy.type=flow_coarse server). Like predict-task the
        # client carries a langact forward and reads `reasoning` back, but the wire
        # key is `prev_langact` and the reset value is "" (empty) -- see below.
        self.coarse_prior = getattr(config, "lap_coarse_prior", False)
        self._prev_langact = ""
        # Reasoning trace: every chunk's decoded langact for the CURRENT episode.
        # Populated whenever the server returns a `reasoning` key, independent of
        # whether the policy consumes it (see LAP_REASONING_OBSERVE_ONLY).
        self.reasoning_trace: list[dict] = []
        # Diagnostic post-scale on returned pose-delta dims (xyz+rpy); gripper
        # dims (index 6, 13) are excluded so they stay in their calibrated [0,1]
        # range. 1.0 = no-op.
        self.action_scale = float(getattr(config, "lap_action_scale", 1.0))
        self.client = _WebsocketClient(config.lap_host, config.lap_port)

    def reset_episode(self) -> None:
        """Clear carried language-action history at the start of each rollout."""
        if self.bridge is not None:
            # Drop last episode's command anchor so the first chunk of the new one
            # is anchored on the (freshly reset) measured pose.
            self.bridge.reset()
        self._prev_language_actions = "none"
        # Empty (falsy) -> the coarse server seeds the first chunk from pure
        # Gaussian noise. "none" would be truthy and parse to a biased zero-delta
        # seed, so it must NOT be reused as the coarse reset sentinel.
        self._prev_langact = ""
        self.reasoning_trace = []

    def _build_request(self, obs: dict[str, Any], state: np.ndarray) -> dict[str, Any]:
        observation: dict[str, Any] = {"state": np.asarray(state, dtype=np.float32)}
        for cam, image in obs["images"].items():
            role = LAP_CAMERA_ROLES.get(cam)
            if role is None:
                raise KeyError(
                    f"No LAP camera role mapping for camera '{cam}'; "
                    f"known cameras: {sorted(LAP_CAMERA_ROLES)}"
                )
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] == 3:  # CHW (sim render) -> HWC (LAP contract)
                image = np.transpose(image, (1, 2, 0))
            observation[role] = np.ascontiguousarray(image)
        request = {
            "observation": observation,
            "prompt": obs.get("prompt", self.config.prompt),
            "frame_description": self.frame_description,
            # Route the server-side ImageHandler / frame policy down the bimanual
            # path so cameras slot as [base_0_rgb, left_wrist_0_rgb,
            # right_wrist_0_rgb]; without this the single-arm path misplaces the
            # top view and drops the right wrist.
            "is_bimanual": bool(self.config.lap_is_bimanual),
        }
        if self.predict_task:
            # Carried-forward language action of the previously executed chunk
            # ("none" on the first chunk). Ignored by non-predict-task servers.
            request["prev_language_actions"] = self._prev_language_actions
        if self.coarse_prior:
            # Previous chunk's decoded langact ("" on the first chunk of an
            # episode -> the coarse server falls back to Gaussian noise). Sent
            # under `prev_langact`, the key CoarsePriorFlowPolicy.infer reads.
            request["prev_langact"] = self._prev_langact
        return request

    def infer(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,  # LAP samples its own noise server-side.
        action_prefix: np.ndarray | None = None,
        prefix_length: int = 0,
    ) -> np.ndarray:
        if action_prefix is not None:
            raise NotImplementedError("RTC / action_prefix is not supported for the LAP backend")
        state = self.bridge.state20_from_joints(obs["state"]) if self.bridge is not None else obs["state"]
        request = self._build_request(obs, state)
        # ABC_DUMP_OBS=<dir>: save every request exactly as sent to the server, so the
        # language head can be re-run offline on in-distribution sim observations.
        dump_dir = os.environ.get("ABC_DUMP_OBS")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            self._dump_idx = getattr(self, "_dump_idx", 0)
            np.savez(
                os.path.join(dump_dir, f"req_{os.getpid()}_{self._dump_idx:05d}.npz"),
                prompt=np.array(request["prompt"]), frame_description=np.array(request["frame_description"]),
                is_bimanual=np.array(request["is_bimanual"]), **request["observation"],
            )
            self._dump_idx += 1
        response = self.client.infer(request)

        if self.head == "ar":
            return self._ar_actions(response, obs, state)

        if self.predict_task:
            # Carry this chunk's summary forward as the next query's previous
            # language action (empty/absent reasoning falls back to "none").
            reasoning = response.get("reasoning")
            if self.config.log_every_chunk:
                prev = self._prev_language_actions
                task = obs.get("prompt", self.config.prompt)
                # Print the exact task plus the FULL (untruncated) previous and
                # current language actions so both arms' movement is visible --
                # an 80-char clip hid the right-arm clause.
                print(
                    f"[predict-task] task={task!r}\n"
                    f"  prev={repr(prev or 'none')}\n"
                    f"  curr={repr(reasoning or 'none')}",
                    flush=True,
                )
            self._prev_language_actions = reasoning if reasoning else "none"

        if self.coarse_prior:
            # Carry this chunk's decoded langact forward to seed the next chunk's
            # flow base. Empty/absent reasoning resets to "" (Gaussian seed) rather
            # than a stale carry.
            reasoning = response.get("reasoning")
            if self.config.log_every_chunk:
                task = obs.get("prompt", self.config.prompt)
                print(
                    f"[coarse-prior] task={task!r}\n"
                    f"  prev={repr(self._prev_langact or '')}\n"
                    f"  curr={repr(reasoning or '')}",
                    flush=True,
                )
            self._prev_langact = reasoning if reasoning else ""

        # Record the trace before the action path, so a chunk is logged even when the
        # policy ignores the text. `reasoning` is absent for servers that do not
        # decode langact at all, which is simply an empty trace rather than an error.
        if "reasoning" in response:
            self.reasoning_trace.append(
                {"chunk": len(self.reasoning_trace), "reasoning": response.get("reasoning") or ""}
            )

        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if self.config.log_every_chunk and actions.shape[-1] >= 14:
            # Continuous per-arm action sanity: the raw EEF chunk's last step is the
            # net delta from chunk start (the bridge adds [0:3] to the start pose),
            # so this shows each arm's actual predicted motion (cm) alongside the
            # langact -- e.g. to confirm the right arm [7:14] isn't collapsing to ~0.
            eef = actions
            lnet, rnet = eef[-1, 0:3] * 100.0, eef[-1, 7:10] * 100.0
            print(
                f"[coarse-action] "
                f"L xyz_cm=[{lnet[0]:+.1f},{lnet[1]:+.1f},{lnet[2]:+.1f}] grip={eef[-1,6]:+.2f} "
                f"|L|={np.linalg.norm(lnet):.1f}  "
                f"R xyz_cm=[{rnet[0]:+.1f},{rnet[1]:+.1f},{rnet[2]:+.1f}] grip={eef[-1,13]:+.2f} "
                f"|R|={np.linalg.norm(rnet):.1f}",
                flush=True,
            )
        if self.action_scale != 1.0 and actions.shape[-1] >= 14:
            # Uniform post-scale on pose-delta dims only (gripper excluded so it
            # stays in its calibrated [0,1] range) -- applied AFTER the diagnostic
            # print above so that print always shows the raw model output.
            scale_vec = np.full(actions.shape[-1], self.action_scale, dtype=np.float32)
            scale_vec[6] = 1.0
            scale_vec[13] = 1.0
            actions = actions * scale_vec
        self._sanity_check_actions(actions)
        eef_raw = actions.copy()
        if self.bridge is not None:
            actions = self.bridge.joint_chunk_from_eef(actions, obs["state"])
            self._note_executed_command(actions)
        self._log_action_rows(eef_raw, actions if self.bridge is not None else None)
        return actions

    # Diagnostic magnitudes for a PHYSICAL (unnormalized) 14-D chunk. These are
    # warnings, never clipping: a chunk row is a cumulative displacement from chunk
    # start, so at h16/30 Hz translations beyond ~25 cm or rotations beyond ~1 rad
    # mean the server is returning normalized values, the wrong horizon's stats, or a
    # convention mismatch -- not a fast trajectory.
    TRANSLATION_WARN_M = 0.25
    ROTATION_WARN_RAD = 1.0

    def _sanity_check_actions(self, actions: np.ndarray) -> None:
        if actions.shape[-1] < 14:
            return
        t = float(np.max(np.abs(actions[:, [0, 1, 2, 7, 8, 9]])))
        r = float(np.max(np.abs(actions[:, [3, 4, 5, 10, 11, 12]])))
        g = actions[:, [6, 13]]
        if t > self.TRANSLATION_WARN_M and not getattr(self, "_warned_translation", False):
            self._warned_translation = True
            print(f"[sanity] max |translation| {t:.3f} m exceeds {self.TRANSLATION_WARN_M} m -- "
                  f"check that the server is unnormalizing and that norm stats match the "
                  f"trained action_horizon", flush=True)
        if r > self.ROTATION_WARN_RAD and not getattr(self, "_warned_rotation", False):
            self._warned_rotation = True
            print(f"[sanity] max |rotation| {r:.3f} rad exceeds {self.ROTATION_WARN_RAD} rad -- "
                  f"check the rotation convention", flush=True)
        if (g.min() < -0.05 or g.max() > 1.05) and not getattr(self, "_warned_gripper", False):
            self._warned_gripper = True
            print(f"[sanity] gripper outside [0,1]: min {g.min():.3f} max {g.max():.3f} -- "
                  f"the gripper column is ABSOLUTE, not a delta", flush=True)

    # -- gripper / action-row diagnostics -------------------------------------

    def _log_action_rows(self, eef: np.ndarray, joints: np.ndarray | None) -> None:
        """Print each chunk row as raw EEF action + the joint target it became.

        Exists to answer one question directly: does the policy ever instruct a
        close, and does that instruction survive the bridge? The raw gripper
        (`eef[:, 6]` / `eef[:, 13]`, absolute in [0, 1], NOT a delta) is what the
        model asked for; the joint-chunk gripper (`joints[:, 6]` / `[:, 13]`) is
        what the sim will be commanded after `_grip` clips and -- when
        `lap_binarize_gripper` is set -- thresholds it. Divergence between the two
        columns is the binarization; agreement at ~1.0 throughout means the model
        itself never asked to close.
        """
        if eef.shape[-1] < 14:
            return
        thr = float(self.config.lap_close_threshold)
        self._chunk_index = getattr(self, "_chunk_index", 0) + 1

        # Whole-rollout accounting, always on once row logging is configured.
        acc = getattr(self, "_grip_acc", None)
        if acc is None:
            acc = self._grip_acc = {
                "rows": 0, "raw_close_l": 0, "raw_close_r": 0,
                "cmd_close_l": 0, "cmd_close_r": 0,
                "raw_min_l": float("inf"), "raw_min_r": float("inf"),
            }
        acc["rows"] += len(eef)
        acc["raw_close_l"] += int((eef[:, 6] < thr).sum())
        acc["raw_close_r"] += int((eef[:, 13] < thr).sum())
        acc["raw_min_l"] = min(acc["raw_min_l"], float(eef[:, 6].min()))
        acc["raw_min_r"] = min(acc["raw_min_r"], float(eef[:, 13].min()))
        if joints is not None:
            acc["cmd_close_l"] += int((joints[:, 6] < thr).sum())
            acc["cmd_close_r"] += int((joints[:, 13] < thr).sum())

        if self._chunk_index > int(self.config.lap_log_action_rows):
            return

        print(f"[chunk {self._chunk_index:04d}] {len(eef)} rows  "
              f"(raw EEF action -> absolute joint target)", flush=True)
        for k in range(len(eef)):
            e = eef[k]
            tag_l = "CLOSE" if e[6] < thr else "open "
            tag_r = "CLOSE" if e[13] < thr else "open "
            print(f"  r{k:02d} eef  L d_cm=[{e[0]*100:+6.2f},{e[1]*100:+6.2f},{e[2]*100:+6.2f}] "
                  f"drpy=[{e[3]:+.3f},{e[4]:+.3f},{e[5]:+.3f}] g={e[6]:.3f} {tag_l} | "
                  f"R d_cm=[{e[7]*100:+6.2f},{e[8]*100:+6.2f},{e[9]*100:+6.2f}] "
                  f"drpy=[{e[10]:+.3f},{e[11]:+.3f},{e[12]:+.3f}] g={e[13]:.3f} {tag_r}",
                  flush=True)
            if joints is None:
                continue
            j = joints[k]
            jl = ",".join(f"{np.rad2deg(v):+7.2f}" for v in j[0:6])
            jr = ",".join(f"{np.rad2deg(v):+7.2f}" for v in j[7:13])
            ctag_l = "CLOSE" if j[6] < thr else "open "
            ctag_r = "CLOSE" if j[13] < thr else "open "
            print(f"      jnt  L deg=[{jl}] g={j[6]:.3f} {ctag_l} | "
                  f"R deg=[{jr}] g={j[13]:.3f} {ctag_r}", flush=True)

    def report_gripper_summary(self) -> None:
        """One-line verdict on whether a close was ever instructed."""
        acc = getattr(self, "_grip_acc", None)
        if not acc or not acc["rows"]:
            return
        n = acc["rows"]
        print(
            f"[gripper-summary] rows={n}  threshold={self.config.lap_close_threshold}\n"
            f"  raw model asked to close:  L {acc['raw_close_l']:6d} ({100*acc['raw_close_l']/n:5.1f}%)"
            f"   R {acc['raw_close_r']:6d} ({100*acc['raw_close_r']/n:5.1f}%)\n"
            f"  bridge commanded close:    L {acc['cmd_close_l']:6d} ({100*acc['cmd_close_l']/n:5.1f}%)"
            f"   R {acc['cmd_close_r']:6d} ({100*acc['cmd_close_r']/n:5.1f}%)\n"
            f"  min raw gripper seen:      L {acc['raw_min_l']:.3f}   R {acc['raw_min_r']:.3f}",
            flush=True,
        )

    def _note_executed_command(self, joint_chunk: np.ndarray) -> None:
        """Tell the bridge which joint target the sim will last have been given.

        The rollout loop executes only the first ``execute_chunk_dim`` entries of the
        chunk before replanning, so the anchor for the next chunk is that entry --
        not the chunk's final step, which is never commanded.
        """
        if not len(joint_chunk):
            return
        executed = min(int(self.config.execute_chunk_dim), len(joint_chunk))
        self.bridge.note_command(joint_chunk[executed - 1])

    def _ar_actions(self, response: dict, obs: dict[str, Any], state: np.ndarray) -> np.ndarray:
        """Decode the language action into a bimanual EEF delta and (if bridged)
        an interpolated joint chunk that drives toward the implied target."""
        from abc_minimal.lap_langact import parse_bimanual_langact

        reasoning = response.get("reasoning")
        if reasoning is None:
            raise RuntimeError(
                "lap_head='ar' but the server returned no 'reasoning'; "
                "start the policy server with --env LAP_AR."
            )
        delta = parse_bimanual_langact(
            reasoning,
            np.asarray(state, dtype=np.float32),
            # Bimanual langacts keep base-frame numerics (ego-lap's
            # choose_action_frame returns use_eef_transform=False for them and only
            # the prompt label says "egocentric frame"), so the eef->base rotation
            # applies to single-arm checkpoints only.
            eef_frame=not bool(self.config.lap_is_bimanual),
            dataset_name=self.config.lap_langact_convention_dataset,
        )
        if self.bridge is None:
            return delta[None]
        target = self.bridge.joint_chunk_from_eef(delta[None], obs["state"])[0]
        horizon = max(int(self.config.execute_chunk_dim), 1)
        start = np.asarray(obs["state"], dtype=np.float32)
        # Linearly interpolate joint targets from the current pose to the IK
        # target so the position controller approaches smoothly; replan next call.
        chunk = np.linspace(start, target, horizon + 1)[1:].astype(np.float32)
        self._note_executed_command(chunk)
        return chunk
