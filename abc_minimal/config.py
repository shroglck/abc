"""Training and model configuration dataclasses."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"


def default_cache_root() -> Path:
    return Path(os.environ.get("ABC_CACHE", str(DEFAULT_CACHE_ROOT))).expanduser()


@dataclass
class OptimConfig:
    """AdamW with a linear-warmup-then-constant LR schedule."""
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    vision_lr_scale: float = 1.0


@dataclass
class FlowConfig:
    """Rectified-flow matching + action-prefix conditioning."""
    mask_state_ratio: float = 0.1
    max_action_prefix: int = 4
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.05
    num_diffusion_steps: int = 10


@dataclass
class ClipConfig:
    """CLIP ViT-B/32 text asset locations."""
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "clip"))
    model_url: str = (
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    )
    bpe_url: str = (
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    )
    model_name: str = "ViT-B-32.pt"
    bpe_name: str = "bpe_simple_vocab_16e6.txt.gz"


@dataclass
class MixtureComponent:
    """One source in the train/val mixture."""
    train_dir: str
    val_dir: str
    weight: float
    task_name: str


@dataclass
class DiTConfig:
    """ABC-DiT architecture defaults."""
    hidden_size: int = 1536
    depth: int = 32
    num_heads: int = 24
    mlp_ratio: float = 4.0
    state_dim: int = 14
    action_dim: int = 14
    chunk_length: int = 30
    camera_keys: tuple[str, ...] = ("top", "left", "right")
    task_embed_dim: int = 512

    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vision_pool_num_queries: int = 12
    vision_pool_num_heads: int = 8
    vision_pool_mlp_ratio: int = 4


@dataclass
class PutBottlesSimConfig:
    """Scene, randomization, and task metric defaults for put-bottles sim eval."""
    gripper_ctrl_max: float = 0.0475
    bottle_count: int = 6
    init_q: tuple[float, ...] = (
        0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0,
    )
    timestep: float = 0.002
    control_decimation: int = 17

    table_z: float = 0.75
    table_bounds: tuple[float, float, float, float] = (0.3025, 0.8975, -0.65, 0.65)
    bottle_spawn_clearance: float = 0.005
    bottle_sample_attempts: int = 200
    bottle_collision_margin: float = 0.04
    bottle_scale_range: tuple[float, float] = (0.9, 1.1)
    bottle_side_radii: tuple[float, ...] = (0.025667, 0.024014, 0.020589, 0.026359, 0.023689, 0.021823)
    bottle_flat_lengths: tuple[float, ...] = (0.166718, 0.165000, 0.156531, 0.160000, 0.166689, 0.159200)
    bottle_flat_half_widths: tuple[float, ...] = (0.025672, 0.024013, 0.020566, 0.025957, 0.023689, 0.021823)

    bin_scale_range: tuple[float, float] = (0.95, 1.05)
    bin_yaw_range: tuple[float, float] = (-0.75, 0.75)
    bin_xy_range: tuple[float, float, float, float] = (0.57, 0.73, -0.25, 0.25)
    bin_z_scale: float = 0.83
    bin_occupied_radius: float = 0.13
    bin_base_quat: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)

    eval_bin_radius: float = 0.155
    eval_min_rel_z: float = -0.06
    eval_max_rel_z: float = 0.26


# Reference hours-weighted real+sim bottles mix.
MIXTURE_PRESETS: dict[str, list[MixtureComponent]] = {
    "bottles": [
        MixtureComponent("train_real", "val_real", 0.8172, "throw_plastic_bottles_in_bin"),
        MixtureComponent("train_sim", "val_sim", 0.1828, "sim_put_the_plastic_bottles_in_the_bin"),
    ],
}


@dataclass
class TrainConfig:
    """Minimal ABC-DiT bottles-in-bin training."""
    cache_root: str = field(
        default_factory=lambda: str(default_cache_root())
    )
    seed: int = 123
    batch_size: int = 90
    num_workers: int = 16
    train_steps: int = 75_000

    mixture_preset: Literal["bottles"] = "bottles"
    mixture: list[MixtureComponent] = field(default_factory=list)

    load_pretrained: bool = False
    dino_bf16: bool = True
    compile: bool = True

    log_every: int = 20
    val_every: int = 2500
    val_batches: int = 4
    ckpt_every: int = 5000
    log_wandb: bool = False
    wandb_project: str = "minimal-abc"

    optim: OptimConfig = field(default_factory=OptimConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)

    def resolve_mixture(self) -> list[MixtureComponent]:
        return self.mixture if self.mixture else MIXTURE_PRESETS[self.mixture_preset]


@dataclass
class SimEvalConfig:
    """MuJoCo-Warp put-bottles evaluation."""
    # Local ABC-DiT checkpoint (.pt / s3://...). Required for the "dit" backend;
    # ignored for "lap" (the policy lives on the remote LAP server).
    checkpoint: str = ""
    norm_stats_path: str | None = None

    # Policy backend: "dit" runs a local ABC-DiT checkpoint; "lap" queries a
    # remote ego-lap policy server (openpi WebsocketPolicyServer).
    policy_backend: Literal["dit", "lap"] = "dit"
    # LAP inference head: "flow" = continuous action-expert EEF chunk;
    # "ar" = decode the language action (needs server --env LAP_AR).
    lap_head: Literal["flow", "ar"] = "flow"
    lap_host: str = "0.0.0.0"
    lap_port: int = 8000
    # Convert the sim's joint state/actions to/from LAP's EEF space via FK/IK
    # (abc_minimal.lap_bridge). Disable to pass raw joint state in / EEF out
    # (inspection only -- the sim cannot execute raw EEF actions).
    lap_bridge: bool = True
    # Apply the real-YAM EEF rotation convention (i2rt canonical-base + axis-flip,
    # ported verbatim in lap_bridge.CANONICAL_*): the bridge conjugates the
    # grasp_site FK rotation by the axis flip, matching the deployed encoder
    # (verified to ~1e-8 against an independent i2rt-model FK ground truth).
    lap_align_eef_rot: bool = True
    # Which link the checkpoint's TRAINING data measured end-effector pose at. This is
    # a property of the dataset, not of the robot, and the two YAM sources disagree:
    #   abc_yam        -> "grasp_site"  (converter FK; reproduces the shipped pose to 1e-8)
    #   molmoact2_yam  -> "link_6"      (sidecar metadata_json: urdf_target_link=link_6)
    # They are 13.47 cm and exactly 90 deg apart -- grasp_site hangs off link_6 at
    # pos="0 0 0.1347" quat="1 0 0 -1". Serving with the wrong one puts every state the
    # policy sees off its training manifold. Resolved as "{arm}_{lap_eef_link}", tried
    # as a site then a body.
    lap_eef_link: str = "grasp_site"
    # Frame string injected into the prompt to condition the flow expert. The
    # served bimanual checkpoint (ckeckpoint_bimanip, run lap_rhb_yam_armrand...)
    # trained two-arm samples with BaseFrameProb bimanual=0.0 -> labeled
    # "egocentric frame", so the request must advertise that (sending "robot base
    # frame" conditions on an unseen label). Override per-checkpoint if a run used
    # BaseFrameProb bimanual=1.0.
    lap_frame_description: str = "egocentric frame"
    # Rotation-delta multiply order in the FK/IK bridge. Training/real-client
    # encode the rotation delta as a BODY-LOCAL relative rotation (post-multiply:
    # euler_diff = R_start^T @ R_t), so the bridge must post-multiply
    # (target_rot = start_rot @ delta_rot). Verified by the offline round-trip
    # arbiter abc_minimal.lap_convention_probe: False round-trips to ~3e-5 rad,
    # True is wrong by ~0.33 rad. Keep False; expose as a flag only for A/B.
    lap_rotation_delta_in_base: bool = False
    # Execute the gripper as the checkpoint's serving contract specifies
    # (`binary_model_open_gt_0p5_else_closed_v1`): threshold the model's continuous
    # gripper at 0.5 and command a FULL open/close. Passing the raw value through
    # instead scales it (ctrl = g * gripper_ctrl_max), so a mid-range prediction near
    # a grasp transition becomes a partially-closed hand that hovers without gripping.
    lap_binarize_gripper: bool = True
    # Anchor each chunk's EEF deltas on the last COMMANDED joint target (FK'd) rather
    # than on the measured joint state. LAP's action chunks are cmd[t+i] - cmd[t]
    # over the commanded EEF stream (MolmoAct2-YAM's sidecar *_action_eef_pose is FK
    # of the commanded joints; the observed pose is a separate stream), so anchoring
    # on the measured pose silently drops the controller's tracking error every
    # chunk -- a bias along the direction of travel, i.e. compounding under-travel.
    # Set False to restore measured-pose anchoring for an A/B.
    lap_anchor_on_command: bool = True
    # Dataset whose raw rotation-axis convention the checkpoint's language actions
    # carry, inverted when a langact is parsed back into numbers (lap_head="ar").
    # Only "molmoact2_yam_dataset" has one (rotation stored as [pitch, roll, -yaw]).
    # The coarse-prior seed is parsed server-side instead, so it needs the matching
    # ego-lap flag: serve_policy.py --langact-convention-dataset.
    lap_langact_convention_dataset: str = ""
    # Advertise the sample as bimanual in the request (`is_bimanual`). LAP's
    # server-side ImageHandler slots cameras differently per morphology: bimanual
    # -> [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]; single-arm ->
    # [left_wrist_0_rgb, base_0_rgb, blank] (top view misplaced, right wrist
    # dropped). It also selects the bimanual base-frame convention. Must be True
    # for the put-bottles bimanual checkpoints or the policy sees scrambled views.
    lap_is_bimanual: bool = True
    # Serve a predict-task checkpoint (predict_task=True config, e.g. ego-lap
    # branch shrg/dataset_ablation). The client carries the previous chunk's
    # language action forward as `prev_language_actions` and reads this chunk's
    # summary back from `reasoning`. The action still arrives as a flow EEF chunk,
    # so keep lap_head="flow". Ignored by plain bimanual servers.
    lap_predict_task: bool = False
    # Serve a coarse-prior checkpoint (use_coarse_prior=True config, e.g.
    # lap_coarse_prior; start the server with --policy.type=flow_coarse). The flow
    # action expert is seeded from the PREVIOUS chunk's language action instead of
    # Gaussian noise: the client carries the previous chunk's decoded langact
    # forward under `prev_langact` and reads this chunk's summary back from
    # `reasoning`. The action still arrives as a flow EEF chunk, so keep
    # lap_head="flow". Distinct from predict-task: the wire key is `prev_langact`
    # (not `prev_language_actions`) and the reset sentinel is "" (empty), NOT
    # "none" -- coarse treats any truthy prev as a langact to parse, so "none"
    # would seed a biased zero-delta instead of the intended first-query Gaussian.
    lap_coarse_prior: bool = False
    # Diagnostic post-scale on the returned action's pose-delta dims (xyz+rpy per
    # arm; gripper dims untouched, already calibrated in [0,1]). The bimanual
    # category's bounds_q99 translation range is only ~1-2cm regardless of what
    # the model predicts in normalized space (see coarse_prior_ckpt norm stats);
    # a uniform scalar keeps the model's own learned per-axis/per-arm ratios
    # intact while testing whether simple linear amplification (rather than
    # switching to a differently-scaled norm category) produces usable motion.
    lap_action_scale: float = 1.0
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[1] / "outputs" / "sim_eval_put_bottles"
        )
    )
    num_worlds: int = 5
    seed: int = 20260511
    num_chunks: int = 120
    execute_chunk_dim: int = 15
    diffusion_steps: int = 10
    policy_seed: int = 0
    camera_height: int = 168
    camera_width: int = 224
    device: str = "auto"
    gpu_id: int | None = None
    fast_inference: bool = True
    fast_compile_mode: str = "max-autotune-no-cudagraphs"
    vanilla_physics: bool = False
    rtc: bool = False
    rtc_prefix_length: int = 4
    rtc_inference_lead_steps: int = 4
    # Dump EVERY row of each action chunk: the raw 14-D EEF action as the server
    # returned it, and the 14-D absolute joint target the FK/IK bridge turned it
    # into, side by side. The value is how many chunks to dump in full (0 = off);
    # the gripper accounting below it runs for the whole rollout regardless, so
    # "did it ever command a close?" is answered even when the dump is capped.
    lap_log_action_rows: int = 0
    # Raw-gripper threshold counted as a "close" instruction in that accounting.
    # 0.5 mirrors lap_binarize_gripper's own threshold.
    lap_close_threshold: float = 0.5
    log_every_chunk: bool = False
    save_video: bool = False
    video_fps: int = 30
    video_every_n_actions: int = 1
    prompt: str = "sim put the plastic bottles in the bin"

    scene: PutBottlesSimConfig = field(default_factory=PutBottlesSimConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class VizSimEvalConfig(SimEvalConfig):
    """Single-world sim config defaults for the live Viser viewer."""
    num_chunks: int = 200


@dataclass
class VizPolicyConfig:
    """Live viser viewer over a single ABC-DiT sim rollout."""
    sim: VizSimEvalConfig
    port: int = 8080
    fast_inference: bool = True
    fast_compile_mode: str = "max-autotune-no-cudagraphs"


def validate_model_config(model: DiTConfig) -> list[str]:
    model_dims = [
        model.hidden_size, model.depth, model.num_heads, model.mlp_ratio,
        model.state_dim, model.action_dim, model.chunk_length, model.task_embed_dim,
        model.vit_embed_dim, model.vit_depth, model.vit_num_heads,
        model.vision_pool_num_queries, model.vision_pool_num_heads,
        model.vision_pool_mlp_ratio,
    ]
    errors = []
    if min(model_dims) <= 0 or not model.camera_keys:
        errors.append("model dimensions and camera_keys must be positive/non-empty")
    if (
        model.hidden_size % model.num_heads
        or model.vit_embed_dim % model.vit_num_heads
        or model.vit_embed_dim % model.vision_pool_num_heads
        or model.hidden_size % 2
        or (model.vit_embed_dim // model.vit_num_heads) % 4
    ):
        errors.append("attention dimensions must be compatible with their head counts")
    return errors


def validate_train_config(
    config: TrainConfig, cache_root: Path, checkpoint_path: Path
) -> list[MixtureComponent]:
    components = config.resolve_mixture()
    weights = [c.weight for c in components]
    errors = []

    if (
        min(config.batch_size, config.train_steps, config.log_every, config.val_every,
            config.val_batches, config.ckpt_every) <= 0
        or config.num_workers < 0
    ):
        errors.append(
            "batch size, step intervals, and val_batches must be positive; "
            "num_workers must be non-negative"
        )
    if (
        not 0 <= config.flow.mask_state_ratio <= 1
        or not 0 <= config.flow.prefix_conditioning_prob <= 1
        or config.flow.prefix_noise_scale < 0
    ):
        errors.append(
            "flow probabilities must be in [0, 1] and prefix_noise_scale must be non-negative"
        )
    errors.extend(validate_model_config(config.model))
    if (
        not components
        or any(not math.isfinite(w) or w <= 0 for w in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        errors.append(
            f"mixture weights must be positive and sum to 1.0, "
            f"got {sum(weights) if weights else 0:.8g}"
        )

    required = [cache_root / "norm_stats.json"]
    required += [cache_root / p for c in components for p in (c.train_dir, c.val_dir)]
    if config.load_pretrained:
        required.append(checkpoint_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        errors.append("missing required paths: " + ", ".join(missing))

    if errors:
        raise ValueError("Invalid training config:\n  - " + "\n  - ".join(errors))
    return components
