#!/bin/bash
#SBATCH --job-name=lap_eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# 5 worlds x 120 chunks runs ~3.3s/chunk plus ~2min of server load/compile, which
# overruns the old 0:40:00 by a few minutes and kills the last world mid-rollout
# (no summary.json). Headroom for the default NUM_WORLDS/NUM_CHUNKS; raise further
# for longer episodes.
#SBATCH --time=1:20:00
#SBATCH --output=/n/fs/prism-data/shresth/abc/slurm_logs/%x_%j.out
#SBATCH --error=/n/fs/prism-data/shresth/abc/slurm_logs/%x_%j.err

# LAP put-bottles eval, self-contained in one job (sbatch OR `srun bash`):
#   LAP policy server (ego-lap scripts/serve_policy.py) + MuJoCo-Warp put-bottles
#   eval, sharing the allocated GPU(s). Conventions are source-verified in
#   abc_minimal/config.py's defaults; see also eval_lap_slurm.sbatch.
#
# MODE picks the head (details at the MODE block below):
#   coarse (default) -- flow_coarse server; flow seeded from the previous chunk's
#                       decoded langact. Needs a use_coarse_prior=True checkpoint.
#   flow             -- plain flow, Gaussian base. Any flow checkpoint.
#   ar               -- client IK-executes the decoded language action directly.
#
# Defaults target the coarse-prior checkpoint (coarse_prior_ckpt/20000 under
# lap_coarse_prior); override CKPT_DIR + CKPT_CONFIG for anything else, e.g.
#   MODE=flow CKPT_CONFIG=auto CKPT_DIR=$EGO/checkpoint_yam_new/7000 NUM_WORLDS=5
# CKPT_CONFIG=auto makes the server replay the training config from the
# run_manifest.json beside the step dir (and validate it against the checkpoint
# contract) instead of trusting a hand-passed config name -- prefer it whenever the
# manifest is present. NUM_WORLDS defaults to 2 for a quick smoke run.

set -euo pipefail

ABC=/n/fs/prism-data/shresth/abc
EGO=/n/fs/prism-data/shresth/ego-lap
CKPT_TAG_DIR=${CKPT_TAG_DIR:-20000}
# Overridable so non-coarse checkpoints (e.g. checkpoint_predict_task, plain flow
# under lap_robot_human_bimanual) can be pointed at directly.
CKPT_DIR=${CKPT_DIR:-$EGO/coarse_prior_ckpt/$CKPT_TAG_DIR}
CKPT_CONFIG=${CKPT_CONFIG:-lap_coarse_prior}
NUM_WORLDS=${NUM_WORLDS:-2}
# Episode length = num_chunks * execute_chunk_dim(15) steps; default 120 -> 1800
# steps. Override to run longer episodes (e.g. 600 = 5x default).
NUM_CHUNKS=${NUM_CHUNKS:-120}
# AR langact decode temperature (0.0=greedy ~1cm collapse; >0 recovers magnitudes).
DECODE_TEMP=${DECODE_TEMP:-0.5}
# Frame the server parses langact/coarse deltas in. abc default "egocentric frame";
# set FRAME_DESC="end-effector frame" to match the molmospaces/eef-trained convention.
FRAME_DESC=${FRAME_DESC:-egocentric frame}
# MODE=coarse  -> flow_coarse server, flow action expert seeded from langact.
# MODE=ar      -> ar server; the client IK-executes the DECODED language action
#                 directly (bypasses the flow expert, which outputs ~0 translation).
# MODE=flow    -> plain flow server, NO coarse seed (Gaussian noise base) -- for
#                 non-coarse checkpoints (e.g. checkpoint_predict_task served under
#                 lap_robot_human_bimanual), to compare raw deltas against coarse.
MODE=${MODE:-coarse}
# Dataset whose raw rotation-axis convention this checkpoint's language actions
# carry. MolmoAct2-YAM stores rotations as [pitch, roll, -yaw] relative to the
# repo's language convention, and the training-time coarse seed is built from the
# pre-remap numeric tensor -- so parsing a YAM langact back into a seed (coarse) or
# into an executed delta (ar) must invert it. Empty = no conversion, which is right
# for checkpoints whose robot data is not YAM.
LANGACT_CONVENTION_DATASET=${LANGACT_CONVENTION_DATASET:-molmoact2_yam_dataset}
# Ablation (coarse mode only): build the coarse seed single-arm (left clause) while
# langact still decodes both arms. Set COARSE_SINGLE_ARM=1 to enable.
SERVER_EXTRA=""
EVAL_EXTRA=""
# EEF rotation convention of the checkpoint's TRAINING data. The two YAM datasets
# disagree, so this must follow the checkpoint:
#   molmoact2_yam_dataset -- poses come from the deployed i2rt encoder, which reports
#     R_af^T R_cb^T R_raw R_af (canonical-base + axis-flip conjugation)  -> 1
#   abc_yam_dataset -- the RLDS converter (rlds_dataset_builder/abc_pipeline/fk.py,
#     YamFK("grasp_site")) stores the RAW MuJoCo grasp_site rotation in the arm's own
#     base frame, no conjugation at all                                  -> 0
# Position is the raw arm-base FK xyz either way, so only orientation is affected --
# which is exactly what a grasp needs to get right. Serving abc_yam with 1 feeds the
# model conjugated rotations and un-conjugates its rotation deltas on the way out.
ALIGN_EEF_ROT=${ALIGN_EEF_ROT:-1}
[ "$ALIGN_EEF_ROT" = "0" ] && EVAL_EXTRA="$EVAL_EXTRA --no-lap-align-eef-rot"

# Render observations with MuJoCo's OpenGL renderer instead of the Warp ray tracer
# (physics stays on Warp either way). Warp ignores the scene's `headlight ambient`,
# which is where put_bottle.xml gets nearly all its light, so its surround renders
# ~3x darker than the repo's sim eval: floor corner (28,28,31) vs GL's (90,90,90),
# and the white skybox never appears at all. Needs MUJOCO_GL=egl, set below.
# Pairs with --vanilla-physics: mj_step then advances the HOST MjData, which is what
# the GL renderer reads, so no per-frame device->host copy is needed. README calls
# vanilla physics the faster choice for single-world rollouts anyway.
RENDER_GL=${RENDER_GL:-0}
EVAL_PHYSICS_FLAG=""
if [ "$RENDER_GL" = "1" ]; then
  export ABC_RENDER_GL=1
  export MUJOCO_GL=egl
  EVAL_PHYSICS_FLAG="--vanilla-physics"
fi
if [ "$MODE" = "ar" ]; then
  POLICY_TYPE=ar
  EVAL_HEAD_FLAGS="--lap-head ar"
  # AR executes the decoded langact directly, so the client does the inversion.
  EVAL_EXTRA="--lap-langact-convention-dataset $LANGACT_CONVENTION_DATASET"
elif [ "$MODE" = "flow" ]; then
  POLICY_TYPE=flow
  EVAL_HEAD_FLAGS="--lap-head flow"
  # Plain flow never round-trips through langact text -- no conversion applies.
else
  POLICY_TYPE=flow_coarse
  EVAL_HEAD_FLAGS="--lap-head flow --lap-coarse-prior"
  # The coarse seed is parsed server-side, so the server needs the convention.
  SERVER_EXTRA="--langact-convention-dataset $LANGACT_CONVENTION_DATASET"
  [ "${COARSE_SINGLE_ARM:-0}" = "1" ] && SERVER_EXTRA="$SERVER_EXTRA --coarse-single-arm"
fi
# Diagnostic override: unnormalize actions with true min/max ("bounds") instead of
# the training config's 1st/99th-percentile clip ("bounds_q99"). Only affects the
# action output (and coarse seed) unnorm -- state/proprio input stays at the
# training type. Empty (default) = use the training config's type.
if [ -n "${ACTION_NORM_TYPE:-}" ]; then
  SERVER_EXTRA="$SERVER_EXTRA --action-normalization-type $ACTION_NORM_TYPE"
fi
# Diagnostic override: use a different category's norm stats for the ACTION output
# (e.g. "single_arm" instead of the training config's "bimanual" for lap_coarse_prior,
# whose q01/q99 translation is only ~1-2cm vs single_arm's ~40-60cm).
if [ -n "${POLICY_NORM_CATEGORY:-}" ]; then
  SERVER_EXTRA="$SERVER_EXTRA --policy-normalization-category $POLICY_NORM_CATEGORY"
fi

PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))
# Name the run after what it actually is. Two ways the old name lied:
#   * the "lap_coarse..." prefix was hardcoded, so MODE=flow/ar runs landed in
#     directories that read as coarse-prior runs;
#   * the step came from CKPT_TAG_DIR rather than from CKPT_DIR, so passing only
#     CKPT_DIR (the natural way to point at another run's checkpoint) kept the
#     default step in the name -- e.g. serving checkpoint_yam/8000 wrote to
#     .../lap_coarse20000_flow_<job>.
# Both parts now come from CKPT_DIR itself, so the name cannot disagree with what
# was served. The parent dir (checkpoint_yam / coarse_prior_ckpt / ...) also
# disambiguates checkpoints that share a step number.
CKPT_STEP=$(basename "$CKPT_DIR")
CKPT_TAG=$(basename "$(dirname "$CKPT_DIR")")
OUT_DIR=$ABC/outputs/${CKPT_TAG}_${CKPT_STEP}_${MODE}_${SLURM_JOB_ID}
SERVER_LOG=$ABC/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.server.log
mkdir -p "$OUT_DIR" "$ABC/slurm_logs"

IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_EVAL=${GPU_ARR[0]}
GPU_SERVER=${GPU_ARR[1]:-${GPU_ARR[0]}}

echo "host=$(hostname) job=$SLURM_JOB_ID port=$PORT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "gpu_eval=$GPU_EVAL gpu_server=$GPU_SERVER num_worlds=$NUM_WORLDS out=$OUT_DIR"

# --- launch flow_coarse policy server on the 2nd allocated GPU -------------
(
  cd "$EGO"
  # Single-GPU sharing: stop JAX from preallocating ~75% of the device (default),
  # so MuJoCo-Warp in the eval process has room on the same GPU. On-demand growth.
  PYTHONPATH="$EGO/src" JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPU_SERVER" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$EGO/.venv/bin/python" scripts/serve_policy.py \
      --env LAP --port "$PORT" \
      --decode-temperature "$DECODE_TEMP" \
      $SERVER_EXTRA \
      policy:checkpoint \
      --policy.config "$CKPT_CONFIG" \
      --policy.dir "$CKPT_DIR" \
      --policy.type "$POLICY_TYPE"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "cleanup: stopping server $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for server on :$PORT ..."
for i in $(seq 1 120); do   # up to 20 min
  if grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" 2>/dev/null; then
    echo "server ready after ~$(( i * 10 ))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED before ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1
  fi
  sleep 10
done
if ! grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" 2>/dev/null; then
  echo "server never became ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1
fi

# Diagnostic post-scale on the client's returned pose-delta dims (gripper excluded).
# Default 1.0 = no-op.
ACTION_SCALE=${ACTION_SCALE:-1.0}

# --- run the coarse-prior eval on the 1st allocated GPU --------------------
cd "$ABC"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" "$ABC/.venv/bin/python" -m abc_minimal.eval_policy \
    --policy-backend lap \
    --lap-host 127.0.0.1 --lap-port "$PORT" \
    $EVAL_HEAD_FLAGS \
    $EVAL_EXTRA \
    $EVAL_PHYSICS_FLAG \
    --lap-frame-description "$FRAME_DESC" \
    --lap-action-scale "$ACTION_SCALE" \
    --log-every-chunk \
    --num-worlds "$NUM_WORLDS" \
    --num-chunks "$NUM_CHUNKS" \
    --save-video \
    --output-dir "$OUT_DIR"

echo "=== eval done; summary ==="
cat "$OUT_DIR/summary.json" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:v for k,v in d.items() if not isinstance(v,(list,dict))})" || true
