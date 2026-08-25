#!/bin/bash
# Run under srun (resources are passed on the srun command line, not here):
#
#   srun --job-name=abc_yam_gripper_probe --partition=all --nodes=1 \
#        --gres=gpu:l40:1 --cpus-per-task=8 --mem=64G --time=2:00:00 \
#        bash scripts/eval_abc_yam_gripper_probe.sh
#
# With one GPU the server and the MuJoCo eval share it (GPU_SERVER falls back to
# GPU_ARR[0]); ask for :2 to split them.

# Why: the abc_yam checkpoint never closes the gripper in closed loop. This dumps
# EVERY chunk row as (raw 14-D EEF action the server returned) -> (14-D absolute
# joint target the FK/IK bridge produced), so the close instruction can be traced
# from the model output to the sim command, and prints a whole-rollout tally of
# how often each side asked for a close.
#
# Two runs in one allocation, one variable apart:
#   A  defaults as shipped -- lap_align_eef_rot=True, lap_binarize_gripper=True.
#      Reproduces the reported behaviour.
#   B  lap_align_eef_rot=False (abc's converter wrote the RAW grasp_site rotation,
#      no i2rt encoder conjugation -- verified against the shards to 1.2e-7 rad;
#      the align path is off by a median 58 deg) and lap_binarize_gripper=False
#      (abc's gripper is analog: 46% of its commands sit strictly between 0.1 and
#      0.9, and the 0.5 threshold lands on its 0.444 median).
set -uo pipefail

ABC=/n/fs/prism-data/shresth/abc
EGO=/n/fs/prism-data/shresth/ego-lap
# Local checkpoint on purpose: gsutil on the compute nodes authenticates
# ANONYMOUSLY even with GOOGLE_APPLICATION_CREDENTIALS set, so the GCS path
# dies with "401 Anonymous caller ... storage.objects.list". The manifest
# check is satisfied here too (checkpoint_abc_yam/run_manifest.json, config
# lap_abc_yam). Override CKPT_DIR for a GCS step only after fixing auth.
CKPT_DIR=${CKPT_DIR:-/n/fs/prism-data/shresth/ego-lap/checkpoint_abc_yam/22000}
CKPT_CONFIG=${CKPT_CONFIG:-lap_abc_yam}
NUM_WORLDS=${NUM_WORLDS:-2}
DUMP_CHUNKS=${DUMP_CHUNKS:-6}

PORT=$(( 8000 + ${SLURM_JOB_ID:-$$} % 1000 ))
STAMP=${SLURM_JOB_ID:-$$}
OUT_BASE=$ABC/outputs/abc_yam_gripper_probe_${STAMP}
SERVER_LOG=$ABC/slurm_logs/abc_yam_gripper_probe_${STAMP}.server.log
mkdir -p "$OUT_BASE" "$ABC/slurm_logs"

IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_EVAL=${GPU_ARR[0]}
GPU_SERVER=${GPU_ARR[1]:-${GPU_ARR[0]}}
echo "host=$(hostname) port=$PORT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ckpt=$CKPT_DIR config=$CKPT_CONFIG worlds=$NUM_WORLDS dump_chunks=$DUMP_CHUNKS"

export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/u/sg0803/irom-service-account-key.json}
export CURL_CA_BUNDLE=${CURL_CA_BUNDLE:-$EGO/.venv/lib/python3.11/site-packages/certifi/cacert.pem}

(
  cd "$EGO"
  PYTHONPATH="$EGO/src" JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPU_SERVER" \
    "$EGO/.venv/bin/python" scripts/serve_policy.py \
      --env LAP --port "$PORT" \
      --require-checkpoint-manifest \
      policy:checkpoint \
      --policy.config "$CKPT_CONFIG" \
      --policy.dir "$CKPT_DIR" \
      --policy.type flow
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for server on :$PORT ..."
for i in $(seq 1 120); do
  grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" 2>/dev/null && { echo "server ready after ~$(( i * 10 ))s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED before ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1; }
  sleep 10
done
grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" || { echo "server never ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1; }

run_variant () {  # $1=label  $2...=extra flags
  local label=$1; shift
  echo ""
  echo "==================== VARIANT $label ===================="
  echo "extra flags: $*"
  cd "$ABC"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" "$ABC/.venv/bin/python" -m abc_minimal.eval_policy \
      --policy-backend lap \
      --lap-host 127.0.0.1 --lap-port "$PORT" \
      --num-worlds "$NUM_WORLDS" \
      --lap-log-action-rows "$DUMP_CHUNKS" \
      --save-video \
      --output-dir "$OUT_BASE/$label" \
      "$@"
  echo "==================== END VARIANT $label ===================="
}

run_variant A_defaults
run_variant B_rawrot_continuous --no-lap-align-eef-rot --no-lap-binarize-gripper

echo ""
echo "=== gripper verdict (both variants) ==="
grep -A4 "gripper-summary" "$ABC/slurm_logs/abc_yam_gripper_probe_${STAMP}.out" 2>/dev/null || \
  echo "(see [gripper-summary] blocks above)"
echo "outputs: $OUT_BASE"
