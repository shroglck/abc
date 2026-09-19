#!/bin/bash
# Online turn-the-mug-right-side-up eval of an abc_yam LAP checkpoint.
#
# Second sim task on ABC's scene (the first is put-bottles). Same two-GPU split as
# eval_abc_yam_lap.sh: LAP policy server on one GPU, MuJoCo/MJWarp eval on the
# other. The only task-specific arguments are --task, --prompt, and the
# --mug-scene.* randomization knobs.
#
# The prompt is ABC-130k's real task string verbatim ("turn the mug right side
# up", 673 episodes, in the abc_yam mixture) -- do NOT prefix it with "sim", which
# is ABC's own marker for their sim-teleop data and is out of distribution here.
set -euo pipefail

ABC=/n/fs/prism-data/shresth/abc
EGO=/n/fs/prism-data/shresth/ego-lap
CKPT_DIR=${CKPT_DIR:-$EGO/checkpoint_abc_yam_29k/29000}
CKPT_CONFIG=${CKPT_CONFIG:-lap_abc_yam}
NUM_WORLDS=${NUM_WORLDS:-10}
# One mug scores a placement; success is all-or-nothing across the mugs present.
MUG_COUNT=${MUG_COUNT:-1}
PROMPT=${PROMPT:-"turn the mug right side up"}

PORT=$(( 8000 + ${SLURM_JOB_ID:-$$} % 1000 ))
OUT_DIR=${OUT_DIR:-$ABC/outputs/mug_flip_${SLURM_JOB_ID:-$$}}
SERVER_LOG=$ABC/slurm_logs/mug_flip_${SLURM_JOB_ID:-$$}.server.log
mkdir -p "$OUT_DIR" "$ABC/slurm_logs"

IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_EVAL=${GPU_ARR[0]}
GPU_SERVER=${GPU_ARR[1]:-${GPU_ARR[0]}}
echo "host=$(hostname) port=$PORT gpus=${CUDA_VISIBLE_DEVICES:-unset} ckpt=$CKPT_DIR worlds=$NUM_WORLDS mugs=$MUG_COUNT"

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
  if grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" 2>/dev/null; then
    echo "server ready after ~$(( i * 10 ))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED before ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1
  fi
  sleep 10
done
grep -q "server listening on 0.0.0.0:$PORT" "$SERVER_LOG" || { echo "server never ready; tail:"; tail -n 40 "$SERVER_LOG"; exit 1; }

cd "$ABC"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" "$ABC/.venv/bin/python" -m abc_minimal.eval_policy \
    --task mug_flip \
    --prompt "$PROMPT" \
    --policy-backend lap \
    --lap-host 127.0.0.1 --lap-port "$PORT" \
    --num-worlds "$NUM_WORLDS" \
    --mug-scene.mug-count "$MUG_COUNT" \
    `# Spawn box (x0 x1 y0 y1), tighter than the table so both arms can reach.` \
    --mug-scene.mug-xy-range ${MUG_XY_RANGE:-0.42 0.72 -0.34 0.34} \
    `# abc_yam stores RAW grasp_site FK orientation -- the i2rt axis-flip` \
    `# conjugation corrupts both the policy state and the decoded angular action.` \
    --no-lap-align-eef-rot \
    `# abc trains on ANALOG gripper commands (median 0.444); BINARIZE=1 to A/B it.` \
    ${BINARIZE_GRIPPER:+--lap-binarize-gripper} ${BINARIZE_GRIPPER:---no-lap-binarize-gripper} \
    --save-video \
    --output-dir "$OUT_DIR"

echo "=== summary ==="
python3 -c "import json;d=json.load(open('$OUT_DIR/summary.json'));print({k:v for k,v in d.items() if not isinstance(v,(list,dict))})" 2>/dev/null || true
echo "outputs: $OUT_DIR"
