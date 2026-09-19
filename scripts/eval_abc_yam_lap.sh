#!/bin/bash
# Online put-bottles eval of the abc_yam checkpoint, to see whether the gripper
# actually closes in closed loop. Adapted from eval_lap_slurm.sbatch: same two-GPU
# split (server on one, MuJoCo eval on the other), different checkpoint/config.
set -euo pipefail

ABC=/n/fs/prism-data/shresth/abc
EGO=/n/fs/prism-data/shresth/ego-lap
CKPT_DIR=${CKPT_DIR:-gs://v6_east1d/checkpoints/lap_abc_yam/abc_yam_smoke_test/20000}
CKPT_CONFIG=${CKPT_CONFIG:-lap_abc_yam}
NUM_WORLDS=${NUM_WORLDS:-10}

PORT=$(( 8000 + ${SLURM_JOB_ID:-$$} % 1000 ))
OUT_DIR=$ABC/outputs/abc_yam_20k_binarize_${SLURM_JOB_ID:-$$}
SERVER_LOG=$ABC/slurm_logs/abc_yam_eval_${SLURM_JOB_ID:-$$}.server.log
mkdir -p "$OUT_DIR" "$ABC/slurm_logs"

IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_EVAL=${GPU_ARR[0]}
GPU_SERVER=${GPU_ARR[1]:-${GPU_ARR[0]}}
echo "host=$(hostname) port=$PORT gpus=${CUDA_VISIBLE_DEVICES:-unset} ckpt=$CKPT_DIR worlds=$NUM_WORLDS"

# Checkpoint lives on GCS, so the server needs credentials the login node's gcloud
# does not pass through to a compute node.
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
    --policy-backend lap \
    --lap-host 127.0.0.1 --lap-port "$PORT" \
    --num-worlds "$NUM_WORLDS" \
    `# Bottles in the scene. success = ALL bottles in the bin at once, so 6 is` \
    `# all-or-nothing; BOTTLE_COUNT=1 makes a single placement score.` \
    --scene.bottle-count "${BOTTLE_COUNT:-6}" \
    `# Bottle spawn region (x0 x1 y0 y1). The default spans the whole table, which` \
    `# can drop the bottle beside an arm base where the top camera cannot see it.` \
    --scene.table-bounds ${TABLE_BOUNDS:-0.3025 0.8975 -0.65 0.65} \
    `# abc_yam stores RAW grasp_site FK orientation -- the i2rt axis-flip` \
    `# conjugation corrupts both the policy state and the decoded angular action.` \
    --no-lap-align-eef-rot \
    `# trained on absolute ANALOG gripper commands (median 0.444); thresholding at` \
    `# 0.5 is not faithful to training.` \
    `# Gripper execution. abc trains on ANALOG commands (46% strictly between 0.1 and` \
    `# 0.9, median 0.444), so analog is the faithful default. The served contract's` \
    `# binary profile is a hardcoded serving constant, not evidence. BINARIZE=1 to A/B it.` \
    ${BINARIZE_GRIPPER:+--lap-binarize-gripper} ${BINARIZE_GRIPPER:---no-lap-binarize-gripper} \
    --save-video \
    --output-dir "$OUT_DIR"

echo "=== summary ==="
python3 -c "import json;d=json.load(open('$OUT_DIR/summary.json'));print({k:v for k,v in d.items() if not isinstance(v,(list,dict))})" 2>/dev/null || true
echo "outputs: $OUT_DIR"
