#!/bin/bash
#SBATCH --job-name=lap_putbottles_eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/n/fs/prism-data/shresth/abc/slurm_logs/%x_%j.out
#SBATCH --error=/n/fs/prism-data/shresth/abc/slurm_logs/%x_%j.err

# Longer LAP (ego-lap) put-bottles eval, self-contained in one SLURM job:
#   GPU 1 -> LAP flow policy server (ckeckpoint_bimanip/20000)
#   GPU 0 -> MuJoCo-Warp put-bottles eval, --num-worlds worlds
# Conventions are the source-verified ones baked into abc_minimal/config.py
# defaults (frame="egocentric frame", rotation_delta_in_base=False, is_bimanual).

set -euo pipefail

ABC=/n/fs/prism-data/shresth/abc
EGO=/n/fs/prism-data/shresth/ego-lap
CKPT_DIR=$EGO/ckeckpoint_bimanip/20000
CKPT_CONFIG=lap_robot_human_and_bimanual
NUM_WORLDS=${NUM_WORLDS:-25}

# Per-job port so multiple jobs can share a node without colliding.
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))
OUT_DIR=$ABC/outputs/lap_bimanip20k_slurm_${SLURM_JOB_ID}
SERVER_LOG=$ABC/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.server.log
mkdir -p "$OUT_DIR"

# Split the GPUs SLURM allocated to this job (robust to cgroup renumbering vs
# raw physical ids): eval -> first, server -> second (fallback to first).
IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_EVAL=${GPU_ARR[0]}
GPU_SERVER=${GPU_ARR[1]:-${GPU_ARR[0]}}

echo "host=$(hostname) job=$SLURM_JOB_ID port=$PORT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "gpu_eval=$GPU_EVAL gpu_server=$GPU_SERVER num_worlds=$NUM_WORLDS out=$OUT_DIR"

# --- launch policy server on the 2nd allocated GPU ------------------------
(
  cd "$EGO"
  PYTHONPATH="$EGO/src" JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPU_SERVER" \
    "$EGO/.venv/bin/python" scripts/serve_policy.py \
      --env LAP --port "$PORT" \
      policy:checkpoint \
      --policy.config "$CKPT_CONFIG" \
      --policy.dir "$CKPT_DIR" \
      --policy.type flow
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Kill the server no matter how the eval exits.
cleanup() {
  echo "cleanup: stopping server $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for the server to come up (JAX+params load is slow) --------------
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

# --- run the eval on the 1st allocated GPU ---------------------------------
cd "$ABC"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" "$ABC/.venv/bin/python" -m abc_minimal.eval_policy \
    --policy-backend lap \
    --lap-host 127.0.0.1 --lap-port "$PORT" \
    --num-worlds "$NUM_WORLDS" \
    --save-video \
    --output-dir "$OUT_DIR"

echo "=== eval done; summary ==="
cat "$OUT_DIR/summary.json" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:v for k,v in d.items() if not isinstance(v,(list,dict))})" || true
