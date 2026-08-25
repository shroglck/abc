#!/bin/bash
# Fetch the lap_yam_h30/yam_h30/25000 checkpoint for serving (params + assets only;
# train_state is optimizer state the policy server never reads, and it doubles the
# transfer). Run under srun WITHOUT a GPU -- this is pure network/IO:
#
#   srun --partition=all --cpus-per-task=8 --mem=16G --time=1:00:00 \
#        bash scripts/fetch_abc_yam_29k.sh
#
# Uses `gcloud storage`, not gsutil: gsutil on the compute nodes authenticates
# anonymously even with GOOGLE_APPLICATION_CREDENTIALS set and dies with a 401.
# CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE points gcloud at the service-account key
# for THIS process only, so the user's own gcloud config is left untouched.
set -uo pipefail

SRC=gs://v6_east1d/checkpoints/lap_yam_h30/yam_h30
DST=${DST:-/n/fs/prism-data/shresth/ego-lap/checkpoint_yam_h30}
STEP=25000

export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/u/sg0803/irom-service-account-key.json}
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$GOOGLE_APPLICATION_CREDENTIALS"
export CURL_CA_BUNDLE=${CURL_CA_BUNDLE:-/n/fs/prism-data/shresth/ego-lap/.venv/lib/python3.11/site-packages/certifi/cacert.pem}

echo "host=$(hostname)  src=$SRC/$STEP  dst=$DST"
mkdir -p "$DST/$STEP"

# Fail fast and loudly if credentials did not survive the hop to the compute node,
# rather than after a partial 12 GB transfer.
if ! gcloud storage ls "$SRC/$STEP/" > /dev/null 2>&1; then
  echo "FATAL: cannot list $SRC/$STEP -- credentials did not reach this node"; exit 1
fi
echo "auth OK"

# run_manifest.json lives BESIDE the numeric step dir, which is where
# serving_contract.checkpoint_manifest_path() looks for it.
gcloud storage cp "$SRC/run_manifest.json" "$DST/run_manifest.json" || exit 1
for item in _CHECKPOINT_METADATA commit_success.txt; do
  gcloud storage cp "$SRC/$STEP/$item" "$DST/$STEP/$item" || exit 1
done
for d in assets params; do
  echo "== $d =="
  gcloud storage rsync --recursive "$SRC/$STEP/$d" "$DST/$STEP/$d" || exit 1
done

echo "=== local tree ==="
ls -la "$DST" "$DST/$STEP"
du -sh "$DST/$STEP/params"
echo "done: $DST/$STEP"
