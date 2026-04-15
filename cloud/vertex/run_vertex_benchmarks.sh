#!/usr/bin/env bash
# Master Vertex AI benchmark runner.
# Deploys FP16, INT8, INT4 sequentially, runs c=1 and c=8 benchmarks,
# deletes each endpoint immediately after to avoid extra cost.
# On any error, cleans up all endpoints before exiting.
#
# Usage:
#   bash cloud/vertex/run_vertex_benchmarks.sh PROJECT_ID HF_TOKEN
#
# Prerequisites:
#   gcloud auth application-default login
#   uv add google-cloud-aiplatform google-auth httpx

set -euo pipefail

PROJECT="${1:?Usage: $0 PROJECT_ID HF_TOKEN}"
HF_TOKEN="${2:?Usage: $0 PROJECT_ID HF_TOKEN}"
LOG="results/raw/run_vertex_benchmarks.log"
mkdir -p results/raw

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cleanup_all() {
    log "ERROR encountered — deleting all Vertex AI endpoints to avoid extra fees."
    uv run --python 3.11 python cloud/vertex/vertex_cleanup.py \
        --project "$PROJECT" --all 2>&1 | tee -a "$LOG" || true
    log "Emergency cleanup done. Exiting."
}
trap cleanup_all ERR

run_one() {
    local QUANT="$1"
    local MAX_LEN="$2"

    log "===== $QUANT ====="

    log "Deploying $QUANT endpoint..."
    uv run --python 3.11 python cloud/vertex/vertex_deploy.py \
        --project "$PROJECT" --hf-token "$HF_TOKEN" --quantize "$QUANT" \
        2>&1 | tee -a "$LOG"

    # Read endpoint ID saved by deploy script
    ENDPOINT_ID=$(cat "results/raw/vertex_${QUANT}_endpoint_id.txt")
    log "Endpoint ID: $ENDPOINT_ID"

    log "$QUANT benchmark c=1"
    uv run --python 3.11 python cloud/vertex/vertex_benchmark.py \
        --project "$PROJECT" --endpoint "$ENDPOINT_ID" \
        --concurrency 1 --requests 100 --max-len "$MAX_LEN" --quant "$QUANT" \
        2>&1 | tee -a "$LOG"

    log "$QUANT benchmark c=8"
    uv run --python 3.11 python cloud/vertex/vertex_benchmark.py \
        --project "$PROJECT" --endpoint "$ENDPOINT_ID" \
        --concurrency 8 --requests 100 --max-len "$MAX_LEN" --quant "$QUANT" \
        2>&1 | tee -a "$LOG"

    log "Deleting $QUANT endpoint..."
    uv run --python 3.11 python cloud/vertex/vertex_cleanup.py \
        --project "$PROJECT" --quant "$QUANT" \
        2>&1 | tee -a "$LOG"
    log "$QUANT done and deleted."
}

log "===== Vertex AI Benchmark Run ====="
log "Project: $PROJECT | Region: us-central1 | Machine: g2-standard-4 (L4)"

run_one "fp16" 1024
run_one "int8" 2048
run_one "int4" 4096

log "Push results to GitHub"
git add results/raw/vertex_*.csv results/raw/vertex_*.txt results/raw/run_vertex_benchmarks.log \
    cloud/vertex/
git commit -m "Add Vertex AI benchmark results: FP16, INT8, INT4 (c=1, c=8)"
git push

log "===== All Vertex AI benchmarks complete ====="
