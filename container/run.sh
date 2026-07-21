#!/usr/bin/env bash
# Entry point for the AMD ACT II Docker container.
# Starts llama-server, runs multi-prompt ensemble, shutdown.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/model.gguf}"
INPUT_PATH="${INPUT_PATH:-/input/tasks.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
N_CTX="${N_CTX:-2048}"
N_THREADS="${N_THREADS:-2}"
TIMEOUT="${TIMEOUT:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run.sh] Starting multi-prompt ensemble container"
echo "  Model:   $MODEL_PATH"
echo "  Input:   $INPUT_PATH"
echo "  Output:  $OUTPUT_DIR"
echo "  Threads: $N_THREADS"
echo "  Timeout: ${TIMEOUT}s"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[run.sh] ERROR: Model not found at $MODEL_PATH" >&2
    exit 1
fi

# Verify input exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "[run.sh] ERROR: Input not found at $INPUT_PATH" >&2
    exit 1
fi

exec python3 -m container.runner \
    --model "$MODEL_PATH" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-ctx "$N_CTX" \
    --n-threads "$N_THREADS" \
    --timeout "$TIMEOUT"
