#!/bin/bash
# Docker entrypoint
# Starts llama.cpp server in background, waits for it, then runs agent.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/qwen3.5-4b-q4.gguf}"
LLAMA_N_CTX="${LLAMA_N_CTX:-2048}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-0}"  # CPU-only: 0 (override via env var for local GPU)

echo "[start] Starting llama.cpp server with Qwen3.5-4B..."

# Cleanup handler for container shutdown
cleanup() {
    echo "[start] Shutting down..."
    kill $LLAMA_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start llama.cpp server in background
llama-server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8081 \
    --ctx-size "$LLAMA_N_CTX" \
    --n-gpu-layers "$LLAMA_N_GPU_LAYERS" \
    --cont-batching \
    --no-kv-offload \
    --flash-attn auto \
    --temp 0.0 \
    --repeat-penalty 1.0 \
    &
LLAMA_PID=$!

# Wait for server to be ready (health endpoint)
echo "[start] Waiting for llama.cpp server..."
MAX_WAIT=50  # seconds
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://127.0.0.1:8081/health > /dev/null 2>&1; then
        echo "[start] llama.cpp server ready (${i}s)"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "[start] ERROR: llama.cpp server failed to start in ${MAX_WAIT}s"
        kill $LLAMA_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo "Running agent..."
# Ensure /output directory exists for results
mkdir -p /output
# Run agent (reads from /input/tasks.json or stdin, writes to /output/results.json)
PYTHONPATH=/ python3 -u /agent/main.py

AGENT_EXIT=$?

# Cleanup
kill $LLAMA_PID 2>/dev/null || true
wait $LLAMA_PID 2>/dev/null || true

echo "[start] Agent finished with exit code $AGENT_EXIT"
exit $AGENT_EXIT
