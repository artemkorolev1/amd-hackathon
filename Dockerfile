# amd-hackathon-submit — modular Pipeline agent
# Build: docker buildx build --platform linux/amd64 -t ghcr.io/artemkorolev1/amd-hackathon-submit:v15 --load .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /

# Runtime libs for llama-cpp-python CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && apt-get purge -y --auto-remove build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && find /usr/local/lib -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy agent module (plug-and-play Pipeline class)
COPY agent/ /agent/

# Copy scripts, staging (parallel pool system), and runner (eval wrapper)
COPY scripts/ /scripts/
COPY staging/ /staging/
COPY runner/ /runner/

# Copy CLI harness entrypoint
COPY harness.py /harness.py

# Copy root dispatcher (routes to harness or staging/entrypoint)
COPY dispatcher.py /dispatcher.py

# Copy models (GGUF files — explicit for layer caching)
COPY models/qwen2.5-1.5b-instruct-q4_k_m.gguf /models/qwen2.5-1.5b-instruct-q4_k_m.gguf
COPY models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf /models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
COPY models/gemma-3-1b-it-Q4_K_M.gguf /models/gemma-3-1b-it-Q4_K_M.gguf

# Copy fact database (FTS5 knowledge base for factual QA)
COPY data/facts/ /data/facts/

# Ensure grader I/O paths
RUN mkdir -p /input /output

ENV PYTHONPATH=/ \
    MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    N_GPU_LAYERS=-1 \
    N_THREADS=4 \
    N_CTX=2048

ENTRYPOINT ["python3", "-u", "dispatcher.py"]
