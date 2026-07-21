"""
Configuration — self-consistency voting with GPU local model (100% local pipeline).
"""
import os

TASK_COUNT: int = 40
MAX_RUNTIME_SEC: int = 600
STARTUP_LIMIT_SEC: int = 60
LLAMA_SERVER_PORT: int = 8081
LLAMA_SERVER_URL: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
LLAMA_ENABLE: bool = os.environ.get("LLAMA_ENABLE", "1") == "1"
LOCAL_MODEL_PATH: str = os.environ.get("MODEL_PATH", "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

# Self-consistency voting (no parallelization — single sample)
CONSENSUS_SAMPLES: int = 1
CONSENSUS_SAMPLES_MAX: int = 1
LOCAL_MAX_TOKENS: int = 512

CONSENSUS_THRESHOLDS: dict[str, float] = {
    "ner": 0.6,
    "sentiment": 0.5,
    "code_debug": 0.6,
    "code_gen": 0.6,
    "math": 0.75,
    "logic": 0.75,
    "summarization": 0.0,
    "factual": 0.5,
    "general": 0.5,
}

# Complexity thresholds for decision table routing
COMPLEXITY_THRESHOLDS = {
    "simple_max": 0.3,   # < 0.3: simple → deterministic solver (no API)
    "medium_max": 0.6,   # 0.3-0.6: medium → local LLM
}

# Parallelism (none — sequential, single-sample)
WORKERS: int = 1
REMOTE_WORKERS: int = 1
REMOTE_ATTEMPTS: int = 1

# Time budget ratchets (env var overridable for local testing)
DEGRADE_50: float = float(os.environ.get("DEGRADE_50", "0.50"))
DEGRADE_70: float = float(os.environ.get("DEGRADE_70", "0.70"))
DEGRADE_85: float = float(os.environ.get("DEGRADE_85", "0.85"))


# ── Per-role resource profiles ──────────────────────────────────────────────
# Managed by the orchestrator, not by cells — these tell the scheduler what
# each role needs so it can bin-pack work across available hardware.
ROLE_RESOURCE_DEFAULTS = {
    "deterministic": {"gpu_vram_gb": 0.0, "cpu_cores": 0.1, "ram_gb": 0.1, "max_concurrent": 8},
    "local_llm":     {"gpu_vram_gb": 1.5, "cpu_cores": 1.0, "ram_gb": 1.0, "max_concurrent": 2},
    "api_llm":       {"gpu_vram_gb": 0.0, "cpu_cores": 0.2, "ram_gb": 0.2, "max_concurrent": 8},
    "workflow":      {"gpu_vram_gb": 1.5, "cpu_cores": 1.0, "ram_gb": 1.5, "max_concurrent": 1},
    "aggregator":    {"gpu_vram_gb": 0.0, "cpu_cores": 0.2, "ram_gb": 0.2, "max_concurrent": 2},
}
