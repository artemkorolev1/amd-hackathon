"""
Configuration — self-consistency voting with GPU local model + Fireworks escalation.
"""
import os

TASK_COUNT: int = 40
MAX_RUNTIME_SEC: int = 600
STARTUP_LIMIT_SEC: int = 60
LLAMA_SERVER_PORT: int = 8081
LLAMA_SERVER_URL: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
LLAMA_ENABLE: bool = os.environ.get("LLAMA_ENABLE", "1") == "1"
LOCAL_MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/nvidia-nemotron3-nano-4b-q4_k_m.gguf")

# Fireworks
DEFAULT_MODEL = "accounts/fireworks/models/kimi-k2p7-code"

# Sanitize FIREWORKS_BASE_URL: strip quotes, trailing slash, /chat/completions suffix
_raw_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
_raw_url = _raw_url.strip().strip("\"'").rstrip("/")
if _raw_url.endswith("/chat/completions"):
    _raw_url = _raw_url.removesuffix("/chat/completions").rstrip("/")
FIREWORKS_BASE_URL: str = _raw_url
del _raw_url

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
    "medium_max": 0.6,   # 0.3-0.6: medium → Fireworks medium model
}

# Parallelism (none — sequential, single-sample)
WORKERS: int = 1
REMOTE_WORKERS: int = 1
REMOTE_ATTEMPTS: int = 1

# Circuit breaker
REMOTE_CIRCUIT_BREAKER_LIMIT: int = 5
REMOTE_CIRCUIT_RETRY_AFTER: int = 30

# Time budget ratchets (env var overridable for local testing)
DEGRADE_50: float = float(os.environ.get("DEGRADE_50", "0.50"))
DEGRADE_70: float = float(os.environ.get("DEGRADE_70", "0.70"))
DEGRADE_85: float = float(os.environ.get("DEGRADE_85", "0.85"))


def resolve_model(complexity_score: float = 0.5) -> str:
    """Return the Fireworks model. No tier routing — always uses DEFAULT_MODEL."""
    allowed = os.environ.get("ALLOWED_MODELS", "")
    if allowed:
        first = allowed.split(",")[0].strip()
        if first:
            return first if "/" in first else f"accounts/fireworks/models/{first}"
    return DEFAULT_MODEL
