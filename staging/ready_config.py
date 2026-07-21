"""ReadyConfig — Configuration loaded from environment variables.

All STAGING_* variables prefixed for the new parallel submission path.
Uses STAGING_ prefix so env vars don't collide with the existing pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from agent.resource_manager import WorkerResourceDemand


@dataclass
class ReadyConfig:
    """Central configuration for the staging parallel submission system."""

    # ── Worker counts ──
    fw_workers: int = 1
    loc_workers: int = 1
    det_workers: int = 2

    # ── Judgment ──
    judgment_votes: int = 5           # how many sequential tries per task
    vote_min_agreement: float = 0.5   # minimum fraction for majority (0.0-1.0)

    # ── Timeouts ──
    worker_timeout_s: float = 30.0    # per-try timeout
    deadline_s: float = 600.0         # total wall-clock deadline
    judge_timeout_s: float = 60.0     # how long before judge force-judges with available votes

    # ── Fireworks ──
    fw_api_key: str = ""
    fw_models: list[str] = field(default_factory=lambda: [
        "accounts/fireworks/models/deepseek-v4-flash",
    ])

    # ── Local model (backward compat: single model path) ──
    loc_model_path: str = "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    loc_model_configs: list[dict] = field(default_factory=lambda: [
        {"id": "qwen2.5-instruct", "path": "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
         "categories": ["factual", "logic", "math", "summarization", "code_debug"]},
        {"id": "qwen2.5-coder", "path": "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
         "categories": ["ner", "code_gen", "code_debug"]},
        {"id": "gemma-3", "path": "/models/gemma-3-1b-it-Q4_K_M.gguf",
         "categories": ["sentiment", "code_gen", "code_debug"]},
    ])
    loc_n_gpu_layers: int = 0            # 0 = CPU only (grader has no GPU)
    loc_n_threads: int = 2               # grader has 2 vCPU

    # ── Strategy ──
    tiebreaker_strategy: str = "fw_priority"  # fw_priority | fastest | deterministic
    fallback_strategy: str = "best_available"

    # ── Tuning / emergency / heartbeat ──
    steal_threshold: int = 2                # tasks before worker tries to steal
    steal_timeout_s: float = 0.5            # timeout on steal-request replies
    heartbeat_timeout_s: float = 60.0       # worker declared dead after this
    reservation_timeout_s: float = 30.0     # how long before reserved task released to all workers
    monitor_interval_s: float = 5.0         # how often monitor loop checks
    judge_poll_interval_s: float = 0.05     # how often judge polls for votes
    emergency_vote_reduction: int = 2       # reduce votes in deadline mode
    per_worker_inbox_size: int = 3          # max backlog per-worker inbox

    # ── Category → worker type priority map ──
    category_priority: dict[str, list[str]] = field(default_factory=lambda: {
        "math":          ["deterministic", "local", "fireworks"],
        "logic":         ["fireworks", "local"],
        "factual":       ["deterministic", "fireworks", "local"],
        "sentiment":     ["deterministic", "local", "fireworks"],
        "ner":           ["deterministic", "fireworks", "local"],
        "summarization": ["deterministic", "fireworks", "local"],
        "code_gen":      ["fireworks", "local"],
        "code_debug":    ["fireworks", "local", "deterministic"],
    })

    # ── Ablation toggles ──
    ablation_disable_fireworks: int = 0
    ablation_disable_local: int = 0
    ablation_disable_deterministic: int = 0
    ablation_disable_voting: int = 0
    ablation_votes: int = 0           # 0 = use default judgment_votes
    ablation_temperature_sweep: str = ""  # comma-separated floats, e.g. "0.1,0.3,0.5"
    ablation_tiebreaker: str = ""
    ablation_single_worker: str = ""   # e.g. "fireworks", "local", "deterministic"
    ablation_force_crash: int = 0
    ablation_empty_model_path: int = 0

    @classmethod
    def from_env(cls) -> "ReadyConfig":
        """Load configuration from environment variables with sensible defaults."""
        # ── Backward compat: MODEL_PATH env var → single-model mode ──
        model_path_env = os.environ.get("MODEL_PATH", "")
        if model_path_env:
            loc_model_configs = [{
                "id": "qwen2.5-instruct",
                "path": model_path_env,
                "categories": ["factual", "logic", "math", "sentiment",
                               "ner", "summarization", "code_gen", "code_debug"],
            }]
            loc_workers = int(os.environ.get("STAGING_LOC_WORKERS", "1"))
        else:
            # Multi-model mode: try JSON env var or use defaults
            configs_json = os.environ.get("STAGING_LOC_MODEL_CONFIGS", "")
            if configs_json:
                loc_model_configs = json.loads(configs_json)
            else:
                # Default 3-model config
                loc_model_configs = [
                    {"id": "qwen2.5-instruct", "path": "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                     "categories": ["factual", "logic", "math", "summarization", "code_debug"]},
                    {"id": "qwen2.5-coder", "path": "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
                     "categories": ["ner", "code_gen", "code_debug"]},
                    {"id": "gemma-3", "path": "/models/gemma-3-1b-it-Q4_K_M.gguf",
                     "categories": ["sentiment", "code_gen", "code_debug"]},
                ]
            loc_workers = len(loc_model_configs)

        return cls(
            fw_workers=int(os.environ.get("STAGING_FW_WORKERS", "1")),
            loc_workers=loc_workers,
            det_workers=int(os.environ.get("STAGING_DET_WORKERS", "2")),
            judgment_votes=int(os.environ.get("STAGING_JUDGMENT_VOTES", "5")),
            vote_min_agreement=float(os.environ.get("STAGING_VOTE_MIN_AGREEMENT", "0.5")),
            worker_timeout_s=float(os.environ.get("STAGING_WORKER_TIMEOUT", "30.0")),
            deadline_s=float(os.environ.get("DEADLINE_S", "600")),
            judge_timeout_s=float(os.environ.get("STAGING_JUDGE_TIMEOUT_S", "60.0")),
            fw_api_key=os.environ.get("FIREWORKS_API_KEY", ""),
            fw_models=[m.strip() for m in
                       os.environ.get("STAGING_FW_MODELS",
                                      "accounts/fireworks/models/deepseek-v4-flash")
                       .split(",") if m.strip()],
            loc_model_path=model_path_env or
                           "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            loc_model_configs=loc_model_configs,
            loc_n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),
            loc_n_threads=int(os.environ.get("N_THREADS", "2")),
            tiebreaker_strategy=os.environ.get("STAGING_TIEBREAKER", "fw_priority"),
            fallback_strategy=os.environ.get("STAGING_FALLBACK", "best_available"),
            # ── Tuning / emergency / heartbeat ──
            steal_threshold=int(os.environ.get("STAGING_STEAL_THRESHOLD", "2")),
            steal_timeout_s=float(os.environ.get("STAGING_STEAL_TIMEOUT", "0.5")),
            heartbeat_timeout_s=float(os.environ.get("STAGING_HEARTBEAT_TIMEOUT", "60.0")),
            reservation_timeout_s=float(os.environ.get("STAGING_RESERVATION_TIMEOUT_S", "30.0")),
            monitor_interval_s=float(os.environ.get("STAGING_MONITOR_INTERVAL", "5.0")),
            judge_poll_interval_s=float(os.environ.get("STAGING_JUDGE_POLL_INTERVAL", "0.05")),
            emergency_vote_reduction=int(os.environ.get("STAGING_EMERGENCY_VOTE_REDUCTION", "2")),
            per_worker_inbox_size=int(os.environ.get("STAGING_PER_WORKER_INBOX_SIZE", "3")),
            # ── Ablation toggles ──
            ablation_disable_fireworks=int(os.environ.get("ABLATION_DISABLE_FIREWORKS", "0")),
            ablation_disable_local=int(os.environ.get("ABLATION_DISABLE_LOCAL", "0")),
            ablation_disable_deterministic=int(os.environ.get("ABLATION_DISABLE_DETERMINISTIC", "0")),
            ablation_disable_voting=int(os.environ.get("ABLATION_DISABLE_VOTING", "0")),
            ablation_votes=int(os.environ.get("ABLATION_VOTES", "0")),
            ablation_temperature_sweep=os.environ.get("ABLATION_TEMPERATURE_SWEEP", ""),
            ablation_tiebreaker=os.environ.get("ABLATION_TIEBREAKER", ""),
            ablation_single_worker=os.environ.get("ABLATION_SINGLE_WORKER", ""),
            ablation_force_crash=int(os.environ.get("ABLATION_FORCE_CRASH", "0")),
            ablation_empty_model_path=int(os.environ.get("ABLATION_EMPTY_MODEL_PATH", "0")),
        )

    @property
    def total_workers(self) -> int:
        return self.fw_workers + self.loc_workers + self.det_workers

    def build_resource_demands(self) -> list[WorkerResourceDemand]:
        """Build worker resource demands from config counts."""
        demands = []
        if self.loc_workers > 0:
            for mc in self.loc_model_configs:
                demands.append(WorkerResourceDemand(
                    worker_type="local",
                    gpu_vram_gb=1.5,
                    cpu_cores=1.0,
                    ram_gb=1.0,
                    count=1,  # one per model
                ))
        if self.det_workers > 0:
            demands.append(WorkerResourceDemand(
                worker_type="deterministic",
                gpu_vram_gb=0.0,
                cpu_cores=0.1,
                ram_gb=0.1,
                count=self.det_workers,
            ))
        if self.fw_workers > 0:
            demands.append(WorkerResourceDemand(
                worker_type="fireworks",
                gpu_vram_gb=0.0,
                cpu_cores=0.2,
                ram_gb=0.2,
                count=self.fw_workers,
            ))
        return demands
