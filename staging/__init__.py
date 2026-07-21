"""Staging — parallel submission architecture for AMD ACT II Track 1.

This module builds AROUND the existing agent pipeline (agent/pipeline.py)
without touching any files in agent/.

Components:
- ready_config  : Configuration from env vars
- ready_queue   : Multi-category task queue
- ready_pool    : Worker pool manager (Fireworks, Local, Deterministic)
- ready_worker  : Worker base class with 5-try sequential processing
- ready_judge   : Voting/judgment module
- workers/      : Worker implementations
- entrypoint    : Container entrypoint (replaces harness.py for staging)
"""

from .ready_config import ReadyConfig
from .ready_queue import ReadyQueue, ReadyTask
from .ready_pool import ReadyMonitor
from .ready_judge import ReadyJudge

__all__ = [
    "ReadyConfig",
    "ReadyQueue", "ReadyTask",
    "ReadyMonitor",
    "ReadyJudge",
]
