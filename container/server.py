"""
llama-server lifecycle manager for container module.

Handles: server start (with parallel slots), health check, shutdown.
Uses subprocess to manage llama-server as a background process.
"""

import json
import logging
import os
import platform
import signal
import subprocess
import time
import urllib.request
import urllib.error

logger = logging.getLogger("container.server")

DEFAULT_LLAMA_SERVER = "llama-server"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8081
HEALTH_URL = "http://127.0.0.1:8081/health"
SLOTS = 4


def find_llama_server() -> str:
    """Locate llama-server binary. Check common paths."""
    candidates = [
        DEFAULT_LLAMA_SERVER,
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        os.path.expanduser("~/.local/bin/llama-server"),
    ]
    # Check if in repo
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "bin", "llama-server")
    if os.path.exists(repo_path):
        candidates.insert(0, repo_path)
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return DEFAULT_LLAMA_SERVER  # hope it's on PATH


class ServerManager:
    """Manages a llama-server process with parallel slots."""

    def __init__(
        self,
        model_path: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        n_ctx: int = 2048,
        n_parallel: int = SLOTS,
        n_threads: int = 2,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.n_parallel = n_parallel
        self.n_threads = n_threads
        self.process: subprocess.Popen = None
        self.server_binary = find_llama_server()

    def start(self, timeout: float = 30.0) -> bool:
        """Start llama-server. Blocks until healthy or timeout."""
        if self.process and self.process.poll() is None:
            logger.info("Server already running")
            return True

        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return False

        cmd = [
            self.server_binary,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-np", str(self.n_parallel),
            "-t", str(self.n_threads),
            "--mlock",  # lock weights in RAM
            "--no-mmap",  # avoid mmap overhead on small RAM
            "--embedding",  # disabled, keep default
            "-ngl", "0",  # CPU-only
        ]

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )

        # Wait for health endpoint
        deadline = time.monotonic() + timeout
        last_err = ""
        while time.monotonic() < deadline:
            try:
                resp = urllib.request.urlopen(HEALTH_URL, timeout=2)
                if resp.status == 200:
                    logger.info("llama-server healthy on %s:%d",
                                self.host, self.port)
                    return True
            except (urllib.error.URLError, ConnectionRefusedError) as e:
                last_err = str(e)
                time.sleep(0.5)

        # Check if process crashed
        if self.process.poll() is not None:
            stderr_out = self.process.stderr.read().decode(errors="replace")[:500]
            logger.error("llama-server crashed: %s", stderr_out)

        logger.error("llama-server failed to start within %.0fs: %s",
                     timeout, last_err)
        return False

    def stop(self):
        """Stop the server gracefully."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            logger.info("llama-server stopped")
        self.process = None

    def is_healthy(self) -> bool:
        """Quick health check."""
        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=2)
            return resp.status == 200
        except Exception:
            return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
