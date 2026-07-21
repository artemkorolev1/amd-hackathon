#!/usr/bin/env python3
"""Experiment logger — structured versioned JSON tracking for GEPA runs.

Logs every generation snapshot: cell configs, metrics, Pareto fronts,
routing table changes, and agent decisions.

Usage:
    logger = ExperimentLogger("gepa_logs/run_001")
    logger.log_generation(gen=0, population=..., pareto_fronts=...)
    logger.log_decision("routing_update", details={...})
    logger.summary()  # prints compact report
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """Structured experiment tracker that writes versioned JSON snapshots.

    Directory layout:
        <run_dir>/
            meta.json              — run metadata (timestamp, config, tags)
            gen_000.json           — generation 0 snapshot
            gen_001.json           — generation 1 snapshot
            ...
            decisions.jsonl        — append-only log of agent decisions
            final.json             — final results / Pareto front
    """

    def __init__(
        self,
        run_dir: str = "gepa_logs",
        run_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.run_dir = Path(run_dir)
        if run_name:
            self.run_dir = self.run_dir / run_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.run_dir / f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Write meta.json
        meta = {
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "run_name": self.run_dir.name,
            "tags": [],
        }
        if metadata:
            meta.update(metadata)
        self._write_json("meta.json", meta)

        self._decisions_path = self.run_dir / "decisions.jsonl"

    # ── Public API ──────────────────────────────────────────────────────────

    def log_generation(
        self,
        gen: int,
        population: list,
        pareto_fronts: Optional[dict] = None,
        metrics: Optional[dict] = None,
        extra: Optional[dict] = None,
    ):
        """Log a full generation snapshot.

        Args:
            population: list of Cell objects (or dicts).
            pareto_fronts: dict mapping model_key -> list of cell names.
            metrics: summary metrics for this generation.
            extra: any additional metadata to include.
        """
        snapshot = {
            "generation": gen,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "population_size": len(population),
            "cells": [self._cell_to_dict(c) for c in population],
            "pareto_fronts": pareto_fronts or {},
            "metrics": metrics or {},
        }
        if extra:
            snapshot.update(extra)

        fname = f"gen_{gen:03d}.json"
        self._write_json(fname, snapshot)

    def log_decision(self, action: str, details: Optional[dict] = None):
        """Append a structured agent decision to the decisions log."""
        entry = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "action": action,
        }
        if details:
            entry["details"] = details
        self._append_jsonl(self._decisions_path, entry)

    def log_routing_table(self, version: int, table: dict, backtest_ok: bool, note: str = ""):
        """Log a routing table update or proposal."""
        self.log_decision("routing_table_update", {
            "version": version,
            "table": table,
            "backtest_passed": backtest_ok,
            "note": note,
        })

    def save_final(self, results: dict):
        """Write the final results summary."""
        self._write_json("final.json", results)
        self.log_decision("run_complete", {"total_generations": results.get("num_generations")})

    def add_tag(self, tag: str):
        """Add a human-readable tag to the run (e.g. 'smollm2-factual-only')."""
        meta_path = self.run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        tags = meta.get("tags", [])
        if tag not in tags:
            tags.append(tag)
        meta["tags"] = tags
        self._write_json("meta.json", meta)

    def summary(self) -> str:
        """Print a compact one-line summary of the run."""
        meta_path = self.run_dir / "meta.json"
        if not meta_path.exists():
            return f"[{self.run_dir.name}] no meta.json"

        with open(meta_path) as f:
            meta = json.load(f)

        gen_files = sorted(self.run_dir.glob("gen_*.json"))
        dec_count = 0
        dec_path = self.run_dir / "decisions.jsonl"
        if dec_path.exists():
            with open(dec_path) as f:
                dec_count = sum(1 for _ in f)

        return (
            f"[{self.run_dir.name}] "
            f"gens={len(gen_files)} "
            f"decisions={dec_count} "
            f"tags={meta.get('tags', [])} "
            f"created={meta.get('created_at_iso', '?')}"
        )

    def get_latest_gen(self) -> Optional[dict]:
        """Return the most recent generation snapshot, or None if none exist."""
        gen_files = sorted(self.run_dir.glob("gen_*.json"))
        if not gen_files:
            return None
        with open(gen_files[-1]) as f:
            return json.load(f)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _cell_to_dict(self, cell) -> dict:
        if hasattr(cell, "to_dict"):
            return cell.to_dict()
        return dict(cell) if isinstance(cell, dict) else {"repr": str(cell)}

    def _write_json(self, fname: str, data: dict):
        path = self.run_dir / fname
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @staticmethod
    def _append_jsonl(path: Path, entry: dict):
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ── Top-level helpers for quick use ─────────────────────────────────────────

def _auto_run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def create_run(
    base_dir: str = "gepa_logs",
    tags: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> ExperimentLogger:
    """Convenience: create a new experiment run with metadata."""
    meta = {}
    if tags:
        meta["tags"] = tags
    if config:
        meta["config"] = config
    return ExperimentLogger(base_dir, metadata=meta)
