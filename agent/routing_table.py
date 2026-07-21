#!/usr/bin/env python3
"""Routing table — bridges GEPA Pareto-optimal cells into the production Pipeline.

The GEPA Orchestrator maintains Pareto fronts per task type.
After each generation (or on demand), the best cell per task is promoted
into a RoutingTable, which the production Pipeline reads to decide
which cell to use for a given query.

The RoutingTable is versioned and gated by a backtest check, so only
verifiably-better cells replace existing routes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class RoutingTable:
    """Versioned mapping from (task_category) → cell configuration.

    Each entry describes the cell to use for a given task category:
        category: str     — pipeline category name (e.g. "factual", "math")
        cell_name: str    — name of the selected cell
        model_key: str    — which model to load
        system_prompt: str
        decoding: dict    — temperature, max_tokens, etc.
        aggregation: str  — "single", "majority_vote", etc.
        accuracy: float   — last-measured accuracy
        updated_at: float — timestamp of this entry

    The table is immutable once published — replace the whole table atomically.
    """

    def __init__(self, initial_entries: Optional[list[dict]] = None):
        self._version: int = 0
        self._entries: dict[str, dict] = {}  # category → entry dict
        self._history: list[dict] = []
        if initial_entries:
            for entry in initial_entries:
                self._upsert_entry(entry)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def categories(self) -> list[str]:
        return list(self._entries.keys())

    def get(self, category: str) -> Optional[dict]:
        """Return the routing entry for a category, or None."""
        return self._entries.get(category)

    def select(self, category: str) -> Optional[dict]:
        """Alias for get(). Returns None if no route exists."""
        return self.get(category)

    # ── Workflow step support ──────────────────────────────────────────

    def get_role_for_category(self, category: str) -> Optional[str]:
        """Return the execution-backend role configured for this category, or None."""
        entry = self._entries.get(category)
        if entry and "role" in entry:
            return entry["role"]
        return None

    def get_workflow_steps(self, category: str) -> Optional[list]:
        """Return the workflow steps for a category, or None if single-shot.

        Steps are returned as a list of dicts (JSON-serialisable) ready
        to be converted to StepConfig objects via StepConfig.from_dict().
        """
        entry = self._entries.get(category)
        if entry is None:
            return None
        return entry.get("steps")

    def has_workflow(self, category: str) -> bool:
        """Return True if the route for *category* uses a multi-step workflow."""
        steps = self.get_workflow_steps(category)
        return steps is not None and len(steps) > 0

    def store_workflow_template(
        self,
        category: str,
        steps: list,
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a workflow template (list of StepConfig dicts) for a category.

        Args:
            category: Pipeline category name (e.g. "math", "ner").
            steps: List of StepConfig dicts (name, system_prompt, tool, etc.).
            metadata: Optional dict with template_name, version, description.

        Returns:
            New routing table version number.

        This creates or updates the routing entry for the category to point
        at the given workflow steps. The entry's ``aggregation`` is set to
        ``"workflow"`` automatically.
        """
        entry = self._entries.get(category, {})
        entry["category"] = category
        entry["steps"] = steps
        entry["aggregation"] = "workflow"
        entry["workflow_metadata"] = metadata or {}
        entry["updated_at"] = time.time()
        self._upsert_entry(entry)
        return self._publish([entry])

    def update_from_cells(
        self,
        cells: list,
        backtest_results: Optional[dict] = None,
        strict: bool = True,
    ) -> int:
        """Update routing table from a population of Cell objects.

        For each task category, picks the cell with the highest accuracy
        (or the first cell if no accuracy metadata is available).

        Args:
            cells: list of Cell objects (or dicts with at minimum
                   task_id/pipeline_category + model_key + system_prompt).
            backtest_results: optional dict of {cell_name: passed_bool}.
            strict: if True, only update categories where the new cell
                    passes backtest (backtest_results required).

        Returns:
            new version number, or -1 if the update was rejected.
        """
        # Group cells by category
        by_cat: dict[str, list] = {}
        for c in cells:
            cat = self._cell_category(c)
            by_cat.setdefault(cat, []).append(c)

        new_entries: list[dict] = []

        for cat, cat_cells in by_cat.items():
            # Pick best (highest accuracy) or first
            best = self._pick_best(cat_cells)
            if best is None:
                continue

            entry = self._cell_to_entry(cat, best)
            if backtest_results is not None:
                passed = backtest_results.get(entry.get("cell_name", best.get("name", "")), False)
                if strict and not passed:
                    continue  # skip — not backtest-verified
                entry["backtest_passed"] = passed

            new_entries.append(entry)

        if not new_entries:
            return -1

        return self._publish(new_entries)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot."""
        return {
            "version": self._version,
            "entries": self._entries,
            "history_count": len(self._history),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> RoutingTable:
        with open(path) as f:
            data = json.load(f)
        table = cls()
        table._version = data.get("version", 0)
        table._entries = data.get("entries", {})
        table._history = data.get("history", [])
        return table

    # ── Internal ────────────────────────────────────────────────────────────

    def _cell_category(self, cell) -> str:
        """Get pipeline category name from a cell dict or object."""
        # Cell object
        if hasattr(cell, "pipeline_category"):
            return cell.pipeline_category
        if hasattr(cell, "task_label"):
            return cell.task_label
        # Dict
        if isinstance(cell, dict):
            return cell.get("pipeline_category",
                            cell.get("task_id",
                                     cell.get("category", "unknown")))
        return "unknown"

    def _pick_best(self, cells: list) -> Optional[dict]:
        """Pick the best cell from a list by accuracy, else first."""
        best = None
        best_acc = -1.0
        for c in cells:
            meta = c.metadata if hasattr(c, "metadata") else c.get("metadata", {})
            acc = meta.get("accuracy", -1.0) if isinstance(meta, dict) else -1.0
            if acc > best_acc:
                best_acc = acc
                best = c
        return best or (cells[0] if cells else None)

    def _cell_to_entry(self, cat: str, cell) -> dict:
        """Convert a Cell (object or dict) into a routing entry."""
        if hasattr(cell, "to_dict"):
            cd = cell.to_dict(include_metadata=False)
            meta = cell.metadata
            entry = {
                "category": cat,
                "cell_name": cd.get("name", cell.name),
                "model_key": cd.get("model_key", cell.model_key),
                "system_prompt": cd.get("system_prompt", cell.system_prompt),
                "decoding": cd.get("decoding", {}),
                "aggregation": cd.get("aggregation", cell.aggregation),
                "accuracy": meta.get("accuracy", 0.0) if meta else 0.0,
                "updated_at": time.time(),
            }
            # Capture workflow steps if this is a workflow cell
            if hasattr(cell, "steps") and cell.steps:
                entry["steps"] = [s.to_dict() if hasattr(s, "to_dict") else s
                                  for s in cell.steps]
                entry["aggregation"] = "workflow"
            # Capture execution-backend role if present
            if hasattr(cell, "role") and cell.role:
                entry["role"] = cell.role
            return entry
        if isinstance(cell, dict):
            meta = cell.get("metadata", {})
            entry = {
                "category": cat,
                "cell_name": cell.get("name", ""),
                "model_key": cell.get("model_key", ""),
                "system_prompt": cell.get("system_prompt", ""),
                "decoding": cell.get("decoding", {}),
                "aggregation": cell.get("aggregation", "single"),
                "accuracy": meta.get("accuracy", 0.0) if isinstance(meta, dict) else 0.0,
                "updated_at": time.time(),
            }
            # Capture workflow steps from dict-based cells
            steps_raw = cell.get("steps")
            if steps_raw:
                entry["steps"] = steps_raw
                entry["aggregation"] = "workflow"
            role_val = cell.get("role")
            if role_val:
                entry["role"] = role_val
            return entry
        return {"category": cat, "cell_name": str(cell)}

    def _upsert_entry(self, entry: dict):
        cat = entry.get("category", "unknown")
        self._entries[cat] = entry

    def _publish(self, new_entries: list[dict]) -> int:
        """Atomically replace entries and bump version."""
        old = dict(self._entries)
        for entry in new_entries:
            self._upsert_entry(entry)
        self._version += 1
        self._history.append({
            "version": self._version,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "changes": len(new_entries),
            "categories": [e.get("category") for e in new_entries],
        })
        return self._version
