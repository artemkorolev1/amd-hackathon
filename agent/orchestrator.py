#!/usr/bin/env python3
"""GEPA Orchestrator — central coordinator for the agentic GEPA system.

Manages the population lifecycle, coordinates sub-agents (mutation,
evaluation, analysis), maintains Pareto fronts per task/model,
and publishes routing table updates.

Usage:
    orchestrator = GEPAOrchestrator(model_cache)
    orchestrator.seed_generation_0()
    orchestrator.run_generations(n=3)
    orchestrator.publish_routing_table()
    orchestrator.report()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from agent.cell import (
    Cell,
    TASK_IDS,
    TASK_LABELS,
    TASK_TO_PIPELINE_CAT,
    deduplicate_population,
    population_from_json,
    population_to_json,
)
from agent.experiment_logger import ExperimentLogger
from agent.routing_table import RoutingTable

logger = logging.getLogger("gepa_orchestrator")


# ── Pareto helpers ──────────────────────────────────────────────────────────

def _pareto_dominates(a: dict, b: dict) -> bool:
    """Pareto dominance for 3 objectives (all higher=better after normalisation).

    Objectives: accuracy_norm, tokens_norm, latency_norm.
    """
    objs = ["acc_norm", "tokens_norm", "lat_norm"]
    better_or_eq = all(a.get(o, 0) >= b.get(o, 0) - 1e-10 for o in objs)
    strictly = any(a.get(o, 0) > b.get(o, 0) + 1e-10 for o in objs)
    return better_or_eq and strictly


def _fast_non_dominated_sort(population: list[dict]) -> list[list[int]]:
    """NSGA-II fast non-dominated sort. Returns fronts (list of indices)."""
    n = len(population)
    S = [set() for _ in range(n)]
    n_count = [0] * n
    fronts = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _pareto_dominates(population[i], population[j]):
                S[i].add(j)
            elif _pareto_dominates(population[j], population[i]):
                n_count[i] += 1

    current = [i for i in range(n) if n_count[i] == 0]
    if not current:
        return [list(range(n))]
    fronts.append(current)

    while current:
        next_front = []
        for i in current:
            for j in S[i]:
                n_count[j] -= 1
                if n_count[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        current = next_front

    return fronts


def _pareto_front_indices(
    cells: list[Cell], objectives: tuple = ("accuracy", "avg_output_tokens", "avg_latency_ms")
) -> list[int]:
    """Return indices of cells on the Pareto front (front 0)."""
    if len(cells) <= 1:
        return list(range(len(cells)))

    # Build normalised model_pop entries
    accs = [c.metadata.get(objectives[0], 0.0) for c in cells]
    toks = [c.metadata.get(objectives[1], 0) for c in cells]
    lats = [c.metadata.get(objectives[2], 0.0) for c in cells]

    min_a, max_a = min(accs), max(accs)
    min_t, max_t = min(toks), max(toks)
    min_l, max_l = min(lats), max(lats)

    model_pop = []
    for i in range(len(cells)):
        model_pop.append({
            "acc_norm": (accs[i] - min_a) / (max_a - min_a + 1e-9),
            "tokens_norm": 1.0 - (toks[i] - min_t) / (max_t - min_t + 1e-9),
            "lat_norm": 1.0 - (lats[i] - min_l) / (max_l - min_l + 1e-9),
            "idx": i,
        })

    fronts = _fast_non_dominated_sort(model_pop)
    return fronts[0] if fronts else []


# ── Orchestrator ────────────────────────────────────────────────────────────

class GEPAOrchestrator:
    """Central coordinator for the agentic GEPA system.

    Args:
        model_cache: ModelCache instance for lazy-loaded LLMs.
        model_keys: list of model keys to use.
        mutation_agent: pre-configured MutationAgent instance.
        evaluation_agent: pre-configured EvaluationAgent instance.
        analysis_agent: pre-configured AnalysisAgent instance.
        experiment_logger: ExperimentLogger for tracking runs.
        routing_table: RoutingTable for publishing optimal cells.
        questions: list of eval question dicts.
        population: optional initial population.
    """

    def __init__(
        self,
        model_cache: Any,
        model_keys: Optional[list[str]] = None,
        mutation_agent: Any = None,
        evaluation_agent: Any = None,
        analysis_agent: Any = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        routing_table: Optional[RoutingTable] = None,
        questions: Optional[list[dict]] = None,
        population: Optional[list[Cell]] = None,
    ):
        self.model_cache = model_cache
        self.model_keys = model_keys or []

        # Lazy import to avoid circular deps at module level
        from agent.mutation_agent import MutationAgent
        from agent.evaluation_agent import EvaluationAgent
        from agent.analysis_agent import AnalysisAgent

        self.mutation_agent = mutation_agent or MutationAgent(model_keys=self.model_keys)
        self.evaluation_agent = evaluation_agent or EvaluationAgent(self.model_cache)
        self.analysis_agent = analysis_agent or AnalysisAgent()

        self.experiment_logger = experiment_logger
        self.routing_table = routing_table or RoutingTable()

        self.questions = questions or []
        self.population = population or []
        self.tags: list[dict] = []
        self.history: list[dict] = []
        self.generation = 0
        self._start_time = time.time()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def seed_generation_0(self, known_good: Optional[dict[str, list[str]]] = None,
                          cells_per_task: int = 3):
        """Create and optionally evaluate the initial population."""
        self.population = self.mutation_agent.seed_generation_0(
            known_good_prompts=known_good,
            model_keys=self.model_keys,
            cells_per_task=cells_per_task,
        )
        logger.info("Generation 0 seeded: %d cells", len(self.population))
        if self.experiment_logger:
            self.experiment_logger.log_decision("seed_generation_0", {
                "population_size": len(self.population),
                "model_keys": self.model_keys,
            })

    def load_population(self, json_path: str):
        """Load a population from a JSON file."""
        self.population = population_from_json(json_path)
        logger.info("Loaded %d cells from %s", len(self.population), json_path)

    def save_population(self, json_path: str):
        """Save the current population to a JSON file."""
        population_to_json(self.population, json_path)
        logger.info("Saved %d cells to %s", len(self.population), json_path)

    # ── Generation Loop ─────────────────────────────────────────────────────

    def run_generation(self) -> dict:
        """Execute one full generation cycle.

        Returns a summary dict with generation metrics.
        """
        if not self.population:
            raise RuntimeError("No population — call seed_generation_0() or load_population() first")

        gen = self.generation
        logger.info("─" * 60)
        logger.info("Generation %d — evaluating %d cells", gen, len(self.population))

        # 1. Evaluate
        eval_start = time.time()
        self.population = self.evaluation_agent.evaluate(self.population, self.questions)
        eval_time = time.time() - eval_start
        logger.info("Evaluation complete in %.1fs", eval_time)

        # 2. Deduplicate
        before = len(self.population)
        self.population = deduplicate_population(self.population)
        if len(self.population) < before:
            logger.info("Dedup removed %d duplicate cells", before - len(self.population))

        # 3. Compute Pareto fronts (per-task, across models)
        fronts = self._compute_pareto_fronts()

        # 4. Generation metrics
        metrics = self._generation_metrics(fronts)

        # 5. Analysis
        self.tags = self.analysis_agent.analyze(self.population, self.tags)
        if self.tags:
            logger.info("Analysis tags: %s", ", ".join(t["tag"] for t in self.tags))
            if self.experiment_logger:
                for t in self.tags:
                    self.experiment_logger.log_decision("analysis_tag", t)

        # 6. Log generation
        if self.experiment_logger:
            self.experiment_logger.log_generation(
                gen=gen,
                population=self.population,
                pareto_fronts={k: list(v) for k, v in fronts.items()},
                metrics=metrics,
                extra={"tags": self.tags, "eval_time_s": round(eval_time, 1)},
            )

        # 7. Record history
        record = {
            "generation": gen,
            "population_size": len(self.population),
            "eval_time_s": round(eval_time, 1),
            "metrics": metrics,
            "tags": [t["tag"] for t in self.tags],
            "pareto_front_sizes": {k: len(v) for k, v in fronts.items()},
        }
        self.history.append(record)

        # 8. Evolve next population (unless this is the final gen)
        if gen < 99:  # safety cap
            self.population = self.mutation_agent.evolve(
                self.population,
                tags=[t["tag"] for t in self.tags],
                target_size=max(10, len(self.population)),
            )
            logger.info("Next generation: %d cells", len(self.population))

        self.generation += 1
        return record

    def run_generations(self, n: int = 3):
        """Run N generations in sequence."""
        logger.info("=" * 60)
        logger.info("GEPA Orchestrator: running %d generations", n)
        logger.info("=" * 60)

        for _ in range(n):
            record = self.run_generation()
            # Convergence check
            if len(self.history) >= 2:
                prev = self.history[-2]["metrics"].get("top_accuracy", 0)
                curr = record["metrics"].get("top_accuracy", 0)
                delta = abs(curr - prev) * 100  # percentage points
                if delta < 5.0 and len(self.history) >= 3:
                    logger.info("Converged (delta=%.2fpp, %d generations) — stopping",
                                delta, len(self.history))
                    break

        logger.info("=" * 60)
        logger.info("Completed %d generations", self.generation)
        logger.info("=" * 60)

    # ── Routing Table ───────────────────────────────────────────────────────

    def publish_routing_table(self, backtest_ok: bool = True) -> int:
        """Update the routing table from the Pareto-optimal cells.

        Returns:
            New routing table version, or -1 if no cells available.
        """
        if not self.population:
            logger.warning("No population — cannot publish routing table")
            return -1

        version = self.routing_table.update_from_cells(
            self.population, backtest_results=None, strict=False
        )
        if version > 0 and self.experiment_logger:
            table_dict = self.routing_table.to_dict()
            self.experiment_logger.log_routing_table(
                version=version,
                table=table_dict["entries"],
                backtest_ok=backtest_ok,
                note=f"After generation {self.generation - 1}",
            )
            self.experiment_logger.add_tag(f"routing_v{version}")
            logger.info("Routing table v%d published (%d entries)",
                        version, len(table_dict["entries"]))

        return version

    def get_routing_entries(self) -> dict:
        """Return a dict of {category: entry} for easy integration with Pipeline."""
        entries = {}
        for cat, entry in self.routing_table._entries.items():
            entries[cat] = {
                "model_key": entry.get("model_key", ""),
                "system_prompt": entry.get("system_prompt", ""),
                "decoding": entry.get("decoding", {}),
                "aggregation": entry.get("aggregation", "single"),
                "accuracy": entry.get("accuracy", 0.0),
            }
        return entries

    # ── Reporting ───────────────────────────────────────────────────────────

    def report(self, verbose: bool = True):
        """Print a summary of the run to stderr."""
        elapsed = time.time() - self._start_time
        header = f" GEPA Run: {self.generation} generations in {elapsed:.0f}s "
        logger.info("=" * len(header))
        logger.info(header)

        if self.history:
            best_record = max(self.history, key=lambda r: r["metrics"].get("top_accuracy", 0))
            logger.info("Best generation: %d (top acc=%.3f)",
                        best_record["generation"],
                        best_record["metrics"].get("top_accuracy", 0))

        for record in self.history:
            tags_str = ", ".join(record.get("tags", [])) or "-"
            logger.info("  Gen %d: %d cells, top=%.3f, tags=[%s]",
                        record["generation"],
                        record["population_size"],
                        record["metrics"].get("top_accuracy", 0),
                        tags_str)

        if self.experiment_logger:
            logger.info("Experiment log: %s", self.experiment_logger.summary())

        if self.routing_table.version > 0:
            logger.info("Routing table: v%d, %d entries",
                        self.routing_table.version,
                        len(self.routing_table.categories))
            for cat in self.routing_table.categories:
                entry = self.routing_table.get(cat)
                if entry:
                    logger.info("  %s → %s (%.3f acc)",
                                cat, entry.get("cell_name", "?"),
                                entry.get("accuracy", 0.0))

    def save_final(self, output_dir: str = "gepa_logs"):
        """Save the final state: routing table, population, results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Routing table
        rt_path = os.path.join(output_dir, "routing_table.json")
        self.routing_table.to_json(rt_path)

        # Population
        pop_path = os.path.join(output_dir, "final_population.json")
        self.save_population(pop_path)

        # Summary
        summary = {
            "num_generations": self.generation,
            "elapsed_s": round(time.time() - self._start_time, 1),
            "history": self.history,
            "routing_table_version": self.routing_table.version,
            "population_size": len(self.population),
        }
        summary_path = os.path.join(output_dir, "orchestrator_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if self.experiment_logger:
            self.experiment_logger.save_final(summary)
            self.experiment_logger.add_tag("run_complete")

        logger.info("Final state saved to %s/", output_dir)

    # ── Private ─────────────────────────────────────────────────────────────

    def _compute_pareto_fronts(self) -> dict[str, list[int]]:
        """Compute Pareto fronts per (task_id, model_key) combo.

        Returns dict mapping "{task_id}:{model_key}" → list of cell indices on front.
        """
        if not self.population:
            return {}

        # Group cells
        groups: dict[str, list[tuple[int, Cell]]] = {}
        for idx, c in enumerate(self.population):
            key = f"{c.task_id}:{c.model_key}"
            groups.setdefault(key, []).append((idx, c))

        fronts: dict[str, list[int]] = {}
        for key, items in groups.items():
            indices = [it[0] for it in items]
            cells = [it[1] for it in items]
            # Mark Pareto rank in metadata
            front_idx = _pareto_front_indices(cells)
            for i in front_idx:
                self.population[indices[i]].metadata["pareto_rank"] = 0
            for i, idx in enumerate(indices):
                if i not in front_idx:
                    self.population[idx].metadata["pareto_rank"] = 1
            fronts[key] = [indices[i] for i in front_idx]

        return fronts

    def _generation_metrics(self, fronts: dict[str, list[int]]) -> dict:
        """Compute summary metrics for the current generation."""
        accs = [c.metadata.get("accuracy", 0.0) for c in self.population if c.metadata]
        tokens = [c.metadata.get("avg_output_tokens", 0) for c in self.population if c.metadata]
        latency = [c.metadata.get("avg_latency_ms", 0.0) for c in self.population if c.metadata]

        # Top accuracy per task
        by_task: dict[str, list[float]] = {}
        for c in self.population:
            by_task.setdefault(c.task_id, []).append(c.metadata.get("accuracy", 0.0))

        return {
            "top_accuracy": max(accs) if accs else 0.0,
            "mean_accuracy": sum(accs) / len(accs) if accs else 0.0,
            "median_tokens": sorted(tokens)[len(tokens) // 2] if tokens else 0,
            "mean_latency_ms": sum(latency) / len(latency) if latency else 0.0,
            "format_compliance": sum(
                c.metadata.get("format_compliance", 0.0) for c in self.population if c.metadata
            ) / len(self.population) if self.population else 0.0,
            "pareto_front_total": sum(len(v) for v in fronts.values()),
            "per_task_top": {
                tid: max(accs) for tid, accs in by_task.items()
            },
            "frontier_size": sum(len(v) for v in fronts.values()),
        }
