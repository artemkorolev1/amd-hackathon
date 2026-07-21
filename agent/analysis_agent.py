#!/usr/bin/env python3
"""Analysis / diagnostics agent — interprets failures and tags patterns.

After each evaluation cycle, this agent inspects which cells failed
on which questions and produces structured tags that the mutation
agent can use to bias the next generation.

Tags are simple strings like:
    "verbose"          — cell produces overly long outputs
    "imprecise"        — cell has low accuracy despite good format
    "format_skip"      — cell skips format requirements
    "hard_task:math"   — math questions are systemically harder
    "model_weak:smollm2-1.7b"  — a model underperforms on this task
    "seed_improvement" — random seed produced better result
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional

from agent.cell import Cell, TASK_LABELS


# ── Analysis Agent ──────────────────────────────────────────────────────────

class AnalysisAgent:
    """Diagnoses evaluation results and returns actionable tags.

    Uses only the metadata already stored on cells — no extra LLM calls.
    """

    def analyze(
        self,
        population: list[Cell],
        previous_tags: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Analyze a population of evaluated cells.

        Args:
            population: list of Cell objects with metadata filled by EvaluationAgent.
            previous_tags: tags from previous generation (for trend detection).

        Returns:
            List of tag dicts: {"tag": str, "confidence": float, "detail": str}
        """
        tags: list[dict] = []

        if not population:
            return tags

        # 1. Per-model performance
        by_model: dict[str, list[Cell]] = defaultdict(list)
        for c in population:
            by_model[c.model_key].append(c)

        for model_key, cells in by_model.items():
            accs = [c.metadata.get("accuracy", 0.0) for c in cells]
            if not accs:
                continue
            mean_acc = sum(accs) / len(accs)
            if mean_acc < 0.3:
                tags.append({
                    "tag": f"model_weak:{model_key}",
                    "confidence": round(1.0 - mean_acc, 2),
                    "detail": f"{model_key} mean accuracy {mean_acc:.2f} across {len(cells)} cells",
                })

        # 2. Per-task difficulty
        by_task: dict[str, list[Cell]] = defaultdict(list)
        for c in population:
            by_task[c.task_id].append(c)

        for tid, cells in by_task.items():
            accs = [c.metadata.get("accuracy", 0.0) for c in cells]
            if not accs:
                continue
            mean_acc = sum(accs) / len(accs)
            if mean_acc < 0.4:
                label = TASK_LABELS.get(tid, tid)
                tags.append({
                    "tag": f"hard_task:{label}",
                    "confidence": round(1.0 - mean_acc, 2),
                    "detail": f"Task {tid} ({label}) mean accuracy {mean_acc:.2f} across {len(cells)} cells",
                })

        # 3. Verbosity analysis
        for c in population:
            tok = c.metadata.get("avg_output_tokens", 0)
            acc = c.metadata.get("accuracy", 0.0)
            if tok > 100 and acc < 0.5:
                tags.append({
                    "tag": "verbose",
                    "confidence": min(1.0, tok / 200),
                    "detail": f"Cell {c.name}: {tok} tokens/task at {acc:.2f} accuracy",
                })
                break  # one tag per category per analysis

        # 4. Format compliance issues
        for c in population:
            fmt = c.metadata.get("format_compliance", 1.0)
            acc = c.metadata.get("accuracy", 0.0)
            if fmt < 0.6 and acc < 0.5:
                tags.append({
                    "tag": "format_skip",
                    "confidence": round(1.0 - fmt, 2),
                    "detail": f"Cell {c.name}: format compliance {fmt:.2f}, accuracy {acc:.2f}",
                })
                break

        # 5. Check for improvement from random variants
        rand_cells = [c for c in population if "fresh" in c.name or "rand" in c.name]
        if rand_cells:
            best_rand = max(rand_cells, key=lambda c: c.metadata.get("accuracy", 0.0))
            best_overall = max(population, key=lambda c: c.metadata.get("accuracy", 0.0))
            if best_rand.metadata.get("accuracy", 0.0) > 0.4:
                tags.append({
                    "tag": "seed_improvement",
                    "confidence": 0.6,
                    "detail": f"Random variant '{best_rand.name}' achieved "
                              f"{best_rand.metadata.get('accuracy', 0.0):.2f} accuracy",
                })

        # 6. Diversity check
        unique_prompts = len(set(c.system_prompt for c in population))
        if len(population) > 0 and unique_prompts <= len(population) * 0.3:
            tags.append({
                "tag": "low_diversity",
                "confidence": round(1.0 - unique_prompts / len(population), 2),
                "detail": f"{unique_prompts} unique prompts out of {len(population)} cells",
            })

        # 7. Pareto front diversity (if front info in metadata)
        pareto_entries = [c for c in population if c.metadata.get("pareto_rank", -1) == 0]
        if pareto_entries:
            models_on_front = len(set(c.model_key for c in pareto_entries))
            tasks_on_front = len(set(c.task_id for c in pareto_entries))
            if models_on_front <= 1:
                tags.append({
                    "tag": "pareto_single_model",
                    "confidence": 0.7,
                    "detail": f"Pareto front has only {models_on_front} model(s) "
                              f"covering {tasks_on_front} task(s)",
                })

        return tags

    def summarize_failures(self, population: list[Cell], top_n: int = 5) -> list[dict]:
        """Return the hardest questions (by failure rate) across all cells.

        Requires cells to have 'details' in metadata.
        """
        # Collect question-level stats
        question_stats: dict[str, dict[str, Any]] = {}

        for c in population:
            details = c.metadata.get("details")
            if not details or not isinstance(details, list):
                continue
            for d in details:
                qid = d.get("task_id", d.get("question", "?"))
                if qid not in question_stats:
                    question_stats[qid] = {
                        "question": d.get("question", qid)[:80],
                        "total": 0,
                        "correct": 0,
                        "cells_tried": set(),
                    }
                qs = question_stats[qid]
                qs["total"] += 1
                qs["cells_tried"].add(c.name)
                if d.get("correct"):
                    qs["correct"] += 1

        # Convert to sorted list
        failures = []
        for qid, qs in question_stats.items():
            fail_rate = 1.0 - (qs["correct"] / qs["total"] if qs["total"] > 0 else 1.0)
            failures.append({
                "task_id": qid,
                "question": qs["question"],
                "failure_rate": round(fail_rate, 2),
                "total_attempts": qs["total"],
                "correct_attempts": qs["correct"],
                "cells_tried": len(qs["cells_tried"]),
            })

        failures.sort(key=lambda f: f["failure_rate"], reverse=True)
        return failures[:top_n]
