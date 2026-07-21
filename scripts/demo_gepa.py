#!/usr/bin/env python3
"""GEPA Demo — run a lightweight GEPA generation cycle.

This script demonstrates the full agentic GEPA architecture:
 1. Seed generation 0 from known-good prompts (5 task types × 2 models)
 2. Evaluate cells against the dev set (skip if no local model)
 3. Run analysis/diagnostics
 4. Publish routing table
 5. Show wiring into the production Pipeline

Without a GPU or loaded GGUF model, the evaluation step is a no-op
(demo mode). Set --eval to run actual inference (requires model files).

Usage:
    python scripts/demo_gepa.py                          # no-eval demo
    python scripts/demo_gepa.py --eval                    # full eval loop
    python scripts/demo_gepa.py --gen 2 --out my_run      # custom params
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Add project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("demo_gepa")


def load_training_data(path: str = "data/eval/training-v3.json") -> list[dict]:
    """Load the training eval data."""
    path = os.path.join(_PROJECT_ROOT, path)
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded %d training questions from %s", len(data), path)
    return data


def build_model_cache(eval_mode: bool = False):
    """Build a ModelCache (lazy-loads models only when needed)."""
    from agent.gepa_runner import ModelCache

    cache = ModelCache()
    if not eval_mode:
        logger.info("Demo mode — model cache created but empty (no eval)")
    else:
        logger.info("Eval mode — models will be loaded on first use")
    return cache


def main():
    parser = argparse.ArgumentParser(description="GEPA agentic architecture demo")
    parser.add_argument("--eval", action="store_true",
                        help="Run actual model inference (requires GGUF files)")
    parser.add_argument("--gen", type=int, default=1, help="Number of generations (default: 1)")
    parser.add_argument("--out", type=str, default="gepa_logs/demo_run",
                        help="Output directory for experiment logs")
    parser.add_argument("--model-keys", type=str, nargs="+",
                        default=["qwen2.5-1.5b", "smollm2-1.7b"],
                        help="Model keys to use")
    args = parser.parse_args()

    # ── 1. Setup ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GEPA Agentic Architecture Demo")
    logger.info("=" * 60)

    questions = load_training_data()
    model_cache = build_model_cache(args.eval)

    if args.eval:
        # Verify model files exist
        from agent.gepa_runner import MODEL_PATHS
        missing = [k for k in args.model_keys if k not in MODEL_PATHS]
        if missing:
            logger.warning("Model keys not in MODEL_PATHS: %s", missing)

    # Filter questions by task type for demo purposes
    task_map = {"T01": "factual", "T02": "math", "T03": "sentiment",
                "T04": "summarization", "T05": "ner"}
    demo_questions = [q for q in questions if q.get("category") in task_map.values()]
    logger.info("Questions for 5-task GEPA: %d (from %d total)",
                len(demo_questions), len(questions))

    # ── 2. Create Agents ────────────────────────────────────────────────
    from agent import (
        MutationAgent, EvaluationAgent, AnalysisAgent,
        ExperimentLogger, RoutingTable, GEPAOrchestrator, create_run,
    )

    mutation_agent = MutationAgent(model_keys=args.model_keys, seed=42)
    evaluation_agent = EvaluationAgent(model_cache)
    analysis_agent = AnalysisAgent()
    experiment_logger = create_run(args.out, tags=["demo", "5-task"])
    routing_table = RoutingTable()

    orchestrator = GEPAOrchestrator(
        model_cache=model_cache,
        model_keys=args.model_keys,
        mutation_agent=mutation_agent,
        evaluation_agent=evaluation_agent,
        analysis_agent=analysis_agent,
        experiment_logger=experiment_logger,
        routing_table=routing_table,
        questions=demo_questions,
    )

    # ── 3. Seed Generation 0 ───────────────────────────────────────────
    logger.info("\n─── Seeding Generation 0 ───")
    orchestrator.seed_generation_0(cells_per_task=2)
    logger.info("Population: %d cells", len(orchestrator.population))

    # Show cell distribution by task and model
    from collections import Counter
    task_dist = Counter(c.task_id for c in orchestrator.population)
    model_dist = Counter(c.model_key for c in orchestrator.population)
    logger.info("By task: %s", dict(task_dist))
    logger.info("By model: %s", dict(model_dist))

    # Show some sample cells
    logger.info("Sample cells:")
    for c in orchestrator.population[:5]:
        logger.info("  [%s] %s — %s: '%s'",
                    c.task_id, c.model_key, c.name, c.system_prompt[:60])

    # ── 4. Run Generations ──────────────────────────────────────────────
    if args.gen > 0:
        logger.info("\n─── Running %d generation(s) ───", args.gen)
        if not args.eval:
            logger.info("(DEMO MODE — no actual inference; cells get zero accuracy scores)")
            # Populate mock metadata for demo purposes
            for i, c in enumerate(orchestrator.population):
                c.metadata["accuracy"] = max(0.1, min(0.6, (i % 5) * 0.15))
                c.metadata["avg_output_tokens"] = 40 + (i % 7) * 10
                c.metadata["avg_latency_ms"] = 3000 + (i % 4) * 500
                c.metadata["format_compliance"] = 0.7 + (i % 3) * 0.1
                c.metadata["correct"] = int(c.metadata["accuracy"] * 19)
                c.metadata["total"] = 19
                c.metadata["category"] = c.pipeline_category

            # Log the mock generation
            orchestrator.experiment_logger.log_generation(
                gen=0, population=orchestrator.population,
                pareto_fronts={},
                metrics={"top_accuracy": 0.6, "mean_accuracy": 0.35},
                extra={"tags": [], "eval_time_s": 0.0},
            )

        orchestrator.run_generations(n=args.gen)

    # ── 5. Publish Routing Table ────────────────────────────────────────
    logger.info("\n─── Publishing Routing Table ───")
    version = orchestrator.publish_routing_table()
    entries: dict = {}
    if version > 0:
        entries = orchestrator.get_routing_entries()
        logger.info("Routing table v%d:", version)
        for cat, entry in entries.items():
            logger.info("  %s → model=%s, acc=%.3f, prompt='%s'",
                        cat, entry.get("model_key", "?"),
                        entry.get("accuracy", 0.0),
                        entry.get("system_prompt", "")[:50])

    # ── 6. Wire into Pipeline ───────────────────────────────────────────
    logger.info("\n─── Wiring into Production Pipeline ───")
    from agent import Pipeline

    pipe = Pipeline(routing_table=entries)
    logger.info("Pipeline created with %d routing entries", len(entries))

    # Show pipeline category mapping
    for tid, cat in [("T01", "factual"), ("T02", "math"), ("T03", "sentiment"),
                     ("T04", "summarization"), ("T05", "ner")]:
        entry = entries.get(cat)
        if entry:
            logger.info("  Task %s (%s) → '%s' (%.3f acc)",
                        tid, cat, entry.get("system_prompt", "?")[:40],
                        entry.get("accuracy", 0.0))

    # ── 7. Report ──────────────────────────────────────────────────────
    logger.info("\n─── Final Report ───")
    orchestrator.report()

    # ── 8. Save final state ─────────────────────────────────────────────
    orchestrator.save_final(args.out)
    logger.info("\n─── Demo Complete ───")
    logger.info("Experiment logs: %s", experiment_logger.run_dir)
    logger.info("Routing table: %s/routing_table.json", args.out)
    logger.info("Final population: %s/final_population.json", args.out)
    logger.info("Summary: %s/orchestrator_summary.json", args.out)

    # Compact summary line
    print(f"\n[GEPA DEMO] Run: {experiment_logger.run_dir.name}")
    print(f"  Cells: {len(orchestrator.population)}")
    print(f"  Generations: {orchestrator.generation}")
    print(f"  Routing entries: {len(entries)}")
    print(f"  Routing table version: {version}")


if __name__ == "__main__":
    main()
