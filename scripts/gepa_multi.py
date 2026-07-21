#!/usr/bin/env python3
"""
GEPA Multi-Category Runner — runs Pareto-based prompt evolution for ALL task categories.

Extends the GEPA pipeline beyond the original factual-only (T01) to cover
all 8 categories: code_debug, code_gen, factual, logic, math, ner, sentiment,
summarization.

Usage:
    python scripts/gepa_multi.py                                          # all 8 cats, 1 gen each
    python scripts/gepa_multi.py --categories sentiment,math               # specific cats
    python scripts/gepa_multi.py --categories sentiment --generations 3    # 3 gen on 1 cat
    python scripts/gepa_multi.py --categories sentiment --questions 5      # subset of questions
    python scripts/gepa_multi.py --models qwen2.5-1.5b                     # single model
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent dir so agent/ is importable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── GEPA imports ──────────────────────────────────────────────────────────
from agent.gepa_category_runner import (
    run_gepa,
    POPULATION_SIZE,
    NUM_GENERATIONS as DEFAULT_GENERATIONS,
    RESULTS_DIR,
)
from agent.experiment_logger import ExperimentLogger
from agent.routing_table import RoutingTable
from agent.cell import Cell, DecodingConfig, TASK_LABELS, TASK_TO_PIPELINE_CAT, PIPELINE_CAT_TO_TASK

# ── Constants ─────────────────────────────────────────────────────────────
ALL_CATEGORIES = ["code_debug", "code_gen", "factual", "logic", "math", "ner", "sentiment", "summarization"]
DEFAULT_MODELS = ["qwen2.5-1.5b", "qwen2.5-coder-1.5b"]
ROUTING_TABLE_PATH = ROOT / "gepa_plans" / "multi_gepa_routing_table.json"


def build_cell_from_variant(category: str, model_key: str, variant: dict, generation: int) -> Cell:
    """Convert a GEPA variant dict into a Cell object suitable for routing."""
    task_id = PIPELINE_CAT_TO_TASK.get(category, f"T{ALL_CATEGORIES.index(category)+1:02d}")
    return Cell(
        task_id=task_id,
        model_key=model_key,
        system_prompt=variant.get("system_prompt", ""),
        decoding=DecodingConfig(
            temperature=variant.get("temperature", 0.0),
            max_tokens=variant.get("max_tokens", 128),
        ),
        aggregation="single",
        name=f"{category}_best_gen{generation}_{model_key}",
        generation=generation,
        metadata={
            "category": category,
            "accuracy": variant.get("accuracy", {}).get(model_key, 0.0),
            "avg_output_tokens": variant.get("avg_output_tokens", {}).get(model_key, 0),
            "avg_latency_ms": variant.get("avg_latency_ms", {}).get(model_key, 0.0),
        },
    )


def update_routing_table(
    table: RoutingTable,
    all_results: dict[str, dict],
    model_keys: list[str],
    generation: int,
) -> RoutingTable:
    """Update routing table with best cells from each category and model."""
    entries = []
    for category in ALL_CATEGORIES:
        if category not in all_results:
            continue
        result = all_results[category]
        if not result or "final_results" not in result:
            continue

        for mk in model_keys:
            fin = result["final_results"].get(mk)
            if not fin:
                continue

            # Clamp category string to pipeline-recognised values
            cat_key = category if category in TASK_TO_PIPELINE_CAT.values() else category

            entry = {
                "category": cat_key,
                "cell_name": fin.get("variant_name", f"{category}_best_{mk}"),
                "model_key": mk,
                "system_prompt": fin.get("best_prompt", ""),
                "decoding": {
                    "temperature": fin.get("temperature", 0.0),
                    "max_tokens": fin.get("max_tokens", 128),
                },
                "aggregation": "single",
                "accuracy": fin.get("best_accuracy", 0.0),
                "avg_output_tokens": fin.get("avg_output_tokens", 0),
                "avg_latency_ms": fin.get("avg_latency_ms", 0.0),
                "generation": generation,
                "updated_at": time.time(),
            }
            entries.append(entry)

    # Publish entries to routing table
    if entries:
        new_version = table._publish(entries)
        print(f"\n  [routing] Updated routing table → version {new_version} ({len(entries)} entries)")
    else:
        print("\n  [routing] No entries to publish.")

    return table


def main():
    parser = argparse.ArgumentParser(
        description="GEPA Multi-Category Prompt Evolution"
    )
    parser.add_argument(
        "--categories",
        default=",".join(ALL_CATEGORIES),
        help=f"Comma-separated categories (default: all 8: {','.join(ALL_CATEGORIES)})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model keys to use (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--generations", "--gen", type=int, default=1,
        help="Number of GEPA generations per category (default: 1)",
    )
    parser.add_argument(
        "--questions", type=int, default=None,
        help="Number of questions to use per category (subset; default: all)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Experiment run name (default: auto-generated)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip categories that already have results saved",
    )

    args = parser.parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Validate categories
    for cat in categories:
        if cat not in ALL_CATEGORIES:
            print(f"ERROR: Unknown category '{cat}'. Valid: {ALL_CATEGORIES}")
            sys.exit(1)

    print("=" * 70)
    print("GEPA MULTI-CATEGORY EVOLUTION")
    print("=" * 70)
    print(f"  Categories:   {categories}")
    print(f"  Models:       {args.models}")
    print(f"  Generations:  {args.generations}")
    print(f"  Questions:    {args.questions or 'all (19 per cat in training-v3.json)'}")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    # ── Setup experiment logger ───────────────────────────────────────────
    run_name = args.run_name or f"multi_gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(
        run_dir=str(ROOT / "gepa_logs"),
        run_name=run_name,
        metadata={
            "categories": categories,
            "models": args.models,
            "generations": args.generations,
            "num_questions": args.questions,
            "type": "multi_category_gepa",
        },
    )
    logger.add_tag("multi-category")
    logger.add_tag(f"gens-{args.generations}")

    # ── Setup routing table ───────────────────────────────────────────────
    routing_table = RoutingTable()
    # Load existing routing table if available
    if ROUTING_TABLE_PATH.exists():
        try:
            routing_table = RoutingTable.from_json(str(ROUTING_TABLE_PATH))
            print(f"  Loaded existing routing table (version {routing_table.version})")
        except Exception as e:
            print(f"  Could not load existing routing table: {e}")

    # ── Run GEPA for each category ────────────────────────────────────────
    all_results = {}
    for idx, category in enumerate(categories):
        print(f"\n{'#' * 70}")
        print(f"# CATEGORY {idx+1}/{len(categories)}: {category.upper()}")
        print(f"{'#' * 70}")

        # Check if results already exist
        results_path = Path(RESULTS_DIR) / f"{category}_gepa_results.json"
        if args.skip_existing and results_path.exists():
            print(f"  Results already exist at {results_path}, skipping.")
            try:
                with open(results_path) as f:
                    all_results[category] = json.load(f)
            except Exception:
                pass
            continue

        # Run GEPA
        start = time.time()
        try:
            result = run_gepa(
                category=category,
                model_keys=args.models,
                num_generations=args.generations,
                num_questions=args.questions,
            )
            elapsed = time.time() - start
            print(f"\n  Category '{category}' completed in {elapsed:.1f}s")

            if result:
                all_results[category] = result
            else:
                print(f"  WARNING: run_gepa returned None for '{category}'")

        except Exception as e:
            print(f"\n  ERROR on category '{category}': {e}")
            import traceback
            traceback.print_exc()
            continue

        # ── Log to experiment logger ──────────────────────────────────────
        if category in all_results:
            cat_result = all_results[category]
            gen_data = cat_result.get("generations", [])
            for g in gen_data:
                logger.log_generation(
                    gen=g.get("generation", 0),
                    population=g.get("population", []),
                    pareto_fronts=g.get("pareto_front_sizes", {}),
                    metrics={
                        "category": category,
                        "per_model_best": g.get("per_model_best", {}),
                    },
                    extra={"category": category},
                )

            # Log routing decision
            logger.log_decision("routing_update", {
                "category": category,
                "models": args.models,
                "final_results": {
                    mk: fin.get("best_accuracy", 0.0)
                    for mk, fin in cat_result.get("final_results", {}).items()
                },
            })

    # ── Update routing table with Pareto-optimal cells ────────────────────
    print(f"\n{'=' * 70}")
    print("UPDATING ROUTING TABLE")
    print(f"{'=' * 70}")

    update_routing_table(routing_table, all_results, args.models, args.generations)

    # Save routing table
    ROUTING_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    routing_table.to_json(str(ROUTING_TABLE_PATH))
    print(f"  Routing table saved to {ROUTING_TABLE_PATH}")

    # ── Log routing table to experiment logger ────────────────────────────
    logger.log_routing_table(
        version=routing_table.version,
        table=routing_table.to_dict(),
        backtest_ok=True,
        note=f"Multi-category GEPA completed for {len(categories)} categories",
    )

    # ── Save final results ────────────────────────────────────────────────
    final_summary = {
        "num_categories_completed": len(all_results),
        "categories_attempted": categories,
        "categories_completed": list(all_results.keys()),
        "models": args.models,
        "generations": args.generations,
        "routing_table_version": routing_table.version,
        "per_category_summary": {},
    }

    for cat in categories:
        if cat in all_results:
            per_model = {}
            for mk in args.models:
                fin = all_results[cat].get("final_results", {}).get(mk, {})
                per_model[mk] = {
                    "best_accuracy": fin.get("best_accuracy", 0.0),
                    "best_prompt": fin.get("best_prompt", ""),
                    "temperature": fin.get("temperature", 0.0),
                }
            final_summary["per_category_summary"][cat] = per_model

    logger.save_final(final_summary)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Logger:     {logger.summary()}")
    print(f"  Routing:    version {routing_table.version}, saved to {ROUTING_TABLE_PATH}")
    print()

    for cat in categories:
        if cat in all_results:
            cat_result = all_results[cat]
            print(f"  {cat}:")
            for mk in args.models:
                fin = cat_result.get("final_results", {}).get(mk, {})
                acc = fin.get("best_accuracy", 0.0)
                prompt_preview = fin.get("best_prompt", "")[:80]
                print(f"    {mk}: accuracy={acc:.3f}  prompt=\"{prompt_preview}...\"")
        else:
            print(f"  {cat}: FAILED or skipped")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
