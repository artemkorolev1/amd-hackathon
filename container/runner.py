"""
Runner — orchestrates the full container pipeline.

Flow:
  1. Load prompts_config.json
  2. Start llama-server with 4 parallel slots
  3. For each question:
     a. Determine category (from input or classify)
     b. Load 4 system prompts (strategies A-D) for that category
     c. Send all 4 to parallel slots → collect answers
     d. Consensus: vote → judge → quality gate → fallback
     e. Log final answer
  4. Write results.json
  5. Shutdown server
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from container.server import ServerManager
from container.inference import parallel_infer, simple_infer
from container.consensus import merge_answers, is_degenerate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("container.runner")

# ── Paths ───────────────────────────────────────────────────────────────────

PROMPTS_CONFIG = os.path.join(_HERE, "prompts_config.json")
OUTPUT_DIR = os.path.join(_HERE, "output")


def load_prompts() -> dict:
    """Load prompt config JSON."""
    with open(PROMPTS_CONFIG) as f:
        return json.load(f)


def get_category_prompts(
    prompts: dict,
    category: str,
) -> tuple[list[str], dict]:
    """
    Return (list_of_4_system_prompts, per_category_params).
    Falls back to 'defaults' if category not found.
    """
    strategies = ["A", "B", "C", "D"]
    cat_prompts = prompts["system_prompts"].get(category, prompts["defaults"])
    params = prompts["per_category_params"].get(
        category,
        {"max_tokens": 150, "temperature": 0.0, "stop": []},
    )
    system_prompts = []
    for s in strategies:
        sp = cat_prompts.get(s, prompts["defaults"].get(s, ""))
        # Append anti-preamble
        sp += (" Start with the answer directly — "
               "no greeting, no 'I will', no meta-commentary.")
        system_prompts.append(sp)
    return system_prompts, params


def load_input(input_path: str) -> list[dict]:
    """Load eval JSON (array of {task_id, prompt, ...})."""
    with open(input_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", data.get("prompts", list(data.values())) or [])
    if isinstance(data, list):
        return data
    return []


def run(
    model_path: str,
    input_path: str,
    output_dir: str = OUTPUT_DIR,
    n_ctx: int = 2048,
    n_threads: int = 2,
    n_parallel: int = 4,
    timeout_s: int = 600,
):
    """Main entry point."""
    os.makedirs(output_dir, exist_ok=True)
    prompts = load_prompts()
    questions = load_input(input_path)
    judge_template = prompts["judge_prompt"]

    logger.info("Loaded %d questions, %d prompt strategies",
                len(questions), n_parallel)

    results = []
    start_time = time.monotonic()

    with ServerManager(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parallel=n_parallel,
        n_threads=n_threads,
    ) as server:
        if not server.is_healthy():
            logger.error("Server not healthy — aborting")
            sys.exit(1)

        for i, q in enumerate(questions):
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_s:
                logger.warning("Timeout (%.0fs) — finishing early at question %d/%d",
                               elapsed, i, len(questions))
                break

            tid = q.get("task_id", f"q_{i:03d}")
            prompt_text = q.get("prompt", q.get("question", ""))
            category = q.get("category", "factual")

            logger.info("[%s/%d] %s (cat=%s) — inferring 4 variants...",
                        tid, len(questions), prompt_text[:60], category)

            sys_prompts, params = get_category_prompts(prompts, category)

            # Phase 1: Parallel inference (4 slots)
            t0 = time.monotonic()
            answers = parallel_infer(
                system_prompts=sys_prompts,
                user_prompt=prompt_text,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                stop=params.get("stop"),
            )
            infer_ms = (time.monotonic() - t0) * 1000

            # Phase 2: Consensus + judge
            def _judge_call(text: str) -> str:
                return simple_infer(
                    system_prompt=(
                        "You are a strict answer judge. "
                        "Evaluate the candidates and output the best one."
                    ),
                    user_prompt=text,
                    max_tokens=100,
                    temperature=0.0,
                    stop=["\n\n"],
                )

            merged = merge_answers(
                question=prompt_text,
                answers=answers,
                judge_template=judge_template,
                call_judge_fn=_judge_call,
            )

            result = {
                "task_id": tid,
                "prompt": prompt_text,
                "category": category,
                "final_answer": merged["answer"],
                "method": merged["method"],
                "confidence": merged["confidence"],
                "judge_reason": merged.get("judge_reason", ""),
                "inference_ms": int(infer_ms),
                "raw_answers": merged["raw_answers"],
                "degenerate_flags": merged["degenerate"],
            }
            results.append(result)

            logger.info("  → method=%s conf=%.2f answer=%s",
                        merged["method"], merged["confidence"],
                        merged["answer"][:80])

            # Fireworks fallback for degenerate answers
            if merged["method"] in ("degenerate", "fallback") and merged["confidence"] < 0.2:
                logger.warning("  ⚠ Degenerate answer — would trigger Fireworks fallback here")

    # Write output
    output_path = os.path.join(output_dir, "results.json")
    output_data = {
        "meta": {
            "model": model_path,
            "input": input_path,
            "n_questions": len(questions),
            "n_results": len(results),
            "elapsed_s": round(time.monotonic() - start_time, 1),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Done — %d/%d questions answered in %.0fs",
                len(results), len(questions),
                time.monotonic() - start_time)
    logger.info("Output: %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-prompt ensemble runner for AMD ACT II container"
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--input", required=True, help="Input JSON (tasks)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max runtime in seconds")
    args = parser.parse_args()

    run(
        model_path=args.model,
        input_path=args.input,
        output_dir=args.output_dir,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        timeout_s=args.timeout,
    )
