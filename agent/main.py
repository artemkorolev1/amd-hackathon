"""
Filtered Pipeline v9 — pre_filter bypass → category_classifier → complexity_scorer → decision_table → solvers → QC gate.

Flow for each task:
  1. Pre-Filter (T0/T1): bypass trivial prompts, route clear-code directly
  2. Category Classifier: 8-way deterministic classifier (85% accuracy on 60-set)
  3. Complexity Scorer: per-category complexity (0.0–1.0)
  4. Decision Table: deterministic bypass if simple + covered
  5. Deterministic solvers (math, logic, sentiment, NER, factual, code_debug)
  6. If not answered → local Qwen2.5-1.5B consensus voting
  7. QC gate: reject degenerate/hedging/too-short answers
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.config import (
    CONSENSUS_SAMPLES, CONSENSUS_THRESHOLDS,
    DEGRADE_50, DEGRADE_70, DEGRADE_85,
    LLAMA_ENABLE, LLAMA_SERVER_URL, LOCAL_MODEL_PATH,
    MAX_RUNTIME_SEC, TASK_COUNT,
    COMPLEXITY_THRESHOLDS,
)
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment,
    solve_factual_qa, solve_code_debugging, solve_summarization,
    solve_math_word_problems,
)
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3
solve_ner = solve_ner_v3  # use prototype_ner_v3 as the NER deterministic solver
from agent.dynamic_prompts import build_system_prompt, build_merged_prompt, get_max_tokens, NER_ONE_SHOT_EXAMPLE
from agent.solvers.local_vote import solve_with_consensus
from agent.solvers.verify import verify as qc_verify, format_and_lint
from agent.pre_filter import pre_filter
from agent.category_filter import classify as classify_category
from agent.complexity_filter import score as score_complexity
from agent.secondary_summarization import resolve_summarization
from agent.quality_config import QC_CONFIG, GLOBAL_QC, QC_POLICY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("agent")

# Categories that have deterministic solvers — simple tasks are handled without any API
DETERMINISTIC_CATEGORIES = {"math", "logic", "sentiment", "ner", "factual", "code_debug", "summarization"}

# Categories where pipeline degrades the model — skip deterministic solvers
# and system prompts. Send prompt naked to local model.
NAKED_CATEGORIES: set[str] = set()

# Map classifier short names to deterministic solver categories
DET_CATEGORY_MAP = {
    "math": "math_arithmetic",
    "logic": "logical_reasoning",
    "sentiment": "sentiment",
    "ner": "named_entity_recognition",
    "factual": "other_complex",
    "code_debug": "code_debugging",
    "code_gen": "code_debugging",
    "summarization": "summarization",
    "general": "other_complex",
}


async def _wait_for_llama_server(url: str, timeout: int = 55, interval: float = 0.5) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    logger.info(f"llama.cpp ready in {time.monotonic() - start:.1f}s")
                    return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


def _read_tasks() -> List[Tuple[str, str]]:
    try:
        task_count = int(os.environ.get("TASK_COUNT", "0")) or TASK_COUNT
    except (ValueError, TypeError):
        task_count = TASK_COUNT
    input_path = "/input/tasks.json"
    if os.path.exists(input_path):
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
            tasks = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    tid = item.get("task_id", f"idx_{i}")
                    prompt = item.get("prompt", str(item))
                    tasks.append((tid, prompt))
                elif isinstance(item, str):
                    tasks.append((f"idx_{i}", item))
                else:
                    tasks.append((f"idx_{i}", str(item)))
            logger.info(f"Read {len(tasks)} tasks from {input_path}")
            return tasks[:task_count]
        except Exception as e:
            logger.warning(f"Failed to read {input_path}: {e}")
    tasks = []
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if line:
            tasks.append((f"idx_{i}", line))
        if len(tasks) >= task_count:
            break
    logger.info(f"Read {len(tasks)} tasks from stdin")
    return tasks[:task_count]


def _run_pipeline(prompt: str) -> Tuple[str, str, float, bool, dict]:
    """Run the full filtered pipeline. Returns (answer, category, complexity, used_api, scores)."""
    
    # ── Pre-Filter (T0/T1) ──
    s0 = pre_filter(prompt)
    if s0.action == "bypass":
        if s0.direct_answer:
            logger.info(f"  T0 bypass: {s0.direct_answer[:40]}")
            return s0.direct_answer, s0.category or "general", 0.0, False, {}
        # No direct answer from T0 — fall through to classifier/complexity/decision which
        # also run deterministic solvers (avoid redundant double-run).
        return "", s0.category or "general", 0.0, False, {}
    
    # ── Category Classifier (8-way) ──
    category, confidence, scores = classify_category(prompt)
    logger.info(f"  Classifier category={category} (conf={confidence:.2f})")
    
    # ── Complexity Scorer ──
    complexity = score_complexity(prompt, category)
    logger.info(f"  Complexity ({category}): {complexity:.3f}")
    
    # ── Decision Table ──
    if complexity < COMPLEXITY_THRESHOLDS["simple_max"] and category in DETERMINISTIC_CATEGORIES and category not in NAKED_CATEGORIES:
        logger.info(f"  Decision: DETERMINISTIC (complexity={complexity:.2f} < {COMPLEXITY_THRESHOLDS['simple_max']})")
        det_cat = DET_CATEGORY_MAP.get(category, "other_complex")
        for solve_fn in (solve_math_word_problems, solve_logic, solve_sentiment, solve_ner,
                         solve_factual_qa, solve_code_debugging, solve_summarization):
            try:
                ans = solve_fn(prompt, det_cat)
                if ans:
                    logger.info(f"  Deterministic solver: {ans[:60]}")
                    return ans, category, complexity, False, scores
            except Exception:
                pass
        return "", category, complexity, False, scores
    
    # ── API path: needs LLM ──
    logger.info(f"  Decision: API (complexity={complexity:.2f} or category={category})")
    return None, category, complexity, True, scores


async def _run_pipeline_impl(llm: Any) -> None:
    raw_tasks = _read_tasks()
    if not raw_tasks:
        env_tasks = os.environ.get("TASKS", "")
        if env_tasks:
            data = json.loads(env_tasks)
            raw_tasks = [
                (f"idx_{i}", item) if isinstance(item, str)
                else (item.get("task_id", f"idx_{i}"), item.get("prompt", str(item)))
                for i, item in enumerate(data)
            ]
    if not raw_tasks:
        logger.error("No tasks received")
        return

    task_ids = [t[0] for t in raw_tasks]
    prompts = [t[1] for t in raw_tasks]
    n = len(prompts)

    if LLAMA_ENABLE:
        logger.info("Local inference enabled (direct Python binding — no server needed)")
    else:
        logger.info("Local inference disabled — only deterministic solvers available")

    answers: Dict[str, str] = {}

    def _flush(answers: Dict[str, str]) -> None:
        ordered = [{"task_id": tid, "answer": answers.get(tid, "")} for tid in task_ids]
        tmp = "/output/results.json.tmp"
        final = "/output/results.json"
        try:
            os.makedirs("/output", exist_ok=True)
            payload = json.dumps(ordered)
            with open(tmp, "w") as f:
                f.write(payload)
            os.replace(tmp, final)
        except Exception as e:
            logger.warning(f"flush failed: {e}")

    # Read DEADLINE_S from grader env var (with safe fallback)
    try:
        deadline_s = int(os.environ.get("DEADLINE_S", str(MAX_RUNTIME_SEC)))
    except (ValueError, TypeError):
        logger.warning("DEADLINE_S invalid, using default %ds", MAX_RUNTIME_SEC)
        deadline_s = MAX_RUNTIME_SEC
    deadline = time.monotonic() + deadline_s

    for idx, prompt in enumerate(prompts):
        tid = task_ids[idx]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("Deadline reached — stopping")
            break

        logger.info(f"[{idx+1}/{n}] {tid}: {prompt[:80]}...")

        # ── Run filtered pipeline ──
        t0 = time.time()
        ans, category, complexity, needs_api, scores = _run_pipeline(prompt)

        # ── QC gate on deterministic answer (only if non-empty) ──
        if ans is not None and ans.strip():
            try:
                qc_result = qc_verify(ans, category=category)
                if qc_result and not qc_result.passed:
                    logger.warning(f"  QC FAILED ({qc_result.reason}): {ans[:60]}")
                    # QC failed — treat as unanswered, let the local LLM handle it
                    needs_api = True
                    ans = None
                else:
                    logger.info(f"  QC passed — answer: {ans[:60]}")
                    # Run code quality validation for code categories
                    if category in ("code_gen", "code_debug"):
                        code_val = format_and_lint(ans)
                        if code_val.get("formatted"):
                            logger.info(f"  Black format OK — {len(code_val.get('lint_errors', []))} lint issues")
                        if code_val.get("lint_errors"):
                            logger.info(f"  Lint issues ({len(code_val['lint_errors'])}): {code_val['lint_errors'][:2]}")
            except Exception as e:
                logger.warning(f"QC error: {e}")
        elif ans is not None and not ans.strip():
            # Deterministic solver returned empty — skip QC, go straight to local LLM
            needs_api = True
            ans = None

        # ── Local LLM path ──
        if needs_api or ans is None:
            logger.info(f"  → Local LLM (category={category}, complexity={complexity:.2f})")
            
            # Naked categories: skip system prompts entirely, go straight to local LLM
            if category in NAKED_CATEGORIES:
                sys_prompt = ""
                logger.info(f"  NAKED mode for {category} — no system prompt")
            else:
                # Build dynamic system prompt with complexity awareness
                # Use merged prompt if classifier top-2 scores are close (high uncertainty)
                s2_scores = scores or {}
                sorted_scores = sorted(s2_scores.items(), key=lambda x: -x[1])
                top_cat = sorted_scores[0][0] if sorted_scores else category
                second_cat = sorted_scores[1][0] if len(sorted_scores) > 1 else ""
                top_score = sorted_scores[0][1] if sorted_scores else 0
                second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                use_merged = (top_score - second_score) < 1.0 and second_score > 0

                ner_example = NER_ONE_SHOT_EXAMPLE if category == "ner" else None
                if use_merged:
                    sys_prompt = build_merged_prompt(
                        primary_category=top_cat,
                        secondary_category=second_cat,
                        complexity_score=complexity,
                        custom_instructions=ner_example or "",
                    )
                else:
                    sys_prompt = build_system_prompt(
                        category=category,
                        complexity_score=complexity,
                        custom_instructions=ner_example,
                    )
            
            # Local model inference (skip NER — local LLMs proven 0% F1)
            if not ans and LLAMA_ENABLE and category != "ner":
                try:
                    result = solve_with_consensus(
                        llm=llm,
                        prompt=prompt,
                        category=category,
                        system_prompt=sys_prompt,
                        k=CONSENSUS_SAMPLES,
                        max_tokens=get_max_tokens(category, complexity),
                    )
                    if result and result.get("majority_answer"):
                        ans = result["majority_answer"]
                        logger.info(f"  Local consensus: {ans[:80]} (agreement={result.get('agreement_score',0):.2f})")
                    else:
                        logger.warning("  Local consensus returned empty")
                except Exception as e:
                    logger.warning(f"  Local model error: {e}")
                    ans = ""

            # ── Code quality retry: if code_gen/code_debug has lint issues, retry ──
            if ans and category in ("code_gen", "code_debug"):
                max_code_retries = 2
                for retry_i in range(max_code_retries):
                    code_val = format_and_lint(ans)
                    if not code_val.get("lint_errors"):
                        if code_val.get("error"):
                            logger.warning(f"  Code validation error (ignored): {code_val['error']}")
                        break
                    # Has lint errors — retry with feedback
                    lint_summary = "; ".join(code_val["lint_errors"][:5])
                    logger.warning(f"  Code retry {retry_i+1}/{max_code_retries}: {len(code_val['lint_errors'])} lint issues")
                    retry_prompt = (
                        f"{prompt}\n\n"
                        f"--\nYour previous answer had these lint issues:\n{lint_summary}\n"
                        f"Please fix them and provide correct Python code."
                    )
                    # Try local retry
                    if LLAMA_ENABLE:
                        try:
                            result = solve_with_consensus(
                                llm=llm,
                                prompt=retry_prompt,
                                category=category,
                                system_prompt=sys_prompt,
                                k=CONSENSUS_SAMPLES,
                                max_tokens=get_max_tokens(category, complexity),
                            )
                            if result and result.get("majority_answer"):
                                ans = result["majority_answer"]
                                logger.info(f"  Local retry result: {ans[:60]}")
                                continue
                        except Exception:
                            pass
                    break  # No more retry options

            if not ans:
                ans = ""
                logger.warning(f"  All answers empty")

        answers[tid] = ans or ""
        _flush(answers)

    _flush(answers)
    # Final stdout output for grader harness: one answer per line
    for tid in task_ids:
        print(answers.get(tid, ""))
    logger.info(f"Completed {len(answers)}/{n} tasks")


async def main() -> None:
    # ── Load local GGUF model ──
    llm = None
    if LLAMA_ENABLE:
        # Wait for llama.cpp server (if configured — not needed for direct Python binding)
        if LLAMA_SERVER_URL:
            logger.info("Waiting for llama.cpp server...")
            ready = await _wait_for_llama_server(LLAMA_SERVER_URL)
            if not ready:
                logger.warning("llama.cpp server not ready — continuing without local inference")
        # Load the model directly via llama-cpp-python
        if os.path.isfile(LOCAL_MODEL_PATH):
            try:
                from llama_cpp import Llama
                logger.info("Loading local model: %s", os.path.basename(LOCAL_MODEL_PATH))
                llm = Llama(
                    model_path=LOCAL_MODEL_PATH,
                    n_ctx=int(os.environ.get("N_CTX", "2048")),
                    n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),
                    n_threads=int(os.environ.get("N_THREADS", "2")),
                    flash_attn=True,
                    verbose=False,
                )
                logger.info("Local model loaded successfully")
            except Exception as e:
                logger.warning("Failed to load local model: %s — continuing without local LLM", e)
        else:
            logger.warning("Model not found at %s — continuing without local LLM", LOCAL_MODEL_PATH)
    else:
        logger.info("Local inference disabled — only deterministic solvers available")

    await _run_pipeline_impl(llm)

    # Flush stdout and stderr
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    asyncio.run(main())
