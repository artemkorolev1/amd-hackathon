"""
Filtered Pipeline v9 — T0/T1 bypass → S2 (8-way 85%) → S3 (complexity) → S4 (decision) → solvers → QC gate.

Flow for each task:
  1. Stage 0 (T0/T1): bypass trivial prompts, route clear-code directly
  2. Stage 2: 8-way category classifier (85% accuracy on 60-set)
  3. Stage 3: per-category complexity (0.0–1.0)
  4. Stage 4: decision table — deterministic bypass if simple + covered
  5. Deterministic solvers (math, logic, sentiment, NER, factual, code_debug)
  6. If not answered → local Qwen2.5-1.5B consensus voting
  7. If still not answered → Fireworks API
  8. QC gate: reject degenerate/hedging/too-short answers
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from agent.config import (
    CONSENSUS_SAMPLES, CONSENSUS_THRESHOLDS,
    DEGRADE_50, DEGRADE_70, DEGRADE_85,
    LLAMA_ENABLE, LLAMA_SERVER_URL,
    MAX_RUNTIME_SEC, REMOTE_CIRCUIT_BREAKER_LIMIT,
    REMOTE_CIRCUIT_RETRY_AFTER, TASK_COUNT,
    COMPLEXITY_THRESHOLDS,
    resolve_model,
)
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
    solve_factual_qa, solve_code_debugging,
)
from agent.dynamic_prompts import build_system_prompt, build_merged_prompt, get_max_tokens, NER_ONE_SHOT_EXAMPLE
from agent.solvers.fireworks import FireworksSolver, REASONING_EFFORT_NONE_TASK_TYPES
from agent.solvers.local_vote import solve_with_consensus
from agent.solvers.verify import verify as qc_verify
from agent.pre_filter import stage0
from agent.category_filter import classify as stage2_classify
from agent.complexity_filter import score as stage3_complexity
from agent.quality_config import QC_CONFIG, GLOBAL_QC, QC_POLICY
from agent.circuit_breaker import RemoteBreaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("agent")

# Categories that have deterministic solvers — simple tasks are handled without any API
DETERMINISTIC_CATEGORIES = {"math", "logic", "sentiment", "ner", "factual", "code_debug"}

# Categories where pipeline degrades the model — skip deterministic solvers,
# system prompts, and Fireworks. Send prompt naked to local model.
NAKED_CATEGORIES = {"ner", "summarization", "factual", "logic", "math"}

# Categories that escalate to Fireworks API instead of local model
FIREWORKS_CATEGORIES = {"sentiment"}

# Map S2 short names to deterministic solver categories
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
    
    # ── Stage 0: T0/T1 pre-filter ──
    s0 = stage0(prompt)
    if s0.action == "bypass":
        if s0.direct_answer:
            logger.info(f"  T0 bypass: {s0.direct_answer[:40]}")
            return s0.direct_answer, s0.category or "general", 0.0, False, {}
        # Route to deterministic solver with hint category
        det_cat = DET_CATEGORY_MAP.get(s0.category or "general", "other_complex")
        for solve_fn in (solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
                         solve_factual_qa, solve_code_debugging):
            try:
                ans = solve_fn(prompt, det_cat)
                if ans:
                    logger.info(f"  T0 route→solver: {ans[:60]}")
                    return ans, s0.category or "general", 0.0, False, {}
            except Exception:
                pass
        return "", s0.category or "general", 0.0, False, {}
    
    # ── Stage 2: 8-way category classifier ──
    category, confidence, scores = stage2_classify(prompt)
    logger.info(f"  S2 category={category} (conf={confidence:.2f})")
    
    # ── Stage 3: Per-category complexity ──
    complexity = stage3_complexity(prompt, category)
    logger.info(f"  S3 complexity ({category}): {complexity:.3f}")
    
    # ── Stage 4: Decision table ──
    if complexity < COMPLEXITY_THRESHOLDS["simple_max"] and category in DETERMINISTIC_CATEGORIES and category not in NAKED_CATEGORIES:
        logger.info(f"  S4 decision: DETERMINISTIC (complexity={complexity:.2f} < {COMPLEXITY_THRESHOLDS['simple_max']})")
        det_cat = DET_CATEGORY_MAP.get(category, "other_complex")
        for solve_fn in (solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
                         solve_factual_qa, solve_code_debugging):
            try:
                ans = solve_fn(prompt, det_cat)
                if ans:
                    logger.info(f"  Deterministic solver: {ans[:60]}")
                    return ans, category, complexity, False, scores
            except Exception:
                pass
        return "", category, complexity, False, scores
    
    # ── API path: needs LLM ──
    logger.info(f"  S4 decision: API (complexity={complexity:.2f} or category={category})")
    return None, category, complexity, True, scores


async def _run_pipeline_impl(fireworks: Optional[FireworksSolver]) -> None:
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
        logger.info("Local inference disabled — all non-deterministic tasks go to Fireworks")

    breaker = RemoteBreaker(
        circuit_breaker_limit=REMOTE_CIRCUIT_BREAKER_LIMIT,
        retry_after=REMOTE_CIRCUIT_RETRY_AFTER,
    )

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
                qc_result = qc_verify(ans, task=category)
                if qc_result and not qc_result.passed:
                    logger.warning(f"  QC FAILED ({qc_result.reason}): {ans[:60]}")
                    # QC failed — treat as unanswered, let the API handle it
                    needs_api = True
                    ans = None
                else:
                    logger.info(f"  QC passed — answer: {ans[:60]}")
            except Exception as e:
                logger.warning(f"QC error: {e}")
        elif ans is not None and not ans.strip():
            # Deterministic solver returned empty — skip QC, go straight to API
            needs_api = True
            ans = None

        # ── API escalation path ──
        if needs_api or ans is None:
            logger.info(f"  → API escalation (category={category}, complexity={complexity:.2f})")
            
            # Naked categories: skip system prompts entirely, go straight to local LLM
            if category in NAKED_CATEGORIES:
                sys_prompt = ""
                logger.info(f"  NAKED mode for {category} — no system prompt, no Fireworks")
            else:
                # Build dynamic system prompt with complexity awareness
                # Use merged prompt if S2 top-2 scores are close (high uncertainty)
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
            det_hint = None
            
            if fireworks and not breaker.is_open() and category in FIREWORKS_CATEGORIES:
                try:
                    # Try deterministic solvers as hints first
                    for solve_fn in (solve_arithmetic, solve_logic, solve_sentiment,
                                     solve_ner, solve_factual_qa, solve_code_debugging):
                        try:
                            hint_cat = DET_CATEGORY_MAP.get(category, "other_complex")
                            det_hint = solve_fn(prompt, hint_cat)
                            if det_hint:
                                break
                        except Exception:
                            pass
                    
                    ans = await asyncio.to_thread(
                        fireworks.solve,
                        model=resolve_model(complexity),
                        user_prompt=prompt,
                        system_prompt=sys_prompt,
                        max_tokens=get_max_tokens(category, complexity),
                        task_type=category,
                        det_hint=det_hint,
                    )
                    if ans:
                        logger.info(f"  Fireworks: {ans[:80]}")
                    else:
                        logger.warning("  Fireworks returned empty — falling back to local")
                except Exception as e:
                    logger.warning(f"  Fireworks error: {e} — falling back")
                    breaker.record_failure(str(e))
                    ans = None
            
            # Local model fallback
            if not ans and LLAMA_ENABLE:
                try:
                    result = solve_with_consensus(
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
    # Wait for llama.cpp (if server-based — not needed for direct Python binding)
    if LLAMA_ENABLE and LLAMA_SERVER_URL:
        logger.info("Waiting for llama.cpp server...")
        ready = await _wait_for_llama_server(LLAMA_SERVER_URL)
        if not ready:
            logger.warning("llama.cpp server not ready — continuing without local inference")
    
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    fireworks = FireworksSolver(api_key) if api_key else None
    
    if fireworks:
        logger.info("Fireworks solver ready")
    else:
        logger.warning("No FIREWORKS_API_KEY — local inference only where available")
    
    await _run_pipeline_impl(fireworks)

    # Flush and flush stderr
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    asyncio.run(main())
