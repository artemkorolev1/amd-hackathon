#!/usr/bin/env python3
"""Multi-model runner — runs one eval set through multiple GGUF models.

Each model answers ALL questions independently. Results are logged to a single
Excel file with a "Model" column so you can compare answers side-by-side.

Usage:
    # Run models from ~/models/ directory
    python3 multi_runner.py --eval input/two_questions.json --model-dir ~/models/

    # Run specific models (comma-separated paths)
    python3 multi_runner.py --eval input/two_questions.json \\
        --models ~/models/qwen2.5-1.5b.q4,~/models/gemma-3-1b-it.q4

    # GPU mode (default: all layers)
    python3 multi_runner.py --gpu --eval input/two_questions.json --model-dir ~/models/

    # CPU mode
    python3 multi_runner.py --cpu --eval input/two_questions.json --model-dir ~/models/
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("multi-runner")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Pipeline imports (shared across models)
from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
from agent.complexity import score as mlm_complexity
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment,
    solve_ner, solve_factual_qa, solve_code_debugging,
)
from agent.dynamic_prompts import (
    build_system_prompt, build_merged_prompt,
    get_max_tokens as dp_max_tokens,
    get_stop_sequences as dp_stop_sequences,
)
from agent.run_logger import RunLogger

# ── Regexes from harness.py ────────────────────────────────────────────────
_HARD_MATH_RE = re.compile(
    r"\b(law of sines|law of cosines|geometric series|cofactor|determinant"
    r"|inclusion.exclusion|bayes(?:ian)?|conditional probability"
    r"|permutations?|combinations?|integral|derivative|matrix|eigenvalu"
    r"|logarithm|log base|\bmod\b|modular arithmetic|chinese remainder"
    r"|ratio|proportion|in the ratio|lcm|gcd|least common multiple"
    r"|greatest common divisor|rate.*time|time.*rate|work rate"
    r"|how many (?:different |distinct |possible )?ways"
    r"|distinct.*\bdigits?|distinct.*\bnumbers?"
    r"|nCr|nPr)\b", re.IGNORECASE,
)
_MULTI_PERSON_RE = re.compile(
    r"\b(friends?|colleagues?|neighbors?|students?|candidates?|people)\b"
    r".{0,40}\b(each|all)\b", re.IGNORECASE,
)
_RATIO_RE = re.compile(r"\b\d+\s*:\s*\d+\b")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_DOC_HEADER_RE = re.compile(
    r"^(HEADLINE:|LEGAL BRIEF|STATEMENT BY|ARTICLE:|TRANSCRIPT:|MEMO:|PRESS RELEASE:)",
    re.IGNORECASE | re.MULTILINE,
)


def is_hard_math(prompt: str) -> bool:
    if _HARD_MATH_RE.search(prompt):
        return True
    if len(re.findall(r"\\[a-zA-Z]+", prompt)) >= 2:
        return True
    if _RATIO_RE.search(prompt):
        return True
    return False


def is_hard_logic(prompt: str, category: str) -> bool:
    if category != "logic":
        return False
    return (len(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)) >= 4
            or bool(_MULTI_PERSON_RE.search(prompt)))


def strip_think(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    if stripped != text.strip():
        return stripped
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def strip_kimi_preamble(text: str) -> str:
    idx = text.find("```python")
    if idx == -1:
        idx = text.find("```")
    if idx > 0:
        return text[idx:]
    return text


def secondary_category(raw_scores: dict, primary: str) -> str:
    sorted_cats = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    for cat, _ in sorted_cats:
        if cat != primary:
            return cat
    return ""


def _get_prompt_version(category: str, score_delta: float,
                         is_reasoning: bool, is_merged: bool,
                         secondary_cat: str, complexity: float) -> str:
    if is_reasoning:
        return f"reasoning/{category}"
    if is_merged and secondary_cat:
        return f"merged/{category}+{secondary_cat}"
    return f"standard/{category}/cx={complexity:.2f}"


# ── Model discovery ────────────────────────────────────────────────────────

def find_gguf_files(directory: str) -> list[dict]:
    """Scan a directory for .gguf files, return list of {path, name, size_gb}."""
    models = []
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        logger.warning("Model directory %s does not exist", path)
        return models
    for f in sorted(path.glob("*.gguf")):
        size_gb = f.stat().st_size / (1024**3)
        models.append({
            "path": str(f),
            "name": f.name,
            "size_gb": round(size_gb, 1),
        })
    return models


# ── Per-model runner ───────────────────────────────────────────────────────

def run_model(model_cfg: dict, questions: list[dict],
              n_gpu_layers: int, n_ctx: int, n_threads: int,
              logger_instance: RunLogger):
    """Load one model, run all questions, log results, unload."""
    from llama_cpp import Llama

    model_path = model_cfg["path"]
    model_name = model_cfg["name"]
    logger.warning("Loading %s  (%.1f GB, n_gpu_layers=%d)",
                   model_name, model_cfg["size_gb"], n_gpu_layers)

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        flash_attn=True,
        verbose=False,
    )
    logger.warning("Model %s ready", model_name)

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def infer(messages, max_tok, stop_seq, timeout=60):
        def _call():
            return llm.create_chat_completion(
                messages=messages, max_tokens=max_tok,
                temperature=0.0, stop=stop_seq,
            )
        try:
            future = executor.submit(_call)
            resp = future.result(timeout=timeout)
            raw = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage", {})
            return strip_think(raw), usage
        except concurrent.futures.TimeoutError:
            logger.warning("[%s] inference timed out", model_name)
            return "", {}
        except Exception as exc:
            logger.warning("[%s] inference error: %s", model_name, exc)
            return "", {}

    for i, q in enumerate(questions):
        tid = q.get("task_id", f"idx_{i}")
        prompt = q.get("prompt", q.get("question", ""))
        t_start = time.monotonic()

        logger_instance.start_question(tid, prompt, model_name=model_name,
                                         difficulty=q.get("difficulty", ""))

        # Stage 0 — pre-filter
        t0 = time.monotonic()
        s0 = stage0(prompt)
        s0_ms = (time.monotonic() - t0) * 1000
        logger_instance.log_pre_filter(
            action=s0.action, answer=s0.direct_answer or "",
            category_hint=s0.category or "", flags=str(s0.flags),
            elapsed_ms=s0_ms,
        )
        if s0.action == "bypass" and s0.direct_answer:
            logger_instance.finish_question(s0.direct_answer, solver_ms=0)
            print(s0.direct_answer.replace("\n", "\\n"))
            continue

        # Stage 2 — category filter
        t0 = time.monotonic()
        detail = _stage2_detail(prompt)
        s2_ms = (time.monotonic() - t0) * 1000
        category = detail["category"]
        score_delta = detail["score_delta"]
        raw_scores = detail["raw_scores"]

        # Keyword overrides
        lower = prompt.lower()
        overrides = []
        if category not in ("summarization",) and _DOC_HEADER_RE.search(prompt):
            category = "summarization"; score_delta = 1.0
            overrides.append("doc_header→summarization")
        if category != "summarization" and re.search(r"\bsummariz[ei]", lower):
            category = "summarization"; score_delta = 1.0
            overrides.append("summarize_keyword→summarization")
        if category == "math" and len(prompt) > 600 and not re.search(
            r'[=×÷]|\\frac|\\int|\\sum|\bsolve\b|\bcalculate\b|\bcompute\b|\bfind\b.*\b(?:value|sum|product|ratio)\b', prompt, re.I):
            category = "summarization"; score_delta = 1.0
            overrides.append("long_math→summarization")
        if category not in ("ner",) and re.search(
            r"\b(extract|identify|list)\b.{0,40}\b(named entity|entities|people mentioned|organizations mentioned)\b", lower):
            category = "ner"; score_delta = 1.0
            overrides.append("extract_keyword→ner")
        if category == "sentiment" and re.search(
            r"\bfor (someone|a person)\b.{0,50}\b(with your|of your)\b", prompt, re.I):
            logger_instance.log_post_processing("backhanded_compliment→negative")
            logger_instance.finish_question("negative", solver_ms=0)
            print("negative")
            continue

        logger_instance.log_category_filter(
            category=category, category_4way=detail.get("category_4way", ""),
            confidence=detail.get("confidence", 0.0),
            score_delta=score_delta, raw_scores=raw_scores,
            overrides="; ".join(overrides), elapsed_ms=s2_ms,
        )
        if overrides and logger_instance._current:
            logger_instance._current.keyword_overrides_applied = "; ".join(overrides)

        # Complexity
        t0 = time.monotonic()
        complexity = mlm_complexity(prompt)
        cx_ms = (time.monotonic() - t0) * 1000
        logger_instance.log_complexity(complexity, "MiniLM-L6-v2+LogReg", cx_ms)

        # Build prompt
        mc_options = re.findall(r"(?<!\w)[a-dA-D]\)\s", prompt)
        is_mc = len(set(o.strip().lower() for o in mc_options)) >= 3
        is_reasoning, is_merged, secondary_cat = False, False, ""
        max_tok = int(dp_max_tokens(category, complexity))
        stop_seq = dp_stop_sequences(category)

        sys_prompt = build_system_prompt(category, complexity)
        t_decision = time.monotonic()
        logger_instance.log_decision(
            solver_name="local_llm", model=model_name,
            max_tokens=max_tok, temperature=0.0,
            system_prompt=sys_prompt[:300],
            prompt_version=_get_prompt_version(category, score_delta,
                                                is_reasoning, is_merged,
                                                secondary_cat, complexity),
            is_merged=is_merged, is_reasoning_prompt=is_reasoning,
            elapsed_ms=(time.monotonic() - t_decision) * 1000,
        )

        # Deterministic solvers
        det_cat_map = {
            "math": "math_arithmetic", "sentiment": "sentiment",
            "factual": "other_complex", "code_debug": "code_debugging",
        }
        det_answer = None
        if category in det_cat_map:
            for solver_fn in [solve_arithmetic, solve_logic, solve_sentiment,
                              solve_ner, solve_factual_qa, solve_code_debugging]:
                try:
                    a = solver_fn(prompt, det_cat_map[category])
                    if a:
                        det_answer = a
                        break
                except Exception:
                    pass
        if det_answer:
            logger_instance.log_deterministic(
                solver_results=[("det", det_answer)], hint=det_answer, elapsed_ms=0)
            logger_instance.finish_question(det_answer, solver_ms=0)
            print(det_answer.replace("\n", "\\n"))
            continue

        # Local LLM
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}]
        t0 = time.monotonic()
        answer, usage = infer(messages, max_tok, stop_seq)
        llm_ms = (time.monotonic() - t0) * 1000

        retry = False
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        total_tok = usage.get("total_tokens", 0)
        if not answer:
            t0 = time.monotonic()
            answer2, usage2 = infer([{"role": "user", "content": prompt}],
                                    max_tok, stop_seq)
            llm_ms += (time.monotonic() - t0) * 1000
            if answer2:
                answer = answer2
                prompt_tok += usage2.get("prompt_tokens", 0)
                comp_tok += usage2.get("completion_tokens", 0)
                total_tok += usage2.get("total_tokens", 0)
            retry = True

        logger_instance.log_local_llm(
            elapsed_ms=llm_ms, retry=retry,
            prompt_tokens=prompt_tok, completion_tokens=comp_tok,
            total_tokens=total_tok,
        )

        # Post-processing
        post_proc = ""
        if category == "math" and answer:
            m = re.search(r"\bAnswer:\s*(.+)", answer, re.I | re.DOTALL)
            if m:
                answer = m.group(1).strip().split("\n")[0].strip()
                post_proc = "math_answer_extract"
        if category == "logic" and answer:
            m = re.search(r"\bAnswer:\s*(.+)", answer, re.I | re.DOTALL)
            if m:
                answer = m.group(1).strip()
                post_proc = "logic_answer_extract"
        logger_instance.log_post_processing(post_proc)

        logger_instance.finish_question(answer or "", solver_ms=llm_ms)
        print(answer.replace("\n", "\\n"))

    # Unload model
    del llm
    import gc
    gc.collect()
    logger.warning("Unloaded %s", model_name)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-model runner")
    parser.add_argument("--eval", required=True, help="Eval JSON file")
    parser.add_argument("--model-dir", default="~/models/",
                        help="Directory containing .gguf files (default: ~/models/)")
    parser.add_argument("--models", help="Comma-separated model paths (overrides --model-dir)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="All layers on GPU (default)")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU only (zero layers)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window")
    parser.add_argument("--n-threads", type=int, default=2, help="Threads")
    args = parser.parse_args()

    n_gpu = 0 if args.cpu else -1

    # Discover models
    if args.models:
        models = []
        for p in args.models.split(","):
            p = p.strip()
            if os.path.exists(p):
                models.append({
                    "path": os.path.abspath(p),
                    "name": os.path.basename(p),
                    "size_gb": round(os.path.getsize(p) / (1024**3), 1),
                })
    else:
        models = find_gguf_files(args.model_dir)

    if not models:
        logger.error("No GGUF models found")
        sys.exit(1)

    logger.warning("Found %d models:", len(models))
    for m in models:
        logger.warning("  [%.1f GB] %s", m["size_gb"], m["name"])

    # Load eval questions
    with open(args.eval) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    logger.warning("Loaded %d questions", len(questions))

    # Git version for run log
    import subprocess
    try:
        pipeline_version = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, cwd=_HERE, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        pipeline_version = "unknown"

    # Create run logger (one logger for all models — produces one Excel)
    run_logger = RunLogger(
        run_number=None,
        pipeline_version=pipeline_version,
        model_path="multi: " + ", ".join(m["name"] for m in models),
        fireworks_model="(disabled for multi-runner)",
        fireworks_key_set=False,
        n_gpu_layers=n_gpu,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        num_questions=len(questions) * len(models),
        eval_source=os.path.basename(args.eval),
    )

    # Run each model through all questions
    for model_cfg in models:
        run_model(
            model_cfg, questions,
            n_gpu_layers=n_gpu,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            logger_instance=run_logger,
        )

    # Write Excel
    xlsx_dir = os.path.join(_HERE, "eval_results")
    try:
        fpath = run_logger.write_xlsx(xlsx_dir)
        logger.warning("Run log written to %s", fpath)
    except Exception as exc:
        logger.warning("Failed to write run log: %s", exc)

    logger.warning("Done — %d questions × %d models = %d total rows",
                   len(questions), len(models),
                   len(questions) * len(models))


if __name__ == "__main__":
    main()
