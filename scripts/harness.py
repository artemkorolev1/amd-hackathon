#!/usr/bin/env python3
"""v14 harness — switchable local GGUF model on GPU (default: Qwen2.5-3B).

Routing: deterministic 8-way regex classifier (stage2), zero API tokens.
Cascade:
  1. Stage 0  – immediate deterministic bypass (greetings, pure arithmetic)
  2. Stage 2  – regex 8-way classifier with score-delta confidence
  3. Fireworks escalation – hard math / hard logic → API model (if key set)
  4. Deterministic solvers – math/logic/sentiment/NER/factual/code_debug
  5. Local LLM – complexity-adaptive prompt via dynamic_prompts
  6. code_gen syntax fallback – AST failure → Fireworks retry
"""

import ast
import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("v12h")

# ── Early arg parse (before model loads) ──────────────────────────────────
# Parse --gpu/--cpu so N_GPU_LAYERS env var is set before the model reads it.
if not hasattr(sys, '_gpu_parsed'):
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument('--gpu',  action='store_true', help='All layers on GPU')
    _parser.add_argument('--cpu',  action='store_true', help='CPU only (zero layers)')
    _parser.add_argument('eval_path', nargs='?', default=None)
    _early_args, _ = _parser.parse_known_args()
    if _early_args.gpu:
        os.environ['N_GPU_LAYERS'] = '-1'
        logger.warning("GPU mode: N_GPU_LAYERS=-1 (all layers on GPU)")
    elif _early_args.cpu:
        os.environ['N_GPU_LAYERS'] = '0'
        logger.warning("CPU mode: N_GPU_LAYERS=0 (zero layers offloaded)")
    sys._gpu_parsed = True
    sys._early_eval_path = _early_args.eval_path

# ── Model ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(_HERE, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))   # -1 = all layers on GPU
N_CTX        = int(os.environ.get("N_CTX",         "2048"))
N_THREADS    = int(os.environ.get("N_THREADS",     "2"))

# ── Fireworks escalation model ─────────────────────────────────────────────
# Override with FIREWORKS_MODEL env var.
# Bearer token read from FIREWORKS_API_KEY; if unset, escalation is silently skipped.
FIREWORKS_MODEL = os.environ.get(
    "FIREWORKS_MODEL",
    "accounts/fireworks/models/llama-3.1-nemotron-70b-instruct",
)

# ── Reasoning headroom ─────────────────────────────────────────────────────
# Reasoning models (Nemotron, DeepSeek-R1, QwQ) emit <think> blocks before
# the visible answer.  Nematron benchmarks at ~85 tok/s on the RTX A4000,
# giving a 30s budget of ~2550 tokens.  With 4× headroom the largest
# category (code_gen high-complexity) reaches ~1560 tokens = 18 s, leaving
# ~11 s for a Fireworks fallback if local inference still fails.
_REASONING_HEADROOM = 4 if "nemotron" in MODEL_PATH.lower() else 1

# Safety timeout per inference call — generous 60s prevents stuck model hangs.
# The grader hard-kills at 30s per question, so this should never actually fire.
INFERENCE_TIMEOUT_S = 60.0
FIREWORKS_TIMEOUT_S = 30.0

# On CPU (N_GPU_LAYERS=0) reduce token budget to avoid timeouts at
# ~12 tok/s (2-vCPU estimate). Still gives ample tokens per question.
_CPU_TOKEN_FACTOR = 0.75 if N_GPU_LAYERS == 0 else 1.0

DEFAULT_MAX_TOKENS = 150
DEFAULT_STOP: list[str] = ["\n\n"]

# ── Imports ────────────────────────────────────────────────────────────────
from agent.pre_filter import stage0
from agent.classifier import classify_with_detail as _stage2_detail
from agent.complexity import score as mlm_complexity   # MiniLM-L6-v2 + LogReg
from agent.solvers.deterministic import (
    solve_arithmetic,
    solve_logic,
    solve_sentiment,
    solve_ner,
    solve_factual_qa,
    solve_code_debugging,
    solve_code_generation,
    solve_summarization,
)
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3
from agent.solvers.prototype_zebra_v2 import solve_zebra_puzzle
from agent.solvers.logic_reasoning import solve_logical_reasoning
from agent.solvers.fireworks import FireworksSolver
from agent.solvers.fw_router import route as _fw_route
from agent.dynamic_prompts import (
    build_system_prompt,
    build_merged_prompt,
    get_max_tokens as dp_max_tokens,
    get_stop_sequences as dp_stop_sequences,
)

_fw = FireworksSolver()  # reads FIREWORKS_API_KEY from env; api_key="" if unset

# ── Run Logger (initialised by main()) ─────────────────────────────────────
from agent.run_logger import RunLogger
_run_logger = None  # RunLogger instance, initialised by main()

# ── Reasoning-model system prompts ─────────────────────────────────────────
# Standard dynamic_prompts contains "First word = the answer" / "Start with the
# answer directly."  Reasoning models (Nematron, QwQ, DeepSeek-R1) put ALL work
# inside <think> blocks FIRST, then output the answer — so the anti-preamble
# instruction creates a contradiction the model spends all its think-budget on.
# These prompts are non-contradictory: tell the model WHAT to output after
# thinking, not HOW to structure its internal reasoning.
_REASONING_PROMPTS: dict[str, str] = {
    "sentiment": (
        "Classify the sentiment of the text. Watch for sarcasm and hedging — "
        "these are NEGATIVE or NEUTRAL. After thinking, output EXACTLY one word: "
        "positive, negative, neutral, or mixed."
    ),
    "ner": (
        "Extract all named entities from the text. After thinking, output them "
        "grouped by type: CATEGORY: v1, v2; CATEGORY: v3. "
        "Labels: PERSON, ORGANIZATION, LOCATION, DATE, GENE, DISEASE, TICKER, "
        "MONETARY, PERCENTAGE, DRUG, LEGISLATION. Only include entities that "
        "explicitly appear in the text."
    ),
    "math": (
        "Solve the math problem. After thinking, output ONLY: Answer: <value>. "
        "Standard decimal format. No units unless the problem requires them."
    ),
    "logic": (
        "Solve the logic puzzle. Output ONLY the final answer — NO preamble, NO 'To solve...', "
        "NO step-by-step reasoning in your response. "
        "For assignment puzzles (floors, seats, positions, days): immediately output ALL assignments "
        "on ONE line: 'Position 1: Name (Role); Position 2: Name (Role); ...'. "
        "For option-letter questions: output ONLY the option letter and its full text. "
        "For procedural puzzles: output the sequence of steps concisely. "
        "For yes/no or single-conclusion puzzles: output ONLY: Answer: <word or short phrase>."
    ),
    "factual": (
        "Answer the question. After thinking, output the answer directly. "
        "Under 100 words. Address every sub-part. Use exact names, dates, numbers."
    ),
    "summarization": (
        "Summarize the text. After thinking, output the summary directly. "
        "Obey any length constraint stated in the prompt. If none given, max 2 sentences."
    ),
    "code_gen": (
        "Write the requested Python function. After thinking, output ONLY the "
        "function inside ```python\\n...\\n```. Preserve exact function name and "
        "signature. Handle edge cases. No explanation, no docstring."
    ),
    "code_debug": (
        "Fix the bug in the Python function. After thinking, output ONLY the "
        "corrected function inside ```python\\n...\\n```. Preserve the original "
        "function name and signature. No explanation."
    ),
}

# Fireworks task_type hint per category (controls reasoning suppression in FireworksSolver)
_FW_TASK_TYPES: dict[str, str] = {
    "code_gen":     "code",
    "code_debug":   "code",
    "math":         "math",
    "logic":        "general",
    "factual":      "general",
    "sentiment":    "sentiment",
    "ner":          "ner",
    "summarization":"summarization",
}

# Map stage2 category → solver-expected category name.
# All 8 categories now have deterministic solver entries, ordered by
# effectiveness (pipeline-context eval with fuzzy_match grading).
_DET_CAT_MAP: dict[str, str] = {
    "math":       "math_arithmetic",
    "sentiment":  "sentiment",
    "factual":    "other_complex",
    "code_debug": "code_debugging",
    "logic":      "logical_reasoning",
    "ner":        "ner",
    "code_gen":   "code_gen",
    "summarization": "summarization",
}
_DET_SOLVERS = [
    # NER: old regex (80%) before v3 (66%)
    solve_ner,
    solve_ner_v3,
    # Logic
    solve_zebra_puzzle,
    solve_logical_reasoning,
    solve_logic,
    # Rest
    solve_arithmetic,
    solve_sentiment,
    solve_factual_qa,
    solve_code_debugging,
    solve_code_generation,
    solve_summarization,
]

# ── Hard-case escalation filters ───────────────────────────────────────────
# Prompts matching these patterns reliably exceed what the local 3B can handle;
# route to a stronger model via Fireworks when a key is available.
_HARD_MATH_RE = re.compile(
    r"\b(law of sines|law of cosines|geometric series|cofactor|determinant"
    r"|inclusion.exclusion|bayes(?:ian)?|conditional probability"
    r"|permutations?|combinations?|integral|derivative|matrix|eigenvalu"
    r"|logarithm|log base|\bmod\b|modular arithmetic|chinese remainder"
    r"|ratio|proportion|in the ratio|lcm|gcd|least common multiple"
    r"|greatest common divisor|rate.*time|time.*rate|work rate"
    r"|how many (?:different |distinct |possible )?ways"
    r"|distinct.*\bdigits?|distinct.*\bnumbers?"
    r"|nCr|nPr)\b",
    re.IGNORECASE,
)
_MULTI_PERSON_RE = re.compile(
    r"\b(friends?|colleagues?|neighbors?|students?|candidates?|people)\b"
    r".{0,40}\b(each|all)\b",
    re.IGNORECASE,
)
# Ratio expression like "3:2" or "9:2" appearing in the prompt
_RATIO_RE = re.compile(r"\b\d+\s*:\s*\d+\b")


def is_hard_math(prompt: str) -> bool:
    if _HARD_MATH_RE.search(prompt):
        return True
    if len(re.findall(r"\\[a-zA-Z]+", prompt)) >= 2:
        return True
    # Explicit numeric ratio (e.g., "3:2", "9:2") → ratio/proportion problem
    if _RATIO_RE.search(prompt):
        return True
    return False


def is_hard_logic(prompt: str, category: str) -> bool:
    if category != "logic":
        return False
    return (
        len(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)) >= 4
        or bool(_MULTI_PERSON_RE.search(prompt))
    )


def _strip_kimi_preamble(text: str) -> str:
    """Remove kimi thinking preamble that appears before the first code fence.

    Kimi often outputs "We need solve... Let me design..." before ```python even
    when KIMI_KILL is in the system prompt.  If a code fence exists anywhere in
    the text, discard everything before it so only the code is returned.
    """
    idx = text.find("```python")
    if idx == -1:
        idx = text.find("```")
    if idx > 0:
        return text[idx:]
    return text


def syntax_ok(code: str) -> bool:
    """Return True if `code` contains syntactically valid Python."""
    m = re.search(r"```(?:python)?\n([\s\S]+?)\n```", code)
    src = m.group(1) if m else code
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _secondary_category(raw_scores: dict[str, float], primary: str) -> str:
    """Return the second-highest scoring category (used for merged prompts)."""
    sorted_cats = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    for cat, _ in sorted_cats:
        if cat != primary:
            return cat
    return ""


# ── Think-block stripper ───────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning blocks from model output.

    Handles two forms:
    - Standard: <think>...</think>ANSWER  (full tag pair present)
    - Nematron/llama-cpp: REASONING...</think>ANSWER  (opening tag consumed
      before the content field is populated; only closing tag visible)
    """
    stripped = _THINK_RE.sub("", text).strip()
    if stripped != text.strip():
        return stripped
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def _stop_sequences(category: str) -> list[str]:
    """Stop sequences adjusted for reasoning models.

    Reasoning models put all work inside <think> blocks, so the think content
    contains newlines, code fences, and other tokens that would normally act as
    stops.  Any stop sequence would fire mid-think and truncate the response
    before </think> and the actual answer appear.  With 4× budget headroom the
    model always finishes within the wall-clock limit, so no stops are needed.
    """
    if _REASONING_HEADROOM > 1:
        return []   # let </think> close naturally; budget handles the time limit
    # Logic answers can span multiple semicolon-separated assignments on one line,
    # or be multi-line floor/seat breakdowns — \n\n fires too early after any
    # preamble paragraph and cuts the model off before the final answer.
    if category == "logic":
        return ["Question:", "Context:"]
    return dp_stop_sequences(category)


# ── Model loader ───────────────────────────────────────────────────────────
def _load_model():
    from llama_cpp import Llama
    logger.warning("Loading %s  (n_gpu_layers=%d)", os.path.basename(MODEL_PATH), N_GPU_LAYERS)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=N_THREADS,
        flash_attn=True,
        verbose=False,
    )
    logger.warning("Model ready")
    return llm


llm = _load_model()

# Pre-warm MiniLM complexity model before the eval loop so any loading output
# is swallowed here, not mixed with per-question answers on stdout.
import contextlib, io as _io
with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
    mlm_complexity("warmup")

# ── Inference ──────────────────────────────────────────────────────────────
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _infer(
    messages: list[dict],
    max_tok: int,
    stop_seq: list[str],
    timeout: float = INFERENCE_TIMEOUT_S,
) -> tuple[str, dict]:
    """Call the local LLM. Returns (answer_text, usage_dict)."""
    def _call():
        return llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tok,
            temperature=0.0,
            stop=stop_seq,
        )

    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        future = _executor.submit(_call)
        resp   = future.result(timeout=timeout)
        raw    = resp["choices"][0]["message"]["content"] or ""
        usage  = resp.get("usage", empty_usage)
        return _strip_think(raw), usage
    except concurrent.futures.TimeoutError:
        logger.warning("Inference timed out after %.0fs — returning empty", timeout)
        return "", empty_usage
    except Exception as exc:
        logger.warning("Inference error: %s", exc)
        return "", empty_usage


# ── Main loop ──────────────────────────────────────────────────────────────

# Each question runs without per-task timeout — the grader hard-kills at 30 s.
# INFERENCE_TIMEOUT_S is a safety net for stuck model inference (not a budget).


def _process(prompt: str, task_id: str = "", difficulty: str = "") -> str:
    t0 = time.monotonic()

    if _run_logger:
        _run_logger.start_question(task_id, prompt, model_name=os.path.basename(MODEL_PATH),
                                   difficulty=difficulty)

    # 1. Stage 0 — immediate deterministic bypass (greetings, pure arithmetic)
    t_s0 = time.monotonic()
    s0 = stage0(prompt)
    t_s0 = (time.monotonic() - t_s0) * 1000
    if _run_logger:
        _run_logger.log_pre_filter(
            action=s0.action,
            answer=s0.direct_answer or "",
            category_hint=s0.category or "",
            flags=str(s0.flags),
            elapsed_ms=t_s0,
        )
    if s0.action == "bypass" and s0.direct_answer:
        if _run_logger:
            _run_logger.finish_question(s0.direct_answer, solver_ms=0)
        return s0.direct_answer

    # 2. Category classification with score delta (zero API tokens)
    t_s2 = time.monotonic()
    detail      = _stage2_detail(prompt)
    t_s2 = (time.monotonic() - t_s2) * 1000
    category    = detail["category"]
    score_delta = detail["score_delta"]
    raw_scores  = detail["raw_scores"]

    # Log stage 2 results
    overrides = ""
    orig_category = category
    if _run_logger:
        _run_logger.log_category_filter(
            category=category,
            category_4way=detail.get("category_4way", ""),
            confidence=detail.get("confidence", 0.0),
            score_delta=score_delta,
            raw_scores=raw_scores,
            overrides="",
            elapsed_ms=t_s2,
        )

    # 2b. Keyword overrides — correct common stage2 misclassifications
    _p_lower = prompt.lower()
    applied_overrides: list[str] = []

    # Document-header patterns always indicate summarization regardless of classifier
    _DOC_HEADER_RE = re.compile(
        r'^(HEADLINE:|LEGAL BRIEF|STATEMENT BY|ARTICLE:|TRANSCRIPT:|MEMO:|PRESS RELEASE:)',
        re.IGNORECASE | re.MULTILINE,
    )
    if category not in ("summarization",) and _DOC_HEADER_RE.search(prompt):
        category    = "summarization"
        score_delta = 1.0
        applied_overrides.append("doc_header→summarization")

    # Explicit "summarize" keyword or multi-source SOURCE 1/2 pattern — catches any category
    # (previously only rescued math/code; Q19 showed sentiment can misclassify "Summarize: ...")
    if category != "summarization":
        if re.search(r'\bsummariz[ei]', _p_lower) or re.search(r'source\s+[12]\b', _p_lower, re.I):
            category    = "summarization"
            score_delta = 1.0
            applied_overrides.append("summarize_keyword→summarization")

    # Long narrative text (>600 chars, no explicit equation) misclassified as math
    if category == "math" and len(prompt) > 600:
        # If no explicit computation operators in the text, it is more likely a summary task
        if not re.search(r'[=×÷]|\\frac|\\int|\\sum|\bsolve\b|\bcalculate\b|\bcompute\b|\bfind\b.*\b(?:value|sum|product|ratio)\b', prompt, re.I):
            category    = "summarization"
            score_delta = 1.0
            applied_overrides.append("long_math→summarization")

    # Prompts asking to "extract", "find", "identify" named entities but classified wrong
    if category not in ("ner",) and re.search(
        r'\b(extract|identify|list)\b.{0,40}\b(named entity|entities|people mentioned|organizations mentioned)\b',
        _p_lower
    ):
        category    = "ner"
        score_delta = 1.0
        applied_overrides.append("extract_keyword→ner")

    # Backhanded compliment detection: "for someone with your X, you did Y" → dismissive → negative
    # These sound positive on the surface but carry implicit low expectations (condescending).
    if category == "sentiment" and re.search(
        r'\bfor (someone|a person)\b.{0,50}\b(with your|of your)\b',
        prompt, re.IGNORECASE,
    ):
        if _run_logger:
            _run_logger.log_post_processing("backhanded_compliment→negative")
            _run_logger.finish_question("negative", solver_ms=0)
        return "negative"

    # Log overrides if any
    if applied_overrides and _run_logger:
        if _run_logger._current:
            _run_logger._current.keyword_overrides_applied = "; ".join(applied_overrides)

    # 3. Complexity scoring (drives prompt tier and token budget)
    t_cx = time.monotonic()
    complexity = mlm_complexity(prompt)   # MiniLM+LogReg, task-agnostic 0-1
    t_cx = (time.monotonic() - t_cx) * 1000
    if _run_logger:
        _run_logger.log_complexity(
            complexity=complexity,
            model_info="MiniLM-L6-v2+LogReg",
            elapsed_ms=t_cx,
        )

    # 4. Build system prompt
    # Detect multiple-choice questions (≥3 distinct a)/b)/c)/d) options in prompt)
    _mc_options = re.findall(r'(?<!\w)[a-dA-D]\)\s', prompt)
    _is_mc = len(set(o.strip().lower() for o in _mc_options)) >= 3

    _MC_MATH_PROMPT = (
        "Select the correct option from the choices given. "
        "Output ONLY the letter and its full text exactly as written, e.g., 'c) 4 hours'. "
        "No working. No explanation."
    )
    _MC_LOGIC_PROMPT = (
        "Select the correct option from the choices given. "
        "Output ONLY the letter and its full text exactly as written, e.g., 'B) Some logical people are not musicians.' "
        "No explanation."
    )

    prompt_version = "standard"
    is_merged = False
    is_reasoning = False
    secondary_cat = ""

    if _REASONING_HEADROOM > 1:
        sys_prompt = _REASONING_PROMPTS.get(category, _REASONING_PROMPTS["factual"])
        prompt_version = f"reasoning/{category}"
        is_reasoning = True
    elif category == "math" and _is_mc:
        sys_prompt = _MC_MATH_PROMPT
        prompt_version = f"mc_math"
    elif category == "logic" and _is_mc:
        sys_prompt = _MC_LOGIC_PROMPT
        prompt_version = f"mc_logic"
    elif category == "logic":
        sys_prompt = _REASONING_PROMPTS["logic"]
        prompt_version = "reasoning/logic"
        is_reasoning = True
    else:
        secondary = _secondary_category(raw_scores, category)
        secondary_cat = secondary or ""
        if score_delta < 0.5 and secondary:
            sys_prompt = build_merged_prompt(category, secondary, complexity)
            prompt_version = f"merged/{category}+{secondary}"
            is_merged = True
        else:
            sys_prompt = build_system_prompt(category, complexity)
            prompt_version = f"standard/{category}/cx={complexity:.2f}"

    # Token budget: complexity-adaptive × reasoning headroom × CPU factor
    # MC questions need very few tokens — cap at 80 to avoid wasting time
    if _is_mc and category in ("math", "logic"):
        max_tok = 80
    else:
        max_tok  = int(dp_max_tokens(category, complexity) * _REASONING_HEADROOM * _CPU_TOKEN_FACTOR)
    stop_seq = _stop_sequences(category)

    t_decision = time.monotonic()
    decision_elapsed = 0.0  # set after decision logging

    def _fw_fallback(reason: str) -> str:
        """Call Fireworks if key is set."""
        if not _fw.api_key:
            return ""
        try:
            logger.warning("Fireworks fallback (%s) — %s", category, reason)
            cfg = _fw_route(category, prompt, complexity)
            return _fw.solve(
                cfg.model_id, cfg.system_prompt, prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                prefill=cfg.prefill,
                task_type=_FW_TASK_TYPES.get(category, "general"),
                timeout=FIREWORKS_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("Fireworks fallback failed (%s)", exc)
            return ""

    # Log decision
    t_decision = (time.monotonic() - t_decision) * 1000
    decision_solver = "local_llm"
    decision_model = os.path.basename(MODEL_PATH)
    if _run_logger:
        _run_logger.log_decision(
            solver_name="local_llm",
            model=os.path.basename(MODEL_PATH),
            max_tokens=max_tok,
            temperature=0.0,
            reasoning_effort="",
            prefill="",
            skip_api=False,
            system_prompt=sys_prompt[:300],
            prompt_version=prompt_version,
            is_merged=is_merged,
            is_reasoning_prompt=is_reasoning,
            elapsed_ms=t_decision,
        )

    # 5. Hard-case Fireworks escalation — bypass local for reliably-hard inputs
    if _fw.api_key:
        try:
            if category == "sentiment":
                t_fw = time.monotonic()
                # Local 3B can't resolve sarcasm/irony (≤60%); gpt-oss-120b got 83% in v12d.
                # Route ALL sentiment to FW — fast + cheap (60 tokens max).
                cfg = _fw_route("sentiment", prompt, complexity)
                answer = _fw.solve(
                    cfg.model_id, cfg.system_prompt, prompt,
                    max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    prefill=cfg.prefill, task_type="sentiment",
                )
                t_fw = (time.monotonic() - t_fw) * 1000
                if _run_logger:
                    _run_logger.log_fireworks("sentiment", cfg.model_id, t_fw)
                if answer:
                    if _run_logger:
                        _run_logger.finish_question(answer, solver_ms=t_fw)
                    return answer
            elif category == "math" and is_hard_math(prompt):
                t_fw = time.monotonic()
                cfg = _fw_route("math", prompt, complexity)
                answer = _fw.solve(
                    cfg.model_id, cfg.system_prompt, prompt,
                    max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    prefill=cfg.prefill, task_type="math",
                )
                t_fw = (time.monotonic() - t_fw) * 1000
                if _run_logger:
                    _run_logger.log_fireworks("hard_math", cfg.model_id, t_fw)
                if answer:
                    if _run_logger:
                        _run_logger.finish_question(answer, solver_ms=t_fw)
                    return answer
            elif category == "logic" and is_hard_logic(prompt, category):
                t_fw = time.monotonic()
                cfg = _fw_route("logic", prompt, complexity)
                answer = _fw.solve(
                    cfg.model_id, cfg.system_prompt, prompt,
                    max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    prefill=cfg.prefill, task_type="general",
                )
                t_fw = (time.monotonic() - t_fw) * 1000
                if _run_logger:
                    _run_logger.log_fireworks("hard_logic", cfg.model_id, t_fw)
                if answer:
                    if _run_logger:
                        _run_logger.finish_question(answer, solver_ms=t_fw)
                    return answer
            elif category in ("code_gen", "code_debug"):
                t_fw = time.monotonic()
                # code_gen / code_debug always route to Fireworks when a key is set.
                # Local 3B on CPU needs 20-35s for complex functions — exceeds the 22s
                # local budget, leaving too little time for the FW fallback. Bypass entirely.
                # code_debug also had this problem: local timeout consumed all slack time
                # leaving FW only ~6s which wasn't enough → now routes upfront like code_gen.
                cfg = _fw_route(category, prompt, complexity)
                answer = _fw.solve(
                    cfg.model_id, cfg.system_prompt, prompt,
                    max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    prefill=cfg.prefill, task_type="code",
                )
                t_fw = (time.monotonic() - t_fw) * 1000
                if _run_logger:
                    _run_logger.log_fireworks(category, cfg.model_id, t_fw)
                if answer:
                    cleaned = _strip_kimi_preamble(answer)
                    if _run_logger:
                        _run_logger.finish_question(cleaned, solver_ms=t_fw)
                    return cleaned
        except Exception as exc:
            logger.warning("Fireworks escalation failed (%s) — falling through to local", exc)

    # 6. Deterministic solvers — try before the LLM, free and fast
    det_results: list[tuple[str, str]] = []
    if category in _DET_CAT_MAP:
        solver_cat = _DET_CAT_MAP[category]
        t_det = time.monotonic()
        for solver_fn in _DET_SOLVERS:
            try:
                s_answer = solver_fn(prompt, solver_cat)
                if s_answer:
                    det_results.append((solver_fn.__name__, s_answer))
                    break
            except Exception:
                pass
        t_det = (time.monotonic() - t_det) * 1000
        if det_results and _run_logger:
            _run_logger.log_deterministic(
                solver_results=det_results,
                hint=det_results[0][1] if det_results else "",
                elapsed_ms=t_det,
            )
        if det_results:
            if _run_logger:
                _run_logger.finish_question(det_results[0][1], solver_ms=t_det)
            return det_results[0][1]

    # 7. Local LLM — use default inference timeout
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": prompt},
    ]
    t_llm = time.monotonic()
    answer, usage = _infer(messages, max_tok, stop_seq, timeout=INFERENCE_TIMEOUT_S)
    t_llm = (time.monotonic() - t_llm) * 1000

    # Retry without system prompt if first attempt returned empty
    retry = False
    tok_u = usage or {}
    prompt_tok = tok_u.get("prompt_tokens", 0)
    completion_tok = tok_u.get("completion_tokens", 0)
    total_tok = tok_u.get("total_tokens", 0)
    if not answer:
        t_llm2 = time.monotonic()
        answer2, usage2 = _infer([{"role": "user", "content": prompt}], max_tok, stop_seq, timeout=INFERENCE_TIMEOUT_S)
        t_llm += (time.monotonic() - t_llm2) * 1000
        if answer2:
            answer = answer2
            tok_u2 = usage2 or {}
            prompt_tok += tok_u2.get("prompt_tokens", 0)
            completion_tok += tok_u2.get("completion_tokens", 0)
            total_tok += tok_u2.get("total_tokens", 0)
        retry = True

    if _run_logger:
        _run_logger.log_local_llm(elapsed_ms=t_llm, retry=retry,
                                   prompt_tokens=prompt_tok,
                                   completion_tokens=completion_tok,
                                   total_tokens=total_tok)

    # 7b. Post-processing — extract final answer from reasoning text
    post_proc = ""

    # Math: extract Answer: marker or last option letter+text from reasoning
    if category == "math" and answer:
        m = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip().split('\n')[0].strip()
            post_proc = "math_answer_extract"
        elif _is_mc:
            # Pull the last "x) some text" from reasoning output
            mc_all = list(re.finditer(r'\b([a-dA-D]\))\s*([^\n]{2,60})', answer))
            if mc_all:
                last = mc_all[-1]
                answer = f"{last.group(1).lower()} {last.group(2).strip().rstrip('.,;')}"
                post_proc = "math_mc_extract"

    # Logic: strip preamble or expand bare option letter to full text
    if category == "logic" and answer:
        # Bare option letter like "C)" — expand to full text from prompt
        bare_letter = re.match(r'^([A-Da-d]\)?)\s*$', answer.strip())
        if bare_letter:
            letter = bare_letter.group(1).upper().rstrip(")") if bare_letter else ""
            letter = letter or ""
            opt_m = re.search(
                rf'(?<!\w){re.escape(letter)}\)\s*([^\n]{{3,120}})', prompt
            )
            if opt_m:
                answer = f"{letter}) {opt_m.group(1).strip()}"
                post_proc = "logic_bare_letter_expand"
        else:
            m = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
            if m:
                answer = m.group(1).strip()
                post_proc = "logic_answer_extract"
            elif re.match(r'^(to solve|let me|we need|first|step|consider|given)', answer, re.I):
                m2 = re.search(r'((?:floor|position|seat|day|slot|person|name)\s*\d+\s*:.*)', answer, re.I | re.DOTALL)
                if m2:
                    answer = m2.group(1).strip()
                    post_proc = "logic_assignment_extract"

    if _run_logger:
        _run_logger.log_post_processing(post_proc)

    # 8. code_gen syntax fallback — AST failure → Fireworks retry
    if category == "code_gen" and not syntax_ok(answer):
        t_fall = time.monotonic()
        fw_answer = _fw_fallback("syntax check failed")
        t_fall = (time.monotonic() - t_fall) * 1000
        if fw_answer:
            answer = _strip_kimi_preamble(fw_answer)
            if _run_logger:
                _run_logger.log_fallback("syntax_check_failed")
                _run_logger.log_fireworks("code_gen_syntax_fallback", "", t_fall)

    # 9. Universal fallback — local timed out or returned empty → Fireworks
    if not answer:
        t_fall = time.monotonic()
        fw_answer = _fw_fallback("local inference empty")
        t_fall = (time.monotonic() - t_fall) * 1000
        if fw_answer:
            answer = _strip_kimi_preamble(fw_answer) if category in ("code_gen", "code_debug") else fw_answer
            if _run_logger:
                _run_logger.log_fallback("local_empty")
                _run_logger.log_fireworks("universal_fallback", "", t_fall)
        else:
            if _run_logger:
                _run_logger.log_fallback("local_empty_no_fw_key")

    if _run_logger:
        _run_logger.finish_question(answer or "", solver_ms=t_llm)
    return answer or ""

# ── Entry point ────────────────────────────────────────────────────────────
# Hackathon output contract: writes /output/results.json AND prints to stdout.
def _write_output(results: list[dict]) -> None:
    os.makedirs("/output", exist_ok=True)
    tmp = "/output/results.json.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, "/output/results.json")

if __name__ == "__main__":
    eval_path = sys._early_eval_path or sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_HERE, "..", "data", "eval", "primary", "eval_mini_10.json")

    with open(eval_path) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data

    # Initialise run logger
    import subprocess
    try:
        _pipeline_version = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, cwd=_HERE, timeout=5
        ).stdout.strip()
    except Exception:
        _pipeline_version = "unknown"
    if not _pipeline_version:
        _pipeline_version = "unknown"

    _run_logger = RunLogger(
        run_number=None,  # auto-increment
        pipeline_version=_pipeline_version,
        model_path=MODEL_PATH,
        fireworks_model=FIREWORKS_MODEL,
        fireworks_key_set=bool(_fw.api_key),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        num_questions=len(questions),
        eval_source=os.path.basename(eval_path),
    )

    results = []
    for i, q in enumerate(questions):
        tid = q.get("task_id", f"idx_{i}")
        prompt = q.get("prompt", q.get("question", ""))
        answer = _process(prompt, task_id=tid, difficulty=q.get("difficulty", ""))
        results.append({"task_id": tid, "answer": answer})
        # Flush partial output every 5 tasks for crash safety
        if (i + 1) % 5 == 0:
            _write_output(results)
        # Hackathon contract: one flat line per answer (newlines escaped)
        print(answer.replace("\n", "\\n"))

    _write_output(results)

    # Write Excel log
    xlsx_dir = os.path.join(_HERE, "eval_results")
    try:
        fpath = _run_logger.write_xlsx(xlsx_dir)
        logger.warning("Run log written to %s", fpath)
    except Exception as exc:
        logger.warning("Failed to write run log: %s", exc)

    logger.warning("Wrote %d results to /output/results.json", len(results))
