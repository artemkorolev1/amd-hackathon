#!/usr/bin/env python3
"""
Pipeline — plug-and-play agent pipeline for AMD ACT II Track 1. 100% local.

Usage:
    from agent.pipeline import Pipeline

    pipe = Pipeline()
    answer = pipe.process("What is 2+2?")
    results = pipe.process_batch([{"task_id": "t1", "prompt": "..."}])

All configuration via constructor kwargs (override) or env vars (defaults).
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import io as _io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.workflow import WorkflowEngine
import agent.dynamic_prompts as _dp
from agent.classifier import classify_with_detail as _classify_detail
from agent.complexity import score as mlm_complexity
from agent.pre_filter import pre_filter
from agent.solvers.deterministic import (
    solve_arithmetic,
    solve_code_debugging,
    solve_factual_qa,
    solve_logic,
    solve_math_word_problems,
    solve_ner,
    solve_sentiment,
    solve_summarization,
)
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3
from agent.solvers.prototype_zebra_v2 import solve_zebra_puzzle
from agent.solvers.logic_reasoning import solve_logical_reasoning
from agent.solvers.deterministic import solve_code_generation
from agent.solvers.code_tool_cascade import route_code as _cascade_route
from agent.solvers.local_vote import solve_with_consensus
from agent.solvers.verify import verify
from agent.secondary_summarization import resolve_summarization

logger = logging.getLogger("pipeline")

# ── Regex constants ─────────────────────────────────────────────────────────
_DOC_HEADER_RE = re.compile(
    r'^(HEADLINE:|LEGAL BRIEF|STATEMENT BY|ARTICLE:|TRANSCRIPT:|MEMO:|PRESS RELEASE:)',
    re.IGNORECASE | re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_HARD_MATH_RE = re.compile(
    r"\b(law of sines|law of cosines|geometric series|cofactor|determinant"
    r"|inclusion.exclusion|bayes(?:ian)?|conditional probability"
    r"|permutations?|combinations?|integral|derivative|matrix|eigenvalu"
    r"|logarithm|log base|\bmod\b|modular arithmetic|chinese remainder"
    r"|ratio|proportion|in the ratio|lcm|gcd|least common multiple"
    r"|greatest common divisor|rate.*time|time.*rate|work rate"
    r"|how many (?:different |distinct |possible )?ways"
    r"|distinct.*\bdigits?|distinct.*\bnumbers?"
    r"|nCr|nPr"
    r"|compound interest|simple interest|annual interest|interest rate"
    r"|pipes|fill.*pool|drain.*pool|fill.*tank|drain.*tank"
    r"|upstream|downstream|boat.*current|current.*boat"
    r"|cone|sphere|cylinder|pyramid|volume of|surface area"
    r"|age.*(?:years|old)|years.*(?:older|younger)"
    r"|mixture|mixed|alloy|concentration|percentage of"
    r"|discount|profit.*percent|percent.*profit|markup"
    r"|speed.*distance|distance.*speed|time.*distance"
    r"|together.*work|together.*paint|work.*together|job.*alone"
    r"|if.*then.*how many|many.*more.*than"
    r"|number.*word|digit.*number|digit.*arrang"
    r"|train.*speed|train.*distance|plane.*speed|plane.*distance"
    r"|average speed|average of|mean of|median of"
    r"|\d+\s*%\s+of\s+\d+|\d+\s*percent\s+(of|more|less)"
    r")\b",
    re.IGNORECASE,
)
_MULTI_PERSON_RE = re.compile(
    r"\b(friends?|colleagues?|neighbors?|students?|candidates?|people)\b"
    r".{0,40}\b(each|all)\b",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(r"\b\d+\s*:\s*\d+\b")


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for a Pipeline instance. All fields have env-var defaults."""

    # Model
    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        )
    )
    n_gpu_layers: int = field(
        default_factory=lambda: int(os.environ.get("N_GPU_LAYERS", "0"))
    )
    n_ctx: int = field(
        default_factory=lambda: int(os.environ.get("N_CTX", "2048"))
    )
    n_threads: int = field(
        default_factory=lambda: int(os.environ.get("N_THREADS", "2"))
    )

    # Multi-model routing: category → local model path
    # Based on 300-set evaluation (July 12, 2026):
    #   qwen2.5-1.5b wins factual(81%), logic(68%), math(63%), summarization(75%)
    #   qwen2.5-coder wins NER(100%), code_debug(100%), code_gen(95%)
    #   gemma-3-1b wins code_gen(100%), 2x faster on small tasks
    #   Qwen2.5-Math is 37% overall — terrible, only good for sentiment(92%)
    category_model_map: dict[str, str] = field(default_factory=lambda: {
        "math": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "logic": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "factual": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "summarization": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "code_debug": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "code_gen": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
        "ner": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "sentiment": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    })

    # Timing
    inference_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("INFERENCE_TIMEOUT_S", "28.0"))
    )

    # Deadline (grader env var)
    deadline_s: float = field(default=600.0)

    # Consensus voting (default OFF)
    consensus_samples: int = field(
        default_factory=lambda: int(os.environ.get("CONSENSUS_SAMPLES", "1"))
    )
    consensus_categories: set[str] = field(
        default_factory=lambda: {"math", "sentiment", "ner"}
    )

    # Task routing
    det_cat_map: dict[str, str] = field(default_factory=lambda: {
        "math": "math_arithmetic",
        "logic": "logical_reasoning",
        "sentiment": "sentiment",
        "summarization": "summarization",
        "factual": "other_complex",
        "code_debug": "code_debugging",
        "code_gen": "code_gen",
        "ner": "ner",
    })
    det_solvers: list[Callable] = field(default_factory=lambda: [
        # NER: old regex (80%) BEFORE v3 (66%) — order matters
        solve_ner,
        solve_ner_v3,
        # Logic
        solve_zebra_puzzle,
        solve_logical_reasoning,
        solve_logic,
        # Rest (each fires independently per category)
        solve_sentiment,
        solve_factual_qa,
        solve_code_debugging,
        solve_code_generation,
        solve_summarization,
        solve_math_word_problems,
    ])

    # Reasoning model detection
    reasoning_keywords: list[str] = field(
        default_factory=lambda: ["nemotron"]
    )

    def __post_init__(self):
        """Parse deadline from env."""
        try:
            self.deadline_s = float(os.environ.get("DEADLINE_S", str(self.deadline_s)))
        except (ValueError, TypeError):
            pass  # keep default


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_hard_math(prompt: str) -> bool:
    if _HARD_MATH_RE.search(prompt):
        return True
    if len(re.findall(r"\\[a-zA-Z]+", prompt)) >= 2:
        return True
    if _RATIO_RE.search(prompt):
        return True
    return False


def _multi_step_math(prompt: str) -> bool:
    """Detect multi-step word problems that keyword patterns miss."""
    p_lower = prompt.lower()
    # GSM8K-style: "Solve:" prefix
    if p_lower.startswith("solve:"):
        return True
    # 3+ numeric values in a narrative suggests multi-step
    numbers = re.findall(r"\b\d+(\.\d+)?\b", prompt)
    if len(numbers) >= 3 and p_lower.startswith(("if ", "a ", "the ", "there ")):
        return True
    # Narrative word problem patterns
    word_problem = bool(re.search(r"\b(how many|how much|what is the (?:total|value|sum))"
                                  r"|find (?:the |how )|compute|determine|calculate"
                                  r"|what (?:would|will|must|should)", p_lower))
    if word_problem and len(re.findall(r"\b\d+\b", prompt)) >= 2:
        return True
    return False


def _is_hard_logic(prompt: str, category: str) -> bool:
    if category != "logic":
        return False
    # Named-entity puzzles (zebra, seating, ordering)
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)) >= 4:
        return True
    if _MULTI_PERSON_RE.search(prompt):
        return True
    # LogiQA argument analysis patterns
    p_lower = prompt.lower()
    if re.search(r"which\s+of\s+the\s+following\s+(weakens?|strengthens?|assumes?|supports?|conclusion|infer|follows|must\s+be\s+true|explains?|resolves?)", p_lower):
        return True
    if re.search(r"(?:which|what)\s+(conclusion|assumption|inference|argument|reasoning|flaw|fallacy)", p_lower):
        return True
    if re.search(r"\(1\)\s", prompt) and re.search(r"(?:above|argument|conclusion|reasoning)", p_lower):
        return True
    if len(re.findall(r"\b\d+\.\s", prompt)) >= 3 and "reasoning" in p_lower:
        return True
    return False


def _strip_kimi_preamble(text: str, model_id: str = "") -> str:
    """Remove thinking preamble before the first code fence (Kimi-specific)."""
    if "kimi" not in model_id.lower():
        return text
    for marker in ("```python", "```"):
        idx = text.find(marker)
        if idx > 0:
            return text[idx:]
        if idx == 0:
            return text
    return text


def _syntax_ok(code: str) -> bool:
    """Return True if `code` contains syntactically valid Python."""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"```(?:python\w*)?\n?([\s\S]+?)\n?```", code)
    src = m.group(1) if m else code
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _secondary_category(raw_scores: dict[str, float], primary: str) -> str:
    sorted_cats = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    for cat, _ in sorted_cats:
        if cat != primary:
            return cat
    return ""


def _strip_think(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    if stripped != text.strip():
        return stripped
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


# ── Pipeline ────────────────────────────────────────────────────────────────

class Pipeline:
    """Plug-and-play agent pipeline.

    Usage:
        pipe = Pipeline()                         # reads env vars
        pipe = Pipeline(infrence_timeout_s=30)   # override specific fields
        answer = pipe.process("What is 2+2?")
        results = pipe.process_batch(tasks)
        pipe.close()
    """

    def __init__(self, config: Optional[PipelineConfig] = None,
                 routing_table: Optional[dict] = None):
        self.cfg = config or PipelineConfig()
        self._log = logger

        # Optional routing table from GEPA Orchestrator
        # Dict of {category: {model_key, system_prompt, decoding, aggregation}}
        self._routing_table = routing_table or {}

        # Reasoning headroom
        _mp_lower = self.cfg.model_path.lower()
        self._reasoning_headroom = (
            4 if any(kw in _mp_lower for kw in self.cfg.reasoning_keywords) else 1
        )
        self._cpu_token_factor = 0.75 if self.cfg.n_gpu_layers == 0 else 1.0

        # Reasoning-model prompts
        self._reasoning_prompts: dict[str, str] = {
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
                "Solve the logic puzzle. Output ONLY the final answer — NO preamble, "
                "NO 'To solve...', NO step-by-step reasoning in your response. "
                "For assignment puzzles: immediately output ALL assignments on ONE line: "
                "'Position 1: Name (Role); Position 2: Name (Role); ...'. "
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

        # Load local model
        self._llm: Any = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._category_llms: dict[str, Any] = {}  # per-category cached models
        self._load_local_model()

        # Pre-warm complexity model
        with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
            mlm_complexity("warmup")

        self._log.info("Pipeline ready (headroom=%d, cpu_factor=%.2f)",
                       self._reasoning_headroom, self._cpu_token_factor)

    # ── Model lifecycle ─────────────────────────────────────────────────

    def _load_local_model(self):
        """Load the local GGUF model."""
        if not os.path.isfile(self.cfg.model_path):
            self._log.warning("Model not found at %s — continuing without local LLM",
                              self.cfg.model_path)
            return
        try:
            from llama_cpp import Llama
            self._log.warning("Loading %s  (n_gpu_layers=%d)",
                              os.path.basename(self.cfg.model_path),
                              self.cfg.n_gpu_layers)
            self._llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_gpu_layers=self.cfg.n_gpu_layers,
                n_threads=self.cfg.n_threads,
                flash_attn=True,
                verbose=False,
            )
            self._log.warning("Model ready")
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        except Exception as exc:
            self._log.warning("Model load failed: %s — continuing without local LLM", exc)

    def _get_llm_for_category(self, category: str):
        """Return the cached LLM for a category, loading it if needed.
        
        Falls back to the default self._llm if no category-specific model is
        configured or the model file doesn't exist.
        """
        if category in self._category_llms:
            return self._category_llms[category]
        
        model_path = self.cfg.category_model_map.get(category)
        if model_path and os.path.isfile(model_path):
            try:
                from llama_cpp import Llama
                self._log.warning("Loading %s  (n_gpu_layers=%d) for category %s",
                                  os.path.basename(model_path),
                                  self.cfg.n_gpu_layers, category)
                llm = Llama(
                    model_path=model_path,
                    n_ctx=self.cfg.n_ctx,
                    n_gpu_layers=self.cfg.n_gpu_layers,
                    n_threads=self.cfg.n_threads,
                    flash_attn=True,
                    verbose=False,
                )
                self._category_llms[category] = llm
                self._log.warning("Model ready for category %s", category)
                return llm
            except Exception as exc:
                self._log.warning("Failed to load model for %s: %s — using default", category, exc)
        return self._llm

    def close(self):
        """Release resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        # Clean up per-category models
        for cat, llm in self._category_llms.items():
            try:
                del llm
            except Exception:
                pass
        self._category_llms.clear()

    # ── Deterministic solver dispatch ────────────────────────────────────

    def _run_deterministic(self, category: str, prompt: str) -> str:
        """Try all registered deterministic solvers. Returns first success or ''."""
        if category not in self.cfg.det_cat_map:
            return ""
        solver_cat = self.cfg.det_cat_map[category]
        for solver_fn in self.cfg.det_solvers:
            try:
                answer = solver_fn(prompt, solver_cat)
                if answer:
                    return answer
            except Exception:
                pass
        return ""


    # ── Local inference ─────────────────────────────────────────────────

    def _infer(self, messages: list[dict], max_tok: int,
               stop_seq: list[str], timeout: float = 28.0,
               category: str = "") -> str:
        llm = self._get_llm_for_category(category) if category else self._llm
        if llm is None or self._executor is None:
            self._log.warning("Local LLM not loaded — skipping inference")
            return ""

        def _call():
            return llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tok,
                temperature=0.0,
                top_p=1.0,
                top_k=40,
                min_p=0.0,
                repeat_penalty=1.0,
                seed=None,
                stop=stop_seq,
            )

        try:
            future = self._executor.submit(_call)
            resp = future.result(timeout=timeout)
            raw = resp["choices"][0]["message"]["content"] or ""
            return _strip_think(raw)
        except concurrent.futures.TimeoutError:
            self._log.warning("Inference timed out after %.0fs — returning empty", timeout)
            return ""
        except Exception as exc:
            self._log.warning("Inference error: %s", exc)
            return ""

    # ── Core process ────────────────────────────────────────────────────

    def process(self, prompt: str) -> str:
        """Run the full pipeline on a single prompt and return the answer."""
        if not prompt or not prompt.strip():
            self._log.warning("Empty prompt received — returning empty answer")
            return ""

        # 1. Pre-Filter — immediate deterministic bypass
        s0 = pre_filter(prompt)
        if s0.action == "bypass" and s0.direct_answer:
            return s0.direct_answer

        # 2. Category classification
        detail = _classify_detail(prompt)
        category = detail["category"]
        score_delta = detail["score_delta"]
        raw_scores = detail["raw_scores"]

        # 2b. Keyword overrides
        _p_lower = prompt.lower()

        if category not in ("summarization",) and _DOC_HEADER_RE.search(prompt):
            category = "summarization"
            score_delta = 1.0

        if category != "summarization":
            if re.search(r'\bsummariz[ei]', _p_lower) or re.search(r'source\s+[12]\b', _p_lower, re.I):
                category = "summarization"
                score_delta = 1.0

        if category == "math" and len(prompt) > 600:
            if not re.search(
                r'[=×÷]|\\frac|\\int|\\sum|\bsolve\b|\bcalculate\b|\bcompute\b'
                r'|\bfind\b.*\b(?:value|sum|product|ratio)\b', prompt, re.I
            ):
                category = "summarization"
                score_delta = 1.0

        if category not in ("ner",) and re.search(
            r'\b(extract|identify|list)\b.{0,40}\b(named entity|entities|people mentioned|organizations mentioned)\b',
            _p_lower
        ):
            category = "ner"
            score_delta = 1.0

        if category in ("math",) and re.search(
            r'\bhow many\b.{0,30}\bare there\b',
            _p_lower
        ):
            category = "factual"
            score_delta = 1.0

        if category == "sentiment" and re.search(
            r'\bfor (someone|a person)\b.{0,50}\b(with your|of your)\b',
            prompt, re.IGNORECASE,
        ):
            return "negative"

        # 2c. Secondary summarization classifier — catches misroutes the
        #     keyword overrides miss (e.g., prose with incidental numbers
        #     that primary scored as math, or doc headers in narrative form)
        corrected = resolve_summarization(category, prompt)
        if corrected != category:
            category = corrected
            score_delta = 1.0

        # 3. Complexity scoring
        complexity = mlm_complexity(prompt)

        # 3a. Code tool cascade — route code_gen/code_debug through binary decision tree
        #     Returns solver_fn for template matches, prompt_key for LLM paths.
        _cascade_key: str | None = None
        _cascade_cleaned: str | None = None
        if category in ('code_gen', 'code_debug'):
            _cascade_result = _cascade_route(prompt, category)
            _cascade_answer = _cascade_result.solve()
            if _cascade_answer:
                return _cascade_answer
            _cascade_key = _cascade_result.prompt_key or None
            _cascade_cleaned = _cascade_result.cleaned or None

        # 3ab. Logic cascade — route through binary reject cascade for solver selection
        #       Covers: truth-teller, sequence, syllogism, constraint/zebra, argument analysis.
        #       Returns answer directly if solved, or None to continue to LLM path.
        if category == "logic":
            try:
                from agent.solvers.logic_classifier_cascade import route_logic
                logic_answer = route_logic(prompt)
                if logic_answer:
                    return logic_answer
            except Exception:
                self._log.warning("Logic cascade failed — continuing to deterministic path")

        # 3ac. NER cascade — route through binary reject cascade for solver selection
        #       Covers: tweet NER ({@...@}), spaCy general NER.
        #       Returns answer directly if solved, or None to continue to LLM path.
        if category == "ner":
            try:
                from agent.solvers.ner_classifier_cascade import route_ner
                ner_answer = route_ner(prompt)
                if ner_answer:
                    return ner_answer
            except Exception:
                self._log.warning("NER cascade failed — continuing to deterministic path")

        # 3b. Routing table override (if available)
        route_entry = self._routing_table.get(category)

        # New: Role-based dispatch (execution-backend routing)
        role = route_entry.get("role", "local_llm") if route_entry else "local_llm"

        # If role is "deterministic", try deterministic solvers first (they're free)
        if role == "deterministic":
            det_answer = self._run_deterministic(category, prompt)
            if det_answer:
                return det_answer

        # Role "api_llm" is deprecated (was Fireworks) — treat as regular local_llm
        if route_entry:
            # Check if this is a workflow cell
            steps = route_entry.get("steps")
            if steps:
                try:
                    from agent.cell import Cell, StepConfig, DecodingConfig
                    model_key = route_entry.get("model_key", "qwen2.5-1.5b")
                    decoding_dict = route_entry.get("decoding", {})
                    step_configs = [
                        StepConfig.from_dict(s) if isinstance(s, dict) else s
                        for s in steps
                    ]
                    cell = Cell(
                        task_id=category,
                        model_key=model_key,
                        steps=step_configs,
                    )
                    # Apply decoding overrides from routing entry
                    if decoding_dict:
                        cell.decoding = DecodingConfig.from_dict(decoding_dict)

                    engine = WorkflowEngine(self._infer)
                    result = engine.run(cell, prompt)

                    # Log per-step metrics
                    for step_res in result.get("step_results", []):
                        self._log.info(
                            "Workflow step %s: latency_ms=%.0f tokens=%d",
                            step_res.get("step", "?"),
                            step_res.get("latency_ms", 0),
                            step_res.get("tokens_est", 0),
                        )
                    self._log.info(
                        "Workflow complete for category=%s cell=%s "
                        "total_latency=%.0fms total_tokens=%d",
                        category, route_entry.get("cell_name", "?"),
                        result.get("total_latency", 0),
                        result.get("total_tokens_estimate", 0),
                    )
                    return result["final_answer"] or ""
                except Exception as exc:
                    self._log.exception(
                        "Workflow execution failed for category=%s: %s — "
                        "falling through to regular processing",
                        category, exc,
                    )
                    # Fall through to regular processing

            sys_prompt = route_entry.get("system_prompt", "")
            decoding = route_entry.get("decoding", {})
            max_tok = decoding.get("max_tokens", 128)
            temperature = decoding.get("temperature", 0.0)
            self._log.info("Routing table hit for category=%s → cell=%s",
                           category, route_entry.get("cell_name", "?"))
            # Use routing table's system prompt for local inference
            if self._llm is not None:
                messages = [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}]
                answer = self._infer(messages, max_tok, [],
                                     timeout=self.cfg.inference_timeout_s,
                                     category=category)
                return answer or ""
            # Fall through to default processing if no local LLM

        # 4. Build system prompt
        _mc_options = re.findall(r'(?<!\w)[a-dA-D]\)\s', prompt)
        _is_mc = len(set(o.strip().lower() for o in _mc_options)) >= 3

        _MC_MATH_PROMPT = (
            "Select the correct option from the choices given. "
            "Output ONLY the letter and its full text exactly as written, e.g., 'c) 4 hours'. "
            "No working. No explanation."
        )
        _MC_LOGIC_PROMPT = (
            "Select the correct option from the choices given. "
            "Output ONLY the letter and its full text exactly as written, "
            "e.g., 'B) Some logical people are not musicians.' "
            "No explanation."
        )

        if _cascade_key:
            sys_prompt = _dp.build_system_prompt(_cascade_key, complexity)
        elif self._reasoning_headroom > 1:
            sys_prompt = self._reasoning_prompts.get(category, self._reasoning_prompts["factual"])
        elif category == "math" and _is_mc:
            sys_prompt = _MC_MATH_PROMPT
        elif category == "logic" and _is_mc:
            sys_prompt = _MC_LOGIC_PROMPT
        elif category == "logic":
            sys_prompt = self._reasoning_prompts["logic"]
        else:
            secondary = _secondary_category(raw_scores, category)
            if score_delta < 0.5 and secondary:
                sys_prompt = _dp.build_merged_prompt(category, secondary, complexity)
            else:
                sys_prompt = _dp.build_system_prompt(category, complexity)

        # Token budget
        if _is_mc and category in ("math", "logic"):
            max_tok = 80
        else:
            max_tok = int(
                _dp.get_max_tokens(category, complexity)
                * self._reasoning_headroom
                * self._cpu_token_factor
            )

        # Stop sequences
        if self._reasoning_headroom > 1:
            stop_seq: list[str] = []
        elif category == "logic":
            stop_seq = ["Question:", "Context:"]
        else:
            stop_seq = _dp.get_stop_sequences(category)

        # 5. Deterministic solvers — run BEFORE local LLM so local knowledge
        #    (FactDB, arithmetic, NER rules) answers without an API call.
        #    SKIP code_gen/code_debug — the cascade binary tree (3a) already
        #    handled template/pattern matching.
        if category not in ('code_gen', 'code_debug') and category in self.cfg.det_cat_map:
            solver_cat = self.cfg.det_cat_map[category]
            for solver_fn in self.cfg.det_solvers:
                try:
                    answer = solver_fn(prompt, solver_cat)
                    if answer:
                        return answer
                except Exception:
                    pass

        # 5b. ToRA solver for math — LLM generates Python code, executed deterministically.
        #     Uses the category-specific LLM (not self._llm) to match the model the
        #     pipeline would use for math inference.
        #     When consensus_samples > 1, runs multiple ToRA attempts and votes.
        if category == "math":
            tora_llm = self._get_llm_for_category("math")
            if tora_llm is not None:

                # 5a. Variable extraction pre-filter — extract structured
                # variables from the word problem, then run ToRA on a clean
                # prompt with named variables instead of raw text.
                # If extraction fails, falls back to regular ToRA below.
                try:
                    from agent.solvers.variable_extractor import solve_with_extraction
                    ext_answer = solve_with_extraction(
                        prompt,
                        tora_llm,
                        self._infer,
                        max_tokens=512,
                    )
                    if ext_answer:
                        self._log.info("Extraction+ToRA solved: %s → %s", prompt[:60], ext_answer)
                        return ext_answer.strip()
                except Exception:
                    pass

                # Consensus ToRA: run ToRA N times at different temperatures
                # and take majority vote on extracted numeric answers.
                if self.cfg.consensus_samples > 1 and category in self.cfg.consensus_categories:
                    from agent.solvers.tora_solver import solve_with_tora
                    tora_results = []
                    temps = [0.1, 0.5, 0.7, 0.9]
                    n_attempts = min(self.cfg.consensus_samples, len(temps))
                    saved_temp = getattr(self._llm, 'temperature', None)

                    for attempt in range(n_attempts):
                        try:
                            if hasattr(self._llm, 'temperature'):
                                self._llm.temperature = temps[attempt]
                            ans = solve_with_tora(
                                prompt, tora_llm, self._infer,
                                max_tokens=512, timeout=10,
                            )
                            if ans and ans.strip():
                                tora_results.append(ans.strip())
                        except Exception:
                            continue
                        finally:
                            if saved_temp is not None and hasattr(self._llm, 'temperature'):
                                self._llm.temperature = saved_temp

                    if len(tora_results) >= 2:
                        import re as _re
                        from collections import Counter
                        def _extract_num(s):
                            ns = _re.findall(r"-?\d+(?:\.\d+)?", s)
                            return ns[-1] if ns else s
                        normalized = [_extract_num(s) for s in tora_results]
                        counter = Counter(normalized)
                        majority, count = counter.most_common(1)[0]
                        agreement = count / len(normalized)
                        self._log.info(
                            "ToRA consensus (%d/%d samples): %s (agree=%.2f)",
                            count, len(normalized), majority, agreement
                        )
                        if agreement >= 0.4:
                            return majority

                # Single-shot ToRA (when consensus disabled, or didn't agree)
                try:
                    from agent.solvers.tora_solver import solve_with_tora
                    tora_answer = solve_with_tora(
                        prompt,
                        tora_llm,
                        self._infer,
                        max_tokens=512,
                        timeout=10,
                    )
                    if tora_answer:
                        self._log.info("ToRA solved: %s → %s", prompt[:60], tora_answer)
                        return tora_answer.strip()
                except Exception as e:
                    self._log.warning("ToRA solver failed: %s", e)

                # 5b-ii. Iterative ToRA — fallback when single-shot fails.
                # Decomposes the problem into sub-steps and solves each independently,
                # passing intermediate results forward. Handles multi-step problems
                # where the model loses track in a single generation.
                try:
                    from agent.solvers.iterative_tora import solve_iterative_tora
                    self._log.info("Single-shot ToRA failed, trying iterative...")
                    iter_answer = solve_iterative_tora(
                        prompt,
                        tora_llm,
                        self._infer,
                        max_tokens=256,
                        timeout=10,
                    )
                    if iter_answer:
                        self._log.info("Iterative ToRA solved: %s → %s", prompt[:60], iter_answer)
                        return iter_answer.strip()
                    self._log.debug("Iterative ToRA also failed")
                except Exception as e:
                    self._log.warning("Iterative ToRA failed: %s", e)

        # 6. Local LLM with optional self-consistency voting
        answer: str = ""
        if (self.cfg.consensus_samples > 1
                and category in self.cfg.consensus_categories
                and self._reasoning_headroom <= 1):
            budget_per_sample = 30.0 / self.cfg.consensus_samples
            result = solve_with_consensus(
                llm=self._llm,
                prompt=prompt,
                category=category,
                system_prompt=sys_prompt,
                k=self.cfg.consensus_samples,
                max_tokens=max_tok,
                timeout_per_sample=budget_per_sample,
            )
            answer = result["majority_answer"]
            v = verify(answer, category)
            if not v.passed or result["agreement_score"] < 0.5:
                # Fall back to simple inference if consensus fails
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": _cascade_cleaned or prompt},
                ]
                answer = self._infer(messages, max_tok, stop_seq,
                                     timeout=self.cfg.inference_timeout_s,
                                     category=category)
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _cascade_cleaned or prompt},
            ]
            answer = self._infer(messages, max_tok, stop_seq,
                                 timeout=self.cfg.inference_timeout_s,
                                 category=category)
            if not answer:
                answer = self._infer(
                    [{"role": "user", "content": _cascade_cleaned or prompt}],
                    max_tok, stop_seq,
                    timeout=self.cfg.inference_timeout_s,
                    category=category,
                )

        # 7b. Post-processing
        if category == "math" and answer:
            m = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
            if m:
                answer = m.group(1).strip().split('\n')[0].strip()
            elif _is_mc:
                mc_all = list(re.finditer(r'\b([a-dA-D])\)\s*([^\n]{2,60})', answer))
                if mc_all:
                    last = mc_all[-1]
                    answer = f"{last.group(1).lower()}) {last.group(2).strip().rstrip('.,;')}"
            else:
                # Fallback: extract the last number from verbose output
                numbers = re.findall(r'-?\d+(?:\.\d+)?', answer)
                if numbers:
                    answer = numbers[-1]

        if category == "logic" and answer:
            bare_letter = re.match(r'^([A-Da-d])\)?\s*$', answer.strip())
            if bare_letter:
                letter = bare_letter.group(1).upper()
                opt_m = re.search(
                    rf'(?<!\w){re.escape(letter)}\)\s*([^\n]{{3,120}})', prompt
                )
                if opt_m:
                    answer = f"{letter}) {opt_m.group(1).strip()}"
            else:
                m = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
                if m:
                    answer = m.group(1).strip()
                elif re.match(r'^(to solve|let me|we need|first|step|consider|given)', answer, re.I):
                    m2 = re.search(
                        r'((?:floor|position|seat|day|slot|person|name)\s*\d+\s*:.*)',
                        answer, re.I | re.DOTALL,
                    )
                    if m2:
                        answer = m2.group(1).strip()

        # 8. code_gen syntax fallback removed (was Fireworks — 100% local now)

        # 9. Universal fallback (no Fireworks — just return what we have)
        # if not answer, we return empty string below

        return answer or ""

    def process_batch(self, tasks: list[dict], deadline_s: Optional[float] = None) -> list[dict]:
        """Process a batch of tasks with deadline management.

        Args:
            tasks: List of dicts with 'task_id' and 'prompt' (or 'question').
            deadline_s: Override the configured deadline.

        Returns:
            List of dicts with 'task_id' and 'answer'.
        """
        deadline = time.monotonic() + (deadline_s or self.cfg.deadline_s)
        results: list[dict] = []

        for i, q in enumerate(tasks):
            if time.monotonic() >= deadline:
                self._log.warning("Deadline reached after %d/%d tasks — stopping",
                                  i, len(tasks))
                break

            tid = q.get("task_id", f"idx_{i}")
            prompt = q.get("prompt", q.get("question", ""))
            try:
                answer = self.process(prompt)
            except Exception as exc:
                self._log.exception("Task %s failed — returning empty", tid)
                answer = ""
            results.append({"task_id": tid, "answer": answer})

        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
