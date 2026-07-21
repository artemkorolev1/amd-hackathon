#!/usr/bin/env python3
"""Evaluation agent — runs cells on dev sets and collects multi-objective metrics.

Interface:
    agent = EvaluationAgent(model_cache: ModelCache)
    scored = agent.evaluate(cells, questions)

Where `questions` is a list of dicts with:
    category, prompt, expected_answer, task_id, difficulty

Returns scored cells with accuracy, avg_output_tokens, avg_latency_ms,
and format_compliance filled into .metadata.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from agent.cell import Cell

logger = logging.getLogger("evaluation_agent")


# ── Format / validation checkers ────────────────────────────────────────────

def _check_json_validity(text: str) -> bool:
    """Return True if text is valid JSON."""
    text = text.strip()
    if not text:
        return False
    if text.startswith("```"):
        # Strip code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_sentence_count(text: str, expected: Optional[int] = None) -> bool:
    """Check sentence count against expected (if provided) or basic sanity."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    count = len(sentences)
    if expected is not None:
        return count == expected
    return 1 <= count <= 5  # basic sanity


def _check_word_limit(text: str, max_words: Optional[int] = None) -> bool:
    words = len(text.split())
    if max_words is not None:
        return words <= max_words
    return True


def _check_has_entities(text: str) -> bool:
    """Rough check that NER-style output has entity labels."""
    labels = ["PERSON:", "ORG:", "LOC:", "DATE:", "ORGANIZATION:", "LOCATION:"]
    return any(label in text.upper() for label in labels)


# ── Metric registry ─────────────────────────────────────────────────────────

def _compute_extra_metrics(category: str, text: str, expected: str,
                            format_confidence: str = "low") -> dict:
    """Compute format compliance metrics specific to the task category.

    Args:
        category: Task category (e.g. "sentiment", "math").
        text: The model's answer (post-normalization).
        expected: The expected answer.
        format_confidence: Normalizer confidence level ("high", "medium", "low").
    """
    metrics = {
        "format_compliant": 1.0,
        "is_json": 0.0,
        "sentence_count_ok": 0.0,
        "word_limit_ok": 1.0,
    }
    if not text:
        return {k: 0.0 for k in metrics}

    if category in ("math",):
        # Must contain a number
        metrics["format_compliant"] = 1.0 if re.search(r"\d", text) else 0.0

    elif category in ("code_gen", "code_debug"):
        metrics["is_json"] = 1.0 if _check_json_validity(text) else 0.0
        metrics["format_compliant"] = metrics["is_json"]

    elif category == "factual":
        # Not too long, no preamble
        words = len(text.split())
        metrics["word_limit_ok"] = 1.0 if words <= 120 else 0.0
        metrics["format_compliant"] = metrics["word_limit_ok"]

    elif category == "ner":
        metrics["format_compliant"] = 1.0 if _check_has_entities(text) else 0.0

    elif category == "sentiment":
        valid_labels = {"positive", "negative", "neutral", "mixed"}
        is_valid = text.strip().lower() in valid_labels
        metrics["format_compliant"] = 1.0 if is_valid else 0.0
        metrics["format_confidence"] = format_confidence

    elif category == "summarization":
        metrics["sentence_count_ok"] = 1.0 if _check_sentence_count(text) else 0.0
        metrics["format_compliant"] = metrics["sentence_count_ok"]

    return metrics


# ── Fuzzy match (from gepa_runner.py — canonical version) ───────────────────

def fuzzy_match(answer: str, expected: str) -> bool:
    """4-cascade fuzzy match: exact → substring → numeric 1% → token overlap ≥80%."""
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False


# ── Evaluation Agent ─────────────────────────────────────────────────────────

class EvaluationAgent:
    """Runs cells against eval questions and produces multi-objective scores.

    Uses a ModelCache (lazy-loaded llama_cpp.Llama instances) for inference.
    """

    def __init__(self, model_cache: Any):
        self._model_cache = model_cache
        self._eval_cache: dict[str, dict] = {}  # key → result (eval cache)

    def evaluate(
        self,
        cells: list[Cell],
        questions: list[dict],
        cache_key: Optional[str] = None,
    ) -> list[Cell]:
        """Run each cell on the given questions. Returns cells with metadata.

        Args:
            cells: list of Cell objects.
            questions: list of dicts with 'category', 'prompt', 'expected_answer'.
            cache_key: optional string to cache results for idempotency.

        Returns:
            Modified cells with accuracy, avg_output_tokens, avg_latency_ms,
            format_compliance filled into .metadata.
        """
        # Group cells by model_key so we can share loaded models
        by_model: dict[str, list[tuple[int, Cell]]] = {}
        for idx, c in enumerate(cells):
            by_model.setdefault(c.model_key, []).append((idx, c))

        for model_key, cell_list in by_model.items():
            logger.info("Evaluating %d cells on model %s", len(cell_list), model_key)
            try:
                llm = self._model_cache.get(model_key)
            except Exception as e:
                logger.warning("Failed to load model %s: %s", model_key, e)
                for _, c in cell_list:
                    c.metadata["accuracy"] = 0.0
                    c.metadata["avg_output_tokens"] = 0
                    c.metadata["avg_latency_ms"] = 99999.0
                    c.metadata["format_compliance"] = 0.0
                    c.metadata["error"] = str(e)
                continue

            for _, cell in cell_list:
                self._evaluate_cell(llm, cell, questions)

        return cells

    def evaluate_single(self, cell: Cell, questions: list[dict]) -> Cell:
        """Evaluate a single cell. Convenience wrapper."""
        return self.evaluate([cell], questions)[0]

    def clear_cache(self):
        self._eval_cache.clear()

    # ── Internal ────────────────────────────────────────────────────────────

    def _evaluate_cell(self, llm: Any, cell: Cell, questions: list[dict]):
        """Run inference for this cell on its task-specific questions."""
        # Filter questions by cell's task category
        cat = cell.pipeline_category
        task_questions = [q for q in questions if q.get("category") == cat]
        if not task_questions:
            # Fall back to generic questions — use all
            task_questions = questions
            if not task_questions:
                logger.warning("No questions for category %s (cell %s)", cat, cell.name)
                cell.metadata["accuracy"] = 0.0
                cell.metadata["avg_output_tokens"] = 0
                cell.metadata["avg_latency_ms"] = 0.0
                cell.metadata["format_compliance"] = 0.0
                return

        correct = 0
        total_tokens = 0
        total_latency = 0.0
        total_format = 0.0
        details: list[dict] = []
        temperature = cell.decoding.temperature
        max_tokens = cell.decoding.max_tokens

        for q in task_questions:
            prompt_text = q.get("prompt", "")
            expected = q.get("expected_answer", "")
            q_cat = q.get("category", cat)

            messages = [
                {"role": "system", "content": cell.system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            start = time.time()
            try:
                # Pass ALL decoding params from cell.decoding (not just temperature + max_tokens)
                dec = cell.decoding
                resp = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=dec.max_tokens,
                    temperature=dec.temperature,
                    top_p=dec.top_p,
                    top_k=dec.top_k,
                    min_p=dec.min_p,
                    repeat_penalty=dec.repeat_penalty,
                    seed=dec.seed,
                )
                elapsed = (time.time() - start) * 1000  # ms
                answer = resp["choices"][0]["message"]["content"].strip()
                # NEW: pass through format normalizer for deterministic cleanup
                from agent.solvers.format_normalizer import normalize_sentiment_output
                answer, format_confidence = normalize_sentiment_output(answer)
                usage = resp.get("usage", {})
                tok_count = usage.get("completion_tokens", len(answer.split()))

                is_correct = fuzzy_match(answer, expected)
                if is_correct:
                    correct += 1
                total_tokens += tok_count
                total_latency += elapsed

                # Format compliance
                extra = _compute_extra_metrics(q_cat, answer, expected, format_confidence)
                total_format += extra.get("format_compliant", 1.0)

                details.append({
                    "task_id": q.get("task_id", ""),
                    "question": prompt_text[:60],
                    "expected": expected,
                    "got": answer[:120],
                    "correct": is_correct,
                    "latency_ms": round(elapsed, 1),
                    "tokens": tok_count,
                    "format": extra,
                })

            except Exception as e:
                logger.warning("Cell %s question %s failed: %s",
                               cell.name, q.get("task_id", "?"), e)
                total_tokens += 0
                total_latency += 100  # penalty
                total_format += 0.0
                details.append({
                    "task_id": q.get("task_id", ""),
                    "question": prompt_text[:60],
                    "error": str(e),
                    "correct": False,
                })

        n = len(task_questions)
        cell.metadata["accuracy"] = round(correct / n, 4) if n else 0.0
        cell.metadata["correct"] = correct
        cell.metadata["total"] = n
        cell.metadata["avg_output_tokens"] = round(total_tokens / n, 1) if n else 0
        cell.metadata["avg_latency_ms"] = round(total_latency / n, 1) if n else 0.0
        cell.metadata["format_compliance"] = round(total_format / n, 4) if n else 0.0
        cell.metadata["category"] = cat
        # Store full details for analysis
        cell.metadata["details"] = details

        logger.info("  Cell %s: acc=%.3f tok=%.0f lat=%.0fms fmt=%.3f (n=%d)",
                    cell.name, cell.metadata["accuracy"],
                    cell.metadata["avg_output_tokens"],
                    cell.metadata["avg_latency_ms"],
                    cell.metadata["format_compliance"], n)
