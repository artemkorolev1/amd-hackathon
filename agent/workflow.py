#!/usr/bin/env python3
"""Workflow engine for Plan-and-Solve multi-step cells.

Executes a sequence of steps (LLM inference or deterministic tool calls),
passing intermediate artifacts between steps.

Part of Plan A (Extended Cell with Inline Steps).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from agent.cell import Cell, StepConfig, DecodingConfig


# ── Step type constants ──────────────────────────────────────────────────────
STEP_TYPE_PLAN = "plan"
STEP_TYPE_SOLVE = "solve"
STEP_TYPE_CRITIQUE = "critique"
STEP_TYPE_COMPOSE = "compose"
STEP_TYPE_TOOL = "tool"

# Known tool names
TOOL_SYMPY = "sympy"
TOOL_PYTHON = "python"
TOOL_SPACY = "spacy"
TOOL_CHUNK_TEXT = "chunk_text"

# ── Task-specific workflow templates ─────────────────────────────────────────

MATH_3STEP_WORKFLOW = [
    StepConfig(name="plan", system_prompt="Analyze the math problem and list the steps needed to solve it. Extract variables and numbers. Output ONLY the plan, not the solution."),
    StepConfig(name="solve", system_prompt="Execute the plan step by step. Show calculations clearly. Put the final numeric answer in \\boxed{}."),
    StepConfig(name="compose", system_prompt="Present the final answer clearly. Format: 'The answer is \\boxed{number}'."),
]

LOGIC_3STEP_WORKFLOW = [
    StepConfig(name="plan", system_prompt="Identify the premises, conclusion, and any hidden assumptions. Outline the reasoning steps."),
    StepConfig(name="reason", system_prompt="Work through the reasoning step by step. Be explicit about each inference."),
    StepConfig(name="compose", system_prompt="Present the final answer clearly."),
]

NER_2STEP_WORKFLOW = [
    StepConfig(name="extract", system_prompt="Extract all named entities. Label each as PERSON, ORG, LOC, or DATE. Format: TYPE: name"),
    StepConfig(name="verify", system_prompt="Review the extracted entities. Remove any that aren't explicitly in the text. Fix any incorrect labels.", input_from="extract"),
]


TEMPLATE_REGISTRY: dict[str, list[StepConfig]] = {
    "math_3step": MATH_3STEP_WORKFLOW,
    "logic_3step": LOGIC_3STEP_WORKFLOW,
    "ner_2step": NER_2STEP_WORKFLOW,
}


# ── Artifact helpers ─────────────────────────────────────────────────────────

def build_step_messages(step: StepConfig, artifact: str, all_artifacts: dict[str, str]) -> list[dict]:
    """Build messages for a workflow step.

    - Step 0 (first step): user message = original prompt, system = step prompt
    - Steps 1+ (continuation): user message = previous step's output, system = step prompt
    """
    prior_context = _summarize_artifacts(all_artifacts, exclude=step.name)

    system_content = step.system_prompt
    user_content = all_artifacts.get("_input", artifact)

    prior_keys = [k for k in all_artifacts if k != "_input" and k != step.name]
    if prior_keys:
        # Continuation step: embed prior output + step prompt into user message
        last_key = prior_keys[-1]
        last_output = all_artifacts[last_key]
        original = all_artifacts.get("_input", "")
        if len(last_output) > 400:
            last_output = last_output[:400] + "\n...[truncated]"
        user_content = f"Original problem:\n{original}\n\n=== [{last_key}] output ===\n{last_output}\n\nNow: {step.system_prompt}"
        if prior_context and len(prior_keys) > 1:
            system_content += f"\n(There were also earlier steps: {', '.join(prior_keys[:-1])})"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def _summarize_artifacts(artifacts: dict[str, str], exclude: str) -> str:
    """Build a compact summary of all prior step outputs for context injection."""
    parts = []
    for key, val in artifacts.items():
        if key == "_input" or key == exclude:
            continue
        # Truncate long artifacts
        text = val[:300] + "..." if len(val) > 300 else val
        parts.append(f"[{key}]: {text}")
    return "\n\n".join(parts)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} if present."""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    return m.group(1) if m else None


# ── Tool registry ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200) -> str:
    """Split text into N-word chunks, joined by '\\n---CHUNK---\\n'.

    Args:
        text: Input text to split.
        chunk_size: Maximum number of words per chunk (default: 200).

    Returns:
        Chunks separated by '\\n---CHUNK---\\n' for downstream processing.
    """
    words = text.split()
    if not words:
        return ""
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return '\n---CHUNK---\n'.join(chunks)


class ToolRegistry:
    """In-process registry for deterministic tools used in workflow steps."""

    def __init__(self):
        self._tools: dict[str, callable] = {}
        # Register built-in tools
        self.register(TOOL_CHUNK_TEXT, chunk_text)

    def register(self, name: str, fn: callable):
        self._tools[name] = fn

    def dispatch(self, name: str, context: str) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return f"<ERROR: tool '{name}' not registered>"
        try:
            result = fn(context)
            return str(result)
        except Exception as e:
            return f"<ERROR: {e}>"


# ── Workflow Engine ──────────────────────────────────────────────────────────

class WorkflowEngine:
    """Executes a multi-step workflow for a given Cell.

    Usage:
        engine = WorkflowEngine(llm_provider, tool_registry)
        result = engine.run(cell, user_prompt)
        # result = {"final_answer": "...", "artifacts": {...}, "step_results": [...]}
    """

    def __init__(self, llm_infer_fn: callable, tool_registry: Optional[ToolRegistry] = None):
        """
        Args:
            llm_infer_fn: Callable(messages, max_tokens, temperature) -> str
                The function that performs a single LLM inference call.
            tool_registry: Optional ToolRegistry for tool steps.
        """
        self._infer = llm_infer_fn
        self._tool_registry = tool_registry or ToolRegistry()

    def run(self, cell: Cell, prompt: str) -> dict[str, Any]:
        """Execute a workflow cell's steps against a user prompt.

        Returns:
            dict with:
                - final_answer: str (output of the last step)
                - artifacts: dict[str, str] (all step outputs keyed by name)
                - step_results: list[dict] (per-step metrics)
                - total_latency: float
                - total_tokens_estimate: int
        """
        steps = cell.steps
        if not steps:
            # Single-shot fallback
            messages = [
                {"role": "system", "content": cell.system_prompt},
                {"role": "user", "content": prompt},
            ]
            t0 = time.time()
            result = self._infer(messages, cell.decoding.max_tokens, cell.decoding.temperature)
            elapsed = time.time() - t0
            return {
                "final_answer": result,
                "artifacts": {"_input": prompt, "output": result},
                "step_results": [{
                    "step": "single", "latency_ms": round(elapsed * 1000, 1), "output": result,
                }],
                "total_latency": round(elapsed * 1000, 1),
                "total_tokens_estimate": len(result.split()),
            }

        # Multi-step workflow
        artifacts: dict[str, str] = {"_input": prompt}
        step_results = []

        for step_idx, step in enumerate(steps):
            # Resolve model and decoding for this step
            step_model_key = step.model_key or cell.model_key
            step_decoding = step.get_decoding(cell.decoding)

            # Get input text
            input_key = step.input_from or "_input"
            input_text = artifacts.get(input_key, prompt)

            step_start = time.time()

            if step.tool:
                # Deterministic tool step
                result = self._tool_registry.dispatch(step.tool, input_text)
            else:
                # LLM inference step
                messages = build_step_messages(step, input_text, artifacts)
                result = self._infer(messages, step_decoding.max_tokens, step_decoding.temperature)

            step_elapsed = time.time() - step_start

            # Store artifact
            artifacts[step.name] = result
            step_results.append({
                "step": step.name,
                "model": step_model_key,
                "latency_ms": round(step_elapsed * 1000, 1),
                "tokens_est": len(result.split()),
                "output": result[:200] + "..." if len(result) > 200 else result,
            })

        # Final answer = last step's output
        final_answer = artifacts[steps[-1].name]

        # Try to extract from \boxed{} if present
        boxed = extract_boxed_answer(final_answer)
        if boxed:
            final_answer = boxed

        total_latency = sum(s["latency_ms"] for s in step_results)
        total_tokens = sum(s["tokens_est"] for s in step_results)

        return {
            "final_answer": final_answer,
            "artifacts": artifacts,
            "step_results": step_results,
            "total_latency": round(total_latency, 1),
            "total_tokens_estimate": total_tokens,
        }
