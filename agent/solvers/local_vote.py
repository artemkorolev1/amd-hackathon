"""
Self-consistency voting — plug-and-play module.

Accepts a pre-loaded llama-cpp-python Llama instance so the caller
(harness.py) owns model lifecycle.  No singleton, no double-load risk.

Usage inside harness.py::

    from agent.solvers.local_vote import solve_with_consensus

    result = solve_with_consensus(
        llm=llm,
        prompt=prompt,
        category=category,
        system_prompt=sys_prompt,
        k=CONSENSUS_SAMPLES,
        max_tokens=max_tok,
        timeout_per_sample=10.0,
    )
"""

import concurrent.futures
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Think-block stripper (mirrors harness._strip_think) ──────────────────────
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning blocks from model output."""
    stripped = _THINK_RE.sub("", text).strip()
    if stripped != text.strip():
        return stripped
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


# ── Category-aware answer normalisation (for vote counting) ─────────────────
def normalize_answer(category: str, text: str) -> str:
    """Canonical form for vote counting — category-aware.

    Returns a normalised string.  The majority vote is determined on these
    normalised forms, but the caller receives the **raw** answer text whose
    normalised form matches the majority.
    """
    text = text.strip()
    if not text:
        return ""

    cat = category.lower()

    # MATH — extract last numeric value
    if cat == "math":
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if nums:
            num = float(nums[-1])
            return str(int(num)) if num == int(num) else str(num)
        return text.lower()

    # SENTIMENT — map to canonical label
    if cat == "sentiment":
        lower = text.lower()
        if "positive" in lower:
            return "positive"
        if "negative" in lower:
            return "negative"
        if "neutral" in lower:
            return "neutral"
        return lower

    # NER — extract proper nouns, sort, join with semicolons
    if cat == "ner":
        entities = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
        if entities:
            return "; ".join(sorted(entities))
        return text.lower()

    # CODE — raw (no normalisation, agreement is unlikely)
    if cat in ("code_debug", "code_gen", "code"):
        return text

    # LOGIC / FACTUAL — strip punctuation, lowercase
    if cat in ("logic", "factual"):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    # SUMMARISATION — raw (never voted on)
    if cat == "summarization":
        return text

    # Default fallback
    return text.lower().strip()


# ── Self-consistency voting ──────────────────────────────────────────────────
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def solve_with_consensus(
    llm: Any,
    prompt: str,
    category: str,
    system_prompt: str,
    k: int = 3,
    max_tokens: int = 512,
    timeout_per_sample: float = 30.0,
    prompt_variants: Optional[List[str]] = None,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.0,
    repeat_penalty: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample the model *k* times, normalise answers, and vote.

    ``llm`` must be a pre-loaded ``llama_cpp.Llama`` instance.
    Samples run sequentially (llama-cpp-python is NOT thread-safe).

    Args:
        llm: Pre-loaded Llama instance (caller owns lifecycle).
        prompt: The user task prompt.
        category: Task category (used for answer normalisation).
        system_prompt: System-level instructions.
        k: Number of samples (default 3).
        max_tokens: Max tokens per completion.
        timeout_per_sample: Wall-clock timeout per sample.
        prompt_variants: Optional list of alternative system prompts
            (cycled round-robin across samples for prompt diversity).

    Returns:
        dict with keys:
          - majority_answer (str):    Raw text of the majority-consensus answer.
          - agreement_score (float):  Fraction of samples agreeing (0.0-1.0).
          - all_answers (list[str]):  All raw sample outputs.
          - samples (list[str]):      Normalised versions of each sample.
    """
    samples: List[str] = []
    prompts_used: List[str] = []

    for i in range(k):
        temperature = 0.1 if i == 0 else 0.7

        if prompt_variants:
            variant = prompt_variants[i % len(prompt_variants)]
            msgs = [
                {"role": "system", "content": variant},
                {"role": "user", "content": prompt},
            ]
            prompts_used.append(variant[:60])
        else:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            prompts_used.append(system_prompt[:60])

        # Single-threaded call with timeout — harness's _executor pattern
        def _call():
            return llm.create_chat_completion(
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )

        try:
            future = _executor.submit(_call)
            resp = future.result(timeout=timeout_per_sample)
            content = resp["choices"][0]["message"]["content"] or ""
            content = _strip_think(content)
            samples.append(content)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Consensus sample %d/%d timed out after %.0fs", i + 1, k, timeout_per_sample
            )
        except Exception as exc:
            logger.warning("Consensus sample %d/%d failed: %s", i + 1, k, exc)

    if not samples:
        return {
            "majority_answer": "",
            "agreement_score": 0.0,
            "all_answers": [],
            "samples": [],
            "prompts_used": prompts_used,
        }

    normalised = [normalize_answer(category, s) for s in samples]
    counter = Counter(normalised)
    majority_norm, count = counter.most_common(1)[0]
    agreement_score = count / len(samples)

    # Return the FIRST raw sample whose normalised form matches the majority
    majority_answer = ""
    for raw, norm in zip(samples, normalised):
        if norm == majority_norm:
            majority_answer = raw
            break

    return {
        "majority_answer": majority_answer,
        "agreement_score": agreement_score,
        "all_answers": samples,
        "samples": normalised,
    }
