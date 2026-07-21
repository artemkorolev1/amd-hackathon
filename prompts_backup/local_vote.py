"""
Self-consistency voting with direct GPU inference via llama-cpp-python.

Stateless module. No tool calling, no multi-round.
Model loaded once as module-level singleton (lazy, thread-safe).
"""
import logging
import os
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Model path resolution ──────────────────────────────────────────
# Priority: config.MODEL_PATH env var > project-relative > Docker default
MODEL_PATH: str = os.environ.get("MODEL_PATH", "")
if not MODEL_PATH:
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    _project_model = os.path.join(_project_root, "models", "nvidia-nemotron3-nano-4b-q4_k_m.gguf")
    if os.path.exists(_project_model):
        MODEL_PATH = _project_model
    else:
        MODEL_PATH = "/models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

# ── Lazy model singleton (thread-safe) ─────────────────────────────
_LLM: Optional[object] = None
_LLM_LOCK = threading.Lock()


def _get_llm() -> Optional[object]:
    """Load the Llama model once on first call (thread-safe lazy singleton).

    CPU-only (n_gpu_layers=0, threads=2) for grader's 2 vCPU / 4 GB RAM.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM

        try:
            from llama_cpp import Llama

            ngl = int(os.environ.get("N_GPU_LAYERS", "0"))
            logger.info(
                "Loading model from %s (n_gpu_layers=%d)",
                MODEL_PATH, ngl,
            )
            _LLM = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_gpu_layers=ngl,
                n_threads=4 if ngl == 0 else 2,
                verbose=False,
            )
            logger.info("Model loaded (n_gpu_layers=%d)", ngl)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            _LLM = None

    return _LLM


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Single chat completion via llama-cpp-python (direct GPU inference).

    Thread-safe: acquires the model lock for each call because
    llama-cpp-python is NOT safe for concurrent inference.

    Args:
        messages: OpenAI-format message list
            (e.g. [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}]).
        temperature: Sampling temperature (default 0.7).
        max_tokens: Maximum tokens in the response.

    Returns:
        Raw response text, or empty string on failure.
    """
    llm = _get_llm()
    if llm is None:
        return ""

    try:
        with _LLM_LOCK:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        content = response["choices"][0]["message"]["content"]
        return content.strip() if content else ""
    except Exception as exc:
        logger.warning("Inference failed: %s", exc)
        return ""


def normalize_answer(category: str, text: str) -> str:
    """Category-aware canonical form for vote counting.

    Normalization rules:
        MATH          → extract last number, canonical (6.0 → 6)
        SENTIMENT     → positive | negative | neutral
        NER           → sorted semicolon-separated list of proper nouns
        CODE          → raw text (agreement not expected)
        LOGIC/FACTUAL → strip punctuation and lowercase
        SUMMARIZATION → raw (threshold=0.0, never voted on)
        default       → lowercase + strip
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

    # CODE — raw (no normalization, agreement is unlikely)
    if cat in ("code_debug", "code_gen", "code"):
        return text

    # LOGIC / FACTUAL — strip punctuation, lowercase
    if cat in ("logic", "factual"):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    # SUMMARIZATION — raw (never voted on, threshold=0.0)
    if cat == "summarization":
        return text

    # Default fallback
    return text.lower().strip()


def solve_with_consensus(
    prompt: str,
    category: str,
    system_prompt: str,
    k: int = 3,
    max_tokens: int = 512,
    system_prompt_variants: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Sample the model *k* times, normalize answers, and vote.

    Strategy:
      - First sample at T=0.1 (greedy-ish, reproducible)
      - Remaining samples at T=0.7 (diverse exploration)
      - If *system_prompt_variants* is provided, each sample draws a different
        prompt variant (cycles round-robin) — this creates prompt diversity in
        addition to temperature diversity, making agreement a stronger signal.
      - All inferences run SEQUENTIALLY to avoid OOM on 8 GB VRAM.

    Args:
        prompt: The user task prompt.
        category: Task category (used for normalization).
        system_prompt: System-level instructions.
        k: Number of samples (default 3).
        max_tokens: Max tokens per completion.

    Returns:
        dict with keys:
          - majority_answer (str):   Raw text of the majority-consensus answer.
          - agreement_score (float): Fraction of samples agreeing (0.0‑1.0).
          - all_answers (list[str]): All raw sample outputs.
          - samples (list[str]):     Normalized versions of each sample.
    """
    messages_base = []
    if system_prompt:
        messages_base.append({"role": "system", "content": system_prompt})
    messages_base.append({"role": "user", "content": prompt})
    prompt_variants = system_prompt_variants or []

    samples: List[str] = []
    prompts_used: List[str] = []
    for i in range(k):
        temperature = 0.1 if i == 0 else 0.7
        if prompt_variants:
            variant = prompt_variants[i % len(prompt_variants)]
            messages = [
                {"role": "system", "content": variant},
                {"role": "user", "content": prompt},
            ]
            prompts_used.append(variant[:60])
        else:
            messages = messages_base
            prompts_used.append(system_prompt[:60])
        response = chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        if response:
            samples.append(response)

    if not samples:
        return {
            "majority_answer": "",
            "agreement_score": 0.0,
            "all_answers": [],
            "samples": [],
            "prompts_used": prompts_used,
        }

    normalized = [normalize_answer(category, s) for s in samples]
    counter = Counter(normalized)
    majority_norm, count = counter.most_common(1)[0]
    agreement_score = count / len(samples)

    # Find the first raw sample whose normalized form matches the majority
    majority_answer = ""
    for raw, norm in zip(samples, normalized):
        if norm == majority_norm:
            majority_answer = raw
            break

    return {
        "majority_answer": majority_answer,
        "agreement_score": agreement_score,
        "all_answers": samples,
        "samples": normalized,
    }
