"""
Parallel inference — sends concurrent POST requests to llama-server slots.

Each slot gets a different system prompt (same user question). Returns
4 answers in ~the time of 1 inference (server handles parallelism internally).
"""

import concurrent.futures
import json
import logging
import time
import urllib.request
import urllib.error

logger = logging.getLogger("container.inference")

CHAT_URL = "http://127.0.0.1:8081/v1/chat/completions"
REQUEST_TIMEOUT = 60.0


def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> str:
    """Single POST to llama-server /v1/chat/completions."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"] or ""
        return content.strip()
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, KeyError) as e:
        logger.warning("Inference error: %s", e)
        return ""


def parallel_infer(
    system_prompts: list[str],
    user_prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> list[str]:
    """
    Send one request per system prompt concurrently.
    Returns answers in same order as system_prompts.

    With llama-server --parallel N, requests are served by different slots.
    """
    if not system_prompts:
        return []

    answers: list[str | None] = [None] * len(system_prompts)

    def _do_infer(i: int) -> tuple[int, str]:
        ans = _chat_completion(
            system_prompts[i], user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return (i, ans)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(system_prompts)) as ex:
        futures = [ex.submit(_do_infer, i) for i in range(len(system_prompts))]
        for future in concurrent.futures.as_completed(futures):
            idx, ans = future.result()
            answers[idx] = ans

    # Fill any None with empty string
    return [a or "" for a in answers]


def simple_infer(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> str:
    """Single inference. Used for judge calls (one slot)."""
    return _chat_completion(system_prompt, user_prompt, max_tokens, temperature, stop)
