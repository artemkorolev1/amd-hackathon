"""
Fireworks API fallback — used when all 4 prompts produce degenerate/no answers.

Calls the Fireworks API with a simple prompt to get a last-resort answer.
"""

import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger("container.fallback")

FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1",
).strip("\"'").rstrip("/")
if FIREWORKS_BASE_URL.endswith("/chat/completions"):
    FIREWORKS_BASE_URL = FIREWORKS_BASE_URL.replace("/chat/completions", "")
CHAT_URL = f"{FIREWORKS_BASE_URL}/chat/completions"
API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FALLBACK_MODEL = os.environ.get(
    "FALLBACK_MODEL",
    "accounts/fireworks/models/kimi-k2p7-code",
)


def is_available() -> bool:
    """Check if Fireworks API is configured."""
    return bool(API_KEY)


def fallback_answer(
    question: str,
    category: str = "factual",
    timeout: int = 15,
) -> str | None:
    """Call Fireworks API for a single answer. Returns None on failure."""
    if not API_KEY:
        logger.warning("No FIREWORKS_API_KEY set — skipping fallback")
        return None

    system_prompt = (
        f"You are a {category} assistant. "
        "Answer the question directly and concisely. "
        "No preamble, no meta-commentary. "
        "First word = the answer."
    )

    body = {
        "model": FALLBACK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        CHAT_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"] or ""
        return content.strip()
    except Exception as e:
        logger.warning("Fireworks fallback failed: %s", e)
        return None
