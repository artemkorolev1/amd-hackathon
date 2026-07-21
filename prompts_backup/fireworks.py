"""
Fireworks API solver: calls the Fireworks inference API.

Uses stdlib urllib only (no external deps like httpx or requests).
Pattern taken from top competitors (shankerram3, MarcusVinicius) who
use pure stdlib to minimize image size.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from agent.config import FIREWORKS_BASE_URL

# Models that support reasoning_effort (only minimax/gpt-oss per Fireworks docs)
REASONING_EFFORT_MODELS = {"minimax", "gpt-oss", "gptoss"}
# Kimi supports reasoning_effort too — use "none" for tasks that don't need chain-of-thought
REASONING_EFFORT_NONE_TASK_TYPES = {"ner", "sentiment", "code", "summarization"}
from agent.answer_cleaner import clean_response

logger = logging.getLogger(__name__)


def qualify_model_id(raw: str) -> str:
    """Add accounts/fireworks/models/ prefix if the id is bare.

    Fireworks serving ids are account-scoped. Bare ids (as published
    in the hackathon announcement) get the standard prefix; already-qualified
    ids pass through.
    Matches source2destination/act2-agent Qualify() pattern.
    """
    return raw if "/" in raw else "accounts/fireworks/models/" + raw


class FireworksSolver:
    """Solver that calls Fireworks AI API via stdlib urllib."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        if not self.api_key:
            logger.warning("FIREWORKS_API_KEY not set")

    def _reasoning_effort(self, model: str) -> Optional[str]:
        """Determine reasoning_effort parameter for this model.

        Only send for models that accept it. Sending to gemma/deepseek/kimi
        causes them to reject the call entirely (act2-agent discovered this).
        Returns None (don't send), "low", or "none".
        """
        model_lower = model.lower()
        for family in REASONING_EFFORT_MODELS:
            if family in model_lower:
                # minimax-m3 burns tokens on hidden reasoning if not suppressed
                return "none" if "minimax" in model_lower else "low"
        return None

    def solve(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: int = 29,
        prefill: str = "",
        task_type: str = "general",
        det_hint: Optional[str] = None,
        conflicting_answers: Optional[list] = None,
    ) -> str:
        """
        Call a Fireworks model and return the cleaned response text.

        Uses urllib (stdlib) - no external HTTP dependencies.
        Disables thinking mode on supported models to save tokens.
        If prefill is set, appends an assistant message to force the model
        to continue from that text (useful for code generation).

        If det_hint is provided, prepends "HINT: {det_hint}\\n\\n" to system_prompt.
        If conflicting_answers is provided, appends the conflicting-answers note
        to system_prompt.
        """
        # Merge det_hint and conflicting_answers into system_prompt
        if det_hint:
            system_prompt = f"HINT: {det_hint}\n\n{system_prompt}"
        if conflicting_answers:
            conflict_text = "; ".join(conflicting_answers)
            system_prompt = f"{system_prompt}\n\nLocal model could not agree. Conflicting answers: {conflict_text}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Pre-fill assistant response (forces model to continue from this text)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        payload = {
            "model": qualify_model_id(model),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Conditional reasoning suppression: only send for models that accept it
        reff = self._reasoning_effort(model)
        # Override: for kimi, use "none" for non-reasoning task types
        if task_type in REASONING_EFFORT_NONE_TASK_TYPES and "kimi" in model.lower():
            reff = "none"
        elif reff is None and "kimi" in model.lower():
            reff = None  # leave thinking enabled for math/logic/factual
        if reff is not None:
            payload["reasoning_effort"] = reff

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{FIREWORKS_BASE_URL}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.error(f"Fireworks HTTP {e.code}: {e.read().decode()[:300]}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"Fireworks connection error: {e.reason}")
            raise
        except Exception as e:
            logger.error(f"Fireworks call failed: {e}")
            raise

        content = data["choices"][0]["message"]["content"].strip()
        # Prepend prefill if we used one and the model didn't already include it
        if prefill and not content.startswith(prefill.strip()):
            content = prefill + content

        # Clean response (strip preambles, code fence artifacts)
        return clean_response(content)
