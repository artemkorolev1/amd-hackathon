"""
Summarization Module for Sub-1B LLMs — Reroutes headline generation to Fireworks API.

The 1B models (Qwen2.5-1.5B, Gemma-3-1B, SmolLM2-1.7B, Llama-3.2-1B) 
score 0% on xsum headline generation across all prompt variants tried.
This module provides a Fireworks API fallback for the summarization category.

Usage:
    from agent.summarization import SummarizationRouter
    router = SummarizationRouter()
    answer = router.solve(prompt)  # prompt starts with "Summarize: ..."
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def is_headline_task(prompt: str) -> bool:
    """Detect if this is a headline-style summarization task (xsum format).
    
    Checks for:
    - "Summarize:" prefix (by far the most common in training-v3.json)
    - BBC-style article body patterns
    """
    prompt_stripped = prompt.strip()
    if prompt_stripped.startswith("Summarize:"):
        return True
    # Detect xsum-style articles: ~300 chars, date+news patterns
    if re.match(r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', 
                prompt_stripped, re.IGNORECASE):
        return True
    # Detect article-like content with news structure
    if (re.search(r'Last updated at \d{2}:\d{2}', prompt_stripped) or
        re.search(r'Media playback is unsupported', prompt_stripped)):
        return True
    return False


class SummarizationRouter:
    """Routes headline-generation summarization tasks to the best available solver.
    
    Strategy:
    1. If FIREWORKS_API_KEY is set and the task is headline-style → use Fireworks API
       with kimi-k2p7-code model
    2. Otherwise → fall through to the local LLM (which may score 0%)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self._fw_solver = None
        
    @property
    def fw_solver(self):
        if self._fw_solver is None and self.api_key:
            from agent.solvers.fireworks import FireworksSolver
            self._fw_solver = FireworksSolver(api_key=self.api_key)
        return self._fw_solver
    
    def solve_headline(self, prompt: str) -> Optional[str]:
        """Generate a headline for the given prompt using the API.
        
        Returns None if no API key is set or the call fails.
        """
        if not self.fw_solver:
            logger.warning("No Fireworks API key set — summarization will score 0%")
            return None
        
        # Best system prompt for kimi-k2p7-code (reasoning model)
        system_prompt = (
            "You write BBC-style news headlines. "
            "Given a news article, output a single headline sentence (8-20 words) "
            "that captures the core event. Use exact names, numbers, and places. "
            "Do NOT start with 'Headline:' or any prefix. "
            "Do NOT include quotes around the headline. "
            "Just output the bare headline text."
        )
        
        # Clean up the prompt: remove the "Summarize:" prefix if present
        user_prompt = prompt
        if prompt.strip().startswith("Summarize:"):
            user_prompt = prompt.strip()[len("Summarize:"):].strip()
            user_prompt = f"Article: {user_prompt}"
        
        try:
            answer = self.fw_solver.solve(
                model="accounts/fireworks/models/kimi-k2p7-code",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=80,
                temperature=0.0,
                task_type="summarization",
                timeout=29,
            )
            
            # Clean the response
            answer = answer.strip().strip('"').strip("'")
            if answer.startswith("Headline:") or answer.startswith("HEADLINE:"):
                answer = answer.split(":", 1)[1].strip()
            if answer.startswith("Here") or answer.startswith("The article"):
                answer = _extract_first_sentence(answer)
            
            logger.info(f"Fireworks headline: {answer[:80]}...")
            return answer
            
        except Exception as exc:
            logger.error(f"Fireworks summarization failed: {exc}")
            return None
    
    def solve(self, prompt: str, local_fallback=None) -> str:
        """Main entry point. Returns a headline or falls through to local model."""
        if is_headline_task(prompt):
            fw_answer = self.solve_headline(prompt)
            if fw_answer:
                return fw_answer
            logger.warning("Fireworks summarization returned empty — falling through to local model")
        # Not a headline task, or no API — pass through to local model
        if local_fallback:
            return local_fallback(prompt)
        return ""


def _extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences[0] if sentences else text


# ─── Alternative approach: two-step extractive (for testing without API) ───

def extractive_headline(prompt: str, llm) -> str:
    """Two-step extractive headline generation using a local model.
    
    Step 1: Extract WHO, WHAT, WHERE from article
    Step 2: Format into headline template
    
    This is a research approach — accuracy is expected to be ~20-40% at best.
    """
    article_text = prompt
    if prompt.strip().startswith("Summarize:"):
        article_text = prompt.strip()[len("Summarize:"):].strip()
    
    # Step 1: Extract structured info
    step1_sys = (
        "Extract from the news article in this exact format:\n"
        "WHO: [main person, group, or organization]\n"
        "WHAT: [the single most important action or event]\n"
        "WHERE: [location]\n"
        "Output ONLY these three fields. No explanation."
    )
    
    messages = [
        {"role": "system", "content": step1_sys},
        {"role": "user", "content": article_text},
    ]
    
    r = llm.create_chat_completion(messages=messages, max_tokens=100, temperature=0.0)
    extracted = r["choices"][0]["message"]["content"].strip()
    
    # Parse extracted fields
    who = _extract_field(extracted, "WHO")
    what = _extract_field(extracted, "WHAT")
    where = _extract_field(extracted, "WHERE")
    
    # Step 2: Format as headline
    step2_sys = (
        "Write a news headline in this format:\n"
        '"WHO has WHAT" or "WHO has WHAT, WHERE"\n\n'
        "Example:\n"
        "WHO: Peterborough United defender Miles Addison\n"
        "WHAT: signed a new one-month contract\n"
        "WHERE: with the League One side\n"
        '→ Headline: Peterborough United defender Miles Addison has signed a new one-month contract with the League One side.\n\n'
        "Now write the headline. Output ONLY the headline, one sentence."
    )
    
    step2_input = f"WHO: {who}\nWHAT: {what}\nWHERE: {where}"
    
    messages = [
        {"role": "system", "content": step2_sys},
        {"role": "user", "content": step2_input},
    ]
    
    r = llm.create_chat_completion(messages=messages, max_tokens=80, temperature=0.0)
    headline = r["choices"][0]["message"]["content"].strip()
    
    return headline


def _extract_field(text: str, field: str) -> str:
    """Extract a field value from structured output like 'WHO: value'."""
    m = re.search(rf'{field}:\s*(.+?)(?:\n|$)', text)
    return m.group(1).strip() if m else ""
