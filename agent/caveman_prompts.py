""""
CAVEMAN-MODE per-category prompts. 🪨

Same answers, ~50-65% fewer output tokens. Brain still big. Mouth small.
Removes filler, preamble, hedge words, restated questions, closing remarks.
Forces the model to say ONLY what's needed.

Based on JuliusBrussee/caveman schema and act2-agent's tight per-category format.
"""
from typing import Optional

# ── FIREWORKS SYSTEM PROMPTS (caveman terse) ────────────────────────────────
# Each line is a direct instruction on format. No extra fluff.
# Tokens here cost ~5-15 (input side) but save 50-200 completion tokens.
SYSTEM_PROMPTS: dict[str, str] = {
    "code_debug":
        "Point out the bug in 1 sentence. Fix in fenced block. No commentary after.",
    "code_gen":
        "Write the code. Fenced block. No explanation.",

    "math":
        "Output ONLY the number. No units. No explanation. Just the numeric answer.",
    "logic":
        "Output ONLY the answer. One word or letter. No explanation. No reasoning steps.",

    "sentiment":
        "Output EXACTLY one word: Positive, Negative, or Neutral. No explanation.",
    "ner":
        "Entities: Person=..., Org=..., Loc=..., Date=... No prose.",

    "summarization":
        "Only the summary obeying prompt's length constraint. No intro.",

    "factual":
        "Answer directly. No preamble. No restate question. No closing.",
    "general":
        "Answer directly. No filler.",

    "default":
        "Answer concise. No filler.",
}

# ── LOCAL MODEL TEMPLATES (equally terse) ────────────────────────────────────
# Shorter labels = less to generate. Model already sees the task in context.
TEMPLATES: dict[str, str] = {
    "summarization":            "Summarize:\n{task}",
    "sentiment":                "Sentiment:\n{task}",
    "ner":                      "Entities:\n{task}",
    "code_gen":                 "Code:\n{task}",
    "code_debug":               "Fix:\n{task}",
    "math":                     "Calc:\n{task}",
    "logic":                    "Logic:\n{task}",
    "factual":                  "Answer:\n{task}",
    "general":                  "{task}",
}


# ── Anti-preamble suffix appended to EVERY system prompt ───────────────────
# Deepseek models (v4-flash, v4-pro) and kimi-k2p6 love to generate
# "We need to...", "The user wants...", "I will..." before the actual answer.
# This suffix is a final, non-negotiable instruction that the first token
# must be the answer itself — no greetings, no meta, no self-talk.
_ANTI_PREAMBLE_SUFFIX = (
    " OUTPUT EXACTLY the answer. No preamble, no 'we', no 'the user', "
    "no meta-commentary, no self-talk. First word = the answer."
)


def get_system_prompt(category: str) -> str:
    """Get a caveman-terse system prompt for the category."""
    base = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["default"])
    return base + _ANTI_PREAMBLE_SUFFIX


def get_prompt(category: str, task: str) -> str:
    """Build the prompt. Uses shortened template if available."""
    template = TEMPLATES.get(category, "{task}")
    return template.format(task=task.rstrip())
