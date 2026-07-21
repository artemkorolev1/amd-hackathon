"""Solvers package — deterministic tools for the 8-way prompt classifier pipeline.

Exports all solver functions and the ToolRegistry with all registered tools.
Classifiers live in a separate module (agent/classifiers/ or agent/ml_classifier.py).

NOTE: Heavy imports (spaCy, VADER, Sumy, torch) are lazy — they load on first
call, not at module import time. This avoids 2+ second cold starts.
"""
# Sub-modules (no eager imports from deterministic.py due to spacy→torch chain)
from agent.solvers import logic_solver
from agent.solvers import code_sandbox
from agent.solvers import easter_egg_shelf
from agent.solvers import spell_check
from agent.solvers import web_search

# Re-export functions from lightweight modules
from agent.solvers.logic_solver import solve_logic_puzzle, solve_syllogism
from agent.solvers.code_sandbox import execute_code_safe
from agent.solvers.spell_check import spell_check, list_misspellings
from agent.solvers.web_search import search_web, search_factual

# NOTE: solve_truth_teller_liar and solve_number_sequence live in
# deterministic.py but are NOT eagerly imported here due to spacy/torch.
# Use: from agent.solvers.deterministic import solve_truth_teller_liar
# or access via ToolRegistry (which uses lazy imports).

__all__ = [
    # Logic
    "solve_logic_puzzle",
    "solve_syllogism",
    # Code
    "execute_code_safe",
    # Spell
    "spell_check",
    "list_misspellings",
    # Web search
    "search_web",
    "search_factual",
    # Easter egg
    "format_csv",
    "text_stats",
    "reverse_text",
    "top_words",
    "to_leetspeak",
    "is_palindrome",
    "days_until_april_fools",
    "weather_hot_take",
    "to_emoji",
    "flip_coin",
]
