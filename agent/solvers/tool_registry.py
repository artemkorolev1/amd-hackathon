"""
Lightweight deterministic tool registry.
Pattern stolen from smolagents @tool decorator, zero external dependencies.
"""
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, get_type_hints


@dataclass
class Tool:
    name: str
    description: str
    inputs: Dict[str, Dict[str, str]]
    output_type: str
    fn: Callable = field(repr=False)
    category: str = ""  # Which of the 8 categories this tool serves
    is_deterministic: bool = True
    fallback_response: Optional[str] = None
    timeout_seconds: int = 30

    def __call__(self, **kwargs) -> Any:
        try:
            return {"status": "success", "data": self.fn(**kwargs), "tool": self.name}
        except Exception as e:
            if self.fallback_response:
                return {"status": "fallback", "data": self.fallback_response, "tool": self.name}
            return {"status": "error", "error": str(e), "tool": self.name}


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "",
    is_deterministic: bool = True,
    fallback_response: Optional[str] = None,
    timeout_seconds: int = 30,
) -> Tool:
    """
    Decorator that wraps a function into a Tool with auto-inferred schema.
    
    Usage:
        @tool
        def my_tool(x: int) -> str: ...
        
        @tool(name="custom_name", category="math")
        def my_tool(x: int) -> str: ...
    """
    def _make_tool(fn: Callable) -> Tool:
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        
        # Tool name from function name or override
        t_name = name or fn.__name__
        
        # Description from docstring first line
        desc_lines = doc.strip().split("\n")
        t_desc = description or (desc_lines[0].strip() if desc_lines else "")
        
        # Inputs from function signature + docstring
        inputs = {}
        for p_name, param in sig.parameters.items():
            typ = hints.get(p_name, str).__name__
            inputs[p_name] = {
                "type": typ,
                "description": f"Parameter {p_name}",
            }
            # Try to extract description from docstring
            for line in desc_lines:
                stripped = line.strip()
                if stripped.startswith(f"{p_name}:") or stripped.startswith(f"{p_name} "):
                    inputs[p_name]["description"] = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
                    break
        
        output_type = hints.get("return", str).__name__
        
        return Tool(
            name=t_name,
            description=t_desc,
            inputs=inputs,
            output_type=output_type,
            fn=fn,
            category=category,
            is_deterministic=is_deterministic,
            fallback_response=fallback_response,
            timeout_seconds=timeout_seconds,
        )
    
    if func is not None:
        return _make_tool(func)
    return _make_tool


class ToolRegistry:
    """Central registry for all pipeline tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, t: Tool):
        self._tools[t.name] = t
    
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)
    
    def get_by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]
    
    def list_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "deterministic": t.is_deterministic,
                "inputs": t.inputs,
                "output_type": t.output_type,
            }
            for t in self._tools.values()
        ]
    
    def __len__(self):
        return len(self._tools)


# Create global registry
registry = ToolRegistry()


# ── Register tools (existing) ──

@tool(name="factual_qa", category="factual", fallback_response="I don't have this fact")
def factual_qa(question: str) -> str:
    """Answer factual questions using the FTS5 database."""
    from agent.solvers.fact_db import FactDB
    db = FactDB()
    results = db.query(question, k=1)
    if results and results[0][0] >= -2.0:
        return results[0][2]
    return "I don't know"

@tool(name="sentiment_analysis", category="sentiment", is_deterministic=True)
def sentiment_analysis(text: str) -> str:
    """Classify text sentiment as positive, negative, or neutral."""
    from agent.solvers.deterministic import solve_sentiment
    result = solve_sentiment(text, "sentiment")
    return result or "neutral"

@tool(name="summarize", category="summarization", is_deterministic=True)
def summarize(text: str) -> str:
    """Extractive summarization using Sumy LexRank."""
    from agent.solvers.deterministic import solve_summarization
    result = solve_summarization(text, "summarization")
    return result or "Could not summarize"

@tool(name="math_solve", category="math", is_deterministic=True)
def math_solve(expression: str) -> str:
    """Solve a mathematical expression or equation."""
    from agent.solvers.deterministic import solve_arithmetic
    result = solve_arithmetic(expression, "math_arithmetic")
    return result or "Could not solve"

@tool(name="ner_extract", category="ner", is_deterministic=True)
def ner_extract(text: str) -> str:
    """Extract named entities using spaCy NER."""
    from agent.solvers.deterministic import solve_ner
    result = solve_ner(text, "ner")
    return result or "No entities found"

@tool(name="format_python", category="code", is_deterministic=True)
def format_python(code: str) -> str:
    """Format and lint Python code."""
    from agent.solvers.verify import format_and_lint
    result = format_and_lint(code)
    return result.get("formatted", code)


# ── Register tools (Logic Solver) ──

@tool(name="solve_logic_puzzle", category="logic", is_deterministic=True)
def tool_solve_logic_puzzle(prompt: str) -> str:
    """Solve constraint-based logic puzzles (seating, ordering, attribute matching, scheduling)."""
    from agent.solvers.logic_solver import solve_logic_puzzle
    result = solve_logic_puzzle(prompt)
    return result or "Could not solve this logic puzzle"

@tool(name="solve_syllogism", category="logic", is_deterministic=True)
def tool_solve_syllogism(premises: str) -> str:
    """Solve categorical syllogisms. Pass premises as a list of statements separated by newlines."""
    premises_list = [p.strip() for p in premises.split("\n") if p.strip()]
    from agent.solvers.logic_solver import solve_syllogism
    result = solve_syllogism(premises_list)
    return result or "Could not derive a conclusion"

@tool(name="solve_truth_teller_liar", category="logic", is_deterministic=True)
def tool_solve_truth_teller_liar(prompt: str) -> str:
    """Solve truth-teller/liar puzzles (knights and knaves). Each person is either a knight (always tells truth) or knave (always lies)."""
    from agent.solvers.deterministic import solve_truth_teller_liar
    result = solve_truth_teller_liar(prompt)
    return result or "Could not solve this truth-teller/liar puzzle"

@tool(name="solve_number_sequence", category="logic", is_deterministic=True)
def tool_solve_number_sequence(prompt: str) -> str:
    """Solve number/letter sequence puzzles (arithmetic, geometric, fibonacci-like, squares, alternating)."""
    from agent.solvers.deterministic import solve_number_sequence
    result = solve_number_sequence(prompt)
    return result or "Could not determine the next in sequence"

@tool(name="solve_logical_reasoning", category="logic", is_deterministic=True)
def tool_solve_logical_reasoning(prompt: str) -> str:
    """Solve LSAT-style logical reasoning questions (strengthen, weaken, assumption, inference, flaw, main point, explain)."""
    from agent.solvers.logic_reasoning import solve_logical_reasoning
    result = solve_logical_reasoning(prompt, "logic")
    return result or "Could not classify this logical reasoning question"


# ── Register tools (Code Sandbox) ──

@tool(name="execute_code_safe", category="code", is_deterministic=True)
def tool_execute_code_safe(code: str, timeout: int = 10) -> str:
    """Execute Python code safely in a sandbox with RestrictedPython."""
    from agent.solvers.code_sandbox import execute_code_safe
    result = execute_code_safe(code, timeout=timeout)
    return json.dumps(result)

@tool(name="code_debug", category="code", is_deterministic=True)
def tool_code_debug(task: str) -> str:
    """Fix common bugs in Python code (off-by-one, product init, assignment vs comparison, etc.)."""
    from agent.solvers.deterministic import solve_code_debugging
    result = solve_code_debugging(task, "code_debug")
    return result or "No bug fixes applied"

@tool(name="code_gen_templates", category="code", is_deterministic=True)
def tool_code_gen_templates(task: str) -> str:
    """Generate Python code for common patterns (two-sum, palindrome, fizzbuzz, fibonacci, etc.)."""
    from agent.solvers.deterministic import solve_code_generation
    result = solve_code_generation(task, "code_gen")
    return result or "No matching code template found"


# ── Register tools (Spell Check) ──

@tool(name="spell_check", category="factual", is_deterministic=True)
def tool_spell_check(text: str, max_edit: int = 2) -> str:
    """Check and correct spelling in text using SymSpell. Returns corrected text."""
    from agent.solvers.spell_check import spell_check
    result = spell_check(text, max_edit=max_edit)
    return result

@tool(name="list_misspellings", category="factual", is_deterministic=True)
def tool_list_misspellings(text: str, max_edit: int = 1) -> str:
    """List potentially misspelled words with suggestions. Does NOT auto-correct."""
    from agent.solvers.spell_check import list_misspellings
    import json
    result = list_misspellings(text, max_edit=max_edit)
    return json.dumps(result, indent=2)


# ── Register tools (Web Search) ──

@tool(name="search_web", category="factual", fallback_response="Web search unavailable")
def tool_search_web(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns snippet-style results with titles and URLs."""
    from agent.solvers.web_search import search_web
    result = search_web(query, max_results=max_results)
    return result

@tool(name="search_factual", category="factual", fallback_response="Factual search unavailable")
def tool_search_factual(question: str, max_results: int = 3) -> str:
    """Answer a factual question by searching the web. Prefers Wikipedia, falls back to general search."""
    from agent.solvers.web_search import search_factual
    result = search_factual(question, max_results=max_results)
    return result


# ── Register tools (Easter Egg Shelf — category="fun") ──

@tool(name="format_csv", category="fun", is_deterministic=True)
def tool_format_csv(text: str) -> str:
    """Pretty-print CSV data with aligned columns."""
    from agent.solvers.easter_egg_shelf import format_csv
    return format_csv(text)

@tool(name="text_stats", category="fun", is_deterministic=True)
def tool_text_stats(text: str) -> str:
    """Return quirky text statistics: reading time, grade level, most common words."""
    from agent.solvers.easter_egg_shelf import text_stats
    result = text_stats(text)
    return json.dumps(result)

@tool(name="reverse_text", category="fun", is_deterministic=True)
def tool_reverse_text(text: str) -> str:
    """Reverse the input text (just for fun)."""
    from agent.solvers.easter_egg_shelf import reverse_text
    return reverse_text(text)

@tool(name="top_words", category="fun", is_deterministic=True)
def tool_top_words(text: str, n: int = 10) -> str:
    """Return the N most common words with frequencies (word cloud)."""
    from agent.solvers.easter_egg_shelf import top_words
    return top_words(text, n)

@tool(name="to_leetspeak", category="fun", is_deterministic=True)
def tool_to_leetspeak(text: str) -> str:
    """Convert text to leetspeak (e -> 3, a -> 4, etc.)."""
    from agent.solvers.easter_egg_shelf import to_leetspeak
    return to_leetspeak(text)

@tool(name="is_palindrome", category="fun", is_deterministic=True)
def tool_is_palindrome(text: str) -> str:
    """Check if text (cleaned) is a palindrome."""
    from agent.solvers.easter_egg_shelf import is_palindrome
    return str(is_palindrome(text))

@tool(name="days_until_april_fools", category="fun", is_deterministic=True)
def tool_days_until_april_fools() -> str:
    """Days until next April Fools' Day (fun counter)."""
    from agent.solvers.easter_egg_shelf import days_until_april_fools
    return str(days_until_april_fools())

@tool(name="weather_hot_take", category="fun", is_deterministic=True)
def tool_weather_hot_take(text: str = "20") -> str:
    """Given a temperature in Celsius, return a hot take. Defaults to 20°C if no temperature given."""
    try:
        temp_c = float(text)
    except (ValueError, TypeError):
        temp_c = 20.0
    from agent.solvers.easter_egg_shelf import weather_hot_take
    return weather_hot_take(temp_c)

@tool(name="to_emoji", category="fun", is_deterministic=True)
def tool_to_emoji(text: str) -> str:
    """Convert common words/phrases to emoji."""
    from agent.solvers.easter_egg_shelf import to_emoji
    return to_emoji(text)

@tool(name="flip_coin", category="fun", is_deterministic=True)
def tool_flip_coin() -> str:
    """Flip a virtual coin — Heads or Tails."""
    from agent.solvers.easter_egg_shelf import flip_coin
    return flip_coin()


# Register all tools
existing = [factual_qa, sentiment_analysis, summarize, math_solve, ner_extract, format_python]
logic = [tool_solve_logic_puzzle, tool_solve_syllogism, tool_solve_truth_teller_liar, tool_solve_number_sequence, tool_solve_logical_reasoning]
code = [tool_execute_code_safe, tool_code_debug, tool_code_gen_templates]
spell = [tool_spell_check, tool_list_misspellings]
web = [tool_search_web, tool_search_factual]
fun = [
    tool_format_csv, tool_text_stats, tool_reverse_text, tool_top_words,
    tool_to_leetspeak, tool_is_palindrome, tool_days_until_april_fools,
    tool_weather_hot_take, tool_to_emoji, tool_flip_coin,
]

all_tools = existing + logic + code + spell + web + fun

for t in all_tools:
    registry.register(t)

print(f"Registered {len(registry)} tools")
for t in registry.list_tools():
    print(f"  {t['name']:20s} [{t['category']:12s}] {t['description'][:50]}")
