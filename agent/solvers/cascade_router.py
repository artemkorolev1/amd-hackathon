"""
agent/solvers/cascade_router.py — Cascade determination classifier.

Routes each (category, prompt) pair to the optimal solver with cleaned input.
Performs:
  1. Input preprocessing (strip boilerplate, extract key info)
  2. Solver match scoring (determine best solver for this prompt)
  3. Input transformation (convert prompt to solver-expected format)
  4. Output post-processing (format match expected answer)
  5. Fallback to LLM when no solver scores sufficiently

Single-pass: The first solver with sufficient match score wins.
"""

import re
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Solver descriptors
# ═══════════════════════════════════════════════════════════════════════════

Solver = Callable[[str, str], Optional[str]]


class RouteEntry:
    """A single solver + its pre/post processing + match criteria."""

    def __init__(
        self,
        name: str,
        solver_fn: Solver,
        categories: set[str],
        match_score: Callable[[str], float],
        preprocess: Optional[Callable[[str], str]] = None,
        postprocess: Optional[Callable[[str, str], str]] = None,
        min_score: float = 0.3,
    ):
        self.name = name
        self.solver_fn = solver_fn
        self.categories = categories
        self.match_score = match_score  # returns 0.0-1.0 match confidence
        self.preprocess = preprocess or (lambda x: x)
        self.postprocess = postprocess or (lambda ans, prompt: ans)
        self.min_score = min_score


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

def strip_task_prefix(text: str) -> str:
    """Remove common task instruction prefixes like 'Write a Python function:'"""
    return re.sub(
        r'^(?:Write\s+(?:a\s+)?(?:Python\s+)?(?:function|program|script)'
        r'|Create\s+(?:a\s+)?(?:Python\s+)?(?:function|program|script)'
        r'|Implement\s+(?:a\s+)?(?:Python\s+)?(?:function|program|script)'
        r'|Define\s+(?:a\s+)?(?:Python\s+)?(?:function|program|script)'
        r'|Task:?|Problem:?)\s*[:\n]*',
        '', text, flags=re.IGNORECASE
    ).strip()


def extract_function_name(prompt: str) -> Optional[str]:
    """Extract the function name from a 'def name(...)' signature."""
    m = re.search(r'def\s+(\w+)\s*\(', prompt)
    return m.group(1) if m else None


def extract_function_signature(prompt: str) -> Optional[str]:
    """Extract the full def line from a prompt."""
    m = re.search(r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?:)', prompt)
    return m.group(1) if m else None


def strip_docstring(text: str) -> str:
    """Strip Python docstrings and multi-line comments."""
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
    return text.strip()


def extract_business_logic(prompt: str) -> str:
    """Extract the core logic description from a code gen prompt."""
    # Remove the def line and docstring, keep the description
    text = re.sub(r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?:', '', prompt)
    text = strip_docstring(text)
    # Remove example/test cases
    text = re.sub(r'>>>.*', '', text)
    text = re.sub(r'>>>\n.*?\n', '', text, flags=re.DOTALL)
    return text.strip()


def strip_squad_formatting(prompt: str) -> str:
    """Remove 'Context:', 'Question:' prefixes for factual QA."""
    return re.sub(
        r'^(?:Context|Passage|Question|Query|Q)\s*:\s*',
        '', prompt, flags=re.IGNORECASE | re.MULTILINE
    ).strip()


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for sentiment/NER."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing helpers
# ═══════════════════════════════════════════════════════════════════════════

def inject_function_name(answer: str, prompt: str) -> str:
    """Replace the solver's generic function name with the prompt's function name."""
    prompt_fname = extract_function_name(prompt)
    if not prompt_fname:
        return answer
    # Find the first def in the answer and replace the function name
    answer_fname = extract_function_name(answer)
    if answer_fname and answer_fname != prompt_fname:
        answer = answer.replace(f'def {answer_fname}(', f'def {prompt_fname}(', 1)
    return answer


def inject_signature(answer: str, prompt: str) -> str:
    """Replace the solver's def line with the prompt's exact def line."""
    prompt_sig = extract_function_signature(prompt)
    if not prompt_sig:
        return answer
    answer_sig = extract_function_signature(answer)
    if answer_sig and answer_sig != prompt_sig:
        answer = answer.replace(answer_sig, prompt_sig, 1)
    return answer


def normalize_ner_output(answer: str, prompt: str) -> str:
    """Normalize NER output to TYPE: entity per line format."""
    lines = []
    for line in answer.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Already in TYPE: entity format
        if re.match(r'^[A-Z]+\s*:\s*', line):
            lines.append(line)
        # Remove numbered prefixes like "1. " or "1) "
        else:
            line = re.sub(r'^\d+[.)]\s*', '', line).strip()
            if line:
                lines.append(line)
    return '\n'.join(lines)


def normalize_sentiment(answer: str, prompt: str) -> str:
    """Normalize sentiment to exact one-word label."""
    label = answer.strip().lower().rstrip('.,!?;')
    if label in ('positive', 'negative', 'neutral', 'mixed'):
        return label
    # If longer, try to extract the label word
    for w in ('positive', 'negative', 'neutral', 'mixed'):
        if w in label:
            return w
    return answer.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Per-solver match scorers
# ═══════════════════════════════════════════════════════════════════════════

# ── Code gen ──
_CODE_GEN_TEMPLATE_NAMES = {
    'freq_count', 'closest_num', 'is_pal', 'two_sum', 'reverse_str',
    'factorial', 'fib', 'merge_lists', 'fizzbuzz', 'count_vowels',
    'is_anagram', 'find_max_min', 'remove_duplicates', 'flatten_list',
    'is_prime', 'gcd', 'lcm', 'binary_search', 'linear_search',
    'bubble_sort', 'capitalize_words', 'word_count', 'is_even',
    'power', 'sum_digits', 'is_armstrong', 'is_perfect', 'transpose',
    'caesar_cipher', 'sort_dict_by_value', 'intersection', 'list_difference',
    'second_largest',
}


def match_code_gen_template(prompt: str) -> float:
    """Score code gen -> template match.

    Returns 1.0 if the function name matches a template EXACTLY.
    Returns 0.3-0.8 for keyword overlap but only if problem mentions
    a known algorithm by name in the prompt text.
    Returns 0.0 for HumanEval-style unique prompts.
    """
    fname = extract_function_name(prompt)
    if fname and fname in _CODE_GEN_TEMPLATE_NAMES:
        return 1.0  # Exact function name match — template will work

    lower = prompt.lower()
    # Check for explicit algorithm names
    if re.search(
        r'\b(fibonacci|palindrome|anagram|fizz.?buzz|'
        r'two.?sum|binary.?search|bubble.?sort|merge.?sort|'
        r'prime number|factorial|tower of hanoi|caesar.?cipher|'
        r'armstrong|perfect number|reverse string)\b',
        lower,
    ):
        return 0.6

    return 0.0  # HumanEval-style — no template match


def preprocess_code_gen(prompt: str) -> str:
    """Clean up code gen prompt for solver or LLM.

    If the prompt has a def line, extract it + the business logic,
    stripping docstrings and test cases.
    """
    fname = extract_function_name(prompt)
    if fname:
        sig = extract_function_signature(prompt)
        logic = extract_business_logic(prompt)
        if logic:
            return f'{sig}\n    """{logic}"""'
    return strip_task_prefix(prompt)


def postprocess_code_gen(answer: str, prompt: str) -> str:
    """Fix up code gen output — inject correct function name/signature."""
    answer = inject_signature(answer, prompt)
    answer = inject_function_name(answer, prompt)
    # Also fix function name references inside the body
    answer_fname = extract_function_name(answer)
    prompt_fname = extract_function_name(prompt)
    if answer_fname and prompt_fname and answer_fname != prompt_fname:
        # Replace all calls to the old function name inside the body
        answer = re.sub(
            rf'\b{re.escape(answer_fname)}\b(?!\s*=)',
            prompt_fname, answer,
        )
    return answer


# ── Factual QA ──
def match_factual(prompt: str) -> float:
    """Factual QA — always try FactDB first."""
    return 1.0


def preprocess_factual(prompt: str) -> str:
    """Strip SQuAD formatting for FactDB."""
    return strip_squad_formatting(prompt)


# ── Sentiment ──
_COMMON_ANALYSIS_WORDS = {'summarize', 'summary', 'review', 'analyse'}

def match_sentiment(prompt: str) -> float:
    """Sentiment — VADER handles most, but not analytical texts."""
    lower = prompt.lower()
    # If it's an analysis/review prompt, LLM is better
    if _COMMON_ANALYSIS_WORDS & set(re.findall(r'\w+', lower)):
        return 0.0
    return 0.8


def preprocess_sentiment(prompt: str) -> str:
    """Strip markdown formatting for VADER."""
    return strip_markdown(prompt)


# ── NER ──
def match_ner(prompt: str) -> float:
    """NER — route based on subtype detection."""
    if re.search(r'\{@|#\w+', prompt):
        return 0.7  # Tweet NER → old regex solver
    if re.search(r'(?:extract all (?:disease|gene|protein) names'
                 r'|biomedical text)', prompt, re.I):
        return 0.7  # Biomedical → old regex solver
    return 0.9  # General NER → old regex solver


def preprocess_ner(prompt: str) -> str:
    """Strip instruction boilerplate for NER solver."""
    return re.sub(
        r'^(?:Extract|Identify|Find|List|Tag|Locate|Retrieve|Annotate)\s+'
        r'(?:all\s+)?(?:the\s+)?(?:named\s+)?(?:entities?|names?|people|'
        r'persons?|organizations?|locations?|diseases?|genes?|proteins?)'
        r'.*?(?:from|in)\s+(?:the\s+)?(?:following\s+)?(?:text|sentence|tweet|passage)\s*:\s*',
        '', prompt, flags=re.IGNORECASE | re.DOTALL
    ).strip() or prompt  # Return original if nothing extracted


# ── Logic ──
def match_logic(prompt: str) -> float:
    """Logic — route to zebra solver for named-entity puzzles, else LLM."""
    # Named-entity constraint puzzle → zebra solver (high confidence)
    names = re.findall(r'\b[A-Z][a-z]+\b', prompt)
    has_constraints = bool(re.search(
        r'\b(each|different|distinct|adjacent|to the (left|right) of|between)\b',
        prompt, re.I,
    ))
    if len(names) >= 3 and has_constraints:
        return 0.6  # zebra puzzle
    # LogiQA / argument analysis → logical_reasoning
    if re.search(r'\b(which of the following|argument|conclusion|premise|'
                 r'weakens?|strengthens?|infer|deduce|syllogism)\b', prompt, re.I):
        return 0.5
    return 0.0


# ── Math ──
def match_math(prompt: str) -> float:
    """Math — only route to arithmetic solver for bare expressions."""
    # Pure arithmetic expression with =, +, -, *, /
    has_operators = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', prompt))
    is_short = len(prompt.split()) < 30
    no_narrative = not re.search(
        r'\b(if |then|how many|there are|each|total of|buy|sells?|cost|price|'
        r'distance|speed|time|age|work together|mixture)',
        prompt, re.I,
    )
    if has_operators and is_short and no_narrative:
        return 0.8
    return 0.0  # Word problems → LLM


# ═══════════════════════════════════════════════════════════════════════════
# Route registry
# ═══════════════════════════════════════════════════════════════════════════

from agent.solvers.deterministic import (
    solve_ner,
    solve_factual_qa as solve_factual_qa_fn,
    solve_sentiment as solve_sentiment_fn,
    solve_code_generation as solve_code_gen_template_fn,
    solve_logic as solve_logic_fn,
    solve_arithmetic as solve_arithmetic_fn,
)
from agent.solvers.logic_reasoning import solve_logical_reasoning as solve_logical_reasoning_fn


# Route table: per category, ordered solver list with pre/post/match
ROUTE_TABLE = {
    'code_gen': [
        RouteEntry(
            name='code_gen_template',
            solver_fn=solve_code_gen_template_fn,
            categories={'code_gen'},
            match_score=match_code_gen_template,
            preprocess=preprocess_code_gen,
            postprocess=postprocess_code_gen,
            min_score=0.5,
        ),
    ],
    'code_debug': [],  # solver fires <10% → always LLM
    'factual': [
        RouteEntry(
            name='factual_qa',
            solver_fn=solve_factual_qa_fn,
            categories={'factual'},
            match_score=match_factual,
            preprocess=preprocess_factual,
            min_score=0.0,
        ),
    ],
    'sentiment': [
        RouteEntry(
            name='sentiment',
            solver_fn=solve_sentiment_fn,
            categories={'sentiment'},
            match_score=match_sentiment,
            preprocess=preprocess_sentiment,
            postprocess=lambda a, p: normalize_sentiment(a, p),
            min_score=0.5,
        ),
    ],
    'ner': [
        RouteEntry(
            name='ner_regex',
            solver_fn=solve_ner,
            categories={'ner'},
            match_score=match_ner,
            preprocess=preprocess_ner,
            postprocess=normalize_ner_output,
            min_score=0.3,
        ),
    ],
    'logic': [
        RouteEntry(
            name='logical_reasoning',
            solver_fn=solve_logical_reasoning_fn,
            categories={'logic'},
            match_score=match_logic,
            min_score=0.4,
        ),
    ],
    'math': [
        RouteEntry(
            name='arithmetic',
            solver_fn=solve_arithmetic_fn,
            categories={'math'},
            match_score=match_math,
            min_score=0.5,
        ),
    ],
    'summarization': [],  # Always LLM
}


# ═══════════════════════════════════════════════════════════════════════════
# Main cascade dispatch
# ═══════════════════════════════════════════════════════════════════════════

def route(category: str, prompt: str) -> Optional[str]:
    """Route a (category, prompt) to the optimal solver.

    Returns:
        solver answer string, or None if no solver qualifies (→ LLM fallback)
    """
    entries = ROUTE_TABLE.get(category, [])

    for entry in entries:
        score = entry.match_score(prompt)
        if score < entry.min_score:
            continue

        # Preprocess
        cleaned = entry.preprocess(prompt)

        # Run solver
        try:
            answer = entry.solver_fn(cleaned, category)
        except Exception:
            answer = None

        if not answer:
            continue

        # Postprocess
        answer = entry.postprocess(answer, prompt)

        return answer

    return None  # No solver qualified → LLM fallback
