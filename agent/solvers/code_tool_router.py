"""
agent/solvers/code_tool_router.py — Step-by-step cascade classifier for coding tools.

Same principle as the 8-way cascade classifier: a sequence of simple binary/ternary
decisions that route a code prompt to the correct tool with cleaned input.

Decision tree:
  Step 0: code_gen or code_debug?
  Step 1: Exact function name matches a template?
  Step 2: Known algorithm mentioned by name?
  Step 3: Clear spec with test cases?
  Step 4 (debug): Traceback/error pattern in prompt?
  Step 5 (fallback): LLM
"""

import re
from typing import Optional

# ── Known template function names ──
_TEMPLATE_FUNCTIONS = {
    'freq_count', 'closest_num', 'is_pal', 'two_sum', 'reverse_str',
    'factorial', 'fib', 'merge_lists', 'fizzbuzz', 'count_vowels',
    'is_anagram', 'find_max_min', 'remove_duplicates', 'flatten_list',
    'is_prime', 'gcd', 'lcm', 'binary_search', 'linear_search',
    'bubble_sort', 'capitalize_words', 'word_count', 'is_even',
    'power', 'sum_digits', 'is_armstrong', 'is_perfect', 'transpose',
    'caesar_cipher', 'sort_dict_by_value', 'intersection', 'list_difference',
    'second_largest',
}

_KNOWN_ALGORITHMS = re.compile(
    r'\b(fibonacci|palindrome|anagram|fizz.?buzz|'
    r'two.?sum|binary.?search|bubble.?sort|merge.?sort|'
    r'prime\s+number|factorial|tower\s+of\s+hanoi|'
    r'caesar.?cipher|armstrong|perfect\s+number|'
    r'reverse\s+string|linear\s+search|gcd|greatest\s+common\s+divisor|'
    r'lcm|least\s+common\s+multiple|transpose\s+matrix)\b',
    re.IGNORECASE,
)

_DEBUG_PATTERNS = re.compile(
    r'\b(traceback|error|exception|bug|debug|fix\s+(the\s+)?(bug|error|function)|'
    r'broken|not\s+working|wrong\s+(output|result|answer)|'
    r'incorrect|failing|crash|typerror|valueerror|indexerror|keyerror)\b',
    re.IGNORECASE,
)

_SPEC_WITH_TESTS = re.compile(
    r'(?:>>>\s|Examples?:|Test cases?:|Input:|Output:|'
    r'For\s+example|should\s+return|expected\s+output|'
    r'assert\s+|should\s+be\s+)',
    re.IGNORECASE,
)


# ── Input preprocessing ──

def extract_function_name(prompt: str) -> Optional[str]:
    m = re.search(r'def\s+(\w+)\s*\(', prompt)
    return m.group(1) if m else None


def extract_function_signature(prompt: str) -> Optional[str]:
    m = re.search(r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?:)', prompt)
    return m.group(1) if m else None


def strip_task_prefix(prompt: str) -> str:
    return re.sub(
        r'^(?:Write|Create|Implement|Define|Code)\s+(?:a\s+)?(?:Python\s+)?'
        r'(?:function|program|script|class)\s*[:\n]*',
        '', prompt, flags=re.IGNORECASE
    ).strip()


def strip_docstrings(prompt: str) -> str:
    prompt = re.sub(r'""".*?"""', '', prompt, flags=re.DOTALL)
    prompt = re.sub(r"'''.*?'''", '', prompt, flags=re.DOTALL)
    return prompt.strip()


def extract_business_logic(prompt: str) -> str:
    """Extract just the core logic description from a code prompt."""
    text = strip_docstrings(prompt)
    text = re.sub(r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?:', '', text)
    # Remove test cases
    text = re.sub(r'>>>\s.*', '', text)
    text = re.sub(r'(?:Example|Test|Input|Output)s?.*', '', text)
    return text.strip()


# ── Template solvers (lazy-loaded) ──

_SOLVERS = {}

def _get_code_gen_template():
    if 'code_gen_template' not in _SOLVERS:
        from agent.solvers.deterministic import solve_code_generation
        _SOLVERS['code_gen_template'] = solve_code_generation
    return _SOLVERS['code_gen_template']


def _get_code_debug_solver():
    if 'code_debug_solver' not in _SOLVERS:
        from agent.solvers.deterministic import solve_code_debugging
        _SOLVERS['code_debug_solver'] = solve_code_debugging
    return _SOLVERS['code_debug_solver']


# ── Post-processing ──

def inject_function_name(answer: str, target_name: str) -> str:
    """Replace the solver's function name with the prompt's function name."""
    answer_fname = extract_function_name(answer)
    if answer_fname and answer_fname != target_name:
        answer = answer.replace(f'def {answer_fname}(', f'def {target_name}(', 1)
        # Also fix recursive calls inside the body
        answer = re.sub(
            rf'\b{re.escape(answer_fname)}\b(?=\s*\()',
            target_name, answer,
        )
    return answer


def inject_signature(answer: str, target_sig: str) -> str:
    """Replace the solver's def line with the prompt's exact def line."""
    answer_sig = extract_function_signature(answer)
    if answer_sig and answer_sig != target_sig:
        answer = answer.replace(answer_sig, target_sig, 1)
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# Cascade decision steps
# ═══════════════════════════════════════════════════════════════════════════

class CodeRoutingResult:
    """Result of cascade routing."""
    def __init__(self, tool: str, cleaned_prompt: str, answer: Optional[str] = None):
        self.tool = tool       # Which tool was selected
        self.cleaned = cleaned_prompt  # Cleaned input for that tool
        self.answer = answer   # Solver output (None if not yet run)


def route_code_gen(prompt: str) -> CodeRoutingResult:
    """Cascade classifier for code_gen prompts.

    Returns a CodeRoutingResult with the selected tool + cleaned input.
    """
    fname = extract_function_name(prompt)
    lower = prompt.lower()

    # ── Step 1: Exact function name matches a template ──
    if fname and fname in _TEMPLATE_FUNCTIONS:
        # Clean prompt: just the function signature + business logic
        sig = extract_function_signature(prompt)
        logic = extract_business_logic(prompt)
        cleaned = f"{sig}\n    \"\"\"{logic}\"\"\"" if sig and logic else strip_task_prefix(prompt)
        solver = _get_code_gen_template()
        answer = solver(prompt, 'code_gen')
        # Post-process: inject correct function name and signature
        if answer:
            if sig:
                answer = inject_signature(answer, sig)
            if fname:
                answer = inject_function_name(answer, fname)
        return CodeRoutingResult('template_exact_match', cleaned, answer)

    # ── Step 2: Known algorithm mentioned by name ──
    alg_match = _KNOWN_ALGORITHMS.search(lower)
    if alg_match:
        # Still use template solver with full prompt (it handles keyword matching)
        cleaned = strip_task_prefix(prompt)
        solver = _get_code_gen_template()
        answer = solver(prompt, 'code_gen')
        if answer and fname:
            answer = inject_function_name(answer, fname)
        return CodeRoutingResult('template_algorithm_match', cleaned, answer)

    # ── Step 3: Has clear spec with test cases → LLM ──
    has_tests = bool(_SPEC_WITH_TESTS.search(prompt))
    has_def = bool(re.search(r'def\s+\w+\s*\(', prompt))
    if has_tests or has_def:
        # Complex HumanEval-style — use described function name as cleaned output
        cleaned = strip_task_prefix(prompt)
        cleaned = strip_docstrings(cleaned)
        return CodeRoutingResult('llm_complex', cleaned, None)

    # ── Step 4: Simple / ambiguous → LLM ──
    cleaned = strip_task_prefix(prompt)
    cleaned = strip_docstrings(cleaned)
    return CodeRoutingResult('llm_simple', cleaned, None)


def route_code_debug(prompt: str) -> CodeRoutingResult:
    """Cascade classifier for code_debug prompts."""
    lower = prompt.lower()

    # ── Step 1: Has explicit error/traceback pattern ──
    has_debug = bool(_DEBUG_PATTERNS.search(lower))
    has_code = bool(re.search(r'def\s+\w+\(|```|return\s+', prompt))

    if has_debug and has_code:
        cleaned = strip_docstrings(prompt)
        cleaned = re.sub(
            r'^(?:Fix|Debug|Correct|Repair)\s+(?:the\s+)?(?:bug\s+in\s+)?(?:this\s+)?'
            r'(?:Python\s+)?(?:function|code|program)\s*[:\n]*',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        solver = _get_code_debug_solver()
        answer = solver(cleaned, 'code_debug')
        return CodeRoutingResult('pattern_solver', cleaned, answer)

    # ── Step 2: No clear debug pattern → LLM ──
    return CodeRoutingResult('llm', prompt, None)


def route_code(prompt: str, category: str) -> CodeRoutingResult:
    """Main entry point — routes code_gen or code_debug prompts.

    Returns:
        CodeRoutingResult with tool name, cleaned prompt, and solver answer if available.
    """
    if category == 'code_gen':
        return route_code_gen(prompt)
    elif category == 'code_debug':
        return route_code_debug(prompt)
    else:
        return CodeRoutingResult('llm_fallback', prompt, None)


def cleanup_cleaned_prompt(prompt: str) -> str:
    """Final cleaning — normalize whitespace for any route."""
    return re.sub(r'\n{3,}', '\n\n', prompt).strip()
