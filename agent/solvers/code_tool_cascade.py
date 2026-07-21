"""
agent/solvers/code_tool_cascade.py — Binary cascade decision tree for coding tools.

Each node is a single yes/no classifier. The tree grows one binary split at a time.
No multi-way decisions — every fork is True/False.

Current tree:
  N0:  Has structured I/O (JSON examples + test cases)?
       ├─YES→ N1:  Involves tree/graph/linked-list data structure?
       │         ├─YES→ [data_structure_solver]
       │         └─NO→  N2:  Involves optimal substructure / DP?
       │                 ├─YES→ [dp_solver]
       │                 └─NO→  N3:  Involves sorting/searching/math?
       │                          ├─YES→ [sort_search_solver]
       │                          └─NO→  [llm]
       └─NO→  N4:  Function name matches template exactly?
                ├─YES→ [template_solver]
                └─NO→  N5:  Known algorithm named in prompt text?
                         ├─YES→ [template_solver_renamed]
                         └─NO→  [llm]
"""

import re
from typing import Optional, Callable


# ═══════════════════════════════════════════════════════════════════════════
# Node definitions — each is a binary (yes/no) decision
# ═══════════════════════════════════════════════════════════════════════════

# ── N0: Has structured I/O ──
_HAS_STRUCTURED_IO = re.compile(
    r'(?:\[?\{\s*"input"|\[?\{\s*"input"|test_cases|examples\s*:|'
    r'example\s+input|expected\s+output|constraints)', re.IGNORECASE,
)

def has_structured_io(prompt: str) -> bool:
    """N0: Does the prompt contain JSON examples/test cases/structured I/O?"""
    return bool(_HAS_STRUCTURED_IO.search(prompt))


# ── N1: Data structure problem ──
_N1_TREE = re.compile(
    r'\b(binary\s*tree|bst|root|node|leaf|traversal|inorder|preorder|postorder|'
    r'level.?order|subtree|tree\s+node|tree\s+path|tree\s+sum|bst\s+|'
    r'trie|prefix\s*tree|segment\s*tree|fenwick|binary\s+indexed)\b',
    re.IGNORECASE,
)
_N1_GRAPH = re.compile(
    r'\b(graph|dfs|bfs|dijkstra|topological|adjacency|edge|vertex|nodes?\s+\d+|'
    r'cycle\s+detect|connected\s+component|shortest\s+path|minimum\s+spanning|'
    r'strongly\s+connected|bipartite|directed|undirected)\b',
    re.IGNORECASE,
)
_N1_LINKED = re.compile(
    r'\b(linked\s*list|node\s*value|next\s*pointer|head|tail|'
    r'singly|doubly|circular\s+list|list\s+cycle|list\s+reversal)\b',
    re.IGNORECASE,
)

def is_data_structure(prompt: str) -> bool:
    """N1: Does the problem involve tree/graph/linked-list DS?"""
    lower = prompt.lower()
    return bool(_N1_TREE.search(lower) or _N1_GRAPH.search(lower) or _N1_LINKED.search(lower))


# ── N2: DP problem ──
_N2_DP = re.compile(
    r'\b(dynamic\s*programming|dp\b|knapsack|subsequence|subarray|substring|'
    r'edit\s+distance|lcs|lis|longest\s+(increasing|common|palindromic)|'
    r'minimum\s+(cost|path|sum|steps|operations)|maximum\s+(sum|profit|product)|'
    r'best\s+time|coin\s+change|word\s+break|partition|palindromic\s+substrings|'
    r'matrix\s+chain|optimal\s+(bst|path)|memoization|tabulation)\b',
    re.IGNORECASE,
)
_N2_DP_WORDS = re.compile(
    r'\b(maximize|minimize|optimal|maximum|minimum|'
    r'ways?\s+to|number\s+of\s+ways|count\s+the\s+number|'
    r'(?:can|could|possible)\s+(?:you\s+)?(?:reach|obtain|make|achieve))\b',
    re.IGNORECASE,
)

def is_dp_problem(prompt: str) -> bool:
    """N2: Does the problem involve optimal substructure / DP?"""
    lower = prompt.lower()
    return bool(_N2_DP.search(lower))


# ── N3: Sort / search / math problem ──
_N3_SORT = re.compile(
    r'\b(sort|merge|quick|heap|bucket|radix|counting|selection|insertion|'
    r'ordered|sorted|kth\s+(smallest|largest|closest|nearest)|median|'
    r'priority\s*queue|top\s*k|frequency\s+sort)\b',
    re.IGNORECASE,
)
_N3_SEARCH = re.compile(
    r'\b(binary\s*search|search\s+in|find\s+(element|index|position)|'
    r'index\s+of|locate|search\s+algorithm|two\s*sum|three\s*sum|'
    r'k\s*sum|search\s+rotated|search\s+matrix)\b',
    re.IGNORECASE,
)
_N3_MATH = re.compile(
    r'\b(math|prime|gcd|lcm|factorial|fibonacci|'
    r'geometric|arithmetic|modulo|exponent|power\s+of|'
    r'bit\s+manipulation|xor|bitwise)\b',
    re.IGNORECASE,
)

def is_sort_search_math(prompt: str) -> bool:
    """N3: Does the problem involve sorting/searching/math operators?"""
    lower = prompt.lower()
    return bool(_N3_SORT.search(lower) or _N3_SEARCH.search(lower) or _N3_MATH.search(lower))


# ── N4: Function name matches template exactly ──
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

def _extract_fn_name(prompt: str) -> Optional[str]:
    m = re.search(r'def\s+(\w+)\s*\(', prompt)
    return m.group(1) if m else None

def fn_name_matches_template(prompt: str) -> bool:
    """N4: Does the prompt's function name exactly match a template?"""
    fn = _extract_fn_name(prompt)
    return fn is not None and fn in _TEMPLATE_FUNCTIONS


# ── N5: Known algorithm named ──
_KNOWN_ALGORITHM = re.compile(
    r'\b(fibonacci|palindrome|anagram|fizz.?buzz|'
    r'two.?sum|binary.?search|bubble.?sort|merge.?sort|'
    r'prime\s+number|factorial|tower\s+of\s+hanoi|'
    r'caesar.?cipher|armstrong|perfect\s+number|'
    r'reverse\s+string|linear\s+search|gcd|greatest\s+common\s+divisor|'
    r'lcm|least\s+common\s+multiple|transpose)\b',
    re.IGNORECASE,
)

def known_algorithm_named(prompt: str) -> bool:
    """N5: Does the prompt text mention a known algorithm by name?"""
    return bool(_KNOWN_ALGORITHM.search(prompt.lower()))


# ═══════════════════════════════════════════════════════════════════════════
# Solver dispatch
# ═══════════════════════════════════════════════════════════════════════════

class RoutingResult:
    def __init__(self, tool: str, cleaned: str, prompt_key: str = "",
                 solver_fn: Optional[Callable] = None):
        self.tool = tool
        self.cleaned = cleaned
        self.prompt_key = prompt_key  # maps to dynamic_prompts._CATEGORY_PROMPTS key
        self._solver_fn = solver_fn

    def solve(self) -> Optional[str]:
        if self._solver_fn:
            return self._solver_fn(self.cleaned, 'code_gen')
        return None

    def __repr__(self):
        return f'RoutingResult(tool={self.tool})'


# ── Solver accessors (lazy) ──
def _get_solver(name):
    if name == 'template':
        from agent.solvers.deterministic import solve_code_generation
        return solve_code_generation
    if name == 'llm_none':
        return None
    raise ValueError(f'Unknown solver: {name}')


# ── Preprocessing ──
def _strip_task_hdr(prompt: str) -> str:
    return re.sub(
        r'^(?:Write|Create|Implement|Define|Code)\s+(?:a\s+)?(?:Python\s+)?'
        r'(?:function|program|script|class)\s*[:\n]*',
        '', prompt, flags=re.IGNORECASE
    ).strip()


def _strip_docs(text: str) -> str:
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
    return text.strip()


def _clean_for_template(prompt: str) -> str:
    return _strip_task_hdr(_strip_docs(prompt))


def _clean_for_llm(prompt: str) -> str:
    """For LLM: strip JSON test cases but keep the spec + examples as text."""
    cleaned = re.sub(r'"test_cases"\s*:\s*\[.*?\]', '', prompt, flags=re.DOTALL)
    cleaned = re.sub(r'"examples"\s*:\s*\[.*?\]', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Binary cascade tree — route!
# ═══════════════════════════════════════════════════════════════════════════

def route_code(prompt: str, category: str) -> RoutingResult:
    """Walk the binary cascade tree and return the selected tool."""
    
    # ── Only route code_gen and code_debug ──
    if category not in ('code_gen', 'code_debug'):
        return RoutingResult('llm_fallback', prompt)

    if category == 'code_debug':
        # Single binary decision for code_debug (for now)
        has_debug_pattern = bool(re.search(
            r'\b(traceback|error|exception|bug|debug|fix|broken|crash)', prompt, re.I
        ))
        if has_debug_pattern:
            from agent.solvers.deterministic import solve_code_debugging
            return RoutingResult('pattern_debug', prompt,
                                 solver_fn=solve_code_debugging)
        return RoutingResult('llm_debug', prompt)

    # ── code_gen tree ──
    lower = prompt.lower()

    # N0: Has structured I/O?
    if has_structured_io(prompt):
        # N1: Data structure?
        if is_data_structure(prompt):
            return RoutingResult('llm_data_structure', _clean_for_llm(prompt),
                                 prompt_key='coding_challenge_ds')
        # N2: DP?
        if is_dp_problem(prompt):
            return RoutingResult('llm_dp', _clean_for_llm(prompt),
                                 prompt_key='coding_challenge_dp')
        # N3: Sort/search/math?
        if is_sort_search_math(prompt):
            return RoutingResult('llm_sort_search', _clean_for_llm(prompt),
                                 prompt_key='coding_challenge_sort_search')
        # Everything else formal → generic LLM
        return RoutingResult('llm_formal', _clean_for_llm(prompt),
                             prompt_key='coding_challenge_formal')

    # Natural language path
    # N4: Function name matches template?
    if fn_name_matches_template(prompt):
        return RoutingResult('template_exact', _clean_for_template(prompt),
                             solver_fn=_get_solver('template'))

    # N5: Known algorithm named?
    if known_algorithm_named(prompt):
        return RoutingResult('template_renamed', _clean_for_template(prompt),
                             solver_fn=_get_solver('template'))

    # LLM fallback
    return RoutingResult('llm_simple', _strip_task_hdr(prompt))
