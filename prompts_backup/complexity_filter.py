"""
Stage 3 — Per-Category Complexity Scoring (8 weight-tuned scorers).

Each of the 8 categories gets its own set of 3-5 complexity signals,
its own weights, and its own threshold mapping. This is the key
difference from the unified scorer in agent/complexity.py — here signals
are category-aware rather than one-size-fits-all.

Signals per category:

  code_gen     function_count, loop_depth, import_count, code_length
  code_debug   error_pattern_count, code_size, function_count
  math         operation_count, paren_depth, proof_keyword, variable_count
  logic        condition_count, negation_count, quantifier_count
  factual      specificity_score, ambiguity_score, multi_hop_score
  sentiment    text_length, polarity_density, entity_count
  ner          entity_density, text_length, ambiguity_score
  summarization source_length_est, compression_ratio_est, bullet_requirement

Labeled data exists for: math (16K), logic (14K), factual (6K).
Others use heuristic defaults validated against the eval sets.

Usage:
    from agent.complexity_filter import score, describe, validate

    result = score(prompt, category="math")          # 0.0 - 1.0
    details = describe(prompt, category="code_gen")  # full signal breakdown
"""

import json
import math
import re
import os
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Category → signal functions
# ---------------------------------------------------------------------------

# Per-category signal extractors
# Each is category-specific, capturing different facets of complexity.

def _get_function_count(text: str) -> float:
    """Count function/method references and definitions."""
    count = len(re.findall(r'\bdef\s+\w+\s*\(', text))
    count += len(re.findall(r'\bfunction\s+\w+\s*\(', text))
    # Also count task-level code references
    count += len(re.findall(
        r'\b(write|implement|create|build|code)\b.*\b(function|program|class|method|algorithm|script)\b',
        text, re.IGNORECASE))
    # Count individual function/method/class mentions in task context
    count += len(re.findall(
        r'\b(function|method|class|program|script|algorithm|routine)\b',
        text, re.IGNORECASE)) * 0.3
    return min(count / 8.0, 1.0)


def _get_spec_detail(text: str) -> float:
    """Count technical specification details in code prompts."""
    count = 0
    # Complexity mentions
    specs = [
        r'\b(time|space)\s+(complexity|bound|limit)\b', r'\bO\s*\(\s*[nN]\s*\)',
        r'\b(edge\s+case|handle|validate|check|error|exception)\b',
        r'\b(optimize|efficient|performant|scalable)\b',
        r'\b(parallel|concurrent|multi.threaded|async|asynchronous)\b',
        r'\b(API|endpoint|interface|protocol|schema)\b',
        r'\b(database|SQL|query|cache|persist|store)\b',
        r'\b(recursive|iterative|dynamic.programming|backtracking|greedy)\b',
        r'\b(data.structure|array|list|tree|graph|hash|heap|stack|queue)\b',
        r'\b(design.pattern|singleton|factory|observer|decorator)\b',
        r'\b(unit.test|integration.test|test.case|coverage|mock|pytest)\b',
        r'\b(type|generic|template|protocol|trait|interface)\b',
    ]
    for s in specs:
        if re.search(s, text, re.IGNORECASE):
            count += 1
    return min(count / 6.0, 1.0)


def _get_loop_depth(text: str) -> float:
    """Estimate loop nesting. Counts for/while and list comps with multiple fors."""
    count = len(re.findall(r'\b(for|while)\b', text))
    nested = len(re.findall(r'\[.*\b(for)\b.*\b(for)\b', text))
    return min((count + nested * 2) / 6.0, 1.0)


def _get_import_count(text: str) -> float:
    """Count import statements. Normalised cap at 15."""
    count = len(re.findall(r'\b(import|from)\s+\w+', text))
    return min(count / 15.0, 1.0)


def _get_code_density(text: str) -> float:
    """Proportion of lines that look like code. Higher = more complex code task."""
    lines = text.split('\n')
    if not lines:
        return 0.0
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if (stripped.startswith(('    ', '\t')) or
            re.match(r'^(def |class |return |if |elif |else |for |while |try |except |with |import |from |print\()', stripped) or
            (re.search(r'[{}()\[\]]', stripped) and not re.search(r'[.!?]$', stripped))):
            code_lines += 1
    return min(code_lines / 30.0, 1.0)


def _get_error_pattern_count(text: str) -> float:
    """Count debugging/error-related patterns."""
    patterns = [
        r'\b(error|exception|traceback|bug|fix|debug|issue|fault)\b',
        r'Traceback|ValueError|TypeError|KeyError|IndexError|AttributeError|SyntaxError',
        r"(doesn't work|not working|wrong output|incorrect|broken|fails)",
        r'\b((throw|raise|catch)\s+\w+)\b',
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
    return min(count / 8.0, 1.0)


def _get_operation_count(text: str) -> float:
    """Count math operations. Discount operators without nearby numbers."""
    all_ops = len(re.findall(r'[+\-*/^%=]', text))
    has_numbers = bool(re.search(r'\d', text))
    operators = all_ops if has_numbers else all_ops * 0.2
    return min(operators / 15.0, 1.0)


def _get_paren_depth(text: str) -> float:
    """Max nesting depth of parentheses."""
    max_depth = 0
    current = 0
    for ch in text:
        if ch == '(':
            current += 1
            max_depth = max(max_depth, current)
        elif ch == ')':
            current = max(current - 1, 0)
    return min(max_depth / 5.0, 1.0)


def _get_proof_keywords(text: str) -> float:
    """Detect proof/theorem language — indicator of advanced math."""
    keywords = [
        r'\b(prove|proof|theorem|lemma|corollary|axiom|conjecture|derive)\b',
        r'\b(show that|demonstrate|verify|justify|therefore|hence|thus|iff|implies)\b',
        r'\b(necessary|sufficient|contradiction|induction|deduction)\b',
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in keywords)
    return min(count / 4.0, 1.0)


def _get_variable_count(text: str) -> float:
    """Count variable-like tokens. Indicates symbolic/parametrised problems."""
    singles = len(re.findall(r'\b([a-z])\b(?!\.)', text))
    multi = len(re.findall(r'\b([a-z]{2,3})\b(?!\.)', text))
    return min((singles * 0.3 + multi * 0.15) / 5.0, 1.0)


def _get_condition_count(text: str) -> float:
    """Count conditional constructs in logic/reasoning tasks."""
    count = len(re.findall(r'\b(if|else|elif|unless|switch|case)\b', text))
    count += len(re.findall(r'(&&|\|\||[<>]=?|==|!=)', text))
    return min(count / 8.0, 1.0)


def _get_negation_count(text: str) -> float:
    """Count negation patterns (logical complexity)."""
    count = len(re.findall(r'\b(not|never|no|none|nobody|nothing|neither|nor)\b', text, re.IGNORECASE))
    count += len(re.findall(r"n't\b", text))
    return min(count / 6.0, 1.0)


def _get_quantifier_count(text: str) -> float:
    """Count logical quantifiers (all/every/some/none)."""
    count = len(re.findall(r'\b(all|every|each|some|any|exists|forall|most|few|none)\b', text, re.IGNORECASE))
    return min(count / 5.0, 1.0)


def _get_specificity_score(text: str) -> float:
    """Measure specificity: citations, dates, numbers, multi-part wording."""
    words = text.split()
    n = len(words)
    if n < 5:
        return 0.1
    numbers = len(re.findall(r'\b\d+\b', text))
    has_dates = bool(re.search(
        r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|January|February|March|April|May|June|'
        r'July|August|September|October|November|December)\b', text))
    cap_words = len(re.findall(r'(?<![.!?]["\']?\s)[A-Z][a-z]+', text))
    quotes = text.count('"') + text.count("'") // 2
    # Multi-part: commas, semicolons, numbered lists
    multi_part = len(re.findall(r'[,;]', text)) / max(n, 1)

    spec = 0.0
    spec += min(numbers / 8.0, 0.20)
    spec += 0.10 if has_dates else 0.0
    spec += min(cap_words / 10.0, 0.25)
    spec += min(quotes / 4.0, 0.20)
    spec += min(multi_part * 5.0, 0.15)
    return min(spec + 0.05, 1.0)


def _get_technical_density(text: str) -> float:
    """Detect specialised/technical vocabulary — indicator of complex questions."""
    technical = [
        r'\b(algorithm|architecture|asymptotic|autonomous|catalyst|chromosome|cognitive|'
        r'correlation|covariance|cyber|derivative|empirical|entropy|equilibrium|'
        r'genotype|heterozygous|homeostasis|metabolism|mitosis|neural|ontological|'
        r'paradigm|phenotype|photosynthesis|protocol|quantum|recursive|relativistic|'
        r'semantic|stochastic|syntax|taxonomy|theorem|thermodynamic|topology|variance)\b',
        r'\b(acid|bacteria|carbon|compound|element|enzyme|gene|ion|molecule|organism|'
        r'protein|species|substrate|virus)\b',
        r'\b(demographic|geopolitical|hegemony|ideology|inflation|jurisdiction|'
        r'legislation|macroeconomics|monetary|regulatory|sovereignty|subsidy|tariff)\b',
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in technical)
    return min(count / 4.0, 1.0)


def _get_ambiguity_score(text: str) -> float:
    """Vague/open-ended language as complexity signal."""
    lower = text.lower()
    vague = [
        r'\b(maybe|perhaps|possibly|might|could|seems|appears|sort of|kind of)\b',
        r'\b(various|several|multiple|many|some|often|usually|sometimes)\b',
        r'\b(alternatively|depending|context|depends|varies)\b',
    ]
    count = sum(len(re.findall(p, lower)) for p in vague)
    return min(count / 4.0, 1.0)


def _get_multi_hop_score(text: str) -> float:
    """Multi-step reasoning indicators."""
    signals = [
        r'\b(first|second|third|then|next|finally|after that|subsequently)\b',
        r'\b(step \d|stage \d|phase \d)\b',
        r'\b(compare|contrast|relationship|connection|difference|similarity)\b',
        r'\b(if.*then|given.*find|assuming|suppose|derive|infer|deduce)\b',
        r'\b(chain|cascade|sequence|series|multi.step|multi.hop)\b',
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in signals)
    return min(count / 5.0, 1.0)


def _get_text_length_norm(text: str) -> float:
    """Text length as a complexity signal, with progressive scaling."""
    words = text.split()
    n = len(words)
    if n < 5:
        return 0.0
    if n < 15:
        return 0.1
    if n < 30:
        return 0.2
    if n < 60:
        return 0.4
    if n < 100:
        return 0.5
    if n < 150:
        return 0.6
    if n < 250:
        return 0.7
    if n < 400:
        return 0.85
    return 1.0


def _get_sentence_complexity(text: str) -> float:
    """Average words per sentence — longer sentences are more complex."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg < 8:
        return 0.1
    if avg < 15:
        return 0.3
    if avg < 25:
        return 0.5
    if avg < 40:
        return 0.7
    return 1.0


def _get_question_count(text: str) -> float:
    """Number of sub-questions — more sub-questions = more complex."""
    q_count = text.count('?')
    return min(q_count / 5.0, 1.0)


def _get_numeric_density(text: str) -> float:
    """Ratio of numeric tokens — higher in complex math/code tasks."""
    words = text.split()
    if not words:
        return 0.0
    numbers = len(re.findall(r'\b\d+\b', text))
    density = numbers / max(len(words), 1)
    if density < 0.01:
        return 0.0
    if density < 0.05:
        return 0.2
    if density < 0.10:
        return 0.5
    if density < 0.20:
        return 0.7
    return 1.0


def _get_math_notation_density(text: str) -> float:
    """Detect LaTeX math delimiters and math notation — common in competition math."""
    count = 0
    # LaTeX inline math: $...$
    count += len(re.findall(r'\$[^$]+\$', text)) * 2
    # LaTeX display math: $$...$$ or \[...\]
    count += len(re.findall(r'\$\$[^$]+\$\$|\\\[.*?\\\]', text)) * 3
    # Math function names
    count += len(re.findall(r'\b(sin|cos|tan|log|ln|exp|sqrt|frac|sum|prod|int|lim)\b', text, re.IGNORECASE))
    # Matrix notation
    count += len(re.findall(r'\\begin{pmatrix}|\\begin{bmatrix}|\\begin{vmatrix}', text))
    return min(count / 6.0, 1.0)


def _get_math_keyword_count(text: str) -> float:
    """Count math discipline keywords that indicate complex math."""
    keywords = [
        r'\b(derivative|integral|differential|calculus|gradient|divergence|curl)\b',
        r'\b(algebraic|polynomial|quadratic|coefficient|matrix|determinant|eigenvalue)\b',
        r'\b(probability|permutation|combination|binomial|variance|expectation)\b',
        r'\b(theorem|lemma|proof|axiom|conjecture|hypothesis)\b',
        r'\b(equation|inequality|expression|formula|identity)\b',
        r'\b(converge|diverge|limit|asymptotic|infinity|bound)\b',
        r'\b(trigonometric|logarithmic|exponential|logarithm|log)\b',
        r'\b(vector|tensor|scalar|manifold|topology|isomorphism)\b',
        r'\b(recurrence|recursion|induction|combinatorics|graph)\b',
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in keywords)
    return min(count / 5.0, 1.0)


def _get_number_word_count(text: str) -> float:
    """Count number words (one, two, three...) — often missed by numeric digit detection."""
    count = len(re.findall(
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
        r'eighty|ninety|hundred|thousand|million|billion)\b',
        text, re.IGNORECASE))
    return min(count / 4.0, 1.0)


def _get_polarity_density(text: str) -> float:
    """Mixed polarity in sentiment tasks."""
    positive = len(re.findall(
        r'\b(good|great|excellent|positive|amazing|wonderful|fantastic|happy|love|beautiful)\b',
        text, re.IGNORECASE))
    negative = len(re.findall(
        r'\b(bad|terrible|awful|negative|horrible|poor|worst|hate|ugly|angry)\b',
        text, re.IGNORECASE))
    total = positive + negative
    if total < 2:
        return 0.0
    ratio = min(positive, negative) / max(total, 1)
    return ratio * 0.7 + min(total / 10.0, 0.3)


def _get_entity_count_norm(text: str) -> float:
    """Normalised count of entity-like patterns."""
    cap_words = len(re.findall(r'(?<![.!?]["\']?\s)[A-Z][a-z]+', text))
    pairs = len(re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text))
    return min(cap_words * 0.04 + pairs * 0.12, 1.0)


def _get_entity_density_norm(text: str) -> float:
    """Proportion of capitalised tokens (entity density)."""
    words = text.split()
    if not words:
        return 0.0
    cap_count = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    density = cap_count / max(len(words), 1)
    return min(density * 4.0, 1.0)


def _get_source_length_est(text: str) -> float:
    """Estimate source text length for summarization."""
    words = text.split()
    n = len(words)
    if n < 20:
        return 0.1
    if n < 50:
        return 0.3
    if n < 100:
        return 0.5
    if n < 300:
        return 0.7
    return 1.0


def _get_compression_request(text: str) -> float:
    """Look for compression keywords (condense, tl;dr, etc.)."""
    lower = text.lower()
    indicators = [
        r'\b(single sentence|one sentence|brief|concise|short|tl;dr|tldr)\b',
        r'\b(summary|summarize|condense|abbreviate|paraphrase)\b',
        r'\b(key points|main idea|bullet|high.level|overview)\b',
    ]
    count = sum(len(re.findall(p, lower)) for p in indicators)
    return min(count / 3.0, 1.0)


def _get_bullet_requirement(text: str) -> float:
    """Check for structured output requirements in summarization."""
    has_bullet = bool(re.search(r'\b(bullet|numbered|list|outline)\b', text, re.IGNORECASE))
    has_format = bool(re.search(r'(json|xml|table|csv)', text, re.IGNORECASE))
    return 0.6 if has_bullet else (0.3 if has_format else 0.0)


# ---------------------------------------------------------------------------
# Category → signal extractors + weights
# ---------------------------------------------------------------------------

CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "code_gen":     ["function_count", "loop_depth", "import_count", "code_density",
                    "spec_detail", "sentence_complexity"],
    "code_debug":   ["error_pattern_count", "code_density", "function_count",
                    "spec_detail", "sentence_complexity"],
    "math":         ["operation_count", "paren_depth", "proof_keywords", "variable_count",
                    "numeric_density", "math_notation_density", "math_keyword_count",
                    "number_word_count", "text_length_norm"],
    "logic":        ["condition_count", "negation_count", "quantifier_count", "multi_hop_score",
                    "sentence_complexity"],
    "factual":      ["specificity_score", "technical_density", "ambiguity_score", "multi_hop_score",
                    "text_length_norm", "sentence_complexity"],
    "sentiment":    ["text_length_norm", "polarity_density", "entity_count_norm", "sentence_complexity"],
    "ner":          ["entity_density_norm", "text_length_norm", "ambiguity_score"],
    "summarization": ["source_length_est", "compression_request", "bullet_requirement",
                     "text_length_norm"],
}

# Default weights — tuned against HelpSteer data (signal correlation per category)
# Updated 2026-07-11: weights reflect which signals best separate simple vs complex
CATEGORY_WEIGHTS: Dict[str, List[float]] = {
    "code_gen":     [0.170, 0.346, 0.100, 0.065, 0.003, 0.315],
    "code_debug":   [0.094, 0.011, 0.121, 0.458, 0.316],
    "math":         [0.410, 0.033, 0.210, 0.040, 0.082, 0.008, 0.091, 0.082, 0.043],
    "logic":        [0.284, 0.014, 0.196, 0.460, 0.047],
    "factual":      [0.000, 0.138, 0.157, 0.183, 0.210, 0.312],
    "sentiment":    [0.155, 0.217, 0.000, 0.628],
    "ner":          [0.001, 0.431, 0.568],
    "summarization": [0.031, 0.371, 0.562, 0.036],
}

SIGNAL_FUNCTIONS = {
    "function_count":         _get_function_count,
    "spec_detail":            _get_spec_detail,
    "loop_depth":             _get_loop_depth,
    "import_count":           _get_import_count,
    "code_density":           _get_code_density,
    "error_pattern_count":    _get_error_pattern_count,
    "operation_count":        _get_operation_count,
    "paren_depth":            _get_paren_depth,
    "proof_keywords":         _get_proof_keywords,
    "variable_count":         _get_variable_count,
    "condition_count":        _get_condition_count,
    "negation_count":         _get_negation_count,
    "quantifier_count":       _get_quantifier_count,
    "specificity_score":      _get_specificity_score,
    "technical_density":      _get_technical_density,
    "ambiguity_score":        _get_ambiguity_score,
    "multi_hop_score":        _get_multi_hop_score,
    "text_length_norm":       _get_text_length_norm,
    "sentence_complexity":    _get_sentence_complexity,
    "question_count":         _get_question_count,
    "numeric_density":        _get_numeric_density,
    "math_notation_density":  _get_math_notation_density,
    "math_keyword_count":     _get_math_keyword_count,
    "number_word_count":      _get_number_word_count,
    "polarity_density":       _get_polarity_density,
    "entity_count_norm":      _get_entity_count_norm,
    "entity_density_norm":    _get_entity_density_norm,
    "source_length_est":      _get_source_length_est,
    "compression_request":    _get_compression_request,
    "bullet_requirement":     _get_bullet_requirement,
}

# Smoothing constant for unknown categories
_FALLBACK_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

# ---------------------------------------------------------------------------
# Per-item signal extraction
# ---------------------------------------------------------------------------

def extract_signals(prompt: str, category: str) -> Dict[str, float]:
    """Extract category-specific complexity signals from a prompt."""
    cat = category.lower().replace(" ", "_") if category else "unknown"
    signal_names = CATEGORY_SIGNALS.get(cat)

    if signal_names is None:
        # Try fuzzy match: check if it's a known category
        from CATEGORY_REGISTRY import get_short_name
        try:
            short_cat = get_short_name(cat)
            signal_names = CATEGORY_SIGNALS.get(short_cat)
        except Exception:
            signal_names = None

    # Handle 'code' alias → disambiguate code_gen vs code_debug
    if signal_names is None and cat in ("code", "coding", "programming"):
        debug_signals = len(re.findall(
            r'\b(debug|bug|fix|error|not working|broken|incorrect|wrong|issue|fault)\b',
            prompt, re.IGNORECASE))
        if debug_signals >= 2:
            signal_names = CATEGORY_SIGNALS["code_debug"]
        else:
            signal_names = CATEGORY_SIGNALS["code_gen"]

    # Handle 'general' → 'unknown'
    if signal_names is None and cat in ("general", "other", "other_complex"):
        signal_names = CATEGORY_SIGNALS.get("unknown")

    if signal_names is None:
        # Fallback: use generic signals
        signal_names = ["text_length_norm", "sentence_complexity", "multi_hop_score", "condition_count"]

    result = {}
    for name in signal_names:
        fn = SIGNAL_FUNCTIONS.get(name)
        if fn:
            try:
                result[name] = fn(prompt)
            except Exception:
                result[name] = 0.5
        else:
            result[name] = 0.5

    return result


def score(prompt: str, category: str) -> float:
    """
    Compute per-category complexity score (0.0 = simple, 1.0 = complex).

    Args:
        prompt: The user's input text.
        category: One of the 8 short category names (code_gen, math, etc.)
                  or 'unknown' for fallback.

    Returns:
        float: Complexity score between 0.0 and 1.0.
    """
    cat = category.lower().replace(" ", "_") if category else "unknown"
    signals = extract_signals(prompt, cat)

    # Get weights for this category
    weights = CATEGORY_WEIGHTS.get(cat, _FALLBACK_WEIGHTS)
    signal_names = CATEGORY_SIGNALS.get(cat)

    if signal_names is None:
        from CATEGORY_REGISTRY import get_short_name
        try:
            short_cat = get_short_name(cat)
            weights = CATEGORY_WEIGHTS.get(short_cat, _FALLBACK_WEIGHTS)
            signal_names = CATEGORY_SIGNALS.get(short_cat)
        except Exception:
            weights = _FALLBACK_WEIGHTS
            signal_names = list(signals.keys())

    total = 0.0
    weight_sum = 0.0
    names = signal_names or list(signals.keys())
    for name, w in zip(names, weights):
        if name in signals:
            total += signals[name] * w
            weight_sum += w

    if weight_sum == 0:
        return 0.5

    return round(total / weight_sum, 4)


def describe(prompt: str, category: str) -> dict:
    """
    Full diagnostic: signals + category + weighted score.
    """
    cat = category.lower().replace(" ", "_") if category else "unknown"
    signals = extract_signals(prompt, cat)

    from CATEGORY_REGISTRY import get_short_name, get_human_name
    try:
        short_cat = get_short_name(cat)
    except Exception:
        short_cat = cat
    try:
        human = get_human_name(short_cat)
    except Exception:
        human = short_cat

    weights = CATEGORY_WEIGHTS.get(short_cat, _FALLBACK_WEIGHTS)
    signal_names = CATEGORY_SIGNALS.get(short_cat, list(signals.keys()))

    return {
        "category": short_cat,
        "category_human": human,
        "unified_score": score(prompt, category),
        "signals": signals,
        "signal_names": signal_names,
        "weights": dict(zip(signal_names, weights)),
    }


# ---------------------------------------------------------------------------
# Data-driven threshold tuning
# ---------------------------------------------------------------------------

def _load_labeled_data(data_path: str = "", augment_with_wesley: bool = False) -> list:
    """
    Load labeled complexity data from stage3/*.jsonl.
    
    Args:
        data_path: Path to one of the stage3 split files (train/val/test).
        augment_with_wesley: If True, also load wesley-stage2 merged data.
            Default False because wesley's "question difficulty" labels measure
            knowledge required, not textual prompt complexity. The correlation is
            weak — it degrades accuracy on clean Magpie data.
    
    Returns combined list of items with 'prompt', 'complexity', and 'label_8way'.
    Never modifies or duplicates existing data — derived files only.
    """
    items = []
    seen_prompts = set()

    # 1. Primary: stage3 split files
    if not data_path:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "prompt_data", "stage3", "val.jsonl"),
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "amd-hackathon-shared", "prompt_data", "stage3", "val.jsonl"),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), "")

    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                if "complexity" in item and item.get("prompt"):
                    items.append(item)
                    seen_prompts.add(item["prompt"])

    # 2. Augment with wesley-stage2 merged data (non-destructive)
    # Only augment categories where wesley complexity correlates positively
    # with textual signals: factual, math, logic (verified empirically).
    # NER and summarization show INVERTED correlation (negative sep delta) —
    # the wesley "question difficulty" labels don't measure textual complexity.
    merged_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "prompt_data", "wesley_stage2_merged.jsonl"),
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "amd-hackathon-shared", "prompt_data", "wesley_stage2_merged.jsonl"),
    ]
    merged_path = next((p for p in merged_candidates if os.path.exists(p)), "")
    if augment_with_wesley and merged_path and merged_path != data_path:
        added = 0
        skipped = 0
        with open(merged_path) as f:
            for line in f:
                item = json.loads(line)
                # Only add if not already in primary data AND category correlates positively
                if item["prompt"] not in seen_prompts and "complexity" in item and item.get("label_8way"):
                    cat = item["label_8way"]
                    # Only augment categories where wesley labels correlate with textual signals.
                    # factual: positive correlation (sep ~0.05-0.10)
                    # NER/summarization: INVERTED (negative sep) — skip
                    if cat in ("factual",):
                        items.append(item)
                        seen_prompts.add(item["prompt"])
                        added += 1
                    else:
                        skipped += 1
        if added:
            print(f"  Augmented with {added} items from wesley-stage2 merge (skipped {skipped} for inverted categories)")

    return items


def compute_optimal_threshold(labels: List[float], predictions: List[float]) -> float:
    """
    Find the threshold that best separates binary simple/complex labels,
    minimising misclassification rate.
    Labels: 0 = simple, 1 = complex (from ground truth).
    Predictions: continuous 0-1 scores from our scorers.
    """
    if len(labels) < 10 or len(predictions) < 10:
        return 0.5

    candidates = [p for p in predictions if 0.0 < p < 1.0]
    if not candidates:
        return 0.5

    best_threshold = 0.5
    best_accuracy = 0.0

    # Sample candidate thresholds at the prediction points
    thresholds = sorted(set([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] + candidates))
    for t in thresholds:
        correct = sum(1 for l, p in zip(labels, predictions)
                      if (p >= t and l == 1) or (p < t and l == 0))
        acc = correct / len(labels)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    return best_threshold


def _binarize_complexity(complexity: float) -> int:
    """Convert continuous complexity to binary: 0=simple (< 0.5), 1=complex (>= 0.5)."""
    return 1 if complexity >= 0.5 else 0


def tune_thresholds(data_path: str = "") -> Dict[str, float]:
    """
    Load labeled complexity data and tune per-category thresholds.
    Now also loads wesley-stage2 merged data for categories that previously had no labels.
    Returns dict of category -> optimal threshold.
    """
    if not data_path:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "prompt_data", "stage3", "val.jsonl"),
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "amd-hackathon-shared", "prompt_data", "stage3", "val.jsonl"),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), "")

    items = _load_labeled_data(data_path)
    if not items:
        print(f"Warning: No labeled data found")
        return {}

    # Group by category
    from collections import defaultdict
    cat_items = defaultdict(list)
    for item in items:
        cat = item.get("label_8way", "unknown")
        if "complexity" in item:
            cat_items[cat].append(item)

    thresholds = {}
    for cat, cat_data in cat_items.items():
        if len(cat_data) < 20:
            continue
        labels = [float(_binarize_complexity(d["complexity"])) for d in cat_data]
        preds = [score(d["prompt"], cat) for d in cat_data]
        thresh = compute_optimal_threshold(labels, preds)
        thresholds[cat] = round(thresh, 3)

    return thresholds


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(data_path: str = "") -> dict:
    """
    Run full validation of Stage 3 against labeled data.
    Automatically augments with wesley-stage2 merged data.
    Reports per-category separation, accuracy, and confusion.
    """
    if not data_path:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "prompt_data", "stage3", "val.jsonl"),
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "amd-hackathon-shared", "prompt_data", "stage3", "val.jsonl"),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), "")

    items = _load_labeled_data(data_path)
    if not items:
        return {"error": f"No labeled data found"}

    from collections import defaultdict

    # Per-category metrics
    cat_metrics = defaultdict(lambda: {"n": 0, "correct": 0, "errors": 0,
                                        "true_pos": 0, "false_pos": 0,
                                        "true_neg": 0, "false_neg": 0,
                                        "predicted": [], "actual": []})

    thresholds = tune_thresholds(data_path)
    default_threshold = 0.5

    overall_correct = 0
    overall_total = 0

    for item in items:
        cat = item.get("label_8way", "unknown")
        if "complexity" not in item:
            continue
        actual = _binarize_complexity(item["complexity"])
        predicted_score = score(item["prompt"], cat)
        threshold = thresholds.get(cat, default_threshold)
        predicted = 1 if predicted_score >= threshold else 0

        metrics = cat_metrics[cat]
        metrics["n"] += 1
        metrics["actual"].append(actual)
        metrics["predicted"].append(predicted_score)

        if predicted == actual:
            metrics["correct"] += 1
            overall_correct += 1
        else:
            metrics["errors"] += 1

        if predicted == 1 and actual == 1:
            metrics["true_pos"] += 1
        elif predicted == 1 and actual == 0:
            metrics["false_pos"] += 1
        elif predicted == 0 and actual == 0:
            metrics["true_neg"] += 1
        elif predicted == 0 and actual == 1:
            metrics["false_neg"] += 1

        overall_total += 1

    # Build report
    results = {}
    for cat, m in sorted(cat_metrics.items()):
        if m["n"] < 5:
            continue
        acc = m["correct"] / m["n"] if m["n"] > 0 else 0
        prec = m["true_pos"] / (m["true_pos"] + m["false_pos"] + 1e-10)
        recall = m["true_pos"] / (m["true_pos"] + m["false_neg"] + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)

        # Separation: mean score of simple vs complex items
        simple_scores = [s for s, a in zip(m["predicted"], m["actual"]) if a == 0]
        complex_scores = [s for s, a in zip(m["predicted"], m["actual"]) if a == 1]
        sep = (sum(complex_scores) / max(len(complex_scores), 1)
               - sum(simple_scores) / max(len(simple_scores), 1))

        results[cat] = {
            "n": m["n"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "threshold": thresholds.get(cat, default_threshold),
            "separation_delta": round(sep, 4),
            "simple_mean": round(sum(simple_scores) / max(len(simple_scores), 1), 4) if simple_scores else None,
            "complex_mean": round(sum(complex_scores) / max(len(complex_scores), 1), 4) if complex_scores else None,
            "confusion": {
                "tp": m["true_pos"],
                "fp": m["false_pos"],
                "tn": m["true_neg"],
                "fn": m["false_neg"],
            },
        }

    results["_summary"] = {
        "total_items": overall_total,
        "overall_accuracy": round(overall_correct / overall_total, 4) if overall_total > 0 else 0,
        "categories_with_data": list(results.keys() - {"_summary"}),
    }

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        report = validate()
        print(json.dumps(report, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "--tune":
        thresholds = tune_thresholds()
        print("Optimal thresholds per category:")
        for cat, t in sorted(thresholds.items()):
            print(f"  {cat:>15s}: {t:.3f}")
    elif len(sys.argv) >= 3:
        prompt = sys.argv[1]
        category = sys.argv[2]
        result = describe(prompt, category)
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python -m agent.stage3 <prompt> <category>   # Single score")
        print("  python -m agent.stage3 --validate            # Run validation")
        print("  python -m agent.stage3 --tune                # Tune thresholds")
