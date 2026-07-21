"""
math_step_classifier.py — Deterministic step type classifier for GSM8K.

For each step position in a math word problem, predicts the operation type
needed: 'add', 'subtract', 'multiply', 'divide', or 'combined'.

Uses keyword patterns, position-specific heuristics, and typical
GSM8K problem structures. No ML involved.
"""

import re
from typing import Dict, List, Optional, Tuple

# Available operation types
OP_TYPES = ["add", "subtract", "multiply", "divide", "combined", "other"]

# ============================================================
# Keyword → operation mappings (based on GSM8K training data analysis)
# ============================================================

# Strong indicators (high precision)
ADD_KEYWORDS = [
    "total", "altogether", "together", "combined", "sum",
    "more", "plus", "add", "additional", "both", "extra",
    "increase", "increased", "increases", "grew",
    "joined", "join", "added",
    "how many in total", "how many altogether",
    "what is the total", "what is the sum",
]

SUBTRACT_KEYWORDS = [
    "remaining", "left", "difference", "fewer", "less",
    "minus", "subtract",
    "how many more", "how many fewer", "how much more",
    "how many are left", "how many remain",
    "decrease", "decreased", "decreases", "reduced",
    "subtracted", "deduct", "deducted",
    "spent", "spends", "cost", "costs",
    "give", "gave", "gave away",
]

MULTIPLY_KEYWORDS = [
    "each", "per", "every", "times", "twice", "double",
    "product", "multiply", "multiplied", "multiplies",
    "dozen", "dozens",
    "an hour", "per hour", "per day", "per week", "per month", "per year",
    "a day", "a week", "a month", "a year",
    "apiece",
]

DIVIDE_KEYWORDS = [
    "half", "split", "divide", "divided", "divides",
    "share", "shared", "shares",
    "average", "averages", "averaged",
    "ratio", "quotient",
    "equal", "equally",
    "cut into", "divided into",
    "distribute", "distributed",
    "split equally",
]


def _match_keywords(text: str, keywords: List[str]) -> int:
    """Count how many keywords from the list appear in text."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count


def _word_count(text: str) -> int:
    return len(text.split())


def _is_unit_conversion(question: str) -> bool:
    """Detect if question involves unit conversion (e.g., hours↔minutes, days↔weeks)."""
    ql = question.lower()
    patterns = [
        r'per (hour|day|week|month|year|minute|second|dozen|pound|kilogram)',
        r'an? (hour|day|week|month|year|minute|second)',
        r'(minutes|hours|days|weeks|months|years) per',
        r'convert',
        r'earns?\s+\$?\d+\s+(an?|per)',
        r'makes?\s+\$?\d+\s+(an?|per)',
        r'charges?\s+\$?\d+\s+(an?|per)',
        r'paid?\s+\$?\d+\s+(an?|per)',
        r'(\d+) (minutes|hours|days|weeks|months|years)',
        r'(\d+) (dozen|pounds|kilograms|miles|kilometers)',
    ]
    for pat in patterns:
        if re.search(pat, ql):
            return True
    return False


def _detect_final_operation(question: str) -> Optional[str]:
    """
    Detect what the final question is asking (the computation needed at the last step).
    """
    ql = question.lower()
    
    # The question usually starts with the narrative and ends with the query
    # Look at the last sentence or question
    sentences = re.split(r'[.!?]', ql)
    
    # Check the final interrogative sentence
    final_sentences = [s.strip() for s in sentences if s.strip() and 'how' in s or 'what' in s]
    if not final_sentences:
        final_sentences = [s.strip() for s in sentences if s.strip()]
    
    if final_sentences:
        last_q = final_sentences[-1]
        
        # Addition signals
        if re.search(r'(total|altogether|together|combined|sum|in total)', last_q):
            return 'add'
        if re.search(r'how many (in all|altogether|total)', last_q):
            return 'add'
        if re.search(r'how many .* (total|altogether|altogether\?)', last_q):
            return 'add'
        
        # Subtraction signals
        if re.search(r'(remaining|how many (are )?left|how many more|difference|how much more)', last_q):
            return 'subtract'
        if re.search(r'how many (more|fewer|additional)', last_q):
            return 'subtract'
        if re.search(r'how much (more|less|additional)', last_q):
            return 'subtract'
        
        # Division signals
        if re.search(r'(each|per|every|per person|each person|apiece|each one)', last_q):
            if not re.search(r'(total|altogether)', last_q):
                return 'divide'
    
    return None


def _detect_first_operation(question: str) -> Optional[str]:
    """
    Detect what the first operation should be based on the question content.
    """
    ql = question.lower()
    
    # Strong division signals for first step
    if ('half' in ql or 'half of' in ql) and _word_count(ql) > 10:
        return 'divide'
    
    if 'split' in ql or 'divided' in ql or 'divides' in ql or 'share equally' in ql:
        return 'divide'
    
    if 'average' in ql:
        return 'divide'
    
    # Strong multiplication signals for first step
    if _is_unit_conversion(question):
        return 'multiply'
    
    if 'each' in ql or 'per' in ql or 'every' in ql:
        kw_count = _match_keywords(ql, ['each', 'per', 'every', 'times', 'twice'])
        if kw_count >= 1:
            return 'multiply'
    
    # If there are multiple numbers and comparison keywords, first step may be finding difference
    num_count = len(re.findall(r'\d+', question))
    comparison_words = ['more than', 'less than', 'fewer than', 'times as many']
    has_comparison = any(cw in ql for cw in comparison_words)
    
    if has_comparison and num_count >= 2:
        return 'subtract'
    
    return None


def predict_step_type(
    question: str,
    step_position: int,
    total_steps: int,
    step_context: Optional[str] = None,
) -> str:
    """
    Predict the operation type for a specific step in a math problem.

    Args:
        question: The full GSM8K question text.
        step_position: Which step this is (1-indexed).
        total_steps: Total number of steps predicted for this problem.
        step_context: Optional context text describing this specific step.

    Returns:
        One of: 'add', 'subtract', 'multiply', 'divide', 'combined'.
    """
    ql = question.lower()
    num_count = len(re.findall(r'\d+', question))
    wc = _word_count(question)

    # ---------------------------------------------------------------
    # POSITION-SPECIFIC STRATEGIES
    # ---------------------------------------------------------------

    # Strategy A: Single-step problem → detect the final operation needed
    if total_steps == 1:
        final_op = _detect_final_operation(question)
        if final_op:
            return final_op
        # Fallback for single step: check strongest signal
        add_s = _match_keywords(ql, ADD_KEYWORDS)
        sub_s = _match_keywords(ql, SUBTRACT_KEYWORDS)
        mul_s = _match_keywords(ql, MULTIPLY_KEYWORDS)
        div_s = _match_keywords(ql, DIVIDE_KEYWORDS)
        
        scores = {'add': add_s, 'subtract': sub_s, 'multiply': mul_s * 0.8, 'divide': div_s * 0.8}
        return max(scores, key=lambda k: scores[k])

    # Strategy B: First step of multi-step problem
    if step_position == 1:
        first_op = _detect_first_operation(question)
        if first_op:
            return first_op
        
        # Check division keywords specifically
        div_score = _match_keywords(ql, DIVIDE_KEYWORDS)
        mul_score = _match_keywords(ql, MULTIPLY_KEYWORDS)
        
        # Keywords that specifically indicate multiply for step 1
        if 'each' in ql or 'per' in ql or 'every' in ql:
            if 'half' not in ql and 'split' not in ql:
                return 'multiply'
        
        if 'times' in ql or 'twice' in ql or 'double' in ql:
            return 'multiply'
        
        if 'half' in ql or 'split' in ql or 'share' in ql:
            if div_score > mul_score:
                return 'divide'
            return 'divide'  # These are very often division
        
        # Unit conversion → multiply
        if _is_unit_conversion(question):
            return 'multiply'
        
        # Default for step 1
        if mul_score >= div_score:
            return 'multiply'
        return 'divide'

    # Strategy C: Last step of multi-step problem
    if step_position == total_steps:
        final_op = _detect_final_operation(question)
        if final_op:
            return final_op
        
        add_s = _match_keywords(ql, ADD_KEYWORDS)
        sub_s = _match_keywords(ql, SUBTRACT_KEYWORDS)
        
        # If strong subtraction signal in final question
        if sub_s >= add_s:
            return 'subtract'
        if add_s > sub_s:
            return 'add'
        
        # Default based on typical last-step operations
        # After multiplying/dividing, the last step is usually add or subtract
        if _match_keywords(ql, MULTIPLY_KEYWORDS) >= 2:
            return 'subtract'
        
        return 'add'

    # Strategy D: Middle steps (neither first nor last)
    # Middle steps often combine operations or continue the pattern from first step
    
    # If question has "remaining" or "left", middle steps often involve subtraction
    if 'remaining' in ql or ' left ' in f' {ql} ':
        return 'subtract'
    
    # If the first step would be multiplication, middle steps tend toward add/subtract
    add_score = _match_keywords(ql, ADD_KEYWORDS)
    sub_score = _match_keywords(ql, SUBTRACT_KEYWORDS)
    mul_score = _match_keywords(ql, MULTIPLY_KEYWORDS)
    div_score = _match_keywords(ql, DIVIDE_KEYWORDS)
    
    # Check if there's a narrative of "first X, then Y"
    has_narrative_middle = bool(re.search(r'(then|next|after|finally)', ql))
    
    if has_narrative_middle:
        if sub_score >= add_score:
            return 'subtract'
        if add_score >= sub_score:
            return 'add'
    
    # For problems with 2-3 total steps, middle steps default to add/subtract
    if total_steps <= 4:
        if add_score >= sub_score:
            return 'add'
        return 'subtract'
    
    # Longer problems: middle steps could be any operation
    scores = {
        'add': add_score * 1.5,
        'subtract': sub_score * 1.5,
        'multiply': mul_score * 0.5,
        'divide': div_score * 0.5,
    }
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def classify_operation_from_expr(expr: str) -> str:
    """
    Classify a GSM8K <<expression=result>> expression into operation type.
    Used for ground-truth parsing.
    """
    has_add = '+' in expr
    has_sub = '-' in expr
    has_mul = '*' in expr
    has_div = '/' in expr
    count_ops = sum([has_add, has_sub, has_mul, has_div])
    if count_ops == 0:
        return 'other'
    elif count_ops > 1:
        return 'combined'
    elif has_add:
        return 'add'
    elif has_sub:
        return 'subtract'
    elif has_mul:
        return 'multiply'
    elif has_div:
        return 'divide'
    return 'other'


def parse_ground_truth_plan(answer: str) -> List[Dict]:
    """
    Parse GSM8K answer into a list of step operations.
    Returns: [{'pos': 1, 'op': 'multiply'}, {'pos': 2, 'op': 'add'}, ...]
    """
    steps = []
    exprs = re.findall(r'<<(.*?)=', answer)
    for i, expr in enumerate(exprs):
        op = classify_operation_from_expr(expr)
        steps.append({"pos": i + 1, "op": op})
    return steps


def evaluate_on_dataframe(df) -> Dict:
    """
    Evaluate step type classifier accuracy on a DataFrame.
    Compares per-position predictions vs ground-truth operations.
    """
    correct_positions = 0
    total_positions = 0
    per_position = {}
    per_operation = {}

    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]

        true_steps = parse_ground_truth_plan(answer)
        if not true_steps:
            continue

        total_steps = len(true_steps)

        for step_info in true_steps:
            pos = step_info["pos"]
            true_op = step_info["op"]
            pred_op = predict_step_type(question, pos, total_steps)

            if pos not in per_position:
                per_position[pos] = {"correct": 0, "total": 0}
            per_position[pos]["total"] += 1
            total_positions += 1

            if true_op not in per_operation:
                per_operation[true_op] = {"correct": 0, "total": 0}
            per_operation[true_op]["total"] += 1

            if pred_op == true_op:
                correct_positions += 1
                per_position[pos]["correct"] += 1
                per_operation[true_op]["correct"] += 1

    pos_acc = {p: d["correct"] / d["total"] if d["total"] > 0 else 0.0
               for p, d in sorted(per_position.items())}
    op_acc = {op: d["correct"] / d["total"] if d["total"] > 0 else 0.0
              for op, d in sorted(per_operation.items())}

    return {
        "accuracy": correct_positions / total_positions if total_positions > 0 else 0.0,
        "correct_positions": correct_positions,
        "total_positions": total_positions,
        "per_position_accuracy": pos_acc,
        "per_position_counts": {p: per_position[p]["total"] for p in sorted(per_position)},
        "per_operation_accuracy": op_acc,
    }
