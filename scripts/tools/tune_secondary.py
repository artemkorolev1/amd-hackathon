#!/usr/bin/env python3
"""
Tuning script for secondary module thresholds - EFFICIENT VERSION.
Pre-computes primary predictions, filters to relevant categories per stage.
"""
import importlib.util, json, os, re, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(HERE, "agent")
TRAIN_PATH = os.path.join(HERE, "data/eval/training-v2.json")
VAL_PATH = os.path.join(HERE, "data/eval/validation-v2.json")

start_time = time.time()

def load_primary():
    spec = importlib.util.spec_from_file_location("category_filter", os.path.join(AGENT_DIR, "category_filter.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_data(path):
    with open(path) as f: return json.load(f)

def timelog(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {msg}", flush=True)

# ── Pre-compute primary predictions ──
def precompute_primary(primary, data):
    """Returns list of (true_cat, primary_cat, prompt) for each item."""
    results = []
    for item in data:
        result = primary.classify_with_detail(item["prompt"])
        results.append((item["category"], result["category"], item["prompt"]))
    return results

# ── Optimized evaluator given pre-computed data ──
def evaluate_secondary(data_tuples, cat_filter, secondary_fn):
    """
    Evaluate accuracy impact of a secondary classifier.
    data_tuples: list of (true_cat, primary_cat, prompt)
    cat_filter: set of categories that this secondary affects
    secondary_fn: (primary_cat, prompt) -> corrected_cat
    Returns accuracy over all items.
    """
    correct = 0
    total = len(data_tuples)
    for true_cat, primary_cat, prompt in data_tuples:
        if primary_cat in cat_filter:
            pred = secondary_fn(primary_cat, prompt)
        else:
            pred = primary_cat
        if pred == true_cat:
            correct += 1
    return correct / total if total > 0 else 0.0

def evaluate_full(data_tuples, code_fn, reason_fn, factual_fn):
    """Full pipeline evaluation."""
    correct = 0
    total = len(data_tuples)
    for true_cat, primary_cat, prompt in data_tuples:
        cat = primary_cat
        if cat in ("code_debug", "code_gen"):
            corrected = code_fn(cat, prompt)
            if corrected != cat: cat = corrected
        if cat in ("logic", "math"):
            corrected = reason_fn(cat, prompt)
            if corrected != cat: cat = corrected
        corrected = factual_fn(cat, prompt)
        if corrected != cat: cat = corrected
        if cat == true_cat:
            correct += 1
    return correct / total if total > 0 else 0.0

# ══════════════════════════════════════════════════════════════════
# BASE MODULE LOADING (cached)
# ══════════════════════════════════════════════════════════════════

def _load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(AGENT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_code_base = None
def get_code_base():
    global _code_base
    if _code_base is None:
        _code_base = _load_mod("sc_base", "secondary_code.py")
    return _code_base

_reason_base = None
def get_reason_base():
    global _reason_base
    if _reason_base is None:
        _reason_base = _load_mod("sr_base", "secondary_reasoning.py")
    return _reason_base

_factual_base = None
def get_factual_base():
    global _factual_base
    if _factual_base is None:
        _factual_base = _load_mod("sf_base", "secondary_factual.py")
    return _factual_base

# ══════════════════════════════════════════════════════════════════
# CODE MODULE
# ══════════════════════════════════════════════════════════════════

def make_code_fn(config):
    base = get_code_base()
    dt = config.get("debug_threshold", 5.0)
    gt = config.get("gen_threshold", 3.0)
    ratio = config.get("ratio", 1.4)
    dw = config.get("debug_weights", {})
    gw = config.get("gen_weights", {})

    def fn(cat, prompt):
        lower = prompt.lower()
        ds = 0.0
        if base._FIX_THE_BUG_RE.search(lower): ds += 5.0 * dw.get("fix_the_bug", 1.0)
        if base._BUG_RE.search(lower): ds += 3.0 * dw.get("bug", 1.0)
        if base._DEBUG_RE.search(lower): ds += 3.0 * dw.get("debug", 1.0)
        if base._ERROR_RE.search(lower): ds += 3.0 * dw.get("error", 1.0)
        if base._TRACEBACK_RE.search(lower): ds += 3.0 * dw.get("traceback", 1.0)
        if base._EXCEPTION_RE.search(lower): ds += 3.0 * dw.get("exception", 1.0)
        if base._CRASH_RE.search(lower): ds += 3.0 * dw.get("crash", 1.0)
        if base._BROKEN_RE.search(lower): ds += 3.0 * dw.get("broken", 1.0)
        if base._INCORRECT_RE.search(lower): ds += 3.0 * dw.get("incorrect", 1.0)
        if base._WRONG_RE.search(lower): ds += 3.0 * dw.get("wrong", 1.0)
        if base._FAILING_RE.search(lower): ds += 3.0 * dw.get("failing", 1.0)
        if base._NOT_WORKING_RE.search(lower): ds += 3.0 * dw.get("not_working", 1.0)
        if base._OUTPUT_IS_RE.search(lower): ds += 2.0 * dw.get("output_is", 1.0)
        if base._SHOULD_BE_RE.search(lower): ds += 1.0 * dw.get("should_be", 1.0)
        if base._EXPECTED_ACTUAL_RE.search(lower): ds += 2.0 * dw.get("expected_actual", 1.0)
        if base._FIX_RE.search(lower): ds += 2.0 * dw.get("fix", 1.0)
        
        gs = 0.0
        if base._WRITE_FUNCTION_RE.search(lower): gs += 3.0 * gw.get("write_function", 1.0)
        elif base._WRITE_CODE_RE.search(lower) or base._WRITE_START_RE.search(lower): gs += 2.0 * gw.get("write_code", 1.0)
        if base._DEF_RE.search(lower): gs += 1.0 * gw.get("def", 1.0)
        if base._IMPLEMENT_CODE_RE.search(lower): gs += 3.0 * gw.get("implement_code", 1.0)
        elif base._IMPLEMENT_RE.search(lower): gs += 1.0 * gw.get("implement", 1.0)
        if base._CREATE_CODE_RE.search(lower): gs += 2.0 * gw.get("create_code", 1.0)
        if base._FUNCTION_THAT_RE.search(lower): gs += 1.0 * gw.get("function_that", 1.0)
        if base._IN_PYTHON_RE.search(lower): gs += 1.0 * gw.get("in_python", 1.0)
        
        if ds >= dt and ds >= gs * ratio: return "code_debug"
        if gs >= gt and gs >= ds * ratio: return "code_gen"
        return cat
    return fn

# ══════════════════════════════════════════════════════════════════
# REASONING MODULE
# ══════════════════════════════════════════════════════════════════

def make_reason_fn(config):
    base = get_reason_base()
    lt = config.get("logic_threshold", 3.0)
    mt = config.get("math_threshold", 3.0)
    dm = config.get("decisive_margin", 1.0)
    lw = config.get("logic_weights", {})
    mw = config.get("math_weights", {})

    def score_logic(prompt):
        s = 0.0
        if base._LOGIC_PUZZLE_RE.search(prompt): s += 4.0 * lw.get("logic_puzzle", 1.0)
        if base._WHICH_FOLLOWING_RE.search(prompt): s += 3.0 * lw.get("which_following", 1.0)
        if base._SYLLOGISM_RE.search(prompt): s += 4.0 * lw.get("syllogism", 1.0)
        if base._STATEMENT_REASONING_RE.search(prompt): s += 3.0 * lw.get("statement_reasoning", 1.0)
        if base._CONSTRAINT_RE.search(prompt): s += 3.0 * lw.get("constraint", 1.0)
        if base._ENTITY_PUZZLE_RE.search(prompt): s += 3.0 * lw.get("entity_puzzle", 1.0)
        names = base._NAME_LIST_RE.findall(prompt)
        has_cc = bool(re.search(r"\b(each|different|assigned|allocated|must|condition|rule|requirement|older|younger|taller|shorter|faster|slower|left|right|between|adjacent|before|after|not\s+(the\s+)?same|no\s+two|if\s|then\s|either|neither|nor)\b", prompt, re.IGNORECASE))
        if names and has_cc: s += 3.0 * lw.get("names_constraint", 1.0)
        if base._CHINESE_LOGIC_RE.search(prompt): s += 3.0 * lw.get("chinese_logic", 1.0)
        if base._OPTIONS_RE.search(prompt): s += 2.0 * lw.get("options", 1.0)
        if re.search(r"\b(?:Question|question|Q:|Q\.)\s*\:?", prompt) and base._STATEMENT_REASONING_RE.search(prompt): s += 1.0 * lw.get("question_statement", 1.0)
        if base._QUANTIFIER_RE.search(prompt): s += 1.0 * lw.get("quantifier", 1.0)
        return s
    
    _SR_PAT = re.compile(r"\b(?:structure|format|organize)\s+your\s+answer\b", re.IGNORECASE)
    _NL_PAT = re.compile(r"\(\d+\)\s+\w+")
    _SUM_RE = re.compile(r"\b(?:summary|reasoning|conclusion|answer|output)\b", re.IGNORECASE)

    def score_math(prompt):
        s = 0.0
        if base._SOLVE_PREFIX_RE.search(prompt): s += 4.0 * mw.get("solve_prefix", 1.0)
        if base._STEP_BY_STEP_RE.search(prompt): s += 3.0 * mw.get("step_by_step", 1.0)
        if base._MATH_OPERATION_RE.search(prompt): s += 3.0 * mw.get("math_operation", 1.0)
        if base._MONEY_RATE_RE.search(prompt): s += 2.0 * mw.get("money_rate", 1.0)
        arith = base._ARITHMETIC_RE.findall(prompt)
        if arith and len(arith) >= 2: s += 3.0 * mw.get("arith_multi", 1.0)
        elif arith: s += 2.0 * mw.get("arith_single", 1.0)
        if base._FRACTION_RE.search(prompt): s += 2.0 * mw.get("fraction", 1.0)
        nums = re.findall(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b", prompt)
        if len(nums) >= 5:
            idx = re.search(r"(?:numbered|from|to|row|column|house|room|floor|stage|level|rank|position)\s+\d+\s*(?:to|through|-)\s*\d+", prompt, re.IGNORECASE)
            if not idx: s += min(len(nums) * 0.3, 2.0) * mw.get("numeric_density", 1.0)
        if re.search(r"\b(how\s+many|how\s+much)\b", prompt, re.IGNORECASE) and not base._WHICH_FOLLOWING_RE.search(prompt):
            s += 2.0 * mw.get("how_many", 1.0)
        if re.search(r"\d+\s*[xX×]\s*\d+\s*[=＝]\s*\d*|\w+\s*=\s*\d+", prompt): s += 2.0 * mw.get("equation", 1.0)
        return s

    def fn(cat, prompt):
        ls = score_logic(prompt)
        ms = score_math(prompt)
        
        _is_fmt = bool(_SR_PAT.search(prompt))
        _has_nl = bool(_NL_PAT.search(prompt))
        is_at = _is_fmt or (_has_nl and bool(_SUM_RE.search(prompt)) and not base._STATEMENT_REASONING_RE.search(prompt) and not base._SYLLOGISM_RE.search(prompt))
        if is_at:
            if base._QUANTIFIER_RE.search(prompt) and not (base._STATEMENT_REASONING_RE.search(prompt) or base._SYLLOGISM_RE.search(prompt) or base._CONSTRAINT_RE.search(prompt) or base._ENTITY_PUZZLE_RE.search(prompt) or base._WHICH_FOLLOWING_RE.search(prompt)):
                ls -= 1.0
        
        if cat not in ("logic", "math"):
            if ls >= 6.0 and ls > ms + dm: return "logic"
            if ms >= 6.0 and ms > ls + dm: return "math"
            return cat
        
        if base._LOGIC_PUZZLE_RE.search(prompt): return "logic"
        names = base._NAME_LIST_RE.findall(prompt)
        has_cc = bool(re.search(r"\b(each|different|assigned|allocated|must|condition|rule|requirement|older|younger|taller|shorter|left|right|between|adjacent|before|after|not\s+(the\s+)?same|no\s+two|if\s|then\s|either|neither|nor)\b", prompt, re.IGNORECASE))
        if names and has_cc and ls >= 3.0: return "logic"
        if base._SYLLOGISM_RE.search(prompt) and ls >= 3.0: return "logic"
        if base._WHICH_FOLLOWING_RE.search(prompt) and base._STATEMENT_REASONING_RE.search(prompt): return "logic"
        if base._CHINESE_LOGIC_RE.search(prompt) and ls >= 3.0: return "logic"
        if base._STATEMENT_REASONING_RE.search(prompt) and ls >= 3.0 and ls > ms: return "logic"
        if base._SOLVE_PREFIX_RE.search(prompt) and ms >= 3.0 and ms > ls: return "math"
        if base._STEP_BY_STEP_RE.search(prompt) and ms >= 3.0 and not base._LOGIC_PUZZLE_RE.search(prompt) and not base._WHICH_FOLLOWING_RE.search(prompt) and not base._CHINESE_LOGIC_RE.search(prompt): return "math"
        if base._MONEY_RATE_RE.search(prompt) and base._ARITHMETIC_RE.search(prompt) and ms >= ls: return "math"
        if ls >= lt and ms >= mt:
            if ls >= ms + dm: return "logic"
            if ms >= ls + dm: return "math"
            return cat
        if ls >= lt and ls > ms: return "logic"
        if ms >= mt and ms > ls: return "math"
        return cat
    return fn

# ══════════════════════════════════════════════════════════════════
# FACTUAL MODULE
# ══════════════════════════════════════════════════════════════════

def make_factual_fn(config):
    base = get_factual_base()
    delta = config.get("factual_override_delta", 1.0)
    fw = config.get("factual_weights", {})
    lw = config.get("logic_weights", {})
    mw = config.get("math_weights", {})
    _NUM_RE = re.compile(r"\d+(?:\.\d+)?")

    def sf(p, lower):
        s = 0.0
        if base._FACTUAL_QA_FORMAT_RE.search(p): s += 6.0 * fw.get("squad_format", 1.0)
        if base._FACTUAL_SOURCE_RE.search(p): s += 4.0 * fw.get("source", 1.0)
        if base._FACTUAL_CHOICES_RE.search(p): s += 5.0 * fw.get("choices", 1.0)
        kq = base._FACTUAL_KNOWLEDGE_QUESTIONS.findall(lower); s += len(kq) * 1.5 * fw.get("knowledge_questions", 1.0)
        if base._FACTUAL_WOTF_RE.search(lower): s += 2.0 * fw.get("factual_wotf", 1.0)
        fd = base._FACTUAL_DOMAIN_RE.findall(lower); s += len(fd) * 2.0 * fw.get("domain", 1.0)
        if base._FACTUAL_DEF_RE.search(lower): s += 3.0 * fw.get("definition", 1.0)
        if re.search(r"\b(context|passage|article|excerpt|paragraph)\s*:", lower) and re.search(r"\b(question\s*:|q\s*:|query\s*:)", lower): s += 3.0 * fw.get("context_question", 1.0)
        return s
    
    def sl(p, lower):
        s = 0.0
        constraints = base._LOGIC_CONSTRAINT_RE.findall(p); s += len(constraints) * 2.0 * lw.get("constraint", 1.0)
        if re.search(r"\bif\b.{0,60}\bthen\b", lower, re.DOTALL): s += 3.0 * lw.get("if_then", 1.0)
        if base._LOGIC_WOTF_RE.search(lower): s += 3.0 * lw.get("logic_wotf", 1.0)
        if base._LOGIC_PUZZLE_STRUCTURE_RE.search(p): s += 2.0 * lw.get("puzzle_structure", 1.0)
        names = base._LOGIC_NAME_DENSITY_RE.findall(p); un = len(set(names))
        if un >= 3 and len(constraints) >= 1: s += un * 0.5 * lw.get("name_density", 1.0)
        if base._LOGIC_EXPLAIN_RE.search(lower): s += 2.0 * lw.get("explain", 1.0)
        cw = {"each","every","all","none","no","neither","either","both","only","unless","except","must","if","then","hence","thus","therefore"}
        cc = sum(1 for w in cw if re.search(rf"\b{w}\b", lower))
        if cc >= 4: s += 3.0 * lw.get("constraint_density_high", 1.0)
        elif cc >= 3: s += 1.5 * lw.get("constraint_density_mid", 1.0)
        return s
    
    def sm(p, lower):
        s = 0.0
        cm = base._MATH_CALC_RE.findall(lower); s += len(cm) * 2.0 * mw.get("calc_keywords", 1.0)
        if base._MATH_OPERATORS_RE.search(p): s += 2.5 * mw.get("operators", 1.0)
        if base._MATH_WORDPROB_RE.search(lower): s += 2.0 * mw.get("wordprob", 1.0)
        if re.search(r"^(?:solve|problem|question)\b", lower): s += 2.0 * mw.get("solve_prefix", 1.0)
        nd = base._numeric_density(p)
        if nd > 5 and (s > 0 or len(_NUM_RE.findall(p)) >= 3): s += 2.0 * mw.get("density_high", 1.0)
        elif nd > 3 and s > 0: s += 1.0 * mw.get("density_mid", 1.0)
        if re.search(r"\$.*?[\\\\=+*/^_{}\d].*?\$|\\\\[a-zA-Z]+", p): s += 2.0 * mw.get("latex", 1.0)
        return s

    def fn(cat, prompt):
        if cat not in ("factual", "logic", "math"): return cat
        lower = prompt.lower()
        fs = sf(prompt, lower); ls = sl(prompt, lower); ms = sm(prompt, lower)
        if base._has_squad_structure(prompt, lower): return "factual"
        if base._has_mmlu_structure(prompt, lower):
            cons = base._LOGIC_CONSTRAINT_RE.findall(prompt)
            if len(cons) < 3: return "factual"
            cw = {"each","every","all","must","if","then","either","neither","unless"}
            cc = sum(1 for w in cw if re.search(rf"\b{w}\b", lower))
            if cc < 4: return "factual"
        if base._FACTUAL_SOURCE_RE.search(prompt) and fs >= 3.0: return "factual"
        if base._FACTUAL_DEF_RE.search(lower) and fs >= 3.0: return "factual"
        if cat == "factual":
            if base._is_reasoning_intent(prompt, lower) and ls > fs + delta: return "logic"
            if base._is_calculation_intent(prompt, lower) and ms > fs + delta: return "math"
            if ls > fs + 3.0 and base._is_reasoning_intent(prompt, lower): return "logic"
            if ms > fs + 3.0 and base._is_calculation_intent(prompt, lower): return "math"
            return "factual"
        if cat == "logic":
            _lq = re.search(r"question\s*:.*?(?:infer(?:red|ence)?|conclu(?:de|ded|ding|sion|sive)|deduc(?:e|ed|ing|tion)|imply|implied|implication|assum(?:e|ed|ing|ption)|weaken|strengthen|justify|support|must\s+be\s+(?:true|false)|can\s+be\s+(?:inferred|concluded|deduced)|refu(?:te|ting|tal)|explain(?:\s+the\s+above|\s+this|\s+the\s+seemingly)|anomal(y|ies)|closest\s+to\s+the\s+meaning|best\s+(?:explain|refute|describes?|characterizes?|account|argument)|argument(?:\s+against|\s+take\s+place|s\s+above)?|reasoning|logically|raise\s+(?:the\s+most\s+)?doubts?|opinions?\s+of\s+the\s+above|above\s+argument|above\s+(?:reasoning|conclusion|speculation|point))", lower, re.DOTALL)
            if _lq: return "logic"
            if fs >= 4.0 and fs > ls + delta: return "factual"
            hk = bool(base._FACTUAL_KNOWLEDGE_QUESTIONS.search(lower)) or bool(base._FACTUAL_DOMAIN_RE.search(lower))
            hwr = not base._is_reasoning_intent(prompt, lower)
            if hk and hwr and fs >= 2.0: return "factual"
            return "logic"
        if cat == "math":
            if fs >= 4.0 and fs > ms + delta: return "factual"
            hk = bool(base._FACTUAL_KNOWLEDGE_QUESTIONS.search(lower)) or bool(base._FACTUAL_DOMAIN_RE.search(lower))
            hnc = not base._is_calculation_intent(prompt, lower)
            if hk and hnc and fs >= 2.0: return "factual"
            return "math"
        return cat
    return fn


# ══════════════════════════════════════════════════════════════════
# MAIN TUNING
# ══════════════════════════════════════════════════════════════════

def main():
    timelog("Loading data...")
    train_raw = load_data(TRAIN_PATH)
    val_raw = load_data(VAL_PATH)
    print(f"  Train: {len(train_raw)} items, Val: {len(val_raw)} items")
    
    timelog("Loading primary classifier...")
    primary = load_primary()
    
    timelog("Pre-computing primary predictions...")
    train_data = precompute_primary(primary, train_raw)
    val_data = precompute_primary(primary, val_raw)
    
    # Count categories
    train_cats = Counter(t[0] for t in train_data)
    val_cats = Counter(t[0] for t in val_data)
    print(f"  Train distribution: {dict(train_cats)}")
    print(f"  Val distribution:   {dict(val_cats)}")
    
    # Baseline: primary only accuracy
    primary_correct = sum(1 for true, prim, _ in train_data if prim == true)
    primary_train = primary_correct / len(train_data)
    primary_correct_v = sum(1 for true, prim, _ in val_data if prim == true)
    primary_val = primary_correct_v / len(val_data)
    timelog(f"Baseline (primary only): train={primary_train:.4f}, val={primary_val:.4f}")
    
    # Default secondary
    default_code = make_code_fn({})
    default_reason = make_reason_fn({})
    default_factual = make_factual_fn({})
    def_train = evaluate_full(train_data, default_code, default_reason, default_factual)
    def_val = evaluate_full(val_data, default_code, default_reason, default_factual)
    timelog(f"Baseline (default secondaries): train={def_train:.4f}, val={def_val:.4f}")
    
    # Code items filter (for focused tuning)
    code_train = [(t, p, r) for t, p, r in train_data if t in ("code_debug", "code_gen")]
    code_val = [(t, p, r) for t, p, r in val_data if t in ("code_debug", "code_gen")]
    print(f"  Code items: train={len(code_train)}, val={len(code_val)}")
    
    # ════════════════════════════════════════════════════════════════
    # CODE TUNING
    # ════════════════════════════════════════════════════════════════
    timelog("=== CODE MODULE TUNING ===")
    
    # Fast sweep of thresholds
    timelog("  Phase 1: Threshold sweep...")
    code_best = (0.0, {}, 0.0)
    trials = 0
    
    for dt in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        for gt in [2.0, 3.0, 4.0, 5.0]:
            for ratio in [1.2, 1.4, 1.6, 1.8, 2.0]:
                cfg = {"debug_threshold": dt, "gen_threshold": gt, "ratio": ratio}
                fn = make_code_fn(cfg)
                acc = evaluate_secondary(train_data, {"code_debug", "code_gen"}, fn)
                trials += 1
                if acc > code_best[0]:
                    val_acc = evaluate_secondary(val_data, {"code_debug", "code_gen"}, fn)
                    code_best = (acc, cfg, val_acc)
                    print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - dt={dt:.0f} gt={gt:.0f} r={ratio:.1f}")
    
    timelog(f"  Phase 1 done ({trials} trials). Best: {code_best[0]:.4f}")
    best_code_cfg = code_best[1]
    
    # Quick weight sweep around best thresholds
    timelog("  Phase 2: Weight sweep...")
    dt, gt, r = best_code_cfg["debug_threshold"], best_code_cfg["gen_threshold"], best_code_cfg["ratio"]
    
    for ftb in [0.8, 1.0, 1.2, 1.5]:
        for bug_w in [0.7, 1.0, 1.3]:
            for wf in [0.8, 1.0, 1.3, 1.5]:
                cfg = {"debug_threshold": dt, "gen_threshold": gt, "ratio": r,
                       "debug_weights": {"fix_the_bug": ftb, "bug": bug_w},
                       "gen_weights": {"write_function": wf}}
                fn = make_code_fn(cfg)
                acc = evaluate_secondary(train_data, {"code_debug", "code_gen"}, fn)
                trials += 1
                if acc > code_best[0]:
                    val_acc = evaluate_secondary(val_data, {"code_debug", "code_gen"}, fn)
                    code_best = (acc, cfg, val_acc)
                    print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - ftb={ftb:.1f} bug={bug_w:.1f} wf={wf:.1f}")
    
    timelog(f"  Best code config: train={code_best[0]:.4f}, val={code_best[2]:.4f}")
    best_code_fn = make_code_fn(code_best[1])
    
    # ════════════════════════════════════════════════════════════════
    # REASONING TUNING
    # ════════════════════════════════════════════════════════════════
    timelog("=== REASONING MODULE TUNING ===")
    
    def eval_reason(train_or_val, code_fn):
        """Evaluate with a given reasoning fn + best code + default factual."""
        def fake_reason(cat, prompt): return cat  # placeholder
        def test_with_reason(reason_fn):
            correct = 0
            total = len(train_or_val)
            for true_cat, primary_cat, prompt in train_or_val:
                cat = primary_cat
                if cat in ("code_debug", "code_gen"):
                    corrected = code_fn(cat, prompt)
                    if corrected != cat: cat = corrected
                if cat in ("logic", "math"):
                    corrected = reason_fn(cat, prompt)
                    if corrected != cat: cat = corrected
                if cat == true_cat: correct += 1
            return correct / total
        return test_with_reason
    
    def eval_reason_combined(train_or_val, code_fn, reason_fn, factual_fn):
        """Full pipeline with given functions."""
        correct = 0
        total = len(train_or_val)
        for true_cat, primary_cat, prompt in train_or_val:
            cat = primary_cat
            if cat in ("code_debug", "code_gen"):
                corrected = code_fn(cat, prompt)
                if corrected != cat: cat = corrected
            if cat in ("logic", "math"):
                corrected = reason_fn(cat, prompt)
                if corrected != cat: cat = corrected
            corrected = factual_fn(cat, prompt)
            if corrected != cat: cat = corrected
            if cat == true_cat: correct += 1
        return correct / total
    
    # Phase 1: Threshold/margin sweep
    timelog("  Phase 1: Threshold/margin sweep...")
    reason_best = (0.0, {}, 0.0)
    trials = 0
    
    for lt_val in [2.0, 3.0, 4.0, 5.0]:
        for mt_val in [2.0, 3.0, 4.0, 5.0]:
            for dm_val in [0.5, 1.0, 1.5, 2.0]:
                cfg = {"logic_threshold": lt_val, "math_threshold": mt_val, "decisive_margin": dm_val}
                fn = make_reason_fn(cfg)
                acc = eval_reason_combined(train_data, best_code_fn, fn, default_factual)
                trials += 1
                if acc > reason_best[0]:
                    val_acc = eval_reason_combined(val_data, best_code_fn, fn, default_factual)
                    reason_best = (acc, cfg, val_acc)
                    print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - lt={lt_val:.0f} mt={mt_val:.0f} dm={dm_val:.1f}")
    
    timelog(f"  Phase 1 done ({trials} trials). Best: {reason_best[0]:.4f}")
    best_reason_cfg = reason_best[1]
    
    # Phase 2: Quick weight sweep
    timelog("  Phase 2: Weight sweep...")
    lt_val, mt_val, dm_val = best_reason_cfg["logic_threshold"], best_reason_cfg["math_threshold"], best_reason_cfg["decisive_margin"]
    
    for lp in [0.7, 1.0, 1.3]:
        for sy in [0.7, 1.0, 1.3]:
            for cn in [0.7, 1.0, 1.3]:
                cfg = {"logic_threshold": lt_val, "math_threshold": mt_val, "decisive_margin": dm_val,
                       "logic_weights": {"logic_puzzle": lp, "syllogism": sy, "constraint": cn}}
                fn = make_reason_fn(cfg)
                acc = eval_reason_combined(train_data, best_code_fn, fn, default_factual)
                trials += 1
                if acc > reason_best[0]:
                    val_acc = eval_reason_combined(val_data, best_code_fn, fn, default_factual)
                    reason_best = (acc, cfg, val_acc)
                    print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - lp={lp:.1f} sy={sy:.1f} cn={cn:.1f}")
    
    # Also try math weight variations
    for sp in [0.7, 1.0, 1.3]:
        for mo in [0.7, 1.0, 1.3]:
            cfg = {"logic_threshold": lt_val, "math_threshold": mt_val, "decisive_margin": dm_val,
                   "math_weights": {"solve_prefix": sp, "math_operation": mo}}
            fn = make_reason_fn(cfg)
            acc = eval_reason_combined(train_data, best_code_fn, fn, default_factual)
            trials += 1
            if acc > reason_best[0]:
                val_acc = eval_reason_combined(val_data, best_code_fn, fn, default_factual)
                reason_best = (acc, cfg, val_acc)
                print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - sp={sp:.1f} mo={mo:.1f}")
    
    timelog(f"  Best reason config: train={reason_best[0]:.4f}, val={reason_best[2]:.4f}")
    best_reason_fn = make_reason_fn(reason_best[1])
    
    # ════════════════════════════════════════════════════════════════
    # FACTUAL TUNING
    # ════════════════════════════════════════════════════════════════
    timelog("=== FACTUAL MODULE TUNING ===")
    
    def eval_factual_combined(train_or_val, code_fn, reason_fn, factual_fn):
        correct = 0
        total = len(train_or_val)
        for true_cat, primary_cat, prompt in train_or_val:
            cat = primary_cat
            if cat in ("code_debug", "code_gen"):
                corrected = code_fn(cat, prompt)
                if corrected != cat: cat = corrected
            if cat in ("logic", "math"):
                corrected = reason_fn(cat, prompt)
                if corrected != cat: cat = corrected
            corrected = factual_fn(cat, prompt)
            if corrected != cat: cat = corrected
            if cat == true_cat: correct += 1
        return correct / total
    
    timelog("  Phase 1: Delta sweep...")
    factual_best = (0.0, {}, 0.0)
    trials = 0
    
    for d in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        cfg = {"factual_override_delta": d, "math_override_delta": d}
        fn = make_factual_fn(cfg)
        acc = eval_factual_combined(train_data, best_code_fn, best_reason_fn, fn)
        trials += 1
        print(f"    delta={d:.1f}: train={acc:.4f}")
        if acc > factual_best[0]:
            val_acc = eval_factual_combined(val_data, best_code_fn, best_reason_fn, fn)
            factual_best = (acc, cfg, val_acc)
            print(f"      ★ New best: val={val_acc:.4f}")
    
    timelog(f"  Best delta: {factual_best[1]['factual_override_delta']:.1f}")
    best_delta = factual_best[1]["factual_override_delta"]
    
    # Phase 2: Quick weight sweep
    timelog("  Phase 2: Weight sweep...")
    for sq in [0.7, 1.0, 1.3]:
        for ch in [0.7, 1.0, 1.3]:
            for dm in [0.7, 1.0, 1.3]:
                for lc in [0.7, 1.0, 1.3]:
                    cfg = {"factual_override_delta": best_delta, "math_override_delta": best_delta,
                           "factual_weights": {"squad_format": sq, "choices": ch, "domain": dm},
                           "logic_weights": {"constraint": lc}}
                    fn = make_factual_fn(cfg)
                    acc = eval_factual_combined(train_data, best_code_fn, best_reason_fn, fn)
                    trials += 1
                    if acc > factual_best[0]:
                        val_acc = eval_factual_combined(val_data, best_code_fn, best_reason_fn, fn)
                        factual_best = (acc, cfg, val_acc)
                        print(f"    ★ New best: train={acc:.4f}, val={val_acc:.4f} - sq={sq:.1f} ch={ch:.1f} dom={dm:.1f} lc={lc:.1f}")
    
    timelog(f"  Best factual config: train={factual_best[0]:.4f}, val={factual_best[2]:.4f}")
    best_factual_fn = make_factual_fn(factual_best[1])
    
    # ════════════════════════════════════════════════════════════════
    # FINAL
    # ════════════════════════════════════════════════════════════════
    final_train = eval_factual_combined(train_data, best_code_fn, best_reason_fn, best_factual_fn)
    final_val = eval_factual_combined(val_data, best_code_fn, best_reason_fn, best_factual_fn)
    
    timelog("=" * 60)
    timelog("FINAL RESULTS")
    timelog("=" * 60)
    
    print(f"""
Baseline (primary only):     train={primary_train:.4f}, val={primary_val:.4f}
Baseline (default sec.):     train={def_train:.4f}, val={def_val:.4f}

=== Best secondary_code.py Config ===
  Config: {code_best[1]}
  Train:  {code_best[0]:.4f}
  Val:    {code_best[2]:.4f}

=== Best secondary_reasoning.py Config ===
  Config: {reason_best[1]}
  Train:  {reason_best[0]:.4f}
  Val:    {reason_best[2]:.4f}

=== Best secondary_factual.py Config ===
  Config: {factual_best[1]}
  Train:  {factual_best[0]:.4f}
  Val:    {factual_best[2]:.4f}

=== COMBINED ===
  Train: {def_train:.4f} -> {final_train:.4f} (Δ{final_train-def_train:+.4f})
  Val:   {def_val:.4f} -> {final_val:.4f} (Δ{final_val-def_val:+.4f})
""")
    
    timelog("DONE")


if __name__ == "__main__":
    main()
