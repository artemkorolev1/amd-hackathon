#!/usr/bin/env python3
"""
Benchmark sentiment solver — both regex-based and VADER-based.
Tests all sentiment questions with gold answers across all eval datasets.
"""
import json, os, sys, re
from collections import defaultdict
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from agent.category_filter import get_short_name
from agent.solvers.deterministic import solve_sentiment as regex_solve_sentiment
from agent.solvers.deterministic import _classify_sentiment as regex_classify

# VADER import
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

KNOWN_PATHS = [
    "input/dev_40.json",
    "input/heldout_40.json",
    "input/complexity_40.json",
    "input/cx_300.json",
    "data/eval/primary/eval_60_medium_hard.json",
    "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json",
    "data/eval/primary/eval_mini_10.json",
    "data/eval/training-v1.json",
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
    "data/eval/validation-v3.json",
    "data/eval/tests/complexity_eval_40.json",
    "data/eval/tests/complexity_eval_candidates.json",
    "data/eval/tests/eval_longform_20.json",
    "data/eval/tests/eval_v14_test_20.json",
    "data/eval/tests/eval_v14_remaining_20.json",
    "data/eval/tests/eval_v14_timeout_stress_19.json",
    "data/eval/tests/fireworks_eval_20.json",
    "data/eval/generated/build-A-40.json",
    "data/eval/generated/build-B-40.json",
    "data/eval/generated/eval_from_datasets_20260712_172357.json",
    "data/eval/generated/eval_from_datasets_20260712_172426.json",
    "data/eval/generated/eval_from_datasets_20260712_172443.json",
]

def resolve_path(p):
    if os.path.isabs(p): return p
    return os.path.join(_HERE, p)

def load_sentiment_questions():
    """Load all sentiment questions with gold answers."""
    questions = []
    for path in KNOWN_PATHS:
        fp = resolve_path(path)
        if not os.path.isfile(fp): continue
        with open(fp) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("questions", data.get("items", [data]))
        for idx, item in enumerate(items):
            if not isinstance(item, dict): continue
            cat = item.get("category") or item.get("category_label") or item.get("label") or item.get("label_8way") or ""
            short = get_short_name(cat)
            if short != "sentiment": continue
            prompt = item.get("prompt", item.get("question", ""))
            gold = item.get("gold", {})
            gold_answer = ""
            if isinstance(gold, dict):
                gold_answer = gold.get("answer", "") 
            elif isinstance(gold, str):
                gold_answer = gold
            if not gold_answer:
                gold_answer = item.get("expected_answer", "")
            questions.append({
                "task_id": item.get("task_id", f"{os.path.basename(path)}_q{idx}"),
                "prompt": prompt,
                "gold_answer": gold_answer,
                "source": path,
            })
    return questions

# Normalize gold answers to a canonical form
def normalize_gold(gold):
    """Extract the core sentiment label from gold answer."""
    g = gold.lower().strip()
    # Extract the key sentiment label from verbose gold answers
    for label in ["positive", "negative", "neutral", "mixed"]:
        if label in g:
            return label
    return g

# === VADER configs to try ===
VADER_CONFIGS = [
    {
        "name": "vader_default",
        "desc": "Standard VADER thresholds (compound >= 0.05 POS, <= -0.05 NEG), no MIXED",
        "pos_thresh": 0.05,
        "neg_thresh": -0.05,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
        "enable_mixed": False,
    },
    {
        "name": "vader_strict",
        "desc": "Stricter thresholds (compound >= 0.2 POS, <= -0.2 NEG), more NEUTRAL",
        "pos_thresh": 0.2,
        "neg_thresh": -0.2,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
        "enable_mixed": False,
    },
    {
        "name": "vader_mixed_enabled",
        "desc": "Default thresholds + MIXED detection (pos>0.3 and neg>0.3)",
        "pos_thresh": 0.05,
        "neg_thresh": -0.05,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
        "enable_mixed": True,
    },
    {
        "name": "vader_loose",
        "desc": "Loose thresholds (compound >= 0.0 POS, < 0.0 NEG), fewer NEUTRAL",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
        "enable_mixed": False,
    },
    {
        "name": "vader_very_strict",
        "desc": "Very strict (compound >= 0.4 POS, <= -0.4 NEG), many NEUTRAL",
        "pos_thresh": 0.4,
        "neg_thresh": -0.4,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
        "enable_mixed": False,
    },
    {
        "name": "vader_mixed_loose",
        "desc": "Loose thresholds + MIXED detection",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "mixed_pos": 0.2,
        "mixed_neg": 0.2,
        "enable_mixed": True,
    },
    {
        "name": "vader_mixed_only_high",
        "desc": "Strict thresholds + MIXED with higher bar",
        "pos_thresh": 0.15,
        "neg_thresh": -0.15,
        "mixed_pos": 0.4,
        "mixed_neg": 0.4,
        "enable_mixed": True,
    },
]

def classify_vader(text, config):
    """Classify sentiment using VADER with given config."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    
    # Check for MIXED
    if config["enable_mixed"] and pos > config["mixed_pos"] and neg > config["mixed_neg"]:
        return "MIXED"
    
    if compound >= config["pos_thresh"]:
        return "POSITIVE"
    elif compound <= config["neg_thresh"]:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def extract_target_text(prompt):
    """Extract the text to analyze from a sentiment prompt."""
    # Try to extract quoted text first
    quoted = re.findall(r'"([^"]+)"', prompt)
    if quoted:
        return quoted[-1]  # Usually the last quoted text is the review
    
    # Try to extract after "text:" markers
    m = re.search(r'(?:review|text|statement|passage|following)[\s:]*[:](.+?)$', prompt, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip().rstrip('.')
        # Remove trailing instruction like "Respond in JSON"
        text = re.sub(r'\s*Respond\s+(in|with).*', '', text, flags=re.IGNORECASE | re.DOTALL)
        # Remove "Keep your answer under 50 words"
        text = re.sub(r'\s*Keep your.*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    # Try after ":-" or just the whole thing
    m = re.search(r':\s*(.+?)$', prompt, re.DOTALL)
    if m:
        text = m.group(1).strip()
        text = re.sub(r'\s*Respond\s+(in|with).*', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()
    
    return prompt

def score_accuracy(predictions, gold_map):
    """Calculate accuracy, counting a prediction as correct if the normalized label matches."""
    correct = 0
    total = 0
    details = []
    for tid, pred in predictions.items():
        gold = gold_map.get(tid, "")
        if not gold:
            continue
        gold_norm = normalize_gold(gold)
        pred_norm = pred.lower().strip() if pred else "none"
        is_correct = gold_norm == pred_norm
        if is_correct:
            correct += 1
        total += 1
        details.append((tid, pred_norm, gold_norm, gold, is_correct))
    acc = correct / total * 100 if total > 0 else 0
    return acc, correct, total, details

def main():
    questions = load_sentiment_questions()
    print(f"Loaded {len(questions)} sentiment questions with gold answers\n")
    
    # Build gold map
    gold_map = {q["task_id"]: q["gold_answer"] for q in questions}
    
    # === BASELINE: regex solver ===
    print("=" * 80)
    print("BASELINE: Regex-based sentiment solver")
    print("=" * 80)
    regex_preds = {}
    for q in questions:
        ans = regex_solve_sentiment(q["prompt"], "sentiment")
        regex_preds[q["task_id"]] = ans if ans else "none"
    
    acc, correct, total, details = score_accuracy(regex_preds, gold_map)
    print(f"Regex solver accuracy: {acc:.1f}% ({correct}/{total})")
    
    # Show some examples
    errors = [(tid, p, g, gg) for tid, p, g, gg, ok in details if not ok]
    print(f"\nSample errors ({min(10, len(errors))} shown):")
    for tid, pred, gold_norm, gold_raw in errors[:10]:
        q = [q for q in questions if q["task_id"] == tid][0]
        target = extract_target_text(q["prompt"])
        print(f"  [{tid}] gold={gold_raw} | pred={pred} | text='{target[:80]}...'")
    
    # === VADER configs ===
    print("\n" + "=" * 80)
    print("VADER CONFIGURATIONS")
    print("=" * 80)
    
    results = []
    for config in VADER_CONFIGS:
        print(f"\n--- Config: {config['name']} ---")
        print(f"  {config['desc']}")
        vader_preds = {}
        for q in questions:
            target = extract_target_text(q["prompt"])
            ans = classify_vader(target, config)
            vader_preds[q["task_id"]] = ans
        
        acc, correct, total, details = score_accuracy(vader_preds, gold_map)
        print(f"  Accuracy: {acc:.1f}% ({correct}/{total})")
        
        # Per-gold accuracy
        gold_labels = defaultdict(lambda: {"correct": 0, "total": 0})
        for tid, pn, gn, gr, ok in details:
            gold_labels[gn]["total"] += 1
            if ok:
                gold_labels[gn]["correct"] += 1
        
        for label in sorted(gold_labels.keys()):
            v = gold_labels[label]
            pct = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"    Gold={label}: {v['correct']}/{v['total']} ({pct:.1f}%)")
        
        results.append((config["name"], acc, correct, total, config))
    
    # Find best config
    best = max(results, key=lambda r: r[1])
    print(f"\n{'=' * 80}")
    print(f"BEST CONFIG: {best[0]} with {best[1]:.1f}% accuracy ({best[2]}/{best[3]})")
    print(f"{'=' * 80}")
    
    # Show per-config comparison
    print(f"\n{'Config':<25} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 50)
    for name, acc, correct, total, _ in sorted(results, key=lambda r: -r[1]):
        print(f"{name:<25} {acc:>8.1f}%  {correct:>4}/{total:<4}")
    
    # Detailed error analysis for best config
    print(f"\n{'=' * 80}")
    print(f"ERROR ANALYSIS: Best config ({best[0]})")
    print(f"{'=' * 80}")
    
    best_config = best[4]
    best_preds = {}
    for q in questions:
        target = extract_target_text(q["prompt"])
        best_preds[q["task_id"]] = classify_vader(target, best_config)
    
    _, _, _, best_details = score_accuracy(best_preds, gold_map)
    errors = [(tid, p, g, gg) for tid, p, g, gg, ok in best_details if not ok]
    
    # Categorize errors
    error_categories = defaultdict(list)
    for tid, pred, gold_norm, gold_raw in errors:
        q = next((q for q in questions if q["task_id"] == tid), None)
        target = extract_target_text(q["prompt"]) if q else ""
        error_categories[f"gold={gold_norm}_pred={pred}"].append((tid, target[:100], gold_raw))
    
    print(f"Total errors: {len(errors)}/{best[3]}")
    print("\nError patterns:")
    for pattern, items in sorted(error_categories.items(), key=lambda x: -len(x[1])):
        print(f"\n  {pattern}: {len(items)} errors")
        for tid, text, gold_raw in items[:3]:
            print(f"    [{tid}] text='{text}'")
    
    # Regex vs VADER comparison
    print(f"\n{'=' * 80}")
    print("REGEX vs VADER (best config) comparison")
    print(f"{'=' * 80}")
    
    both_correct = 0
    regex_only = 0
    vader_only = 0
    both_wrong = 0
    
    for q in questions:
        tid = q["task_id"]
        gold = gold_map.get(tid, "")
        if not gold:
            continue
        gold_norm = normalize_gold(gold)
        
        r = regex_preds.get(tid, "none")
        v = best_preds.get(tid, "none").lower()
        
        r_ok = r == gold_norm
        v_ok = v == gold_norm
        
        if r_ok and v_ok:
            both_correct += 1
        elif r_ok and not v_ok:
            regex_only += 1
        elif not r_ok and v_ok:
            vader_only += 1
        else:
            both_wrong += 1
    
    print(f"Both correct:   {both_correct}")
    print(f"Regex only:     {regex_only} (VADER regressed)")
    print(f"VADER only:     {vader_only} (VADER improved)")
    print(f"Both wrong:     {both_wrong}")
    print(f"Net improvement: {vader_only - regex_only}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
