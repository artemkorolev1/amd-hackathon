#!/usr/bin/env python3
"""
Final comprehensive VADER sentiment solver benchmark.
Tests the integrated VADER solver (which uses solve_sentiment with VADER)
plus standalone VADER configs for parameter tuning.
"""
import json, os, sys, re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

# Import the actual solver module
from agent.solvers.deterministic import (
    solve_sentiment as vader_solve_sentiment,
    _VADER_POS_THRESH, _VADER_NEG_THRESH,
)
from agent.category_filter import get_short_name

# VADER direct access
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

def normalize_gold(gold):
    """Extract the core sentiment label from gold answer."""
    g = gold.lower().strip()
    for label in ["positive", "negative", "neutral", "mixed"]:
        if label in g:
            return label
    return g

def classify_vader_direct(text, config):
    """Classify sentiment using VADER directly."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    
    if config.get("enable_mixed") and pos > config.get("mixed_pos", 0.3) and neg > config.get("mixed_neg", 0.3):
        return "mixed"
    
    if compound >= config["pos_thresh"]:
        return "positive"
    elif compound <= config["neg_thresh"]:
        return "negative"
    else:
        return "neutral"

def extract_target_text(prompt):
    """Extract the actual text to analyze from a sentiment prompt."""
    quoted = re.findall(r'"([^"]+)"', prompt)
    if quoted:
        return quoted[-1]
    
    m = re.search(r'(?:review|text|statement|passage|following)[\s:]*[:]\s*(.+?)$', prompt, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip().rstrip('.')
        text = re.sub(r'\s*Respond\s+(in|with).*', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\s*Keep your.*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    m = re.search(r':\s*(.+?)$', prompt, re.DOTALL)
    if m:
        text = m.group(1).strip()
        text = re.sub(r'\s*Respond\s+(in|with).*', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()
    
    return prompt

VADER_CONFIGS = [
    {
        "name": "vader_loose",
        "desc": "compound>=0.0=POS, compound<0.0=NEG, no MIXED (best individual config)",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,  # compound < 0 → NEG, compound == 0 → POS (rare)
        "enable_mixed": False,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
    {
        "name": "vader_default",
        "desc": "compound>=0.05=POS, compound<=-0.05=NEG, no MIXED (standard VADER)",
        "pos_thresh": 0.05,
        "neg_thresh": -0.05,
        "enable_mixed": False,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
    {
        "name": "vader_strict",
        "desc": "compound>=0.2=POS, compound<=-0.2=NEG, no MIXED",
        "pos_thresh": 0.2,
        "neg_thresh": -0.2,
        "enable_mixed": False,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
    {
        "name": "vader_very_strict",
        "desc": "compound>=0.4=POS, compound<=-0.4=NEG, no MIXED",
        "pos_thresh": 0.4,
        "neg_thresh": -0.4,
        "enable_mixed": False,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
    {
        "name": "vader_mixed_loose",
        "desc": "compound>=0.0=POS, compound<0.0=NEG, MIXED when pos>0.2 and neg>0.2",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "enable_mixed": True,
        "mixed_pos": 0.2,
        "mixed_neg": 0.2,
    },
    {
        "name": "vader_mixed_default",
        "desc": "compound>=0.05=POS, compound<=-0.05=NEG, MIXED when pos>0.3 and neg>0.3",
        "pos_thresh": 0.05,
        "neg_thresh": -0.05,
        "enable_mixed": True,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
    {
        "name": "vader_neutral_zone",
        "desc": "compound>=0.1=POS, compound<=-0.1=NEG, balanced neutral zone",
        "pos_thresh": 0.1,
        "neg_thresh": -0.1,
        "enable_mixed": False,
        "mixed_pos": 0.3,
        "mixed_neg": 0.3,
    },
]

def score_accuracy(predictions, gold_map):
    correct = 0
    total = 0
    details = []
    for tid, pred in predictions.items():
        gold = gold_map.get(tid, "")
        if not gold:
            continue
        gold_norm = normalize_gold(gold)
        pred_norm = pred.lower().strip() if pred else "none"
        # Allow "none" to match "none" (but we won't have any)
        if pred_norm == "none":
            is_correct = False
        else:
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
    
    gold_map = {q["task_id"]: q["gold_answer"] for q in questions}
    
    # === TEST 1: Integrated VADER solver (via solve_sentiment) ===
    print("=" * 80)
    print("TEST 1: Integrated VADER solver (solve_sentiment with default config)")
    print("=" * 80)
    
    integrated_preds = {}
    for q in questions:
        ans = vader_solve_sentiment(q["prompt"], "sentiment")
        integrated_preds[q["task_id"]] = ans if ans else "none"
    
    acc_i, correct_i, total_i, details_i = score_accuracy(integrated_preds, gold_map)
    print(f"Integrated VADER solver accuracy: {acc_i:.1f}% ({correct_i}/{total_i})")
    
    # Show error examples
    errors_i = [(tid, p, g, gg) for tid, p, g, gg, ok in details_i if not ok]
    print(f"\nSample errors ({min(5, len(errors_i))} shown):")
    for tid, pred, gold_norm, gold_raw in errors_i[:5]:
        q = [q for q in questions if q["task_id"] == tid][0]
        target = extract_target_text(q["prompt"])
        print(f"  [{tid}] gold={gold_raw} | pred={pred} | text='{target[:100]}...'")
    
    # === TEST 2: Direct VADER on extracted target text ===
    print("\n" + "=" * 80)
    print("TEST 2: Direct VADER on extracted target text — config sweep")
    print("=" * 80)
    
    # Extract target texts
    target_texts = {}
    for q in questions:
        target_texts[q["task_id"]] = extract_target_text(q["prompt"])
    
    all_results = []
    for config in VADER_CONFIGS:
        print(f"\n--- Config: {config['name']} ---")
        print(f"  {config['desc']}")
        
        config_preds = {}
        for q in questions:
            ans = classify_vader_direct(target_texts[q["task_id"]], config)
            config_preds[q["task_id"]] = ans
        
        acc, correct, total, details = score_accuracy(config_preds, gold_map)
        print(f"  Accuracy: {acc:.1f}% ({correct}/{total})")
        
        gold_labels = defaultdict(lambda: {"correct": 0, "total": 0})
        for tid, pn, gn, gr, ok in details:
            gold_labels[gn]["total"] += 1
            if ok:
                gold_labels[gn]["correct"] += 1
        
        for label in sorted(gold_labels.keys()):
            v = gold_labels[label]
            pct = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"    Gold={label}: {v['correct']}/{v['total']} ({pct:.1f}%)")
        
        all_results.append((config["name"], acc, correct, total, config))
    
    # Sort and show results
    print("\n" + "=" * 80)
    print("CONFIG RANKINGS")
    print("=" * 80)
    print(f"{'Config':<25} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 50)
    for name, acc, correct, total, _ in sorted(all_results, key=lambda r: -r[1]):
        print(f"{name:<25} {acc:>8.1f}%  {correct:>4}/{total:<4}")
    
    best = max(all_results, key=lambda r: r[1])
    print(f"\nBEST CONFIG: {best[0]} with {best[1]:.1f}% accuracy ({best[2]}/{best[3]})")
    
    # === TEST 3: Compare integrated solver vs best direct VADER ===
    print("\n" + "=" * 80)
    print("TEST 3: Comparison — Integrated solver vs Direct VADER (best)")
    print("=" * 80)
    
    best_config = best[4]
    best_direct_preds = {}
    for q in questions:
        best_direct_preds[q["task_id"]] = classify_vader_direct(target_texts[q["task_id"]], best_config)
    
    # Compare integrated vs direct
    agreements = 0
    integrated_correct_direct_wrong = 0
    integrated_wrong_direct_correct = 0
    both_wrong = 0
    
    for tid in gold_map:
        if not gold_map[tid]:
            continue
        ip = integrated_preds.get(tid, "none")
        dp = best_direct_preds.get(tid, "none").lower()
        gold_norm = normalize_gold(gold_map[tid])
        
        i_ok = ip == gold_norm
        d_ok = dp == gold_norm
        
        if i_ok and d_ok:
            agreements += 1
        elif i_ok and not d_ok:
            integrated_correct_direct_wrong += 1
        elif not i_ok and d_ok:
            integrated_wrong_direct_correct += 1
        else:
            both_wrong += 1
    
    print(f"Both correct:            {agreements}")
    print(f"Integrated only correct: {integrated_correct_direct_wrong}")
    print(f"Direct VADER only correct: {integrated_wrong_direct_correct}")
    print(f"Both wrong:              {both_wrong}")
    
    # If integrated solver's text extraction differs from direct, show examples
    diff_examples = []
    for q in questions[:20]:
        tid = q["task_id"]
        if integrated_preds.get(tid, "none") != best_direct_preds.get(tid, "none").lower():
            ip = integrated_preds.get(tid, "none")
            dp = best_direct_preds.get(tid, "none").lower()
            diff_examples.append((tid, ip, dp, q["prompt"][:120]))
    
    if diff_examples:
        print(f"\nDifferences between integrated and direct ({len(diff_examples)} total across all Q):")
        for tid, ip, dp, prompt in diff_examples[:5]:
            print(f"  [{tid}] integrated={ip} vs direct={dp}")
            print(f"         prompt: {prompt}")
    
    # === FINAL: Detailed error analysis for best config ===
    print("\n" + "=" * 80)
    print(f"FINAL: Error analysis for best config ({best[0]})")
    print("=" * 80)
    
    _, _, _, best_details = score_accuracy(best_direct_preds, gold_map)
    final_errors = [(tid, p, g, gg) for tid, p, g, gg, ok in best_details if not ok]
    
    error_categories = defaultdict(list)
    for tid, pred, gold_norm, gold_raw in final_errors:
        q = next((q for q in questions if q["task_id"] == tid), None)
        target_text = target_texts.get(tid, "")
        error_categories[f"gold={gold_norm}_pred={pred}"].append((tid, target_text[:120], gold_raw))
    
    print(f"Total errors: {len(final_errors)}/{best[3]}")
    print("\nError patterns:")
    for pattern, items in sorted(error_categories.items(), key=lambda x: -len(x[1])):
        print(f"\n  {pattern}: {len(items)} errors")
        for tid, text, gold_raw in items[:3]:
            print(f"    [{tid}] '{text}'")
    
    # Per-source accuracy
    print("\n--- Per-source accuracy ---")
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for tid, pn, gn, gr, ok in best_details:
        q = next((q for q in questions if q["task_id"] == tid), None)
        src = os.path.basename(q["source"]) if q else "unknown"
        source_stats[src]["total"] += 1
        if ok:
            source_stats[src]["correct"] += 1
    for src in sorted(source_stats.keys()):
        v = source_stats[src]
        pct = v["correct"] / v["total"] * 100 if v["total"] else 0
        print(f"  {src:<50} {v['correct']:>4}/{v['total']:<4} ({pct:.1f}%)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
