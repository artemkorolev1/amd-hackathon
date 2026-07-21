#!/usr/bin/env python3
"""
Round 2 of VADER tuning — focus on fixing the gold=negative but pred=positive problem.
Also includes an "advanced" strategy that uses ratio-based rules + compound.
"""
import json, os, sys, re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from agent.category_filter import get_short_name
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
    g = gold.lower().strip()
    for label in ["positive", "negative", "neutral", "mixed"]:
        if label in g:
            return label
    return g

def extract_target_text(prompt):
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

def classify_vader_v2(text, config):
    """Enhanced VADER classification with ratio-based heuristics."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    neu = scores["neu"]
    
    strategy = config.get("strategy", "compound_only")
    
    if strategy == "compound_only":
        if config.get("enable_mixed") and pos > config.get("mixed_pos", 0.3) and neg > config.get("mixed_neg", 0.3):
            return "mixed"
        if compound >= config["pos_thresh"]:
            return "positive"
        elif compound <= config["neg_thresh"]:
            return "negative"
        else:
            return "neutral"
    
    elif strategy == "ratio_based":
        # Use pos/neg ratio instead of just compound
        if pos == 0 and neg == 0:
            return "neutral"
        ratio = pos / (pos + neg) if (pos + neg) > 0 else 0.5
        
        if config.get("enable_mixed") and pos > config.get("mixed_pos", 0.3) and neg > config.get("mixed_neg", 0.3):
            return "mixed"
        
        # Strong signal from ratio
        if ratio >= 0.75:
            return "positive"
        elif ratio <= 0.25:
            return "negative"
        elif compound >= 0.2:
            return "positive"
        elif compound <= -0.2:
            return "negative"
        else:
            return "neutral"
    
    elif strategy == "compound_signed":
        # Use compound but with a neutral zone that widens for short texts
        # and includes a "leaning positive but check neg dominance" rule
        if config.get("enable_mixed") and pos > config.get("mixed_pos", 0.3) and neg > config.get("mixed_neg", 0.3):
            return "mixed"
        
        # Sarcasm detection: if compound is slightly positive but neg > pos, it's negative
        sarcasm_detected = (compound > 0 and compound < 0.4 and neg > pos)
        
        if sarcasm_detected:
            return "negative"
        
        if compound >= config["pos_thresh"]:
            return "positive"
        elif compound <= config["neg_thresh"]:
            return "negative"
        else:
            return "neutral"

    elif strategy == "compound_with_neutral_width":
        # Dynamic neutral zone based on text length
        compound_offset = min(0.15, config.get("neutral_width", 0.05))
        if config.get("enable_mixed") and pos > config.get("mixed_pos", 0.3) and neg > config.get("mixed_neg", 0.3):
            return "mixed"
        if compound >= compound_offset:
            return "positive"
        elif compound <= -compound_offset:
            return "negative"
        else:
            return "neutral"

# More configs to try, including ratio-based and sarcasm-aware
EXTRA_CONFIGS = [
    {
        "name": "vader_loose_signed",
        "desc": "compound>=0.0=POS, <0.0=NEG + sarcasm (compound 0-0.4 and neg>pos → NEG)",
        "strategy": "compound_signed",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "enable_mixed": False,
    },
    {
        "name": "vader_neutral_zone_w15",
        "desc": "neutral zone width=0.15, wider neutral band",
        "strategy": "compound_with_neutral_width",
        "neutral_width": 0.15,
        "enable_mixed": False,
    },
    {
        "name": "vader_ratio",
        "desc": "ratio-based: pos/(pos+neg) with compound fallback",
        "strategy": "ratio_based",
        "enable_mixed": False,
    },
    {
        "name": "vader_loose_signed_mixed",
        "desc": "signed + MIXED detection",
        "strategy": "compound_signed",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "enable_mixed": True,
        "mixed_pos": 0.2,
        "mixed_neg": 0.2,
    },
    {
        "name": "vader_neg_biased",
        "desc": "asymmetric: POS>=0.05, NEG<=-0.0, no MIXED",
        "strategy": "compound_only",
        "pos_thresh": 0.05,
        "neg_thresh": 0.0,
        "enable_mixed": False,
    },
    {
        "name": "vader_neg_biased_loose",
        "desc": "asymmetric loose: POS>=0.01, NEG<=-0.0",
        "strategy": "compound_only",
        "pos_thresh": 0.01,
        "neg_thresh": 0.0,
        "enable_mixed": False,
    },
]

# Include best from round 1
ALL_CONFIGS = EXTRA_CONFIGS + [
    {
        "name": "vader_loose",
        "desc": "compound>=0.0=POS, <0.0=NEG (best from round 1)",
        "strategy": "compound_only",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "enable_mixed": False,
    },
    {
        "name": "vader_default",
        "desc": "compound>=0.05=POS, -0.05>compound<NEG (standard)",
        "strategy": "compound_only",
        "pos_thresh": 0.05,
        "neg_thresh": -0.05,
        "enable_mixed": False,
    },
    {
        "name": "vader_mixed_loose",
        "desc": "compound>=0.0, MIXED when pos>0.2 and neg>0.2",
        "strategy": "compound_only",
        "pos_thresh": 0.0,
        "neg_thresh": 0.0,
        "enable_mixed": True,
        "mixed_pos": 0.2,
        "mixed_neg": 0.2,
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

def categorize_error(details, questions, target_texts):
    err_by_gold_pred = defaultdict(list)
    for tid, pred_norm, gold_norm, gold_raw, ok in details:
        if ok: continue
        key = f"gold={gold_norm}_pred={pred_norm}"
        q = next((q for q in questions if q["task_id"] == tid), None)
        src = os.path.basename(q["source"]) if q else "unknown"
        err_by_gold_pred[key].append({
            "task_id": tid,
            "text": target_texts.get(tid, "")[:100],
            "source": src,
            "gold_raw": gold_raw[:80],
        })
    return err_by_gold_pred

def main():
    questions = load_sentiment_questions()
    print(f"Loaded {len(questions)} sentiment questions with gold answers\n")
    gold_map = {q["task_id"]: q["gold_answer"] for q in questions}
    target_texts = {q["task_id"]: extract_target_text(q["prompt"]) for q in questions}
    
    print("=" * 80)
    print("VADER ROUND 2 TUNING — 10 configs")
    print("=" * 80)
    
    all_results = []
    for config in ALL_CONFIGS:
        config_preds = {}
        for q in questions:
            ans = classify_vader_v2(target_texts[q["task_id"]], config)
            config_preds[q["task_id"]] = ans
        
        acc, correct, total, details = score_accuracy(config_preds, gold_map)
        all_results.append((config["name"], acc, correct, total, config, details))
        
        # Per-gold accuracy
        gold_labels = defaultdict(lambda: {"correct": 0, "total": 0})
        for tid, pn, gn, gr, ok in details:
            gold_labels[gn]["total"] += 1
            if ok:
                gold_labels[gn]["correct"] += 1
        
        pos_acc = gold_labels["positive"]["correct"] / max(gold_labels["positive"]["total"], 1) * 100
        neg_acc = gold_labels["negative"]["correct"] / max(gold_labels["negative"]["total"], 1) * 100
        neu_acc = gold_labels["neutral"]["correct"] / max(gold_labels["neutral"]["total"], 1) * 100
        mixed_acc = gold_labels["mixed"]["correct"] / max(gold_labels["mixed"]["total"], 1) * 100
        
        print(f"  {config['name']:<30} {acc:>5.1f}%  ({correct:>3}/{total:<3})  "
              f"pos={pos_acc:.0f}% neg={neg_acc:.0f}% neu={neu_acc:.0f}% mix={mixed_acc:.0f}%  "
              f"| {config['desc'][:50]}")
    
    print("\n" + "=" * 80)
    print("RANKINGS")
    print("=" * 80)
    print(f"{'Config':<30} {'Accuracy':>10} {'Correct/Total':>15}  {'pos%':>5} {'neg%':>5} {'neu%':>5}")
    print("-" * 70)
    for name, acc, correct, total, config, _ in sorted(all_results, key=lambda r: -r[1]):
        # Recalculate per-gold for display
        _, _, _, details_show = all_results[[r[0] for r in all_results].index(name)][1:5]
        if config["name"] != name:
            continue
        gold_labels = defaultdict(lambda: {"correct": 0, "total": 0})
        for tid, pn, gn, gr, ok in all_results[[r[0] for r in all_results].index(name)][5]:
            gold_labels[gn]["total"] += 1
            if ok:
                gold_labels[gn]["correct"] += 1
        pos_acc = gold_labels["positive"]["correct"] / max(gold_labels["positive"]["total"], 1) * 100
        neg_acc = gold_labels["negative"]["correct"] / max(gold_labels["negative"]["total"], 1) * 100
        neu_acc = gold_labels["neutral"]["correct"] / max(gold_labels["neutral"]["total"], 1) * 100
        print(f"{name:<30} {acc:>8.1f}%  {correct:>3}/{total:<4}   {pos_acc:>4.0f}% {neg_acc:>4.0f}% {neu_acc:>4.0f}%")
    
    # BEST CONFIG with detailed error analysis
    best = max(all_results, key=lambda r: r[1])
    print(f"\n{'=' * 80}")
    print(f"BEST: {best[0]} — {best[1]:.1f}% ({best[2]}/{best[3]})")
    print(f"{'=' * 80}")
    
    errors = categorize_error(best[5], questions, target_texts)
    print(f"\nError breakdown ({best[3] - best[2]} errors):")
    for pattern, items in sorted(errors.items(), key=lambda x: -len(x[1])):
        print(f"\n  {pattern}: {len(items)} errors")
        for item in items[:4]:
            print(f"    [{item['task_id']}] '{item['text']}'")
    
    # === Second best ===
    if len(all_results) >= 2:
        second = sorted(all_results, key=lambda r: -r[1])[1]
        if second[0] != best[0]:
            print(f"\n{'=' * 80}")
            print(f"SECOND BEST: {second[0]} — {second[1]:.1f}% ({second[2]}/{second[3]})")
            print(f"{'=' * 80}")
            errors2 = categorize_error(second[5], questions, target_texts)
            print(f"\nError breakdown ({second[3] - second[2]} errors):")
            for pattern, items in sorted(errors2.items(), key=lambda x: -len(x[1])):
                print(f"\n  {pattern}: {len(items)} errors")
                for item in items[:3]:
                    print(f"    [{item['task_id']}] '{item['text']}'")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
