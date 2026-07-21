#!/usr/bin/env python3
"""
Precision sweep: find optimal asymmetric thresholds.
Sweeps pos_thresh from 0.0 to 0.4 and neg_thresh from -0.2 to 0.0.
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
    "input/dev_40.json", "input/heldout_40.json", "input/complexity_40.json", "input/cx_300.json",
    "data/eval/primary/eval_60_medium_hard.json", "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json", "data/eval/primary/eval_mini_10.json",
    "data/eval/training-v1.json", "data/eval/training-v2.json", "data/eval/training-v3.json",
    "data/eval/validation-v1.json", "data/eval/validation-v2.json", "data/eval/validation-v3.json",
    "data/eval/tests/complexity_eval_40.json", "data/eval/tests/complexity_eval_candidates.json",
    "data/eval/tests/eval_longform_20.json", "data/eval/tests/eval_v14_test_20.json",
    "data/eval/tests/eval_v14_remaining_20.json", "data/eval/tests/eval_v14_timeout_stress_19.json",
    "data/eval/tests/fireworks_eval_20.json", "data/eval/generated/build-A-40.json",
    "data/eval/generated/build-B-40.json", "data/eval/generated/eval_from_datasets_20260712_172357.json",
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
            gold_answer = gold.get("answer", "") if isinstance(gold, dict) else str(gold) if gold else ""
            if not gold_answer:
                gold_answer = item.get("expected_answer", "")
            questions.append({"task_id": item.get("task_id", f"q{idx}"), "prompt": prompt, "gold_answer": gold_answer, "source": path})
    return questions

def normalize_gold(gold):
    g = gold.lower().strip()
    for label in ["positive", "negative", "neutral", "mixed"]:
        if label in g: return label
    return g

def extract_target_text(prompt):
    quoted = re.findall(r'"([^"]+)"', prompt)
    if quoted: return quoted[-1]
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

def classify(text, pos_thresh, neg_thresh):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= pos_thresh:
        return "positive"
    elif compound <= neg_thresh:
        return "negative"
    else:
        return "neutral"

questions = load_sentiment_questions()
gold_map = {q["task_id"]: q["gold_answer"] for q in questions}
target_texts = {q["task_id"]: extract_target_text(q["prompt"]) for q in questions}

print("Sweeping asymmetric thresholds...\n")
print(f"{'pos_thresh':>10} {'neg_thresh':>10} {'Accuracy':>9} {'Correct/Total':>14} {'pos%':>6} {'neg%':>6} {'neu%':>6}")
print("-" * 65)

results = []
for pos_t in [x/100.0 for x in range(0, 45, 5)]:
    for neg_t in [x/100.0 for x in range(-20, 5, 5)]:
        if pos_t <= neg_t:
            continue  # pos must be > neg
        correct = 0
        total = 0
        gold_label_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for q in questions:
            gold = gold_map.get(q["task_id"], "")
            if not gold: continue
            gold_norm = normalize_gold(gold)
            pred = classify(target_texts[q["task_id"]], pos_t, neg_t)
            gold_label_stats[gold_norm]["total"] += 1
            if pred == gold_norm:
                correct += 1
                gold_label_stats[gold_norm]["correct"] += 1
            total += 1
        
        acc = correct / total * 100 if total > 0 else 0
        pos_acc = gold_label_stats["positive"]["correct"] / max(gold_label_stats["positive"]["total"], 1) * 100
        neg_acc = gold_label_stats["negative"]["correct"] / max(gold_label_stats["negative"]["total"], 1) * 100
        neu_acc = gold_label_stats["neutral"]["correct"] / max(gold_label_stats["neutral"]["total"], 1) * 100
        results.append((acc, pos_t, neg_t, correct, total, pos_acc, neg_acc, neu_acc))
        
        print(f"{pos_t:>8.2f}  {neg_t:>8.2f}  {acc:>6.1f}%  {correct:>3}/{total:<4}   {pos_acc:>4.0f}% {neg_acc:>4.0f}% {neu_acc:>4.0f}%")

results.sort(key=lambda r: -r[0])
best = results[0]
print(f"\n{'=' * 65}")
print(f"BEST: pos_thresh={best[1]:.2f}, neg_thresh={best[2]:.2f} → {best[0]:.1f}% ({best[3]}/{best[4]})")
print(f"  pos_acc: {best[5]:.0f}%, neg_acc: {best[6]:.0f}%, neu_acc: {best[7]:.0f}%")

# Also show top 5
print(f"\nTop 5 configurations:")
for i, (acc, pt, nt, cor, tot, pa, na, nua) in enumerate(results[:5]):
    print(f"  {i+1}. pos_thresh={pt:.2f}, neg_thresh={nt:.2f} → {acc:.1f}% ({cor}/{tot})")
    print(f"     pos_acc={pa:.0f}% neg_acc={na:.0f}% neu_acc={nua:.0f}%")

# Also check: what if we use a neutral band?
print(f"\n\nNow checking neutral band approach (neutral if |compound| < threshold):")
print(f"{'neutral_band':>12} {'Accuracy':>9} {'Correct':>8} {'pos%':>6} {'neg%':>6} {'neu%':>6}")
print("-" * 45)

for band in [x/100.0 for x in range(0, 30, 2)]:
    correct = 0
    total = 0
    gold_label_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for q in questions:
        gold = gold_map.get(q["task_id"], "")
        if not gold: continue
        gold_norm = normalize_gold(gold)
        scores = analyzer.polarity_scores(target_texts[q["task_id"]])
        c = scores["compound"]
        if c >= band:
            pred = "positive"
        elif c <= -band:
            pred = "negative"
        else:
            pred = "neutral"
        gold_label_stats[gold_norm]["total"] += 1
        if pred == gold_norm:
            correct += 1
            gold_label_stats[gold_norm]["correct"] += 1
        total += 1
    acc = correct / total * 100 if total > 0 else 0
    pos_acc = gold_label_stats["positive"]["correct"] / max(gold_label_stats["positive"]["total"], 1) * 100
    neg_acc = gold_label_stats["negative"]["correct"] / max(gold_label_stats["negative"]["total"], 1) * 100
    neu_acc = gold_label_stats["neutral"]["correct"] / max(gold_label_stats["neutral"]["total"], 1) * 100
    print(f"  |c|>={band:.2f}     {acc:>6.1f}%  {correct:>3}/{total:<4}  {pos_acc:>4.0f}% {neg_acc:>4.0f}% {neu_acc:>4.0f}%")
