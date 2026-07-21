#!/usr/bin/env python3
"""Step 1: Audit all available sentiment data — load, dedup, classify, and report."""
import json
import os
import re
from collections import defaultdict, Counter

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon/data/eval")

def safe_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None

def get_items(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("per_question", data.get("questions", data.get("items", [])))
    return []

# ── All source files that may contain sentiment ──────────────────────────────
source_files = [
    f"{BASE}/generated/sentiment_comprehensive_hard.json",
    f"{BASE}/sentiment_combined_25.json",
    f"{BASE}/sentiment_failure_analysis.json",
    f"{BASE}/tests/sst2_100.json",
    f"{BASE}/training-v1.json",
    f"{BASE}/training-v2.json",
    f"{BASE}/training-v3.json",
    f"{BASE}/validation-v1.json",
    f"{BASE}/validation-v2.json",
    f"{BASE}/validation-v3.json",
    f"{BASE}/primary/eval_60_medium_hard.json",
    f"{BASE}/primary/eval_hard_218.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172357.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172426.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172443.json",
]

# ── Load everything ──────────────────────────────────────────────────────────
raw_by_source = defaultdict(list)

for fp in source_files:
    data = safe_load_json(fp)
    if data is None:
        continue
    items = get_items(data)
    key = os.path.relpath(fp, BASE)
    for item in items:
        if not isinstance(item, dict):
            continue
        # Filter for sentiment category only
        cat = item.get("category", "")
        # Items without category field are still included (e.g. failure_analysis is all sentiment)
        if cat and cat != "sentiment":
            continue
        prompt = item.get("prompt", "") or item.get("text", "")
        answer = item.get("expected_answer", "") or item.get("expected", "")
        if not prompt or not answer:
            continue
        raw_by_source[key].append(item)

# ── Normalize answers ────────────────────────────────────────────────────────
def normalize_answer(ans):
    a = ans.lower().strip()
    if a.startswith("sentiment:"):
        a = a.replace("sentiment:", "").strip()
    a = a.strip().rstrip(".")
    # Strip parenthetical qualifiers
    a = re.sub(r'\(.*?\)', '', a).strip()
    
    if "mixed" in a:
        return "mixed"
    if "negative" in a and "positive" in a:
        return "mixed"
    if "negative" in a:
        return "negative"
    if "positive" in a:
        return "positive"
    if "neutral" in a:
        return "neutral"
    if a.startswith("pos"):
        return "positive"
    if a.startswith("neg"):
        return "negative"
    return a

# ── Build unified list ───────────────────────────────────────────────────────
all_items = []
seen_prompts = set()

for source_key, items in raw_by_source.items():
    for item in items:
        prompt = item.get("prompt", "") or item.get("text", "")
        answer_raw = item.get("expected_answer", "") or item.get("expected", "")
        
        prompt_key = prompt.strip().lower()
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        
        difficulty = item.get("difficulty", "unknown")
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "unknown"
        
        # Detect failure type from the failure analysis
        failure_type = item.get("failure_type", item.get("would_classify_as", ""))
        
        all_items.append({
            "prompt": prompt,
            "expected_answer_raw": answer_raw,
            "expected_answer": normalize_answer(answer_raw),
            "difficulty": difficulty,
            "source": source_key,
            "failure_type": failure_type,
        })

# ── Print audit report ───────────────────────────────────────────────────────
print("=" * 70)
print("SENTIMENT DATA AUDIT REPORT")
print("=" * 70)
print(f"\nTotal unique sentiment questions: {len(all_items)}")

# Per source
print(f"\n--- Per Source ---")
src_counts = Counter(it["source"] for it in all_items)
for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f"  {src}: {cnt}")

# Per difficulty
print(f"\n--- Per Difficulty ---")
diff_counts = Counter(it["difficulty"] for it in all_items)
for d in ["easy", "medium", "hard", "unknown"]:
    print(f"  {d}: {diff_counts.get(d, 0)}")

# Per expected answer
print(f"\n--- Per Expected Answer ---")
ans_counts = Counter(it["expected_answer"] for it in all_items)
for a in ["positive", "negative", "neutral", "mixed"]:
    print(f"  {a}: {ans_counts.get(a, 0)}")

# Cross-tab: difficulty × expected_answer
print(f"\n--- Difficulty × Expected Answer ---")
print(f"{'':12s} {'positive':>10s} {'negative':>10s} {'neutral':>10s} {'mixed':>10s} {'other':>10s}")
for d in ["easy", "medium", "hard", "unknown"]:
    counts = Counter(it["expected_answer"] for it in all_items if it["difficulty"] == d)
    print(f"{d:12s} {counts.get('positive', 0):>10d} {counts.get('negative', 0):>10d} "
          f"{counts.get('neutral', 0):>10d} {counts.get('mixed', 0):>10d} "
          f"{sum(v for k, v in counts.items() if k not in ('positive','negative','neutral','mixed')):>10d}")

# Failure type distribution (from failure analysis only)
print(f"\n--- Failure Type Distribution (failure_analysis) ---")
fa_items = [it for it in all_items if "failure_analysis" in it["source"]]
ft_counts = Counter(it.get("failure_type", "") for it in fa_items)
for ft, cnt in sorted(ft_counts.items(), key=lambda x: -x[1]):
    print(f"  {ft}: {cnt}")

# Save audit data
audit_path = f"{BASE}/sentiment_audit.json"
with open(audit_path, "w") as f:
    json.dump({"total": len(all_items),
               "per_source": dict(src_counts),
               "per_difficulty": dict(diff_counts),
               "per_answer": dict(ans_counts),
               "items": all_items}, f, indent=2)
print(f"\n✓ Audit data saved to {audit_path}")
print(f"\nDone.")
