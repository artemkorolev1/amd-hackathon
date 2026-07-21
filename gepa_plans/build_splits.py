#!/usr/bin/env python3
"""
Build train/val/hard-test splits from all available sentiment data.
Also generate additional hard questions to reach ~100 in the hard test set.

Outputs:
  data/eval/sentiment_train.json      ~400 questions
  data/eval/sentiment_val.json        ~100 questions
  data/eval/sentiment_hard_test.json  ~100 questions
"""
import json
import os
import random
import re
from collections import defaultdict, Counter

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon/data/eval")
SEED = 42
random.seed(SEED)

# ── Helper functions ─────────────────────────────────────────────────────────

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

def normalize_answer(ans):
    a = ans.lower().strip()
    if a.startswith("sentiment:"):
        a = a.replace("sentiment:", "").strip()
    a = re.sub(r'\(.*?\)', '', a).strip().rstrip(".")
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

def classify_difficulty_from_source(item):
    """Determine actual difficulty based on content patterns and failure type."""
    prompt = (item.get("prompt", "") or "").lower()
    text = prompt
    failure_type = item.get("failure_type", "")
    
    # Use failure type from failure analysis when available (authoritative)
    if failure_type in ("SUBTLE_LANGUAGE", "MIXED_SIGNALS", "SARCASM",
                        "KEYWORD_MISMATCH", "HEDGING", "FAINT_PRAISE"):
        return "hard"
    if failure_type == "NEUTRAL_NEGATIVE":
        return "medium"
    
    # ── HARD patterns ──
    # Sarcasm indicators
    sarcasm_phrases = [
        "oh brilliant", "efficiency at its finest", "admire your ability",
        "you're so brave", "about as well as expected",
        "for someone with your qualifications", "just the highlight of my day",
        "i really admire",
    ]
    for p in sarcasm_phrases:
        if p in text:
            return "hard"
    
    # Mixed signals: "great X but terrible Y"
    if "but" in text and any(w in text for w in ["great", "good", "excellent", "wonderful"]):
        if any(w in text for w in ["terrible", "awful", "horrible", "bad", "disappointing"]):
            return "hard"
    
    # Hedging
    hedging = ["not entirely terrible", "could be worse", "could do worse", 
               "one could do worse", "interesting enough"]
    for h in hedging:
        if h in text:
            return "hard"
    
    # Sarcastic brilliant
    if "brilliant" in text and any(w in text for w in ["cancel", "flight"]):
        return "hard"
    
    # Fraudulent/charged neutral
    if "fraudulent" in text:
        return "hard"
    
    # Backhanded
    if ("you're so" in text or "you are so" in text) and any(w in text for w in ["brave", "confident"]):
        return "hard"
    
    # Multi-entity mixed (earnings)
    if "earnings" in text and "missed" in text:
        return "hard"
    
    # ── MEDIUM patterns ──
    word_count = len(text.split())
    has_contrast = any(w in text for w in ["although", "however", "but", "while", "though", "yet"])
    
    if word_count > 30 and has_contrast:
        return "medium"
    
    if word_count > 50:
        return "medium"
    
    # Check original difficulty
    orig = item.get("difficulty", "easy")
    if orig in ("hard", "medium") and word_count > 20:
        return orig
    
    # ── EASY ──
    return "easy"

# ── STEP 1: Load ALL sentiment questions ────────────────────────────────────

source_files = [
    f"{BASE}/generated/sentiment_comprehensive_hard.json",
    f"{BASE}/sentiment_failure_analysis.json",
    f"{BASE}/sentiment_combined_25.json",
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

all_items = []
seen_prompts = set()

for fp in source_files:
    data = safe_load_json(fp)
    if data is None:
        continue
    items = get_items(data)
    key = os.path.relpath(fp, BASE)
    for item in items:
        if not isinstance(item, dict):
            continue
        cat = item.get("category", "")
        if cat and cat != "sentiment":
            continue
        prompt = item.get("prompt", "") or item.get("text", "")
        answer_raw = item.get("expected_answer", "") or item.get("expected", "")
        if not prompt or not answer_raw:
            continue
        
        prompt_key = prompt.strip().lower()
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        
        normalized = normalize_answer(answer_raw)
        if normalized not in ("positive", "negative", "neutral", "mixed"):
            # Try harder normalization
            if "positive" in normalized:
                normalized = "positive"
            elif "negative" in normalized:
                normalized = "negative"
            else:
                continue  # skip unclassifiable
        
        difficulty = classify_difficulty_from_source(item)
        
        all_items.append({
            "prompt": prompt,
            "expected_answer": normalized,
            "difficulty": difficulty,
            "source": key,
            "task_id": item.get("task_id", f"sent-{hash(prompt) & 0xffffffff:08x}"),
            "failure_type": item.get("failure_type", ""),
        })

print(f"Total unique sentiment items: {len(all_items)}")

# Count by difficulty
diff_counts = Counter(it["difficulty"] for it in all_items)
print(f"\nDifficulty breakdown:")
for d in ["easy", "medium", "hard"]:
    print(f"  {d}: {diff_counts.get(d, 0)}")

# Count by answer
ans_counts = Counter(it["expected_answer"] for it in all_items)
print(f"\nAnswer breakdown:")
for a in ["positive", "negative", "neutral", "mixed"]:
    print(f"  {a}: {ans_counts.get(a, 0)}")

# ── STEP 2: Create splits ───────────────────────────────────────────────────

# Separate by difficulty
easy = [it for it in all_items if it["difficulty"] == "easy"]
medium = [it for it in all_items if it["difficulty"] == "medium"]
hard = [it for it in all_items if it["difficulty"] == "hard"]

random.shuffle(easy)
random.shuffle(medium)
random.shuffle(hard)

print(f"\nAvailable: Easy={len(easy)} Medium={len(medium)} Hard={len(hard)}")

# Strategy:
# Hard test set: 100 hardest questions (all hard + some from medium if needed)
# Validation: ~100 questions (mix of easy/medium)
# Training: everyone else (~500-600)

# Reserve 100 for hard test
hard_test = hard[:min(100, len(hard))]
remaining_hard = hard[min(100, len(hard)):]

# If hard test < 100, supplement from remaining hard + medium-hard
if len(hard_test) < 100:
    needed = 100 - len(hard_test)
    # Take from medium items - the longest/most complex ones
    medium_sorted = sorted(remaining_hard + medium, key=lambda x: len(x["prompt"]), reverse=True)
    hard_test.extend(medium_sorted[:needed])

# Cap at 100
hard_test = hard_test[:100]
print(f"\nHard test set: {len(hard_test)}")

# Validation: ~100 from easy + medium (not in hard test)
hard_test_prompts = {it["prompt"].strip().lower() for it in hard_test}
remaining_easy = [it for it in easy if it["prompt"].strip().lower() not in hard_test_prompts]
remaining_medium = [it for it in medium if it["prompt"].strip().lower() not in hard_test_prompts]

val_target = 100
val_set = []
# 30 hard-ish medium + 70 easy
val_medium_pool = remaining_medium[:]
random.shuffle(val_medium_pool)
val_set.extend(val_medium_pool[:30])
val_easy_pool = [it for it in remaining_easy if it["prompt"].strip().lower() not in {x["prompt"].strip().lower() for x in val_set}]
random.shuffle(val_easy_pool)
val_set.extend(val_easy_pool[:70])
# Trim to 100
val_set = val_set[:100]
print(f"Validation set: {len(val_set)}")

# Training: everything else
val_prompts = {it["prompt"].strip().lower() for it in val_set}
train_set = [it for it in all_items 
             if it["prompt"].strip().lower() not in hard_test_prompts
             and it["prompt"].strip().lower() not in val_prompts]
# Shuffle
random.shuffle(train_set)
print(f"Training set: {len(train_set)}")

# ── STEP 3: Generate additional hard questions ──────────────────────────────

def generate_hard_variations(base_item):
    """Generate harder variations from an existing hard item."""
    prompt = base_item["prompt"]
    answer = base_item["expected_answer"]
    variations = []
    
    # Strategy 1: Add contradictory follow-up sentence
    if answer == "positive":
        contra = " However, the more I think about it, the more I realize it was deeply flawed."
        variations.append({
            "prompt": prompt + contra,
            "expected_answer": "mixed",
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (mixed): positive review with contradictory negative follow-up added."
        })
    elif answer == "negative":
        contra = " But there were some genuinely redeeming moments that I keep coming back to."
        variations.append({
            "prompt": prompt + contra,
            "expected_answer": "mixed",
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (mixed): negative review with contradictory positive follow-up added."
        })
    
    # Strategy 2: Add a sarcastic twist
    if len(prompt) > 40:
        variations.append({
            "prompt": prompt + " Yeah, right — because THAT makes perfect sense.",
            "expected_answer": "negative",
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (sarcastic amplification): original sentiment preserved with added sarcastic remark."
        })
    
    # Strategy 3: Flip the context (add negation/contrast at start)
    if len(prompt) < 200:
        prefix = "I've been told this is great, but honestly... "
        if answer == "negative":
            prefix = "Everyone says this is terrible, but personally... "
        variations.append({
            "prompt": prefix + prompt,
            "expected_answer": answer,
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (contextual flip): original sentiment buried under contrasting setup."
        })
    
    # Strategy 4: Add a backhanded compliment structure
    if answer == "negative":
        variations.append({
            "prompt": f"The production values are undeniably impressive. That said, {prompt[0].lower()}{prompt[1:]}",
            "expected_answer": "negative",
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (backhanded): opens with surface-level praise then undercuts it."
        })
    elif answer == "positive":
        variations.append({
            "prompt": f"I had low expectations, but I have to admit: {prompt[0].lower()}{prompt[1:]}",
            "expected_answer": "positive",
            "difficulty": "hard",
            "source": "augmented-hard",
            "reasoning": "Hard (reluctant admission): framed as surprising positive against low expectations."
        })
    return variations

# Count unique prompts to see how many variations we need
existing_test_prompts = {it["prompt"] for it in hard_test}
print(f"\nExisting hard test has {len(hard_test)} items")

# Generate variations from hard items (to fill hard test)
augmented_hard = []
for item in hard_test[:40]:  # Use 40 base items for generation
    for variant in generate_hard_variations(item):
        if variant["prompt"] not in existing_test_prompts:
            augmented_hard.append(variant)
            existing_test_prompts.add(variant["prompt"])

print(f"Generated {len(augmented_hard)} augmented hard questions")

# Trim hard test to exactly 100
# First, make sure we include the original 92 from comprehensive_hard.json where possible
orig_92_path = f"{BASE}/generated/sentiment_comprehensive_hard.json"
orig_92 = []
for item in hard_test:
    if item["source"] == "generated/sentiment_comprehensive_hard.json":
        orig_92.append(item)

# Build final hard test: keep all true-hard items, use augmented items, fill with medium
hard_true = [it for it in hard_test if it["difficulty"] == "hard"]
hard_medium = [it for it in hard_test if it["difficulty"] != "hard"]
print(f"\nTrue hard items: {len(hard_true)}, Medium fillers: {len(hard_medium)}")

# Replace medium fillers with augmented hard items where possible
final_hard_test = list(hard_true)
# Add augmented items that aren't already in
for aug in augmented_hard:
    if aug["prompt"] not in {it["prompt"] for it in final_hard_test}:
        final_hard_test.append(aug)
# Fill remaining slots from medium pool
remaining_slots = 100 - len(final_hard_test)
if remaining_slots > 0:
    final_hard_test.extend([it for it in hard_medium if it["prompt"] not in {x["prompt"] for x in final_hard_test}][:remaining_slots])

final_hard_test = final_hard_test[:100]
random.shuffle(final_hard_test)

print(f"\nFinal hard test: {len(final_hard_test)}")
print(f"  Original comprehensive items: {sum(1 for it in final_hard_test if it['source'] == 'generated/sentiment_comprehensive_hard.json')}")
print(f"  Failure analysis items: {sum(1 for it in final_hard_test if 'failure' in it['source'])}")
print(f"  Augmented items: {sum(1 for it in final_hard_test if it['source'] == 'augmented-hard')}")

# ── STEP 4: Write output files ──────────────────────────────────────────────

def write_split(items, filepath, name):
    """Write split to JSON, ensuring consistent format."""
    output = []
    for item in items:
        output.append({
            "category": "sentiment",
            "task_id": item.get("task_id", f"sent-{hash(item['prompt']) & 0xffffffff:08x}"),
            "prompt": item["prompt"],
            "expected_answer": item["expected_answer"],
            "difficulty": item["difficulty"],
            "source": item.get("source", "unknown"),
            "failure_type": item.get("failure_type", ""),
            "reasoning": item.get("reasoning", ""),
        })
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    diff_c = Counter(it["difficulty"] for it in output)
    ans_c = Counter(it["expected_answer"] for it in output)
    print(f"\n{name} ({len(output)} questions):")
    print(f"  Difficulty: easy={diff_c.get('easy',0)} medium={diff_c.get('medium',0)} hard={diff_c.get('hard',0)}")
    print(f"  Answers: pos={ans_c.get('positive',0)} neg={ans_c.get('negative',0)} neu={ans_c.get('neutral',0)} mix={ans_c.get('mixed',0)}")
    print(f"  Written to: {filepath}")

os.makedirs(BASE, exist_ok=True)

write_split(train_set, f"{BASE}/sentiment_train.json", "TRAINING SET")
write_split(val_set, f"{BASE}/sentiment_val.json", "VALIDATION SET")
write_split(final_hard_test, f"{BASE}/sentiment_hard_test.json", "HARD TEST SET")

print("\n✓ All splits created successfully.")
