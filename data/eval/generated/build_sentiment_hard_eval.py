#!/usr/bin/env python3
"""
Build a comprehensive hard sentiment evaluation set from all available AMD hackathon data sources.
Output: data/eval/generated/sentiment_comprehensive_hard.json
"""

import json
import os
import random
import sys
from collections import defaultdict

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon/data/eval")

def safe_load_json(path):
    """Load JSON from a file, return None if it fails."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None

def get_items(data):
    """Return a list of items regardless of whether data is a list, or a dict with 'questions'."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("questions", data.get("items", []))
    return []

# ─── All source files with sentiment data ─────────────────────────────────────

source_files = [
    f"{BASE}/sentiment_combined_25.json",
    f"{BASE}/training-v1.json",
    f"{BASE}/training-v2.json",
    f"{BASE}/training-v3.json",
    f"{BASE}/validation-v1.json",
    f"{BASE}/validation-v2.json",
    f"{BASE}/validation-v3.json",
    f"{BASE}/tests/sst2_100.json",
    f"{BASE}/tests/complexity_eval_40.json",
    f"{BASE}/tests/fireworks_eval_20.json",
    f"{BASE}/primary/eval_60_medium_hard.json",
    f"{BASE}/primary/eval_hard_218.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172357.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172426.json",
    f"{BASE}/generated/eval_from_datasets_20260712_172443.json",
]

all_sentiment_items = []

for fp in source_files:
    data = safe_load_json(fp)
    if data is None:
        continue
    items = get_items(data)
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("category") != "sentiment":
            continue
        prompt = item.get("prompt", "")
        answer = item.get("expected_answer", "")
        if not prompt or not answer:
            continue
        all_sentiment_items.append({
            "prompt": prompt,
            "expected_answer": answer,
            "difficulty": item.get("difficulty", "unknown"),
            "source": item.get("source", "unknown"),
            "reasoning": item.get("reasoning", ""),
        })

print(f"Total raw sentiment items loaded: {len(all_sentiment_items)}")

# ─── Deduplicate by full prompt text ─────────────────────────────────────────

seen_prompts = set()
deduped = []
for item in all_sentiment_items:
    prompt_key = item["prompt"].strip().lower()
    if prompt_key not in seen_prompts:
        seen_prompts.add(prompt_key)
        deduped.append(item)

print(f"After deduplication: {len(deduped)} unique items")

# ─── Normalize expected answers ──────────────────────────────────────────────

def normalize_answer(ans):
    """Normalize expected_answer to one of: positive, negative, neutral, mixed"""
    a = ans.lower().strip()
    # Remove prefix
    if a.startswith("sentiment:"):
        a = a.replace("sentiment:", "").strip()
    # Remove parenthetical qualifiers
    for phrase in ["(sarcastic)", "(hedging)", "(dismissive)", "(backhanded)",
                   "(indirect)", "(or neutral)", "(sarcastic/negative)",
                   "(predominantly negative with a faint positive note)"]:
        a = a.replace(phrase, "").strip()
    a = a.strip().rstrip(".")
    
    # Classification logic
    if "mixed" in a:
        return "mixed"
    if "negative" in a and "positive" in a:
        return "mixed"
    if "negative" in a:
        return "negative"
    if "positive" in a:
        if "not positive" in a:
            return "neutral"
        return "positive"
    if "neutral" in a:
        return "neutral"
    
    # Catch remaining
    if a.startswith("pos"):
        return "positive"
    if a.startswith("neg"):
        return "negative"
    
    return a

for item in deduped:
    item["expected_answer_normalized"] = normalize_answer(item["expected_answer"])

# ─── Re-classify items by actual difficulty ──────────────────────────────────

def classify_actual_difficulty(item):
    """
    Re-classify difficulty based on actual content.
    Returns one of: easy, medium, hard
    """
    prompt = item["prompt"]
    text = prompt.lower()
    ans = str(item.get("expected_answer", "")).lower()
    orig_diff = item.get("difficulty", "unknown")
    
    # ── HARD patterns ──
    
    # Sarcasm: positive words used negatively
    sarcasm_indicators = [
        "oh brilliant", "oh wow", "efficiency at its finest",
        "admire your ability", "i really admire",
        "you're so brave", "you're so confident",
        "about as well as expected",
        "for someone with your qualifications",
        "truly 'improved'", "so 'intuitive'",
        "just the highlight of my day"
    ]
    has_sarcasm = any(ind in text for ind in sarcasm_indicators)
    
    # "brilliant" in negative context
    has_brilliant_negative = "brilliant" in text and any(
        w in text for w in ["canceled", "cancelled", "flight"])
    
    # Hedging/faint praise
    hedging_indicators = [
        "not entirely terrible", "could be worse", "could do worse", "one could do worse",
        "interesting enough", "i suppose it's not",
        "enough to keep me from checking my phone"
    ]
    has_hedging = any(ind in text for ind in hedging_indicators)
    
    # Mixed signals: "Great X but terrible Y"
    great_terrible = "great" in text and "terrible" in text
    has_mixed_structure = great_terrible
    
    # Faint praise / hedging
    faint_praise_indicators = [
        "works. mostly", "mostly works",
        "i wouldn't recommend",
        "does the job i guess"
    ]
    has_faint_praise = any(ind in text for ind in faint_praise_indicators)
    
    # Multi-sentence contradiction
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if len(s.strip()) > 10]
    has_contradictory = False
    if len(sentences) >= 2:
        pos_words = ["great", "good", "excellent", "amazing", "wonderful", "beautiful",
                     "love", "perfect", "best", "fantastic", "impressive", "charming",
                     "mesmerizing", "riveted"]
        neg_words = ["terrible", "awful", "horrible", "bad", "worst", "disappointing",
                    "poor", "broken", "failure", "sucks", "hate", "boring", "slow",
                    "bleak", "desperate"]
        pos_scores = [sum(1 for w in pos_words if w in s.lower()) for s in sentences]
        neg_scores = [sum(1 for w in neg_words if w in s.lower()) for s in sentences]
        # Check if first half positive, second half negative or vice versa
        if len(sentences) >= 2:
            first_half = sum(pos_scores[:len(pos_scores)//2])
            second_half_pos = sum(pos_scores[len(pos_scores)//2:])
            first_neg = sum(neg_scores[:len(neg_scores)//2])
            second_neg = sum(neg_scores[len(neg_scores)//2:])
            if (first_half > 0 and second_neg > 0) or (first_neg > 0 and second_half_pos > 0):
                if first_half > 0 and second_neg > 0:
                    has_contradictory = True
    
    # Negated positive: "Amazing!" + negative follow-up
    has_negated = "amazing" in text and any(
        w in text for w in ["but", "however", "unfortunately", "returning", "disappointed"])
    
    # Multi-entity mixed (earnings report style)
    has_multi_entity_mixed = ("earnings" in text and "missed" in text and 
                             any(w in text for w in ["cloud", "raised guidance", "increase in"]) or
                             "earnings" in text and "declining" in text and "increase" in text)
    
    # Neutral with charged keywords
    has_neutral_charged = "fraudulent" in text and "neutral" in ans
    
    # Explicit sarcasm/backhanded labels in answer
    answer_is_hard = any(w in ans for w in ["sarcastic", "backhanded", "dismissive", "condescending"])
    
    if has_sarcasm or has_brilliant_negative or has_hedging or has_neutral_charged or answer_is_hard:
        return "hard"
    if has_mixed_structure or has_contradictory or has_faint_praise or has_negated:
        return "hard"
    if has_multi_entity_mixed:
        return "hard"
    
    # ── MEDIUM patterns ──
    medium_indicators = [
        "mixed", "balanc", "both positive", "both negative",
        "neither positive nor negative", "formal", "factual",
        "without emotional"
    ]
    has_medium_markers = any(ind in text for ind in medium_indicators)
    
    # Assess text complexity
    word_count = len(text.split())
    has_contrast = any(w in text for w in ["although", "however", "but", "while", "though"])
    has_conditional = any(w in text for w in ["if", "unless", "provided that"])
    
    is_complex = word_count > 25 and has_contrast
    is_objective = has_medium_markers
    
    if is_complex or is_objective:
        return "medium"
    
    # ── Fallback to original label ──
    if orig_diff in ("hard", "medium", "easy"):
        return orig_diff
    
    return "easy"


# Apply classification
for item in deduped:
    item["difficulty"] = classify_actual_difficulty(item)

# ─── Separate by difficulty ──────────────────────────────────────────────────

easy_items = [it for it in deduped if it["difficulty"] == "easy"]
medium_items = [it for it in deduped if it["difficulty"] == "medium"]
hard_items = [it for it in deduped if it["difficulty"] == "hard"]

print(f"\nBy difficulty after reclassification:")
print(f"  Easy:   {len(easy_items)}")
print(f"  Medium: {len(medium_items)}")
print(f"  Hard:   {len(hard_items)}")
print(f"  Total:  {len(easy_items) + len(medium_items) + len(hard_items)}")

# ─── Build curated set with target distribution: ~40% hard, ~30% med, ~30% easy ──

target = 92
target_hard = int(target * 0.40)  # 37
target_medium = int(target * 0.30)  # 28
target_easy = int(target * 0.30)  # 27

random.seed(42)

# Selection with variety
def select_n(items, n):
    shuffled = list(items)
    random.shuffle(shuffled)
    return shuffled[:n]

sel_hard = select_n(hard_items, min(target_hard, len(hard_items)))
sel_medium = select_n(medium_items, min(target_medium, len(medium_items)))
sel_easy = select_n(easy_items, min(target_easy, len(easy_items)))

# Ensure SPECIFIC hard variants are covered
required_variant_checks = [
    ("brilliant_sarcasm", lambda t: "brilliant" in t.lower() and "canceled" in t.lower()),
    ("not_entirely_terrible", lambda t: "not entirely terrible" in t.lower()),
    ("could_be_worse", lambda t: "could be worse" in t.lower() or "could do worse" in t.lower()),
    ("great_terrible_mixed", lambda t: "great" in t.lower() and "terrible" in t.lower()),
    ("dismissive_qualified", lambda t: "for someone with your qualifications" in t.lower()),
    ("admire_sarcasm", lambda t: "admire your ability" in t.lower()),
    ("deadpan_expected", lambda t: "about as well as expected" in t.lower()),
    ("neutral_charged", lambda t: "fraudulent" in t.lower()),
    ("youre_so_brave", lambda t: "you're so brave" in t.lower()),
    ("efficiency_finest", lambda t: "efficiency at its finest" in t.lower()),
]

added_variant_names = []
for vname, vcheck in required_variant_checks:
    found = any(vcheck(h["prompt"]) for h in sel_hard)
    if not found:
        for h in hard_items:
            if vcheck(h["prompt"]) and h not in sel_hard:
                sel_hard.append(h)
                added_variant_names.append(vname)
                break

if added_variant_names:
    print(f"  + Added variants: {added_variant_names}")

# ─── Final trimming ──────────────────────────────────────────────────────────

actual_total = len(sel_hard) + len(sel_medium) + len(sel_easy)

# Identify items we MUST keep (required hard variants)
protected_prompts = set()
for vname, vcheck in required_variant_checks:
    for h in sel_hard:
        if vcheck(h["prompt"]):
            protected_prompts.add(h["prompt"].strip().lower())

while actual_total > target:
    # Remove from largest pool, but don't remove protected items
    pools = [("hard", sel_hard), ("medium", sel_medium), ("easy", sel_easy)]
    # Sort by size descending, prefer non-hard for trimming
    pools.sort(key=lambda x: (0 if x[0] == "hard" else 1, len(x[1])), reverse=True)
    
    removed = False
    for pname, pool in pools:
        if pool:
            # Try to find a non-protected item to remove
            for idx in range(len(pool) - 1, -1, -1):
                if pname != "hard" or pool[idx]["prompt"].strip().lower() not in protected_prompts:
                    pool.pop(idx)
                    actual_total -= 1
                    removed = True
                    break
        if removed:
            break
    
    if not removed:
        # Fallback: remove from the largest pool regardless of protection
        for pname, pool in pools:
            if pool:
                pool.pop()
                actual_total -= 1
                break

# ─── Build output with reasoning ─────────────────────────────────────────────

def build_reasoning(item):
    if item.get("reasoning") and len(item["reasoning"]) > 15:
        return item["reasoning"]
    
    prompt = item["prompt"]
    text = prompt.lower()
    diff = item["difficulty"]
    
    if "brilliant" in text and ("canceled" in text or "cancelled" in text):
        return "Sarcasm: 'brilliant' used in clearly negative context (canceled flight). LLMs relying on keyword sentiment often label this positive."
    if "not entirely terrible" in text:
        return "Hedging/understatement: 'not entirely terrible' is faint praise — neither truly positive nor overtly negative. LLMs often mislabel as negative due to 'terrible'."
    if "could be worse" in text:
        return "Hedging: 'could be worse' expresses reluctant acceptance, not genuine positivity."
    if "great" in text and "terrible" in text:
        return "Mixed signals: combines positive ('great') and negative ('terrible') in the same review. Requires 4-class output (positive/negative/neutral/mixed)."
    if "admire your ability" in text and "wrong" in text:
        return "Passive-aggressive sarcasm: 'admire your ability' is surface-positive but context reveals mockery. LLMs often label this positive."
    if "for someone with your qualifications" in text:
        return "Dismissive backhanded compliment: 'for someone with your qualifications' implies the person is underqualified. 'Remarkably well' is condescending."
    if "about as well as expected" in text:
        return "Deadpan sarcasm: uses understatement to signal disappointment. Without pragmatic context, LLMs often call this neutral."
    if "fraudulent" in text:
        return "Neutral factual report despite charged keyword ('fraudulent'). LLMs often mislabel as negative due to 'fraudulent'."
    if "earnings" in text and "missed" in text:
        return "Mixed sentiment with balanced positive and negative signals. First half negative (missed estimates), second half positive (cloud growth). Hard because it doesn't fit a single category."
    if "you're so brave" in text:
        return "Backhanded compliment: 'brave' coupled with 'wear that outfit in public' signals mockery. LLMs relying on surface-level positivity miss the sarcastic tone."
    if "efficiency at its finest" in text:
        return "Sarcasm: 'efficiency at its finest' sounds positive but the context (excessive follow-up emails) frames it as bureaucratic annoyance."
    if "amazing" in text and any(w in text for w in ["but", "returning", "disappointed"]):
        return "Negated positive: initial exclamation is undercut by negative follow-up. Requires reading beyond the first sentence."
    
    if diff == "hard":
        return "Hard because it requires pragmatic/sarcasm detection beyond keyword matching, or involves mixed/hedged sentiment."
    elif diff == "medium":
        return "Medium because it contains mixed signals or indirect language requiring inference beyond surface keywords."
    else:
        return "Easy because clear positive/negative keywords make sentiment unambiguous."

output = []
for item in sel_hard + sel_medium + sel_easy:
    output.append({
        "category": "sentiment",
        "prompt": item["prompt"],
        "expected_answer": item["expected_answer_normalized"],
        "difficulty": item["difficulty"],
        "source": "sentiment-hard-v1",
        "reasoning": build_reasoning(item),
    })

# Shuffle for variety
random.shuffle(output)

# ─── Write output ────────────────────────────────────────────────────────────

output_path = f"{BASE}/generated/sentiment_comprehensive_hard.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Written {len(output)} questions to {output_path}")

# ─── Validation ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALIDATION REPORT")
print("=" * 60)

total = len(output)
print(f"\nTotal questions: {total}")
assert 80 <= total <= 100, f"FAIL: Expected 80-100 questions, got {total}"

diff_counts = defaultdict(int)
ans_counts = defaultdict(int)
for item in output:
    diff_counts[item["difficulty"]] += 1
    ans_counts[item["expected_answer"]] += 1

hard_pct = diff_counts.get("hard", 0) / total * 100
medium_pct = diff_counts.get("medium", 0) / total * 100
easy_pct = diff_counts.get("easy", 0) / total * 100

print(f"\nDifficulty distribution:")
print(f"  Hard:   {diff_counts.get('hard', 0)} ({hard_pct:.1f}%) — target ~40%")
print(f"  Medium: {diff_counts.get('medium', 0)} ({medium_pct:.1f}%) — target ~30%")
print(f"  Easy:   {diff_counts.get('easy', 0)} ({easy_pct:.1f}%) — target ~30%")

print(f"\nExpected answer distribution:")
for a in ["positive", "negative", "neutral", "mixed"]:
    print(f"  {a}: {ans_counts.get(a, 0)}")

# Validate required fields
required_fields = ["category", "prompt", "expected_answer", "difficulty", "source", "reasoning"]
errors = []
for i, item in enumerate(output):
    for field in required_fields:
        if field not in item or not item[field]:
            errors.append(f"Item {i}: missing/empty '{field}'")
    if item["category"] != "sentiment":
        errors.append(f"Item {i}: wrong category '{item['category']}'")
    if item["expected_answer"] not in ("positive", "negative", "neutral", "mixed"):
        errors.append(f"Item {i}: invalid expected_answer '{item['expected_answer']}'")
    if item["difficulty"] not in ("easy", "medium", "hard"):
        errors.append(f"Item {i}: invalid difficulty '{item['difficulty']}'")

# Check duplicate prompts
prompts = [item["prompt"].strip().lower() for item in output]
dupes = len(prompts) - len(set(prompts))
if dupes:
    errors.append(f"Found {dupes} duplicate prompts")

if errors:
    for e in errors:
        print(f"  ✗ {e}")
else:
    print(f"\nAll field validations: ✓")
    print(f"Duplicate prompts:    ✓ (0)")

# Check specific hard variant coverage
print(f"\nHard variant coverage:")
variant_checks = [
    ("brilliant sarcasm", lambda t: "brilliant" in t.lower() and any(w in t.lower() for w in ["canceled"])),
    ("'not entirely terrible' hedging", lambda t: "not entirely terrible" in t.lower()),
    ("'could be worse' hedging", lambda t: "could be worse" in t.lower() or "could do worse" in t.lower()),
    ("'Great X but terrible Y' mixed", lambda t: "great" in t.lower() and "terrible" in t.lower()),
    ("backhanded compliment (brave)", lambda t: "you're so brave" in t.lower()),
    ("dismissive (qualified)", lambda t: "for someone with your qualifications" in t.lower()),
    ("passive-aggressive (admire)", lambda t: "admire your ability" in t.lower()),
    ("deadpan sarcasm", lambda t: "about as well as expected" in t.lower()),
    ("neutral with charged keywords", lambda t: "fraudulent" in t.lower()),
    ("earnings mixed sentiment", lambda t: "earnings" in t.lower() and "missed" in t.lower()),
    ("sarcastic efficiency", lambda t: "efficiency at its finest" in t.lower()),
]
all_covered = True
for vname, vcheck in variant_checks:
    found = sum(1 for item in output if vcheck(item["prompt"]))
    status = "✓" if found > 0 else "✗ MISSING!"
    if found == 0:
        all_covered = False
    print(f"  {vname}: {found} {status}")

if all_covered:
    print(f"\nAll required hard variants covered! ✓")
else:
    print(f"\nSome hard variants missing (see above)")

print(f"\n{'=' * 60}")
print(f"ALL VALIDATIONS {'PASSED ✓' if not errors and all_covered else 'COMPLETED (see warnings)'}")
print(f"{'=' * 60}")

print(f"\nFinal summary:")
print(f"  Total: {total}")
print(f"  Hard: {diff_counts.get('hard', 0)} ({hard_pct:.1f}%)")
print(f"  Medium: {diff_counts.get('medium', 0)} ({medium_pct:.1f}%)")
print(f"  Easy: {diff_counts.get('easy', 0)} ({easy_pct:.1f}%)")
print(f"  Mixed answers: {ans_counts.get('mixed', 0)}")
