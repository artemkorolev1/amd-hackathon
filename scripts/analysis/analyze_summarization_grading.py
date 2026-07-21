#!/usr/bin/env python3
"""
Analyze why fuzzy_match grading fails for summarization answers.
Extracts all summarization items from eval results, runs fuzzy_match,
ROUGE-1, entity recall, keyword overlap, and numeric extraction.
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from grade_answer import fuzzy_match, tokenize, extract_numbers, grade_answer

# ── Load eval results ──────────────────────────────────────────────
EVAL_RESULTS = os.path.join(
    os.path.dirname(__file__),
    "eval_results/comprehensive_eval_qwen2.5-1.5b-instruct_20260713_083928.json"
)

with open(EVAL_RESULTS) as f:
    data = json.load(f)

# ── Extract all summarization items ─────────────────────────────────
summarization_items = [r for r in data["results"] if r.get("category") == "summarization"]
print(f"\n{'='*80}")
print(f"TOTAL SUMMARIZATION ITEMS: {len(summarization_items)}")
print(f"  Reported correct: {data['by_category']['summarization']['correct']}")
print(f"  Reported total:   {data['by_category']['summarization']['total']}")
print(f"  Reported acc:     {data['by_category']['summarization']['accuracy']:.4f}")
print(f"{'='*80}\n")

# ── Helper: ROUGE-1 F1 ─────────────────────────────────────────────
def rouge1_f1(answer, expected):
    """Compute ROUGE-1 F1 score (unigram overlap precision+recall)."""
    a_tokens = tokenize(answer)
    e_tokens = tokenize(expected)
    # Remove stopwords for consistency with fuzzy_match
    STOPWORDS = {"the", "a", "an", "is", "to", "of", "in", "and", "that",
                 "for", "it", "on", "with", "as", "at", "by", "or", "be"}
    a_tokens -= STOPWORDS
    e_tokens -= STOPWORDS
    overlap = len(a_tokens & e_tokens)
    if not a_tokens or not e_tokens:
        return 0.0, overlap, len(e_tokens)
    precision = overlap / len(a_tokens) if a_tokens else 0
    recall = overlap / len(e_tokens) if e_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall, precision

# ── Helper: Entity recall (from worker_summarization.py) ────────────
def extract_entities(text):
    return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

def entity_recall(answer, expected):
    exp_entities = extract_entities(expected)
    out_entities = extract_entities(answer)
    if not exp_entities:
        return 1.0, 0, 0, set()  # no entities to recall
    overlap = exp_entities & out_entities
    return len(overlap) / len(exp_entities), len(overlap), len(exp_entities), overlap

# ── Helper: Keyword overlap (from worker_summarization.py) ──────────
def keyword_overlap(answer, expected):
    exp_words = set(re.findall(r'[a-zA-Z]{4,}', expected.lower()))
    out_words = set(re.findall(r'[a-zA-Z]{4,}', answer.lower()))
    if not exp_words:
        return 1.0, 0, 0
    overlap = len(exp_words & out_words)
    return overlap / len(exp_words), overlap, len(exp_words)

# ── Helper: Numeric extraction comparison ───────────────────────────
def compare_numbers(answer, expected):
    an = extract_numbers(answer)
    en = extract_numbers(expected)
    matching = len(set(an) & set(en))
    return matching, len(en), an, en

# ── Helper: Manual token overlap computation ────────────────────────
def compute_token_overlap(answer, expected):
    """Compute the exact token overlap percentage used by fuzzy_match."""
    a = answer.strip()
    e = expected.strip()
    a_tokens = tokenize(a)
    e_tokens = tokenize(e)
    STOPWORDS = {"the", "a", "an", "is", "to", "of", "in", "and", "that",
                 "for", "it", "on", "with", "as", "at", "by", "or", "be"}
    e_tokens -= STOPWORDS
    a_tokens -= STOPWORDS
    if not e_tokens:
        return 0.0, 0, 0, 0.3
    overlap = len(e_tokens & a_tokens)
    threshold = 0.5 if len(e) < 50 else 0.3
    pct = overlap / len(e_tokens)
    return pct, overlap, len(e_tokens), threshold

# ── Helper: worker_summarization grading ────────────────────────────
def worker_summarization_grade(output, expected):
    """Replicate worker_summarization.py's grading logic."""
    # 1. fuzzy_match cascade
    if fuzzy_match(output, expected):
        return True, "fuzzy_match"

    # 2. Entity overlap
    exp_entities = extract_entities(expected)
    out_entities = extract_entities(output)
    if len(exp_entities) > 0:
        overlap = exp_entities & out_entities
        recall = len(overlap) / len(exp_entities)
        if recall >= 0.5 or len(overlap) >= 2:
            return True, f"entity_recall={recall:.2f}"

    # 3. Keyword overlap (4+ char words)
    exp_words = set(re.findall(r'[a-zA-Z]{4,}', expected.lower()))
    out_words = set(re.findall(r'[a-zA-Z]{4,}', output.lower()))
    if len(exp_words) > 0:
        word_overlap = len(exp_words & out_words) / len(exp_words)
        if word_overlap >= 0.4:
            return True, f"keyword_overlap={word_overlap:.2f}"

    # 4. Number overlap
    exp_nums = set(re.findall(r'\d+(?:\.\d+)?', expected))
    out_nums = set(re.findall(r'\d+(?:\.\d+)?', output))
    if exp_nums and exp_nums & out_nums:
        return True, f"number_overlap"

    return False, "all_failed"


# ── Analyze each item ───────────────────────────────────────────────
failures = []
passes_actual = 0
false_negatives = []
false_positives_by_worker = []
would_pass_worker = 0

for i, item in enumerate(summarization_items):
    expected = item["expected"]
    answer = item["answer"]
    original_result = item["correct"]
    task_id = item["task_id"]

    # fuzzy_match result
    fuzzy_result = fuzzy_match(answer, expected)
    fm_pct, fm_overlap, fm_total, fm_thresh = compute_token_overlap(answer, expected)

    # ROUGE-1
    r1_f1, r1_recall, r1_precision = rouge1_f1(answer, expected)

    # Entity recall
    er, er_overlap, er_total, er_entities = entity_recall(answer, expected)

    # Keyword overlap
    ko, ko_overlap, ko_total = keyword_overlap(answer, expected)

    # Numbers
    num_match, num_total, answer_nums, expect_nums = compare_numbers(answer, expected)

    # Worker summarization grade
    worker_pass, worker_reason = worker_summarization_grade(answer, expected)

    status = "PASS" if fuzzy_result else "FAIL"
    if original_result:
        passes_actual += 1

    # Collect false negatives (original=fail but content looks good)
    if not original_result and (r1_recall >= 0.3 or er >= 0.5 or ko >= 0.4):
        false_negatives.append({
            "task_id": task_id,
            "rouge1_f1": round(r1_f1, 3),
            "rouge1_recall": round(r1_recall, 3),
            "token_overlap_pct": round(fm_pct, 3),
            "token_overlap_threshold": fm_thresh,
            "entity_recall": round(er, 3),
            "keyword_overlap": round(ko, 3),
            "num_extractions": list(zip(answer_nums[:5], expect_nums[:5])),
            "fm_reason": item.get("reason", ""),
            "expected": expected[:150],
            "got": answer[:150],
        })

    if worker_pass and not original_result:
        would_pass_worker += 1

    if fuzzy_result:
        passes_count = 1
    else:
        passes_count = 0

    # Print a detailed line for each item
    er_tag = f"ER={er:.2f}" if er >= 0.5 else f"er={er:.2f}"
    ko_tag = f"KO={ko:.2f}" if ko >= 0.4 else f"ko={ko:.2f}"
    fm_tag = f"FM={fm_pct:.2f}" if fuzzy_result else f"fm={fm_pct:.2f}"
    w_tag = "W+" if worker_pass else "W-"
    print(f"[{'OK' if original_result else 'XX'}] {task_id[:32]:32s} | "
          f"R1_rcl={r1_recall:.3f} R1_f1={r1_f1:.3f} | {fm_tag} | {er_tag} | {ko_tag} | nums={num_match}/{num_total} | {w_tag}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"  Original accuracy (from eval):    {data['by_category']['summarization']['correct']}/{data['by_category']['summarization']['total']} = {data['by_category']['summarization']['accuracy']:.3f}")
print(f"  FuzzyMatch would give:            {sum(1 for i in summarization_items if fuzzy_match(i['answer'], i['expected']))}/{len(summarization_items)}")
print(f"  Worker_summarization would give:  {sum(1 for i in summarization_items if worker_summarization_grade(i['answer'], i['expected'])[0])}/{len(summarization_items)}")
print(f"{'='*80}\n")


# ── Detailed false negative report ──────────────────────────────────
print(f"\n{'='*80}")
print(f"FALSE NEGATIVES ANALYSIS (original=fail but content has merit)")
print(f"{'='*80}")
print(f"Total false negatives with good content: {len(false_negatives)}")
print(f"  - Would pass worker_summarization_grade: {would_pass_worker}")
print()

# Sort by how much content overlap there is (descending ROUGE recall)
false_negatives.sort(key=lambda x: x["rouge1_recall"], reverse=True)

for fn in false_negatives:
    print(f"  ┌─ Task: {fn['task_id']}")
    print(f"  ├─ Token overlap:       {fn['token_overlap_pct']:.1%} (threshold={fn['token_overlap_threshold']:.0%})")
    print(f"  ├─ ROUGE-1 F1:          {fn['rouge1_f1']:.1%}")
    print(f"  ├─ ROUGE-1 Recall:      {fn['rouge1_recall']:.1%}")
    print(f"  ├─ Entity recall:       {fn['entity_recall']:.1%}")
    print(f"  ├─ Keyword overlap:     {fn['keyword_overlap']:.1%}")
    print(f"  ├─ Numeric comparison:  {fn['num_extractions']}")
    print(f"  ├─ FM reason:           {fn['fm_reason'][:100]}")
    print(f"  ├─ Expected (first 150): {fn['expected'][:150]}")
    print(f"  └─ Got (first 150):      {fn['got'][:150]}")
    print()

# ── Category-level grading method comparison ────────────────────────
print(f"\n{'='*80}")
print(f"GRADING METHOD COMPARISON")
print(f"{'='*80}")

# How many issues does each strategy catch?
strategies = {
    "exact_match": 0,
    "substring": 0,
    "numeric": 0,
    "token_overlap": 0,
    "worker_entity": 0,
    "worker_keyword": 0,
    "worker_number": 0,
}

for item in summarization_items:
    a = item["answer"].strip()
    e = item["expected"].strip()

    # Which fuzzy_match strategy catches it?
    a_low = a.lower()
    e_low = e.lower()
    if a_low == e_low:
        strategies["exact_match"] += 1
    elif e_low in a_low or (len(a) >= 3 and a_low in e_low):
        strategies["substring"] += 1
    else:
        na = extract_numbers(a)
        ne = extract_numbers(e)
        numeric_ok = False
        if na and ne:
            if len(na) == len(ne):
                if all(
                    abs(na[i] - ne[i]) <= 0.01 if ne[i] == 0
                    else abs(na[i] - ne[i]) / abs(ne[i]) <= 0.01
                    for i in range(len(ne))
                ):
                    numeric_ok = True
            elif len(ne) == 1:
                for n in na:
                    if ne[0] == 0:
                        if abs(n - ne[0]) <= 0.01:
                            numeric_ok = True
                    elif abs(n - ne[0]) / abs(ne[0]) <= 0.01:
                        numeric_ok = True
        if numeric_ok:
            strategies["numeric"] += 1
        else:
            a_tokens = tokenize(a)
            e_tokens = tokenize(e)
            stopwords = {"the", "a", "an", "is", "to", "of", "in", "and", "that",
                         "for", "it", "on", "with", "as", "at", "by", "or", "be"}
            e_tokens -= stopwords
            a_tokens -= stopwords
            if e_tokens:
                overlap = e_tokens & a_tokens
                threshold = 0.5 if len(e) < 50 else 0.3
                if len(overlap) >= len(e_tokens) * threshold:
                    strategies["token_overlap"] += 1

    # Worker strategies (beyond fuzzy_match)
    if not fuzzy_match(a, e):
        _, reason = worker_summarization_grade(a, e)
        if "entity" in reason:
            strategies["worker_entity"] += 1
        elif "keyword" in reason:
            strategies["worker_keyword"] += 1
        elif "number" in reason:
            strategies["worker_number"] += 1

print(f"  fuzzy_match Exact:      {strategies['exact_match']:3d}")
print(f"  fuzzy_match Substring:  {strategies['substring']:3d}")
print(f"  fuzzy_match Numeric:    {strategies['numeric']:3d}")
print(f"  fuzzy_match TokenOv:    {strategies['token_overlap']:3d}")
print(f"  ──────────────────────────")
print(f"  fuzzy_match total:      {sum(strategies[s] for s in ['exact_match','substring','numeric','token_overlap']):3d}")
print()
print(f"  worker Entity (extra):  {strategies['worker_entity']:3d}")
print(f"  worker Keyword (extra): {strategies['worker_keyword']:3d}")
print(f"  worker Number (extra):  {strategies['worker_number']:3d}")
print(f"  ──────────────────────────")
print(f"  worker total:           {sum(1 for i in summarization_items if worker_summarization_grade(i['answer'], i['expected'])[0]):3d}")
print()

# ── Token overlap threshold analysis ────────────────────────────────
print(f"\n{'='*80}")
print(f"TOKEN OVERLAP THRESHOLD SENSITIVITY")
print(f"{'='*80}")

# For each failure, show the actual overlap pct and what threshold would have passed it
fm_only_fails = [item for item in summarization_items
                 if not fuzzy_match(item['answer'], item['expected'])]
print(f"  Items failing fuzzy_match: {len(fm_only_fails)}")
print()

# What if we lower the token overlap threshold?
for thresh in [0.2, 0.15, 0.1, 0.05]:
    extra = 0
    for item in summarization_items:
        a = item["answer"].strip()
        e = item["expected"].strip()
        if fuzzy_match(a, e):
            continue
        # Try with custom threshold
        a_tokens = tokenize(a)
        e_tokens = tokenize(e)
        stopwords = {"the", "a", "an", "is", "to", "of", "in", "and", "that",
                     "for", "it", "on", "with", "as", "at", "by", "or", "be"}
        e_tokens -= stopwords
        a_tokens -= stopwords
        if e_tokens:
            overlap = len(e_tokens & a_tokens)
            if overlap >= len(e_tokens) * thresh:
                extra += 1
    print(f"  At threshold >= {thresh:.0%}: +{extra} more passes")

# What overlap % distribution do the failures have?
overlap_pcts = []
for item in summarization_items:
    pct, _, _, _ = compute_token_overlap(item["answer"], item["expected"])
    overlap_pcts.append(pct)

print(f"\n  Token overlap distribution for ALL items:")
buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in buckets:
    count = sum(1 for p in overlap_pcts if lo <= p < hi)
    bar = "█" * count
    print(f"    [{lo:.1f}-{hi:.1f}): {count:3d} {bar}")

# ── Expected length distribution ────────────────────────────────────
print(f"\n{'='*80}")
print(f"EXPECTED ANSWER LENGTHS")
print(f"{'='*80}")
lengths = [len(item["expected"]) for item in summarization_items]
print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}")
print(f"  Items with len < 50:  {sum(1 for l in lengths if l < 50)} (use 50% threshold)")
print(f"  Items with len >= 50: {sum(1 for l in lengths if l >= 50)} (use 30% threshold)")
for item in summarization_items:
    e = item["expected"]
    short = "SHORT" if len(e) < 50 else "LONG"
    if not fuzzy_match(item["answer"], e):
        pct, _, _, thresh = compute_token_overlap(item["answer"], e)
        print(f"    [{short}] len={len(e):3d} overlap={pct:.1%} (thresh={thresh:.0%}) fail <-- {item['task_id'][:30]}")

# ── Answer length vs quality analysis ───────────────────────────────
print(f"\n{'='*80}")
print(f"ANSWER LENGTH ANALYSIS")
print(f"{'='*80}")
for item in summarization_items:
    a = item["answer"]
    e = item["expected"]
    if fuzzy_match(a, e):
        continue
    # Check: is the answer too short to cover the expected content?
    if len(a) < len(e) * 0.5:
        print(f"  Short answer ({len(a)} vs {len(e)} exp chars): {item['task_id'][:30]}")

# ── RECOMMENDATIONS ─────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"RECOMMENDATIONS")
print(f"{'='*80}")

print("""
PROBLEM SUMMARY:
────────────────
The fuzzy_match() function uses a token overlap threshold of 30% for answers
>= 50 characters. For summarization, where expected answers are 100-450 chars
of free-form text, the model can produce semantically correct summaries using
completely different vocabulary. The 30% threshold is too high and doesn't
capture synonym usage or paraphrasing.

KEY FINDINGS:
  1. Most summarization failures have token overlap in the 10-25% range
     (relevant content words but different vocabulary)
  2. Entity recall (capitalized names, orgs) often succeeds where fuzzy_match fails
  3. Keyword overlap (4+ char words) catches many more cases than token overlap
  4. The numeric comparison can cause false negatives (e.g., matching "9" vs "19"
     because the wrong number was extracted)
  5. Some XSum items are genuine failures (model just repeats source text verbatim
     instead of summarizing)

RECOMMENDED FIX:
  Option A (Recommended): Replace fuzzy_match for summarization with the
     worker_summarization.py approach (entity_recall + keyword_overlap + number overlap),
     which uses 3 signals that are more robust to vocabulary differences.

  Option B: Lower the token overlap threshold for summarization from 0.3 to 0.15
     and add entity/keyword bonus signals.

  Option C: Add ROUGE-1/ROUGE-L as an alternative grading pathway for the
     summarization category specifically, with a threshold of F1 >= 0.30.

IMPLEMENTATION:
  The simplest fix is to add a 'summarization' branch in grade_answer() or
  grade_answer() that calls worker_summarization_grade() logic for summarization
  items. This catches ~50% of the current false negatives.
""")

# ── Save detailed results to JSON ───────────────────────────────────
output = {
    "total_items": len(summarization_items),
    "original_accuracy": data['by_category']['summarization']['accuracy'],
    "fuzzy_match_accuracy": sum(1 for i in summarization_items if fuzzy_match(i['answer'], i['expected'])) / len(summarization_items),
    "worker_grade_accuracy": sum(1 for i in summarization_items if worker_summarization_grade(i['answer'], i['expected'])[0]) / len(summarization_items),
    "false_negatives": false_negatives,
    "strategy_counts": strategies,
}
with open("/tmp/summarization_grading_analysis.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nDetailed results saved to /tmp/summarization_grading_analysis.json")
