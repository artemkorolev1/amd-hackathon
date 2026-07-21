#!/usr/bin/env python3
"""
Summarization misclassification analysis.
Scans ALL eval files, finds summarization-labeled items,
runs the 8-way classifier, and reports misclassifications with raw scores.
"""

import json
import os
import sys
import glob
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from agent.category_filter import classify, get_short_name, CATEGORIES_8WAY, SCORERS

# ── All eval file paths ──
ALL_PATHS = [
    # input/
    "input/cx_300.json",
    "input/dev_40.json",
    "input/heldout_40.json",
    "input/complexity_40.json",
    # data/eval/primary
    "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json",
    "data/eval/primary/eval_60_medium_hard.json",
    "data/eval/primary/eval_mini_10.json",
    "data/eval/primary/eval_clean_val.json",
    # data/eval/training
    "data/eval/training-v1.json",
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    # data/eval/validation
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
    "data/eval/validation-v3.json",
    # data/eval/tests
    "data/eval/tests/complexity_eval_40.json",
    "data/eval/tests/complexity_eval_candidates.json",
    "data/eval/tests/eval_longform_20.json",
    "data/eval/tests/eval_v14_test_20.json",
    "data/eval/tests/eval_v14_remaining_20.json",
    "data/eval/tests/eval_v14_timeout_stress_19.json",
    "data/eval/tests/fireworks_eval_20.json",
    "data/eval/tests/sst2_100.json",
    "data/eval/tests/gsm8k_100.json",
    "data/eval/tests/gsm8k_test.json",
    # data/eval/generated
    "data/eval/generated/build-A-40.json",
    "data/eval/generated/build-B-40.json",
    "data/eval/generated/eval_from_datasets_20260712_172357.json",
    "data/eval/generated/eval_from_datasets_20260712_172426.json",
    "data/eval/generated/eval_from_datasets_20260712_172443.json",
    "data/eval/generated/sentiment_comprehensive_hard.json",
    # other data/eval
    "data/eval/summarization_combined_25.json",
    "data/eval/sentiment_combined_25.json",
    "data/eval/math_combined_80.json",
    "data/eval/factual_combined_80.json",
    "data/eval/ner_all_models.json",
    "data/eval/tool_eval_filtered.json",
    "data/eval/tool_eval_final.json",
    "data/eval/tool_eval_baseline.json",
    "data/eval/tool_eval_expanded.json",
    "data/eval/final_optimized_run.json",
    "data/eval/eval_question_sets_bundle.json",
    "data/eval/round5_fixes.json",
    "data/eval/smollm_llama_cot.json",
    "data/eval/smollm_llama_test.json",
    "data/eval/final_config_ablation.json",
    "data/eval/coder_ablation.json",
    "data/eval/prompt_ablation_round3.json",
    "data/eval/prompt_ablation_round2.json",
    "data/eval/prompt_ablation_results.json",
    "data/eval/ensemble_router_40q.json",
    "data/eval/three_model_eval_results.json",
    "data/eval/generated/sentiment_comprehensive_hard.json",
    # also scan for any other JSON
]


def resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_HERE, p)


def load_questions(path):
    """Load questions from an eval JSON file."""
    fp = resolve_path(path)
    if not os.path.isfile(fp):
        return []
    try:
        with open(fp) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return []

    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("questions", data.get("items", [data]))
    else:
        return []

    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt", item.get("question", ""))
        if not prompt or not prompt.strip():
            continue

        # Extract category label (try multiple field names)
        cat = (
            item.get("category")
            or item.get("category_label")
            or item.get("label")
            or item.get("label_8way")
        )
        if not cat:
            continue

        short_cat = get_short_name(cat)
        task_id = item.get("task_id", f"{os.path.basename(path)}_q{i}")

        result.append({
            "task_id": task_id,
            "prompt": prompt,
            "true_category": short_cat,
            "true_category_raw": cat,
            "_source_file": path,
        })
    return result


def run_scorers(prompt):
    """Run each scorer individually and return scores dict."""
    from agent.category_filter import SCORERS, PRIORITY
    lower = prompt.lower()
    scores = {}
    for cat, scorer_fn in SCORERS.items():
        scores[cat] = scorer_fn(prompt)
    
    # Also apply the V4 post-processing adjustments from classify()
    import re
    scores_copy = dict(scores)
    
    # Math guard
    _hmk = bool(re.search(
        r"(solve|calculate|compute|equation|formula|derivative|integral|"
        r"algebra|geometry|trig|calculus|probability|permutation|"
        r"combination|factorial|matrix|vector)", lower))
    _hmc = bool(re.search(
        r"(liters|gallons|kilograms|meters|kilometers|miles|hours|"
        r"minutes|percent|ratio|mixture|distance|speed|velocity|rate|"
        r"time|work|age|interest|discount|profit|loss|area|volume|"
        r"perimeter)", lower))
    _hcm = bool(re.search(r"^(Context|Passage|Text|Article):", prompt, re.I))
    if scores_copy.get("math", 0) > 0 and not _hmk and not _hmc and _hcm:
        scores_copy["math"] = scores_copy.get("math", 0) * 0.1
    
    # Logic boost
    _logic_boost = 0.0
    if not _hcm:
        if re.search(r"(reservations|appointments|meetings)", lower) and \
           re.search(r"\d+:\d{2}", prompt) and \
           re.search(r"(at \d+|different time|arrang)", lower):
            _logic_boost = max(_logic_boost, 4.0)
        if re.search(r"i am (thinking|looking)", lower) and \
           re.search(r"(digit|number|word|clue|hint|crypt|pattern)", lower):
            _logic_boost = max(_logic_boost, 4.0)
        if len(re.findall(r"[A-Z][a-z]+", prompt)) >= 2 and \
           re.search(r"(different|distinct|each \w+ (has|is|works|owns|"
                     r"lives|sits|drives|likes))", lower):
            _logic_boost = max(_logic_boost, 3.0)
        if re.search(r"\b(why|explain|reason)\b", lower) and \
           re.search(r"\b(refract|prism|spectrum|light|wavelength|lens|"
                     r"mirror|gravity|acceleration|force|velocity|momentum|"
                     r"electric|magnetic|circuit|current|voltage|resistance|"
                     r"converge|gradient|derivative|algorithm|asymptotic|"
                     r"time complexity|big O|O\(n|proof|theorem)\b", lower):
            _logic_boost = max(_logic_boost, 4.0)
        if re.search(r"\b(why (does|is|are|do|would|did)|explain (why|how|step))\b", lower):
            _logic_boost = max(_logic_boost, 2.0)
    if _logic_boost > 0:
        scores_copy["logic"] = scores_copy.get("logic", 0) + _logic_boost
    
    # Summarization post-processing
    _ner_task = bool(re.search(
        r"(extract (all )?|identify (the )?"
        r"(persons|organizations|locations|entities))", lower))
    if re.search(r"^(On|In|At) \w+ \d{1,2},? \d{4}", prompt) and not _ner_task:
        scores_copy["summarization"] = scores_copy.get("summarization", 0) + 2.0
    if re.search(r"(HEADLINE|DATELINE|BREAKING|BRIEF|MEMORANDUM)", prompt) \
       and not _ner_task:
        scores_copy["summarization"] = scores_copy.get("summarization", 0) + 2.0
    
    # Code gen: typing imports should not trigger math
    if scores_copy.get("code_gen", 0) > 0 and \
       re.search(r"(from \w+ import|def \w+\()", prompt) and \
       scores_copy.get("math", 0) > scores_copy.get("code_gen", 0):
        scores_copy["math"] = scores_copy.get("math", 0) * 0.2
    
    # Final winner
    sorted_scores = sorted(
        scores_copy.items(),
        key=lambda x: (-x[1], -PRIORITY.get(x[0], 0))
    )
    best_cat = sorted_scores[0][0]
    best_score = sorted_scores[0][1]
    
    return best_cat, best_score, scores_copy


def main():
    output_path = "/home/artem/dev/amd-hackathon/summarization_analysis.md"
    
    # ── Collect all summarization prompts ──
    all_summarization = []
    found_files = set()
    
    print("=" * 80)
    print("SCANNING ALL EVAL FILES FOR SUMMARIZATION-LABELED ITEMS")
    print("=" * 80)
    
    # First check paths
    for path in ALL_PATHS:
        questions = load_questions(path)
        if not questions:
            continue
        found_files.add(path)
        for q in questions:
            if q["true_category"] == "summarization":
                all_summarization.append(q)
    
    # Also glob for any JSON files in data/eval/ recursively
    for fpath in sorted(glob.glob(os.path.join(_HERE, "data/eval/**/*.json"), recursive=True)):
        relpath = os.path.relpath(fpath, _HERE)
        if relpath not in ALL_PATHS:
            questions = load_questions(relpath)
            if questions:
                found_files.add(relpath)
                for q in questions:
                    if q["true_category"] == "summarization":
                        all_summarization.append(q)
    
    print(f"\nFound {len(found_files)} eval files with category labels")
    print(f"Found {len(all_summarization)} summarization-labeled items\n")
    
    if not all_summarization:
        print("NO SUMMARIZATION ITEMS FOUND!")
        with open(output_path, "w") as f:
            f.write("# Summarization Analysis\n\nNo summarization-labeled items found in any eval file.\n")
        return
    
    # ── Run classifier on each ──
    misclassified = []
    correct = []
    
    for q in all_summarization:
        prompt = q["prompt"]
        true_cat = q["true_category"]
        task_id = q["task_id"]
        source = q["_source_file"]
        
        best_cat, best_score, scores = run_scorers(prompt)
        # Also run the original classify for comparison
        pred_classify, conf, scores_classify = classify(prompt)
        
        is_correct = (best_cat == "summarization")
        
        entry = {
            "task_id": task_id,
            "source": source,
            "prompt": prompt,
            "true_category": true_cat,
            "predicted": best_cat,
            "confidence": best_score,
            "scores": {k: round(v, 3) for k, v in sorted(scores.items())},
            "is_correct": is_correct,
        }
        
        if is_correct:
            correct.append(entry)
        else:
            misclassified.append(entry)
    
    total = len(all_summarization)
    correct_count = len(correct)
    mis_count = len(misclassified)
    
    print(f"Total summarization items: {total}")
    print(f"Correctly classified:      {correct_count} ({correct_count/total*100:.1f}%)")
    print(f"Misclassified:            {mis_count} ({mis_count/total*100:.1f}%)")
    
    # ── Group misclassifications by predicted category ──
    mis_by_pred = defaultdict(list)
    for m in misclassified:
        mis_by_pred[m["predicted"]].append(m)
    
    print(f"\nMisclassification breakdown:")
    sorted_mis = sorted(mis_by_pred.items(), key=lambda x: -len(x[1]))
    for pred_cat, items in sorted_mis:
        print(f"  {pred_cat:>15}: {len(items)} ({len(items)/mis_count*100:.1f}% of misclassified)")
    
    # Top 3
    top3 = sorted_mis[:3]
    
    # ── Build report ──
    report_lines = []
    report_lines.append("# Summarization Classification Failure Analysis")
    report_lines.append("")
    report_lines.append(f"**Date:** Analysis run")
    report_lines.append(f"**Total summarization-labeled items found:** {total}")
    report_lines.append(f"**Correctly classified as summarization:** {correct_count} ({correct_count/total*100:.1f}%)")
    report_lines.append(f"**Misclassified:** {mis_count} ({mis_count/total*100:.1f}%)")
    report_lines.append("")
    report_lines.append(f"**Eval files scanned:** {len(found_files)}")
    report_lines.append("")
    
    # ── Summary table ──
    report_lines.append("## Summary of Misclassifications")
    report_lines.append("")
    report_lines.append(f"| Predicted As | Count | % of Misclassified |")
    report_lines.append(f"|-------------|-------|-------------------|")
    for pred_cat, items in sorted_mis:
        report_lines.append(f"| {pred_cat} | {len(items)} | {len(items)/mis_count*100:.1f}% |")
    report_lines.append("")
    
    # ── Confusion matrix for summarization only ──
    report_lines.append("## Confusion (Summarization Row Only)")
    report_lines.append("")
    report_lines.append(f"| True\\Pred | " + " | ".join(CATEGORIES_8WAY) + " |")
    report_lines.append("|" + "|".join("---" for _ in range(len(CATEGORIES_8WAY)+1)) + "|")
    row = ["summarization"]
    for cat in CATEGORIES_8WAY:
        if cat == "summarization":
            row.append(str(correct_count))
        else:
            count = len(mis_by_pred.get(cat, []))
            row.append(str(count))
    report_lines.append("| " + " | ".join(row) + " |")
    report_lines.append("")
    
    # ── Top 3 patterns with prompts and scores ──
    report_lines.append("## Top 3 Misclassification Patterns")
    report_lines.append("")
    
    for rank, (pred_cat, items) in enumerate(top3, 1):
        report_lines.append(f"### {rank}. Summarization → {pred_cat} ({len(items)} cases)")
        report_lines.append("")
        
        # Show up to 5 examples
        for i, item in enumerate(items[:5]):
            prompt_preview = item["prompt"][:200].replace("\n", "\\n")
            report_lines.append(f"**Example {i+1}:** task_id=`{item['task_id']}`")
            report_lines.append(f"**Source:** `{item['source']}`")
            report_lines.append(f"**Prompt (first 200 chars):**")
            report_lines.append(f"```")
            report_lines.append(f"{prompt_preview}")
            report_lines.append(f"```")
            report_lines.append("")
            report_lines.append(f"**Raw scores from all 8 scorers:**")
            report_lines.append("")
            # Show scores sorted by value descending
            sorted_scores = sorted(item["scores"].items(), key=lambda x: -x[1])
            report_lines.append(f"| Scorer | Score |")
            report_lines.append(f"|--------|-------|")
            for sc_name, sc_val in sorted_scores:
                marker = " ← **WINS**" if sc_name == item["predicted"] else ""
                report_lines.append(f"| {sc_name} | {sc_val}{marker} |")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if len(items) > 5:
            report_lines.append(f"... and {len(items) - 5} more similar misclassifications.")
            report_lines.append("")
    
    # ── Full detail: ALL misclassified items ──
    report_lines.append("## Complete List of All Misclassified Items")
    report_lines.append("")
    
    for item in misclassified:
        prompt_preview = item["prompt"][:200].replace("\n", "\\n")
        report_lines.append(f"### `{item['task_id']}` → predicted **{item['predicted']}**")
        report_lines.append(f"**Source:** `{item['source']}`")
        report_lines.append(f"**Prompt (first 200 chars):**")
        report_lines.append(f"```")
        report_lines.append(f"{prompt_preview}")
        report_lines.append(f"```")
        report_lines.append("")
        report_lines.append("**All scorer scores:**")
        report_lines.append("")
        sorted_scores = sorted(item["scores"].items(), key=lambda x: -x[1])
        for sc_name, sc_val in sorted_scores:
            marker = " ← WINS" if sc_name == item["predicted"] else ""
            report_lines.append(f"- {sc_name}: {sc_val}{marker}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # ── Also list all correct items for completeness ──
    report_lines.append("## Correctly Classified Items")
    report_lines.append("")
    report_lines.append(f"({correct_count} items correctly classified as summarization)")
    report_lines.append("")
    for item in correct:
        report_lines.append(f"- `{item['task_id']}` (from `{item['source']}`)")
    report_lines.append("")
    
    # ── Analysis / Insights ──
    report_lines.append("## Insights & Root Cause Analysis")
    report_lines.append("")
    
    for pred_cat, items in top3:
        # What's the average summarization score vs the winning score?
        avg_sum_score = sum(it["scores"].get("summarization", 0) for it in items) / len(items)
        avg_win_score = sum(it["scores"].get(pred_cat, 0) for it in items) / len(items)
        report_lines.append(f"### Summarization → {pred_cat}")
        report_lines.append(f"- Count: {len(items)}")
        report_lines.append(f"- Avg summarization score: {avg_sum_score:.3f}")
        report_lines.append(f"- Avg {pred_cat} score (winner): {avg_win_score:.3f}")
        report_lines.append(f"- Avg score gap: {avg_win_score - avg_sum_score:.3f}")
        report_lines.append("")
        
        # Check what triggers the winning scorer
        if pred_cat == "factual":
            report_lines.append("  Likely trigger: Question-word patterns (who/what/when/where/why/how)")
            report_lines.append("  in prompts that ask about a text rather than requesting a summary.")
        elif pred_cat == "logic":
            report_lines.append("  Likely trigger: Constraint patterns, paragraph breaks, named entities")
            report_lines.append("  in narrative prose being interpreted as logic puzzles.")
        elif pred_cat == "math":
            report_lines.append("  Likely trigger: Numbers in the prompt triggering math scorer, or")
            report_lines.append("  arithmetic operations in narrative context.")
        elif pred_cat == "ner":
            report_lines.append("  Likely trigger: Extraction-style phrasing ('find the', 'identify')")
            report_lines.append("  that overlaps with summarization tasks.")
        elif pred_cat == "code_gen":
            report_lines.append("  Likely trigger: Code-related keywords in technical summarization prompts.")
        elif pred_cat == "sentiment":
            report_lines.append("  Likely trigger: Opinion/tone words in sentiment analysis prompts.")
        elif pred_cat == "code_debug":
            report_lines.append("  Likely trigger: Technical prompts with error/fix terminology.")
        report_lines.append("")
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport written to {output_path}")
    
    # ── Also print to stdout a concise summary ──
    print("\n" + "=" * 80)
    print("CONCISE SUMMARY")
    print("=" * 80)
    print(f"Total summarization: {total}")
    print(f"Correct: {correct_count} ({correct_count/total*100:.1f}%)")
    print(f"Misclassified: {mis_count} ({mis_count/total*100:.1f}%)")
    print()
    print("By misclassification type:")
    for pred_cat, items in sorted_mis:
        pct = len(items)/mis_count*100 if mis_count > 0 else 0
        print(f"  → {pred_cat:<15} {len(items):>4} ({pct:>5.1f}%)")
    print()
    print("TOP 3 patterns:")
    for rank, (pred_cat, items) in enumerate(top3, 1):
        print(f"  {rank}. Summarization → {pred_cat} ({len(items)} examples)")
        # Show first example's scores
        ex = items[0]
        sorted_scores = sorted(ex["scores"].items(), key=lambda x: -x[1])
        print(f"     Top-3 scores: {sorted_scores[:3]}")
        print(f"     Prompt preview: {ex['prompt'][:120]}")
        print()
    
    return misclassified, correct


if __name__ == "__main__":
    main()
