#!/usr/bin/env python3
"""
Benchmark V3: Compare strategies by pre-computing all classifier outputs once.
Batch ML inference to avoid OOM.
"""
import importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cat_mod = _load_module(_PROJECT_ROOT / "agent" / "category_filter.py", "category_filter")
CATEGORIES_8WAY = _cat_mod.CATEGORIES_8WAY
regex_classify = _cat_mod.classify
get_short_name = _cat_mod.get_short_name

_ml_mod = _load_module(_PROJECT_ROOT / "agent" / "ml_classifier.py", "ml_classifier")
pipeline = _ml_mod._get_pipeline()

EVAL_FILES = [
    "input/dev_40.json", "input/heldout_40.json", "input/complexity_40.json", "input/cx_300.json",
    "data/eval/primary/eval_60_medium_hard.json",
    "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json",
    "data/eval/primary/eval_mini_10.json",
    "data/eval/training-v1.json", "data/eval/training-v2.json", "data/eval/training-v3.json",
    "data/eval/validation-v1.json", "data/eval/validation-v2.json", "data/eval/validation-v3.json",
    "data/eval/tests/complexity_eval_40.json", "data/eval/tests/eval_longform_20.json",
    "data/eval/tests/eval_v14_test_20.json", "data/eval/tests/eval_v14_remaining_20.json",
    "data/eval/tests/eval_v14_timeout_stress_19.json", "data/eval/tests/fireworks_eval_20.json",
    "data/eval/generated/build-A-40.json", "data/eval/generated/build-B-40.json",
]

def load_questions(path):
    fp = _PROJECT_ROOT / path if not os.path.isabs(path) else Path(path)
    if not fp.exists():
        return []
    with open(fp) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("questions", data.get("items", [data]))
    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt", item.get("question", ""))
        if not prompt or not prompt.strip():
            continue
        cat = item.get("category") or item.get("category_label") or item.get("label") or item.get("label_8way")
        if not cat:
            continue
        result.append({"prompt": prompt, "true_category": get_short_name(cat)})
    return result

def main():
    print("=" * 80)
    print("  BENCHMARK V3: All Strategies (batch ML)")
    print("=" * 80)
    
    # Load
    all_qs = []
    for path in EVAL_FILES:
        qs = load_questions(path)
        if qs:
            all_qs.extend(qs)
    N = len(all_qs)
    print(f"\n  Questions: {N}")
    print()
    
    # PRE-COMPUTE: batch regex scores once
    print("  Computing regex scores...")
    t0 = time.monotonic()
    regex_results = []
    for q in all_qs:
        pred, conf, scores = regex_classify(q["prompt"])
        regex_results.append({"pred": pred, "conf": conf, "scores": scores})
    print(f"    Done in {time.monotonic()-t0:.2f}s")
    
    # PRE-COMPUTE: batch ML predictions once
    print("  Computing ML predictions...")
    t0 = time.monotonic()
    ml_probs = pipeline.predict_proba([q["prompt"] for q in all_qs])
    ml_classes = list(pipeline.classes_)
    print(f"    Done in {time.monotonic()-t0:.2f}s")
    
    cat_to_idx = {c: i for i, c in enumerate(ml_classes)}
    
    def accuracy(predictions):
        return sum(1 for q, p in zip(all_qs, predictions) if p == q["true_category"])
    
    def per_cat(predictions):
        pc = defaultdict(lambda: {"correct": 0, "total": 0})
        for q, p in zip(all_qs, predictions):
            pc[q["true_category"]]["total"] += 1
            if p == q["true_category"]:
                pc[q["true_category"]]["correct"] += 1
        return dict(pc)
    
    def print_strat(name, predictions):
        correct = accuracy(predictions)
        acc = round(correct / N * 100, 1)
        return correct, N, acc
    
    def per_cat_table(predictions):
        pc = per_cat(predictions)
        print(f"  {'Category':<20} {'Correct':>8} {'Total':>6} {'Acc%':>8}")
        print(f"  {'-'*42}")
        for cat in sorted(CATEGORIES_8WAY):
            v = pc.get(cat, {"correct": 0, "total": 0})
            a = round(v["correct"] / max(v["total"], 1) * 100, 1)
            print(f"  {cat:<20} {v['correct']:>8} {v['total']:>6} {a:>7.1f}%")
        return pc
    
    def diff_table(r_preds, c_preds, label="Strategy"):
        r_pc = per_cat(r_preds)
        c_pc = per_cat(c_preds)
        print(f"  {'Category':<20} {'Regex%':>8} {label:>10} {'Change':>8}")
        print(f"  {'-'*46}")
        for cat in sorted(CATEGORIES_8WAY):
            rv = r_pc.get(cat, {"correct": 0, "total": 1})
            cv = c_pc.get(cat, {"correct": 0, "total": 1})
            rp = round(rv["correct"] / max(rv["total"], 1) * 100, 1)
            cp = round(cv["correct"] / max(cv["total"], 1) * 100, 1)
            cd = cp - rp
            sign = "+" if cd > 0 else ""
            print(f"  {cat:<20} {rp:>7.1f}% {cp:>8.1f}% {sign}{cd:>6.1f}%")
    
    # ===================================================================
    # Strategy 1: Regex-only
    # ===================================================================
    r_preds = [r["pred"] for r in regex_results]
    r_correct, r_N, r_acc = print_strat("1. Regex-Only", r_preds)
    print(f"    Accuracy: {r_correct}/{r_N} = {r_acc}%\n")
    
    # ===================================================================
    # Strategy 2: ML-only
    # ===================================================================
    m_preds = [get_short_name(ml_classes[np.argmax(ml_probs[i])]) for i in range(N)]
    m_correct, m_N, m_acc = print_strat("2. ML-Only", m_preds)
    print(f"    Accuracy: {m_correct}/{m_N} = {m_acc}%")
    diff_table(r_preds, m_preds, "ML-Only%")
    print()
    
    # ===================================================================
    # Strategy 3: ML-first, regex fallback (vary threshold)
    # ===================================================================
    best_ml_first = None
    best_ml_first_acc = 0
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = []
        ml_count = 0
        reg_count = 0
        for i in range(N):
            probs = ml_probs[i]
            best_idx = np.argmax(probs)
            ml_conf = float(probs[best_idx])
            if ml_conf >= thresh:
                preds.append(get_short_name(ml_classes[best_idx]))
                ml_count += 1
            else:
                preds.append(r_preds[i])
                reg_count += 1
        c, _, a = print_strat(f"3. ML-first (thresh={thresh})", preds)
        print(f"    ML={ml_count}, regex_fallback={reg_count}")
        if a > best_ml_first_acc:
            best_ml_first = preds
            best_ml_first_acc = a
        if thresh == 0.5:
            diff_table(r_preds, preds, "ML-first%")
            print()
    
    # ===================================================================
    # Strategy 4: Regex-first, ML fallback (vary threshold)
    # ===================================================================
    for thresh in [0.5, 0.6, 0.7]:
        preds = []
        r_count = 0
        ml_count = 0
        fall_count = 0
        for i in range(N):
            rr = regex_results[i]
            if rr["conf"] >= thresh:
                preds.append(rr["pred"])
                r_count += 1
            else:
                probs = ml_probs[i]
                best_idx = np.argmax(probs)
                ml_conf = float(probs[best_idx])
                if ml_conf >= thresh:
                    preds.append(get_short_name(ml_classes[best_idx]))
                    ml_count += 1
                else:
                    preds.append(rr["pred"])
                    fall_count += 1
        c, _, a = print_strat(f"4. Regex-first (thresh={thresh})", preds)
        print(f"    regex={r_count}, ml={ml_count}, fallback={fall_count}")
    
    # ===================================================================
    # Strategy 5: Ensemble (weighted average)
    # ===================================================================
    for ml_w in [0.3, 0.5, 0.7]:
        preds = []
        for i in range(N):
            rr = regex_results[i]
            max_reg = max(list(rr["scores"].values())) if rr["scores"] else 1
            ensemble = {}
            for cat in CATEGORIES_8WAY:
                reg_val = rr["scores"].get(cat, 0) / max_reg if max_reg > 0 else 0
                ml_val = float(ml_probs[i][cat_to_idx[cat]]) if cat in cat_to_idx else 0
                ensemble[cat] = (1 - ml_w) * reg_val + ml_w * ml_val
            preds.append(max(ensemble, key=ensemble.get))
        c, _, a = print_strat(f"5. Ensemble (ml_w={ml_w})", preds)
        if ml_w == 0.5:
            diff_table(r_preds, preds, "Ensemble%")
            print()
    
    # ===================================================================
    # Summary
    # ===================================================================
    print()
    print(f"  {'='*60}")
    print(f"  SUMMARY TABLE")
    print(f"  {'='*60}")
    print(f"  {'Strategy':<45} {'Correct':>8} {'Total':>6} {'Acc%':>6} {'Delta':>6}")
    print(f"  {'-'*71}")
    
    entries = [("1. Regex-Only (baseline)", r_preds)]
    entries.append(("2. ML-Only", m_preds))
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = []
        for i in range(N):
            probs = ml_probs[i]
            best_idx = np.argmax(probs)
            if float(probs[best_idx]) >= thresh:
                preds.append(get_short_name(ml_classes[best_idx]))
            else:
                preds.append(r_preds[i])
        entries.append((f"3. ML-first (thresh={thresh})", preds))
    
    for thresh in [0.5, 0.6, 0.7]:
        preds = []
        for i in range(N):
            rr = regex_results[i]
            if rr["conf"] >= thresh:
                preds.append(rr["pred"])
            else:
                probs = ml_probs[i]
                best_idx = np.argmax(probs)
                if float(probs[best_idx]) >= thresh:
                    preds.append(get_short_name(ml_classes[best_idx]))
                else:
                    preds.append(rr["pred"])
        entries.append((f"4. Regex-first (thresh={thresh})", preds))
    
    for ml_w in [0.3, 0.5, 0.7]:
        preds = []
        for i in range(N):
            rr = regex_results[i]
            max_reg = max(list(rr["scores"].values())) if rr["scores"] else 1
            ensemble = {}
            for cat in CATEGORIES_8WAY:
                reg_val = rr["scores"].get(cat, 0) / max_reg if max_reg > 0 else 0
                ml_val = float(ml_probs[i][cat_to_idx[cat]]) if cat in cat_to_idx else 0
                ensemble[cat] = (1 - ml_w) * reg_val + ml_w * ml_val
            preds.append(max(ensemble, key=ensemble.get))
        entries.append((f"5. Ensemble (ml_w={ml_w})", preds))
    
    for name, preds in entries:
        c = accuracy(preds)
        a = round(c / N * 100, 1)
        d = round(a - r_acc, 1)
        ds = f"+{d}" if d >= 0 else str(d)
        print(f"  {name:<45} {c:>8} {N:>6} {a:>5.1f}% {ds:>6}pp")
    
    # ===================================================================
    # CODE_DEBUG DEEP DIVE with best strategy
    # ===================================================================
    if best_ml_first is not None:
        print()
        print(f"  {'='*60}")
        print(f"  BEST STRATEGY ANALYSIS: ML-first (thresh=0.5)")
        print(f"  {'='*60}")
        
        r_db = per_cat(r_preds).get("code_debug", {"correct": 0, "total": 0})
        c_db = per_cat(best_ml_first).get("code_debug", {"correct": 0, "total": 0})
        print(f"    Regex code_debug:   {r_db['correct']}/{r_db['total']} = {round(r_db['correct']/max(r_db['total'],1)*100,1)}%")
        print(f"    Cascade code_debug: {c_db['correct']}/{c_db['total']} = {round(c_db['correct']/max(c_db['total'],1)*100,1)}%")
        
        fixed = 0
        regressed = 0
        for q, rp, cp in zip(all_qs, r_preds, best_ml_first):
            true_cat = q["true_category"]
            if rp != true_cat and cp == true_cat:
                fixed += 1
            elif rp == true_cat and cp != true_cat:
                regressed += 1
        print(f"    Regex errors FIXED:        {fixed}")
        print(f"    Cascade regressions:       {regressed}")
        print(f"    Net improvement:           {fixed - regressed}")

if __name__ == "__main__":
    main()
