#!/usr/bin/env python3
"""v12e comprehensive eval — Qwen2.5-1.5B + LoRA + MiniLM complexity."""
import json, logging, os, sys, time
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("eval")

os.environ["COMPLEXITY_MODEL_DIR"] = "/home/artem/dev/amd-hackathon-shared/classifiers/best_complexity_model"
os.environ["FIREWORKS_API_KEY"] = ""

# Import pipeline components
from agent.complexity import score as c_score  # MiniLM ML scorer
from agent.category_filter import classify
from agent.solvers import local_model
from agent.dynamic_prompts import build_system_prompt, get_max_tokens, NER_ONE_SHOT_EXAMPLE, SENTIMENT_EXAMPLES, MATH_EXAMPLES
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
    solve_factual_qa, solve_code_debugging,
)
from agent.pre_filter import stage0

# Attach official grader
MAIN_REPO = "/home/artem/dev/amd-hackathon"
sys.path.insert(1, MAIN_REPO)
from evaluate import fuzzy_match

EVAL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/artem/dev/amd-hackathon-shared/eval_all_300.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DET_CAT_MAP = {
    "math": "math_arithmetic", "logic": "logical_reasoning",
    "sentiment": "sentiment", "ner": "named_entity_recognition",
    "factual": "factual_knowledge", "code_debug": "code_debugging",
    "summarization": "summarization",
    "general": "other_complex",
}
LORA_CATEGORIES = {"logic", "ner", "summarization", "sentiment", "factual", "code_debug", "code_gen", "math"}
DET_SOLVERS = {
    "solve_arithmetic": solve_arithmetic, "solve_logic": solve_logic,
    "solve_sentiment": solve_sentiment, "solve_ner": solve_ner,
    "solve_factual_qa": solve_factual_qa, "solve_code_debugging": solve_code_debugging,
}

def load_tasks(path):
    with open(path) as f:
        raw = json.load(f)
    questions = raw if isinstance(raw, list) else raw.get("questions", [])
    tasks = []
    for i, q in enumerate(questions):
        tasks.append({
            "idx": i, "task_id": q.get("task_id", f"q_{i}"),
            "prompt": q.get("prompt", ""), "expected": q.get("expected_answer", ""),
            "category_label": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
        })
    return tasks

def run_one(q):
    prompt, expected = q["prompt"], q["expected"]
    rec = {"idx": q["idx"], "task_id": q["task_id"], "prompt": prompt,
           "expected": expected, "category_label": q["category_label"],
           "difficulty": q["difficulty"], "correct": False, "final_answer": "",
           "stages": {}, "latency": {}, "answer_source": "none"}

    t_start = time.monotonic()
    try:
        s0 = stage0(prompt)
        if s0.action == "bypass" and s0.direct_answer:
            rec["final_answer"] = s0.direct_answer
            rec["correct"] = fuzzy_match(s0.direct_answer, expected)
            rec["stages"]["stage0"] = {"bypassed": True}
            rec["latency"]["total"] = time.monotonic() - t_start
            rec["answer_source"] = "stage0"
            return rec
    except Exception as e:
        rec["stages"]["stage0"] = {"error": str(e)}

    # Stage 2
    t0 = time.monotonic()
    try:
        cat, conf, scores = classify(prompt)
        rec["stages"]["stage2"] = {"category": cat, "confidence": conf, "scores": scores}
    except Exception as e:
        cat, conf, scores = "general", 0, {}
        rec["stages"]["stage2"] = {"error": str(e), "category": cat}
    rec["latency"]["stage2"] = time.monotonic() - t0

    # Complexity
    t0 = time.monotonic()
    try:
        cx = c_score(prompt)
        rec["stages"]["complexity"] = {"score": cx}
    except Exception as e:
        cx = 0.5
        rec["stages"]["complexity"] = {"error": str(e), "score": cx}
    rec["latency"]["complexity"] = time.monotonic() - t0

    # Try deterministic solvers
    t0 = time.monotonic()
    det_answer = None
    for sname, sfn in DET_SOLVERS.items():
        try:
            ans = sfn(prompt, DET_CAT_MAP.get(cat, "general"))
            if ans:
                det_answer = ans
                logger.info(f"  [{q['task_id']}] DET {sname}: {ans[:60]}")
                break
        except Exception:
            logger.debug("DET solver %s skipped for %s", sname, q.get("task_id", "?"))
    rec["latency"]["deterministic"] = time.monotonic() - t0

    # Decision
    needs_api = not bool(det_answer)
    if needs_api:
        custom_instr = None
        if os.environ.get("FEW_SHOT", "1") == "1":
            if cat == "ner":
                custom_instr = NER_ONE_SHOT_EXAMPLE
            elif cat == "sentiment":
                custom_instr = SENTIMENT_EXAMPLES
            elif cat == "math":
                custom_instr = MATH_EXAMPLES
        sys_prompt = build_system_prompt(cat, cx, custom_instructions=custom_instr)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        t0 = time.monotonic()
        try:
            ans = local_model.chat_completion(messages, category=cat if cat in LORA_CATEGORIES else "",
                                              max_tokens=get_max_tokens(cat, cx))
            rec["final_answer"] = ans or ""
            rec["answer_source"] = f"local_{cat}" if ans else "empty"
            rec["latency"]["local_model"] = time.monotonic() - t0
        except Exception as e:
            rec["final_answer"] = ""
            rec["answer_source"] = f"local_error:{e}"
            rec["latency"]["local_model"] = time.monotonic() - t0
    else:
        rec["final_answer"] = det_answer
        rec["answer_source"] = "deterministic"

    rec["correct"] = fuzzy_match(rec["final_answer"], expected)
    rec["latency"]["total"] = time.monotonic() - t_start
    return rec

def main():
    tasks = load_tasks(EVAL_PATH)
    n = len(tasks)
    logger.info(f"Loaded {n} tasks")

    results = []
    cat_stats = defaultdict(lambda: {"n": 0, "correct": 0, "lat": []})
    diff_stats = defaultdict(lambda: {"n": 0, "correct": 0, "lat": []})
    source_stats = Counter()

    for idx, q in enumerate(tasks):
        rec = run_one(q)
        results.append(rec)

        cat_short = q["category_label"].split("_")[0]
        cat_stats[cat_short]["n"] += 1
        cat_stats[cat_short]["lat"].append(rec["latency"]["total"])
        diff_stats[q["difficulty"]]["n"] += 1
        diff_stats[q["difficulty"]]["lat"].append(rec["latency"]["total"])
        source_stats[rec["answer_source"]] += 1

        if rec["correct"]:
            cat_stats[cat_short]["correct"] += 1
            diff_stats[q["difficulty"]]["correct"] += 1

        mark = "✓" if rec["correct"] else "✗"
        agg = rec.get("answer_source", "?")[:16]
        print(f"  [{idx+1:>3d}/{n}] {mark} | {agg:>16s} | {rec['latency']['total']:>5.1f}s | {rec['final_answer'][:50] if rec['final_answer'] else '(empty)'}")
        sys.stdout.flush()

    total_correct = sum(1 for r in results if r["correct"])
    total_time = sum(r["latency"]["total"] for r in results)
    acc = 100 * total_correct / n

    print(f"\n{'='*70}")
    print(f"  RESULTS: {total_correct}/{n} = {acc:.1f}% | Total time: {total_time:.0f}s")
    print(f"{'='*70}")
    print(f"  {'Category':<18s} {'Acc':>8s} {'Count':>6s} {'AvgLat':>7s}")
    for cat, s in sorted(cat_stats.items()):
        cacc = 100 * s["correct"] / s["n"]
        al = sum(s["lat"]) / len(s["lat"]) if s["lat"] else 0
        print(f"  {cat:<18s} {cacc:>6.1f}% ({s['correct']}/{s['n']}) {al:>6.1f}s")

    print(f"\n  {'Difficulty':<12s} {'Acc':>8s} {'Count':>6s}")
    for diff, s in sorted(diff_stats.items()):
        dacc = 100 * s["correct"] / s["n"] if s["n"] else 0
        print(f"  {diff:<12s} {dacc:>6.1f}% ({s['correct']}/{s['n']})")

    print(f"\n  {'Answer Source':<30s} {'Count':>6s}")
    for src, cnt in source_stats.most_common():
        print(f"  {src:<30s} {cnt:>4d}/{n}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(OUTPUT_DIR, f"eval_v12e_{ts}.json")
    with open(outpath, "w") as f:
        json.dump({"summary": {"total": n, "correct": total_correct, "accuracy": round(acc, 1),
                                "total_time_s": round(total_time, 1),
                                "per_category": dict(cat_stats),
                                "per_difficulty": dict(diff_stats)},
                    "questions": results}, f, indent=2, default=str)
    logger.info(f"Saved to {outpath}")
    print(f"\n  Results: {outpath}")

if __name__ == "__main__":
    main()
