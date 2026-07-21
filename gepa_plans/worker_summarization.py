#!/usr/bin/env python3
"""Worker: runs a single model on summarization prompts. Uses summarization-specific grading (keyword overlap + entity recall) instead of strict fuzzy_match."""
import sys, json, os, time, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_common import fuzzy_match

os.environ["PYTHONUNBUFFERED"] = "1"

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/summarization_combined_25.json"
RESULTS_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/summarization_worker_results.json"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else ""

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty", "system_prompt": ""},
    {"index": 1, "label": "label_prefix", "system_prompt": "Summarize:"},
    {"index": 2, "label": "explicit_instruction",
     "system_prompt": "Summarize the text in at most 2 sentences. Include key names, numbers, and facts."},
    {"index": 3, "label": "verbose_instruction",
     "system_prompt": "Read the following text carefully and produce a concise summary. Capture the main point, key entities, and any numerical data. Output 1-3 sentences maximum. Do not add opinions or commentary."},
]

def summarization_grade(output: str, expected: str) -> bool:
    """Grade summarization: check entity/keyword overlap in expected appears in output.
    Uses 3 signals: fuzzy_match cascade, entity overlap, noun-phrase overlap."""
    # 1. Standard fuzzy_match cascade (catches near-exact)
    if fuzzy_match(output, expected):
        return True

    # 2. Extract capitalized entities (names, orgs, places)
    def extract_entities(text):
        return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    exp_entities = extract_entities(expected)
    out_entities = extract_entities(output)
    if len(exp_entities) > 0:
        overlap = exp_entities & out_entities
        recall = len(overlap) / len(exp_entities)
        # Entity recall >= 50% or at least 2 entities match
        if recall >= 0.5 or len(overlap) >= 2:
            return True

    # 3. Keyword overlap: significant shared content words
    exp_words = set(re.findall(r'[a-zA-Z]{4,}', expected.lower()))
    out_words = set(re.findall(r'[a-zA-Z]{4,}', output.lower()))
    if len(exp_words) > 0:
        word_overlap = len(exp_words & out_words) / len(exp_words)
        if word_overlap >= 0.4:
            return True

    # 4. Extract numbers and check overlap
    exp_nums = set(re.findall(r'\d+(?:\.\d+)?', expected))
    out_nums = set(re.findall(r'\d+(?:\.\d+)?', output))
    if exp_nums and exp_nums & out_nums:
        return True

    return False

with open(DATA_PATH) as f:
    questions = json.load(f)

from llama_cpp import Llama

t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False, seed=42)
load_time = time.time() - t0

results = {}
for strat in PROMPT_STRATEGIES:
    idx = strat["index"]
    sys_prompt = strat["system_prompt"]
    correct = 0
    total = 0
    total_latency = 0.0
    per_q = []

    for q in questions:
        prompt = q["prompt"]
        expected = q.get("expected_answer", q.get("answer", ""))
        messages = [{"role": "user", "content": prompt}]
        if sys_prompt:
            messages.insert(0, {"role": "system", "content": sys_prompt})

        t1 = time.time()
        try:
            resp = llm.create_chat_completion(
                messages=messages, max_tokens=96, temperature=0.0,
            )
            output = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            output = ""
        latency = (time.time() - t1) * 1000

        ok = summarization_grade(output, expected)
        if ok:
            correct += 1
        total += 1
        total_latency += latency
        per_q.append({"expected": expected[:60], "got": output[:60], "ok": ok})

    avg_latency = total_latency / total if total else 0
    results[f"prompt_{idx}"] = {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "avg_latency_ms": round(avg_latency, 1),
        "label": strat["label"],
    }
    print(f"  [{strat['label']:25s}] acc={results[f'prompt_{idx}']['accuracy']:.4f} ({correct}/{total}) lat={avg_latency:.1f}ms")

output = {"load_time_s": round(load_time, 2), "results": results}
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"  [save] {RESULTS_PATH}")
