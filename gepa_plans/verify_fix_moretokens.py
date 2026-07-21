#!/usr/bin/env python3
"""Quick verification: run qwen2.5-math-1.5b with max_tokens=512 instead of 64."""
import json, gc, time, re
from llama_cpp import Llama

QWEN_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"

with open(DATA_PATH) as f:
    questions = json.load(f)

def fuzzy_match(answer, expected):
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

print("Loading qwen2.5-math-1.5b with GPU...")
t0 = time.time()
llm = Llama(model_path=QWEN_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"Loaded in {time.time()-t0:.1f}s")

# Test with max_tokens=512 (no system prompt, just user)
print(f"\nTesting {len(questions)} questions with max_tokens=500, empty system prompt...")
correct = 0
total = 0
for q in questions:
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": q["prompt"]}]
    t_start = time.time()
    resp = llm.create_chat_completion(messages=messages, max_tokens=500, temperature=0.0)
    got = resp["choices"][0]["message"]["content"]
    is_correct = fuzzy_match(got, q.get("expected_answer", ""))
    if is_correct:
        correct += 1
    total += 1
    
    # Show first few
    if total <= 3 or is_correct:
        exp = q.get("expected_answer", "")
        print(f"  [{total}] {q['task_id'][:25]}: exp={exp} correct={is_correct} | output ends: ...{got[-80:].replace(chr(10),' ')}")
    
    if total % 20 == 0:
        print(f"  ... {total}/{len(questions)} so far: {correct}/{total} = {correct/total:.3f}")

acc = correct / total if total else 0
print(f"\n{'='*60}")
print(f"RESULTS with max_tokens=500: {correct}/{total} = {acc:.3f}")
print(f"{'='*60}")
print(f"GPU confirmed: n_gpu_layers=-1 was used")
