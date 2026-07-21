#!/usr/bin/env python3
"""
Investigate why qwen2.5-math-1.5b scores 0.021 on math word problems.
Tests chat template, formats, raw outputs, and compares with smollm2-1.7b.
Saves report to /home/artem/dev/amd-hackathon/gepa_plans/math_model_investigation.md
"""

import json
import re
import os
import sys
import gc
import time
import textwrap

os.environ['PYTHONUNBUFFERED'] = '1'

# ── Paths ────────────────────────────────────────────────────────────────
QWEN_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
SMOLLM_PATH = "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf"
DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
REPORT_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/math_model_investigation.md"

# ── Data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    ALL_QUESTIONS = json.load(f)

# Pick 5 diverse questions for detailed testing
TEST_INDICES = [0, 3, 6, 10, 15]  # 14, 294, 36, 125, ...
TEST_QUESTIONS = [ALL_QUESTIONS[i] for i in TEST_INDICES]

# Also pick a very simple one for sanity check
SANITY_QUESTION = {"task_id": "sanity", "prompt": "What is 2+2?", "expected_answer": "4"}

print(f"Loaded {len(ALL_QUESTIONS)} questions. Testing {len(TEST_QUESTIONS)} detailed + 1 sanity.")

# ── fuzzy_match (same as eval) ───────────────────────────────────────────
def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False

# ── Report helpers ───────────────────────────────────────────────────────
report_lines = []

def R(text=""):
    report_lines.append(text)
    print(text)

def Rcode(label, code):
    R(f"\n### {label}")
    R("```")
    R(code)
    R("```")

def Rtable(rows, header=None):
    if header:
        R(f"| {' | '.join(header)} |")
        R(f"|{'|'.join('---' for _ in header)}|")
    for row in rows:
        R(f"| {' | '.join(str(c) for c in row)} |")

# ========================================================================
# SECTION 1: Chat Template Detection
# ========================================================================
R("\n# Math Model Investigation Report")
R(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
R("\n## 1. Chat Template Detection")
R("\nLoading qwen2.5-math-1.5b with verbose=True to see metadata...")

from llama_cpp import Llama

R("```")
R(f"Loading {QWEN_PATH}...")
sys.stdout.flush()
t0 = time.time()
llm_qwen = Llama(model_path=QWEN_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=True)
R(f"Loaded in {time.time()-t0:.1f}s")
R("```")

# Try to get metadata
R("\n### Model metadata dict:")
R("```")
metadata_keys = []
if hasattr(llm_qwen, 'metadata'):
    for k, v in llm_qwen.metadata.items():
        metadata_keys.append(f"{k}: {v[:200] if isinstance(v, str) and len(v) > 200 else v}")
        R(f"  {k}: {v[:200] if isinstance(v, str) and len(v) > 200 else v}")
else:
    R("  (no metadata attribute)")
R("```")

# Check for chat template
R("\n### Chat template check:")
R("```")
if hasattr(llm_qwen, 'chat_template'):
    R(f"chat_template = {llm_qwen.chat_template!r}")
elif hasattr(llm_qwen, '_chat_template'):
    R(f"_chat_template = {llm_qwen._chat_template!r}")
else:
    R("(no chat_template attribute found)")
R("```")

# Determine what template is being used
R("\n### Default chat handler:")
R("```")
# Test what tokenizer does with a test message
test_messages = [{"role": "user", "content": "Hello"}]
try:
    fmt = llm_qwen.tokenizer().apply_chat_template(test_messages, tokenize=False)
    R(f"Formatted: {fmt!r}")
except Exception as e:
    R(f"Error: {e}")
    # Try the old way
    try:
        fmt = llm_qwen._format_chat_template(test_messages)
        R(f"_format_chat_template: {fmt!r}")
    except Exception as e2:
        R(f"_format_chat_template error: {e2}")

# Check the actual format used by create_chat_completion
try:
    sample = llm_qwen.create_chat_completion(
        messages=[{"role": "user", "content": "Say exactly 'hello'"}],
        max_tokens=10, temperature=0.0
    )
    R(f"\ncreate_chat_completion response: {sample['choices'][0]['message']['content']!r}")
except Exception as e:
    R(f"create_chat_completion failed: {e}")
R("```")

# ========================================================================
# SECTION 2: Manual Inference - qwen2.5-math-1.5b
# ========================================================================
R("\n## 2. Detailed Inference: qwen2.5-math-1.5b on 5 Questions")

# Test 3 prompt strategies
strategies = [
    {"name": "a) Empty system prompt", "system": "", "user_only": False},
    {"name": "b) 'Answer only with a number.'", "system": "Answer only with a number.", "user_only": False},
    {"name": "c) No system, just user", "system": None, "user_only": True},
]

for q in TEST_QUESTIONS:
    R(f"\n--- Question: {q['task_id']} ---")
    R(f"Prompt: {q['prompt']}")
    R(f"Expected: {q['expected_answer']}")

    for strat in strategies:
        R(f"\n#### {strat['name']}")
        if strat['user_only']:
            messages = [{"role": "user", "content": q['prompt']}]
        else:
            messages = [{"role": "system", "content": strat['system']}, {"role": "user", "content": q['prompt']}]

        Rcode("Messages sent", json.dumps(messages, indent=2))

        t_start = time.time()
        try:
            resp = llm_qwen.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
            elapsed = time.time() - t_start
            got = resp["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"<ERROR: {e}>"
            elapsed = 0

        R(f"Raw output: {got!r}")
        R(f"Time: {elapsed:.2f}s")

        is_correct = fuzzy_match(got, q["expected_answer"])
        R(f"fuzzy_match('{got}', '{q['expected_answer']}') = {is_correct}")

        # Show what fuzzy_match extracts
        a_lower = got.strip().lower()
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a_lower)]
        R(f"Numbers extracted from output: {nums}")

    R("\n" + "-"*60)

# ========================================================================
# SECTION 3: Alternative Formats (raw completion)
# ========================================================================
R("\n## 3. Alternative Format Test (Raw Completion)")

R("\nTrying raw completion instead of chat format:")
for q in TEST_QUESTIONS[:3]:
    # Various raw prompt formats
    formats = [
        f"Question: {q['prompt']}\nAnswer:",
        f"Solve: {q['prompt']}\nAnswer: ",
        f"{q['prompt']}\nThe answer is",
    ]
    for prompt_str in formats:
        R(f"\n--- {q['task_id']} ---")
        Rcode("Raw prompt", prompt_str)
        t_start = time.time()
        try:
            resp = llm_qwen(prompt=prompt_str, max_tokens=64, temperature=0.0, stop=["\n", ".\n"])
            elapsed = time.time() - t_start
            got = resp["choices"][0]["text"]
        except Exception as e:
            got = f"<ERROR: {e}>"
            elapsed = 0
        R(f"Raw output: {got!r}")
        R(f"Time: {elapsed:.2f}s")
        is_correct = fuzzy_match(got, q["expected_answer"])
        R(f"fuzzy_match = {is_correct}")
    R("\n" + "-"*40)

# ========================================================================
# SECTION 4: Sanity check - can it even generate coherent math?
# ========================================================================
R("\n## 4. Tokenizer Sanity Check")
R("\n### 'What is 2+2?' - Chat format")

sanity_strategies = [
    {"name": "Empty system", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "What is 2+2?"}]},
    {"name": "No system", "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "Direct instruction", "messages": [{"role": "system", "content": "Answer only with a number."}, {"role": "user", "content": "What is 2+2?"}]},
]

for s in sanity_strategies:
    R(f"\n#### {s['name']}")
    Rcode("Messages", json.dumps(s["messages"]))
    try:
        resp = llm_qwen.create_chat_completion(messages=s["messages"], max_tokens=64, temperature=0.0)
        got = resp["choices"][0]["message"]["content"]
        R(f"Output: {got!r}")
        R(f"Expected: '4'")
        R(f"fuzzy_match: {fuzzy_match(got, '4')}")
    except Exception as e:
        R(f"Error: {e}")

R("\n### Raw completion - 'What is 2+2?'")
raw_prompts = [
    "What is 2+2?",
    "What is 2+2?\nAnswer:",
    "Q: What is 2+2?\nA:",
]
for p in raw_prompts:
    R(f"\nPrompt: {p!r}")
    try:
        resp = llm_qwen(prompt=p, max_tokens=64, temperature=0.0)
        got = resp["choices"][0]["text"]
        R(f"Output: {got!r}")
    except Exception as e:
        R(f"Error: {e}")

# ========================================================================
# Clean up qwen
R("\n\nCleaning up qwen model...")
del llm_qwen
gc.collect()
import time
time.sleep(2)

# ========================================================================
# SECTION 5: Compare with smollm2-1.7b on same 5 questions
# ========================================================================
R("\n# 5. Comparison: smollm2-1.7b on Same 5 Questions")

R(f"\nLoading {SMOLLM_PATH}...")
sys.stdout.flush()
t0 = time.time()
llm_smollm = Llama(model_path=SMOLLM_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
R(f"Loaded in {time.time()-t0:.1f}s")

for q in TEST_QUESTIONS:
    R(f"\n--- {q['task_id']} ---")
    R(f"Prompt: {q['prompt']}")
    R(f"Expected: {q['expected_answer']}")

    for strat in strategies:
        R(f"\n#### {strat['name']}")
        if strat['user_only']:
            messages = [{"role": "user", "content": q['prompt']}]
        else:
            messages = [{"role": "system", "content": strat['system']}, {"role": "user", "content": q['prompt']}]

        t_start = time.time()
        try:
            resp = llm_smollm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
            elapsed = time.time() - t_start
            got = resp["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"<ERROR: {e}>"
            elapsed = 0

        R(f"Raw output: {got!r}")
        R(f"Time: {elapsed:.2f}s")
        is_correct = fuzzy_match(got, q["expected_answer"])
        R(f"fuzzy_match = {is_correct}")

    R("\n" + "-"*60)

# ========================================================================
# SECTION 6: Side-by-side comparison table
# ========================================================================
R("\n# 6. Side-by-Side Comparison Table")

# Re-run both models on the same questions with same format to build table
R("\nUsing format: [system='', user=question] (the 'empty' prompt)")

llm_qwen2 = Llama(model_path=QWEN_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
llm_smollm2 = Llama(model_path=SMOLLM_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)

Rtable(
    ["Question", "Expected", "Qwen Output", "Qwen Correct?", "SmolLM2 Output", "SmolLM2 Correct?"],
    header=["Question", "Expected", "Qwen Output", "Qwen ✓?", "SmolLM2 Output", "SmolLM2 ✓?"]
)

for q in ALL_QUESTIONS:
    msg = [{"role": "system", "content": ""}, {"role": "user", "content": q["prompt"]}]
    
    # Qwen
    r1 = llm_qwen2.create_chat_completion(messages=msg, max_tokens=64, temperature=0.0)
    o1 = r1["choices"][0]["message"]["content"]
    c1 = fuzzy_match(o1, q["expected_answer"])
    
    # SmolLM2
    r2 = llm_smollm2.create_chat_completion(messages=msg, max_tokens=64, temperature=0.0)
    o2 = r2["choices"][0]["message"]["content"]
    c2 = fuzzy_match(o2, q["expected_answer"])
    
    # Truncate outputs for table
    o1_short = o1[:60].replace('\n', '\\n')
    o2_short = o2[:60].replace('\n', '\\n')
    
    R(f"| {q['task_id'][:25]} | {q['expected_answer']} | {o1_short} | {'✓' if c1 else '✗'} | {o2_short} | {'✓' if c2 else '✗'} |")

del llm_qwen2
del llm_smollm2
gc.collect()

# ========================================================================
# SECTION 7: Analysis and Recommendations
# ========================================================================
R("\n# 7. Analysis & Recommendations")
R("\n[To be filled after examining results]")

# ========================================================================
# Write report
# ========================================================================
R("\n\n---")
R("\nGPU confirmed: n_gpu_layers=-1 was used")

report = "\n".join(report_lines)
with open(REPORT_PATH, "w") as f:
    f.write(report)
print(f"\nReport saved to {REPORT_PATH}")
