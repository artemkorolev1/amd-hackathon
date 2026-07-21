# Long-Form Answer Stress Test — 20 Questions

**File:** `eval_longform_20.json`  
**Purpose:** Verify the container handles long answers (>250 tokens) without truncation, timeout, or corruption.

## The Problem

The current `dynamic_prompts.py` caps `max_tokens` per category:

| Category | max_tokens | +30% high-complexity | ×0.75 CPU factor | Effective ceiling |
|----------|-----------|---------------------|-------------------|------------------|
| code_gen | 250 | 325 | 243 | **~243 tokens** |
| code_debug | 200 | 260 | 195 | **~195 tokens** |
| math | 200 | 260 | 195 | **~195 tokens** |
| logic | 200 | 260 | 195 | **~195 tokens** |
| factual | 120 | 156 | 117 | **~117 tokens** |
| sentiment | 60 | 78 | 58 | **~58 tokens** |
| ner | 120 | 156 | 117 | **~117 tokens** |
| summarization | 120 | 156 | 117 | **~117 tokens** |
| general (default) | 300 | 390 | 292 | **~292 tokens** |

If any of these ceilings is too low for the grader's expected answer, the model gets cut off mid-sentence and the output either:
1. Fails `fuzzy_match()` substring/token-overlap checks
2. Wastes a Fireworks API fallback call (counted tokens)
3. Returns empty (if truncated to just preamble)

## Test Design

**20 questions** with the following breakdown:

### Long-form (>250 tokens expected)
- `long_code_01` — LRU Cache (code_gen, ~300-500 tok)
- `long_code_02` — Trie data structure (code_gen, ~400-600 tok)
- `long_code_03` — Rate Limiter (code_gen, ~350-500 tok)
- `long_factual_01` — HTTPS handshake (factual, ~400-600 tok)
- `long_factual_02` — RDBMS architecture (general, ~500-800 tok)
- `long_logic_01` — Einstein's puzzle (logic, ~300-600 tok)
- `long_logic_02` — 4-friend puzzle (logic, ~200-400 tok)
- `long_code_debug_01` — BST bug fix (code_debug, ~400-600 tok)
- `long_summarization_01` — AI safety essay (summarization, ~200-400 tok)
- `long_math_01` — Pool volume (math, ~300-500 tok)
- `long_math_02` — Compound interest (math, ~250-400 tok)
- `long_general_01` — Map-reduce model (general, ~600-1000 tok)
- `medium_code_gen_01` — Merge sorted lists (code_gen, ~150-300 tok — borderline)
- `medium_factual_01` — Greenhouse effect (factual, ~100-200 tok — borderline)

### Control questions (<100 tokens expected)
- `control_code_debug_01` — find_max bug fix (very short, ~20 tok)
- `control_math_01` — 15% of 240 (numeric, <10 tok)
- `control_sentiment_01` — Mixed sentiment (single word)
- `control_ner_01` — Entity extraction (moderate, ~80-150 tok)
- `medium_summarization_01` — Short summary (~30-60 tok)
- `medium_logic_01` — Syllogism (single option letter)

## How to Run

### Option 1: Through the container (recommended)

```bash
# Copy test file to a mountable location
cp /home/artem/dev/amd-hackathon-v12h/eval_longform_20.json /tmp/eval_longform_20.json

# Run through container with grader constraints
docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -v /tmp/eval_longform_20.json:/input/tasks.json:ro \
  -v /tmp/longform_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> 2>&1

# Check results
cat /tmp/longform_out/results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data:
    print(f\"{item['task_id']}: {len(item['answer'].split())} words | {len(item['answer'])} chars | {repr(item['answer'][:80])}...\")
"
```

### Option 2: Directly through harness.py (for debugging)

```bash
cd /home/artem/dev/amd-hackathon-v12h
python3 harness.py /path/to/eval_longform_20.json 2>/dev/null
```

This prints one answer per line in the container's output format.

## What to Check

### 1. Truncation detection
For every long-form answer, check if it ends naturally or mid-sentence:

```bash
cat /tmp/longform_out/results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data:
    answer = item['answer']
    ends_natural = answer.endswith(('.', '```', '}', ']', ')', '\"'))
    print(f\"{item['task_id']}: {'OK' if ends_natural else 'TRUNCATED?'} ({len(answer.split())} words)\")
"
```

### 2. Token count per answer
The key metric — how many tokens did the model actually produce before hitting max_tokens or timeout?

Look for:
- Answers that are exactly the category's max_tokens ceiling (means the model hit the limit)
- Answers that are much shorter than expected (model timed out or crashed)
- Empty answers (fireworks fallback or error)

### 3. Per-question timing
Check stderr for timestamps between tasks. Long-generation tasks should not exceed 28-30s. If they do, the container may timeout on the grader.

### 4. Fireworks fallback activity
If a local model hits max_tokens and the residual answer is empty/broken, the pipeline falls back to Fireworks. Count how many Fireworks calls happen — each one costs tokens on the leaderboard.

## What the Results Tell You

| Pattern | Meaning | Fix |
|---------|---------|-----|
| Most answers < 50% of expected length | model is VERY slow or max_tokens is very conservative | Increase max_tokens, check CPU token generation rate |
| Long answers end mid-sentence (`was goi...`) | max_tokens ceiling hit during generation | Increase per-category MAX_TOKENS |
| Long answers end at exactly ` ``` ` or `Answer:` marker | Model finished naturally but within ceiling | Ceiling is OK for that category |
| Long factual/summary answers < 80 tokens | Model hit timeout (not max_tokens) | Need faster model or reduce per-task complexity |
| Some short answers but correct | Pipeline working fine | No fix needed for those categories |
| Empty answers for long questions | Timeout or Fireworks fallback also failed | Increase timeout or fix model loading |
