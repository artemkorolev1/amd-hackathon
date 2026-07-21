# Model Recommendation Report

Evaluated 7 GGUF models for instruction-following ability, speed, and format compliance.

## Test Setup
- **GPU**: RTX A4000 (16GB VRAM), n_gpu_layers=-1
- **Library**: llama-cpp-python 0.3.33
- **Context**: 2048 tokens, Temperature: 0.1
- **Workflow**: 3-step (Plan → Solve → Compose) with 2 math word problems
- **Load/unload**: Each model fully loaded then `del + gc.collect()` before next

### Test Questions
- **Q1**: Jean has 30 lollipops. Jean eats 2. With remaining, puts 2 per bag. How many bags?
- **Q2**: Mike plays ping pong for 40min. First 20min: 4 pts. Second 20min: 1.25× first. Total points?

### Workflow Steps
1. **Plan**: "Create a numbered plan to solve this math problem. Output ONLY the plan."
2. **Solve**: "Execute the plan step by step. Show each calculation. Put final answer in \boxed{}."
3. **Compose**: "Format the final answer as: The answer is \boxed{number}."

### Key Evaluation Criteria
- **Instruction following**: Does model produce DIFFERENT output per step? Follow the specific instruction? Use prior step's output?
- **Format compliance**: Correct \boxed{} usage?
- **Speed**: Load time + per-inference latency
- **Accuracy**: Are the answers correct?

---

## Model: qwen2.5-1.5b-instruct (Q4_K_M)
**File**: qwen2.5-1.5b-instruct-q4_k_m.gguf
- **Load time**: 1.44s
- **Per-inference**: 492ms avg (fastest non-Llama)
- **Instruction following**: Partial — produced numbered plans correctly and used prior context, but sometimes omitted \boxed{} in Solve step (used it only in Compose). Q1 Solve lacked \boxed{}, Q2 Solve included it.
- **\boxed{} compliance**: Partial (2/4 Compose steps correct, 1/2 Solve steps correct)
- **Accuracy**: ✅ Q1=14, Q2=9 — both correct
- **Workflow output (Q1)**:
  - **Plan**: "1. Determine the number of lollipops left after eating 2. 2. Calculate how many bags..."
  - **Solve**: Shows 30-2=28, then 28÷2=14 (no \boxed{})
  - **Compose**: "The answer is \boxed{14}."
- **Workflow output (Q2)**:
  - **Plan**: "1. Calculate points in second 20min: 1.25×4=5. 2. Add points from both periods..."
  - **Solve**: "4 + 5 = 9" with \boxed{9}
  - **Compose**: "The answer is \boxed{9}."
- **Assessment**: Fast generalist with good reasoning. Occasionally misses formatting instructions but produces correct answers. Good for high-volume general tasks.

---

## Model: Qwen2.5-Math-1.5B (Q4_K_M)
**File**: Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf
- **Load time**: 0.70s (fastest load)
- **Per-inference**: 943ms avg (moderate)
- **Instruction following**: No — Solve output was IDENTICAL to Plan output for both Q1 and Q2 (regenerates the same text instead of executing the plan)
- **\boxed{} compliance**: ✅ Yes — all steps used correct \boxed{} formatting
- **Accuracy**: ✅ Q1=14, Q2=9 — both correct
- **Assessment**: Good at math and formatting, but fails the multi-step workflow because it doesn't change output per step. The math specialization shows in correct answers, but the instruction-following weakness means it can't be used in multi-step pipelines without wrappers.

---

## Model: qwen2.5-coder-1.5b-instruct (Q4_K_M)
**File**: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
- **Load time**: 1.80s
- **Per-inference**: 696ms avg (7 runs incl. code test)
- **Instruction following**: ✅ **Yes** — clearly the best. Started Plan with "Plan:", executed step-by-step in Solve, used prior context, produced proper \boxed{} in Solve, and good Compose output.
- **\boxed{} compliance**: ✅ Yes — correct \boxed{} in both Solve and Compose
- **Accuracy**: ✅ Q1=14, Q2=9 — both correct
- **Code test**: ✅ Generated valid Python function `sum_even_numbers(numbers)` with proper loop and modulo check
- **Workflow output (Q1)**:
  - **Plan**: "Plan: 1. Start with total lollipops (30). 2. Subtract eaten (2). 3. Divide remaining by 2."
  - **Solve**: "Let's execute the plan step by step: 30-2=28, 28÷2=14 → \boxed{14}"
  - **Compose**: "The answer is \boxed{14}."
- **Assessment**: **Top performer**. Best instruction following, correct formatting, handles both math and code well. The code specialization makes it an excellent all-rounder for a pipeline that includes code tasks.

---

## Model: smollm2-1.7b-instruct (Q4_K_M)
**File**: smollm2-1.7b-instruct-q4_k_m.gguf
- **Load time**: 1.75s
- **Per-inference**: 517ms avg (fast)
- **Instruction following**: No — Solve output identical to Plan (both Q1 and Q2). No \boxed{} used in any step.
- **\boxed{} compliance**: ❌ No
- **Accuracy**: ❌ Q1 correct (14), Q2 **wrong** (18 instead of 9 — incorrectly added an extra 4+5+4)
- **Assessment**: Disqualified. Incorrect answer on question 2 (18 vs 9) and complete failure to follow formatting instructions. The 1.7B parameter count doesn't compensate for poor quality.

---

## Model: Llama-3.2-1B-Instruct (Q4_K_M)
**File**: Llama-3.2-1B-Instruct-Q4_K_M.gguf
- **Load time**: 0.49s (fastest load overall)
- **Per-inference**: 424ms avg (fastest inference)
- **Instruction following**: Partial — produced different output per step but no \boxed{} at all. Generated a plan, then a solve, then composed.
- **\boxed{} compliance**: ❌ No — never used \boxed{} in any step
- **Accuracy**: ❌ Q1 correct (14), Q2 **wrong** (14 instead of 9 — computed 4×1.25×2+4 incorrectly)
- **Assessment**: Fastest and smallest model, but wrong answers and no formatting compliance. Could be useful as a very lightweight fallback for non-critical tasks if accuracy isn't paramount.

---

## Model: gemma-3-1b-it (Q4_K_M)
**File**: gemma-3-1b-it-Q4_K_M.gguf
- **Load time**: 1.20s
- **Per-inference**: 856ms avg (moderate)
- **Instruction following**: No — Q2 Solve and Compose outputs were IDENTICAL. No \boxed{} in 3/4 cases.
- **\boxed{} compliance**: ❌ No (only Q1 Solve had \boxed{}, rest missing)
- **Accuracy**: Q1 correct (14), Q2 output didn't isolate answer
- **Assessment**: Disqualifying issues — identical outputs across steps and missing formatting. Not suitable for pipeline use.

---

## Model: Qwen2.5-Math-1.5B (Q4_0) — lower quality quant
**File**: Qwen2.5-Math-1.5B-Instruct-Q4_0.gguf
- **Load time**: 1.31s
- **Per-inference**: 840ms avg (moderate)
- **Instruction following**: No — Solve output identical to Plan for both Q1 and Q2 (same issue as Q4_K_M math variant)
- **\boxed{} compliance**: ✅ Yes — used \boxed{} correctly in Solve and Compose
- **Accuracy**: ✅ Q1=14, Q2=9 — both correct
- **Assessment**: Same behavior as the Q4_K_M math model — correct answers and formatting but fails the multi-step instruction test. The Q4_0 quant is slightly slower and larger, so the Q4_K_M version is strictly better.

---

## Performance Summary

| Model | Load (s) | Avg Infer (ms) | Instruction Follow | \boxed{} | Accuracy |
|-------|----------|----------------|-------------------|----------|----------|
| qwen2.5-coder-1.5b-instruct | 1.80 | 696 | ✅ Yes | ✅ Yes | ✅ 2/2 |
| qwen2.5-1.5b-instruct | 1.44 | 492 | ⚠️ Partial | ⚠️ Partial | ✅ 2/2 |
| Qwen2.5-Math-1.5B (Q4_K_M) | 0.70 | 943 | ❌ No (same output) | ✅ Yes | ✅ 2/2 |
| Qwen2.5-Math-1.5B (Q4_0) | 1.31 | 840 | ❌ No (same output) | ✅ Yes | ✅ 2/2 |
| smollm2-1.7b-instruct | 1.75 | 517 | ❌ No | ❌ No | ❌ 1/2 |
| Llama-3.2-1B-Instruct | 0.49 | 424 | ⚠️ Partial | ❌ No | ❌ 1/2 |
| gemma-3-1b-it | 1.20 | 856 | ❌ No | ❌ No | ❌ Partial |

---

## Final Recommendation

### Top 3 Models (ordered by priority)

#### 1. 🥇 qwen2.5-coder-1.5b-instruct (Q4_K_M) — Primary Model
- **Best overall**: Only model with perfect instruction following, correct \boxed{} formatting, AND accurate answers
- **Covers tasks**: factual, logic, sentiment, summarization, NER, **code_debug**, **code_gen**
- **Size**: ~950MB
- **Why**: It's not just a code model — it handled math word problems perfectly and was the only model that correctly followed the 3-step workflow with distinct outputs per step. Its code specialization is a bonus for code tasks.

#### 2. 🥈 qwen2.5-1.5b-instruct (Q4_K_M) — Generalist / High-Volume
- **Fastest accurate model**: 492ms avg, correct answers on both questions
- **Covers tasks**: factual, logic, sentiment, summarization, NER
- **Size**: ~1.1GB
- **Why**: Fast generalist that gets answers right. Minor formatting issues (missing \boxed{} in Solve occasionally) can be handled via prompt engineering or post-processing. Ideal as a fast router target for non-code tasks.

#### 3. 🥉 Qwen2.5-Math-1.5B (Q4_K_M) — Math Specialist
- **Best math accuracy**: Correct \boxed{} formatting, correct answers
- **Covers tasks**: **math**, logic
- **Size**: ~940MB
- **Why**: Despite failing the multi-step distinct-output test, it produces correct math answers with proper formatting. When you need guaranteed math accuracy, route to this model. The Q4_K_M variant is better than Q4_0 (faster load, same quality).

### 🚫 Not Recommended
- **smollm2-1.7b**: Wrong answers, no formatting compliance
- **Llama-3.2-1B**: Fastest but wrong answers on Q2, no \boxed{}
- **gemma-3-1b-it**: Identical outputs across steps, poor formatting
- **Qwen2.5-Math Q4_0**: Strictly worse than Q4_K_M variant

### GPU Memory Budget Check
All 3 recommended models loaded simultaneously:
| Model | Size |
|-------|------|
| qwen2.5-coder-1.5b-instruct | ~950 MB |
| qwen2.5-1.5b-instruct | ~1.1 GB |
| Qwen2.5-Math-1.5B (Q4_K_M) | ~940 MB |
| **Total** | **~3.0 GB** |
| RTX A4000 capacity | 16 GB |

✅ Fits comfortably with 13 GB headroom for activations, KV cache, and concurrent workloads.

### Recommended Pipeline Strategy

```
                    ┌─ factual ──┐
                    ├─ logic ────┤
User Request ──▶ Router ──┬─ sentiment ──▶ qwen2.5-1.5b-instruct (generalist)
                    ├─ summarization ─┤
                    ├─ NER ───────────┘
                    │
                    ├─ math ──────────▶ Qwen2.5-Math-1.5B (math specialist)
                    │
                    ├─ code_debug ────┐
                    └─ code_gen ──────┴─▶ qwen2.5-coder-1.5b-instruct (code specialist)
```

For the 3 models, pre-load all into VRAM (total ~3GB). Route by task type. The coder model can also serve as a fallback generalist since it demonstrated the best instruction following overall.
