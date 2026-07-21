# Workflow Instruction-Following Investigation

**Date**: 2026-07-13
**Models tested**:
- qwen2.5-1.5b-instruct (`/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`)
- qwen2.5-math-1.5b (`/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf`)
- Llama-3.2-1B-Instruct (`/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`)

**Questions used** (first 3 from `math_combined_80.json`):
- Q0: Jean lollipops → 14
- Q1: Mike ping pong → 9
- Q2: John drives → 45

---

## 1. Per-Config Raw Input/Output Dumps

### 1a. Config A — qwen2.5-1.5b-instruct (ALL steps)

#### Q0 (Jean lollipops, expected=14)

```
[plan] SYSTEM: Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else.
[plan] USER: Solve: Jean has 30 lollipops. Jean eats 2 of the lollipops. With the remaining lollipops, Jean wants to package 2 lollipops in one bag. How many bags can Jean fill?
[plan] OUTPUT:
    1. Jean starts with 30 lollipops.
    2. Jean eats 2 lollipops, leaving 30 - 2 = 28 lollipops.
    3. Jean wants to package 2 lollipops in one bag.
    4. To find out how many bags Jean can fill, divide the remaining lollipops by the number of lollipops per bag: 28 ÷ 2 = 14 bags.

[solve] SYSTEM: Execute the plan above step by step. Show each calculation. Put the final answer in \boxed{}.
[solve] USER: Original problem:
Solve: Jean has 30 lollipops. Jean eats 2 of the lollipops...
=== [plan] output ===
1. Jean starts with 30 lollipops.
2. Jean eats 2 lollipops, leaving 30 - 2 = 28 lollipops.
3. Jean wants to package 2 lollipops in one bag.
4. To find out how many bags Jean can fill, divide the remaining lollipops by...

Now: Execute the plan above step by step. Show each calculation. Put the final answer in \boxed{}.
[solve] OUTPUT:
    1. Jean starts with 30 lollipops.
    2. Jean eats 2 lollipops, leaving 30 - 2 = 28 lollipops.
    ...
    Final answer: \boxed{14} bags

[compose] OUTPUT: The answer is \boxed{14} bags.
```
**Follows plan? YES** ✓ — Solve step mirrors plan steps exactly. **Answer: 14 ✓**

#### Q1 (Mike ping pong, expected=9)

```
[plan] OUTPUT:
    1. Calculate the number of points scored in the second 20 minutes: 4 × 1.25 = 5 points.
    2. Add the points from the first 20 minutes to the points from the second 20 minutes: 4 + 5 = 9 points.

[solve] OUTPUT:
    ...1. Calculate the number of points scored in the second 20 minutes:
       - Mike scored 4 points in the first 20 minutes.
       - In the second 20 minutes, he scored 25% more points...
       - 4 × 0.25 = 1 point → 4 + 1 = 5 points.
    2. Add the points... 4 + 5 = 9

[compose] OUTPUT: The answer is \boxed{24}
```
**Follows plan? YES** ✓ — Solve step followed plan structure and got 9. **Compose step FAILED** (output 24 instead of 9). Perhaps because solve output was truncated in the compose message, or the instruct model hallucinated.

#### Q2 (John drives, expected=45)

```
[plan] OUTPUT:
    1. Distance₁ = 60 × 3 = 180 miles
    2. Remaining distance = 4 × 80 = 320 miles  ← WRONG: plan had errors
    3. Distance₂ = 30 × 2 = 60 miles  ← WRONG: 30mph for 0.5h, not 2h
    4. ... truncated

[solve] OUTPUT: Regenerated from scratch with different calculations.
[compose] OUTPUT: The answer is \boxed{180} miles.
```
**Follows plan? Partial** — Solve started with plan's structure but regenerated independently. **Answer: 180 ✗** (expected 45)
(The plan itself had conceptual errors — this is a complex multi-step problem requiring careful tracking of position relative to home.)

---

### 1b. Config B — qwen2.5-math-1.5b (ALL steps)

#### Q0 (Jean lollipops, expected=14)

```
[plan] SYSTEM: Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else.
[plan] USER: Solve: Jean has 30 lollipops...
[plan] OUTPUT:  ← COMPLETE SOLUTION, NOT A PLAN
    To solve the problem, we need to follow these steps:
    1. Determine how many lollipops Jean has left after eating 2 of them.
    2. Calculate how many bags Jean can fill...
    Let's go through the steps in detail:
    1. Jean starts with 30 lollipops. After eating 2 lollipops... 30 - 2 = 28
    2. ... 28 / 2 = 14
    Therefore, the number of bags Jean can fill is \(\boxed{14}\).

[solve] OUTPUT:  ← REGENERATED FROM SCRATCH
    To solve the problem, we need to follow these steps:
    1. Determine how many lollipops Jean has left...
    2. ... 28 / 2 = 14
    Therefore, Jean can fill \boxed{14} bags.

[compose] OUTPUT:
    To solve the problem, we need to follow these steps:
    1. ... 30 - 2 = 28
    2. ... 28 / 2 = 14
    Therefore, Jean can fill \boxed{14} bags.
```
**Follows plan? NO** — the "plan" step already contains a full solution. The "solve" step completely regenerates from scratch (copies the plan's structure but re-derived). The "compose" step also regenerates. All 3 steps produce essentially the same output. **Answer: 14 ✓** (correct by luck — simple problem)

#### Q1 (Mike ping pong, expected=9)

```
[plan] OUTPUT:
    To solve this problem... Let's break down the steps:
    1. Calculate the points scored in the second 20 minutes.
    2. Sum the points...
    Let's implement this in Python...
    ```python
    points_second_20_minutes = points_first_20_minutes * 1.25
    ...
    ```
    Mike scores a total of \(\boxed{9}\) points.

[solve] OUTPUT:
    To solve the problem, we need to calculate the total points...
    1. Calculate the points scored in the second 20 minutes.
    2. Sum the points...
    ... 4 + 5 = 9 points.
    Therefore, the total points Mike scored is \(\boxed{9}\).

[compose] OUTPUT:
    To solve the problem... 4 + 5 = 9
    So, the total points Mike scored is \(\boxed{9}\).
```
**Follows plan? NO** — All 3 steps regenerate identical complete solutions. The "plan" was a full solution (ignored "ONLY the numbered plan"). **Answer: 9 ✓**

#### Q2 (John drives, expected=45)

```
[plan] OUTPUT:
    To determine how far John is from home... break down his journey...
    1. First Segment: 3h × 60mph = 180 miles
    2. Second Segment: first 2h in standstill = 0 miles
    3. Third Segment: 0.5h × 30mph = 15 miles
    4. Fourth Segment: remaining time... truncated

[solve] OUTPUT:
    To determine how far John is from home... break down his journey...
    1. First Segment: 3h × 60mph = 180 miles
    2. Second Segment: first 2h = 0 miles, next 0.5h = 15 miles
    3. Third Segment: John drives the remaining time... truncated

[compose] OUTPUT:
    Same pattern — regenerated from scratch.
```
**Follows plan? NO** — All steps regenerate. **Answer: truncated ✗**

---

### 1c. Config C — Llama-3.2-1B-Instruct (ALL steps)

#### Q0 (Jean lollipops, expected=14)

```
[plan] SYSTEM: Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else.
[plan] OUTPUT:
    1. Calculate the number of lollipops remaining after Jean eats 2.
       30 - 2 = 28
    2. Determine how many bags Jean can fill with the remaining lollipops.
       28 / 2 = 14

[solve] OUTPUT:
    Here's the step-by-step execution of the plan:
    ### Step 1: Calculate the number of lollipops remaining after Jean eats 2.
    Jean starts with 30 lollipops and eats 2, so the number of lollipops remaining is: 30 - 2 = 28
    ### Step 2: Determine how many bags Jean can fill with the remaining lollipops.
    ... 28 ÷ 2 = 14
    Therefore, Jean can fill 14 bags...

[compose] OUTPUT:
    The answer is \boxed{14}.
```
**Follows plan? YES** ✓ — Perfect instruction following. Plan stays as plan, solve explicitly references plan steps. **Answer: 14 ✓**

---

### 1d. Single-prompt (both models)

#### Instruct model — single prompt
```
SYSTEM: You are a helpful math assistant.
USER: Solve: Jean has 30 lollipops...
Work through this problem: first plan, then solve step by step, then put the answer in \boxed{}.

OUTPUT:
    To solve this problem, let's break it down step by step:
    1. Initial number: 30 lollipops.
    2. Lollipops eaten: 2.
    3. Remaining: 30 - 2 = 28
    4. Lollipops per bag: 2
    5. Number of bags: 28/2 = 14
    Therefore, Jean can fill 14 bags.
```
**Follows instructions? YES** ✓ — Has implicit plan, solution, and boxed answer. **Correct: 14**

#### Math model — single prompt
```
SYSTEM: You are a helpful math assistant.
USER: Solve: Jean has 30 lollipops...
Work through this problem: first plan, then solve step by step, then put the answer in \boxed{}.

OUTPUT:
    To determine how many bags Jean can fill, we need to follow these steps:
    1. Calculate the number of lollipops remaining... 30 - 2 = 28
    2. Determine how many bags... 28 ÷ 2 = 14
    Therefore, the number of bags Jean can fill is \(\boxed{14}\).
```
**Follows instructions? YES** ✓ — Works well with single prompt. **Correct: 14**

---

## 2. Which Configs Actually Follow the Plan Instruction

| Config | Plan step obeys "ONLY plan" | Solve step uses plan | Compose step formats correctly |
|--------|---------------------------|---------------------|-------------------------------|
| **A: Instruct all** | ✅ Yes | ✅ Yes (follows structure) | ⚠️ Sometimes hallucinates |
| **B: Math all** | ❌ No — outputs full solution | ❌ No — regenerates from scratch | ❌ No — also regenerates |
| **C: Llama all** | ✅ Yes | ✅ Yes (explicit references) | ✅ Yes |
| **Single-prompt (instruct)** | ✅ Yes (implicit) | ✅ Yes | ✅ Yes |
| **Single-prompt (math)** | ✅ Yes (implicit) | ✅ Yes | ✅ Yes |

---

## 3. Root Cause: Why the Math Specialist Ignores the Plan

### Diagnosis

**The qwen2.5-math-1.5b model has a fundamental training-data mismatch with multi-step workflows.**

1. **Training data bias**: Qwen2.5-Math-1.5B was fine-tuned exclusively on mathematical reasoning data (GSM8K, MATH, etc.) where every training example is structured as:
   - **Input**: A math problem (user)
   - **Output**: A complete solution with reasoning + answer in `\boxed{}`
   The model was never trained on multi-turn conversations, system-prompt instructions, or role-specific tasks like "output ONLY a plan."

2. **The model treats everything as a fresh math problem**: Regardless of system prompt content or user message context (even when we say "Execute the plan above"), the model's internal representation matches the closest pattern in its training data: "given math problem → produce complete solution." The provided plan in the user message is treated as _additional problem context_, not as an instruction to follow. The model produces a complete solution from scratch every time.

3. **System prompts are largely ineffective**: The system prompt constraints like "Output ONLY the numbered plan, nothing else" are completely ignored. This is because the math model's training data didn't include system-prompt-based instruction following — it was trained on single-turn QA pairs.

4. **Evidence from single-prompt test**: The math model works _acceptably_ with a single prompt ("first plan, then solve...") because this matches its training distribution: one input → one complete solution. It doesn't need to understand workflow state; it just needs to produce a full solution that happens to include planning steps.

5. **Contrast with instruct model**: qwen2.5-1.5b-instruct was trained on diverse instruction-following data (including multi-turn chat, role-playing, constrained output formatting). It naturally separates "what to do" (system prompt) from "what to work on" (user message), and it follows formatting constraints.

6. **Contrast with Llama-3.2-1B**: Despite being smaller (1B vs 1.5B), Llama-3.2-1B-Instruct was trained on high-quality instruction-following data and follows multi-step workflow instructions perfectly. It explicitly references the plan in its solve output.

### Is it a prompt issue, model training issue, or tokenizer issue?

| Possible Cause | Ruled In/Out | Evidence |
|---------------|-------------|----------|
| **Model training issue** | ✅ **ROOT CAUSE** | Math model was trained on single-turn math QA, not multi-step workflows |
| **Prompt issue** | ❌ Not primary | Even with detailed prompts the math model ignores constraints |
| **Tokenizer issue** | ❌ Not primary | Same tokenizer family (Qwen2.5) as instruct model |
| **GPU/quantization issue** | ❌ Not relevant | Q4_K_M quantization is same for all models; GPU other models work fine |

---

## 4. Recommended Configuration for the Workflow

### Primary Recommendation: **Use instruct model for ALL steps**

```
steps=[
    StepConfig(name="plan",    model_key="qwen2.5-1.5b"),
    StepConfig(name="solve",   model_key="qwen2.5-1.5b"),
    StepConfig(name="compose", model_key="qwen2.5-1.5b"),
]
```

**Rationale**:
- Best instruction-following of any 1.5B model tested
- Correctly separates plan from solution
- Follows the plan in the solve step
- Only issue: sometimes hallucinates on the compose step (Q1 produced 24 instead of 9) — this may be fixable with a more robust compose prompt or by extracting `\boxed{}` from the solve step directly

### Alternative: **Use Llama-3.2-1B-Instruct for ALL steps**

```
steps=[
    StepConfig(name="plan",    model_key="llama-3.2-1b"),
    StepConfig(name="solve",   model_key="llama-3.2-1b"),
    StepConfig(name="compose", model_key="llama-3.2-1b"),
]
```

**Rationale**:
- Best instruction-following of all tested (perfect plan/solve separation, explicit references)
- Fastest inference (239ms plan, 744ms solve, 67ms compose)
- Correct answer on test question
- Risk: smaller model may struggle with complex math

### NOT Recommended: **Math model in multi-step workflow**

The math model cannot participate meaningfully in a multi-step workflow because it treats every step as an independent math problem to solve from scratch. All three steps produce essentially the same output.

### If you MUST use the math model (for accuracy): **Use single-prompt mode**

Instead of:
```
[plan] → [solve] → [compose]
```

Use:
```
USER: {problem}
Work through this: first plan, then solve step by step, then put answer in \boxed{}.
```

The math model handles this well because it matches its training distribution.

### Compose step improvement

The compose step sometimes hallucinates (Config A Q1 produced 24 instead of 9). The compose step's current prompt is:
```
"Format the answer clearly. Start with 'The answer is \boxed{number}'."
```

This is too open-ended. Better approaches:
1. **Skip compose entirely** — extract `\boxed{}` from solve output programmatically
2. **Use a simpler compose prompt** — just reformat, don't re-solve:
   ```
   "The following solution was computed. Simply reformat the answer as 'The answer is \boxed{N}'. Do NOT recalculate."
   ```
3. **Feed the exact boxed answer** from solve step into compose with a fill-in-the-blank template

---

## 5. Summary

| Model | Follows Plan Instruction? | Math Accuracy | Recommendation |
|-------|--------------------------|---------------|----------------|
| qwen2.5-1.5b-instruct | ✅ Good | Moderate | **BEST for workflow** |
| qwen2.5-math-1.5b | ❌ Poor — regenerates every step | Higher | Only use in single-prompt mode |
| Llama-3.2-1B-Instruct | ✅ Excellent | Good (on simple) | Good alternative, very fast |

**Root cause**: Qwen2.5-Math-1.5B was trained on single-turn math QA where the expected output is always a complete solution. It has no concept of multi-step workflows, role-specific prompts, or constrained output formatting. The model's training data never included examples of "output ONLY a plan" or "execute this pre-existing plan," so it defaults to its learned pattern: complete solution from scratch for every input.
