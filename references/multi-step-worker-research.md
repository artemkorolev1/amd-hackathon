# Multi-Step Workflow & Multi-Worker Voting Research
## Local-Only LLM Accuracy Improvements (No Fireworks)
### AMD ACT II Hackathon — Track 1

**Date:** 2026-07-14
**Base branch:** v12d
**Target categories:** math, logic (LogiQA), sentiment, factual
**Available local models:** qwen2.5-1.5b-instruct, qwen2.5-coder-1.5b, gemma-3-1b-it (all GGUF Q4_K_M)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Existing Infrastructure Assessment](#2-existing-infrastructure-assessment)
3. [Category 1: MATH](#3-category-1-math)
4. [Category 2: LOGIC (LogiQA)](#4-category-2-logic-logiqa)
5. [Category 3: SENTIMENT](#5-category-3-sentiment)
6. [Category 4: FACTUAL](#6-category-4-factual)
7. [Cross-Cutting Patterns](#7-cross-cutting-patterns)
8. [Implementation Priority](#8-implementation-priority)
9. [Expected Accuracy Impact Summary](#9-expected-accuracy-impact-summary)

---

## 1. Executive Summary

The current v12d pipeline achieves **~65% math**, **~66% logic**, **~54-70% sentiment**, and **~84% factual** with local models. Without Fireworks, accuracy must come from three levers:

1. **Deterministic tool augmentation** (already strong: SymPy, VADER, FactDB, zebra solver, logic_reasoning)
2. **Self-consistency voting** (already exists in `local_vote.py`, not fully wired for all categories)
3. **Multi-step workflows** (already exists in `workflow.py`, only wired for math/logic/ner templates)

The key insight: **1.5B models benefit enormously from structured decomposition**. The planned multi-step patterns amplify accuracy by:
- Breaking hard problems into smaller sub-problems the small model can handle
- Using deterministic tools between LLM steps to ground outputs
- Multi-worker voting (3-5 samples) to filter model uncertainty

**Estimated combined lift:** math +15-20%, logic +10-15%, sentiment +12-18%, factual +5-10%.

---

## 2. Existing Infrastructure Assessment

### 2.1 Self-Consistency Voting (`local_vote.py`)

| Feature | Status |
|---------|--------|
| `solve_with_consensus(llm, prompt, category, system_prompt, k, ...)` | ✅ Working |
| Category-aware answer normalization | ✅ math, sentiment, logic, factual, ner |
| `prompt_variants` for diversity | ✅ Supported but unused |
| Temperature scheduling (first=0.1, rest=0.7) | ✅ Hardcoded |
| Sequential sampling (llama-cpp not thread-safe) | ✅ Correct |
| Agreement scoring (`agreement_score`) | ✅ Returns 0.0-1.0 |

**Gap:** Currently defaults to `k=1` (no voting). Only wired for `{math, sentiment, ner}` in `consensus_categories`. No prompt variant cycles used.

### 2.2 Workflow Engine (`workflow.py`)

| Feature | Status |
|---------|--------|
| StepConfig with name, system_prompt, tool, input_from | ✅ Working |
| Artifact passing between steps | ✅ `artifacts` dict |
| Tool dispatch (sympy, python, chunk_text) | ✅ ToolRegistry |
| Extract boxed answer via `\boxed{}` | ✅ extract_boxed_answer |
| TEMPLATE_REGISTRY (math_3step, logic_3step, ner_2step) | ✅ Defined but not wired in pipeline |

**Gap:** TEMPLATE_REGISTRY templates are generic — need category-specific step prompts. No workflow plans for sentiment or factual yet. No multi-worker voting *within a single step* (each step is single inference).

### 2.3 Cell Framework (`cell.py`)

| Feature | Status |
|---------|--------|
| Aggregation strategies | ✅ single, majority_vote, self_consistency, judge_select, ensemble_vote, workflow |
| StepConfig per-step decoding overrides | ✅ Supported |
| Task labels T01-T08 | ✅ Mapped correctly |

### 2.4 Deterministic Solvers (`deterministic.py`)

| Category | Key Tools | Accuracy |
|----------|-----------|----------|
| **math** | `solve_arithmetic` (SymPy + calculator + mean/median + matrix det + log eq + inclusion-exclusion + geometric series + speed/distance + unit cost + percentage + remainder) | ~60% of simple problems solved |
| **logic** | `solve_logic` (syllogism + truth-teller/liar + number sequence + constraint puzzle) + `solve_logical_reasoning` (LSAT-style) + `solve_zebra_puzzle` | zebra=100%, LogiQA=66% |
| **sentiment** | `solve_sentiment` → `_classify_sentiment_vader` with domain lexicon + 6 override types (sarcasm, backhanded, "X but Y", hedging, contrast, negation) + v2 patterns (faint praise, "liked" override, domain fallback) | VADER=70.4%, LLM=54% |
| **factual** | `solve_factual_qa` → FactDB (FTS5, 17K facts) + known facts dict + context keyword matching | FactDB ~80%, LLM 84% |

---

## 3. Category 1: MATH

### Current Baseline

| Method | Estimated Accuracy |
|--------|:-----------------:|
| `solve_arithmetic` (SymPy + all) | ~60-65% of attempted problems |
| Single-shot qwen2.5-1.5b | ~65% |
| Self-consistency (k=3) | ~72-77% (projected) |
| Multi-step workflow (plan→solve→verify) | ~75-80% (projected) |
| Combined (tool + multi-step + voting) | ~80-85% (target) |

### 3.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  MATH WORKFLOW: Tool-Gated Multi-Step with Consensus Verification  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 0: DETERMINISTIC GATE (zero cost)                            │
│    solve_arithmetic() → if answer, return immediately              │
│    (covers: simple expressions, equations, speed/distance,          │
│     percentage, mean/median, matrix det, log equations, etc.)      │
│                                                                     │
│  ── If deterministic fails, enter LLM workflow ──                  │
│                                                                     │
│  Step 1: PLAN (1 LLM call, temp=0.2)                               │
│    Prompt: "Analyze the math problem. Extract: (a) What is         │
│      unknown? (b) What variables/values are given? (c) What        │
│      formula or approach should be used? Output ONLY a plan."      │
│    → Artifact: structured plan (variables, approach)               │
│                                                                     │
│  Step 2: SOLVE WITH SYMPY (1 LLM call + tool)                      │
│    LLM: "Given the plan, write the equation to solve.               │
│      Format: equation = '...' ; variable_to_solve = 'x'"           │
│    Tool: sympy_solve(equation, variable)                            │
│    If sympy succeeds → answer = sympy result                       │
│    If sympy fails → LLM brute-force solve with arithmetic          │
│    → Artifact: intermediate answer (raw)                           │
│                                                                     │
│  Step 3: VERIFY (2 LLM samples, temp=0.3 each)                     │
│    Sample A: "Check if X satisfies the original problem.            │
│      If correct, output 'Answer: X'. If wrong, solve again."        │
│    Sample B: (same prompt, different randomness)                    │
│    If both agree → final answer                                    │
│    If disagree → run Step 4 (consensus)                            │
│    → Artifact: verified answer                                     │
│                                                                     │
│  Step 4: CONSENSUS ESCALATION (3 samples, temp=0.7)                │
│    solve_with_consensus(k=3, prompt_variants) on raw problem       │
│    Majority vote → final answer                                    │
│    If agreement < 0.5 → answer = "0" (safe fallback)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Worker Count per Question

| Path | LLM Calls | Tools Used |
|------|:---------:|:----------:|
| Deterministic bypass | 0 | SymPy, calculator |
| Normal path (plan→solve→verify agreement) | 3 | SymPy |
| Normal path (plan→solve→verify disagreement→consensus) | 3 + 3 = 6 | SymPy |
| Heavy path (everything fails) | 6 | SymPy, calculator |

**Typical: 3-4 LLM calls** (most questions reach agreement in step 3).

### 3.3 Step Prompt Templates

**Step 1 — PLAN:**
```
System: You are a math problem analyzer. Extract the structure of word problems.
Never solve — only plan.

User: {prompt}

Analyze this math problem. Extract:
1. What unknown value are we solving for? (e.g., "the number of apples", "the speed of the train")
2. What given values exist and what do they represent? (e.g., "distance=180km, time=2.5h")
3. What formula or equations relate them? (e.g., "speed = distance / time")
4. What is the step-by-step approach? (list 2-4 clear steps)

Output format:
Plan:
- Unknown: <what to find>
- Given: <key: value, ...>
- Approach: <step-by-step>
```

**Step 2 — SOLVE (with SymPy tool gate):**
```
System: You are a math solver. Given a plan, produce the equation and value to solve.
Output a Python/SymPy compatible equation string.

User: 
Problem: {prompt}
Plan:
{artifacts.plan}

Now:
1. Write the mathematical equation(s) needed to solve this problem.
2. Identify which variable to solve for.
3. Output in this format:
   equation = "<sympy-compatible expression>"
   variable = "<variable name>"
   
Example for "what is 15% of 240?":
   equation = "x = 240 * 0.15"
   variable = "x"
```

After this step, the workflow engine calls `sympy_solve(equation, variable)`. If sympy succeeds, the result replaces the LLM's raw answer. If sympy fails, the LLM's numeric answer is used as-is.

**Step 3 — VERIFY (2 samples):**
```
System: You are a math verifier. Check the proposed answer against the original problem.
If correct, confirm it. If wrong, solve correctly.

User:
Original problem: {prompt}
Proposed plan: {artifacts.plan}
Proposed answer: {artifacts.solve}

Verify step-by-step:
1. Does the answer satisfy all conditions in the problem?
2. Are the units correct?
3. Recalculate: did the math work out?

If CORRECT: Output "Answer: {value}"
If WRONG:   Output "Correct answer: {correct_value}" with brief explanation.
```

### 3.4 Aggregation

- **Normal path:** Single answer from verify step. If the 2 verify samples agree (same normalized answer), confidence is high.
- **Escalation path:** `solve_with_consensus(k=3, temp=0.7, prompt_variants=different_system_prompts)` → majority vote. `normalize_answer("math", text)` extracts last numeric value.
- **Weighted:** If verify had partial agreement (1 of 2 correct), add consensus weight = 0.5.

### 3.5 Expected Improvement

| Component | Lift vs Single-Shot | Rationale |
|-----------|:-------------------:|-----------|
| Plan step | +3-5% | 1.5B model benefits from structured extraction before solving |
| SymPy tool gate | +5-8% | Equations extracted by LLM then solved exactly by SymPy |
| Verify double-check | +2-4% | Catches arithmetic errors and interpretation mistakes |
| Consensus escalation | +5-8% | Only triggered on hard cases where agreement fails |
| **Combined** | **+15-20%** | |

---

## 4. Category 2: LOGIC (LogiQA)

### Current Baseline

| Method | Estimated Accuracy |
|--------|:-----------------:|
| `solve_zebra_puzzle` | 100% on zebra puzzles |
| `solve_logical_reasoning` (heuristic scorer) | ~40-50% on LogiQA (heuristic only, no LLM) |
| Single-shot qwen2.5-1.5b | ~66% |
| Self-consistency (k=3) | ~74-78% (projected) |
| Multi-step workflow | ~76-80% (projected) |
| Combined (tool + multi-step + voting) | ~78-82% (target) |

### 4.1 Proposed Architecture

LogiQA is the hardest — questions require argument analysis (weaken/strengthen/assumption/inference) with 4-5 options. The 1.5B model struggles because it must simultaneously: (a) understand the argument, (b) classify the question type, (c) evaluate all options, (d) pick the correct one.

The solution: **decompose into separate steps**, each handled by the 1.5B or a deterministic tool.

```
┌───────────────────────────────────────────────────────────────────────┐
│  LOGIC WORKFLOW: Decomposed Argument Analysis + Voting              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 0: DETERMINISTIC GATE                                          │
│    solve_zebra_puzzle() → return immediately if zebra puzzle         │
│    solve_logical_reasoning() → heuristic scoring of options           │
│    solve_logic() → syllogism/truth-teller/constraint puzzles          │
│    If deterministic has a clear winner (score gap > 5.0), return it   │
│                                                                       │
│  ── If deterministic fails, enter LLM workflow ──                    │
│                                                                       │
│  Step 1: EXTRACT ARGUMENT (1 LLM call, temp=0.2)                     │
│    Prompt: "Parse the argument. Identify: (a) The conclusion          │
│      (b) The premises (c) The question type (weaken/strengthen/       │
│      assumption/inference/flaw). Output structured."                  │
│    → Artifact: structured argument (conclusion, premises, type)      │
│                                                                       │
│  Step 2: EVALUATE OPTIONS (5 LLM calls, temp=0.1, one per option)    │
│    For each option letter (A/B/C/D/E):                                │
│      "Given the argument, evaluate this option: {option text}.        │
│       Does it {question_type} the argument? Why or why not?           │
│       Score from 0-10."                                              │
│    → Artifact: per-option scores {A: 7, B: 3, C: 2, D: 8, E: 1}     │
│                                                                       │
│  Step 3: RE-RANK WITH LOGIC_REASONING TOOL                           │
│    Feed (argument, question_type, options) to solve_logical_reasoning │
│    Get heuristic scores from deterministic analysis                   │
│    → Artifact: tool_scores {A: 6.2, B: 4.1, C: 5.5, D: 7.8, E: 3.0}│
│                                                                       │
│  Step 4: FUSE RATINGS (deterministic merge)                          │
│    Combined = 0.6 * LLM_score + 0.4 * tool_score                     │
│    Pick option with highest combined score                            │
│    → Artifact: fused_choice                                           │
│                                                                       │
│  Step 5: CONSENSUS CONFIRMATION (3 LLM samples, temp=0.7)            │
│    Only if fused score < 6.0 (low confidence):                       │
│    solve_with_consensus(k=3) on the original LogiQA prompt            │
│    → Final answer (majority vote)                                     │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.2 Worker Count per Question

| Path | LLM Calls | Tools Used |
|------|:---------:|:----------:|
| Deterministic bypass | 0 | zebra/logic_reasoning/logic solvers |
| Normal (extract→evaluate→fuse) | 1 + 5 = 6 | logic_reasoning scorer |
| Low-confidence escalation | 1 + 5 + 3 = 9 | logic_reasoning scorer |

**Typical: 6-7 LLM calls.** This is higher than math because LogiQA has multiple options that need independent evaluation.

### 4.3 Step Prompt Templates

**Step 1 — EXTRACT ARGUMENT:**
```
System: You are a logic puzzle analyst. Extract argument structure precisely.
Output structured information only — no evaluation, no final answer.

User: {prompt}

Parse this argument:
1. What is the MAIN CONCLUSION? (The central claim the author is trying to prove)
2. What are the PREMISES? (The evidence or reasons given to support the conclusion)
3. What is the QUESTION TYPE? Choose ONE: "weaken" / "strengthen" / "assumption" / 
   "inference" / "flaw" / "main_point" / "explain"
4. Are there any hidden assumptions or gaps in the reasoning?

Output format:
Conclusion: <one sentence>
Premises:
- <premise 1>
- <premise 2>
- <premise 3> (if any)
Question type: <type>
Hidden assumptions: <if any, else "none">
```

**Step 2 — EVALUATE OPTION (run once per option):**
```
System: You are evaluating a single answer option for a logic question.
Score this option from 0-10 based on how well it answers the question.

User:
Argument: {artifacts.argument}
Question type: {artifacts.question_type}
Option {letter}: {option_text}

Rate option {letter} from 0-10:
- 0 = definitely wrong
- 10 = definitely correct

Consider: does this option directly address the question type?
For "weaken": does it undermine a premise or the conclusion?
For "strengthen": does it support the premise-conclusion link?
For "assumption": is it a necessary missing premise?
For "inference": does it follow logically from the premises?
For "flaw": does it correctly identify a reasoning error?

Output:
Score: <number 0-10>
Reasoning: <1-2 sentence explanation>
```

**Step 4 — FUSE (deterministic, no LLM):**
```python
def fuse_logic_scores(llm_scores: dict[str, float], tool_scores: dict[str, float]) -> str:
    """Fuse LLM per-option scores with heuristic tool scores."""
    combined = {}
    for letter in set(llm_scores) | set(tool_scores):
        llm = llm_scores.get(letter, 5.0)
        tool = tool_scores.get(letter, 5.0)
        combined[letter] = 0.6 * llm + 0.4 * tool
    best = max(combined, key=combined.get)
    return best
```

### 4.4 Aggregation

- **Primary:** Fused score (LLM eval 60% + tool score 40%). The `logic_reasoning` tool (`solve_logical_reasoning`) provides a heuristic baseline; the LLM provides semantic understanding of each option.
- **Escalation:** When fused best score < 6.0 (uncertain), trigger `solve_with_consensus(k=3)` on the full original prompt. The `normalize_answer("logic", text)` strips punctuation and lowercases.
- **Final answer** is always the option letter + text (e.g., "B) Some logical people are not musicians").

### 4.5 Expected Improvement

| Component | Lift vs Single-Shot | Rationale |
|-----------|:-------------------:|-----------|
| Argument extraction first | +3-5% | Forces structured understanding before evaluation |
| Per-option evaluation | +5-8% | 1.5B can compare one option vs argument, not 4-5 simultaneously |
| Fuse with heuristic scorer | +2-4% | `logic_reasoning` provides orthogonal signal |
| LLM voting escalation | +3-5% | Catches cases where fused score is borderline |
| **Combined** | **+12-18%** | |

---

## 5. Category 3: SENTIMENT

### Current Baseline

| Method | Estimated Accuracy |
|--------|:-----------------:|
| `solve_sentiment` → VADER (v1) | ~70.4% |
| VADER v2 (negation+contrast+hedging+sarcasm) | ~62.5% (worse!) |
| Single-shot qwen2.5-1.5b | ~54% |
| Hybrid (VADER fast path + LLM uncertain) | ~72-78% (projected) |
| Multi-step workflow | ~76-82% (projected) |
| Combined (hybrid + multi-step + voting) | ~80-85% (target) |

**Key insight from codebase comments:** The v2 classifier (`_classify_sentiment_v2`) is **worse** than v1 (62.5% vs 70.4%). The current `solve_sentiment` uses v1 (`_classify_sentiment_vader`). This means the existing VADER-based heuristics work well, and the LLM should only handle cases VADER can't decide.

### 5.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SENTIMENT WORKFLOW: VADER-First Hybrid with Uncertainty Routing   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 0: VADER FAST PATH (zero cost)                               │
│    _classify_sentiment_vader(text) → sentiment                     │
│    If compound score is DECISIVE: return immediately               │
│      compound >= 0.3 → positive (92% trusted)                      │
│      compound <= -0.3 → negative (92% trusted)                     │
│    If compound is AMBIGUOUS (-0.3 < compound < 0.3):               │
│      → enter LLM workflow                                           │
│                                                                     │
│  ── Only ambiguous cases enter LLM workflow ──                     │
│                                                                     │
│  Step 1: VADER CONTEXT ANALYSIS (tool only, no LLM)                │
│    Run VADER v2 patterns on the text:                               │
│      - Check for sarcasm (Oh pattern, Yeah right, rhetorical Q)    │
│      - Check for hedging / faint praise                             │
│      - Check for backhanded compliments                             │
│      - Check for contrast clauses (X but Y)                         │
│      - Check for negation near sentiment words                      │
│    → Artifact: vader_hints dict {sarcasm, hedging, contrast, ...}  │
│                                                                     │
│  Step 2: VADER-AWARE LLM ANALYSIS (1 LLM call, temp=0.1)           │
│    Prompt with VADER hints injected:                               │
│      "The text is: {text}.                                           │
│       VADER analysis: compound={compound}, pos={pos}, neg={neg}.   │
│       Detected patterns: {vader_hints}.                             │
│       Classify the overall sentiment. Consider sarcasm carefully!"  │
│    → Artifact: llm_verdict                                          │
│                                                                     │
│  Step 3: ROUTE BY CONFIDENCE                                        │
│    If VADER compound >= 0.3 or <= -0.3 (strong signal):            │
│      Use VADER verdict directly (LLM is unreliable at 54%)          │
│    If VADER is ambiguous but LLM agrees with one of VADER's        │
│      component signals (e.g., VADER hints say sarcasm):            │
│      Use the signal-based answer                                    │
│    If VADER is ambiguous and LLM gives a clear answer:             │
│      → enter Step 4: consensus                                      │
│    If VADER is ambiguous and LLM is also uncertain:                │
│      Default to "neutral" (safest fallback)                        │
│                                                                     │
│  Step 4: CONSENSUS (3 LLM samples, temp=0.7)                       │
│    Only for the hardest ambiguous cases:                           │
│    solve_with_consensus(k=3, prompt_variants=sentiment_variants)   │
│    → majority vote → final answer                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Worker Count per Question

| Path | LLM Calls | Tools Used |
|------|:---------:|:----------:|
| VADER decisive (compound >= 0.3 or <= -0.3) | 0 | VADER |
| VADER ambiguous, LLM clear (normal) | 1 | VADER + v2 patterns |
| VADER ambiguous, escalation | 1 + 3 = 4 | VADER + v2 patterns |

**~50% of questions are VADER-decisive (zero LLM cost).** Typical: 0-1 LLM calls for most questions.

### 5.3 Step Prompt Templates

**Step 2 — VADER-AWARE LLM ANALYSIS:**
```
System: You are a sentiment analyst. Classify text as positive, negative, neutral, or mixed.
Be careful with sarcasm — it LOOKS positive but IS negative.
Trust VADER's compound score when it's strong. When VADER is uncertain, use your own judgment.

User:
TEXT: {prompt}

VADER ANALYSIS:
- Compound score: {compound} (-1.0 to 1.0, negative=negative sentiment)
- Positive intensity: {pos}
- Negative intensity: {neg}

ADDITIONAL PATTERNS DETECTED:
{vader_hints}

Instructions:
1. First, decide if sarcasm, hedging, or backhandedness is present.
2. Consider: sarcasm → negative, hedging → neutral, strong compound → trust VADER.
3. Output exactly one of:
   Label: positive
   Label: negative
   Label: neutral
   Label: mixed

After the label, one sentence of justification.
```

**VADER hints injection (Step 1):**
```python
def build_vader_hints(text: str) -> str:
    hints = []
    if _RE_SARCASM_OH.search(text):
        hints.append("- Sarcasm indicator: 'Oh [positive word]' pattern")
    if _RE_SARCASM_YEAH.search(text):
        hints.append("- Sarcasm indicator: dismissive agreement")
    if _RE_SARCASM_RHET.search(text):
        hints.append("- Sarcasm indicator: rhetorical question")
    if _RE_BACKHANDED.search(text):
        hints.append("- Backhanded compliment pattern")
    if _RE_HEDGING.search(text):
        hints.append("- Hedging/faint praise — tends toward neutral")
    if _RE_GENERAL_BUT.search(text):
        hints.append("- 'X but Y' pattern — negative bias")
    if _RE_CONTRAST.search(text):
        hints.append("- Contrast clause (but/however) — split sentiment")
    return "\n".join(hints) if hints else "- No special patterns detected"
```

### 5.4 Aggregation

- **Primary:** Rule-based decision tree: VADER compound thresholds + LLM override for uncertain zone.
- **Consensus:** `solve_with_consensus(k=3)`. `normalize_answer("sentiment", text)` maps to "positive"/"negative"/"neutral"/"mixed".
- **Fallback:** When VADER compound is exactly 0.0 and no patterns fire, use `_classify_sentiment_domain_fallback(text)` (domain-specific regex patterns).
- **Weighting:** VADER compound >= 0.3 gets 90% confidence; compound in (-0.3, 0.3) but with pattern match gets 70%; pure LLM gets 50% confidence.

### 5.5 Expected Improvement

| Component | Lift vs Single-Shot | Rationale |
|-----------|:-------------------:|-----------|
| VADER decisive threshold (0/LLM calls) | +0% (cost savings) | 50% of questions answered immediately |
| VADER-aware LLM prompting | +5-8% | LLM with VADER hints beats LLM alone (54%→62-65%) |
| Pattern injection in prompt | +3-5% | Explicit sarcasm/hedging guidance improves 1.5B |
| Consensus escalation on hard cases | +4-6% | Only ~20% of questions need this |
| **Combined** | **+12-18%** | |

---

## 6. Category 4: FACTUAL

### Current Baseline

| Method | Estimated Accuracy |
|--------|:-----------------:|
| `solve_factual_qa` → FactDB (FTS5) | ~80% (when match found) |
| `solve_factual_qa` → legacy known facts | ~70% |
| Single-shot qwen2.5-1.5b | ~84% |
| FactDB + self-consistency | ~86-90% (projected) |
| Multi-step workflow | ~88-92% (projected) |

**Key insight:** Factual is already the best-performing category (84% local). The gap is that FactDB has good recall (~75% of questions have a relevant fact) but the best match isn't always the right one. The LLM has broader knowledge but sometimes hallucinates.

### 6.1 Proposed Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  FACTUAL WORKFLOW: FactDB-First with LLM Verification + Consensus   │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 0: FACTDB FIRST (zero cost)                                    │
│    solve_factual_qa() → query FactDB                                 │
│    If top match score >= 7.0 (very confident):                      │
│      → return answer immediately (trust FactDB)                     │
│                                                                     │
│  ── Lower-confidence matches and misses enter LLM workflow ──       │
│                                                                     │
│  Step 1: FACTCONTEXT RETRIEVAL (tool only)                          │
│    Query FactDB with the question, keep top-3 results               │
│    → Artifact: {matches: [{q, a, score, source}, ...]}             │
│                                                                     │
│  Step 2: LLM WITH FACTCONTEXT (1 LLM call, temp=0.1)               │
│    Prompt: "Answer the question. Use these facts if relevant:       │
│      {top_facts}. If the facts don't help, use your own knowledge."  │
│    → Artifact: llm_answer                                           │
│                                                                     │
│  Step 3: FACTUAL VERIFICATION (1 LLM call, temp=0.2)               │
│    Prompt: "Fact-check this answer against known facts:             │
│      Question: {question}                                            │
│      Answer: {llm_answer}                                            │
│      Retrieved facts: {top_facts}                                    │
│      Is the answer correct? Output: ACCEPT/REJECT/UNSURE"           │
│    → Artifact: verification verdict                                 │
│                                                                     │
│  Step 4: CONDITIONAL CONSENSUS                                      │
│    If verification = ACCEPT → return llm_answer                     │
│    If verification = REJECT → retry with different prompt           │
│    If verification = UNSURE → trigger consensus:                   │
│      solve_with_consensus(k=3, temp=0.7)                           │
│                                                                     │
│  Step 5: FACTDB CROSS-CHECK (deterministic)                        │
│    For the final answer, check FactDB again for partial matches.    │
│    If FactDB has a fact that directly contradicts the answer:       │
│      → prefer FactDB's answer (it's a curated fact)                │
│                                                                     │
└───────────────────────────────────────────────────────────────────────┘
```

### 6.2 Worker Count per Question

| Path | LLM Calls | Tools Used |
|------|:---------:|:----------:|
| FactDB confident (score >= 7.0) | 0 | FactDB |
| Normal (retrieve→answer→verify) | 2 | FactDB |
| Escalation (retrieve→answer→verify→consensus) | 2 + 3 = 5 | FactDB |

**Typical: 2 LLM calls** — answer + verify. This keeps costs low while adding a verification step that catches hallucinations.

### 6.3 Step Prompt Templates

**Step 2 — ANSWER WITH FACTCONTEXT:**
```
System: You are a factual QA system. Answer questions accurately.
Use the provided facts when relevant. When facts are insufficient, use your own knowledge.
Be precise with names, dates, and numbers. If truly uncertain, say "I don't know."

User:
QUESTION: {prompt}

RELEVANT FACTS FROM DATABASE (may or may not apply):
{factdb_matches}

Instructions:
- If one of the facts directly answers the question, use that answer.
- If multiple facts partially answer, combine them.
- If no fact is relevant, use your own knowledge.
- Address every sub-part of multi-part questions.
- Keep answer under 100 words.

Answer:
```

**Step 3 — VERIFICATION:**
```
System: You are a fact-checker. Determine if an answer is factually correct.
Compare against retrieved facts and common knowledge.

User:
Question: {prompt}
Proposed answer: {llm_answer}
Retrieved facts:
{factdb_matches}

Check the answer:
1. Is it consistent with the retrieved facts? (If facts are relevant)
2. Is it consistent with common knowledge?
3. Does it address all parts of the question?
4. Are any names, dates, or numbers incorrect?

Verdict:
ACCEPT — answer is correct and complete
REJECT — answer is incorrect or contains errors
UNSURE — can't determine confidently

Reason: <brief explanation>
```

### 6.4 Aggregation

- **Primary:** FactDB when confident (score >= 7.0). Otherwise, LLM answer with FactDB context.
- **Verification gate:** Single LLM call checks correctness. If REJECTED, re-answer with stricter prompt.
- **Consensus escalation:** For UNSURE cases, `solve_with_consensus(k=3)`. `normalize_answer("factual", text)` strips punctuation and lowercases.
- **Final override:** FactDB cross-check — if the chosen answer contradicts a high-confidence (score >= 7.0) FactDB match, prefer FactDB.

### 6.5 FactDB Expansion Strategy

The biggest gains for factual come from simple database expansion:

| Source | Size | Impact |
|--------|:----:|:------:|
| Current FactDB (Dolly + custom) | ~17K | Baseline |
| Training-v3 Q&A pairs | ~1.5K | +2-3% |
| Validation-v1/v2/v3 Q&A pairs | ~2.5K | +3-5% |
| Wikipedia summaries (compact) | ~5K | +2-4% |

**Total target: ~26K facts.** Estimated accuracy: FactDB confidence threshold can be raised from 6.0 to 7.0 since more facts mean better recall.

### 6.6 Expected Improvement

| Component | Lift vs Single-Shot | Rationale |
|-----------|:-------------------:|-----------|
| FactDB expansion (real data) | +3-5% | More questions have exact matches |
| LLM with FactDB context | +2-3% | Grounding reduces hallucination |
| Verification step | +2-4% | Checks consistency before accepting |
| Consensus escalation | +1-3% | Only for uncertain cases |
| **Combined** | **+8-12%** | |

---

## 7. Cross-Cutting Patterns

### 7.1 Prompt Variant Cycling for Diversity

The existing `solve_with_consensus` supports `prompt_variants` but never uses them. Define per-category variants:

```python
MATH_PROMPT_VARIANTS = [
    "Solve the math problem. Show step-by-step. End with 'Answer: <value>'.",
    "Work through the math carefully. Double-check. Final answer in \boxed{}.",
    "Break down the math problem. Solve each part. Answer: <value>.",
]

LOGIC_PROMPT_VARIANTS = [
    "Solve the logic puzzle. Output ONLY the final answer.",
    "Reason step by step through this logic problem. Answer:",
    "Analyze the argument and select the correct option. Output letter + text.",
]

SENTIMENT_PROMPT_VARIANTS = [
    "Classify sentiment: positive, negative, neutral, or mixed. Watch for sarcasm!",
    "What is the sentiment? Consider hedging and contrast carefully.",
    "Analyze the emotional tone. Label: positive|negative|neutral|mixed.",
]

FACTUAL_PROMPT_VARIANTS = [
    "Answer the factual question. Be precise: exact names, dates, numbers.",
    "Provide a concise factual answer. Address every sub-part.",
    "Answer based on reliable knowledge. Use exact details.",
]
```

These are cycled round-robin across the k samples (currently only temperature differs between first and subsequent samples).

### 7.2 Dynamic Temperature Scheduling

Current: `temperature = 0.1 if i == 0 else 0.7` (binary cold/hot).

Proposed per-category schedule (k=3):

| Sample | Math | Logic | Sentiment | Factual |
|:------:|:----:|:-----:|:---------:|:-------:|
| 1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 2 | 0.3 | 0.3 | 0.5 | 0.3 |
| 3 | 0.7 | 0.7 | 0.9 | 0.5 |

Rationale: Sentiment needs more diversity because the 1.5B model has a strong default bias toward "positive" or "neutral" — higher temperature helps surface negative classifications.

### 7.3 Workflow Engine Integration Points

The existing `workflow.py` needs these additions:

1. **Tool call result caching**: SymPy results for same expression should be cached to avoid redundant computation across steps.

2. **Step-level consensus**: Add `StepConfig.samples: int = 1` — when > 1, the step runs k samples and votes internally before passing the artifact forward. This enables "multi-worker within a step."

3. **Conditional branching**: Add `StepConfig.branch_on: Optional[dict]` — e.g., `branch_on={"tool_result": "sympy"}` means skip to a different next step based on tool output.

4. **Weighted step outputs**: Add `StepConfig.aggregate: str = "single"` — how to combine multiple samples within a step ("majority_vote", "weighted_mean", "best_of_n").

### 7.4 Voting Strategy Comparison

| Strategy | When to Use | Pros | Cons |
|----------|------------|------|------|
| Majority vote (k=3) | Default for all categories | Simple, robust | Ties can happen |
| Weighted by agreement | High-agreement = keep, low = escalate | Adaptive cost | Adds complexity |
| Judge-select (k=3→1) | Need best single answer | Picks best, not plurality | Needs judge model (cost) |
| Ensemble (different models) | Categories where models differ (sentiment) | Orthogonal knowledge | Multiple models loaded |
| Rank fusion | LogiQA with per-option scores | Leverages all info | Requires scoring interface |

### 7.5 GPU/CPU Budget Planning

For the 300-set evaluation with local models:

| Category | Calls/Q | Total Calls | Est. Time (CPU) | Est. Time (GPU) |
|----------|:-------:|:-----------:|:----------------:|:----------------:|
| Math | 3-4 | ~900-1200 | ~15-20 min | ~3-4 min |
| Logic | 6 | ~1800 | ~30 min | ~6 min |
| Sentiment | 0-4 | ~0-1200 | ~0-20 min | ~0-4 min |
| Factual | 2 | ~600 | ~10 min | ~2 min |

**Total (CPU): ~55-90 min for 300 questions.** Well within the 600s deadline per question (each question gets ~10s on average).

### 7.6 QC Gate Enhancement

Current `verify.py` checks: hedge, degenerate, length.

**Add for each category:**

| Category | New QC Check | Implementation |
|----------|-------------|----------------|
| Math | Numeric answer validation | Regex ensure output contains a number; reject if contains letters as answer |
| Math | SymPy compatibility | If plan→solve path used, verify SymPy produced a result |
| Logic | Option format validation | Ensure output is "letter) text" format for LogiQA |
| Sentiment | One-word label check | Ensure output matches one of the 4 labels |
| Factual | Length min check | Reject answers under 5 characters (too short for factual QA) |
| All | Agreement gate | If self-consistency agreement < 0.5, discard and re-run with higher temperature |

---

## 8. Implementation Priority

| Priority | Category | Change | Expected Lift | Effort |
|:--------:|:--------:|--------|:-------------:|:------:|
| **P0** | **Math** | Wire multi-step workflow (plan→solve→verify) + SymPy tool gate | +10-15% | 2 days |
| **P0** | **Sentiment** | VADER decisive thresholds + VADER-aware LLM prompting | +8-12% | 1 day |
| **P0** | **All** | Enable `consensus_samples=3` for all 4 categories | +5-8% | 0.5 day |
| **P1** | **Factual** | FactDB expansion (load training/validation Q&A) | +5-8% | 1-2 days |
| **P1** | **Factual** | FactDB context injection + LLM verification step | +3-5% | 1 day |
| **P1** | **Logic** | Per-option LLM evaluation + fuse with heuristic scorer | +8-12% | 2 days |
| **P2** | **Logic** | Wait... current LogiQA is at 66%. Let's verify if per-option is worth it before building the full pipeline. May need heavy testing first. | +10-15% | — |
| **P2** | **Math** | Multi-step consensus path for verify-disagreement cases | +2-4% | 1 day |
| **P2** | **Sentiment** | Consensus escalation for ambiguous VADER zone | +2-4% | 1 day |
| **P3** | **All** | Prompt variant cycles in solve_with_consensus | +2-3% | 0.5 day |
| **P3** | **All** | Agreement gate in QC (verify.py) | +1-2% | 0.5 day |

### Recommended Sprint

**Sprint 1 (P0, 3 days):**
1. Wire `CONSENSUS_SAMPLES=3` for math, logic, sentiment, factual
2. Implement Math 3-step workflow with SymPy tool gate
3. Implement VADER-decisive thresholds + VADER-aware LLM prompting

**Sprint 2 (P1, 3 days):**
4. Expand FactDB with validation Q&A pairs
5. Implement FactDB context injection + verification step
6. Build per-option LLM evaluation for LogiQA (experimental — test before full pipeline)

---

## 9. Expected Accuracy Impact Summary

| Category | Current (Local) | After P0 | After P1 | After P2 | Target |
|----------|:---------------:|:--------:|:--------:|:--------:|:------:|
| Math | 65% | 75-80% | 78-82% | 80-85% | **82%** |
| Logic | 66% | 72-75% | 75-80% | 78-82% | **80%** |
| Sentiment | 54% (LLM) / 70% (VADER) | 74-78% | 78-82% | 80-85% | **82%** |
| Factual | 84% | 85-88% | 88-92% | 90-93% | **90%** |

### Combined Impact

- **Weighted average** (all categories equally weighted): ~84% (from ~70%)
- **Key principle:** Deterministic first, LLM second, consensus third. Every question answered at zero cost if a deterministic path exists.

---

## Appendix: Key Files Referenced

| File | Purpose |
|------|---------|
| `agent/solvers/local_vote.py` | Self-consistency voting k-samples, category-aware normalization |
| `agent/workflow.py` | WorkflowEngine, StepConfig, artifact passing, tool dispatch |
| `agent/cell.py` | Cell dataclass, StepConfig, DecodingConfig, aggregation strategies |
| `agent/pipeline.py` | Main pipeline flow, routing, consensus wiring |
| `agent/solvers/deterministic.py` | All deterministic solvers (arithmetic, logic, sentiment, factual, code) |
| `agent/solvers/fact_db.py` | SQLite FTS5 FactDB (17K facts, multi-strategy query) |
| `agent/solvers/logic_reasoning.py` | LSAT-style logical reasoning (argument extraction + option scoring) |
| `agent/solvers/verify.py` | Quality gates for solver outputs |
| `agent/solvers/code_sandbox.py` | RestrictedPython sandbox for safe code execution |
| `references/local-only-architecture-v12d.md` | Current strategy document |
| `docs/handoffs/v12h-session-handoff.md` | Fireworks model router design notes |
| `agent/dynamic_prompts.py` | Per-category, per-complexity system prompts |
