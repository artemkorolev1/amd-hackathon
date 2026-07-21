# Dynamic System Prompt System — Design Document

## Overview

This document describes the **dynamic system prompt system** for the AMD ACT II
Hackathon routing pipeline. The system assembles per-category, per-complexity
system prompts that are optimized for the **grader's fuzzy-matching evaluation
cascade**.

### The Grader's Compare Cascade (from `evaluate.py`)

The grader tries strategies **in order**, returning success on the first match:

1. **Exact match** (case-insensitive, normalized whitespace)
2. **Substring containment** (expected text in answer, or short answer in expected)
3. **Numeric tolerance** (within 1% for single or pairwise numbers)
4. **Token overlap** (≥50% for short expected, ≥30% for longer; stopwords stripped)

### Design Implications

| Cascade Step | Prompt Strategy |
|---|---|
| Exact match | Instruct exact label format (sentiment), exact numeric format (math) |
| Substring match | Use "Answer:" prefix lines so key terms appear as substrings |
| Numeric tolerance | Always use standard decimal (e.g., 17.5 not 17,5); round appropriately |
| Token overlap | Be concise; remove stopwords/filler; include ALL key entities |

---

## Architecture

### Components

```
user_prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│  build_system_prompt(                       │
│    category,         ← 8-way classifier     │
│    complexity_score, ← per-category complexity│
│    feature_scores,   ← multi-axis features   │
│    custom_instructions                       │
│  )                                          │
│                                             │
│  Steps:                                     │
│  1. Map complexity_score → low/medium/high  │
│  2. Select base prompt from _CATEGORY_PROMPTS│
│  3. Inject multi-axis feature instructions  │
│  4. Prepend custom instructions (if any)    │
│  5. Append anti-preamble suffix             │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
          System prompt (string)
```

### Per-Category Base Prompts (3 tiers each)

Each of the 8 categories defines a prompt variant for **low**, **medium**, and
**high** complexity. The complexity dimension controls:

- **Low complexity**: Answer-only, no steps, minimal instructions
- **Medium complexity**: Brief steps, some guidance on format
- **High complexity**: Full reasoning, edge cases, thorough checking

### Multi-Axis Feature Injection

Four feature axes from Stage 1 are conditionally injected when their score
exceeds 0.5:

| Feature | Low | Medium | High |
|---|---|---|---|
| **creativity** | (none) | "Be creative but stay factual" | "Feel free to be creative" |
| **verbosity** | (none) | "Be concise" | "Be extremely terse" |
| **structured_output** | (none) | "Use bullet points" | "Use JSON or labeled sections" |
| **multi_step** | (none) | "Think step by step" | "Break into sub-problems" |

---

## Per-Category Strategy for Grader Optimization

### code_gen
- **Grader checker**: `code_tests` (extracts function → runs test cases)
- **Strategy**: Enforce exact function name and signature via the prompt.
  Output in fenced code block so grader can extract cleanly.
- **Key prompt instruction**: "Preserve the exact function name and signature."

### code_debug
- **Grader checker**: `code_tests` (same as code_gen)
- **Strategy**: One bug sentence (for substring match) + corrected code block.
- **Key prompt instruction**: "Preserve the original function name and signature."

### math
- **Grader checker**: `numeric` (1% tolerance)
- **Strategy**: "Answer: <value>" line ensures the numeric value appears as a
  clean substring. No units unless required. Standard decimal format.
- **Key prompt instruction**: "End with 'Answer: <value>' on its own line."

### logic
- **Grader checker**: substring match (short expected answer)
- **Strategy**: "Answer: <conclusion>" line. Keep conclusion short (single word
  or short phrase) for token overlap match.
- **Key prompt instruction**: "End with 'Answer: <conclusion>' on its own line."

### factual
- **Grader checker**: `contains_all` (multi-part answers)
- **Strategy**: Include ALL requested facts. Use exact names/numbers. Word
  budget carefully tuned (proven best at "under 120 words"; "under 50 words"
  and "1-2 sentences" both failed the gate).
- **Key prompt instruction**: "Address every part of multi-part questions."

### sentiment
- **Grader checker**: `label` (exact label match)
- **Strategy**: Enforce exact label vocabulary. Low complexity = label only.
  Medium/high = label + justification. The label must be one of
  positive/negative/neutral/mixed (matches the grader's exact match step).
- **Key prompt instruction**: "Output EXACTLY one word: positive, negative,
  neutral, or mixed."

### ner
- **Grader checker**: `contains_all` (entity name + entity type)
- **Strategy**: Structured output "EntityText (TYPE)" ensures clean extraction.
  Include ALL entities. Use exact entity types: PERSON, ORGANIZATION, LOCATION,
  DATE.
- **Key prompt instruction**: "Output ONE entity per line as: EntityText (TYPE)."

### summarization
- **Grader checker**: `sentence_count` or `word_max`
- **Strategy**: Strict length constraint obedience. The prompt MUST say "strictly
  obey any length constraint" because the grader checks sentence/word count.
- **Key prompt instruction**: "Strictly obey ANY length constraint stated in the
  prompt."

---

## Integration with the Pipeline

### Stage Connection Points

| Stage | Input | Dynamic Prompt Role |
|---|---|---|
| Stage 0 (Prefilters) | Task text | Not involved |
| Stage 1 (Multi-axis features) | Task text → feature scores | Feature scores feed into prompt injection |
| Stage 2 (8-way category) | Task text → category | Category selects base prompt |
| Stage 3 (Complexity) | Task text → complexity score | Complexity maps to low/medium/high tier |
| Stage 4 (Solver selection) | System prompt + task | Solver receives assembled messages |

### Usage in the Solver

```python
from agent.dynamic_prompts import build_solver_messages, get_max_tokens

# Full dynamic assembly
messages = build_solver_messages(
    category="math",
    task="A store has 240 items. It sells 15% on Monday...",
    complexity_score=0.4,       # from Stage 3
    feature_scores={            # from Stage 1
        "multi_step": 0.7,
        "verbosity": 0.3,
    },
    deterministic_hint="144"    # optional, from deterministic solver
)
max_tokens = get_max_tokens("math", complexity_score=0.4)

# Quick lookup (no dynamic injection)
system_prompt, max_tokens, stop = lookup_prompt_config("math", "medium")
```

### Comparison to Existing Prompt Systems

| Feature | LabLab prompts | amd-track1 prompts | hybrid-token-router | **This system** |
|---|---|---|---|---|
| Per-category | ✅ | ✅ | ✅ | **✅ (8 categories)** |
| Per-complexity | ❌ | ❌ | ❌ (1 tier/cat) | **✅ (3 tiers)** |
| Multi-axis injection | ❌ | ❌ | ❌ | **✅ (4 axes)** |
| Anti-preamble | ❌ | ❌ | ❌ | **✅** |
| Grader-cascade-aware | partial | partial | partial | **✅ full** |
| max_tokens scaling | ❌ (fixed) | ❌ (fixed) | ❌ (fixed) | **✅ scaled** |
| Deterministic hint | ❌ | ❌ | ❌ | **✅** |
| Lookup table | ❌ | ❌ | ❌ (render func) | **✅** |

---

## Proven Best Practices Incorporated

From the winning repos and the project's own experiments:

1. **"Answer:" prefix** — proven by jaeyooniee (100% @ 3753 tokens) and our
   own run 13 (94.7%). Enables substring match on the expected answer.

2. **Anti-preamble** — deepseek and kimi models love generating "I need to..."
   or "The user wants..." before the answer. The anti-preamble suffix fights
   this. From project's templates.py (Jul 2026), proven effective.

3. **Label-only for simple sentiment** — cut #15 showed sentiment can be
   label-only (1 word) for simple cases without accuracy loss.

4. **Code block enforcement** — code categories must output fenced blocks for
   the `code_tests` checker to extract the function.

5. **Two-part question handling** — factual prompts explicitly instruct
   "address every part of multi-part questions" because contains_all checkers
   penalize missing sub-answers.

6. **Length constraint obedience** — summarization prompts STRICTLY obey
   stated constraints, matching the sentence_count/word_max checkers.

7. **Verifier-friendly format** — NER output format (one entity per line,
   Type in parens) enables clean programmatic extraction.

8. **Complexity-adaptive** — simple tasks need NO step-by-step reasoning;
   complex tasks need more scaffolding. This matches the bitmorphic classifier
   output and prevents wasted tokens on obvious problems.

---

## Files

| File | Description |
|---|---|
| `agent/dynamic_prompts.py` | Main prompt assembly module |
| `tests/test_dynamic_prompts.py` | 29 tests validating the system |

## Running the Tests

```bash
cd /home/artem/dev/amd-hackathon
python3 tests/test_dynamic_prompts.py
```
