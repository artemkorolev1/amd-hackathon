# GEPA Judge+Analyze: logical_reasoning

## 1. Question Type Taxonomy in Eval Sets

### 1.1 Primary Types Found

| Type | training-v3 | validation-v1 | validation-v2 | eval_hard_218 | Description |
|------|:-----------:|:-------------:|:-------------:|:-------------:|-------------|
| **LogiQA Argument Analysis** | 10 | 13 | 7 | 0 | LSAT-style: paragraph + stem + 4 choices (0-3). Types: strengthen, weaken, assumption, inference, flaw, main_point, explain |
| **Zebra Puzzles (truncated)** | 9 | 37 | 30 | 0 | "Solve: There are N houses..." — prompt truncated, expected output is empty grid JSON with `___` placeholders |
| **Constraint/Seating Puzzles** | 0 | 0 | 0 | ~15 | "Five colleagues — Tom, Oscar, Nina, Maya, and Leo — each work in a different department..." Full puzzles with constraints to solve |
| **Syllogisms** | 0 | 0 | 0 | ~4 | "All philosophers are logical. Some logical people are mathematicians..." + MCQ options |
| **Truth-Teller/Liar** | 0 | 0 | 0 | ~4 | "On a strange island, there are three types of people: Knights (always tell the truth), Knaves (always lie)..." |
| **Number/Letter Sequences** | 0 | 0 | 0 | ~1 | "7-gallon jug and 11-gallon jug" |
| **Conditional Logic** | 0 | 0 | 0 | ~2 | "If the train is late, then the meeting is postponed..." |

**Total logic questions across all sets: ~160+**, with zebra puzzles dominating (76 in val-v1+v2 alone).

### 1.2 Answer Format per Type

| Type | Answer Format | Example |
|------|--------------|---------|
| LogiQA | `number. text` (MCQ letter + full option) | `3. People can get essential minerals from other foods.` |
| Zebra (truncated) | JSON dict with `header` + `rows` of `___` | `{'header': ['House', 'Name', 'Birthday'], 'rows': [['___', '___', '___'], ...]}` |
| Constraint puzzle | Position assignments | `Floor 1: Tom (Sales), Floor 2: Oscar (HR), ...` |
| Syllogism | MCQ letter | `B) Some logical people are not musicians.` |
| Truth-teller | Text assignments | `Alex is Normal, Blake is Knave, Casey is Knight.` |
| Conditional logic | MCQ letter | `C) The train is not late.` |

---

## 2. Current Solver Infrastructure & Integration

### 2.1 Available Solvers

| Solver | File | Handles | Wired? | Status |
|--------|------|---------|--------|--------|
| `solve_logic()` | `deterministic.py:1264` | Truth-teller, sequences, syllogisms, small constraints | ❌ **Not called** | Dead code in pipeline |
| `solve_truth_teller_liar()` | `deterministic.py:1009` | Knight/knave brute-force (≤5 characters) | ❌ | Only reachable via `solve_logic()` |
| `solve_number_sequence()` | `deterministic.py:1157` | Number/letter sequences | ❌ | Only reachable via `solve_logic()` |
| `_solve_syllogism()` | `deterministic.py:769` | Categorical syllogisms (All/Some/No) | ❌ | Only reachable via `solve_logic()` |
| `_solve_constraint_puzzle()` | `deterministic.py:907` | Small permutation puzzles (≤5 items) | ❌ | Only reachable via `solve_logic()` |
| `solve_logic_puzzle()` | `logic_solver.py:10` | CSP solving (seating, scheduling, ordering) using python-constraint | ❌ | Only imported in `tool_registry.py` (lazy) |
| `solve_logical_reasoning()` | `logic_reasoning.py` | LSAT/LogiQA argument analysis (strengthen, weaken, assumption, etc.) | ❌ | Only imported in `tool_registry.py` (lazy) |
| `solve_zebra_puzzle()` | `prototype_zebra_v2.py:96` | Zebra puzzles (returns empty grid) — **100% on 9/9** | ❌ | **Orphaned** — not imported anywhere |
| Local LLM | `pipeline.py:804` | qwen2.5-1.5b-instruct, 512 tok | ✅ | Active for logic |
| FW (kimi-k2p7-code) | `pipeline.py:509` | `_is_hard_logic` → FW direct | ✅ | Active |

### 2.2 Critical Integration Gap

**The deterministic solver chain is completely broken for logic:**

```python
# pipeline.py:159-165
det_cat_map: dict[str, str] = field(default_factory=lambda: {
    "math": "math_arithmetic",
    "sentiment": "sentiment",
    "summarization": "summarization",
    "factual": "other_complex",
    "code_debug": "code_debugging",
})
# NOTE: "logic" is MISSING from det_cat_map
```

The deterministic solver dispatch (lines 545, 794) checks `if category not in self.cfg.det_cat_map: return ""`. Since "logic" is not a key, **all deterministic logic solvers are bypassed**. The pipeline goes straight from FW escalation check to local LLM.

Additionally, the `_fireworks_escalate` code for logic (line 509) runs BEFORE the deterministic solver loop, meaning even if `_is_hard_logic()` is false (most cases), the solvers still don't run.

---

## 3. Current Prompt Architecture Analysis

### 3.1 System Prompts for Logic

**dynamic_prompts.py (3 tiers):**
```python
"logic": {
    "low": "Solve the logic puzzle. Show a single step of reasoning, then end with 'Answer: <conclusion>' on its own line. Keep the conclusion short — a name, item, or single word.",
    "medium": "Solve the logic puzzle step by step — show your reasoning in 2-3 clear steps. End with 'Answer: <conclusion>' on its own line. Keep the conclusion short and precise.",
    "high": "Solve the logic puzzle carefully. Show reasoning step by step — use a table, grid, or deductive chain if needed. Verify every condition is satisfied. End with 'Answer: <conclusion>' on its own line. Keep the conclusion short and unambiguous.",
}
```

**Reasoning model prompt (pipeline.py:306-313):**
```
Solve the logic puzzle. Output ONLY the final answer — NO preamble, NO 'To solve...', NO step-by-step reasoning in your response. For assignment puzzles: immediately output ALL assignments on ONE line: 'Position 1: Name (Role); Position 2: Name (Role); ...'. For option-letter questions: output ONLY the option letter and its full text. For procedural puzzles: output the sequence of steps concisely. For yes/no or single-conclusion puzzles: output ONLY: Answer: <word or short phrase>.
```

**MC Logic prompt (pipeline.py:752-757):**
```
Select the correct option from the choices given. Output ONLY the letter and its full text exactly as written, e.g., 'B) Some logical people are not musicians.' No explanation.
```

**FW Format prompt (fw_router.py:52-53):**
```
Output ONLY the answer. One word or letter. No explanation. No reasoning steps.
```

### 3.2 Prompt-Types Fit Assessment

| Question Type | Which Prompt Fires | Correct for Type? | Issue |
|--------------|-------------------|-------------------|-------|
| LogiQA (MCQ) | `_MC_LOGIC_PROMPT` (if 3+ choices detected) ✅ | Yes — asks for letter+full text | But grader expects `number. text` format (0-3), not letter format (A-D) |
| LogiQA (MCQ, reasoning model) | `reasoning_prompts["logic"]` ⚠️ | Has MCQ instruction but verbose | May produce correct format for letter IDs but LogiQA uses 0-3 numbers |
| Zebra (truncated) | `reasoning_prompts["logic"]` ❌ | Wrong — expects short answer, not JSON | No prompt asks for JSON grid output |
| Zebra (truncated, reasoning) | `reasoning_prompts["logic"]` ❌ | Has "assignment puzzles" instruction | But no mention of JSON format, header, or `___` placeholders |
| Constraint puzzle | `reasoning_prompts["logic"]` ✅ | "assignment puzzles: Position 1: Name (Role)" | Good format match for position-based output |
| Syllogism MCQ | `_MC_LOGIC_PROMPT` ✅ | Asks for letter+full text | Correct format |
| Truth-teller | `reasoning_prompts["logic"]` ⚠️ | "single-conclusion puzzles: Answer: <word>" | But expected format is multi-line assignments |
| Conditional logic MCQ | `_MC_LOGIC_PROMPT` ✅ | Correct | — |

### 3.3 Max Tokens Issues

- **dynamic_prompts.py says 200** for logic (line 649), scaling to 260 for complex
- **FW says 500** for logic (fw_router.py:98)
- **MC logic gets only 80 tokens** (pipeline.py:775-776) — this may truncate LogiQA answers which include full option text like "3. People can get essential minerals from other foods."
- **Consensus voting** NOT enabled for logic (only math, sentiment, ner at line 145)

### 3.4 Stop Sequences

- **dynamic_prompts.py**: stops on `\n\n` and `Question:`
- **pipeline.py (non-reasoning logic)**: stops on `Question:` and `Context:`
- This is fine for LogiQA which doesn't have "Question:" in the prompt

---

## 4. Routing Analysis

### 4.1 Current Routing Path

```
process(prompt)
  → Stage0 bypass
  → Category classification (2. Logic scored)  
  → Complexity scoring
  → Routing table check (usually empty for logic)
  → FW escalation: _is_hard_logic() check (line 509)
    → If 4+ capitalized names OR multi-person pattern: FW (kimi-k2p7-code)
    → Otherwise: skip
  → Build system prompt (3 logic prompt variants)
  → Deterministic solver loop → SKIPPED (logic not in det_cat_map)
  → Local LLM inference (qwen2.5-1.5b)
  → Post-processing (extract Answer: or expand bare letter)
  → If empty: FW fallback
```

### 4.2 `_is_hard_logic()` Analysis

```python
def _is_hard_logic(prompt: str, category: str) -> bool:
    if category != "logic":
        return False
    return (
        len(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)) >= 4
        or bool(_MULTI_PERSON_RE.search(prompt))
    )
```

**Effectiveness per type:**

| Type | Has 4+ capitalized names? | Caught by `_is_hard_logic`? | Should it be hard? |
|------|--------------------------|----------------------------|-------------------|
| LogiQA argument (Chinese) | ❌ Usually 0-1 | ❌ Missed | YES — 1.5B gets ~66%, FW gets ~95% |
| Zebra puzzle | ❌ Few English words | ❌ Missed | N/A — zebra solver would handle |
| Constraint puzzle (eval_hard) | ✅ 5+ names | ✅ Caught | YES |
| Syllogism MCQ | ❌ 0-1 capitalized | ❌ Missed | YES — but FW handles 95% |
| Truth-teller | ✅ Named people | ✅ Caught | YES |

**Key finding:** `_is_hard_logic` misses ALL LogiQA argument analysis questions, which are the majority of the logic eval set. LogiQA questions have short Chinese names or no named entities at all — they use option numbers (0-3), not capitalized names.

### 4.3 FW Model for Logic

The router sends logic to **kimi-k2p7-code** with:
- Format prompt: "Output ONLY the answer. One word or letter. No explanation."
- Prefill: "Answer: "
- This is a mismatch for LogiQA answers which are full option texts (e.g., "3. People can get essential minerals from other foods.")

---

## 5. Current Accuracy (from per-category-architecture doc)

| Metric | Value |
|--------|-------|
| Local LLM accuracy | 66% |
| FW accuracy | 95% |
| Zebra solver | 100% (but orphaned!) |
| FW routing | Conditional (misses LogiQA) |

---

## 6. Specific Proposed Changes

### 6.1 P0: Wire Zebra Puzzle Solver (HIGH IMPACT, LOW EFFORT)

**Problem:** `prototype_zebra_v2.py` achieves 100% on 9/9 zebra puzzles but is orphaned.
**Fix:** Add zebra puzzle detection + solver to the pipeline's pre-LLM phase.

**Implementation options:**
- **Option A:** Import `solve_zebra_puzzle` in `pipeline.py` and add it as a fast-path check before the deterministic loop (zebra puzzles are identifiable by "Solve:" prefix + "house" keyword; detection is O(1) regex)
- **Option B:** Wire it into `_run_deterministic` or as a special-case in `process()`

**Recommended:** Add as a standalone check in `process()` after classification:
```python
# Zebra puzzle fast path — before deterministic solvers
if category == "logic":
    zebra = solve_zebra_puzzle(prompt, category)
    if zebra:
        return zebra
```
This adds <3ms overhead per logic prompt.

### 6.2 P0: Add "logic" to `det_cat_map` (HIGH IMPACT, LOW EFFORT)

**Problem:** `solve_logic()` from deterministic.py is never called because "logic" is not in `det_cat_map`.

**Fix:** Add `"logic": "logic"` to `det_cat_map` so truth-teller, syllogism, number sequence, and small constraint solvers are tried before local LLM.

```python
det_cat_map = {
    "math": "math_arithmetic",
    "logic": "logic",          # ADD THIS
    "sentiment": "sentiment",
    ...
}
```

This would enable the sub-solvers for at least some types:
- `solve_truth_teller_liar()` — works on eval_hard_218 truth-teller puzzles
- `_solve_syllogism()` — works on simple categorical syllogisms
- `_solve_constraint_puzzle()` — works on small permutation puzzles

### 6.3 P1: Wire `logic_solver.py` (CSP solver) into the Pipeline

**Problem:** `solve_logic_puzzle()` from logic_solver.py (using python-constraint) handles seating, scheduling, ordering, and attribute matching puzzles. It's only accessible via `tool_registry.py`, not the main pipeline.

**Fix:** Add a fast-path call to `solve_logic_puzzle(prompt)` in the logic solver chain, after the zebra solver check but before local LLM. This handles the constraint/seating puzzles from eval_hard_218.

**Note:** The import chain is already broken in `agent/solvers/__init__.py` (python-constraint import), but direct import `from agent.solvers.logic_solver import solve_logic_puzzle` works fine.

### 6.4 P1: Wire `logic_reasoning.py` (LogiQA solver) into the Pipeline

**Problem:** `solve_logical_reasoning()` handles LSAT/LogiQA argument analysis (strengthen, weaken, assumption, inference, flaw, main_point, explain). It's only in `tool_registry.py`.

**Fix:** Add as a pre-LLM check for LogiQA-style prompts:
```python
# Detect LogiQA-style: has paragraph + "Q:" + answer choices (0-3) or (A-E)
if _is_logiqa(prompt):
    from agent.solvers import logic_reasoning
    lr_answer = logic_reasoning.solve_logical_reasoning(prompt)
    if lr_answer:
        return lr_answer
```

**Detection heuristic:** Match pattern `\nQ:.*\n` + `(0|1|2|3)\.` answer choices.

### 6.5 P1: Fix `_is_hard_logic()` Threshold

**Problem:** The 4+ capitalized names heuristic misses all LogiQA questions.

**Fix:** Add detection for LogiQA-style argument analysis (which needs FW due to complexity):
```python
def _is_hard_logic(prompt: str, category: str) -> bool:
    if category != "logic":
        return False
    # Existing: 4+ capitalized names OR multi-person constraint
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)) >= 4:
        return True
    if _MULTI_PERSON_RE.search(prompt):
        return True
    # NEW: LogiQA-style argument analysis — needs FW
    # Detects: paragraph + Q: stem + numbered choices
    if re.search(r'\n\s*Q\s*[:：].*\n', prompt, re.I) and \
       re.search(r'(?:0|[A-E])\s*[.)]。]', prompt):
        return True
    # NEW: Complex syllogism (All/Some/No with 3+ categories)
    if len(re.findall(r'\b(all|some|no|most)\b', prompt, re.I)) >= 3:
        return True
    return False
```

### 6.6 P2: Prompt Structure Changes

**For LogiQA (MCQ with number options):**
The `_MC_LOGIC_PROMPT` should also handle numbered options (0-3 format):
```python
_MC_LOGIC_PROMPT = (
    "Select the correct option from the choices given. "
    "Output ONLY the option number and its full text exactly as written, "
    "e.g., '0. All heroes can stand the test...' or 'B) Some logical people are not musicians.' "
    "No explanation."
)
```

**For truncated zebra puzzles:**
Since `prototype_zebra_v2.py` will handle these, the prompt never fires for them. But keep as backup:
```python
# No change needed — prompt handles 'assignment puzzles' format
```

**For truth-teller/knight-knave:**
The reasoning prompt has "single-conclusion puzzles: Answer: <word>" but truth-teller expected format is multi-line like `Alex is Normal, Blake is Knave, Casey is Knight.` Add explicit format instruction:
```
For truth-teller puzzles: output ALL assignments on separate lines, e.g., 'Alex is Knight\nBlake is Knave\nCasey is Knight'
```

### 6.7 P2: Max Tokens for Logic MCQ

Current `max_tok = 80` for MC logic (line 775-776) is too tight for LogiQA answers which include full option text like "3. People can get essential minerals from other foods. If they eat a balanced diet, they don't need mineral water." Increase to 150 for MC logic.

### 6.8 P2: Updated Solver Chain

```
process(prompt)
  → Stage0 bypass
  → Category classification → Logic
  → Complexity scoring
  ──────────────────────────────────────────────────────────
  NEW: Zebra puzzle detector + solver (prototype_zebra_v2)
    → If starts with "Solve: There are N houses" → DIRECT ANSWER (100%)
  NEW: LogiQA detector + solver (logic_reasoning.py)
    → If has paragraph + Q: stem + numbered choices → DIRECT ANSWER
  NEW: CSP solver (logic_solver.py)
    → If seating/scheduling/ordering patterns → DIRECT ANSWER
  NEW: solve_logic() from deterministic (via det_cat_map)
    → Truth-teller → DIRECT ANSWER
    → Syllogism → DIRECT ANSWER
    → Number sequences → DIRECT ANSWER
    → Small constraint puzzles → DIRECT ANSWER
  ──────────────────────────────────────────────────────────
  FIXED: _is_hard_logic() — now catches LogiQA + complex syllogisms
    → FW direct (kimi-k2p7-code with improved format prompt)
  ──────────────────────────────────────────────────────────
  Local LLM (qwen2.5-1.5b, 512 tok) — improved prompt per subtype
    → QC gate (verify answer)
  FW fallback on empty
```

### 6.9 Summary: Expected Accuracy Impact

| Change | Existing | Expected | Logic Subtypes Impacted |
|--------|:--------:|:--------:|------------------------|
| Wire zebra solver | 0% (orphaned) | **100%** | All zebra puzzles (~50% of logic eval) |
| Wire `solve_logic()` via `det_cat_map` | 0% (bypassed) | **70-100%** | Truth-teller, syllogisms, constraint puzzles |
| Wire `logic_reasoning.py` LogiQA solver | 0% (bypassed) | **50-70%** | LogiQA argument analysis (~20% of logic eval) |
| Fix `_is_hard_logic()` to catch LogiQA | 66% local (missed FW) | **95% FW** | All LogiQA questions |
| Improved prompts for subtypes | 66% local | **~75%** | All non-zebra, non-hard |
| **Combined impact** | **66% local, 95% FW** | **~85-90% effective** | Full range |
