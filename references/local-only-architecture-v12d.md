# Local-Only Per-Category Architecture — v12d (FW-free)

**Principle:** No Fireworks. Everything runs on local models + deterministic tools.
**Strategies:** Multi-step workflows, self-consistency voting, tool augmentation.

---

## Category Ceiling Assessment

| Category | Current Best Local | Ceiling | Limitation |
|----------|:------------------:|:-------:|------------|
| code_debug | 100% | ✅ REACHED | Already perfect — no improvement needed |
| code_gen | 90-100% | ✅ NEAR-CEILING | Template solver + LLM covers everything |
| ner | ~67% (LLM) / F1=0.96 (solver) | ⚠️ SOLVER CEILING | prototype_ner_v3 handles {entity@} format perfectly. LLM prompt fixed. |
| logic | 66% local | 🔶 NEEDS WORK | Zebra 100%, but LogiQA needs multi-vote |
| math | 65% local | 🔶 NEEDS WORK | Format compliance fixed, CoT needs multi-vote |
| sentiment | 54% local / 70% VADER | 🔶 NEEDS WORK | Hybrid routing + multi-sample consensus |
| summarization | 62% local | 🔶 NEEDS WORK | Sumy install + multi-sample consensus |
| factual | 84% local | 🔶 NEEDS WORK | FactDB expansion + voting |

---

## Strategy Per Category

### 1. code_debug — ✅ AT CEILING
**Architecture:** Deterministic bug solver (14 patterns) → qwen2.5-coder-1.5b
**Improvement:** Expand bug patterns from 14→20+. Add sandbox verification.
**No multi-step needed** — already 100%.

### 2. code_gen — ✅ NEAR CEILING
**Architecture:** Template solver (30 templates) → qwen2.5-coder-1.5b / gemma-3-1b
**Improvement:** Add 8 more templates (surface_area, nth_octagonal, split_lowercase, etc.). Sandbox tests for templates.
**No multi-step needed** — single-shot works.

### 3. NER — ⚠️ SOLVER CEILING
**Architecture:** prototype_ner_v3 (F1=0.961) → local LLM (corrected prompt)
**Improvement:** Wire format normalizer for LLM output. Add missing entity type mapping (spaCy→tweetner).
**Multi-step:** Not needed if prototype_ner_v3 covers it. LLM is fallback only.

### 4. Logic — 🔶 NEEDS WORK
**Current:** Zebra 100%, LogiQA 66%, truth-teller/syllogism moderate
**Improvement Strategy:**
- **Deterministic first**: Zebra solver → truth-teller solver → constraint solver
- **Self-consistency voting**: 3-5 samples for LogiQA questions (different temp=0.1/0.2)
- **Multi-step workflow**: For LogiQA: (1) Extract argument components → (2) Classify question type (weaken/strengthen/assumption) → (3) Score each option → (4) Pick best
- **Tool**: Constraint satisfaction for zebra/seating puzzles

### 5. Math — 🔶 NEEDS WORK
**Current:** Arithmetic solver + SymPy → local LLM at 65%
**Improvement Strategy:**
- **Deterministic first**: Arithmetic direct → SymPy word problems → AST-safe calculator
- **Self-consistency voting**: 3-5 samples at temp=0.1 (different CoT paths converge on same answer)
- **Multi-step workflow**: (1) Parse problem → extract variables → (2) Set up equations → (3) Solve → (4) Verify with calculator
- **Tool**: SymPy + AST-safe calculator + sandbox for verification

### 6. Sentiment — 🔶 NEEDS WORK
**Current:** VADER 70.4% → local LLM 54%
**Improvement Strategy:**
- **VADER fast path**: compound < -0.3 → direct (92% trusted)
- **Self-consistency voting**: 3 samples for uncertain cases (compound between -0.3 and 0.3)
- **Multi-step workflow**: (1) VADER pre-analysis → (2) LLM with VADER hint → (3) Format normalizer → (4) If disagreement, trigger 3-way vote
- **Tool**: VADER + domain lexicon + format normalizer
- **v2 pattern backport**: hedging, negation, contrast clause splitting, "liked" override

### 7. Summarization — 🔶 NEEDS WORK
**Current:** Sumy (not installed) → local LLM 62%
**Improvement Strategy:**
- **Sumy install**: pip install sumy → 4 extractive strategies work immediately
- **Self-consistency voting**: 3 samples averaged via entity overlap
- **Multi-step workflow**: (1) Chunk long input → (2) Summarize each chunk → (3) Merge chunk summaries → (4) Keyword/entity overlap quality check
- **Tool**: Sumy + keyword extraction + length constraint parser

### 8. Factual — 🔶 NEEDS WORK (NOT STARTED)
**Current:** FactDB (17K facts, 0.34ms) → local LLM 84%
**Improvement Strategy:**
- **FactDB expansion**: Load training-v3, validation-v1/v2/v3 Q&A pairs (4K+ entries)
- **Self-consistency voting**: 3 samples for questions where FactDB confidence < 6.0
- **Multi-step workflow**: (1) FactDB query → (2) If found (score≥6.0) → direct. (3) If not found → LLM with expanded FactDB context. (4) Verify against FactDB if partial match.

---

## Cross-Cutting Architecture: Multi-Worker + Multi-Step

### Self-Consistency Voting (where applicable)
```
For math, logic, sentiment, factual:
  1. Run 3-5 LLM calls with temp=0.1-0.2 (different seed)
  2. Use solve_with_consensus from agent/solvers/local_vote.py
  3. Majority vote on final answer
  4. If agreement < 0.6 → run verification step
```

### Multi-Step Workflow (where beneficial)
```
For math, logic (LogiQA):
  Step 1: "Analyze the problem. What type is it? Extract key variables/constraints."
  Step 2: "Solve using the extracted information. Show step-by-step."
  Step 3: "Verify. Does the answer satisfy all constraints? If not, fix it."

For sentiment (hard cases):
  Step 1: "Identify sentiment-bearing phrases and their polarity."
  Step 2: "Consider sarcasm, hedging, contrast. Classify overall sentiment."
  Step 3: "Output exactly one word: positive/negative/neutral/mixed."

For factual (uncertain):
  Step 1: "What is the question asking? What type of fact is needed?"
  Step 2: "Answer based on your knowledge. If uncertain, say 'I don't know'."
  Step 3: "Verify the answer is complete and accurate."
```

### QC Gate Enhancement
Current QC (verify.py) checks: hedge, degenerate, length.
Add: **self-consistency agreement gate** — if 3-vote agreement < 0.5, discard and re-run with higher temperature.

---

## Implementation Priority

| Priority | Category | Change | Expected Lift |
|:--------:|----------|--------|:-------------:|
| P0 | **math** | Enable self-consistency voting (CONSENSUS_SAMPLES=3) | +10-15% |
| P0 | **sentiment** | Wire VADER hybrid solver + v2 patterns | +8-12% |
| P0 | **factual** | Expand FactDB with 4K+ training entries | +5-8% |
| P1 | **summarization** | pip install sumy + wire | +10-15% |
| P1 | **logic** | Self-consistency voting for LogiQA | +8-12% |
| P2 | **code_gen** | Expand templates + sandbox tests | +3-5% |
| P2 | **NER** | Format normalizer for LLM output | +2-5% |
| P3 | **code_debug** | Expand bug patterns | +0-2% (already at 100%) |
