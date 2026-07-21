# Per-Category Optimal Architecture Design — v12d Baseline

**Date:** 2026-07-14  
**Goal:** Define optimal solver chain + tool usage per category BEFORE GEPA prompt optimization.

---

## 1. sentiment_classification

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 54% (Qwen1.5B) |
| FW accuracy | 83% (gpt-oss-120b) |
| VADER deterministic | 70.4% (on hard set) |
| FW routing | Always (threshold=0.0 in main.py) |

### Optimal Solver Chain
```
VADER (compound < -0.3) ──► DIRECT ANSWER (92% acc)
       │
       ▼
VADER pattern match ──► DIRECT ANSWER (sarcasm/backhanded/but-clause)
       │
       ▼
Local LLM + VADER hint injection ──► QC gate → FW escalation
       │
       ▼
FW (gpt-oss-120b) for remaining hard cases
```

### Tools
- **VADER** with domain lexicon (96 tech/sentiment words), asymmetric thresholds (pos=0.05, neg=0.0), sarcasm/backhanded/but-clause overrides
- **Format normalizer** — strips markdown, extracts label, handles typos via Levenshtein
- **LLM** — qwen2.5-1.5b-instruct (or Nemotron-4B if headroom)
- **FW** — gpt-oss-120b or kimi-k2p7-code

### Routing Thresholds (to tune)
- VADER compound < -0.3 → direct (92% trusted)
- compound < -0.1 + LLM says positive → override to VADER
- compound > 0.7 → LLM only (VADER unreliable on positive)
- compound == 0.0 → LLM only (no signal)
- LLM confidence < 0.6 → FW escalation

### Prompt Architecture
- 3 tiers: low (explicit, short), medium (add sarcasm guard), high (add nuance detection)
- **Critical:** anti-positive bias only on medium/high tiers (low tier overcorrects)
- VADER hint injected as `[Note: preliminary analysis suggests {label} (score: {compound})]`
- No preamble, output EXACTLY one word

### GEPA Focus
1. Tune VADER routing thresholds on held-out set
2. Optimize sarcasm/backhanded detection patterns
3. Prompt variant ablation (with/without anti-positive bias per tier)

---

## 2. factual_knowledge

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 84% |
| FW accuracy | 92% |
| FactDB | 17K facts, 0.34ms queries |
| FW routing | Conditional (not in FIREWORKS_CATEGORIES) |

### Optimal Solver Chain
```
FactDB SQLite FTS5 (3-tier) ──► if best_score >= 6.0 → DIRECT
       │
       ▼
FactDB prefix/OR fallback ──► if best_score >= 3.0 → DIRECT
       │
       ▼
Local LLM (qwen2.5-1.5b, 64 tok) ──► QC gate → FW escalation
       │
       ▼
FW (gpt-oss-120b) for multi-hop/unanswerable
```

### Tools
- **FactDB** — SQLite FTS5 with BM25 tuning (k1=2.0, b=0.75), column weights (question=10, answer=5), source boosting
- **Keyword matcher** — NQ-normalised question lookup (from deterministic.py)
- **LLM** — qwen2.5-1.5b-instruct, max_tokens=64 (short answers)
- **FW** — gpt-oss-120b for complex multi-hop

### Routing Thresholds
- FactDB >= 6.0: high confidence → direct answer
- FactDB >= 3.0: medium confidence → direct but flag for verification
- FactDB < 3.0: skip, go to LLM
- LLM output == "" or QC fail → FW

### Prompt Architecture
- Single tier (empty/blank system prompt performs best — confirmed by GEPA)
- Just question as user message, no system prompt for local LLM
- FW: "Answer the question. Under 100 words. Include exact names, numbers."

### GEPA Focus
1. Expand FactDB with training-v3 and validation-v3 Q&A pairs
2. Tune BM25 thresholds on held-out set
3. Compare empty vs minimal prompt variants

---

## 3. logical_reasoning

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 66% |
| FW accuracy | 95% |
| Zebra solver | 100% (9/9 puzzles) |
| FW routing | Conditional (_is_hard_logic) |

### Optimal Solver Chain
```
Zebra puzzle detector ──► DIRECT (100% acc)
       │
       ▼
Constraint solver (brute-force) ──► DIRECT (≤5 items)
       │
       ▼
Truth-teller/liar solver ──► DIRECT (knight/knave puzzles)
       │
       ▼
Local LLM (qwen2.5-1.5b, 512 tok) ──► QC → FW
       │
       ▼
FW (deepseek-v4-pro) for all hard logic
```

### Tools
- **prototype_zebra_v2** — 100% on zebra puzzles (N houses, N attributes, constraints)
- **logic_solver.py** — constraint-based SAT for small puzzles
- **Truth-table solver** — knight/knave brute-force (2^n assignments)
- **LLM** — qwen2.5-1.5b-instruct, max_tokens=512 (CoT needed)
- **FW** — deepseek-v4-pro

### Routing Thresholds
- Zebra pattern match → direct (always)
- Truth-teller keywords → direct (always)
- _is_hard_logic() (4+ capitalized names or multi-person constraint) → FW direct
- Local LLM == "" or QC fail → FW

### Prompt Architecture
- **3 tiers:** Low (short syllogism → direct answer), Medium (multi-constraint → structured), High (zebra/complex → full constraint format)
- **Critical format:** "Output ONLY the answer. NO preamble." Small models produce verbosity otherwise
- Reasoning model variants get "After thinking, output ONLY..." prefix
- Use `_strip_kimi_preamble` for Kimi outputs

### GEPA Focus
1. Optimize prompt format for different puzzle types (zebra, constraint, syllogism, truth-teller)
2. Tune _is_hard_logic() threshold
3. Variant ablation: with/without reasoning step instruction

---

## 4. math_reasoning

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 65% |
| FW accuracy | 89% |
| SymPy solver | 5/8 complex word problems |
| FW routing | Conditional (_is_hard_math) |

### Optimal Solver Chain
```
Arithmetic solver (calc+regex) ──► DIRECT (pure arithmetic)
       │
       ▼
SymPy (symbolic math) ──► DIRECT (algebra, word problems)
       │
       ▼
Local LLM (qwen2.5-1.5b, 512 tok) ──► QC → FW
       │
       ▼
FW (deepseek-v4-pro) for all hard math (_is_hard_math)
```

### Tools
- **SymPy solver** — symbolic math with implicit_multiplication fix, 10-step priority pipeline
- **Arithmetic solver** — AST-safe calculator, remainder/mean/word-problem patterns
- **LLM** — qwen2.5-1.5b-instruct, max_tokens=512 (CoT critical)
- **FW** — deepseek-v4-pro

### Routing Thresholds
- is_hard_math() (determinant, bayes, modular, ratio, combinatorics) → FW direct
- SymPy returns non-None → direct (verified)
- LLM: if output has no number → QC fail → FW
- max_tokens=64 is WRONG for math (truncates CoT mid-reasoning) — use 512

### Prompt Architecture
- **3 tiers:** Low (simple arithmetic, "Answer: <value>"), Medium (word problem, step-by-step), High (multi-step, full CoT)
- **Critical:** "End with 'Answer: <value>' on its own line" on ALL tiers
- Local LLM needs explicit CoT instruction ("Solve step by step")
- FW prompt: remove "After thinking" prefix for kimi (it forces blank output in reasoning=none mode)

### GEPA Focus
1. max_tokens scaling per complexity tier
2. CoT vs direct answer variant ablation
3. "Answer: <value>" format compliance on 1.5B model
4. SymPy expansion: inequalities, coordinate geometry, trig

---

## 5. code_generation

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 90-100% |
| FW accuracy | 100% |
| FW routing | pipeline.py _fireworks_escalate |
| Lint retry | 2 retries with lint feedback |

### Optimal Solver Chain
```
Template solver (keyword→function template) ──► DIRECT (simple algorithms)
       │
       ▼
Local LLM (gemma-3-1b-it or qwen2.5-coder, 512 tok) ──► ast.parse + lint → retry loop
       │
       ▼
FW (kimi-k2p7-code) for complex specs (3+ requirements)
```

### Tools
- **Template solver** (deterministic) — fibonacci, factorial, sort, etc. Behavioral verification via sandbox tests
- **Black/Ruff** — code formatting and lint validation (up to 2 retries with lint feedback)
- **LLM** — gemma-3-1b-it (100% on 60-set, 2x faster) or qwen2.5-coder
- **FW** — kimi-k2p7-code

### Routing Thresholds
- Template match + ALL tests pass → direct (rare, <10% coverage)
- Lint retry fails after 2 attempts → FW
- 3+ requirements in spec → FW

### Prompt Architecture
- "Output ONLY the function inside ```python\n...\n```. Preserve exact function name and signature."
- No explanation, no docstring
- Lint retry: inject previous errors as context

### GEPA Focus
1. Expand template solver coverage
2. Lint retry loop effectiveness (how often does 2nd retry fix?)
3. FW vs local threshold for complex specs

---

## 6. code_debugging

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 96-100% |
| FW accuracy | 96% (regression from 100%) |
| Debug solver | ~10 bug patterns |
| FW routing | Same as code_gen |

### Optimal Solver Chain
```
Bug pattern solver (deterministic) ──► DIRECT (common bugs: OBO, NameError, NoneType)
       │
       ▼
Local LLM (qwen2.5-coder, 512 tok) ──► ast.parse + sandbox tests → retry loop
       │
       ▼
FW (kimi-k2p7-code) — rarely needed (FW actually regresses to 96%)
```

### Tools
- **Bug pattern solver** — OBO (off-by-one), NoneType, NameError, mutable default, indentation, import error, type mismatch
- **Sandbox tests** — executes fixed code in RestrictedPython to verify
- **Black/Ruff** — format + lint retry (same retry loop)
- **LLM** — qwen2.5-coder-1.5b (100% on debug tasks)
- **FW** — kimi-k2p7-code

### Routing Thresholds
- Bug pattern match → direct (verified by sandbox test)
- LLM output fails QC → FW
- EXCLUDE from FIREWORKS_CATEGORIES (FW regresses code_debug)

### Prompt Architecture
- "Output ONLY the fully corrected function. No description of the bug."
- "Preserve the original function name and signature."
- Retry: inject lint errors + test failures as feedback

### GEPA Focus
1. Expand bug pattern coverage (from 10 → 25+)
2. Sandbox test integration
3. Retry loop: when does it help vs hurt?

---

## 7. text_summarization

### Current State
| Metric | Value |
|--------|-------|
| Local accuracy | 62% |
| FW accuracy | 92% |
| Sumy solver | Working (lead-biased LexRank, 6-algo ensemble) |
| FW routing | Always via pipeline.py special handler |

### Optimal Solver Chain
```
Sumy extractive (input < 400 words, no multi-source) ──► DIRECT (keyword coverage ~95%)
       │
       ▼
Local LLM (qwen2.5-1.5b, 256 tok) ──► QC gate (entity recall + keyword overlap)
       │
       ▼
FW (gpt-oss-120b) for multi-source or complex summarization
```

### Tools
- **Sumy** — LexRank+LSA+KL+LSA+TextRank+SumBasic ensemble, lead-biased, chunk support
- **Multi-source detector** — SOURCE 1/2/3 pattern → FW direct
- **LLM** — qwen2.5-1.5b-instruct, max_tokens=256
- **FW** — gpt-oss-120b with multi-source synthesis prompt

### Routing Thresholds
- Input < 400 words AND no SOURCE markers → Sumy
- SOURCE markers present → FW direct (multi-source synthesis)
- Sumy output QC fails → LLM
- LLM output entity recall < 50% OR keyword overlap < 40% → FW
- Long input (>800 words) → FW (local 1.5B truncates)

### Prompt Architecture
- "Summarize in 1-2 sentences" with exact name/number requirement
- Multi-source FW: "Summarize key similarities and differences" with exact quotes
- Complex analytical prompts defer to FW (LLM loses nuance)

### GEPA Focus
1. Grading method: entity+keyword overlap (not fuzzy_match — too strict)
2. Sumy vs LLM routing threshold (optimal word count cutoff)
3. Prompt variants: length constraint vs free, entity-focus vs general

---

## 8. named_entity_recognition

### Current State
| Metric | Value |
|--------|-------|
| Local LLM accuracy | 54-67% |
| FW accuracy | 100% |
| prototype_ner_v3 | F1=0.961 (on training-v3) |
| spaCy solver | Built, working |
| FW routing | Always (FIREWORKS_CATEGORIES) |

### Optimal Solver Chain
```
prototype_ner_v3 (marker + regex) ──► DIRECT (F1=0.961)
       │
       ▼
spaCy `en_core_web_sm` ──► DIRECT (general NER: PERSON, ORG, GPE, etc.)
       │
       ▼
FW (deepseek-v4-pro) — 100% on all NER tasks
```

### Tools
- **prototype_ner_v3** — hill-climbed to F1=0.961. Phase 1: `{@entity@}` marker extraction (high precision), Phase 2: unmarked capitalized entities, Phase 3: biomedical regex fallback. Dedup + type classification by 100-char context window.
- **spaCy** — en_core_web_sm (disabled tagger/parser/lemmatizer for speed, 3-13ms)
- **FW** — deepseek-v4-pro at 100%

### Routing Thresholds
- `{@...@}` markers present → prototype_ner_v3 direct
- No markers, clear entity keywords → spaCy
- spaCy returns "" → FW
- Type mapping layer: eval expects PERSONS vs spaCy PERSON

### Prompt Architecture
- NER prompt critical: CATEGORY: value1, value2; CATEGORY: value3 format
- 1-shot example via custom_instructions
- **Critical:** local 1.5B models CANNOT do `{@entity@}` format NER — always use solver or FW
- FW: "Extract all entities. Group by type. Output format: CATEGORY: v1, v2; CATEGORY: v3"

### GEPA Focus
1. Type mapping (spaCy PERSON→PERSONS, ORG→corporation for tweet format)
2. prototype_ner_v3 integration into pipeline (currently standalone)
3. FW vs solver routing: when to skip solver and go straight to FW
4. Format normalizer for LLM NER output

---

## Cross-Cutting Architecture Decisions

### Routing Priority (for Agent Pipeline)
```
1. Pre-filter (stage0): bypass trivial (greetings, pure arithmetic)
2. Deterministic solver (per category chain above)
3. VADER fast-path (sentiment only)
4. Fact DB (factual only)
5. Zebra/constraint solver (logic only)
6. Sumy extractive (summarization only)
7. NER solver (NER only)
8. Bug pattern solver (code_debug only)
9. Local LLM with per-category model routing
10. FW escalation (per-category conditional)
11. Multi-model routing fallback (use category_model_map)
```

### Per-Category Model Selection
| Category | Best Local Model | FW Model |
|----------|-----------------|----------|
| sentiment | qwen2.5-1.5b-instruct | gpt-oss-120b |
| factual | qwen2.5-1.5b-instruct | gpt-oss-120b |
| logic | qwen2.5-1.5b-instruct | deepseek-v4-pro |
| math | qwen2.5-1.5b-instruct | deepseek-v4-pro |
| code_gen | gemma-3-1b-it or qwen2.5-coder | kimi-k2p7-code |
| code_debug | qwen2.5-coder-1.5b | (FW regresses — avoid) |
| summarization | qwen2.5-1.5b-instruct | gpt-oss-120b |
| ner | qwen2.5-coder-1.5b | deepseek-v4-pro |

### Prompt Parameter Guidelines
| Category | max_tokens | Temperature | top_p | top_k | Notes |
|----------|:----------:|:-----------:|:-----:|:-----:|-------|
| sentiment | 16 | 0.0 | 0.9 | 20 | Single word output |
| factual | 64 | 0.0 | 0.9 | 20 | Short answer, factoid |
| logic | 512 | 0.0 | 0.95 | 40 | CoT needs headroom |
| math | 512 | 0.0 | 0.95 | 40 | CoT, step-by-step |
| code_gen | 512 | 0.0 | 0.95 | 40 | Full function output |
| code_debug | 512 | 0.0 | 0.95 | 40 | Full corrected function |
| summarization | 256 | 0.3 | 0.95 | 40 | Slight diversity for vocabulary |
| ner | 128 | 0.0 | 0.9 | 20 | Structured format output |

### Remaining Gaps per Category (v12d)
| Category | Gap | Size | Fix Priority |
|----------|-----|:----:|:-----------:|
| sentiment | Sarcasm/hedging still missed by 1.5B even with VADER hint | 17% | HIGH |
| factual | FactDB covers only 17K facts — lots of long-tail NQ trivia | 8% | MEDIUM |
| logic | LogiQA argument analysis hard for 1.5B — needs FW routing | 5% | HIGH |
| math | Multi-step word problems with fractions/proportions | 11% | HIGH |
| code_gen | Complex specs with 3+ constraints | <10% | LOW |
| code_debug | FW regression (96% vs 100% local) | -4% | FIX (exclude from FW) |
| summarization | Extractive ceiling on analytical/comparison texts | 8% | MEDIUM |
| ner | Type mapping between solver output and eval format | 0% | LOW (FW covers) |
