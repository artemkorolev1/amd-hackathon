# GEPA Cycle: NER (named_entity_recognition) — Judge + Analyze

**Date:** 2026-07-14  
**Analyst:** Hermes Agent (GEPA subagent)  
**Files examined:** See full read list at end.

---

## 1. JUDGE — Current State Assessment

### 1.1 Solver Chain (actual runtime vs documented ideal)

| Stage | Architecture Doc (ideal) | Actual Pipeline | Status |
|-------|------------------------|-----------------|--------|
| 1 | prototype_ner_v3 (F1=0.961) → DIRECT | **NOT CALLED** | ❌ BROKEN |
| 2 | spaCy `en_core_web_sm` → DIRECT | imported in `deterministic.py` but **NOT in `det_cat_map`** | ❌ SILENTLY SKIPPED |
| 3 | Local LLM (qwen2.5-coder-1.5b) | Called with wrong prompt format | ⚠️ RUNS BUT WRONG FORMAT |
| 4 | FW (deepseek-v4-pro) | `_fireworks_escalate` has **no NER handler** — falls through to empty | ⚠️ PARTIALLY DEAD |
| 5 | FW fallback via `_fw_fallback` | Only reached if local LLM returns empty | ⚠️ WEAK |

**Critical finding: `prototype_ner_v3` is the highest-accuracy solver (F1=0.961 on training-v3) but is NEVER invoked during normal pipeline execution.** It is imported only in `classifier.py`'s `classify_ner()` standalone function, which the pipeline does not call.

The deterministic `solve_ner()` from `deterministic.py` (spaCy-based) is imported in pipeline.py line 40 but never executed because `ner` is NOT in `det_cat_map` (lines 159-165). The deterministic solver loop at line 794-802 only runs for categories in `det_cat_map`.

### 1.2 Accuracy Metrics

| Source | Metric | Value | Notes |
|--------|--------|-------|-------|
| prototype_ner_v3 (standalone) | Exact match | 2/19 (10.5%) | F1 evaluation is more lenient |
| prototype_ner_v3 (standalone) | Avg F1 | 0.961 | Line-level token overlap |
| Local LLM qwen2.5-coder (eval_ner_all_models) | Fuzzy match | ~10-15% | Wrong format kills accuracy |
| FW (deepseek-v4-pro) | Expected | ~100% | Not confirmed in current wiring |

### 1.3 Format Mismatch: The Core Problem

There are **4 incompatible output formats** in play:

**A) Expected answer format (grader ground truth, training-v3.json):**
```
type: value\n
type: value\n
```
- Lowercase types: `person`, `group`, `corporation`, `location`, `event`, `product`, `creative_work`
- One entity per line
- Entities with `{@...@}` markers preserved (tweetner7 format)
- Duplicates allowed (e.g., `person: Anti-Christ` appears twice in some answers)

**B) prototype_ner_v3 output:**
```
type: value\n
type: value\n
```
- Also lowercase types, one per line ✅ FORMAT-MATCHES grader
- But type labels differ slightly (uses `"group"` vs grader's `"group"`, same)
- Does NOT duplicate entries (dedup by `(type, text.lower())`) ⚠️ may miss duplicates

**C) dynamic_prompts.py NER prompt instructs:**
```
CATEGORY: value1, value2; CATEGORY: value3
```
- Uppercase types, semicolons between categories, comma-separated values
- **TOTALLY WRONG for tweetner7 format** — grader expects one-per-line, lowercase

**D) NER_ONE_SHOT_EXAMPLE:**
```
GENE: WNT, beta-catenin; DISEASE: medulloblastoma; ORGANIZATION: Cold Spring Harbor Laboratory
```
- Also uppercase + semicolons, the exact opposite of what the grader expects
- This example is about biomedical/general NER, NOT the tweetner7 format in the dataset

**E) deterministic.py solve_ner (spaCy) output:**
```
PERSON: Tim Cook, John; ORG: Apple, OpenAI
```
- Uppercase, semicolons — wrong format for tweetner7

### 1.4 Judge QC (judge.py line 251)

The judge validates NER format against this regex:
```python
r"(PERSON|ORG|ORGANIZATION|LOC|LOCATION|DATE|TIME|GPE|EVENT|PRODUCT|NORP|FAC|LAW|LANGUAGE|WORK_OF_ART|MONEY|PERCENT|QUANTITY|CARDINAL)\s*:\s*.+"
```
This only accepts uppercase spaCy-style types — it would REJECT the correct lowercase tweetner7 types (`person`, `group`, `corporation`, etc.). The judge's validation is misaligned with the actual expected answers.

---

## 2. ANALYZE — Root Causes & Proposed Changes

### 2.1 Solver Cascade Ordering

**Problem:** The architecture doc says `prototype_ner_v3 → spaCy → LLM → FW` but the actual pipeline skips the first two stages entirely.

**Proposed changes:**

**a) Wire `prototype_ner_v3` into pipeline.py as the primary deterministic solver for NER:**
- Import `prototype_ner_v3.solve_ner` in pipeline.py
- Add `"ner": "ner"` to `det_cat_map` in the PipelineConfig (line 159-165)
- The solver loop at line 794 will then try `prototype_ner_v3.solve_ner(prompt, "ner")` first, falling through to spaCy's `deterministic.solve_ner()` second

**b) Add NER to `det_cat_map`** — Currently `det_cat_map` maps categories to the category string passed to solvers. Add:
```python
"ner": "ner",
```
This single change makes the deterministic solver loop call `solve_ner(prompt, "ner")` for both prototype_ner_v3 and the spaCy fallback.

**c) Fix execution priority:** The prototype solver should run before spaCy because:
- F1=0.961 vs spaCy's unknown F1 (likely lower on tweetner7 due to lowercase entities, hashtags, `{@...@}` markers spaCy can't see)
- prototype_ner_v3 handles `{@...@}` markers explicitly
- spaCy is a good fallback for entity types prototype misses

**d) Order in `det_solvers`:** Currently `solve_ner` (from deterministic.py) is at index 3 (line 170). If we keep the list order as priority, we need:
```python
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3
# ... in det_solvers:
det_solvers = [solve_ner_v3, solve_arithmetic, solve_logic, solve_sentiment, 
               solve_ner, solve_factual_qa, ...]
```
Or alternatively, change the pipeline to try `prototype_ner_v3.solve_ner` first specifically for NER category.

### 2.2 Prompt Format Fix (Critical)

**Problem:** The system prompt and one-shot example instruct the LLM to output the wrong format. The grader expects:
```
person: {@Austin McBroom@}
person: Jan
```
But the prompt says:
```
CATEGORY: value1, value2; CATEGORY: value3
```

**Proposed changes to `dynamic_prompts.py`:**

**a) Replace the NER prompt templates (lines 232-262):**
```python
"ner": {
    "low": (
        "Extract all named entities from the text. "
        "Output each entity on its own line in the format: type: entity_name. "
        "Use lowercase entity types (person, group, corporation, location, event, "
        "product, creative_work). "
        "Keep {@...@} markers around entities that have them in the original text. "
        "No preamble, no commentary, no extra text."
    ),
    "medium": (
        "Extract ALL named entities from the text exhaustively. "
        "Format per entity on its own line: type: entity. "
        "Use lowercase types: person, group, corporation, location, event, "
        "product, creative_work. "
        "Preserve {@...@} markers exactly as they appear in the text. "
        "Look for entities both inside and outside markers. "
        "Be thorough — include hashtags that are named entities (events, groups). "
        "No preamble, no commentary."
    ),
    "high": (
        "Extract ALL named entities exhaustively. "
        "Output one entity per line: type: entity. "
        "Use precise lowercase types (person, group, corporation, location, "
        "event, product, creative_work). "
        "Preserve {@...@} markers. "
        "Include ALL entities: marked, unmarked, hashtags, @mentions. "
        "Be careful with ambiguous cases. "
        "Output ONLY the entity lines."
    ),
}
```

**b) Replace `NER_ONE_SHOT_EXAMPLE` (lines 351-374):**
```python
NER_ONE_SHOT_EXAMPLE = (
    "Example entity extraction format:\n\n"
    "Text: \"Extract entities: The {@Cleveland Browns@} are 1 of 8 teams that could win "
    "the SUPER BOWL . That is all . # NFLPlayoffs # BrownsTwitter\"\n"
    "Output:\n"
    "group: {@Cleveland Browns@}\n"
    "event: SUPER BOWL\n"
    "event: NFLPlayoffs\n"
    "corporation: BrownsTwitter\n\n"
    "Text: \"Extract entities: I 'm yet to figure out why I should open "
    "{@WhatsApp@} mobile to get {@WhatsApp@} desktop working !\"\n"
    "Output:\n"
    "product: {@WhatsApp@}\n"
    "product: {@WhatsApp@}\n\n"
    "Text: \"Extract entities: Time for the main event . {@Israel Adesanya@} vs Jan # Ufc259\"\n"
    "Output:\n"
    "person: {@Israel Adesanya@}\n"
    "person: Jan\n"
    "event: Ufc259\n\n"
    "Now extract entities from the following text. "
    "Output each entity on its own line: type: entity."
)
```

### 2.3 Format Normalizer for LLM NER Output

Even with a corrected prompt, LLMs will occasionally deviate. Add a post-processing step after local LLM inference for NER:

```python
def normalize_ner_output(raw: str) -> str:
    """Convert various NER output formats to the expected line-per-entity lowercase format."""
    lines = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        # Match "TYPE: value" patterns (handles uppercase, lowercase, mixed)
        m = re.match(r'^\s*([A-Za-z_]+)\s*:\s*(.+)\s*$', line)
        if m:
            etype = m.group(1).lower()
            evalue = m.group(2).strip()
            # Map spaCy types to tweetner7 types
            type_map = {
                'person': 'person', 'people': 'person',
                'org': 'corporation', 'organization': 'corporation', 'organizations': 'corporation',
                'gpe': 'location', 'loc': 'location', 'location': 'location',
                'date': 'date', 'time': 'time',
                'event': 'event', 'product': 'product',
                'work_of_art': 'creative_work', 'creative_work': 'creative_work',
                'norp': 'group', 'fac': 'location', 'law': 'product',
                'money': 'money', 'percent': 'percent', 'percentage': 'percent',
                'quantity': 'quantity', 'cardinal': 'quantity',
                'disease': 'disease', 'gene': 'gene', 'protein': 'protein',
                'drug': 'product', 'legislation': 'product', 'ticker': 'product',
                'corporation': 'corporation', 'group': 'group',
            }
            normalized_type = type_map.get(etype, etype)
            lines.append(f"{normalized_type}: {evalue}")
        else:
            # Try to salvage: split by semicolons, then by colon
            for part in line.split(';'):
                part = part.strip()
                m2 = re.match(r'^\s*([A-Za-z_]+)\s*:\s*(.+)\s*$', part)
                if m2:
                    etype = m2.group(1).lower()
                    evalue = m2.group(2).strip()
                    normalized_type = type_map.get(etype, etype)
                    for val in evalue.split(','):
                        val = val.strip()
                        if val:
                            lines.append(f"{normalized_type}: {val}")
    return '\n'.join(lines)
```

**Integration point:** Add this in `pipeline.py` process() after local LLM inference (around line 843), right before the NER answer is returned.

### 2.4 Fireworks Escalation for NER

**Problem:** `_fireworks_escalate` has no NER handler. NER is in `FIREWORKS_CATEGORIES` in main.py but the Pipeline class ignores it.

**Proposed change to `_fireworks_escalate`:** Add an NER block:
```python
elif category in ("ner", "named_entity_recognition"):
    cfg = _fw_route("ner", prompt, complexity)
    if not self._check_allowed_model(cfg.model_id):
        return ""
    answer = self._fw.solve(
        cfg.model_id, cfg.system_prompt, prompt,
        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        prefill=cfg.prefill, task_type="ner",
        timeout=self.cfg.fireworks_timeout_s,
    )
    if answer:
        return answer
```

The FW system prompt should also use the corrected format:
```
System: Extract all named entities. Output one per line: type: entity. 
Use lowercase types (person, group, corporation, location, event, product, creative_work).
Preserve {@...@} markers.
```

### 2.5 Judge QC Format Validation Fix

**Problem:** `judge.py` line 251 validates NER against uppercase spaCy types, but expected answers use lowercase tweetner7 types.

**Fix:** Update the validation regex to accept lowercase types:
```python
ep = re.compile(
    r"(person|group|corporation|location|event|product|creative_work|"
    r"disease|gene|protein|date|time|money|percent|"
    r"PERSON|ORG|ORGANIZATION|LOC|LOCATION|DATE|TIME|GPE|EVENT|PRODUCT|NORP|FAC|LAW|"
    r"LANGUAGE|WORK_OF_ART|MONEY|PERCENT|QUANTITY|CARDINAL)\s*:\s*.+", 
    re.IGNORECASE
)
```

### 2.6 FW Routing — Consider Removing NER from FIREWORKS_CATEGORIES

The architecture doc says prototype_ner_v3 achieves F1=0.961 on the training set. If this generalizes to the eval set, NER may not need FW at all. Consider:
1. Wire prototype_ner_v3 as the primary solver
2. If it returns non-None → DIRECT (no LLM, no FW)
3. Only go to FW if prototype returns None AND spaCy returns None
4. This saves 30s+ per NER question

However, prototype_ner_v3 uses a static knowledge base — it doesn't handle truly novel entities well. FW is still a safety net for unseen entity types.

### 2.7 prototype_ner_v3 Quality Issues

Looking at the expected answers for training-v3:

| Shortcoming | Example | Impact |
|------------|---------|--------|
| Dedup removes required duplicates | `person: Anti-Christ` expected twice but deduped once | F1 penalty |
| Some entity types wrong | `BrownsTwitter` → `corporation` (correct), but `Browns` → `group` (correct) | OK |
| Misses some lowercase unmarked entities | `person: Jan`, `person: Conor`, `person: Nick` | Need `_extract_unmarked_entities` improvements |
| Handles wnut17 poorly | `product: & gt ; * The soldier was killed...` — nonsensical ground truth | Low priority (outlier format) |
| Missing `{@CZ 🔶 Binance@}` (emoji in entity) | Emoji-containing entities mishandled | Possibly due to regex issues |

---

## 3. RECOMMENDED CHANGES (Priority Ordered)

| # | Change | File(s) | Effort | Impact |
|---|--------|---------|--------|--------|
| **P0** | Fix NER prompt format from semicolons/uppercase to line-per-entity/lowercase | `dynamic_prompts.py:232-262` | 15 min | HIGH — solves format mismatch |
| **P0** | Fix NER_ONE_SHOT_EXAMPLE to match actual expected format | `dynamic_prompts.py:351-374` | 15 min | HIGH — one-shot drives LLM output |
| **P0** | Add `"ner": "ner"` to `det_cat_map` in Pipeline | `pipeline.py:159` | 1 line | HIGH — enables solver cascade |
| **P1** | Wire `prototype_ner_v3.solve_ner` into pipeline as primary NER solver | `pipeline.py` (import + ordering) | 10 min | HIGH — enables F1=0.961 solver |
| **P1** | Add format normalizer for LLM NER output | `pipeline.py` or new `solvers/ner_normalizer.py` | 30 min | MEDIUM — catches LLM format drift |
| **P2** | Add NER handler to `_fireworks_escalate` | `pipeline.py:535` | 15 min | MEDIUM — enables FW for NER |
| **P2** | Fix judge.py NER validation regex | `judge.py:251` | 1 line | MEDIUM — QC alignment |
| **P3** | Remove dedup in prototype_ner_v3 for required duplicates | `prototype_ner_v3.py:519-526` | 10 min | LOW — edge case |
| **P3** | Improve unmarked lowercase entity detection (Jan, Conor, Nick) | `prototype_ner_v3.py:_extract_unmarked_entities` | 30 min | MEDIUM — catches missing entities |

### Quick Implementation Order

1. **Fix `dynamic_prompts.py`** — new prompt templates + one-shot example (takes 15 min, fixes the biggest problem)
2. **Add `"ner": "ner"` to `det_cat_map`** — enables solver cascade (1 line)
3. **Import + wire `prototype_ner_v3`** — primary solver runs first
4. **Add NER handler to `_fireworks_escalate`** — secondary safety net
5. **Add format normalizer** — post-process LLM output
6. **Fix judge.py** — align QC validation

---

**Files read:**
1. `/home/artem/dev/amd-hackathon/references/per-category-architecture-v12d.md` (section 8, lines 344-443)
2. `/home/artem/dev/amd-hackathon/agent/pipeline.py` (full, 925 lines)
3. `/home/artem/dev/amd-hackathon/agent/solvers/prototype_ner_v3.py` (full, 609 lines)
4. `/home/artem/dev/amd-hackathon/agent/solvers/deterministic.py` (NER section, lines 1918-2185)
5. `/home/artem/dev/amd-hackathon/agent/dynamic_prompts.py` (NER prompt section, lines 230-262 + 351-374 + 640-683)
6. `/home/artem/dev/amd-hackathon/agent/classifier.py` (full, 159 lines)
7. `/home/artem/dev/amd-hackathon/agent/judge.py` (NER validation, lines 250-257)
8. `/home/artem/dev/amd-hackathon/agent/main.py` (NER routing, lines 34-299)
9. `/home/artem/dev/amd-hackathon/data/eval/training-v3.json` (19 NER samples with expected answers)
10. `/home/artem/dev/amd-hackathon/scripts/eval/eval_ner_all_models.py` (NER eval script)
11. `/home/artem/dev/amd-hackathon/scripts/grade_answer.py` (fuzzy_match cascade)
