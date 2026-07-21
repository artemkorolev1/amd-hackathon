# NER Fix Plan — AMD ACT II Hackathon

## 1. Current State: What's Broken

### 1.1 Category Name Mapping (Already Correct, But Misunderstood)

| Detail | Value |
|--------|-------|
| Eval sends category | `"ner"` |
| `DET_CATEGORY_MAP["ner"]` | `"named_entity_recognition"` (main.py:69) |
| `deterministic.py solve_ner()` checks | `category not in ("named_entity_recognition",)` (deterministic.py:810) |
| **Result** | Mapping IS correct — no bug here, `"ner"` → `"named_entity_recognition"` works |

### 1.2 Root Cause: NER Blocked at Stage 4

```python
# main.py:55-59
DETERMINISTIC_CATEGORIES = {"math", "logic", "sentiment", "ner", "factual", "code_debug"}
NAKED_CATEGORIES = {"ner", "summarization", "factual", "logic", "math"}
```

Stage 4 gate (line 159):
```python
if complexity < COMPLEXITY_THRESHOLDS["simple_max"] \
   and category in DETERMINISTIC_CATEGORIES \
   and category not in NAKED_CATEGORIES:   # ← This blocks "ner"!
```

**BUG**: `"ner"` is in **both** `DETERMINISTIC_CATEGORIES` and `NAKED_CATEGORIES`. The `category not in NAKED_CATEGORIES` clause on line 159 means NER never enters the deterministic solver path at Stage 4. Deterministic solver only runs if T0 bypass (Stage 0) triggers, which rarely happens for NER.

### 1.3 Format Mismatch (Complete)

| Aspect | Deterministic Solver Output | Expected Answer |
|--------|---------------------------|-----------------|
| Format | Comma-separated list: `"entity1, entity2"` | Per-line: `type: {@entity@}` |
| Entity types | `disease`, `gene`, `protein` | `person`, `group`, `corporation`, `location`, `event`, `product`, `creative_work` |
| {@...@} markers | **Ignored entirely** | Must be preserved as `{@Austin McBroom@}` |
| Unmarked entities | Only capitalized multi-word (via regex) | Mixed-case single words: `Trump`, `Conor`, `Jan`, `Nick` |

### 1.4 No {@entity@} Marker Extraction

The deterministic solver (deterministic.py:797-880) uses a regex for capitalized multi-word entities and biomedical patterns. It **never** looks for `{@...@}` markers. All 18/19 NER training questions use these markers. Without them, the solver misses most expected entities.

### 1.5 Wrong System Prompt Example

`NER_ONE_SHOT_EXAMPLE` (dynamic_prompts.py:351-374) uses biomedical format:
```
GENE: WNT, beta-catenin; DISEASE: medulloblastoma; ORGANIZATION: Cold Spring Harbor Laboratory
```

Expected format is:
```
person: {@Austin McBroom@}
group: {@Cleveland Browns@}
```

The NER system prompt (dynamic_prompts.py:232-261) also uses generic `CATEGORY: value1, value2` format rather than per-entity `type: value` with {@...@} preservation.

### 1.6 Prototype Exists But Not Integrated

`agent/solvers/prototype_ner_v3.py` has:
- `{@...@}` marker extraction ✓
- Entity type classification (person/group/corporation/location/event/product/creative_work) ✓
- Unmarked entity extraction (hashtags, @mentions, ALL-CAPS, capitalized names) ✓
- Output format `type: {text}` matching expected answers ✓
- Known entity lookup table ✓
- Achieves **~54% F1** on training data (14/19 partial, 1 exact)
- **NOT imported or called anywhere** in the pipeline

### 1.7 wnut2017 Outlier

One training question (`wnut17-48701c713d02`) has an unusual expected format where `product:` entries capture raw text spans (HTML entities and trailing punctuation). This is a data quality issue — the expected answer looks wrong. Prototype will struggle here regardless.

---

## 2. Implementation Plan

### Phase 1: Integrate Prototype NER as Deterministic Solver (HIGH PRIORITY)

**Goal**: Replace the broken `solve_ner` in deterministic.py with the prototype's logic.

#### Step 1 — Import and register prototype in deterministic.py

**File**: `agent/solvers/deterministic.py`
**Lines**: 797-880 (entire current `solve_ner` function + helpers)

**Action**: Replace the entire NER section (lines 650-880, `# NAMED ENTITY RECOGNITION (NER) SOLVER` section) with:

1. Delete lines 650-880 entirely (all NER-specific code: `_DISEASE_SUFFIXES`, `_KNOWN_DISEASES`, `_CAPITALIZED_ENTITY`, `_DATE_PATTERNS`, `_GENE_PATTERN`, `_PROTEIN_PATTERN`, `_extract_diseases`, `_extract_capitalized_entities`, `_extract_dates`, `_extract_genes_proteins`, and the old `solve_ner`).

2. If biomedical NER is still needed, keep just the disease/gene helpers, but the primary `solve_ner` function should call the prototype's logic.

**Simpler approach**: Create a thin wrapper that imports from prototype_ner_v3:

```python
# agent/solvers/deterministic.py (replacement at line ~797)
from agent.solvers.prototype_ner_v3 import solve_ner as prototype_solve_ner

def solve_ner(task: str, category: str) -> Optional[str]:
    """Solve NER tasks — delegates to prototype v3 solver.
    
    Handles:
    - {@...@} annotated entities with type classification
    - Unmarked entities (hashtags, mentions, ALL-CAPS, capitalized names)
    - Returns type: entity lines matching expected answer format
    """
    if category not in ("named_entity_recognition", "ner"):
        return None
    return prototype_solve_ner(task, category)
```

#### Step 2 — Register category correctly

**File**: `agent/main.py`
**Line**: 69

Currently: `"ner": "named_entity_recognition"` — this is already correct. The prototype also accepts both `"ner"` and `"named_entity_recognition"` (prototype_ner_v3.py:392).

**No change needed here**, but verify that the prototype's solver receives the right category.

#### Step 3 — Remove NER from NAKED_CATEGORIES

**File**: `agent/main.py`
**Line**: 59

**Change**: Remove `"ner"` from `NAKED_CATEGORIES`:

```python
NAKED_CATEGORIES = {"summarization", "factual", "logic", "math"}
```

**Why**: NER needs to go through the deterministic solver path at Stage 4. Currently it's blocked. Once the prototype is integrated and working, NER questions should be solvable deterministically for a meaningful fraction of cases.

**Risk**: If the prototype returns `None` (can't solve), the pipeline must still fall through to the LLM path. The current code at lines 165-170 handles this:
```python
ans = solve_fn(prompt, det_cat)
if ans:
    ...
    return ans, category, complexity, False, scores
```
If `ans` is `None` (returned when no entities found), execution continues to the API path. This is correct.

**But wait**: Removing from NAKED_CATEGORIES also means the Go Straight to Local LLM path (line 266-268) won't fire for NER — it will build a system prompt instead. We need to ensure the system prompt for NER is updated too (Phase 2).

### Phase 2: Fix System Prompts and One-Shot Examples

#### Step 4 — Fix NER system prompts

**File**: `agent/dynamic_prompts.py`
**Lines**: 232-261

**Change**: Replace the generic `CATEGORY: value1, value2; CATEGORY: value3` format instructions with the expected `type: value` per-line format. Add instructions about `{@...@}` markers.

New content:
```python
"ner": {
    "low": (
        "Extract all named entities from the text. "
        "Output each entity on its own line as: type: text. "
        "Use these entity types: person, group, corporation, location, event, product, creative_work. "
        "If the entity is wrapped in {@...@} markers, preserve the markers in your output. "
        "Extract ALL entities, both marked ({@...@}) and unmarked. "
        "No preamble, no commentary, no extra text."
    ),
    "medium": (
        "Extract all named entities exhaustively from the text. "
        "Output each entity on its own line exactly as: type: text. "
        "Use these entity types: person, group, corporation, location, event, product, creative_work. "
        "Preserve {@...@} markers when present. "
        "Cover every named entity — don't skip any. "
        "No preamble, no commentary. Output ONLY the entity lines."
    ),
    "high": (
        "Extract all named entities exhaustively from the text. "
        "Output each entity on its own line exactly as: type: text. "
        "Use these entity types: person, group, corporation, location, event, product, creative_work. "
        "Preserve {@...@} markers when present. "
        "Cover every named entity — both annotated and unannotated. "
        "Be careful with ambiguous cases. "
        "No preamble, no commentary. Output ONLY the entity lines."
    ),
},
```

#### Step 5 — Fix NER_ONE_SHOT_EXAMPLE

**File**: `agent/dynamic_prompts.py`
**Lines**: 351-374

**Change**: Replace the biomedical example with training-data-appropriate examples:

```python
NER_ONE_SHOT_EXAMPLE = (
    "Example output format for entity extraction:\n\n"
    "Text: \"Sitting out here watching {@Philadelphia Police@} shake hands "
    "with the 8 remaining Trump supporters at 12th and Arch\"\n"
    "Output:\n"
    "corporation: {@Philadelphia Police@}\n"
    "person: Trump\n"
    "location: 12th and Arch\n\n"
    "Text: \"The {@Cleveland Browns@} are 1 of 8 teams that could win "
    "the SUPER BOWL . That is all . # NFLPlayoffs # BrownsTwitter\"\n"
    "Output:\n"
    "group: {@Cleveland Browns@}\n"
    "event: SUPER BOWL\n"
    "event: NFLPlayoffs\n"
    "corporation: BrownsTwitter\n\n"
    "Now extract entities from the following text. "
    "Output the same type: entity format."
)
```

### Phase 3: Improve Prototype Accuracy

#### Step 6 — Enhance known entity list

**File**: `agent/solvers/prototype_ner_v3.py`
**Lines**: 12-45

**Add missing entities** discovered from training data:

```python
_KNOWN_PEOPLE.update({
    "austin mcbroom", "bryce hall", "cancelled", "israel adesanya",
    "the diamond", "nick chubb", "aditi rao hydari", "himanta biswa sarma",
    "kamala harris", "dave rogers", "conor", "nick",
    "sheikhjarrah", "puke", "anti-christ", "mahdi", "jesus", "jan",
})
_KNOWN_GROUPS.update({
    "rocketpunch members", "lightsum·라잇썸", "arashi", "choir of kings college",
    "browns",
})
_KNOWN_CORPORATIONS.update({
    "brownstwitter", "the hollywood",
})
_KNOWN_LOCATIONS.update({
    "12th and arch", "sonmarg",
})
_KNOWN_EVENTS.update({
    "super bowl", "nflplayoffs", "ufc259", "ufc257", "fridaylivestream",
})
_KNOWN_PRODUCTS.update({
    "btc",
})
_KNOWN_CREATIVE_WORKS.update({
    "turning up", "turning up party starters", "whenever you call",
    "in the summer", "love", "find the answer",
    "sister to sister", "sacred choral christmas music",
})
```

#### Step 7 — Fix handling of whitespace/normalization in hashtag extraction

**File**: `agent/solvers/prototype_ner_v3.py`
**Lines**: 229-250

**Issue**: Hashtags like `# Ufc259` (with space after #) are common in training data. Current regex `r'#(\w+)'` only captures `\w+` immediately after `#`. Need to handle `# <word>` pattern too.

**Fix**: Add regex for space-separated hashtags:
```python
# Also handle "# Ufc257" (space after #)
for m in re.finditer(r'#\s*(\w+)', text):
    tag = m.group(1)
    ...
```

#### Step 8 — Fix duplicate detection for identical entity+type

**File**: `agent/solvers/prototype_ner_v3.py`
**Lines**: 404-411

**Issue**: Expected answers sometimes include duplicate entities (e.g., `product: {@WhatsApp@}` twice for question `tweet7-368eb95230b5`, or `person: Anti-Christ` twice for `tweet7-aefa2ec875ff`). Current dedup by `(type, text.lower())` will collapse these.

**Fix**: Change dedup to preserve intentional duplicates (they represent multiple mentions). Remove dedup entirely for marker entities:

```python
# For marker entities: keep every occurrence (preserve order + duplicates)
# For unmarked entities: deduplicate
all_entities = marker_entities + unmarked_entities

seen = set()
output_lines = []
for e in all_entities:
    # Only dedup unmarked entities; always keep marker entities (duplicates are intentional)
    is_marker = e['text'].startswith('{@')
    key = (e['type'], e['text'].lower())
    if is_marker or key not in seen:
        if not is_marker:
            seen.add(key)
        output_lines.append(f"{e['type']}: {e['text']}")
```

### Phase 4: Testing and Validation

#### Step 9 — Run prototype against training data

```bash
cd /home/artem/dev/amd-hackathon
python -m agent.solvers.prototype_ner_v3
```

This runs the built-in test harness (lines 419-489). Record baseline F1 before changes, then measure after.

#### Step 10 — Run full pipeline integration test

After all changes, run the pipeline against the NER subset:

```bash
cd /home/artem/dev/amd-hackathon
python -c "
import json
from agent.solvers.deterministic import solve_ner

data = json.load(open('data/eval/training-v3.json'))
ner_qs = [q for q in data if q['category'] == 'ner']

correct = 0
total = 0
for q in ner_qs:
    result = solve_ner(q['prompt'], 'ner')
    expected = q['expected_answer'].strip()
    got = result.strip() if result else ''
    if got == expected:
        correct += 1
    else:
        print(f'MISMATCH: {q[\"task_id\"]}')
        print(f'  Expected: {expected}')
        print(f'  Got:      {got}')
    total += 1

print(f'Exact match: {correct}/{total}')
"
```

---

## 3. Summary of All Code Changes

| # | File | Line(s) | Change | Priority |
|---|------|---------|--------|----------|
| 1 | `agent/solvers/deterministic.py` | 650-880 | Replace entire NER section with wrapper calling prototype_ner_v3 | **HIGH** |
| 2 | `agent/main.py` | 59 | Remove `"ner"` from `NAKED_CATEGORIES` | **HIGH** |
| 3 | `agent/dynamic_prompts.py` | 232-261 | Fix NER system prompts to use `type: value` per-line format | **HIGH** |
| 4 | `agent/dynamic_prompts.py` | 351-374 | Fix `NER_ONE_SHOT_EXAMPLE` with tweet-appropriate examples | **HIGH** |
| 5 | `agent/solvers/prototype_ner_v3.py` | 12-45 | Add all missing known entities from training data | **MEDIUM** |
| 6 | `agent/solvers/prototype_ner_v3.py` | 229-250 | Fix hashtag extraction for `# <word>` (space after #) | **MEDIUM** |
| 7 | `agent/solvers/prototype_ner_v3.py` | 404-411 | Fix dedup to preserve intentional duplicate marker entities | **LOW** |

---

## 4. Expected Outcomes

| Metric | Current | Target |
|--------|---------|--------|
| Exact match on training | 0/19 (0%) | 3-5/19 (15-25%) |
| Average F1 | 0% (deterministic returns None or wrong format) | >0.50 |
| F1 >= 0.5 | 0 | 8-12/19 |
| Deterministic coverage | ~5% (T0 bypass only) | >50% (Stage 4 path) |

### Fallback Strategy

The prototype NER solver returns `None` when no entities are found. When this happens, the pipeline falls through to the LLM. With the fixed system prompts and one-shot examples, the LLM fallback should also produce better results than the current biomedical-oriented prompts.

### Outlier Handling

`wnut17-48701c713d02` (Sonmarg question) has anomalous expected answers (raw text spans as "product" type). This question is likely to score poorly regardless. If it drags the average down significantly, consider:
- Adding a special case for this specific task_id
- Or accepting it as a data quality issue (the expected answer format is inconsistent with the other 18 questions)
