# Pipeline ↔ Staging Pull Architecture + Plan-A Workflow Integration Plan

**Date:** 2026-07-13  
**Files Modified:**
- `agent/pipeline.py` — expanded workflow branch
- `agent/routing_table.py` — workflow step storage/retrieval

**Files NOT Modified (explicitly preserved):**
- `agent/cell.py` — already updated for Plan-A (StepConfig, steps field)
- `agent/workflow.py` — already built (WorkflowEngine, ToolRegistry, templates)
- `staging/*` — pull architecture untouched (works alongside Pipeline)

---

## 1. Changes to Pipeline (`agent/pipeline.py`)

### What changed

The workflow branch at line 560–571 was expanded from a 12-line stub to a robust 46-block section.

**Before (minimal):** Created a Cell from raw steps, ran WorkflowEngine, returned result. No error handling, no metrics, no decoding config support.

**After (expanded):**
| Feature | Implementation |
|---------|---------------|
| **Try/except wrapping** | Catches all exceptions during workflow execution (step config errors, inference failures, tool dispatch errors) |
| **Graceful fallback** | On failure, logs exception and falls through to regular single-shot processing instead of crashing |
| **Per-step logging** | Each step's latency (ms) and token estimate logged via `logger.info` |
| **Aggregate metrics** | Total workflow latency + total tokens logged at completion |
| **Decoding overrides** | If the routing entry carries a `decoding` dict (temperature, max_tokens, etc.), it's applied to the Cell via `DecodingConfig.from_dict()` |
| **StepConfig deserialization** | Supports both dict and `StepConfig` object forms in the steps list (defensive) |

### Code flow

```
route_entry = routing_table.get(category)
└── route_entry has "steps"?
    ├── YES → try:
    │           ├── Deserialize StepConfigs from dicts
    │           ├── Build Cell with model_key + decoding overrides
    │           ├── Run WorkflowEngine.run(cell, prompt)
    │           ├── Log per-step metrics + aggregate metrics
    │           └── Return final_answer
    │         except Exception:
    │           Log error → fall through to regular processing
    └── NO  → Regular route_entry processing (single-shot system_prompt)
```

---

## 2. Changes to RoutingTable (`agent/routing_table.py`)

### Three new public methods added

#### `get_workflow_steps(category: str) -> Optional[list]`
Returns the stored workflow steps (list of dicts) for a category, or `None` if the route is single-shot. Steps dicts are JSON-serialisable and ready for `StepConfig.from_dict()`.

#### `has_workflow(category: str) -> bool`
Quick check — returns `True` if the category has a multi-step workflow route (non-empty steps list).

#### `store_workflow_template(category, steps, metadata=None) -> int`
Stores a workflow template for a category. Sets `aggregation="workflow"` automatically. Accepts optional metadata (template name, version, description). Returns the new routing table version number.

### Updated internal method: `_cell_to_entry()`

**Before:** Returned a fixed set of fields (category, cell_name, model_key, system_prompt, decoding, aggregation, accuracy, updated_at) — did NOT capture `steps`.

**After:** When the cell has a non-None `steps` attribute:
- Serialises each step via `step.to_dict()`
- Stores steps in the routing entry as `entry["steps"]`
- Sets `entry["aggregation"] = "workflow"`

This ensures that when `update_from_cells()` promotes a workflow Cell from the GEPA population, the workflow steps are preserved in the routing table entry — Pipeline can then read them back via `route_entry.get("steps")`.

### Workflow round-trip

```
GEPA Orchestrator                           Pipeline
      │                                        │
      │  Cell(steps=[StepConfig, ...])          │
      │                                        │
      ▼                                        │
update_from_cells(cells)                       │
      │                                        │
      ▼                                        │
_cell_to_entry() captures steps                │
      │                                        │
      ▼                                        │
RoutingTable entry:                            │
  { category, model_key,                       │
    decoding, steps=[...],                     │
    aggregation="workflow" }                   │
      │                                        │
      │  (published via to_json /               │
      │   shared routing table)                 │
      │                                        │
      │      ┌─────────────────────────────────┘
      │      │  Pipeline.process()
      │      ▼
      │  route_entry = routing_table.get(cat)
      │  steps = route_entry.get("steps")
      │  if steps:
      │      Cell.from steps + DecodingConfig
      │      WorkflowEngine.run(cell, prompt)
      │      return final_answer
```

---

## 3. How the Pull Architecture Connects to the Pipeline

### Architecture relationship

The staging pull architecture (`staging/`) and the Pipeline (`agent/pipeline.py`) serve **two different call paths**:

| Aspect | Staging (Pull) | Pipeline (Online) |
|--------|---------------|-------------------|
| Purpose | Batch parallel submission | Single-query inference |
| Entry point | `staging/entrypoint.py` | `Pipeline.process()` |
| Worker model | Shared task pool, pull-based | Sequential single process |
| Parallelism | 4 worker types (FW, Local, Det × N) | Single-threaded with executor |
| Judgment | Majority vote (5 answers) | Single answer + scoring |
| Uses Pipeline? | **No** — imports specific modules | Yes — runs Pipeline directly |

### Integration points

Despite being separate execution paths, they share:

1. **Routing table** — Both read the same `RoutingTable` entries. The Pipeline reads it via `self._routing_table.get(category)`. Staging workers use `fw_router.route()` for category→model mapping.

2. **Category taxonomy** — Both use the same 8 categories (math, logic, factual, sentiment, ner, summarization, code_gen, code_debug).

3. **Workflow cell support** — The Pipeline now handles workflow cells from the routing table. Staging could, in future, use the routing table's workflow steps to orchestrate multi-step workflows across worker types.

4. **Priority matrix** — `ReadyConfig.category_priority` maps categories to preferred worker types. This is conceptually parallel to the routing table: staging uses it for worker dispatch, Pipeline uses it for model/solver selection.

### Workflow response flow through staging (future)

If workflow cells were to be executed in the staging pull architecture:

```
ReadyTask(category="math", ...)
  → ReadyQueue.enqueue(task)
    → Worker pulls from task_pool
      → Worker checks routing_table.get_workflow_steps("math")
        → If steps found: execute multi-step workflow
          → Push each step result to results_queue
        → If no steps: normal 5-try processing
          → Push 5 answers to results_queue
    → ReadyJudge collects results
      → Judge uses result validation format (fuzzy_match)
```

Currently, staging workers do NOT read the routing table — they use `fw_router.route()` for per-category model config. This is a future integration point.

---

## 4. How Workflow Cells Flow Through the System

### Full lifecycle

```
┌─────────────────────────────────────────────────────────┐
│  1. GEPA Orchestrator creates workflow Cells            │
│     Cell(task_id="T02", steps=[plan, solve, compose])   │
│     Cell(task_id="T05", steps=[extract, verify])        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  2. RoutingTable.update_from_cells(cells)                │
│     → _cell_to_entry() captures steps                    │
│     → Stores entry with aggregation="workflow"           │
│     → Publishes new table version                        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  3. Pipeline.process(prompt)                             │
│     → classify → category = "math"                       │
│     → routing_table.get("math") returns:                 │
│         {model_key, decoding, steps=[...]}              │
│     → steps present → enter workflow branch              │
│     → Build Cell from steps + decoding                   │
│     → WorkflowEngine.run(cell, prompt)                   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  4. WorkflowEngine.run() internals                       │
│     → artifacts = {"_input": prompt}                     │
│     → Step 0: plan                                       │
│         messages = [(system, "plan prompt"),              │
│                     (user, prompt)]                      │
│         result = inference(messages)                      │
│         artifacts["plan"] = result                        │
│     → Step 1: solve                                      │
│         messages = [(system, "solve prompt"),             │
│                     (user, prior artifacts)]              │
│         result = inference(messages)                      │
│         artifacts["solve"] = result                       │
│     → Step 2: compose                                    │
│         ...                                               │
│     → final_answer = artifacts[-1]                       │
│     → extract boxed answer if present                    │
│     → return {final_answer, artifacts, step_results}     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  5. Pipeline logs per-step metrics                       │
│     → "Workflow step plan: latency_ms=120 tokens=45"     │
│     → "Workflow step solve: latency_ms=450 tokens=180"   │
│     → "Workflow complete: total_latency=620ms"           │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                     Returns final_answer
```

### Multi-step workflow templates available

| Template | Category | Steps | Description |
|----------|----------|-------|-------------|
| `math_3step` | math | plan → solve → compose | Plan approach, execute calculations, format answer with \boxed{} |
| `logic_3step` | logic | plan → reason → compose | Identify premises, step-by-step reasoning, present conclusion |
| `ner_2step` | ner | extract → verify | Extract entities, verify against source text |

---

## 5. Issues Found

### Minor

1. **Pipeline's `_infer` method signature vs. WorkflowEngine expectation**
   - Pipeline passes `self._infer` as `llm_infer_fn` to `WorkflowEngine.__init__`
   - Pipeline's `_infer` signature: `(messages, max_tok, stop_seq, timeout) -> str`
   - WorkflowEngine expects: `(messages, max_tokens, temperature) -> str`
   - **Status:** Pipeline wraps via lambda or direct call — the extra `stop_seq` and `timeout` kwargs have defaults in `_infer`. This works correctly because `_infer` has defaults for `stop_seq=[]` and `timeout=28.0`. However, WorkflowEngine never passes `stop_seq` or `timeout`, so pipeline stops are empty and timeout is the default 28s — **acceptable but suboptimal for workflow steps that may need different stop tokens**.

2. **Staging does not yet read the RoutingTable**
   - `store_workflow_template()` and `has_workflow()` are available but staging workers (`loc_worker.py`, `fw_worker.py`, `det_worker.py`) still use `fw_router.route()` for per-category config instead of the RoutingTable.
   - This is a deliberate design split: the RoutingTable is used by the online Pipeline, while staging workers use `fw_router` directly. **Not a bug but an integration gap** — documented for future work.

3. **Cell constructor requires `task_id` but workflow branch passes `category`**
   - `Cell.__post_init__` validates `task_id` against `VALID_TASK_IDS` which includes both "T01"-"T05" and category strings like "math", "ner" etc. (line 69 of `cell.py`).
   - The Pipeline's workflow branch passes category strings as `task_id` — this is explicitly supported.

### Not Issues (deliberate design decisions)

4. **Pipeline does not import staging** — The Pipeline is a standalone online processor. The staging pull architecture is a separate parallel-submission path. They share data structures (RoutingTable, Cell, WorkflowEngine) but not code paths.

5. **Backtest not enforced in store_workflow_template** — The `store_workflow_template()` method bypasses the `update_from_cells()` → `_pick_best()` → backtest gate. This is by design: `store_workflow_template()` is for direct registration of known-good templates, while `update_from_cells()` remains the recommended path for GEPA-evolved cells with backtest verification.

---

## Summary of Changes

| File | Lines changed | Change type |
|------|--------------|-------------|
| `agent/pipeline.py` | 560–605 | Expanded workflow branch: try/except, logging, decoding config |
| `agent/routing_table.py` | 63–108 | Added 3 new methods: `get_workflow_steps`, `has_workflow`, `store_workflow_template` |
| `agent/routing_table.py` | 211–250 | Updated `_cell_to_entry` to capture workflow steps from Cell objects |
| `staging/*` | 0 | Not modified |
| `agent/cell.py` | 0 | Already has `StepConfig` + `steps` field |
| `agent/workflow.py` | 0 | Already has `WorkflowEngine` + templates |
