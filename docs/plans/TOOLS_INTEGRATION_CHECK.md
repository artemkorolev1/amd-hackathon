# Tools Integration Check: Staging Worker Path vs Agent Solvers

## 1. Which deterministic solver functions does `DetWorker` actually import and use?

`DetWorker.initialize()` (in `staging/workers/det_worker.py`, lines 32-38) imports exactly **5 functions** from `agent.solvers.deterministic`:

| Imported function       | Category key       | Used in process()? |
|-------------------------|--------------------|--------------------|
| `solve_arithmetic`      | `"math"`           | Yes (line 61)      |
| `solve_factual_qa`      | `"factual"`        | Yes                |
| `solve_sentiment`       | `"sentiment"`      | Yes                |
| `solve_summarization`   | `"summarization"`  | Yes                |
| `solve_code_debugging`  | `"code_debug"`     | Yes                |

The import is wrapped in a try/except that silently disables all solvers on `ImportError`. These 5 are called directly as `solver(task.prompt, task.category)` — **not** through any tool registry or intermediate dispatch layer.

## 2. Which solver functions exist in `deterministic.py` but are NOT used by `DetWorker`?

`deterministic.py` defines **5 additional solver functions** that are **never imported** by `DetWorker`:

| Unused function               | Line | Purpose                                    |
|-------------------------------|------|--------------------------------------------|
| `solve_truth_teller_liar`     | 1009 | Knights-and-knaves type puzzles            |
| `solve_number_sequence`       | 1157 | Next-in-sequence puzzles                   |
| `solve_logic`                 | 1264 | General logic (internally calls the two above) |
| `solve_ner`                   | 1761 | Named-entity recognition (requires spaCy)  |
| `solve_code_generation`       | 2574 | Template-based code generation             |

**Note on `solve_logic`:** Although it internally calls `solve_truth_teller_liar` and `solve_number_sequence` as sub-routines, `solve_logic` itself is not imported by `DetWorker`, so those sub-routines are also unreachable from the staging pipeline.

## 3. Is `tool_registry.py` imported by any staging code path?

**No.** Zero imports of `tool_registry` exist anywhere under `staging/`.

The only imports of `tool_registry` are within `agent/solvers/`:

| File                        | Line | Usage                                    |
|-----------------------------|------|------------------------------------------|
| `agent/solvers/eval_tools.py`  | 129  | Lazy import: `from agent.solvers.tool_registry import registry` |
| `agent/solvers/text_processor.py` | 362, 571 | Direct import of `tool` and `registry` |

These files are **internal to the agent module** and are not invoked by the staging worker pipeline. The `tool_registry.py` file and its 28 registered tools are **completely disconnected** from the staging pipeline.

## 4. Are there any tools the judge needs (e.g., Fireworks escalation) that wouldn't work because the import chain is broken?

**The judge's Fireworks escalation is fully wired.** The judge (`staging/ready_judge.py`, lines 416-417) imports directly:

```python
from agent.solvers.fireworks import FireworksSolver
from agent.solvers.fw_router import route
```

This is parallel to `FwWorker` (`staging/workers/fw_worker.py`, lines 32, 45, 85) which uses the same direct imports. There is **no broken chain** here — both import paths work independently of `tool_registry.py`.

**Caveat:** The judge's escalation relies on `answers[0].get("category")` and `answers[0].get("prompt")` — these fields are attached by `ReadyWorker._push_results()` (line 237-239). If a worker doesn't set these (e.g., from an older or custom worker), the escalation silently returns `""`.

## 5. What's the actual set of tools available end-to-end vs. the theoretical 28?

### Theoretical 28 (from `tool_registry.py`)

The registry declares 28 tools across 6 categories, each wrapping a `solve_*` function via the `@tool` decorator:

**Existing (6):** `factual_qa`, `sentiment_analysis`, `summarize`, `math_solve`, `ner_extract`, `format_python`
**Logic (5):** `solve_logic_puzzle`, `solve_syllogism`, `solve_truth_teller_liar`, `solve_number_sequence`, `solve_logical_reasoning`
**Code (3):** `execute_code_safe`, `code_debug`, `code_gen_templates`
**Spell (2):** `spell_check`, `list_misspellings`
**Web (2):** `search_web`, `search_factual`
**Fun (10):** `format_csv`, `text_stats`, `reverse_text`, `top_words`, `to_leetspeak`, `is_palindrome`, `days_until_april_fools`, `weather_hot_take`, `to_emoji`, `flip_coin`

### Actual end-to-end (via staging pipeline)

**Only 5 solver functions are wired in DetWorker**, mapping to 5 category keys:

| Category key     | Function            | Route                                    |
|------------------|---------------------|------------------------------------------|
| `"math"`         | `solve_arithmetic`  | Direct call from DetWorker.process()     |
| `"factual"`      | `solve_factual_qa`  | Direct call from DetWorker.process()     |
| `"sentiment"`    | `solve_sentiment`   | Direct call from DetWorker.process()     |
| `"summarization"`| `solve_summarization`| Direct call from DetWorker.process()    |
| `"code_debug"`   | `solve_code_debugging`| Direct call from DetWorker.process()   |

**Summary:** **5 of 10** deterministic solver functions are integrated. **0 of 28** tool_registry tools are integrated. The **tool_registry itself is an unused artifact** from the staging pipeline's perspective.

## Gap Analysis

| Component                    | Staging integration | Status       |
|------------------------------|---------------------|--------------|
| `tool_registry.py` (28 tools) | Not imported       | 🔴 Disconnected |
| `solve_ner`                  | Not imported       | 🔴 Missing    |
| `solve_code_generation`      | Not imported       | 🔴 Missing    |
| `solve_truth_teller_liar`    | Not imported       | 🔴 Missing    |
| `solve_number_sequence`      | Not imported       | 🔴 Missing    |
| `solve_logic`                | Not imported       | 🔴 Missing    |
| Everything from `logic_solver.py` | Not imported   | 🔴 Missing    |
| Everything from `code_sandbox.py` | Not imported   | 🔴 Missing    |
| Everything from `spell_check.py` | Not imported   | 🔴 Missing    |
| Everything from `web_search.py` | Not imported   | 🔴 Missing    |
| Everything from `easter_egg_shelf.py` | Not imported | 🔴 Missing  |
| Judge Fireworks escalation   | Direct import      | ✅ Wired      |
| `FwWorker` + `fw_router`     | Direct import      | ✅ Wired      |
| `LocWorker` + local model    | Direct import      | ✅ Wired      |

**Recommendation:** If the 28 tools in `tool_registry.py` are meant to be the canonical toolset, the staging pipeline needs either (a) DetWorker should import and dispatch through the registry (instead of directly importing 5 functions), or (b) the registry should be refactored into the dispatch layer used by `ready_judge.py` or the workers. As-is, `tool_registry.py` is a parallel universe with no wires to the staging system.
