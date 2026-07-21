# Integration Audit: amd-hackathon

> Generated: 2026-07-13 (Re-audit)
> Scope: All source files in `/home/artem/dev/amd-hackathon/` excluding `.venv/`, `__pycache__/`, `.git/`
> Constraint: `agent/` module is **UNTOUCHABLE**. Everything else can be modified.

---

## 1. Module Inventory

### 1.1 agent/ — Core Pipeline (UNTOUCHABLE)

| File | Lines | Status | Dependencies (imports from) | Tests |
|------|-------|--------|----------------------------|-------|
| `pipeline.py` | 810 | ✅ Active | agent.workflow, agent.dynamic_prompts, agent.category_filter, agent.complexity, agent.pre_filter, agent.solvers.deterministic, agent.solvers.fireworks, agent.solvers.fw_router, agent.solvers.local_vote, agent.solvers.verify | None |
| `__init__.py` | 27 | ✅ Active | agent.pipeline, agent.cell, agent.experiment_logger, agent.routing_table, agent.evaluation_agent, agent.mutation_agent, agent.analysis_agent, agent.orchestrator, agent.workflow | None |
| `main.py` | 444 | ✅ Active | agent.config, agent.solvers.deterministic, agent.dynamic_prompts, agent.solvers.fireworks, agent.solvers.local_vote, agent.solvers.verify, agent.pre_filter, agent.category_filter, agent.complexity_filter, agent.quality_config, agent.circuit_breaker | None |
| `config.py` | 70 | ✅ Active | stdlib only | None |
| `classifier.py` | 120 | ✅ Active | agent.category_filter (via importlib), **agent.secondary_code** (via importlib), **agent.secondary_factual** (via importlib), **agent.secondary_reasoning** (via importlib), agent.solvers.prototype_ner_v3, agent.solvers.deterministic | None |
| `category_filter.py` | 1058 | ✅ Active | stdlib + re only | None |
| `ml_classifier.py` | 172 | ⚠️ Standalone | sklearn, numpy, agent.category_filter (via importlib) | None |
| `dynamic_prompts.py` | 732 | ✅ Active | stdlib + re only | `tests/test_dynamic_prompts.py` (280 lines) |
| `complexity.py` | 135 | ✅ Active | sentence-transformers, sklearn | None |
| `complexity_filter.py` | 899 | ✅ Active | json | None |
| `pre_filter.py` | 209 | ✅ Active | stdlib + re only | None |
| `workflow.py` | 255 | ⚠️ GEPA only | agent.cell | None |
| `cell.py` | 320 | ⚠️ GEPA only | stdlib only | None |
| `answer_cleaner.py` | 58 | ✅ Active | stdlib + re only | None |
| `circuit_breaker.py` | 175 | ✅ Active | stdlib only | None |
| `quality_config.py` | 60 | ✅ Active | None (data only) | None |
| `summarization.py` | 196 | ⚠️ Active (not main pipeline) | stdlib + urllib | None |
| `caveman_prompts.py` | 78 | 🗑️ Orphaned | Not imported by any module | None |
| `bitmorphic_classifier.py` | 254 | 🗑️ Orphaned (known bugs) | Not imported by any module | None |
| `category_router.py` | 407 | 🗑️ Orphaned | Not imported by any module | None |
| `hierarchical_classifier.py` | 89 | 🗑️ Orphaned | Not imported by any module | None |

> **⚠️ Broken import chain in `agent/solvers/__init__.py`:** This file does `from agent.solvers import logic_solver`, which triggers `from constraint import Problem` (python-constraint package). If python-constraint is missing or wrong version, ANY `import agent.solvers` or `from agent.solvers import ...` will fail at import time. The main pipeline avoids this by importing solver modules directly (`agent.solvers.deterministic`, etc.) rather than through `agent.solvers.__init__`, so normal usage is safe. But `agent/__init__.py` does NOT import `agent.solvers` — the chain is broken transitively if someone does `import agent.solvers`.

| `secondary_code.py` | 198 | ✅ Active (via classifier.py) | stdlib + re only | None |
| `secondary_factual.py` | 723 | ✅ Active (via classifier.py) | stdlib + re only | None |
| `secondary_reasoning.py` | 452 | ✅ Active (via classifier.py) | stdlib + re only | None |

#### classifier.py — Classification Pipeline Chain

`agent/classifier.py` implements a **two-stage classification pipeline** that chains the primary 8-way scorer with three targeted secondary disambiguation classifiers:

```
classify(prompt) → 8-way primary (category_filter.classify_with_detail)
                     │
                     ├── if(code_debug|code_gen) → secondary_code.resolve_code()  → correct debug vs gen
                     ├── if(logic|math)          → secondary_reasoning.resolve_reasoning() → correct logic vs math
                     └── always                  → secondary_factual.resolve_factual() → correct factual↔logic/math
                     │
                     └── return (category, method, confidence)
```

| Stage | Module | Method | What It Does |
|-------|--------|--------|-------------|
| **Primary** | `agent.category_filter` (via importlib) | `classify_with_detail(prompt)` → `{category, confidence, ...}` | 8-way heuristic scoring (code_debug, code_gen, factual, logic, math, ner, sentiment, summarization) |
| **Secondary: Code** | `agent.secondary_code` (via importlib) | `resolve_code(category, prompt)` → corrected category | Regex-based scoring to disambiguate code_debug vs code_gen when primary is uncertain |
| **Secondary: Reasoning** | `agent.secondary_reasoning` (via importlib) | `resolve_reasoning(category, prompt)` → corrected category | Heuristic scoring for logic vs math confusion (especially logic puzzles with numbers) |
| **Secondary: Factual** | `agent.secondary_factual` (via importlib) | `resolve_factual(category, prompt)` → corrected category | Detects SQuAD/MMLU-style factual QA vs logic/reasoning prompts (113 logic→factual and 96 math→factual errors) |

All three secondary classifiers are **pure deterministic** (stdlib + re only) — zero model calls, zero external dependencies. They are loaded lazily via `importlib` to avoid circular imports and minimize startup cost.

Additionally, `classifier.py` exposes `classify_ner(prompt)` which runs the NER solver pipeline separately (used by `agent.solvers.prototype_ner_v3` with fallback to `agent.solvers.deterministic.solve_ner`).

#### agent/solvers/

| File | Lines | Status | Dependencies (imports from) | Tests |
|------|-------|--------|----------------------------|-------|
| `__init__.py` | 50 | ✅ Active | agent.solvers.logic_solver, agent.solvers.code_sandbox, agent.solvers.easter_egg_shelf, agent.solvers.spell_check, agent.solvers.web_search | None |
| `deterministic.py` | 2881 | ✅ Active | agent.solvers.tools, vaderSentiment (lazy), spacy (lazy), agent.solvers.fact_db (lazy) | None |
| `fireworks.py` | 149 | ✅ Active | agent.config, agent.answer_cleaner | None |
| `fw_router.py` | 192 | ✅ Active | stdlib only | None |
| `local.py` | 212 | ⚠️ Standalone (not in main pipeline) | agent.config, agent.answer_cleaner, agent.solvers.tools | None |
| `local_vote.py` | 205 | ✅ Active | stdlib only | None |
| `verify.py` | 462 | ✅ Active | stdlib + subprocess (black/ruff) | None |
| `tools.py` | 311 | ✅ Active | sympy | None |
| `tool_registry.py` | 367 | ✅ Active | stdlib only; lazy-imports from agent.solvers.deterministic, fact_db, logic_solver, logic_reasoning, code_sandbox, verify, spell_check, web_search, easter_egg_shelf (28 tools registered) | None |
| `code_sandbox.py` | 216 | ⚠️ Standalone | RestrictedPython | None |
| `logic_solver.py` | 384 | ⚠️ Standalone | python-constraint | None |
| `logic_reasoning.py` | 582 | ✅ Active | stdlib only (used by tool_registry for logical_reasoning tool) | None |
| `deterministic_filters.py` | 248 | ⚠️ Standalone | stdlib + re only | None |
| `eval_tools.py` | 491 | ⚠️ Standalone | json, agent.solvers.tool_registry, agent.solvers.deterministic_filters | None |
| `spell_check.py` | 140 | ⚠️ Standalone | symspellpy (lazy) | None |
| `web_search.py` | 131 | ⚠️ Standalone | duckduckgo-search (lazy) | None |
| `easter_egg_shelf.py` | 378 | ⚠️ Standalone | stdlib only | None |
| `text_processor.py` | 586 | 🗑️ Orphaned | stdlib + re only; imports agent.solvers.tool_registry | None |
| `fact_db.py` | 207 | ✅ Active (via deterministic.py, tool_registry) | sqlite3 | None |
| `prototype_ner_v3.py` | 609 | ⚠️ Active (via classifier.py) | stdlib + spacy | None |
| `prototype_ner_v2.py` | 372 | 🗑️ Orphaned | Not imported | None |
| `prototype_ner_solver.py` | 338 | 🗑️ Orphaned | Not imported | None |
| `prototype_zebra_solver.py` | 441 | 🗑️ Orphaned | Not imported | None |
| `prototype_zebra_v2.py` | 155 | 🗑️ Orphaned | Not imported | None |
| `upgrade_deterministic.py` | 1178 | 🗑️ Orphaned | Not imported by any module | None |

#### agent/ — GEPA Modules (only via `agent.__init__.py`)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `experiment_logger.py` | 205 | ⚠️ GEPA only | Only imported by agent.__init__ and orchestrator |
| `routing_table.py` | 269 | ⚠️ GEPA only | Only imported by agent.__init__ and orchestrator |
| `evaluation_agent.py` | 294 | ⚠️ GEPA only | Only imported by agent.__init__ and orchestrator |
| `mutation_agent.py` | 482 | ⚠️ GEPA only | Only imported by agent.__init__ and orchestrator |
| `analysis_agent.py` | 189 | ⚠️ GEPA only | Only imported by agent.__init__ and orchestrator |
| `orchestrator.py` | 455 | ⚠️ GEPA only | Only imported by agent.__init__ |
| `run_logger.py` | 471 | 🗑️ Orphaned | Not imported by any module (GEPA) |
| `gepa_runner.py` | 908 | 🗑️ Orphaned | Not imported by any module (GEPA; only called by generate_generation_0.py and eval_gen0_qwen.py) |
| `generate_generation_0.py` | 244 | 🗑️ Orphaned | Imports from agent.gepa_runner but not imported externally |
| `eval_gen0_qwen.py` | 195 | 🗑️ Orphaned | Imports from agent.gepa_runner but not imported externally |

---

### 1.2 staging/ — Pull-Based Parallel Pool System

| File | Lines | Status | Dependencies (imports from) | Tests |
|------|-------|--------|----------------------------|-------|
| `__init__.py` | 26 | ✅ Active | staging.ready_config, staging.ready_queue, staging.ready_pool, staging.ready_judge | None |
| `entrypoint.py` | 212 | ✅ Active | staging (ReadyConfig, ReadyQueue, ReadyMonitor, ReadyJudge), staging.ready_classifier | None |
| `ready_pool.py` | 418 | ✅ Active | staging.ready_config, staging.ready_queue, staging.ready_judge, staging.workers.det_worker, staging.workers.loc_worker, staging.workers.fw_worker | None |
| `ready_worker.py` | 244 | ✅ Active | staging.ready_config, staging.ready_queue | None |
| `ready_judge.py` | 423 | ✅ Active | staging.ready_config, **agent.solvers.fireworks**, **agent.solvers.fw_router** | `staging/test_judge.py` (175 lines) |
| `ready_queue.py` | 157 | ✅ Active | stdlib only | None |
| `ready_config.py` | 99 | ✅ Active | stdlib only | None |
| `ready_classifier.py` | 187 | ✅ Active | stdlib + re only (standalone — NO agent imports) | None |
| `workers/det_worker.py` | 114 | ✅ Active | **agent.solvers.deterministic** | None |
| `workers/loc_worker.py` | 143 | ✅ Active | llama_cpp (no agent imports) | None |
| `workers/fw_worker.py` | 107 | ✅ Active | **agent.solvers.fireworks**, **agent.solvers.fw_router** | None |
| `test_judge.py` | 175 | ✅ Self-test | staging.ready_config, staging.ready_judge | ✅ (inline) |
| `workers/__init__.py` | 7 | ✅ Active | imports det_worker, fw_worker, loc_worker | None |

**staging imports from agent/:**
- `ready_judge.py` → `agent.solvers.fireworks`, `agent.solvers.fw_router`
- `det_worker.py` → `agent.solvers.deterministic` (5 solver functions: solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa)
- `fw_worker.py` → `agent.solvers.fireworks`, `agent.solvers.fw_router`

**staging does NOT import `agent.Pipeline`** — this is intentional (bypasses Pipeline for parallel architecture).

> **Pull-system design:** The file `docs/plans/PULL_SYSTEM_DESIGN.md` (1,472 lines) specifies a complete transformation of this module from the current hybrid push-pool model to a **pure pull-based pool with work stealing**. See section 8-9 of that document for detailed code patterns and the full implementation specification. **Current staging code still uses the hybrid push model** — the pull system is design-only.

---

### 1.3 runner/ — Local Evaluation Wrapper

| File | Lines | Status | Dependencies (imports from) | Tests |
|------|-------|--------|----------------------------|-------|
| `__init__.py` | 29 | ✅ Active | runner.batch_runner, runner.deploy, runner.evaluate | None |
| `batch_runner.py` | 441 | ✅ Active | **agent** (PipelineConfig) | `tests/test_batch_runner.py` (542 lines) |
| `evaluate.py` | 677 | ✅ Active | **scripts.evaluate** (fuzzy_match, grade_answer), openpyxl | `tests/test_evaluate.py` (671 lines) |
| `deploy.py` | 449 | ✅ Active | stdlib + subprocess only | `tests/test_deploy.py` (411 lines) |

**runner imports from agent/:**
- `batch_runner.py` → `agent.PipelineConfig` (and lazy `agent.Pipeline`)

**runner does NOT import from staging/** at all.

---

### 1.4 Root Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `harness.py` | 109 | ✅ Docker entrypoint | Only entrypoint in Dockerfile. Imports `agent.Pipeline` |
| `Dockerfile` | 44 | ⚠️ Outdated | Only copies agent/ + harness.py. Missing staging/, runner/, scripts/ |
| `Dockerfile.staging` | 50 | ⚠️ Parallel container | Copies agent/ + staging/. Uses staging/entrypoint.py entrypoint |
| `Dockerfile.gpu` | 51 | ⚠️ GPU testing variant | Copies agent/ + staging/. CUDA-accelerated llama-cpp-python |
| `Makefile` | 59 | ⚠️ Outdated | Only has targets for harness.py + runner/. No staging targets |
| `requirements.txt` | 14 | ⚠️ Partial | Lists 14 packages. Missing sentence-transformers, scikit-learn, black, ruff |
| `.dockerignore` | 31 lines | ⚠️ Blocks staging/ | Explicitly ignores staging/, runner/, scripts/, tests/, docs/, data/ |
| `multi_runner.py` | 123 | ⚠️ Standalone runner | Directly imports agent modules (pre_filter, category_filter, complexity, etc.) |
| `container/runner.py` | 229 | ⚠️ Third pipeline | Self-contained container pipeline. No imports from agent/ |

---

### 1.5 Tests/

| File | Lines | Tests What | Notes |
|------|-------|-----------|-------|
| `tests/test_batch_runner.py` | 542 | runner.batch_runner | Good mock-based tests (542 lines) |
| `tests/test_evaluate.py` | 671 | runner.evaluate + scripts.evaluate | Tests fuzzy_match, grade_answer, report building |
| `tests/test_deploy.py` | 411 | runner.deploy | Docker build/push/verify mocks |
| `tests/test_dynamic_prompts.py` | 280 | agent.dynamic_prompts | Covers prompt building, complexity levels, merged prompts |
| `staging/test_judge.py` | 175 | staging.ready_judge | Tests voting logic, fuzzy matching, judgment strategies |

**Untested modules:**
- `agent/pipeline.py` (810 lines) — NO tests
- `agent/main.py` (444 lines) — NO tests
- `agent/classifier.py` (120 lines) — NO tests
- `agent/category_filter.py` (1058 lines) — NO tests
- `agent/complexity.py` (135 lines) — NO tests
- `agent/complexity_filter.py` (899 lines) — NO tests
- `agent/pre_filter.py` (209 lines) — NO tests
- `agent/secondary_code.py` (198 lines) — NO tests
- `agent/secondary_factual.py` (723 lines) — NO tests
- `agent/secondary_reasoning.py` (452 lines) — NO tests
- `agent/solvers/deterministic.py` (2881 lines, biggest file!) — NO tests
- `agent/solvers/fireworks.py` (149 lines) — NO tests
- `agent/solvers/verify.py` (462 lines) — NO tests
- `agent/solvers/tool_registry.py` (367 lines, 28 tools) — NO tests
- `agent/solvers/logic_reasoning.py` (582 lines) — NO tests
- `staging/entrypoint.py` (212 lines) — NO tests
- `staging/ready_pool.py` (418 lines) — NO tests
- `staging/ready_worker.py` (244 lines) — NO tests
- `staging/ready_classifier.py` (187 lines) — NO tests
- `container/runner.py` (229 lines) — NO tests

---

### 1.6 docs/

#### docs/plans/ — Integration & Build Plans

| File | Lines | Description |
|------|-------|-------------|
| `INTEGRATION_AUDIT.md` | 691+ | This file — module inventory, dependency graph, integration gaps, phased plan |
| `PULL_SYSTEM_DESIGN.md` | 1472 | Full specification for pure pull-based pool with work stealing (5 pull relationships P1-P5, autonomous judge, health monitor, deadline adaptation) |
| `PULL_SYSTEM_DELTA.md` | 1144 | Delta analysis between current and target pull-based architecture |
| `ASSEMBLY_PLAN.md` | 667 | High-level assembly plan for the agent pipeline |
| `DATA_PLAN.md` | 372 | Data collection and training plan |
| `EVAL_SYSTEM_PLAN.md` | 1537 | Comprehensive evaluation system design |
| `EVAL_STANDARD.md` | 36 | Standards for evaluation |
| `FINDINGS_REVIEW.md` | 315 | Review of experimental findings |
| `GAP_CLOSURE_PLAN.md` | 406 | Plan for closing identified gaps |
| `PARALLEL_SUBMIT_PLAN.md` | 1123 | Plan for parallel submission architecture (precursor to staging) |
| `STAGING_HANDOFF.md` | 191 | Handoff documentation for staging system |
| `SUBMISSION_CONTAINER_PLAN.md` | 856 | Plan for submission container architecture |

#### docs/architecture/ — Architecture Documentation

| File | Lines | Description |
|------|-------|-------------|
| `DETERMINISTIC_ROUTING_ARCHITECTURE.md` | 216 | Architecture of deterministic routing system |
| `PROMPT_OPTIMIZATION_PLAN.md` | 163 | Plan for prompt optimization |
| `ROUTER_SPEC.md` | 71 | Specification for the routing system |

#### docs/handoffs/ — Session Handoffs

| File | Lines | Description |
|------|-------|-------------|
| `v12c-to-v12d-handoff.md` | 73 | Handoff from v12c to v12d |
| `v12e-session-handoff.md` | 80 | Handoff for v12e session |
| `v12h-session-handoff.md` | 95 | Handoff for v12h session |

#### docs/research/ — Research Documentation

| File | Lines | Description |
|------|-------|-------------|
| `CLASSIFIER_ANALYSIS.md` | 209 | Analysis of classifier approaches |
| `RESEARCH_BRIEF.md` | 56 | Research brief |
| `RESEARCH_RECOMMENDATIONS.md` | 712 | Research recommendations |
| `eval-dataset-audit.md` | 216 | Audit of evaluation datasets |
| `eval_answer_length_analysis.md` | 236 | Analysis of answer lengths in eval data |
| `genetic_prompt_evolution_methodology.md` | 675 | Methodology for genetic prompt evolution (GEPA) |

#### docs/eval/ — Evaluation Documentation

| File | Lines | Description |
|------|-------|-------------|
| `EVAL_INFRASTRUCTURE_REVIEW.md` | 261 | Review of evaluation infrastructure |
| `LONGFORM_TEST_README.md` | 127 | Readme for long-form text testing |

#### Other docs

| File | Lines | Description |
|------|-------|-------------|
| `docs/consensus-module-notes.md` | — | Notes on consensus module design |
| `docs/dynamic-prompts-design.md` | — | Design of dynamic prompt system |

---

### 1.7 scripts/

| File | Lines | Description |
|------|-------|-------------|
| `scripts/_gen_report.py` | 99 | Generate step-by-step Excel report from V12E pipeline trace |
| `scripts/_test_local_model.py` | 44 | Quick test of local model with pipeline stages |
| `scripts/_test_v12e_rename.py` | 78 | Rename test for v12e pipeline |
| `scripts/build_colab_notebook.py` | 490 | Build Colab notebook from source |
| `scripts/build_fact_db.py` | 168 | Build SQLite FTS5 fact database |
| `scripts/build_training_data.py` | 587 | Build training data (v1) |
| `scripts/build_training_data_v2.py` | 763 | Build training data (v2 with fixes) |
| `scripts/build_training_data_v3.py` | 494 | Build training data (v3) |
| `scripts/bundle_eval_sets.py` | 111 | Bundle evaluation sets into one JSON |
| `scripts/compare_models.py` | 383 | Compare model performance |
| `scripts/compare_nvidia_helpsteer.py` | 145 | Compare with NVIDIA HelpSteer |
| `scripts/demo_gepa.py` | 213 | Demo of GEPA genetic pipeline optimizer |
| `scripts/eval_ner_models.py` | 197 | Evaluate NER model variants |
| `scripts/eval_ner_models_v2.py` | 122 | Evaluate NER models (v2) |
| `scripts/eval_pipeline.py` | 716 | **Self-contained eval pipeline** — imports agent modules directly, bypasses Pipeline. Uses torch/transformers for model inference |
| `scripts/eval_v12e.py` | 213 | Evaluate v12e pipeline version |
| `scripts/evaluate.py` | 347 | Core evaluation/fuzzy-match/grade functions (imported by runner/evaluate.py) |
| `scripts/grade_v12e.py` | 78 | Grade v12e answers |
| `scripts/harness.py` | 853 | Standalone harness (duplicate of root harness.py?) |
| `scripts/load_popculture_facts.py` | 71 | Load pop culture facts into FactDB |
| `scripts/multi_runner.py` | 449 | Multi-model runner (standalone) |
| `scripts/patch_dup.py` | 85 | Patch duplicate entries |
| `scripts/run_v12e.py` | 68 | Run v12e evaluation |
| `scripts/smoke_test.py` | 48 | Smoke test for pipeline |

---

## 2. Dependency Graph

### 2.1 Module-Level Import Map

```
                        ┌───────────────────────────────────┐
                        │          Dockerfile(s)             │
                        │  ENTRYPOINT: harness.py (default)  │
                        │  ENTRYPOINT: staging (staging)     │
                        │  COPY: agent/ (all variants)       │
                        └────┬──────────────────────────────┘
                             │
              ┌──────────────┼──────────────┬──────────────────┐
              │              │              │                  │
              ▼              │              ▼                  ▼
        ┌──────────┐        │       ┌──────────────┐   ┌──────────────┐
        │ harness  │────────┤       │  Makefile     │   │  .dockerignore│
        │ .py      │        │       │  (runner/*)   │   │ (blocks most) │
        └────┬─────┘        │       └──────┬───────┘   └──────────────┘
             │              │              │
             ▼              │              ▼
     ┌───────────────┐      │      ┌──────────────────┐
     │  agent.       │◄─────┘      │  runner/          │
     │  Pipeline     │              │                  │
     └───────┬───────┘              │ batch_runner ◄───┤
             │                      │   → agent.PipelineConfig
             │                      │ evaluate ◄──────┤
             │                      │   → scripts.evaluate
             │                      │ deploy           │
             │                      └──────────────────┘
             │
             │    ┌─────────────────────────────────────┐
             │    │         staging/                     │
             │    │                                     │
             │    │ entrypoint.py                       │
             │    │   → staging.ready_classifier        │
             │    │      (standalone, no agent imports) │
             │    │                                     │
             │    │ ready_pool.py → workers/:           │
             │    │   det_worker.py                     │
             │    │     → agent.solvers.deterministic ◄─┤
             │    │   fw_worker.py                      │
             │    │     → agent.solvers.fireworks ◄─────┤
             │    │     → agent.solvers.fw_router ◄─────┤
             │    │   loc_worker.py                     │
             │    │     → llama_cpp (no agent)          │
             │    │                                     │
             │    │ ready_judge.py                      │
             │    │   → agent.solvers.fireworks ◄──────┤
             │    │   → agent.solvers.fw_router ◄──────┤
             │    └─────────────────────────────────────┘
             │
             │    ┌─────────────────────────────────────┐
             │    │         container/                   │
             │    │  (THIRD pipeline — self-contained)   │
             │    │  runner.py → server.py              │
             │    │            → inference.py           │
             │    │            → consensus.py           │
             │    │  No imports from agent/             │
             │    └─────────────────────────────────────┘
             │
             ▼
   ┌─────────────────────────────────────────────┐
   │            agent/ Modules                    │
   │                                             │
   │ pipeline.py                                 │
   │  → agent.workflow                           │
   │  → agent.dynamic_prompts                    │
   │  → agent.category_filter                    │
   │  → agent.complexity                         │
   │  → agent.pre_filter                         │
   │  → agent.solvers.deterministic              │
   │  → agent.solvers.fireworks                   │
   │  → agent.solvers.fw_router                   │
   │  → agent.solvers.local_vote                  │
   │  → agent.solvers.verify                      │
   │                                             │
   │ classifier.py                                │
   │  → agent.category_filter (importlib)         │
   │  → agent.secondary_code (importlib)           │
   │  → agent.secondary_factual (importlib)        │
   │  → agent.secondary_reasoning (importlib)      │
   │  → agent.solvers.prototype_ner_v3             │
   │  → agent.solvers.deterministic                │
   │                                             │
   │ tool_registry.py                             │
   │  → agent.solvers.deterministic (lazy)        │
   │  → agent.solvers.fact_db (lazy)              │
   │  → agent.solvers.logic_solver (lazy)         │
   │  → agent.solvers.logic_reasoning (lazy)      │
   │  → agent.solvers.code_sandbox (lazy)         │
   │  → agent.solvers.verify (lazy)               │
   │  → agent.solvers.spell_check (lazy)          │
   │  → agent.solvers.web_search (lazy)           │
   │  → agent.solvers.easter_egg_shelf (lazy)     │
   └─────────────────────────────────────────────┘
```

### 2.2 What Imports What (Explicit Dependency List)

| Source | → Target |
|--------|---------|
| `agent.pipeline` | `agent.workflow`, `agent.dynamic_prompts`, `agent.category_filter`, `agent.complexity`, `agent.pre_filter`, `agent.solvers.deterministic`, `agent.solvers.fireworks`, `agent.solvers.fw_router`, `agent.solvers.local_vote`, `agent.solvers.verify` |
| `agent.main` | `agent.config`, `agent.solvers.deterministic`, `agent.dynamic_prompts`, `agent.solvers.fireworks`, `agent.solvers.local_vote`, `agent.solvers.verify`, `agent.pre_filter`, `agent.category_filter`, `agent.complexity_filter`, `agent.quality_config`, `agent.circuit_breaker` |
| `agent.__init__` | `agent.pipeline`, `agent.cell`, `agent.workflow`, `agent.experiment_logger`, `agent.routing_table`, `agent.evaluation_agent`, `agent.mutation_agent`, `agent.analysis_agent`, `agent.orchestrator` |
| `agent.classifier` | `agent.category_filter` (importlib), `agent.secondary_code` (importlib), `agent.secondary_factual` (importlib), `agent.secondary_reasoning` (importlib), `agent.solvers.prototype_ner_v3`, `agent.solvers.deterministic` |
| `agent.ml_classifier` | `agent.category_filter` (importlib) |
| `agent.solvers.__init__` | `agent.solvers.logic_solver`, `agent.solvers.code_sandbox`, `agent.solvers.easter_egg_shelf`, `agent.solvers.spell_check`, `agent.solvers.web_search` |
| `agent.solvers.fireworks` | `agent.config`, `agent.answer_cleaner` |
| `agent.solvers.local` | `agent.config`, `agent.answer_cleaner`, `agent.solvers.tools` |
| `agent.solvers.deterministic` | `agent.solvers.tools` |
| `agent.solvers.tool_registry` | `agent.solvers.fact_db`, `agent.solvers.deterministic`, `agent.solvers.verify`, `agent.solvers.logic_solver`, `agent.solvers.logic_reasoning`, `agent.solvers.code_sandbox`, `agent.solvers.spell_check`, `agent.solvers.web_search`, `agent.solvers.easter_egg_shelf` (all lazy) |
| `agent.workflow` | `agent.cell` |
| `agent.eval_gen0_qwen` | `agent.gepa_runner` |
| `agent.generate_generation_0` | `agent.gepa_runner` |
| `staging.entrypoint` | `staging.*`, `staging.ready_classifier` |
| `staging.ready_pool` | `staging.ready_config`, `staging.ready_queue`, `staging.ready_judge`, `staging.workers.det_worker`, `staging.workers.loc_worker`, `staging.workers.fw_worker` |
| `staging.ready_worker` | `staging.ready_config`, `staging.ready_queue` |
| `staging.ready_judge` | `staging.ready_config`, `agent.solvers.fireworks`, `agent.solvers.fw_router` |
| `staging.workers.det_worker` | `staging.ready_config`, `staging.ready_queue`, `staging.ready_worker`, `agent.solvers.deterministic` |
| `staging.workers.fw_worker` | `staging.ready_config`, `staging.ready_queue`, `staging.ready_worker`, `agent.solvers.fireworks`, `agent.solvers.fw_router` |
| `staging.workers.loc_worker` | `staging.ready_config`, `staging.ready_queue`, `staging.ready_worker`, `llama_cpp` |
| `runner.batch_runner` | `agent` (PipelineConfig) |
| `runner.evaluate` | `scripts.evaluate` |
| `runner.deploy` | (stdlib only) |
| `harness.py` | `agent.Pipeline` |
| `multi_runner.py` | `agent.pre_filter`, `agent.category_filter`, `agent.complexity`, `agent.solvers.deterministic`, `agent.dynamic_prompts`, `agent.run_logger` |
| `scripts.eval_pipeline` | `agent.pre_filter`, `agent.category_filter`, `agent.complexity_filter`, `agent.solvers.verify` |

### 2.3 No Circular Dependencies Found

The dependency graph is acyclic at both module and package level:
- `agent/` is a pure leaf (no internal cycles, no imports from staging/ or runner/)
- `staging/` imports from `agent/` only (solver-level, not Pipeline-level)
- `runner/` imports from `agent/` only
- `container/` imports from neither agent/ nor staging/ nor runner/
- No imports from staging/ → runner/ or runner/ → staging/

---

## 3. Integration Gaps

### Gap 1: Dockerfile only includes agent/ (CRITICAL)

The main `Dockerfile` only copies `agent/` and `harness.py`:
```dockerfile
COPY agent/ /agent/
COPY harness.py /harness.py
```
**Missing directories:** `staging/`, `runner/`, `scripts/`

The staging entrypoint (`staging/entrypoint.py`) cannot run in the main container. Workarounds exist (`Dockerfile.staging`, `Dockerfile.gpu`) but they're separate images.

### Gap 2: Two competing entrypoints (CRITICAL)

- `harness.py` — Original, uses `agent.Pipeline` (sequential, single-threaded)
- `staging/entrypoint.py` — New, uses parallel worker pool with multiprocessing

**No mechanism to choose between them.** The main Dockerfile ENTRYPOINT hardcodes `harness.py`:
```dockerfile
ENTRYPOINT ["python3", "-u", "harness.py"]
```
Additionally, `Dockerfile.staging` and `Dockerfile.gpu` each hardcode different entrypoints.

### Gap 3: staging bypasses agent.Pipeline (INTENTIONAL, but risky)

staging/ imports individual solver functions (deterministic, fireworks) directly, NOT through Pipeline. This means:
- The Pipeline's routing logic (classifier → complexity → decision table → solver chain) is duplicated/reimplemented in staging/
- staging/ has its OWN classifier (`ready_classifier.py`) that duplicates `category_filter.py`
- staging/ has its OWN system prompts in `loc_worker.py` that duplicate `dynamic_prompts.py`
- Staging workers do NOT use Pipeline's QC gate (`verify.py`), complexity scoring, or circuit breaker

**Changes to agent.solvers.deterministic function signatures** would break staging silently.

### Gap 4: runner/ and staging/ are completely independent (DESIGN QUESTION)

- `runner/` wraps `agent.Pipeline` for local eval — sequential, single-process
- `staging/` replaces `agent.Pipeline` with parallel pool — multiprocessing
- They never interact. `runner/batch_runner.py` uses `PipelineConfig` directly, not staging workers.
- If you wanted to run staging through runner's evaluation pipeline, you'd need adapter code.

### Gap 5: requirements.txt is sparse

Only 14 packages listed. Missing explicit dependencies:
- `llama-cpp-python` is listed (good)
- `sentence-transformers` NOT listed (used by `agent/complexity.py`)
- `scikit-learn` NOT listed (used by `agent/ml_classifier.py`, `agent/complexity.py`)
- `RestrictedPython` IS listed (good)
- `python-constraint` IS listed (good)
- `spacy` IS listed (good — used by prototype NER solvers)
- `black` / `ruff` NOT listed (needed by `agent/solvers/verify.py` QC gate)
- `vaderSentiment` IS listed (good)
- `symspellpy` IS listed (good)
- `duckduckgo-search` IS listed (good)
- `numpy`, `scipy`, `sympy`, `click`, `openpyxl` all listed (good)

### Gap 6: Makefile has no staging targets

Current targets: `build`, `run`, `shell`, `rebuild`, `size`, `local-test`, `test-tasks`, `build-image`, `push`, `deploy`, `evaluate`, `run-batch`

Missing: `staging-build`, `staging-run`, `staging-test-judge`, `staging-verify`

### Gap 7: No integration tests for staging/ + agent/ interaction

- `staging/test_judge.py` tests the judge module in isolation
- But there's no test that verifies `staging/workers/det_worker.py` correctly imports and calls `agent.solvers.deterministic` functions
- No test that the `ready_classifier.py` produces compatible categories with `agent/category_filter.py`
- No end-to-end test of the full staging pipeline

### Gap 8: Orphaned duplicate classifier

`staging/ready_classifier.py` is a simplified standalone reimplementation of `agent/category_filter.py`. It has its own scoring functions that may produce different results. No cross-validation with the original classifier.

### Gap 9: Broken import chain in `agent/solvers/__init__.py` (MEDIUM)

`agent/solvers/__init__.py` eagerly imports `from agent.solvers import logic_solver`, which does `from constraint import Problem` (python-constraint package). This chain breaks if python-constraint is missing or the wrong version. The main pipeline avoids this by importing solver modules directly by full path (e.g., `agent.solvers.deterministic`), but `import agent.solvers` or `from agent.solvers import solve_logic_puzzle` will fail.

The secondary classifiers (`agent/secondary_*.py`) avoid this by using `importlib` to lazily load modules — a pattern `agent/solvers/__init__.py` should adopt.

### Gap 10: Pull-system architecture documented but not implemented (LOW)

The file `docs/plans/PULL_SYSTEM_DESIGN.md` (1,472 lines) documents a comprehensive transformation of the staging module from its current hybrid push-pool model to a **pure pull-based pool with work stealing**. Key proposed changes:

- Replace per-worker private queues + round-robin dispatch with a **single shared `task_pool`** (`multiprocessing.Queue`)
- Workers pull from shared pool when idle (autonomous, no dispatcher)
- **Work stealing protocol**: idle workers can steal tasks from busy workers' inboxes
- **Autonomous judge**: judge runs in its own loop, pulls results from shared `results_queue`
- **Pool → Health Monitor**: pool no longer dispatches tasks; only monitors worker health, heartbeats, deadline enforcement
- **Graceful degradation**: auto-detect dead workers, re-enqueue orphaned tasks, reduce judgment threshold under deadline pressure

**Current status**: Design is fully specified with code patterns (sections 8-9) but **not yet implemented**. The current staging system (`ready_pool.py`, `ready_worker.py`, `ready_judge.py`) still uses the hybrid push model. See `docs/plans/PULL_SYSTEM_DESIGN.md` for the full specification.

### Gap 11: Secondary classifiers not utilized by staging (LOW)

`staging/entrypoint.py` uses `staging.ready_classifier.py` for classification, which is a standalone reimplementation. It does NOT use `agent/classifier.py` or any of the three secondary classifiers (`secondary_code.py`, `secondary_factual.py`, `secondary_reasoning.py`). This means staging misses the debug-vs-gen, logic-vs-math, and factual-vs-logic disambiguation that the agent pipeline benefits from.

**Mitigation**: Either:
- (Option A) Make `staging/ready_classifier.py` delegate to `agent.classifier.classify()` via importlib (same pattern classifier.py uses for category_filter)
- (Option B) Port the secondary heuristic logic into `staging/ready_classifier.py`

### Gap 12: .dockerignore blocks staging, runner, scripts from builds (CRITICAL — NEW)

The `.dockerignore` file explicitly excludes these directories:
```
staging/
runner/
scripts/
tests/
docs/
data/
config/
```

Even if `Dockerfile` is updated with `COPY staging/ /staging/`, the `.dockerignore` will prevent the staging directory from being included in the Docker build context. This means:
- `Dockerfile.staging` also cannot actually include `staging/` (despite its `COPY` instruction)
- Any fix for Gap 1 requires also fixing `.dockerignore`
- The `.dockerignore` was presumably written assuming only `agent/` is needed, but now it actively blocks container parity

### Gap 13: container/ module is a third competing pipeline (MEDIUM — NEW)

The `container/` directory (861 lines, 5 Python files) implements a **completely separate container-based pipeline**:

| File | Lines | Description |
|------|-------|-------------|
| `container/runner.py` | 229 | Pipeline orchestrator with category routing, parallel inference, consensus |
| `container/inference.py` | 102 | LLM inference via llama.cpp server (parallel + simple) |
| `container/consensus.py` | 262 | Merging and validation of multi-model answers |
| `container/server.py` | 147 | llama.cpp server process manager |
| `container/fallback.py` | 79 | Fallback inference via direct HTTP |
| `container/run.sh` | 40 | Shell entrypoint script |

This module:
- Does NOT import from `agent/` at all (entirely self-contained)
- Has its own fuzzy matching logic (duplicates `scripts/evaluate.py`)
- Has its own inference management (duplicates `agent/solvers/local.py`)
- Has its own consensus/voting (duplicates `staging/ready_judge.py`)
- Has its own entrypoint (no integration with `harness.py` or `staging/entrypoint.py`)
- Has no tests

This is a **third independent evaluation pipeline** alongside `harness.py` (Pipeline) and `staging/entrypoint.py` (parallel pool).

### Gap 14: Three competing Dockerfiles with no unified build (MEDIUM — NEW)

| Dockerfile | Base | Entrypoint | Copies staging? |
|------------|------|------------|-----------------|
| `Dockerfile` | python:3.12-slim | `harness.py` (Pipeline) | No |
| `Dockerfile.staging` | python:3.12-slim | `staging.entrypoint` | Yes (intended) |
| `Dockerfile.gpu` | nvidia/cuda:12.8.0 | `staging.entrypoint` | Yes (intended) |

No shared base layer, no Docker Compose orchestration, no unified tag scheme. Each must be maintained separately. `.dockerignore` likely prevents staging/ from being included in all builds (see Gap 12).

### Gap 15: scripts/eval_pipeline.py is a fourth competing evaluation entrypoint (LOW — NEW)

`scripts/eval_pipeline.py` (716 lines) is a self-contained evaluation pipeline that:
- Directly imports `agent.pre_filter`, `agent.category_filter`, `agent.complexity_filter`, `agent.solvers.verify`
- Uses `torch` + `transformers` for model inference (not llama-cpp-python, not fireworks API)
- Has its own comprehensive evaluation framework (modes: single, batch, cascade, sweep)
- Is entirely separate from both `harness.py` and `staging/entrypoint.py`

This fragmentation means there are now **four distinct paths** to run the pipeline: `harness.py`, `staging/entrypoint.py`, `container/runner.py`, and `scripts/eval_pipeline.py`.

---

## 4. Integration Order (Phased Dependency-Based Plan)

### Phase 1: Container Parity (Dockerfile + .dockerignore Update)
**Goal:** staging/ runs in the container alongside the existing setup.

1. **Fix `.dockerignore` FIRST** — Remove `staging/`, `runner/`, `scripts/`, `tests/` from the ignore list. Without this, Dockerfile changes have no effect.

2. Update `Dockerfile`:
   - Add `COPY staging/ /staging/`
   - Add `COPY scripts/ /scripts/` (needed by runner/evaluate.py)
   - Add `COPY runner/ /runner/` (for completeness)

3. Add an env-var switch in a new root dispatcher (e.g., `entrypoint.py` or modify `harness.py`):
   - `STAGING_ENABLED=0` → run `harness.py` (current)
   - `STAGING_ENABLED=1` → run `staging/entrypoint.py`

4. Update `requirements.txt`:
   - Add `sentence-transformers` (for agent/complexity.py)
   - Add `scikit-learn` (for agent/ml_classifier.py, agent/complexity.py)
   - Add `black`, `ruff` (for agent/solvers/verify.py QC)

5. Update `Makefile`:
   - Add `staging-build`, `staging-run`, `staging-test` targets

### Phase 2: staging/ → agent/ Import Hardening
**Goal:** staging/ resists agent.solvers API changes.

1. Create a thin adapter layer in `staging/_agent_shim.py`:
   - Wraps `agent.solvers.deterministic` solver functions
   - Wraps `agent.solvers.fireworks.FireworksSolver`
   - Wraps `agent.solvers.fw_router.route`
   - Version + validation checks on import (catch signature mismatches at startup)

2. Add unit tests for shim:
   - Verify each wrapped function exists and has expected signature
   - Test with mock agent.solvers functions

### Phase 3: Classifier Harmonization (Option A preferred)
**Goal:** One classifier, not two.

1. Audit differences between `agent/category_filter.py` and `staging/ready_classifier.py`
2. Either:
   - (Option A) Make `ready_classifier.py` a thin wrapper that imports `agent.classifier.classify()` via importlib (this gives access to all three secondary disambiguators too), OR
   - (Option B) Keep them separate but add cross-validation tests that verify category assignments match ≥95% of the time

Recommendation: Use **Option A** — the importlib pattern is already established in `agent/classifier.py` and gives staging access to the full classification chain (primary + secondary_code + secondary_factual + secondary_reasoning) with zero additional dependencies.

### Phase 3b: Implement Pull-Based Pool Architecture (from PULL_SYSTEM_DESIGN.md)
**Goal:** Transform staging from hybrid push-pool to pure pull-based with work stealing.

Reference the full specification in `docs/plans/PULL_SYSTEM_DESIGN.md`. Key deliverables:
1. **Replace per-worker queues** with a single shared `multiprocessing.Queue` (`task_pool`)
2. **Rewrite `ready_worker._pull_task()`** priority: stolen > inbox > task_pool > steal
3. **Add work stealing protocol**: `steal_request_queue` + per-worker `stolen_queue`
4. **Make judge autonomous**: judge pulls from shared `results_queue` in its own loop
5. **Rename `ReadyPool` → `ReadyMonitor`**: health checking only, no dispatching
6. **Add heartbeat mechanism**: each worker updates `multiprocessing.Value` timestamp every iteration
7. **Add deadline_emergency protocol**: shared flag broadcast by monitor, checked by workers + judge

See sections 8-9 of `docs/plans/PULL_SYSTEM_DESIGN.md` for detailed code patterns.

### Phase 4: Evaluation Pipeline Integration
**Goal:** Run staging results through runner's evaluation/grading.

1. Add a `runner/evaluate_staging.py` (or extend `runner/evaluate.py`):
   - Accepts staging output format (ReadyTask → {task_id, answer})
   - Runs the same `scripts.evaluate.fuzzy_match` / `grade_answer` grader cascade
   - Produces the same XLSX report format

2. Add a `runner/batch_runner_staging.py` (or extend `runner/batch_runner.py`):
   - Spawns staging pool instead of Pipeline instances
   - Collects results, runs evaluation

### Phase 5: Consolidate container/ Module (NEW)
**Goal:** Eliminate the third competing pipeline.

1. Audit `container/` to determine if it provides unique functionality not covered by `staging/`:
   - `container/consensus.py` — merge_answers and is_degenerate (overlaps with ready_judge)
   - `container/inference.py` — parallel inference (overlaps with staging workers)
   - `container/server.py` — llama.cpp server manager (unique functionality)
   - `container/fallback.py` — fallback inference (useful for resilience)

2. Either:
   - (Option A) Integrate unique functionality into staging/archive the rest
   - (Option B) Keep container/ separate but document its relationship to staging/ and add integration tests

3. Remove redundant consensus/fuzzy-match code in favor of the canonical implementations in `scripts/evaluate.py`

### Phase 6: Dead Code Cleanup (Optional)
**Goal:** Remove orphaned files that add maintenance burden.

1. Archive prototype solvers in `archive/`:
   - `prototype_ner_solver.py`, `prototype_ner_v2.py`
   - `prototype_zebra_solver.py`, `prototype_zebra_v2.py`
   - `upgrade_deterministic.py`

2. Remove orphaned files (confirm no imports first):
   - `agent/caveman_prompts.py`
   - `agent/bitmorphic_classifier.py`
   - `agent/category_router.py`
   - `agent/hierarchical_classifier.py`
   - GEPA-runner files not used by pipeline (`run_logger.py`, `gepa_runner.py`, `generate_generation_0.py`, `eval_gen0_qwen.py`)

3. Clean up stale docs handoffs older than 3 versions (archive if needed)

4. Remove backup files:
   - `staging/ready_pool.py.bak.phase1`
   - `deterministic.py.bak`
   - `prototype_ner_v3.py.bak`

---

## Files to Create/Update

| File | Action | Reason |
|------|--------|--------|
| `docs/plans/INTEGRATION_AUDIT.md` | ✅ Updated | This file — comprehensive re-audit |
| `.dockerignore` | 🔧 Update | Remove staging/, runner/, scripts/, tests/ from ignore list |
| `Dockerfile` | 🔧 Update | Add COPY for staging/, runner/, scripts/ |
| `requirements.txt` | 🔧 Update | Add sentence-transformers, scikit-learn, black, ruff |
| `Makefile` | 🔧 Update | Add staging targets |
| `staging/_agent_shim.py` | ✨ Create | Adapter layer for agent.solvers imports |
| `staging/ready_classifier.py` | 🔧 Option A | Delegate to agent.classifier.classify() via importlib |
| `runner/evaluate_staging.py` | ✨ Future | Accept staging output format |
| `runner/batch_runner_staging.py` | ✨ Future | Spawn staging pool |

## Verification

After writing, verify by checking the file exists, its line count, and the first lines.

---

*End of Integration Audit — 2026-07-13 Re-audit*
