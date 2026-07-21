# Comprehensive Gap Analysis: Research Brief vs. Existing Codebase

**Generated:** July 13, 2026  
**Scope:** `/home/artem/dev/amd-hackathon/`  
**Research Brief:** `deep_research/Deep Research Brief_ Agentic Genetic-Pareto Archit.md`

---

## Executive Summary

The existing codebase has a **solid foundation** but is heavily specialized for factual-QA prompt optimization on a single model (smollm2-1.7b). The research brief proposes a **full agentic architecture** with 6 sub-agents, formalized "cell" composition, multi-task optimization, deterministic routing, experiment tracking, and governance — most of which is **missing or only partially present**.

Rough split: **~20% EXISTS**, **~15% PARTIAL**, **~65% MISSING**.

---

## 1. Conceptual Model of "Cells" in GEPA Framework

### (PARTIAL) Cell concept

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Formal compositional cell: `(task_type T01–T05, ensemble of LLMs, per-LLM prompt + decoding params, aggregation/judge strategy, evaluation metadata)` | `gepa_runner.py` has prompt variants structured as `{name, system_prompt, temperature, max_tokens, accuracy, avg_output_tokens, avg_latency_ms}` — a subset of the cell concept. | **No formal Cell class/dataclass.** No concept of: task type attached to a variant, multiple models per cell, aggregation strategy, or evaluation metadata as a persistent first-class object. |
| Mutation alters any cell dimension (swap LLM, prompt style, decoding params, aggregation method) | `gepa_runner.py` has 8 mutation operators, but they only operate on **prompt text and temperature** — not on model selection, ensemble size, aggregation strategy, or judge type. | Mutations are prompt-only. No "swap model", "change aggregation", "add LLM to ensemble" operators exist. |
| Crossover combines substructures from two high-performing cells | Sentence-level uniform crossover in `crossover_prompts()` works on prompt text only. | Crossover is text-only. No crossover of model sets, decoding configs, or judge strategies. |
| Diversity maintained by: limiting ensemble size (1–2 models), sharing model instances across cells, capping candidates per generation, favoring heterogeneous cells on Pareto front | ModelCache in gepa_runner.py shares loaded models across evaluations. Population size is fixed at 8. | **No explicit diversity enforcement.** No constraint that Pareto front must contain cells using different models/prompts/judges. Ensemble size limit not implemented (1 model per cell always). |

### (MISSING) Cell formalization

- **No Cell dataclass/struct.** Variants are plain dicts with ad-hoc keys.
- **No persistence layer for cells.** Cells exist only during a run; no cell registry or versioned store.
- **No aggregation strategy field.** All variants use single-model single-sample inference. No majority-vote, self-consistency, or judge strategy configuration per cell.
- **No decoding config object.** Only temperature and max_tokens are configurable; top-p, top-k, min-p, repeat_penalty, seed are not exposed.

---

## 2. Overall System Architecture with Agentic Sub-Components

### (MISSING) Central GEPA Orchestrator

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Central orchestrator managing population and Pareto front, scheduling batch evaluation jobs, handing failure traces to analysis agent | `gepa_runner.py` has a `run_gepa()` function acting as a monolithic GEPA loop. | **Not an agent.** It's a script, not an orchestrator with message-passing, sub-agent coordination, or runtime lifecycle. No event loop, no internal bus, no scheduling abstraction. |
| Orchestrator only updates routing table when governance agent confirms multi-objective improvement | No governance agent exists. | Routing table updates are not gated on any verification. |

### (MISSING) Mutation / Evolution Agent

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Applies genetic operators to cells; produces new candidates from labeled parent set | `mutate()` and `crossover_prompts()` exist in `gepa_runner.py`. | These are functions, not an agent with a defined interface. No mutation agent contract, no "produce N candidates from parent set" API, no tagging of mutation types applied. |

### (MISSING) Evaluation Agent

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Runs cells on dev sets, collects multiple metrics per cell | `evaluate_variant_on_model()` and `evaluate_population()` exist. | No formal Evaluation Agent interface. Metrics are hard-coded (accuracy, tokens, latency). No extensibility for task-specific metrics (NER F1, math solve rate, ROUGE, format compliance). |
| Must support sampling across seeds and inputs | Single seed (temperature=0.0) used throughout. | No multi-seed evaluation, no performance distribution estimation. |

### (MISSING) Analysis / Diagnostics Agent

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Inspects failures (wrong labels, format breaks, slow outliers), tags patterns, biases future mutations | **Nothing exists.** | No failure analysis, no pattern tagging, no feedback loop to mutation agent. The `generation_0_results.json` has per-question details but no automated analysis. |

### (MISSING) Routing Agent

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Selects best cell for a live query; reads from persisted routing table, never writes | `agent/pipeline.py` is a routing pipeline — it classifies the query, runs deterministic solvers, falls back to LLM. | **No cell-aware routing.** The pipeline doesn't know about "cells" — it uses a single model with 1 system prompt per category, not a Pareto-optimal cell. No routing table exists. Selection is by category + complexity, not by multi-objective optimization. |

### (MISSING) Reporting / Governance Agent

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Maintains backtesting subset, requires champion/ challenger validation before promotion, tracks evolution, flags regressions | **Nothing exists.** | No governance of routing table updates. No regression detection. No automated backtesting. |

### (MISSING) Tool Layer

| What Brief Asks | What Exists | Gap |
|---|---|---|
| LLM inference, deterministic validators, logging and metrics store | Deterministic solvers exist. LLM inference exists (local + Fireworks). | No unified tool layer with service interfaces. No metrics store for cell configurations, Pareto fronts, or decisions. Validators are not first-class tools with latency tracking. |

### (MISSING) Lightweight message bus / task queue

| What Brief Asks | What Exists | Gap |
|---|---|---|
| In-process queues, SQLite-backed work registry, or minimal orchestration | **Nothing exists.** Agents do not exist, so no inter-agent communication infrastructure exists. | No message bus, no task queue, no work registry. |

---

## 3. Multi-Objective Optimization Goals

### (PARTIAL) Pareto Front Management

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Explicitly maintain Pareto front with multiple objectives | NSGA-II fast non-dominated sort + crowding distance fully implemented in `gepa_runner.py`. 3 objectives: accuracy (max), tokens (min), latency (min). | Currently only for factual QA. Not ported to other task types. No per-task-objective customization. |
| Retain cells that are non-dominated even if slower or less deterministic | Pareto front correctly preserves non-dominated solutions. | Pareto front is computed per-model, not cross-model or cross-task. No cell-level Pareto front. |
| "Time per example" and "examples per evaluation batch" as budget parameters | No budget scheduling exists. | Evaluation budget is implicit (3 generations, 8 variants, 4 models). No parameterized budget control. |

### (MISSING) Task-Specific Objectives

| Task | Brief's Required Objectives | What Exists |
|---|---|---|
| Math (T02) | Correctness (deterministic solver graded), format compliance | `deterministic.py` has `solve_arithmetic()` but no math-grading metric in optimization |
| Sentiment (T03) | Label accuracy, format compliance | `solve_sentiment()` exists as deterministic bypass but not as evaluation metric |
| Summarization (T04) | ROUGE-like length/coverage, sentence/word limit compliance | `solve_summarization()` exists as bypass; no summary-specific metrics |
| NER (T05) | NER F1 from classical tagger, entity boundary accuracy | `solve_ner()` exists as bypass; no F1 computation |
| Format compliance (all) | JSON validity, sentence counts, bullet limits, word limits | **No format compliance metrics exist** for optimization evaluation |

### (MISSING) Robustness Objective

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Variance across seeds, performance quantiles rather than means | Single deterministic eval (temp=0.0) across all runs. | No multi-seed variance tracking, no quantile estimation, no robustness scoring. |

---

## 4. Agentic Optimization Cycle

### (PARTIAL) Generation Cycle

| What Brief Asks | What Exists | Gap |
|---|---|---|
| (1) Start from baseline cells → (2) mutation agent generates new candidates → (3) evaluation agent runs dev sets → (4) orchestrator updates Pareto front → (5) analysis agent inspects failures → (6) mutation agent biases future mutations | `run_gepa()` does: build seed → evaluate → compute fronts → report → check convergence → create next generation via mutation/crossover. | Steps 1-4 partially exist. **Steps 5-6 (analysis/diagnostics feedback loop) do not exist at all.** No failure analysis, no mutation biasing. |

### (MISSING) Convergence Detection

| What Brief Asks | What Exists | Gap |
|---|---|---|
| No Pareto-front improvements above threshold after N generations | `check_convergence()` exists using top-3 mean accuracy delta. | Convergence is accuracy-only, not Pareto-front based. No comprehensive convergence criteria. |
| Evaluation budget exhausted | No budget tracking. | No explicit generation/ time/ evaluation budget enforcement. |
| Stability of routing metrics over time | No routing metrics exist. | No stability monitoring. |

### (MISSING) Regression Guards

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Backtesting subset to validate new cells outperform current champion | **Nothing exists.** | No backtesting dataset. No champion/challenger protocol. |
| Require improvements on both task metrics and safety/format metrics before promotion | **Nothing exists.** | Routing table changes are entirely ungoverned. |

### (MISSING) Experiment Logging

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Cell configurations as versioned artifacts in experiment store | Results are saved to JSON files (`gepa_results.json`, `generation_0_results.json`). | **No MLflow or structured experiment tracking.** JSON files are flat, not queryable, not versioned. No audit trail of why a cell was selected. |
| Log all agent decisions | Agent decisions are not logged (agents don't exist). | No decision logging at all. |

---

## 5. Deterministic Routing and Non-LLM Classifiers

### (PARTIAL) Routing Pipeline

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Deterministic task classifier selects task-specific pool of candidate cells | `category_filter.py` (721 lines) is a sophisticated 8-way deterministic classifier using regex/heuristic scoring. `category_router.py` and `pre_filter.py` provide early bypass. | Classifier routes to a single category, not to a "pool of candidate cells." No cell pool concept exists. |
| Static heuristics or learned routers pick one cell | `pipeline.py` uses a single model with category-dependent system prompt. | No cell selection. No learned router. |
| Non-LLM validators downstream to score cell outputs | `verify.py`, `consensus.py` provide basic quality gates (degeneracy checks). | No task-specific validators integrated as services. No F1 computation for NER, no ROUGE for summarization. |

### (MISSING) Small ML Classifiers

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Small ML models (spaCy, transformer-mini) for task classification and validation | Current classifier is pure regex/heuristic. `complexity.py` uses MiniLM-L6-v2 + LogisticRegression for complexity scoring. | **No learned classifier for task routing.** Complexity uses a trained model, but task classification does not. No spaCy NER integration as downstream validator. |
| Non-LLM validators (NER taggers, sentiment classifiers) | `solve_ner()` and `solve_sentiment()` exist as deterministic regex solvers, not as ML-backed validators. | Deterministic solvers are for answering, not for validation of LLM outputs. Different purpose. |

### (MISSING) Router → Cell → Validator Architecture

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Router → deterministic task classifier → task-specific cell pool → cell selection → LLM ensemble → non-LLM validator → output → feedback to analysis agent | Pipeline flow: pre_filter → category_filter → complexity → build prompt → LLM → post-process → fallback. | Cells, cell pools, non-LLM validators as components, and feedback loops are absent. The pipeline is a single-pass process with no validation feedback. |

---

## 6. Technologies and Tools

### (PARTIAL) Inference

| What Brief Asks | What Exists | Gap |
|---|---|---|
| llama.cpp with GGUF quantization | ✅ Llama.cpp Llama() class used in both `gepa_runner.py` and `pipeline.py`. GGUF Q4_K_M models present. | Ollama not used. Only local GGUF + optional Fireworks API. no concurrent server mode for GEPA eval. |
| Multiple GGUF models in parallel | Container module has `ServerManager` (llama-server) + `parallel_infer()` sending 4 concurrent requests. | Server mode used in container, not in GEPA runner. GEPA runner loads models sequentially/one-at-a-time. |

### (MISSING) Orchestration / Workflow

| What Brief Asks | What Exists | Gap |
|---|---|---|
| Prefect-style workflow engine or lightweight orchestrator | **No orchestrator.** GEPA runs as a standalone script. Pipeline runs as a single-process CLI. | No task graphs, no retries, no scheduling. All orchestration is implicit in procedural code. |

### (MISSING) Experiment Tracking

| What Brief Asks | What Exists | Gap |
|---|---|---|
| MLflow or equivalent for logging parameters, metrics, Pareto fronts, cell configurations | JSON file dumps only. | No MLflow. No structured experiment store. No queryable history. No versioned artifacts. |

### (MISSING) Validation Libraries

| What Brief Asks | What Exists | Gap |
|---|---|---|
| spaCy for NER/sentiment, Promptfoo for deterministic checks, DeepEval for evaluations | Deterministic solvers in `deterministic.py` (1897 lines) cover math, logic, sentiment, NER, summarization, code debugging. | Deterministic solvers are purpose-built, not wrapping standard libraries. No spaCy integration. No DeepEval/Promptfoo integration. No formal evaluation framework. |

---

## 7. Dangers and Failure Modes (Mitigation Status)

### (PARTIAL) Optimization Dangers

| Danger | Brief's Recommended Safeguard | What Exists |
|---|---|---|
| Overfitting to small dev set | Separate held-out and backtesting sets; periodically re-sample evaluation inputs | Training set = 19 factual QA questions only. **No held-out set, no backtesting set, no re-sampling.** Single eval set used throughout. |
| Pareto front collapsing to one objective | Enforce minimum diversity constraints; maintain heterogeneous cells | **No diversity constraints on Pareto front.** No minimum diversity policy. |
| Loss of diversity in population | Explicit diversity constraints | No diversity enforcement via Hamming distance, model diversity, or prompt structure diversity. |

### (MISSING) Agentic/Orchestration Dangers

| Danger | Brief's Recommended Safeguard | What Exists |
|---|---|---|
| Cycle explosion | Explicit evaluation budgets per cycle | No budget parameters. |
| Deadlocks/race conditions | Strict responsibility boundaries, single-writer rules | Agents don't exist, so no boundaries. |
| Misconfigured routes | CI-like gating, multi-objective thresholds | No routing table, no gating. |
| Unclear agent boundaries | Narrow contracts, defined artifacts | No agents, no contracts. |

### (MISSING) Model/Inference Dangers

| Danger | Brief's Recommended Safeguard | What Exists |
|---|---|---|
| Mis-tuned decoding params | Constrain decoding ranges, treat params as part of cell definition with conservative defaults | Only temperature configurable. Top-p, top-k, min-p, repeat_penalty not exposed. |
| Small models beyond capability | "Capability boundaries" for task difficulty | No difficulty routing by model capability. All models get all factual questions. |
| Context length issues | Context-length monitoring and truncation logging | n_ctx=2048 set, no monitoring or logging of truncation. |

### (PARTIAL) Data/Evaluation Dangers

| Danger | Brief's Recommended Safeguard | What Exists |
|---|---|---|
| Dev/test contamination via few-shot examples | Store prompts in version control, keep eval separate | Prompts are hardcoded in `gepa_runner.py`. Eval set (`training-v3.json`) contains all task types but only factual is used. |
| Unrepresentative eval sets | Carefully curated golden datasets with edge cases | 19 factual QA questions — tiny and specialized. No edge cases for other task types. |
| Weak metrics for summarization/sentiment | Multi-faceted metrics combining deterministic checks and model-graded scores | Deterministic regex solvers only. No model-graded scoring. |

---

## 8. "Commands for Builders" — Compliance Status

| Command | Status | Notes |
|---|---|---|
| 1. Define cells as compositional units (task, models, prompts, params, aggregation, metadata) | ❌ MISSING | No cell abstraction exists |
| 2. Use deterministic, non-LLM classifiers for routing and validation | ✅ PARTIAL | Heuristic classifier exists; no ML classifiers; no validator integration |
| 3. Treat GEPA optimization as offline, budgeted process; deploy Pareto-optimal cells to live routing | ❌ PARTIAL | GEPA is offline but no budget; no connection between GEPA output and live pipeline |
| 4. Instrument and log all agent decisions; never update routing table without verifiable improvement | ❌ MISSING | No logging, no routing table, no governance |
| 5. Centralized orchestrator with narrow sub-agents | ❌ MISSING | Monolithic script, no sub-agents |
| 6. Anchor all objectives in explicit task-specific rubrics | ❌ MISSING | Single factual-accuracy metric only |
| 7. Conservatively bound ensemble size, decoding freedom, evaluation population | ✅ PARTIAL | Fixed population size; decoding freedom severely limited |
| 8. Continuously monitor for diversity and regressions | ❌ MISSING | No diversity monitoring, no regression detection |

---

## Critical Dependencies

| # | Dependency | Needed For | Current State |
|---|---|---|---|
| D1 | **Cell formalization** (dataclass/registry) | Every sub-agent, routing, Pareto front, experiment tracking | MISSING — foundational blocker |
| D2 | **Multi-task evaluation data** | Evaluation agent, Pareto front across all 5 task types | MISSING — only factual QA data is used in GEPA |
| D3 | **Evaluation agent** with task-specific metrics | Any multi-task optimization | MISSING — must build before multi-objective optimization can extend |
| D4 | **Experiment tracking (MLflow or similar)** | Governance agent, audit trail, cell versioning | MISSING — no way to track history |
| D5 | **Routing table** (persisted cell selection) | Routing agent, governance agent, live inference | MISSING — no bridge between GEPA output and pipeline |
| D6 | **Analysis/diagnostics agent** | Feedback loop, mutation biasing, regression detection | MISSING — required for iterative improvement |
| D7 | **Deterministic validators as services** (NER, sentiment, format) | Evaluation agent metrics, downstream validation | PARTIAL — solvers exist but not integrated as validators |
| D8 | **Decoding config expansion** (top-p, top-k, min-p, repeat_penalty, seed) | Mutation agent, cell diversity | PARTIAL — only temperature configurable |

---

## Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **8GB RAM constraint** — loading 4 models simultaneously for GEPA eval may exceed budget | HIGH | ModelCache in gepa_runner already shares instances; but adding cross-cell model sharing (proposed in brief) is critical |
| R2 | **Evaluation latency** — full GEPA eval on 4 models × multiple cells × multiple tasks could take hours CPU-only | HIGH | Need budgeted evaluation, smaller populations per generation, and metric-based early stopping |
| R3 | **Cell formalization scope creep** — designing a general Cell abstraction risks over-engineering | MEDIUM | Start minimal: `(task_type, model_key, prompt, temp, top_p, max_tokens, aggregation)` as simple dataclass; extend later |
| R4 | **No held-out test set** — factual QA has only 19 training samples; no validation/test split exists | HIGH | Split existing data or generate synthetic benchmarks. 19 samples is far too few for statistical significance |
| R5 | **Pipeline and GEPA are disconnected** — GEPA output has no path into the production routing pipeline | MEDIUM | Build a routable "cell registry" that pipeline can query; initial integration via a JSON config file |
| R6 | **Agent orchestration complexity** — building 6 sub-agents with message-passing in a container risks deadlock and memory leaks | MEDIUM | Start with 2 agents (orchestrator + evaluation); use in-process queues; add agents incrementally |
| R7 | **spaCy/ML models add memory** — loading additional NLP models for validation competes with LLMs for RAM | MEDIUM | Use tiny models (spaCy sm/en_core_web_sm ~15MB); lazy-load only on demand |

---

## Build Order Recommendations

### Phase 0 — Foundations (Build now, unlock everything else)

1. **(D1) Cell formalization** — Create a `Cell` dataclass: `(task_type: str, model_key: str, system_prompt: str, temperature: float, top_p: float, max_tokens: int, aggregation: str, metadata: dict)`. Create a `CellRegistry` for persistent storage (JSON-backed or SQLite).
2. **(D2) Multi-task evaluation data** — Expand GEPA to use all 5 task types from `training-v3.json`. Create task-specific grading functions (math: sympy-based, sentiment: exact-match, NER: F1 via partial match, summarization: length + keyword coverage).
3. **(D4) Experiment tracking** — Add MLflow with local SQLite backend. Log every cell evaluation, Pareto front snapshot, and routing decision.
4. **(D5) Routing table** — Create a `RoutingTable` that maps `(category, complexity_bucket) -> cell_id`. Expose via JSON config that `pipeline.py` can read.

### Phase 1 — Agent Core (Build next)

5. **(D3) Evaluation Agent** — Extract `evaluate_population()` into a proper agent with a `run_evaluation(cells, task_data) -> EvaluationResult` interface. Support multi-metric evaluation.
6. **(D8) Decoding config expansion** — Add top-p, top-k, min-p, repeat_penalty to Cell. Add mutation operators for each.
7. **Pareto front → routing table bridge** — After GEPA converges, automatically update the RoutingTable with the Pareto-optimal cells.

### Phase 2 — Intelligence (Build after core agents work)

8. **(D6) Analysis Agent** — Create an agent that compares cell outputs vs. expected answers, labels failure modes (verbose, hedging, wrong-entity, format-violation), and produces a `DiagnosticReport`.
9. **Mutation biasing** — The Mutation Agent reads DiagnosticReport and biases genetic operators toward fixing observed failure modes.
10. **Backtesting & regression guards** — Implement champion/challenger: before routing table update, run the proposed cell on a held-out backtesting set. Roll back if metrics regress.

### Phase 3 — Observability & Governance

11. **Governance agent** — Wraps experiment tracker, backtesting, and diversity metrics. Enforces "no routing update without multi-objective improvement."
12. **Dashboard / reporting** — Generate human-readable reports from MLflow data: Pareto front evolution, per-model accuracy trends, cell diversity heatmaps.
13. **Convergence criteria** — Replace simple top-3 accuracy delta with comprehensive convergence: Pareto front volume change, diversity index stagnation, budget exhaustion.

### Phase 4 — Optional Enhancements

14. **Non-LLM classifiers** — Train a tiny BERT/NER classifier for task routing; integrate spaCy for validation.
15. **Online self-tuning** — If latency budget allows, run lightweight GEPA cycles during low-traffic periods.
16. **Decentralized agent mesh** — Only if Phase 0-3 proves the centralized orchestrator is a bottleneck.

---

## File-by-File Component Map

| File | Component | Status | Brief Section |
|---|---|---|---|
| `agent/gepa_runner.py` (908L) | GEPA optimizer: Pareto, NSGA-II, mutation, crossover, eval loop | ✅ EXISTS (factual-only) | Sec 1, 3, 4 |
| `agent/pipeline.py` (735L) | Production routing pipeline | ✅ EXISTS | Sec 2 (routing agent) |
| `harness.py` (98L) | CLI entrypoint | ✅ EXISTS | Sec 6 |
| `agent/generate_generation_0.py` (244L) | Gen-0 seed population creator | ✅ EXISTS | Sec 1, 4 |
| `agent/pre_filter.py` | Stage 0 bypass | ✅ EXISTS | Sec 5 |
| `agent/category_filter.py` (721L) | 8-way heuristic classifier | ✅ EXISTS | Sec 5 |
| `agent/complexity.py` (135L) | MiniLM-L6-v2 complexity scorer | ✅ EXISTS | Sec 5 |
| `agent/solvers/deterministic.py` (1897L) | 7 deterministic solvers | ✅ EXISTS | Sec 5, 6 |
| `container/runner.py` (229L) | Container pipeline orchestration | ✅ EXISTS | Sec 6 |
| `container/consensus.py` (262L) | 4-strategy voting/consensus | ✅ EXISTS | Sec 5 |
| `container/inference.py` (102L) | Parallel llama-server inference | ✅ EXISTS | Sec 6 |
| `container/prompts_config.json` (81L) | 4-strategy prompt config | ✅ EXISTS | Sec 1 |
| `Dockerfile` (42L) | Container build | ✅ EXISTS | Sec 6 |
| `ParetoMethodology.md` (601L) | Pareto methodology doc | ✅ EXISTS | Sec 3 |
| `gepa_plans/generation_0_results.json` | Gen-0 eval results | ✅ EXISTS | Sec 4 |
| — | Cell formalization (dataclass) | ❌ MISSING | Sec 1 |
| — | Agentic sub-agents (6 agents) | ❌ MISSING | Sec 2 |
| — | Analysis/diagnostics agent | ❌ MISSING | Sec 2, 4 |
| — | Mutation biasing from diagnostics | ❌ MISSING | Sec 2, 4 |
| — | Routing table | ❌ MISSING | Sec 2, 5 |
| — | Governance agent | ❌ MISSING | Sec 2, 4 |
| — | Experiment tracking (MLflow) | ❌ MISSING | Sec 6 |
| — | Task-specific metrics (math, NER, sentiment, summarization) | ❌ MISSING | Sec 3 |
| — | Non-LLM validators as services | ❌ MISSING | Sec 5 |
| — | Decoding param expansion | ❌ MISSING | Sec 1 |
| — | Backtesting / regression guards | ❌ MISSING | Sec 4, 7 |
| — | Diversity enforcement on Pareto front | ❌ MISSING | Sec 3, 7 |
| — | Orchestrator / workflow engine | ❌ MISSING | Sec 6 |
