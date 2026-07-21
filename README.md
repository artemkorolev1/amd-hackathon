# AMD ACT II — Track 1: Hybrid Token-Efficient Routing Agent

A submission for the **AMD ACT II Hackathon (Track 1)** — a token-efficient, hybrid routing agent that classifies tasks across 8 categories and solves them using a pipeline of deterministic solvers, cascade classifiers, and local LLM inference.

**Repository:** `artemkorolev1/amd-hackathon` (this repo)  
**Docker image:** `ghcr.io/artemkorolev1/amd-hackathon-submit`  
**CPU submission container:** tagged `:cpu` — optimized for 2 vCPU / 4 GB grader environment  
**Team:** Raiders of Vibehalla | **Project:** Router to Vibehalla

---

## Architecture

```
                               ┌──────────────────────────────────────┐
                               │          /input/tasks.json           │
                               │        (JSON array of strings)       │
                               └────────────────┬─────────────────────┘
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Stage 2: 8-Way Classifier Cascade                     │
│  Primary: 8-way scorer (92.2% accuracy) + 4 secondary resolvers               │
│  ├─ code_secondary → code_debug vs code_gen                                   │
│  ├─ reasoning_secondary → logic vs math                                       │
│  ├─ factual_secondary → factual vs logic/math                                 │
│  └─ summarization_secondary → summarization vs math/code/logic/factual        │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Complexity Scoring (MiniLM + LogisticRegression)            │
│                     Spearman ρ = 0.69 — 7-signal bitmorphic fallback          │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Deterministic Solver Layer                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │  NER v3  │ │  Logic   │ │  Math    │ │Sentiment │ │ Factual  │ │ Code  │ │
│  │Regex+NLP │ │Proposition│ │Template  │ │Weighted  │ │ FactDB   │ │Cascade│ │
│  │          │ │Zebra, SAT │ │Solvers   │ │Keywords  │ │(112K ff) │ │Router │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LLM Inference Layer (local GGUF models)                    │
│   ┌──────────────────────────┐   ┌──────────────────────────────────┐        │
│   │  Qwen2.5-1.5B-Instruct  │   │  Qwen2.5-Coder-1.5B-Instruct     │        │
│   │  (general categories)   │   │  (code generation, math extract)  │        │
│   └──────────┬───────────────┘   └──────────────┬───────────────────┘        │
│              ▼                                    ▼                           │
│   ┌────────────────────────────────────────────────────────────────────┐     │
│   │              ToRA Solver (math reasoning via LLM planning          │     │
│   │               + deterministic PythonExecutor)                      │     │
│   │  ┌───────────┐  ┌──────────────┐  ┌───────────────┐               │     │
│   │  │ Variable  │→│  Single-shot │→│  Iterative    │               │     │
│   │  │Extraction │  │  ToRA       │  │  ToRA         │               │     │
│   │  │ Pre-filter│  │  (+consensus│  │  (sub-step    │               │     │
│   │  │           │  │   voting)   │  │   decomposition│               │     │
│   │  └───────────┘  └──────────────┘  └───────────────┘               │     │
│   └────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 ▼
                    ┌──────────────────────────────────┐
                    │       /output/results.json        │
                    │   (JSON array of answer strings)  │
                    └──────────────────────────────────┘
```

### Key Design Decisions

1. **Hybrid deterministic + ML classifier pipeline** — 8-way scorer cascade (92.2% accuracy) with 4 secondary resolvers for boundary cases. Zero LLM calls during classification.
2. **Deterministic where possible** — NER, sentiment, factual QA, basic logic, template math all solved without any model inference. Reduces token cost and latency.
3. **Local LLM only** — Qwen2.5-1.5B-Instruct + Qwen2.5-Coder-1.5B-Instruct GGUF models run locally via llama.cpp. No external API dependencies.
4. **ToRA pattern for math** — LLM generates step-by-step plan + Python expressions → deterministic `PythonExecutor` computes each step. Variable extraction pre-filter + consensus voting + iterative fallback.
5. **GEPA prompt optimization** — All 8 category prompts optimized via Genetic Pareto Algorithm for min-token, max-accuracy tradeoffs.
6. **Cell-based execution** — Modular pipeline cells with contracts, resource gates, and workflow orchestration.

---

## Solver Categories

| Category | Deterministic | LLM | Best Accuracy | Strategy |
|----------|:-------------:|:---:|:-------------:|----------|
| factual | ✓ | | 92.0% | FactDB FTS5 (112K facts, BM25-scored) |
| sentiment | ✓ | | 76.0% | Weighted keyword + VADER + cascade |
| ner | ✓ | | 80.0% | Regex + biomedical/general NER patterns (v3) |
| code_gen | partial | ✓ | 60.0% | Template solvers + Qwen2.5-Coder |
| code_debug | partial | ✓ | 100% | Pattern matching + LLM for edge cases |
| logic | ✓ | | 26.0% | Propositional logic, zebra puzzles, BFS |
| math | templates + ToRA | ✓ | 100% train / 84.2% test | Template math + ToRA (extraction, single-shot, iterative, consensus voting) |
| summarization | | ✓ | Baseline | LLM-only with two-step entity extraction |

**Overall deterministic-only accuracy:** 36.0% (training-v2), 31.8% (validation-v2)  
**With LLM (Qwen2.5-1.5B):** ~75% overall, **v12d Nemotron-3-Nano-4B:** 93.7% on 300-set eval

---

## Recent Developments (v12d → v12e+)

### v12d — Core Pipeline Overhaul

- **Cascade solvers for all 8 categories** — deterministic classification cascades with format normalization and tool-based dispatch for sentiment, NER, logic, math, code (debug + gen), factual, summarization
- **Cell runner + contract system** — modular execution cells with typed contracts, resource management, and pipeline orchestration
- **Workflow gates** — per-category workflow gates for multi-step reasoning pipelines
- **GEPA prompt evolution** — Genetic Pareto Algorithm optimizing prompts across all 8 categories for min-token, max-accuracy. Results: 100% sentiment & code_debug, 60% code_gen, 50% math, 36.8% NER (format ceiling)
- **Code reorganization** — root-level scripts migrated to `scripts/{analysis,benchmarks,eval,tools}/`
- **Staging system** — parallel worker pool, ready judge, resource-aware dispatch, GPU eval tools
- **Comprehensive eval infrastructure** — 300-question eval sets, pipeline scoring, cascade evaluations, model comparisons

### v12e — Math Pipeline Expansion

- **5 new math templates**: rate_work, buy_sell, pct_of_total, group_sharing, pop_growth (total: 11 templates)
- **Template extraction via Qwen2.5-Coder** — better JSON structure for math parameter extraction
- **T11 type A/B prompts** — anti-hallucination guard for template solver
- **Temperature experiments** — temp=0.3 for extraction balanced accuracy vs regression
- **Validation & broader pattern matching** — improved template solver recall

### Latest Uncommitted Work — ToRA Math Solver Enhancements

- **Variable extraction pre-filter** — `solve_with_extraction()` extracts structured variables from word problems before ToRA, improving accuracy on multi-variable problems
- **Consensus voting** — runs ToRA at 4 temperatures (0.1–0.9), normalizes answers by numeric extraction, majority vote with ≥40% agreement threshold
- **Iterative ToRA fallback** — when single-shot ToRA fails, decomposes into sub-steps solved independently with intermediate results forwarded
- **Expanded ToRA examples** — discount/percentage and multi-step examples added to prevent library import mistakes
- **Coding challenge categories** — 4 new subcategories (dp, ds, formal, sort_search) added to max_tokens and stop sequences
- **FactDB expansion** — 21K → 112K facts by integrating NQ-Open (87,925 train + 3,610 dev), 46.8MB FTS5 database

---

## Docker Build

### CPU Submission Container (grader target)

```bash
make cpu-build    # docker buildx --platform linux/amd64 -t ghcr.io/.../amd-hackathon-submit:cpu
make cpu-push     # tag + push to GHCR
make cpu-run      # docker run with /input /output mounts
```

3 models included (Qwen2.5-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct, Gemma-3-1B-IT — all Q4_K_M GGUF, ~2.6 GB total).  
CPU-safe defaults: `N_GPU_LAYERS=0, N_THREADS=2, N_CTX=2048`.

### GPU Container

```bash
docker build -t amd-agent -f Dockerfile.gpu .
```

### Staging Container

```bash
docker build -t amd-agent -f Dockerfile.staging .
```

### Make Targets

| Target | Description |
|--------|-------------|
| `build` | Standard Docker build |
| `run` | Run with mounted /input /output |
| `local-test` | Run agent directly (requires llama.cpp server) |
| `push` | Push to GHCR |
| `deploy` | Build + push + verify |
| `cpu-build` | CPU submission container (linux/amd64) |
| `cpu-push` | Push CPU container to GHCR |
| `staging-test` | Test judge module |
| `staging-run` | Run staging entrypoint |
| `evaluate` | Evaluate results against ground truth |

---

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `ALLOWED_MODELS` | No | Comma-separated list of allowed model codenames |
| `DEADLINE_S` | No | Pipeline timeout in seconds |
| `MODEL_PATH` | No | Path to GGUF model (default: `/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`) |
| `LLAMA_N_CTX` | No | Context size for llama.cpp (default: `2048`) |
| `FIREWORKS_API_KEY` | No | Optional Fireworks API key (archived — not used in current pipeline) |
| `N_GPU_LAYERS` | No | GPU layers for llama.cpp (default: `0` for CPU) |
| `N_THREADS` | No | CPU threads (default: `2`) |
| `CONSENSUS_SAMPLES` | No | ToRA consensus samples (default: `1` — no voting) |

---

## Project Structure

```
├── agent/                    # Core pipeline
│   ├── main.py               # Entrypoint
│   ├── pipeline.py           # Pipeline orchestrator (ToRA, cascade, LLM dispatch)
│   ├── pipeline_gepa.py      # GEPA-integrated pipeline
│   ├── cell.py / cell_runner.py  # Modular execution cells
│   ├── dynamic_prompts.py    # Per-category LLM prompts (GEPA-optimized)
│   ├── classifier.py         # 8-way classifier cascade
│   ├── category_filter.py    # Category router
│   ├── config.py             # Configuration
│   ├── contracts.py          # Cell execution contracts
│   ├── workflow.py / workflow_gate.py  # Workflow orchestration
│   ├── resource_manager.py   # CPU/memory resource gates
│   ├── cells/                # Cell implementations
│   └── solvers/              # Solver implementations
│       ├── deterministic.py  # Main deterministic solver (134K, 3,476 lines)
│       ├── tora_solver.py    # ToRA math reasoning (LLM + PythonExecutor)
│       ├── iterative_tora.py # Iterative sub-step decomposition
│       ├── variable_extractor.py  # Variable extraction pre-filter
│       ├── stepwise_math.py  # Step-by-step math solver
│       ├── fact_db.py        # FTS5 fact database (112K facts)
│       ├── cascade_router.py # Cascade routing
│       ├── code_tool_cascade.py / code_tool_router.py
│       ├── math_classifier.py / math_tool_router.py
│       ├── math_step_classifier.py / math_binary_step_classifier.py
│       ├── sentiment_cascade.py / sentiment_tree.py / sentiment_hybrid.py
│       ├── ner_classifier_cascade.py / ner_solver.py
│       ├── logic_classifier_cascade.py / logic_reasoning.py
│       ├── prototype_ner_v3.py / prototype_zebra_v2.py
│       └── summarization_solver.py
├── scripts/                  # Utility scripts
│   ├── analysis/             # Data analysis
│   ├── benchmarks/           # Benchmarking
│   ├── eval/                 # Evaluation framework
│   └── tools/                # Tool scripts
├── staging/                  # Parallel submission system
│   ├── entrypoint.py         # Submission entrypoint
│   ├── ready_pool.py         # Worker pool
│   ├── ready_worker.py       # Individual worker
│   ├── ready_judge.py        # Answer judge
│   └── ready_config.py       # Config
├── models/                   # GGUF model files (gitignored)
├── data/                     # Evaluation data, facts, training sets
│   ├── eval/                 # Eval datasets (training-v3, validation-v3, GSM8K splits)
│   └── facts/                # FTS5 fact databases (facts.db = 46.8MB, 112K facts)
├── docs/                     # Documentation
│   ├── architecture/         # System architecture diagrams
│   ├── handoffs/             # Session handoff documents
│   └── research/             # Research notes
├── results/                  # Run output
├── config/                   # Configuration
├── colab/                    # GEPA Evolution colab notebook
├── references/               # Reference architecture docs
├── research/                 # Research findings
├── gepa_plans/               # GEPA optimization plans & results
├── gepa_logs/                # GEPA evolution run logs
├── eval_results/             # Evaluation result files
├── Dockerfile                # Standard Docker build
├── Dockerfile.cpu            # CPU submission container (no GPU)
├── Dockerfile.gpu            # GPU-optimized container
├── Dockerfile.staging        # Staging/testing container
├── Makefile                  # Build automation
├── CONTEXT.md                # Shared session context (drop-box)
├── PROJECT_LOG.md            # Chronological project log
├── RETROSPECTIVE_REPORT.md   # Post-hackathon retrospective
├── HANDOFF-v12d-gepa-complete.md  # Full architecture handoff
└── PROPOSED_STRUCTURE.md     # Proposed refactoring structure
```

---

## I/O Contract

- **Input:** `/input/tasks.json` — JSON array of task strings, or objects with a `"prompt"` key
- **Output:** `/output/results.json` — JSON array of answer strings (one per task, in order)
- **Stdout:** One answer per line (for direct pipe usage)
- **Stderr:** Log output (timestamps, categories, progress)

---

## Constraints

| Constraint | Status |
|------------|--------|
| Max 10 GB compressed | ~4–5 GB expected |
| Startup < 60 s | ~15–20 s model load |
| Runtime < 10 min | ~2–6 min depending on complexity |
| CPU-only submission | llama.cpp on CPU, 0 GPU layers |
| 2 vCPU / 4 GB RAM grader | Thread-limited, small context |

---

## License

MIT License — see [LICENSE](LICENSE).
