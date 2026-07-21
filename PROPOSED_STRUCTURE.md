# AMD Hackathon — Project Structure Audit & Cleanup Proposal

> Generated: 2026-07-13  
> Current branch: `v12d`  
> Total root items: 77 files + 30 directories

---

## 1. Current State Assessment

### 1.1 Existing Directory Map

```
amd-hackathon/
├── agent/                    # Core agent: pipeline, classifiers, routing, GEPA runners
├── archive/                  # Old classifiers
├── archived/                 # LoRA artifacts (DUPLICATE of archive/)
├── colab/                    # Colab-specific pipeline + results
├── config/                   # JSON configs (mcporter, minilm, skills-lock)
├── container/                # Container inference runtime (consensus, server, runner)
├── cuda-wheels/              # llama_cpp_python prebuilt wheel (~220MB)
├── data/                     # Datasets (classifier, eval, facts, raw, training) — 473MB
│   ├── classifier/
│   ├── eval/
│   ├── facts/
│   ├── raw/
│   └── training/
├── deep_research/            # Deep research docs (DUPLICATE of research/ + docs/research/)
├── docs/                     # Documentation (architecture, eval, handoffs, plans, research)
│   ├── architecture/
│   ├── eval/
│   ├── handoffs/             # Session handoffs
│   ├── plans/                # Plans
│   └── research/             # Research docs
├── eval_results/             # Evaluation result files (xlsx, json)
│   ├── archive/
│   └── v12e/
├── gepa_logs/                # GEPA experiment logs
├── gepa_plans/               # GEPA experiment scripts + results (~1.2MB, 50+ files)
├── input/                    # Input data files (cx_300.json, dev_40.json, etc.)
├── models/                   # GGUF model files (~2.9GB)
├── output/                   # Empty output directory
├── presentation/             # Router_to_Vibehalla.mp4/pdf/pptx (~2MB)
├── prompts_backup/           # Backup copies of prompt files (archive material)
├── references/               # External reference docs
├── reports/                  # Evaluation reports (HTML, JSON, MD)
│   └── timing/
├── research/                 # Research findings (MD)
├── results/                  # Colab run CSVs/JSONs (DUPLICATE of eval_results/)
├── runner/                   # Runner code (batch_runner, evaluate, deploy)
│   ├── report_templates/
├── scripts/                  # Utility scripts (build data, eval, compare, smoke test)
├── staging/                  # Production pipeline (ready_pool, ready_worker, ready_judge)
│   └── workers/
└── tests/                    # Test files (test_batch_runner, test_evaluate, test_staging)
```

### 1.2 Root-Level Clutter (77 items)

| Category | Count | Examples |
|---|---|---|
| **Python scripts** | 44 | `eval_*.py`, `benchmark_*.py`, `analyze_*.py`, `research_round2*.py`, `test_*.py` |
| **Markdown docs** | 18 | `CONTEXT.md`, `HANDOFF.md`, `handoff_report.md`, `SESSION_REPORT_*.md`, `BUILD_PLAN.md`, `ParetoMethodology.md`, `bug_fixes_*.md`, `smoke_test_results_*.md`, `gpu_*_report*.md` |
| **JSON files** | 4 | `gpu_run_pure_results.json`, `gpu_run_pure_results_detailed.json`, `gpu_run_pure_timing.json`, `run_counter.json` |
| **Log files** | 2 | `staging_gpu_run_80.log`, `staging_gpu_run_pure.log` |
| **Config/build** | 8 | `Dockerfile`, `Dockerfile.cpu`, `Dockerfile.gpu`, `Dockerfile.staging`, `Makefile`, `LICENSE`, `requirements.txt`, `.gitignore`, `.dockerignore` |
| **README** | 1 | `README.md` |
| **Stray files** | 4 | `json` (17MB PostScript misnamed file), `=3.1.0`, `=3.3.2`, `AMD Hackathon Judging FAQ and Self-Check Guide.docx` |
| **Symlinks** | 4 | `bin → usr/bin`, `lib → usr/lib`, `lib64 → usr/lib64`, `sbin → usr/sbin` (should not be in repo) |

### 1.3 Redundancy & Duplication

- **`archive/`** and **`archived/`** — two separate archive directories, same purpose
- **`deep_research/`** vs **`research/`** vs **`docs/research/`** — three locations for research
- **`reports/`** vs **`eval_results/`** vs **`results/`** — three locations for results
- **`docs/handoffs/`** contains 3 handoff files, but root also has `HANDOFF.md` and `handoff_report.md`
- **`docs/plans/`** has plans, root also has `BUILD_PLAN.md`, `ner_fix_plan.md`
- **`gepa_plans/`** is a messy grab-bag of 50+ scripts, results, and plans unrelated to "plans"
- **`prompts_backup/`** is a snapshot that belongs in `archive/`
- **`staging/ready_pool.py.bak.phase1`** — orphaned backup

### 1.4 Git Branches

| Branch | Status |
|---|---|
| `v12d` | **CURRENT** active branch |
| `master` | Old, probably merged |
| `master-archive` | Archive branch |
| `gemma-4-e4b` | Experiment |
| `parallelization` | Experiment |
| `self-consistency-voting` | Experiment |
| `solvers-experiment` | Experiment |
| `stage1-features` | Experiment |
| `origin/master` | Remote master |
| `origin/v12d` | Remote v12d |
| `origin/v5.1-clean` | Remote old version |
| `origin/v6.5` | Remote old version |
| `origin/stage2-worktree/stage2-8way` | Remote worktree |

Worktrees: Only the main checkout exists on disk. No `k2p7` directory found anywhere.

---

## 2. Proposed Clean Structure

```
amd-hackathon/
│
├── agent/                          # [KEEP] Core agent code
│   ├── solvers/                    #   deterministic, LLM-based solvers
│   ├── pipeline.py
│   ├── classifier.py
│   ├── category_filter.py
│   ├── gepa_category_runner.py     #   GEPA evolutionary runner
│   └── ... (other agent modules)
│
├── runner/                         # [KEEP] Evaluation/execution runners
│   ├── evaluate.py
│   ├── batch_runner.py
│   ├── deploy.py
│   └── report_templates/
│
├── staging/                        # [KEEP] Production pipeline
│   ├── ready_pool.py
│   ├── ready_worker.py
│   ├── ready_judge.py
│   └── workers/
│
├── container/                      # [KEEP] Container runtime
│
├── scripts/                        # [ENHANCE] Merge ALL 44 root .py scripts here
│   ├── eval/                       #   eval_classifiers.py, eval_prompt_ablation*.py, etc.
│   ├── benchmarks/                 #   benchmark_*.py
│   ├── analysis/                   #   analyze_*.py, grade_results.py, extract_mmlu_keys.py
│   ├── tools/                      #   harness.py, multi_runner.py, dispatcher.py, tune_secondary.py
│   └── research/                   #   research_round2*.py
│
│── config/                         # [KEEP] Configuration
│
├── data/                           # [KEEP] Datasets
│
├── tests/                          # [KEEP] Test suite
│
├── models/                         # [KEEP] GGUF model files
│
├── input/                          # [KEEP] Input eval data
│
├── output/                         # [KEEP] Output directory
│
├── REPORTS/                        # [CREATE] All eval results, performance reports
│   ├── sessions/                   #   Session reports (SESSION_REPORT_*.md)
│   ├── gpu/                        #   GPU eval results + logs
│   ├── smoke_tests/                #   Smoke test results
│   └── timing/                     #   Timing analysis (from reports/timing/)
│
├── HANDOFFS/                       # [CREATE] Session-to-session handoff docs
│   ├── v12c-to-v12d-handoff.md     #   (from docs/handoffs/)
│   ├── v12e-session-handoff.md
│   ├── v12h-session-handoff.md
│   ├── HANDOFF.md                  #   (from root)
│   ├── handoff_report.md           #   (from root)
│   ├── bug_fixes_applied_*.md      #   Fix reports
│   ├── judge_fix_applied_*.md
│   └── scheduling_fix_applied_*.md
│
├── RESEARCH/                       # [CREATE] All research findings
│   ├── ParetoMethodology.md        #   (from root)
│   ├── summarization_research_findings.md
│   ├── open_source_model_thoughts.md  # (renamed from "I wonder if...")
│   ├── Best Models & Solutions.md   #   (from deep_research/)
│   ├── Deep Research Brief_*.md
│   ├── GAP_ANALYSIS.md
│   ├── ... (all deep_research/*)
│   └── notes/                      #   Smaller research notes
│
├── PLANS/                          # [CREATE] Architecture & implementation plans
│   ├── BUILD_PLAN.md               #   (from root)
│   ├── ner_fix_plan.md             #   (from root)
│   ├── ASSEMBLY_PLAN.md            #   (from docs/plans/)
│   ├── EVAL_SYSTEM_PLAN.md
│   ├── SUBMISSION_CONTAINER_PLAN.md
│   ├── STAGING_HANDOFF.md
│   ├── ... (all docs/plans/*.md)
│   └── tasks/                      #   (ground_truth.txt, tasks.txt, etc.)
│
├── EXPERIMENTS/                    # [CREATE] Experiment artifacts
│   ├── gepa/                       #   (merge gepa_logs/ + gepa_plans/)
│   ├── colab/                      #   (from colab/)
│   └── ablation/                   #   Ablation study results
│
├── DOCS/                           # [KEEP but prune] Project documentation
│   ├── architecture/               #   Architecture docs
│   ├── eval/                       #   Eval infrastructure docs
│   └── README.md                   #   Index
│
├── archive/                        # [CONSOLIDATE] Single archive dir
│   ├── old_classifiers/            #   (from archive/classifiers)
│   ├── lora/                       #   (from archived/lora)
│   └── prompts_backup/            #   (from prompts_backup/)
│
├── references/                     # [KEEP] Reference materials
│   ├── alternative-plans-B-C.md
│   └── AMD Hackathon Judging FAQ and Self-Check Guide.docx
│
├── presentation/                   # [KEEP or → docs/] Presentation assets
│
├── MEMORY.md                       # [NEW] Running project memory (replaces CONTEXT.md)
│
├── README.md                       # [KEEP] Updated
├── LICENSE                         # [KEEP]
├── Makefile                        # [KEEP]
├── requirements.txt               # [KEEP]
│
├── Dockerfile                      # [KEEP] Container definitions
├── Dockerfile.cpu
├── Dockerfile.gpu
├── Dockerfile.staging
│
├── .gitignore                      # [UPDATE]
├── .dockerignore                   # [KEEP]
│
└── ─── [REMOVE] ─────────────────────
    ├── json                        #   17MB misnamed PostScript → delete or archive
    ├── =3.1.0                      #   Version marker → archive
    ├── =3.3.2                      #   Version marker → archive
    ├── bin/ → usr/bin              #   Accidental symlinks → delete
    ├── lib/ → usr/lib
    ├── lib64/ → usr/lib64
    ├── sbin/ → usr/sbin
    ├── __pycache__/                #   Python cache → delete
    └── .pytest_cache/              #   Test cache → delete
```

---

## 3. File-by-File Migration Plan

### 3.1 Root `.py` → `scripts/` (44 files)

**→ `scripts/eval/` (15 files)**
```
eval_classifiers.py
eval_coder_ablation.py
eval_ensemble_router.py
eval_final_config.py
eval_final_run.py
eval_ner_all_models.py
eval_prompt_ablation.py
eval_prompt_ablation_r2.py
eval_prompt_ablation_r3.py
eval_round5.py
eval_round6.py
eval_smollm_llama.py
eval_smollm_llama_cot.py
eval_specialist_router.py
eval_three_models.py
```

**→ `scripts/benchmarks/` (7 files)**
```
benchmark_code_quality.py
benchmark_math.py
benchmark_math_v2.py
benchmark_ml_cascade.py
benchmark_sentiment.py
benchmark_sentiment_v2.py
benchmark_sentiment_v3.py
benchmark_sweep.py
```

**→ `scripts/analysis/` (6 files)**
```
analyze_data.py
analyze_data_deep.py
analyze_data_extra.py
analyze_json_report.py
analyze_sentiment_data.py
grade_results.py
extract_mmlu_keys.py
```

**→ `scripts/tools/` (5 files)**
```
harness.py
multi_runner.py
dispatcher.py
tune_secondary.py
inspect_training_v3.py
```

**→ `research/` (3 files)**
```
research_round2.py
research_round2_v2.py
research_round2_v3.py
```

**→ `tests/` (4 files)**
```
test_code_validation.py
test_gpu_pipeline.py
test_regex_solver.py
test_summarization_approaches.py
```

**→ `scripts/gpu/` (2 files)**
```
comprehensive_gpu_eval.py
gpu_smoke_test.py
```

### 3.2 Root `.md` → Categorized (18 files)

| Current | Destination |
|---|---|
| `CONTEXT.md` | **Replace with** `MEMORY.md` |
| `BUILD_PLAN.md` | `plans/BUILD_PLAN.md` |
| `HANDOFF.md` | `handoffs/HANDOFF.md` |
| `handoff_report.md` | `handoffs/handoff_report.md` |
| `SESSION_REPORT_20260713.md` | `reports/sessions/20260713.md` |
| `SESSION_REPORT_20260713_Build3.md` | `reports/sessions/20260713_build3.md` |
| `ParetoMethodology.md` | `research/ParetoMethodology.md` |
| `ner_fix_plan.md` | `plans/ner_fix_plan.md` |
| `summarization_research_findings.md` | `research/summarization_research_findings.md` |
| `I wonder if there are some uh good uh open source.md` | `research/open_source_model_thoughts.md` |
| `bug_fixes_applied_20260713_162200.md` | `handoffs/bug_fixes_20260713.md` |
| `judge_fix_applied_20260713_081108.md` | `handoffs/` |
| `judge_fix_applied_20260713_081111.md` | `handoffs/` |
| `scheduling_fix_applied_20260713_082049.md` | `handoffs/` |
| `smoke_test_results_20260713_081044.md` | `reports/smoke_tests/` |
| `smoke_test_results_20260713_081107.md` | `reports/smoke_tests/` |
| `gpu_80_eval_results_20260713_0814.md` | `reports/gpu/` |
| `gpu_pure_graded_report_20260713_081735.md` | `reports/gpu/` |

### 3.3 Root JSON, Logs, Other

| Current | Destination |
|---|---|
| `gpu_run_pure_results.json` | `reports/gpu/` |
| `gpu_run_pure_results_detailed.json` | `reports/gpu/` |
| `gpu_run_pure_timing.json` | `reports/gpu/` |
| `run_counter.json` | `config/` (already one there) |
| `staging_gpu_run_80.log` | `reports/gpu/` |
| `staging_gpu_run_pure.log` | `reports/gpu/` |
| `AMD Hackathon Judging FAQ and Self-Check Guide.docx` | `references/` |
| `json` (17MB PostScript) | Remove (misnamed, belongs nowhere) |
| `=3.1.0`, `=3.3.2` | Remove or archive |
| `bin`, `lib`, `lib64`, `sbin` symlinks | Remove from repo |
| `__pycache__/` | Delete |
| `.pytest_cache/` | Delete |

---

## 4. Project Memory System

Replace `CONTEXT.md` with a structured `MEMORY.md` file using this template:

```markdown
# Project Memory

## Identity
- **Project**: AMD Hackathon — Multi-Model Agentic Pipeline
- **Goal**: Build a classification-routing pipeline for Kaggle's AMD Hackathon
- **Active Branch**: `v12d`

## Current State
- V12 pipeline running end-to-end on GPU
- GEPA evolutionary optimization producing candidate prompts
- Staging infrastructure (ready_pool/ready_worker) deployed

## Architecture Decisions
| ID | Decision | Rationale | Date |
|---|---|---|---|
| ADR-001 | Use GGUF + llama.cpp for local inference | GPU-only container, no HF deps | 2026-07-10 |
| ADR-002 | Genetic-Pareto (GEPA) prompt evolution | Outperforms manual tuning | 2026-07-11 |
| ... | | | |

## Active Experiments
- GPU pure eval run (v12d baseline)
- GEPA category runner (5-task optimization)

## Known Issues
- [ ] Parallelization analysis identifies 3 bottlenecks
- [ ] Module integration audit (14 unintegrated modules)

## Session Log (Recent)
| Date | Focus | Outcome |
|---|---|---|
| 2026-07-13 | GPU staging run | Pure v12d evaluated, results in reports/gpu/ |
| 2026-07-12 | NER fix, prompt ablation | 3 ablation rounds completed |

## Key Contacts
- Kaggle competition page: <link>
- Team: [members]
```

---

## 5. Consolidation Summary

| Action | Count | Details |
|---|---|---|
| **KEEP as-is** | ~15 dirs | agent/, runner/, staging/, container/, config/, data/, tests/, models/, input/, output/, references/, presentation/, docs/, scripts/ |
| **NEW directories** | 5 | `reports/`, `handoffs/`, `research/`, `plans/`, `experiments/` |
| **MERGE into new dirs** | 44 .py + 18 .md + 6 .json/.log | All root clutter categorized |
| **CONSOLIDATE duplicates** | 4 pairs | archive/ + archived/, deep_research/ + research/, results/ + eval_results/, gepa_logs/ + gepa_plans/ → experiments/gepa/ |
| **DELETE** | 8 items | `json`, `=3.1.0`, `=3.3.2`, `bin`, `lib`, `lib64`, `sbin` symlinks, `__pycache__/` |
| **REMOVE from git (but keep locally)** | ? | `cuda-wheels/` (220MB wheel), `.venv/` |
| **UPDATE** | 3 files | `README.md`, `.gitignore`, `MEMORY.md` (new) |
| **STALE branches** | 6 local | gemma-4-e4b, master-archive, parallelization, self-consistency-voting, solvers-experiment, stage1-features — archive or delete |
| **Worktrees** | 0 | No `amd-hackathon-k2p7` directory exists on disk |

---

## 6. Recommended Execution Order

1. **Backup current state** (git tag `pre-refactor` or stash)
2. **Create target directories**: `reports/`, `handoffs/`, `research/`, `plans/`, `experiments/gepa/`
3. **Move root `.md` files** to their new homes (git mv preserves history)
4. **Move root `.py` scripts** into `scripts/{eval,benchmarks,analysis,tools,gpu}`
5. **Move JSON/log/other data** into `reports/gpu/`
6. **Consolidate duplicate directories**: merge `archived/` → `archive/`, `deep_research/` → `research/`, `results/` → `eval_results/`, `gepa_*` → `experiments/gepa/`
7. **Remove clutter**: delete stray files, symlinks, caches
8. **Create `MEMORY.md`** from CONTEXT.md content
9. **Update `.gitignore`** to exclude cache dirs, large wheels, model files
10. **Update `README.md`** to reflect new structure
11. **Delete stale local git branches** or push them to a `git archive/` tag
12. **Commit** the restructuring as a single refactor commit
