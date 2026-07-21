# AMD Hackathon — Work Cell Review & Improvement Analysis

Date: 2026-07-15
Analysis of the full system at /home/artem/dev/amd-hackathon/

---

## SYSTEM OVERVIEW

Two parallel architectures exist:

**A. Production Pipeline (main.py + pipeline.py)** — Sequential single-path flow:
  Pre-Filter → Classifier → Complexity → Decision → Solvers → QC → Output

**B. Staging Parallel System (ready_pool.py)** — Multiprocessing multi-worker flow:
  Classify → Partition into per-type pools → Parallel workers → Judge aggregation

Plus the **GEPA Optimization Framework** that evolves cells (prompts + params) via
genetic operators and publishes into the Routing Table.

---

## 1. PRE-FILTER (agent/pre_filter.py)

### What it does
Deterministic regex-based bypass before any ML cost. 3 tiers:
- T0: Greetings → direct answer, Pure arithmetic → route to solver
- T1: Code fences, arithmetic expressions, summarization keywords → route to category

### Strengths
- Zero-cost filter that catches trivial cases (greetings, simple arithmetic)
- Priority resolution for multi-match (most specific → code > arithmetic > summarization)

### Issues & Improvements

**1a. Tier 0 returns "Hello! How can I help you today?" for greetings**
  In a benchmark/hackathon grader context, greeting prompts are never present.
  This 8-line code path never fires in production but wastes code complexity.
  → Remove or gate behind a DEBUG flag. The grader sends task prompts, not hellos.

**1b. RE_PURE_ARITH only handles single-operator binary expressions**
  "what is 12 + 5" works. But "what is 12 + 5 * 3" falls through to Tier 1 as
  arithmetic → Tier 1 only checks len < 15 words. Longer arithmetic expressions
  are missed entirely.
  → Extend to multi-operator or delegate to the deterministic arithmetic solver earlier.

**1c. Code detection is naive**
  RE_SINGLE_DEF only catches lines starting with def/function/class.
  Indented code blocks, multi-line defs, or code without def (scripts, list comprehensions
  in backticks) are missed. _is_all_code() counts keyword density but the threshold
  (0.5) is arbitrary.
  → Consider byte-level encoding heuristics (entropy, newline density) used by
    production code-vs-text classifiers.

---

## 2. CATEGORY CLASSIFIER (agent/classifier.py + secondary classes)

### What it does
Primary 8-way deterministic classifier (category_filter.py) → Secondary classifiers
for 4 known confusion pairs (code_debug vs code_gen, logic vs math, factual, summarization).

### Strengths
- Amazing engineering: secondary classifiers recover from primary errors without
  recursion or model calls. The summarization detector (secondary_summarization.py)
  is exceptionally thorough with 15+ regex signals, scoring heuristics, and
  counter-signal checks.
- The cascade structure (primary → secondary → keyword override) catches errors
  at progressively coarser granularity.

### Issues & Improvements

**2a. Primary classifier is a black-box import**
  category_filter.py is loaded via importlib to avoid broken __init__.py chains.
  The classifier has known scores (8-way deterministic, ~85% on 60-set) but its
  internal architecture isn't visible in this codebase — it could be a brittle
  keyword-frequency approach or a naive Bayes. No trainable parameters.
  → Document the algorithm. If it's keyword-count-based, consider switching to
    a small fast classifier (e.g., fastText, ONNX logistic regression) that can be
    retrained from the 300-set eval data.

**2b. Secondary classifier cascade is tightly coupled**
  The classifier.py module hard-imports secondary_*.py files by name. Adding a new
  secondary means modifying classify() and creating a new file. No plugin/discovery.
  → Registry-based pattern: secondaries register themselves for specific (category, condition).

**2c. No uncertainty propagation**
  The cascade replaces the category silently. If the primary was confident (Δ > 0.8)
  but the secondary overrides based on a weak signal, there's no confidence penalty.
  → Track override confidence: if primary confidence is high but a secondary overrides
    based on a low-score heuristic, flag the final answer as low-confidence.

**2d. Summary: the classifier is the single most brittle component**
  It's pure regex, no model training, no fallback if all secondaries disagree.
  If the 8-way primary misclassifies (e.g., marks a math problem as logic and
  secondary disagrees), there's no third opinion. Consider a simple 8-way logistic
  regression trained on the 60-set classifier data.

---

## 3. COMPLEXITY SCORER (agent/complexity.py / complexity_filter.py)

### What it does
MLM-based per-category complexity score (0.0 → 1.0). Used in the decision table to
route between deterministic solvers, local LLM, and Fireworks API.

### Strengths
- Single numeric score drives all routing decisions — elegant.
- Pre-warmed at pipeline init to avoid cold-start latency.

### Issues & Improvements

**3a. MLM is a hidden dependency**
  The model (named "mlm" from the `from agent.complexity import score` import) is
  loaded silently. Its size, inference time, and accuracy vs a simpler TF-IDF/linear
  model are unknown. If it's a full transformer MLM, it adds 100-500ms per call.
  → Benchmark against a fast alternative: log tokens+entropy+length as feature vector,
    a 5-feature logistic regression. If within 5% accuracy of the MLM, drop the MLM.

**3b. Single complexity score masks per-model complexity**
  "complexity = 0.3" means different things for math vs summarization vs code_gen.
  The decision table thresholds (simple_max=0.3, medium_max=0.6) are global, not
  per-category.
  → Per-category complexity thresholds from eval data (e.g., math simple_max=0.2,
    summarization simple_max=0.4). These can be learned from the 300-set.

**3c. Complexity is computed but not used in workflow_gate.py**
  workflow_gate.py receives complexity but the gate only uses it for "add_verify"
  on math (complexity > 0.5). All other categories ignore complexity for workflow
  selection.
  → If complexity is truly useful, use it everywhere. If not, remove the parameter.

---

## 4. DETERMINISTIC SOLVERS (agent/solvers/deterministic.py — 3280 lines)

### What it does
The largest file in the project. Contains regex/heuristic solvers for: arithmetic logic,
word problems (mean/median, speed/distance, unit cost, inclusion-exclusion, matrix determinants,
log equations), code generation, code debugging, factual QA, NER, summarization, sentiment.

### Strengths
- Comprehensive coverage of math word problem types (unit cost, speed/distance, mean/median,
  matrix det, log equations, inclusion-exclusion, remainder/root/percentage).
- Code debugging solver has actual Python AST validation.
- Sentiment solver pairs with VADER/pattern-based analysis.

### Issues & Improvements

**4a. 3280 lines in one file — violates single responsibility**
  The file has grown organically, mixing 11+ unrelated solver types. This makes
  testing, benchmarking, and isolated improvement of any single solver impossible
  without touching the whole file.
  → Split into solvers/math/ (arithmetic.py, word_problems.py, matrix.py, log_eq.py),
    solvers/code/, solvers/text/ etc. One file per solver family.

**4b. No solver-level unit tests**
  Each solver is a function that takes (prompt, category) → str | None. This is
  perfectly testable. Yet there are zero test cases for any solver.
  → Add tests/solvers/ with test cases from the 300-set eval data. Start with
    the high-coverage solvers (arithmetic, NER) and extend coverage from eval failures.

**4c. Matrix determinant solver is fragile**
  Relies on exact bracket-matching `[a b c]` per line or `[[a,b],[c,d]]` literal.
  Real prompts (from LogiQA, GSM8K, MMLU) don't format matrices this way.
  → If this solver has never produced a correct answer in the 300-set, remove it.
    It adds maintenance cost for zero value. Log the hit rate and prune by data.

**4d. No timeout on deterministic solvers**
  Several solvers import sympy (external library) for matrix det and log equations.
  sympy.solve() on malformed input can hang or take >30s.
  → Wrap sympy calls in a signal/timer timeout. Every deterministic solver should
    guarantee return within 2s.

**4e. Sentiment solver is VADER-based**
  Hard-depends on the `vaderSentiment` package. If it's not in the Docker image,
  the import fails silently (caught by the generic `except Exception` in pipeline.py).
  → Verify existence at import time with a clear error message. Better: make VADER
    a lazy import with a fallback to the regex/pattern-based sentiment classifier.

**4f. Code generation solver is suspicious**
  It "generates" code by... pattern-matching the prompt? The file lists
  `solve_code_generation` in the export list but I need to confirm what it actually does.
  If it's trying to generate Python code from a natural language spec via regex, it
  will almost certainly fail every time.
  → If code_gen solver never returns a valid answer, remove it from det_solvers list.
    Let code_gen always route to local LLM or Fireworks.

---

## 5. PROTOTYPE SOLVERS (prototype_ner_v3, prototype_zebra_v2, logic_reasoning)

### What it does
Specialized solvers for hard NER (entity extraction with {@...@} markers), zebra/logic
puzzles, and logical reasoning.

### Strengths
- The NER v3 solver addresses the critical grader requirement: {@...@} wrapping of
  exact entity text. This is the difference between 0% and 100% on the NER category.
- Zebra puzzle solver is a rare specialized asset — few competitors attempt this.

### Issues & Improvements

**5a. prototype_ner_v3 imports a training data file path**
  If the training/reference data isn't shipped in the Docker image, the solver silently
  degrades. This is a deployment landmine.
  → Fail loudly at import time if reference data is missing. The Docker build should
    fail rather than silently producing 0% NER scores.

**5b. No overlap detection with the generic NER solver**
  The pipeline tries generic NER first (from deterministic.py), then falls back.
  If generic NER returns a wrong answer (e.g., entities without {@...@} markers), the
  grader gets an invalid answer and the prototype never runs.
  → Remove the generic NER solver from the NER path entirely. Only use prototype_ner_v3
    for NER, since the grader requires {@...@} format.

---

## 6. LOCAL LLM INFERENCE (pipeline.py _infer, main.py)

### What it does
Loads 1-3 GGUF models via llama-cpp-python. Per-category model routing based on
evaluation results (Qwen2.5-1.5B for math/logic/factual/summarization, Qwen-Coder for
NER/code_debug, Gemma-3-1B for code_gen/sentiment). Consensus voting when k > 1.

### Strengths
- Multi-model per-category routing is empirically validated (300-set eval).
- Per-category LLM caching avoids reloading.
- Fallback: if category-specific model fails, uses the default model.

### Issues & Improvements

**6a. All models loaded into memory simultaneously**
  The category_model_map may point to 3 different GGUF files. If a task for category A
  is followed by category B, the first model stays in RAM while the second loads.
  With 3 models × 1-2GB each, this is 3-6GB RAM on a 4GB container.
  → Unload models after use if memory pressure detected. Or use a single multi-task
    model to avoid this entirely.

**6b. Consensus voting is disabled (CONSENSUS_SAMPLES=1)**
  The memory says "v12d: no parallelization (CONSENSUS_SAMPLES=1, WORKERS=1)".
  This means the sophisticated vote aggregation code runs a single sample and
  returns immediately — all the majority_vote/self_consistency/judge_select code
  in the cell system is dead code at runtime.
  → Either remove the dead code paths, or explain why consensus is off (latency?
    cost?). If it's useful for certain categories (math, logic), enable k=3 for
    those only.

**6c. NAKED_CATEGORIES = {ner, summarization, factual, logic, math}**
  These bypass all system prompts and deterministic solvers. The reasoning is that
  local LLMs perform worse with instruction prompts for these categories. But this
  means the carefully engineered dynamic_prompts.py (723 lines) is completely unused
  for 5 out of 8 categories.
  → Audit whether dynamic_prompts.py is still needed. If 5/8 categories bypass it,
    the system-prompt complexity may not justify its maintenance.

**6d. llama-cpp-python has no request queue**
  Sequential tasks queue up in the ThreadPoolExecutor(max_workers=1). If a long
  inference (code_gen with 512 tokens) blocks, all other categories wait.
  → Background: this is already handled by per-category model caching, but within
    a single category, inference is strictly sequential. Consider interleaving short
    tasks (sentiment, NER) before long ones (code_gen) via a priority queue.

---

## 7. FIREWORKS API SOLVER (agent/solvers/fireworks.py)

### What it does
External API calls to Fireworks Inference. Supports multiple models, reasoning_effort
control, det_hint injection, conflicting_answer context, and prefill.

### Strengths
- Pure stdlib (urllib) — no httpx/requests dependency, keeping image size small.
- reasoning_effort parameter is model-aware: only sends for minimax/gpt-oss models.
- Circuit breaker (config.py REMOTE_CIRCUIT_BREAKER_LIMIT=5) prevents cascading failures.

### Issues & Improvements

**7a. No retry logic on transient failures**
  If Fireworks returns a 502 or 429, the solver returns "" immediately and the task
  falls to local LLM (which may be worse for that category).
  → Add exponential backoff retry (3 attempts, 1s/2s/4s delays). The 30s timeout
    leaves room for 2 retries.

**7b. fw_router.py is an opaque configuration layer**
  The route() function in fw_router.py maps (category, prompt, complexity) → model_id +
  system_prompt + params. This is effectively the routing table for Fireworks, maintained
  separately from the Pipeline's routing table.
  → Merge into a single routing configuration so there's one source of truth for
    "which model handles which category at which complexity".

**7c. FIREWORKS_CATEGORIES vs NAKED_CATEGORIES conflict**
  In main.py: NAKED_CATEGORIES = {} (empty set at runtime). But main.py also has
  FIREWORKS_CATEGORIES = {"sentiment", "summarization", "ner", "logic"}.
  Meanwhile the Pipeline class has its own separate FW routing.
  → These two different routing configurations (main.py's FIREWORKS_CATEGORIES vs
    pipeline.py's _fireworks_escalate) should be reconciled. Pick one: either the
    Pipeline handles all FW routing, or main.py does. Currently both attempt FW
    calls.

---

## 8. WORKFLOW ENGINE (agent/workflow.py + workflow_gate.py)

### What it does
Multi-step Plan-and-Solve execution with artifact passing. Templates: math_3step,
logic_3step, ner_2step. Workflow_gate selects when to apply.

### Strengths
- Artifact-passing between steps is well-designed via the `input_from` field.
- Tool registry for deterministic steps (chunk_text).
- Fallback to single-shot for cells without steps.

### Issues & Improvements

**8a. Only 3 templates, all hardcoded**
  math_3step = plan → solve → compose (identical reasoning chains).
  If eval shows "plan" step produces irrelevant artifacts, there's no way to
  switch to a different prompting strategy per prompt.
  → Make templates data-driven (loaded from a JSON registry) so they can be
    evolved or swapped without code changes.

**8b. Workflow gate (workflow_gate.py) returns template name but pipeline.py ignores it**
  pipeline.py checks route_entry.get("steps") from the routing table, not from
  workflow_gate.select_workflow(). The workflow gate's sophisticated per-category
  decisions (181 lines) are only used by the staging system, not the main pipeline.
  → Either integrate workflow_gate into pipeline.process(), or remove it from the
    codebase if unused.

**8c. Truncation in artifact passing**
  build_step_messages truncates prior output to 400 chars, and prior_context to 300
  chars. For long reasoning chains (code gen, multi-hop logic), this loses critical
  context.
  → Make truncation length configurable per-category. Code and logic need more context;
    sentiment and NER don't.

---

## 9. QC GATE (agent/solvers/verify.py)

### What it does
Quality checks on solver outputs: hedge detection, degenerate repetition, too short/long,
code syntax validation (black + ruff), unsafe imports.

### Strengths
- Comprehensive hedge detection (17 patterns including "I don't know", "sorry",
  "as an AI", "cannot").
- Code validation with black formatting and ruff linting is production-grade.
- Category-specific checks (code_gen strict, code_debug relaxed for fragments).

### Issues & Improvements

**9a. Hedge detection is English-only**
  The hackathon grader is English-only per requirements, so this is acceptable.
  But the patterns are hardcoded in a list — if the grader or an ablation experiment
  needed multilingual support, every pattern would need translation.

**9b. _is_degenerate checks repeat-word frequency with 50% threshold**
  "yes yes yes no no no" (50% each) passes. But "the the the the the answer is X"
  (>50% "the") is flagged. This is a reasonable heuristic but may cause false
  positives for lists (e.g., "apple apple banana banana").
  → Normalize by removing stopwords before the repetition check.

**9c. black/ruff dependency is a Docker image risk**
  If black or ruff aren't installed in the container, the formatter returns
  ("error": "black/ruff not installed") and code quality validation is entirely
  bypassed.
  → Make these Docker build-time requirements, with explicit error at startup
    if they're missing.

**9d. QC gate runs AFTER deterministc solver output but BEFORE pipeline returns**
  In pipeline.py (line 876): verify(answer, category) is called, and if it fails,
  the answer is discarded and falls through to Fireworks. This is good design.
  But the same QC doesn't run on Fireworks output or local LLM output in the
  fast path (lines 886-900).
  → Run verify() on ALL outputs before final return. If QC fails on Fireworks,
    retry or fall to local. If QC fails on local, retry with different temperature.

---

## 10. GEPA FRAMEWORK (orchestrator.py, mutation_agent.py, evaluation_agent.py, analysis_agent.py)

### What it does
Genetic algorithm over cell configurations. Evolves prompt templates + decoding params
via 20 mutation operators + crossover + tournament selection. Pareto-optimal cells
published to the Routing Table.

### Strengths
- NSGA-II non-dominated sort for multi-objective optimization (accuracy, tokens, latency).
- Task-aware mutation operators (different prompt styles per category).
- Analysis tags bias mutation direction (verbose → shorter prompts, imprecise → precise).
- Elitism preserves best cells across generations.
- Convergence check (5pp delta over 3 generations) prevents infinite optimization.

### Issues & Improvements

**10a. Evaluation is extremely expensive**
  Each generation evaluates EVERY cell on EVERY question in the dev set.
  With 8 categories × ~40 questions each × ~20 cells × multiple models, a single
  generation costs significant time and API tokens.
  → Use surrogate evaluation: run full eval every 2-3 generations; for intermediate
    gens, evaluate on a 10% subset (stratified by category). The mutation agent's
    ranking will be noisy but still informative.

**10b. Pareto front is computed per (task_id, model_key) — not cross-model**
  A Qwen2.5 cell and a Gemma cell in the same category are in separate Pareto fronts.
  This means a mediocre Qwen cell could be "Pareto-optimal" in its group and get
  published to the routing table, even if every Gemma cell beats it.
  → Global Pareto front per category, with model identity as a third axis in the
    objective vector. Use (accuracy, tokens, latency, gpu_cost) as objectives.

**10c. Mutation operators 15-19 (workflow ops) are never tested against baseline**
  Operators 15-19 create/modify multi-step workflow cells. But the pipeline only
  checks RouteEntry for workflow steps — it never requests workflow_gate for
  workflow selection. Workflow cells evolved by GEPA are published to routing
  table but may never be used if the pipeline doesn't check workflow steps.
  → Confirm workflow cells actually fire in production. If not, either wire them
    in or disable the workflow mutation operators.

**10d. Crossover fraction (0.6) vs pure mutation (0.4) is a fixed ratio**
  The Pareto-optimal cell should get more crossover opportunities, but the current
  tournament selection treats all parents equally (weighted by accuracy rank).
  → Adaptive crossover rate: if diversity (measured by prompt edit distance) is high,
    increase mutation. If convergence is detected, increase crossover.

**10e. Analysis agent is a stub**
  analysis_agent.py generates tags ("verbose", "imprecise", "params", "workflow").
  These are simple heuristics, not learned patterns. The tags bias mutation direction
  but the bias is weak (0.4 probability) and the tag space is coarse.
  → If the analysis agent adds value, make it data-driven: cluster cell errors from
    eval details and generate tags from actual failure modes.

---

## 11. DYNAMIC PROMPTS (agent/dynamic_prompts.py — 723 lines)

### What it does
Per-category, per-complexity system prompt templates. 3 complexity tiers (low/medium/high)
for each of the 8 categories. Anti-preamble suffix appended to every prompt.

### Strengths
- Excellent research foundation: derived from winning hackathon repos and the project's
  own prompt history.
- Complexity-aware tiers provide graduated scaffolding (more detail for hard tasks,
  concise for easy ones).

### Issues & Improvements

**11a. 723 lines of template strings — unmaintainable**
  Every prompt is a Python f-string embedded in a dict. Changing one word in the
  "code_gen:medium" prompt requires finding line ~60, editing a multi-line string,
  and hoping the indentation is correct.
  → Move to a YAML/JSON file: agent/prompts.yaml loaded at init. Editable without
    Python source changes. Enables A/B testing of prompt variants.

**11b. As noted in §6c: 5/8 categories are NAKED**
  If NAKED_CATEGORIES = {ner, summarization, factual, logic, math}, then 5 of 8
  categories never use dynamic_prompts.py. Only sentiment, code_gen, and code_debug
  actually get these prompts.
  → Either audit which categories genuinely benefit from custom prompts and shrink
    the file to match, or eliminate NAKED_CATEGORIES and use dynamic prompts everywhere.

---

## 12. ROUTING TABLE (agent/routing_table.py)

### What it does
Versioned mapping from category → cell configuration (model, prompt, decoding, role).
Published by GEPA Orchestrator, read by Pipeline.

### Strengths
- Versioned with history tracking — enables rollback if a new cell performs worse.
- Backtest gate: update_from_cells can reject cells that fail backtesting.
- Workflow step support: stores multi-step configurations alongside single-shot.

### Issues & Improvements

**12a. Backtest gate is never enforced**
  In publish_routing_table(), backtest_results is always None and strict=False.
  Every cell from the Pareto front is published regardless of actual performance.
  → Run a quick backtest (10 eval questions per category) before publishing. Only
    publish cells that maintain or improve accuracy.

**12b. No expiry or staleness tracking**
  A cell published in generation 0 stays in the routing table forever unless explicitly
  replaced. As the evaluation set grows, generation-0 cells may be overfit to the
  original 60-set questions.
  → Add a freshness field: cells older than N generations are deprioritized or
    re-evaluated before routing.

---

## 13. STAGING PARALLEL SYSTEM (ready_pool.py, ready_queue.py, ready_judge.py)

### What it does
Multiprocessing worker pool with per-type partitioned task pools. 3 worker types:
deterministic (2 workers), local LLM (1-3), Fireworks API (1). Per-task multi-vote
judgment (default 5 votes). Backstop release mechanism for unclaimed tasks.

### Strengths
- Sophisticated partitioning prevents fast workers (det) from draining the queue
  before slow workers (local) get a chance.
- Backstop release after reservation_timeout_s ensures no task is orphaned.
- Heartbeat monitoring and orphan re-enqueue for worker failure tolerance.
- Ablation switches (disable_fireworks, disable_local, disable_deterministic) for
  controlled experiments — excellent research infrastructure.
- Per-worker inbox limiting prevents any single worker from being overwhelmed.

### Issues & Improvements

**12a (sic). Inbox size of 3 is very small**
  per_worker_inbox_size=3 means each worker can queue at most 3 tasks. With 40 tasks
  and 3 worker types, the deterministic workers (2×) will process their 3 tasks in
  seconds and then idle while waiting for the backstop release (30s).
  → Either increase inbox size to match task count / worker count ratio, or reduce
    reservation_timeout_s proportionally.

**12b. Worker startup is sequential**
  Each worker process is started one-by-one in the _start_workers() loop. If a model
  load takes 15s, the pipeline is blocked until all workers are ready.
  → Start worker processes concurrently via a ThreadPoolExecutor. The Python GIL
    doesn't block subprocess startup.

**12c. No worker reuse across runs**
  The ReadyPool is destroyed and recreated for each batch. Worker processes are
  spawned fresh every time, including model loading.
  → For the hackathon grader (single batch per container), this is fine. But if
    this were used in a server context, worker reuse would be critical.

**12d. Synchronization complexity is high**
  multiprocessing.Value, multiprocessing.Event, multiprocessing.Queue, per-worker
  flags, inbox queues, steal queues, heartbeat counters — 7+ synchronization primitives.
  This is fragile: a deadlock in any one can freeze the entire pipeline.
  → Consider a simpler shared-nothing architecture: each worker process reads from
    a pre-partitioned file of tasks, processes independently, writes results. No
    runtime coordination needed for a single-batch benchmark.

---

## 14. CROSS-CUTTING SYSTEMIC ISSUES

**14a. Dead code accumulated across two architectures**
  The production Pipeline and the Staging ReadyPool implement overlapping
  functionality (deterministic solvers, FW calls, voting). The GEPA framework
  adds a third layer. Some code paths are clearly dead:
  - consensus_samples=1 but consensus_vote code is elaborate
  - workflow_gate is sophisticated but pipeline ignores it
  - dynamic_prompts unused for 5/8 categories
  - detection patterns that match greeting/human dialog in a benchmark context
  → Run a coverage trace: instrument every function entry and run the full 300-set
    eval. Delete anything that doesn't fire.

**14b. No central configuration schema**
  Config exists in: config.py (env vars), PipelineConfig (defaults + env overrides),
  ReadyConfig (separate env vars with STAGING_ prefix), fw_router.py (hardcoded dicts),
  category_model_map (hardcoded paths), COMPLEXITY_THRESHOLDS (hardcoded).
  Changing model paths or thresholds requires touching 4+ files.
  → Single YAML schema: agents.yaml loaded by all components. Each component reads
    its section.

**14c. Error handling is too permissive**
  The pattern `try: ... except Exception: pass` appears ~20 times in pipeline.py alone.
  This means silent degradation: if the deterministic solver fails, the issue is logged
  at WARNING level and the pipeline continues — potentially with an empty answer.
  → Log at ERROR level for unexpected exceptions. Distinguish "can't handle" (OK to
    skip) from "implementation bug" (should crash/alert).

**14d. No integration test that validates a full run**
  The only test file (tests/test_container.py) checks Docker build success and basic
  task reading/output writing. There's no end-to-end test that runs 8 tasks through
  the pipeline and validates answer format.
  → Create a tests/e2e/ directory with 1-2 tasks per category. Run against a small
    model or mock. This would catch routing configuration errors immediately.

---

## SUMMARY: PRIORITY IMPROVEMENTS

| Priority | Area | Change | Impact |
|----------|------|--------|--------|
| P0 | Dead code | Coverage trace → prune unused paths | -30% codebase, faster iteration |
| P0 | Config unification | Single YAML for all routing params | Eliminates config drift |
| P1 | solver/deterministic.py | Split into per-type files | Testability, maintainability |
| P1 | Unit tests | Add solver tests from 300-set data | Catch regressions |
| P1 | Evaluate GEPA cells | Enable backtest gate on routing table | Prevents accuracy regression |
| P1 | Error handling | Remove except:pass, log distinct errors | Faster debugging |
| P2 | Complexity scorer | Per-category thresholds | Better routing accuracy |
| P2 | Classifier | Document algorithm or replace with trainable | Long-term robustness |
| P2 | Fireworks retry | Add exponential backoff | Better API reliability |
| P2 | Workflow integration | Wire workflow_gate into pipeline.process() | Actually use multi-step |
| P3 | NAKED categories audit | Confirm bypass still helps or remove | Simplifies code |
| P3 | Model memory | Unload unused models under pressure | Fits 4GB container |
