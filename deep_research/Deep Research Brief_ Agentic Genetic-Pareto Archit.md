<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Deep Research Brief: Agentic Genetic-Pareto Architecture for Multi-Model Small LLM System

Context

I have a working multi-model LLM pipeline running inside a constrained Docker container:

Hardware: 2 CPUs × 4 GB RAM = 8 GB total, no GPU

Container image: ≤10 GB

Inference: llama.cpp / Ollama, GGUF Q4_K_M quantization

Current small models (all ≤1.7B parameters):

Qwen2.5-1.5B-Instruct — generalist

Qwen2.5-Math-1.5B-Instruct — math reasoning

Qwen2.5-Coder-1.5B-Instruct — structured logic / coding

SmolLM2-1.7B-Instruct — general instruction following
arxiv
+1

Tasks:

T01: factual knowledge

T02: mathematical reasoning (word problems)

T03: sentiment classification (including mixed sentiment)

T04: constrained summarization (exact sentences / bullets, word limits)

T05: named-entity recognition (PERSON / ORG / LOC / DATE)

I also have a Genetic-Pareto (GEPA)-style framework for multi-objective prompt optimization (genetic algorithms + Pareto fronts), and I want to extend it into an agentic architecture that:

Optimizes compositional “cells” (combinations of models, prompts, and decoding parameters) per task

Uses sub-agents with distinct skills to run optimization cycles

Emphasizes technologies, architectural patterns, and conceptual design (not raw code)

Explicitly identifies the biggest dangers and errors in such systems and how to avoid them
youtube
proceedings.iclr
+2

Please conduct deep research and provide an architecture-level analysis.

1. Conceptual model of “cells” in a Genetic-Pareto framework

Investigate and propose a conceptual representation for “cells” in a GEPA-style system:

Each cell combines:

A task type (T01–T05)

A set of LLMs chosen from my small model pool

For each LLM: a prompt template (from a prompt library) and a decoding configuration (temperature, top‑p, top‑k, min‑p, repeat penalty, seed, etc.)

An aggregation / judgment strategy (e.g., majority vote, self-consistency, deterministic task-specific judge)

Summarize how similar frameworks (ParetoPrompt, MOPrompt, GAAPO, GEPA, Evolutionary Prompting) model prompts and multi-objective trade-offs conceptually, not in code.
proceedings.iclr
+6
youtube

Describe:

What constitutes mutation and crossover at the concept level (e.g., swapping models, changing prompt style, expanding ensemble size)

How to maintain diversity in cells while respecting resource constraints (8 GB RAM, CPU-only).

Focus on architectural patterns rather than implementation details.

2. Overall system architecture with agentic sub-components

Design and analyze an agentic system architecture consisting of:

A central GEPA Orchestrator responsible for multi-objective optimization and Pareto front management

Several sub-agents, each with distinct roles:

Mutation / evolution agent (applies genetic operators to cells)

Evaluation agent (runs cells on dev sets and collects metrics)

Analysis / diagnostics agent (interprets failures and suggest changes)

Routing agent (selects the best cell for a live query)

Reporting / governance agent (tracks evolution and flags regressions)

A tool layer: LLM inference, deterministic validators (math, NER, summarization), logging and metrics store.
arxiv
+3

For this architecture, please:

Describe recommended technologies and frameworks (e.g., message buses, workflow/orchestration tools, experiment tracking systems) suitable for orchestrating agent cycles in a CPU-only container.

Compare architectural styles:

Centralized orchestrator vs more decentralized multi-agent frameworks

Batch optimization (offline cycles) vs online self-tuning while serving requests

Discuss how to maintain clear interfaces and responsibilities between agents to avoid role confusion and side effects.

The answer should emphasize architectural decisions, trade-offs, and recommended patterns—not code.

3. Multi-objective optimization goals and how they shape the system

Analyze how multi-objective optimization frameworks (Pareto-based) define and manage objectives like:

Task-specific accuracy / pass rate (e.g., math correctness, NER F1, sentiment label quality)

Format compliance (JSON validity, sentence counts, bullet limits, word limits)

Latency / throughput (time per example, CPU utilization)

Stability / determinism (variance across seeds and runs)

For each objective:

Explain how it influences architectural choices, such as:

How many cells can be evaluated per cycle

How large ensembles can be (how many models per cell)

How strict format checkers and validators must be

Summarize methods from literature (e.g., ParetoPrompt, MOPrompt, GAAPO) for balancing multiple objectives and maintaining a Pareto front without collapsing everything into a single scalar score.
proceedings.iclr
+3

Focus on conceptual integration with system design rather than mathematical derivations.

4. Agentic optimization cycle: conceptual flow and control points

Give a conceptual description of the optimization cycle in this agentic system:

How the central orchestrator and sub-agents collaborate over repeated cycles to:

generate new candidate cells,

evaluate them,

update the Pareto front,

diagnose failure patterns,

and refine future mutations.

Discuss control points and governance mechanisms:

When and how to stop or pause optimization (convergence criteria)

How to guard against regressions (new cells performing worse than older ones)

How to log and audit decisions made by sub-agents.

Include references to agentic and evolutionary prompt optimization frameworks (e.g., GEPA, Reflective Prompt Evolution, Evolutionary Prompting) and how they structure cycles and feedback loops conceptually.
emergentmind
youtube
dev

Emphasize architecture and process design, not code.

5. Deterministic routing and non-LLM classifiers

I want deterministic, non-LLM routing and validation to complement my small LLMs.

Please research:

Best practices and architectural patterns for pre-judgment routing in multi-LLM systems (deciding which cell to use for a query before any generation).
arxiv
+4

Classical and small ML technologies suitable for:

Task classification (factual vs math vs summarization vs sentiment vs NER vs code)

Validation (e.g., lightweight NER models to verify LLM outputs, small sentiment classifiers to check labels).

How to integrate these classifiers as first-class components in the architecture:

Where they sit relative to the router agent and GEPA cells

How they interact with the LLM ensemble and judges

What constraints their latency and memory footprint impose on system design.

The focus should be on technologies, patterns, and conceptual integration, not algorithms or code.

6. Technologies and tools: recommended stack for this system

Within the constraints (small LLMs, CPU-only, 8 GB RAM), recommend:

Inference technologies: best options for running multiple GGUF models in parallel (llama.cpp server, Ollama, alternative CPU inference engines) and how they influence the architecture.

Orchestration / workflow technologies: options for running agent cycles and managing GEPA (e.g., Airflow, Prefect, Argo, or lighter-weight orchestrators), pros and cons in this context.

Experiment tracking / metrics: tools for logging fitness scores, Pareto fronts, and cell configurations over generations (e.g., MLflow-like systems, custom logging solutions).

Validation / analysis tooling: libraries for NER, sentiment, summarization, math solving, and format checking that can be integrated as services or modules.

All recommendations should be at the level of what tools / technologies and why, not how to code them.

7. Biggest dangers and failure modes in such a system

Provide a detailed analysis of major pitfalls and dangers in building and operating this agentic GEPA-based multi-model system, grouped into categories:

Optimization-level dangers:

Overfitting to a small dev set

Pareto front degenerating toward one objective (e.g., latency) at the cost of accuracy

Loss of diversity in the population (convergence too fast, genetic drift)

Agentic / orchestration dangers:

Cycle explosion: evaluating too many cells given slow CPU inference

Deadlocks or race conditions between agents

Misconfigured routes (wrong cells selected) due to stale routing or bad classifier decisions

Unclear boundaries between agent responsibilities leading to conflicts

Model / inference dangers:

Misuse of decoding parameters (temperature, top‑p, top‑k, min‑p, repeat penalties) leading to unstable or inconsistent results

Using small models beyond their capability (e.g., expecting GPT‑4-level reasoning)

Context length issues (long prompts truncated, silent errors)

Data / evaluation dangers:

Dev/test contamination through few-shot examples inside prompts

Unrepresentative evaluation sets (e.g., only easy cases)

Lack of robust metrics for summarization and sentiment (subjective scoring)

For each category:

Explain why it is dangerous in this kind of architecture.

Suggest architectural safeguards, monitoring strategies, or governance patterns to mitigate it.

Please draw on documented experiences and research in multi-model routing, evolutionary prompt optimization, agentic systems, and small LLM deployment where possible.
proceedings.iclr
+3
youtube
arxiv
+4

8. High-level “commands for builders”

Finally, translate the findings into a set of architecture-level commands or guidelines that a technical team could follow, such as:

“Define cells as compositional units combining task, models, prompts, and parameters; do not mix optimization and routing logic inside individual models.”

“Use deterministic, non-LLM classifiers for routing; LLMs should focus on problem-solving, not meta-decisions.”

“Treat GEPA optimization as an offline, budgeted process; keep online inference fast and simple by using a small set of Pareto-optimal cells.”

“Instrument and log all agent decisions and cell changes; never update the live routing table without a verifiable improvement in multi-objective metrics.”

I want these commands to be conceptual and architectural (what to do / what to avoid), not code instructions.

Output format requested:

Structured sections corresponding to points 1–8.

Emphasis on technologies, architecture patterns, conceptual models, and pitfalls.

References to relevant papers, frameworks, and implementations (GEPA, ParetoPrompt, MOPrompt, GAAPO, PromptEval, DIPPER, Evolutionary Prompting, multi-model routing).
osti
+6
youtube
proceedings.iclr
+4

You can treat your GEPA-based multi-model system as a population of compositional “cells” that encode task-specific ensembles of small LLMs, prompts, and decoding configs, and evolve them under a Pareto-based multi-objective optimization loop. Below is an architecture-level brief that translates ideas from GEPA, ParetoPrompt, MOPrompt, GAAPO, evolutionary prompting, and multi-model routing into a constrained, CPU-only, small‑LLM setting.[^1][^2][^3][^4][^5][^6][^7][^8][^9]

***

## 1. Conceptual model of “cells” in a GEPA-style framework

A **cell** in your system should be defined as a compositional unit: `(task type T01–T05, ensemble of one or more LLMs, per‑LLM prompt template, per‑LLM decoding configuration, aggregation/judge strategy, and evaluation metadata)`, analogous to how prompt objects bundle template and parameters in evaluation frameworks like DeepEval and PromptEval. Conceptually this matches ParetoPrompt, MOPrompt, GAAPO, and GEPA, which all treat each prompt (or prompt bundle) as an individual in a population, scored along several objectives (accuracy, robustness, cost, latency) and compared via Pareto dominance rather than a single scalar fitness.[^2][^10][^5][^11][^12][^13][^14][^8][^1]

At the **concept level**, mutation means altering any cell dimension (swap an LLM, change prompt style or role structure, tweak decoding hyperparameters, change aggregation from majority vote to self‑consistency, adjust tool calls or validators), while crossover means combining substructures from two high‑performing cells (e.g., reusing the math specialist’s prompt and decoding settings inside a sentiment cell, or combining the judge strategy of one cell with the ensemble of another). Diversity under your 8 GB CPU‑only constraint is maintained by (1) limiting ensemble size per cell (e.g., 1–2 models active per query), (2) sharing loaded model instances across cells, (3) capping the number of candidate cells per generation, and (4) explicitly favoring heterogeneous cells on the Pareto front (different models, different prompts, different judges) rather than many near‑duplicates optimized only for latency.[^3][^4][^8][^9][^15]

***

## 2. Overall system architecture with agentic sub-components

A practical architecture is a **central GEPA Orchestrator** (a single process in your container) coordinating sub‑agents via a lightweight internal message bus or task queue (e.g., in‑process queues, SQLite-backed work registry, or a minimal orchestration framework like Prefect rather than heavyweight Airflow), with each agent exposing a clear interface: mutation agent, evaluation agent, analysis/diagnostics agent, routing agent, and reporting/governance agent. In this pattern, the Orchestrator manages the population and Pareto front, schedules batch evaluation jobs, hands failure traces to the analysis agent, and only updates the routing table when the governance agent confirms multi‑objective improvement, reflecting the “build–judge–optimize” loops described for GEPA‑optimized multi‑agent systems.[^16][^17][^18][^14][^8][^9]

A **centralized orchestrator** keeps responsibilities crisp (sub‑agents are libraries or services with single roles) and is easier to debug in a small CPU container, while more decentralized frameworks (Router‑R1 style LLM-as-router, or multi‑agent platforms where agents route each other) introduce additional overhead and coordination complexity that is usually justified only at larger scales. Batch, offline optimization cycles (nightly or on demand) are strongly preferable in your setting: they align with frameworks like PromptEval and GEPA that work under explicit evaluation budgets, and they let live routing use a small, vetted set of Pareto‑optimal cells so online serving remains simple and low‑latency.[^11][^12][^8][^19][^9][^16]

To avoid **role confusion**, each agent should have a narrow contract (e.g., mutation agent only produces new cells from a labeled parent set; routing agent only reads from a persisted routing table and never writes; analysis agent only annotates failures with labels and suggestions), and interactions should be mediated through well‑defined artifacts (cell definitions, evaluation runs, Pareto front snapshots, routing tables) stored in an experiment store rather than free‑form messages.[^20][^21][^14]

***

## 3. Multi-objective optimization goals and how they shape the system

In Pareto‑based prompt optimization, typical objectives are **task performance metrics** (accuracy, F1, pass rate), **format integrity** (JSON validity, length constraints), **cost/latency**, and **robustness**, and frameworks like ParetoPrompt, MOPrompt, GAAPO, and GEPA show that explicitly treating these as separate axes yields more robust prompt sets than collapsing them into a single weighted score. Architecturally, this means your evaluation agent must compute multiple metrics per cell (e.g., math correctness from a deterministic solver, NER F1 from a classical tagger, JSON/length checks via Promptfoo‑style deterministic validators), and your Orchestrator must maintain a Pareto front where cells are retained if they are non‑dominated, even if they are slightly slower or less deterministic than the current best on some objectives.[^10][^5][^12][^22][^23][^24][^8][^1][^2][^11]

These objectives strongly constrain **ensemble size and evaluation budget**: latency and CPU utilization limit how many models per cell and how many cells per generation you can afford, so your GEPA Orchestrator should explicitly treat “time per example” and “examples per evaluation batch” as budget parameters when scheduling optimization cycles. Format compliance and determinism goals drive stronger tooling: schema‑based validation at the boundary, strict JSON and length checks, and controlled decoding parameters with fixed seeds and penalties for unstable cells, all of which mesh naturally with deterministic assertion frameworks like Promptfoo and evaluation stacks like DeepEval or PromptEval.[^12][^22][^23][^25][^26][^8][^9][^11]

Finally, **robustness objectives** (variance across seeds, performance quantiles rather than means) push you toward evaluation approaches like PromptEval that estimate full performance distributions under limited budgets; integrating this into the architecture means your evaluation agent must support sampling across seeds and inputs, and your Orchestrator should retain cells that offer good lower‑quantile performance even if their mean score is similar.[^8][^11][^12]

***

## 4. Agentic optimization cycle: conceptual flow and control points

Conceptually, your **optimization cycle** mirrors GEPA and evolutionary prompting frameworks: (1) start from a population of baseline cells; (2) mutation agent generates new candidate cells via prompt edits, model swaps, decoding tweaks, or ensemble/aggregation changes; (3) evaluation agent runs these cells on held‑out dev subsets under multiple metrics; (4) Orchestrator updates the Pareto front; (5) analysis agent inspects failures (wrong labels, format breaks, slow outliers) and tags patterns; (6) mutation agent biases future mutations using these tags and GEPA‑style reflective prompt evolution. Convergence and pause criteria can be defined at the architecture level (e.g., no Pareto‑front improvements above a threshold after N generations, evaluation budget exhausted, or stability of routing metrics over time), rather than ad‑hoc per agent, echoing the “budgeted optimization” design emphasized in GEPA and the build–judge–optimize blueprint.[^4][^14][^9][^3][^8]

To guard against **regressions**, your reporting/governance agent should maintain a backtesting subset (as PromptLayer and other evaluation frameworks recommend) and require that any candidate routing table or “champion” cell outperform the current one on both task metrics and safety/format metrics before promotion to production. Every decision (cell creation, metric results, Pareto updates, routing table changes) should be logged in an experiment store like MLflow or an equivalent local registry, with cell configurations treated as versioned artifacts, so you can audit why a particular cell was selected and roll back if necessary.[^27][^21][^26][^9][^11][^12][^20][^8]

***

## 5. Deterministic routing and non-LLM classifiers

Best practice in multi‑model routing is to use **deterministic, non‑LLM routers and validators** wherever possible, reserving LLMs for content generation; recent routing work (Universal Model Routing, RouteLLM, LLMRouter, TensorOpera Router) shows that lightweight classifiers or cost‑aware rules can effectively choose among models under explicit quality–cost trade‑offs. In your setting, task classification (T01–T05) can be done with small ML models (e.g., bag‑of‑words or transformer‑mini classifiers) trained on labeled queries, and validation can rely on spaCy‑style NER and sentiment components or other small CPU‑friendly NLP models to check that LLM outputs match expected labels or entity structures.[^28][^29][^30][^31][^32][^33][^6][^7][^34]

Architecturally, these classifiers should be **first‑class components** sitting between the router agent and the GEPA cells: the router first uses the deterministic task classifier to select a task‑specific pool of candidate cells, then (optionally) uses additional static heuristics or learned routers to pick one cell, and finally uses non‑LLM validators downstream to score the cell’s outputs and detect failures that may feed back into the analysis agent. Latency and memory constraints mean these classifiers must have tiny footprints (e.g., small spaCy pipelines, distilled sentiment models) and run faster than the LLMs; if a classifier becomes a bottleneck, routing and validation throughput collapses and defeats the purpose of using multiple small LLMs, so you should explicitly measure classifier latency and include it as a cost objective in your optimization and routing designs.[^29][^31][^35][^25][^6][^7][^15]

***

## 6. Technologies and tools: recommended stack for this system

For **inference**, llama.cpp with GGUF quantization (Q4_K_M and similar) is the most natural choice for CPU‑only, small‑model deployment: it offers a tiny binary, aggressive quantization, and an OpenAI‑compatible server mode, and multiple studies and tutorials confirm its viability on low‑resource hardware. Ollama is a convenient alternative—especially inside Docker—with simple model management and HTTP APIs, though it expects somewhat more CPU/RAM and disk and is less fine‑grained about custom GGUF loading than raw llama.cpp.[^36][^37][^38][^35][^25]

For **orchestration and agent cycles**, a lightweight Python orchestrator or Prefect‑style workflow engine running in‑process is appropriate: Prefect avoids the heavy database and scheduler overhead of Airflow while still giving you task graphs, retries, and logging, which aligns well with a single‑container, CPU‑only deployment. Experiment tracking and metrics can be handled by local MLflow Tracking (using the default `mlruns` directory or a SQLite backend), logging cell parameters, metrics, routing configurations, and Pareto‑front snapshots in a way that’s queryable and versioned.[^17][^18][^21][^20]

For **validation and analysis tooling**, you can integrate deterministic output validation and evaluation frameworks such as Promptfoo (JSON/XML checks, regex, length, latency, safety assertions), DeepEval (end‑to‑end and component‑level prompt evaluation, G‑Eval LLM-as-judge for hard‑to‑score tasks), and PromptEval (efficient multi‑prompt performance estimation under limited budgets), pairing them with small NLP libraries like spaCy for NER/sentiment and dedicated length/format checkers inspired by best‑practice guidance on structured outputs.[^39][^22][^23][^32][^24][^40][^33][^41][^26][^11][^12]

***

## 7. Biggest dangers and failure modes in such a system

At the **optimization level**, overfitting to a small dev set, Pareto fronts collapsing toward a single objective (e.g., latency), and loss of diversity in the population are all common risks; GEPA and similar prompt optimizers explicitly stress budgeted, sample‑efficient optimization and multi‑domain rubrics to avoid optimizing only for one metric, and evolutionary prompting work highlights the need for diverse, interpretable prompt variations. Architecturally, you mitigate these by maintaining separate held‑out and backtesting sets, enforcing minimum diversity constraints (different models/prompts on the front), and periodically re‑sampling evaluation inputs rather than repeatedly tuning on the same subset.[^42][^14][^9][^3][^4][^11][^12][^27][^8]

In **agentic/orchestration** terms, cycle explosion (too many cells per generation on a slow CPU), deadlocks or race conditions, misconfigured routes, and unclear boundaries between agents are significant dangers; routing research emphasizes cost thresholds and budget constraints, while evaluation frameworks recommend failing fast and blocking problematic changes via automated gates. Safeguards include explicit evaluation budgets per cycle, strict responsibility boundaries and single‑writer rules for routing tables (only the governance agent can commit changes), and CI‑like gating where a routing update is only applied if evaluation pipelines pass predetermined multi‑objective thresholds.[^31][^21][^41][^26][^7][^15][^20][^27]

At the **model/inference** level, mis‑tuned decoding parameters can lead to unstable outputs, small models can be pushed beyond their reasoning capacity, and context length or truncation problems can cause silent errors; CPU‑only deployment studies and local LLM guides warn that aggressive quantization and wrong threading settings can degrade quality or throughput if not carefully benchmarked. Architecturally, you should constrain decoding ranges, treat temperature/top‑p/top‑k/min‑p as part of the cell definition with conservative defaults, and implement strict context‑length monitoring and truncation logging, plus explicit “capability boundaries” for small models (e.g., word‑problem difficulty ranges) so routing does not assign them tasks they are unlikely to solve.[^38][^35][^25][^9][^39][^36][^8]

Finally, **data/evaluation** dangers include dev/test contamination via few‑shot examples, unrepresentative evaluation sets, and weak metrics for subjective tasks like summarization and sentiment; evaluation guidelines stress carefully curated golden datasets covering happy paths and edge cases, clear splits, and multi‑faceted metrics combining deterministic checks and model‑graded scoring. Architectural safeguards are to store prompts in version control, keep evaluation datasets separate from prompt few‑shot examples, use multiple metrics per task (e.g., ROUGE‑like length/coverage plus human‑aligned model‑judge scores), and treat metrics definitions themselves as first‑class artifacts that can be audited and updated as you learn more about failure modes.[^40][^41][^26][^11][^12][^27]

***

## 8. High-level “commands for builders”

1. **Define cells as compositional units** that combine task, models, prompts, decoding parameters, aggregation strategy, and evaluation metadata; keep optimization and routing logic outside the cells and instead in a central GEPA Orchestrator.[^13][^14][^11][^12][^8]
2. **Use deterministic, non‑LLM classifiers and validators for routing and checking outputs**, placing them as first‑class components before and after LLM calls; reserve small LLMs for solving tasks, not for meta‑decisions or format checks.[^22][^23][^32][^33][^6][^7][^29][^31]
3. **Treat GEPA optimization as an offline, budgeted process**: run batch optimization cycles under explicit evaluation budgets, maintain a Pareto front over multi‑objective metrics, and deploy only a small, vetted set of Pareto‑optimal cells into the live routing table.[^5][^9][^1][^2][^11][^12][^8]
4. **Instrument and log all agent decisions and cell changes** in an experiment tracker (e.g., MLflow) and never update the production routing table without a verifiable multi‑objective improvement on backtesting sets and guardrail metrics.[^21][^41][^26][^20][^27]
5. **Prefer a centralized orchestrator with narrow, single‑purpose sub‑agents** over a highly decentralized multi‑agent mesh in a small CPU container, to minimize coordination complexity and make cycles, budgets, and failure analysis easier to control.[^18][^14][^16][^17]
6. **Anchor all objectives and metrics in explicit, task‑specific rubrics and schemas**, including accuracy, format integrity, latency, and robustness, and treat these rubrics as the source of truth for both optimization and governance.[^26][^14][^39][^27][^40][^8]
7. **Conservatively bound ensemble size, decoding freedom, and evaluation population size** to fit within CPU and RAM limits, and explicitly model “cost” (latency, utilization) as an objective alongside accuracy when maintaining your Pareto front and routing policies.[^35][^25][^15][^31][^38][^8]
8. **Continuously monitor for diversity and regressions**, enforcing diversity constraints on the Pareto front (different models/prompts/judges), maintaining backtesting datasets, and blocking any change that reduces performance on core tasks or safety/format metrics—even if it improves latency.[^41][^3][^4][^42][^11][^12][^27]

These commands should give your team an architecture-level playbook for turning your existing GEPA-style framework and small-model pool into a governed, agentic, multi-objective optimization system that remains tractable on CPU-only hardware while still exploiting evolutionary and Pareto-based design patterns.
<span style="display:none">[^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55]</span>

<div align="center">⁂</div>

[^1]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/13b45b44e26c353c64cba9529bf4724f-Abstract-Conference.html

[^2]: https://arxiv.org/abs/2508.01541

[^3]: https://arxiv.org/html/2503.23503v1

[^4]: https://www.emergentmind.com/topics/evolutionary-prompting-eot

[^5]: https://arxiv.org/abs/2504.07157

[^6]: https://github.com/ulab-uiuc/LLMRouter

[^7]: https://github.com/lm-sys/routellm

[^8]: https://arxiv.org/pdf/2507.19457.pdf

[^9]: https://www.deeplearning.ai/the-batch/authors-devised-gepa-an-algorithm-for-better-prompts-to-improve-agentic-systems-performance

[^10]: https://openreview.net/pdf?id=HGCk5aaSvE

[^11]: https://arxiv.org/html/2405.17202v3

[^12]: https://neurips.cc/virtual/2024/poster/93925

[^13]: https://deepeval.com/docs/evaluation-prompts

[^14]: https://arxiv.org/html/2603.03565v1

[^15]: https://neurips.cc/virtual/2025/poster/117032

[^16]: https://www.superannotate.com/blog/multi-agent-llms

[^17]: https://www.linkedin.com/posts/vamsikrrishnna_apacheairflow-prefect-workfloworchestration-activity-7427925886563229696-RBRz

[^18]: https://stackoverflow.com/questions/75641630/stream-data-between-tasks-in-pipeline-orchestration-tool-prefect-dagster-airflow

[^19]: https://neurips.cc/virtual/2025/poster/119214

[^20]: https://www.prophecylabs.com/blog/a-swift-guide-to-experiment-tracking-with-mlflow

[^21]: https://mlflow.org/docs/latest/ml/tracking/

[^22]: https://www.promptfoo.dev/docs/configuration/expected-outputs/deterministic/

[^23]: https://www.promptfoo.dev/docs/configuration/expected-outputs/

[^24]: https://sofius.com/artikelen/automated-llm-evaluation-with-promptfoo/

[^25]: https://ceur-ws.org/Vol-4164/paper11.pdf

[^26]: https://github.com/confident-ai/deepeval

[^27]: https://blog.promptlayer.com/llm-eval-framework/

[^28]: https://github.com/explosion/spaCy/discussions/10031

[^29]: https://ubiops.com/what-is-multi-model-routing/

[^30]: https://github.com/BlueBrain/Search/issues/602

[^31]: https://arxiv.org/html/2502.08773v1

[^32]: https://spacy.io/usage/large-language-models

[^33]: https://explosion.ai/_/project/spacy

[^34]: https://arxiv.org/html/2408.12320v3

[^35]: https://blog.chatbotslife.com/bring-gen-ai-to-your-laptop-hosting-ollama-with-docker-for-llm-mastery-d0a03343a9ae

[^36]: https://www.youtube.com/watch?v=vvrGfqd1oxE

[^37]: https://www.datacamp.com/tutorial/docker-ollama-run-llms-locally

[^38]: https://tech-insider.org/llama-cpp-tutorial-2026/

[^39]: https://mightybot.ai/blog/best-structured-prompt-formats-for-llms/

[^40]: https://deepeval.com/docs/metrics-llm-evals

[^41]: https://www.tmasolutions.com/insights/llm-evaluation-metrics-in-cyber-security-outsourcing-services-with-promptfoo-framework

[^42]: https://aclanthology.org/2025.luhme-1.6.pdf

[^43]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1613007/full

[^44]: https://arxiv.org/abs/2507.19457

[^45]: https://news.ycombinator.com/item?id=44744331

[^46]: https://www.merge.dev/blog/multi-model-routing

[^47]: https://arxiv.org/html/2601.05903v1

[^48]: https://www.reddit.com/r/LocalLLaMA/comments/1p90zzi/cpuonly_llm_performance_ts_with_llamacpp/

[^49]: https://github.com/gepa-ai/gepa

[^50]: https://spacy.io/api/large-language-models

[^51]: https://www.youtube.com/watch?v=zLo64Iqstwk

[^52]: https://arxiv.org/html/2601.17814v1

[^53]: https://arxiv.org/html/2507.19457v1

[^54]: https://github.com/anyscale/llm-router

[^55]: https://shashikantjagtap.net/agent-lightning-vs-superoptix-microsoft-enters-the-agent-optimization-race/

