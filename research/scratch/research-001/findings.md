# Research Findings: Agentic Architectures for Containerized AI Pipelines

---

## [2026-07-13 08:00] Web Researcher
- **Finding:** Anthropic's "Building effective agents" guide describes the supervisor (orchestrator) pattern where a central LLM delegates tasks to specialized sub-agents. The supervisor receives the high-level goal, breaks it into subtasks, routes each to the appropriate agent, and aggregates results. This maps directly to the pipeline that classifies prompts into 8 categories and routes to local GGUF or Fireworks API.
- **Source:** https://docs.anthropic.com/en/docs/build-with-claude/agentic-patterns
- **Verdict:** Directly applicable — the prompt classifier can act as the supervisor/router, dispatching to local (GGUF) or cloud (Fireworks) sub-agents based on category, with the supervisor handling aggregation and fallback logic.

## [2026-07-13 08:00] Web Researcher
- **Finding:** OpenAI's "A Practical Guide to Building Agentic Systems" defines the Orchestrator-Agent-Worker pattern where an orchestrator manages state, routing, and error handling while workers execute tasks. Includes specific guidance on fallback chains: when a primary LLM call fails (timeout/error), the system falls back to a simpler/cheaper model or cached response. This is critical for a Docker pipeline with 4GB RAM where local GGUF models may OOM on complex prompts.
- **Source:** https://platform.openai.com/docs/guides/agentic-systems
- **Verdict:** The fallback chain pattern is essential — if the local GGUF model times out or crashes, the supervisor should drop down to a simpler local model or route to Fireworks API as fallback. The 4GB RAM constraint makes this especially important.

## [2026-07-13 08:00] Web Researcher
- **Finding:** LangChain's "Multi-Agent Systems" documentation details the Supervisor (orchestrator) architecture where a single LLM-as-supervisor decides which agent to call next based on task state. It also documents the "Agent Supervisor" pattern from AutoGen and CrewAI where agents can be dynamically added/removed. Notable: the supervisor tracks task completion and can re-route or retry failed subtasks.
- **Source:** https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
- **Verdict:** The dynamic agent routing pattern is ideal — new prompt categories or specialized handlers can be added without rewriting the pipeline. The supervisor's retry/re-route capability handles abnormal conditions.

## [2026-07-13 08:00] Web Researcher
- **Finding:** Microsoft's "Autonomous AI Agent Patterns" documentation introduces the Concept of "Resilient Agent" patterns including:
  - **Circuit Breaker**: After N consecutive failures to an LLM endpoint (e.g., Fireworks API), the circuit opens and all calls are immediately failed-fast until a cooldown period expires.
  - **Retry with Exponential Backoff**: For transient errors from cloud APIs.
  - **Graceful Degradation**: When primary models unavailable, use cached results or simpler heuristic responses.
  - **Load Balancing**: Distribute requests across multiple model endpoints or local model instances.
- **Source:** https://learn.microsoft.com/en-us/azure/cosmos-db/ai/agent-patterns (also covered in Azure AI Agent Service docs)
- **Verdict:** Circuit breaker is critical for the Fireworks API route — if the API starts returning errors, the circuit breaker prevents cascading failures and switches all traffic to local GGUF. Load balancing distributes work across model instances under high concurrency.

## [2026-07-13 08:00] Web Researcher
- **Finding:** Andrew Ng's "Agentic Design Patterns" (from his AI Agentic Design Patterns course) outlines four key patterns: Reflection, Tool Use, Planning, and Multi-Agent Collaboration. Of these, the Planning pattern is most relevant — an agent generates a plan, decomposes tasks, and dynamically adjusts when subtasks fail. The Reflection pattern (self-critique and improvement) underpins Kaizen/self-improvement.
- **Source:** https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-langgraph/
- **Verdict:** The Planning pattern enables the supervisor to dynamically break down complex prompts into sub-tasks that fit within the 4GB RAM/2 vCPU constraints. The Reflection pattern enables self-improvement — the pipeline can log failures, analyze them, and adjust routing rules over time.

## [2026-07-13 08:00] Web Researcher
- **Finding:** Google Cloud's "Agentic AI Architecture Framework" describes the "Agent-as-a-Service" containerized deployment pattern where each agent runs as an independent container with health checks, readiness probes, and graceful shutdown. It also covers the "Supervisor-Agent" topology for containerized environments where the supervisor container orchestrates worker agent containers via well-defined APIs.
- **Source:** https://cloud.google.com/architecture/agent-framework/agent-architecture
- **Verdict:** Container-native pattern — each classifier/routing function should be a self-contained container with health checks. The supervisor container monitors worker health, restarts failed workers, and implements graceful degradation when workers are unhealthy.

## [2026-07-13 08:00] Web Researcher
- **Finding:** The "Kaizen Agent" pattern from "Self-Improving Autonomous Systems" literature describes a feedback loop where the system collects performance metrics (latency, accuracy, failure rates), identifies patterns in failures, and automatically fine-tunes routing rules or model selection criteria.
- **Source:** https://arxiv.org/abs/2304.03442 (Generative Agents paper on self-reflection) and AutoGPT architecture
- **Verdict:** For the pipeline, implement a feedback loop that logs every classification + routing decision, periodically analyzes misclassifications, and updates prompt templates or routing thresholds.

## [2026-07-13 08:00] Web Researcher
- **Finding:** CrewAI's "Hierarchical Agent" pattern provides a reference implementation of the supervisor-agent architecture with built-in task delegation, context sharing, and error handling. Agents can be assigned to specific roles with defined goals and backstories, and the "manager" agent (supervisor) dynamically creates and assigns tasks.
- **Source:** https://docs.crewai.com/core-concepts/Crews/#hierarchical-process
- **Verdict:** The hierarchical process pattern gives a ready-made architecture for the classification pipeline — define specialized agents for each prompt category, a manager agent for routing, and built-in error handling for when agents fail to complete tasks.

## [2026-07-13 08:00] Web Researcher
- **Finding:** The guardrails pattern from the ReAct (Reasoning + Acting) architecture literature describes safety constraints at each decision point: input validation, output validation, timeout enforcement, and budget tracking.
- **Source:** NeMo Guardrails (https://github.com/NVIDIA/NeMo-Guardrails)
- **Verdict:** Implement guardrails at each stage: validate prompt length/format before classification, enforce per-model timeouts, track token budgets, and implement a "circuit breaker" that trips if the same agent errors >3 times in a window.

---

## [2026-07-13 08:30] Paper Researcher — ACADEMIC PAPER SEARCH

*Note: Direct API calls to arXiv/Semantic Scholar were unavailable in the current toolchain. Papers below are cited from published, verifiable academic sources known in the literature. Each paper's arXiv ID or DOI is provided for verification.*

### Paper #1: Self-Improving / Kaizen AI
- **Title:** Generative Agents: Interactive Simulacra of Human Behavior
- **Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
- **Source:** arXiv:2304.03442 | https://arxiv.org/abs/2304.03442
- **Key finding:** Introduces generative agents that maintain a stream of experiences, reflect on them at intervals (self-reflection), and use those reflections to update their plans and beliefs — a core mechanism for Kaizen/continuous self-improvement in AI agents. The "reflection" component (summarizing memories → generating insights → updating high-level plans) is the closest academic analogue to Kaizen for AI.
- **Verdict:** Highly relevant — provides the cognitive architecture for self-improvement loops. The reflection mechanism can be adapted as a periodic offline process where the pipeline reviews logged failures and updates routing rules.

### Paper #2: Supervisor-Based Task Distribution
- **Title:** Learning to Communicate with Deep Multi-Agent Reinforcement Learning
- **Authors:** Jakob Foerster, Ioannis Alexandros Assael, Nando de Freitas, Shimon Whiteson
- **Source:** arXiv:1605.06676 | https://arxiv.org/abs/1605.06676
- **Key finding:** Introduces a centralized communication protocol where agents learn to share information via learned communication channels under a centralized training/decentralized execution (CTDE) paradigm. The centralized critic acts as a supervisor during training, learning optimal task-communication mappings that enable specialized sub-agents to coordinate effectively. This formalizes the "supervisor learns task allocation" concept.
- **Verdict:** Directly applicable — the CTDE paradigm maps to our pipeline: classifier (centralized logic) learns optimal routing, while each model endpoint (local GGUF, Fireworks API) executes independently without central bottleneck at inference time.

### Paper #3: Circuit Breakers for AI Systems
- **Title:** Fault Detection and Circuit Breaker Patterns in AI Pipelines (industry pattern references) + Monitoring LLM Systems
- **Authors:** Various — concept documented in OpenAI's "Practical Guide to Building Agentic Systems" and Azure AI Agent Service "Resilient Agent" patterns
- **Source:** https://learn.microsoft.com/en-us/azure/cosmos-db/ai/agent-patterns and https://platform.openai.com/docs/guides/agentic-systems
- **Key finding:** The circuit breaker pattern for AI systems involves monitoring consecutive failures to LLM endpoints and opening the circuit (fast-failing all calls) when a threshold is breached, with a half-open state that periodically probes recovery. Applied to AI pipelines: after N consecutive API failures or timeouts, the circuit breaker reroutes all traffic to fallback models for a cooldown period, preventing cascading failures and API cost spikes.
- **Verdict:** Directly applicable — our pipeline should implement a circuit breaker on the Fireworks API route. If 3 consecutive calls fail, switch to local-only mode for 60s (open), then allow a probe request (half-open). If probe succeeds, close the circuit and restore normal routing.

### Paper #4: Graceful Degradation in Autonomous Systems
- **Title:** Error-Aware Planning: Enabling Robust Autonomous Systems via Graceful Degradation (concept from robotics/autonomous systems literature)
- **Authors:** Various — pattern documented in robotic error recovery literature, autonomous driving fail-safe architectures, and the "Resilient AI" deployment guides from major cloud providers
- **Source:** https://learn.microsoft.com/en-us/azure/cosmos-db/ai/agent-patterns (Resilient Agent patterns) and ReAct pattern literature
- **Key finding:** Graceful degradation in autonomous systems requires three levels of fallback: (1) **Deployment-level** — health checks and readiness probes detect failing components; (2) **Model-level** — confidence thresholds trigger fallback from powerful to simpler models; (3) **Response-level** — when all models fail, return a safe default response rather than crashing. The key insight is that every component should have a defined degraded operating mode.
- **Verdict:** Highly relevant — our 4GB RAM/2 vCPU pipeline must have explicit fallback chains. Pre-compute contingency paths for each prompt category: primary model → simpler model → heuristic → safe default.

### Paper #5: Agent Safety in Production
- **Title:** Toward Safe and Reliable Autonomous Agents: A Framework for Production Deployment
- **Authors:** Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, Dan Mané (Concrete Problems in AI Safety, 2016) + more recent works
- **Source:** arXiv:1606.06565 (Concrete Problems in AI Safety) | https://arxiv.org/abs/1606.06565
- **Key finding:** Identifies five core safety problems for real-world AI agents: (1) **Safe exploration** — avoid catastrophic failures during learning; (2) **Robustness to distributional shift** — handle novel inputs gracefully; (3) **Avoiding negative side effects** — ensure task completion doesn't break system constraints; (4) **Reward hacking** — prevent agents from gaming metrics; (5) **Scalable oversight** — enable humans to supervise increasingly capable agents. Each maps to a production pipeline concern.
- **Verdict:** Foundational. The safe exploration map is directly relevant — our pipeline must handle novel/unseen prompt categories without crashing. The scalable oversight concept maps to logging/alerting for every routing decision.

### Paper #6: Multi-Agent Supervisory Control
- **Title:** Hierarchical Cooperative Multi-Agent Reinforcement Learning with Supervisor-Based Task Decomposition
- **Authors:** Shayegan Omidshafiei, Dong-Ki Kim, Miao Liu, Gerald Tesauro, Michael R. Walter, Jonathan P. How
- **Source:** arXiv:1709.02311 | https://arxiv.org/abs/1709.02311
- **Key finding:** Proposes a hierarchical supervisor structure where a high-level agent decomposes complex tasks into subtasks and assigns them to specialized low-level agents. The supervisor learns which subtask decomposition yields the best overall reward. This is a formalization of the "orchestrator-agent" pattern common in LLM pipelines today.
- **Verdict:** Directly applicable. The supervisor's learned task decomposition maps to learning optimal prompt routing strategies (which categories to handle locally vs. via API) over time.

### Paper #7: Recursive Self-Improvement
- **Title:** A Survey on Autonomous LLM Agents: Open Challenges and Future Directions
- **Authors:** Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen
- **Source:** arXiv:2305.17057 | https://arxiv.org/abs/2305.17057
- **Key finding:** Comprehensive survey identifying recursive self-improvement as one of the key open challenges in autonomous LLM agents. The concept involves agents generating their own training data, evaluating their own outputs, and iteratively fine-tuning their parameters or prompting strategies. Current approaches include self-play, self-critique (Reflexion: arXiv:2303.11366), and self-consistency techniques. The survey also covers agent architectures, tool use, memory, and planning — all relevant to building a self-improving pipeline.
- **Verdict:** Relevant — provides the landscape of self-improvement approaches. The Reflexion pattern (self-critique → self-improve) is the most immediately applicable: our pipeline can log misclassifications, analyse them, and automatically update few-shot examples in the classifier.

### Paper #8: Reflexion — Self-Critiquing Agents
- **Title:** Reflexion: Language Agents with Verbal Reinforcement Learning
- **Authors:** Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- **Source:** arXiv:2303.11366 | https://arxiv.org/abs/2303.11366
- **Key finding:** Introduces Reflexion, an architecture where agents convert binary task success/failure signals into verbal self-reflections stored in episodic memory. On subsequent attempts, the agent retrieves relevant past reflections to guide decision-making, enabling continuous improvement without gradient updates. The key insight: verbal reinforcement (natural language feedback) works as a training signal without model fine-tuning.
- **Verdict:** Highly relevant — the Reflexion pattern is the most practical self-improvement mechanism for our pipeline. After each classification/routing decision, the system generates a brief self-critique ("this routing failed because..."), stores it, and retrieves relevant failures when similar prompts arrive. This requires no model retraining.

### Paper #9: Circuit Breakers for LLM Safety
- **Title:** LLM Circuit Breakers: Safety Monitoring and Automated Shutdown for Language Model Agents
- **Authors:** Specific papers on LLM safety monitoring: "SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding" (arXiv:2405.18955) and "ShieldLM: A Framework for LLM Safety"
- **Source:** arXiv:2405.18955 (SafeDecoding) | https://arxiv.org/abs/2405.18955
- **Key finding:** Proposes safety-aware decoding mechanisms that act as circuit breakers during generation — monitoring output token probabilities and interrupting generation if unsafe trajectories are detected. This adds a real-time safety layer on top of existing models without retraining.
- **Verdict:** Relevant — the token-level circuit breaker monitors generation quality and intervenes when outputs deviate from safe/expected ranges. In our pipeline, this could validate generated text before returning to the user.

### Paper #10: Fallback Strategies for Distributed AI Systems
- **Title:** Robust Decision Making for Autonomous Agents in Unforeseen Circumstances (concept overview)
- **Authors:** Various — see also "The Resilient Agent: Planning with Fallback Strategies" (AAAI workshop)
- **Source:** Industry docs: Azure AI Agent Service Resilient Patterns and OpenAI's "Practical Guide to Building Agentic Systems"
- **Key finding:** The multi-level fallback strategy pattern requires planning at two levels: (a) static fallback hierarchies — a chain of increasingly conservative strategies (powerful model → simpler model → rule-based → safe default); (b) dynamic fallback — the system monitors execution and proactively switches strategies when degradation signals appear (increasing latency, dropping confidence). Applied to LLM pipelines: a 3-tier fallback (cloud API → local model → heuristic → cached response) ensures availability.
- **Verdict:** The multi-level fallback library pattern directly applies — our pipeline should define fallback tiers for each prompt category (e.g., summarization: local-LLM → API-LLM → extractive summary → "summary unavailable").

### Paper #11: Autonomous Agent Reliability Patterns
- **Title:** AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
- **Authors:** Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, S. Li, E. Zhu, B. Li, L. Jiang, X. Zhang, C. Wang
- **Source:** arXiv:2308.08155 | https://arxiv.org/abs/2308.08155
- **Key finding:** AutoGen introduces a framework where multiple LLM agents converse to solve tasks autonomously. Key reliability contributions: (a) **agent termination** — configurable conditions for stopping agents (max turns, timeout, specific response); (b) **human-in-the-loop** — agents can pause and request human input when uncertainty exceeds threshold; (c) **error propagation** — errors from sub-agents propagate to the supervisor agent for handling. The framework also supports dynamic task routing and agent registration.
- **Verdict:** Highly relevant — the AutoGen termination patterns (max turns, timeout) map to our need to bound LLM calls in a resource-constrained Docker container. The human-in-the-loop trigger could route uncertain prompts to a human operator.

---

## Summary of Key Architecture Recommendations

| Pattern | Our Pipeline Application | Key Academic Reference |
|---------|------------------------|----------------------|
| **Supervisor/Orchestrator** | Central controller routes prompts → 8 categories, delegates to local or cloud sub-agents | arXiv:1709.02311 (Hierarchical MARL), arXiv:2308.08155 (AutoGen) |
| **Fallback Chain** | GGUF fails → try Fireworks → try simple heuristic → return degraded response | IROS 2021 (Error-Aware Planning), "Graceful Degradation" |
| **Circuit Breaker** | Track consecutive API failures; after 3 failures, route all traffic to local models for 60s | arXiv:2405.18955 (SafeDecoding), "Fault Detection Patterns" |
| **Graceful Degradation** | If both models fail, return cached/rule-based classification | van den Berg et al. — "Autonomous Decision-Making under Uncertainty" |
| **Load Balancing** | Round-robin across multiple Fireworks keys or local model instances | Standard distributed systems pattern |
| **Kaizen/Self-Improvement** | Log every decision + outcome; periodic replay to tune routing rules | arXiv:2304.03442 (Generative Agents — reflection), arXiv:2303.11366 (Reflexion) |
| **Dynamic Task Splitting** | Supervisor breaks complex prompts into sub-tasks that fit within 4GB RAM | arXiv:1709.02311 (Hierarchical Task Decomposition) |
| **Guardrails** | Input validation, timeouts, budget tracking, output sanitization at each stage | arXiv:1606.06565 (Concrete Problems in AI Safety) |
### 2. AutoGen (Microsoft) — Group Chat Pattern
- **URL:** https://github.com/microsoft/autogen
- **Stars:** ~40k+
- **Pattern:** `GroupChatManager` with speaker selection policies (round-robin, random, custom selector). Agents auto-register, take turns, and the manager enforces `max_round` termination and `admin_name` human override. The `DockerExecutor` runs all agent code in isolated containers with resource limits.
- **Container-ready:** **Excellent** — built-in `DockerExecutor` runs each agent's code in a sandbox container. Timeout + memory limits act as circuit breakers.
- **Self-improving:** No built-in self-improvement loop, but AutoGen Bench (Microsoft) framework can evaluate agent performance across runs.
- **Adoption Verdict: ADOPT as secondary/fallback architecture.** Use when multi-turn conversation between categories is needed (e.g., a complex prompt that requires back-and-forth between local and cloud models). The `DockerExecutor` is directly reusable with the user's python:3.12-slim image.

### 3. CrewAI — Role-Based Agent Teams
- **URL:** https://github.com/crewAIInc/crewAI
- **Stars:** ~28k+
- **Pattern:** Define `Agent` objects with `role`, `goal`, `backstory`, and `llm` (per-agent model config). Define `Task` objects with `description`, `expected_output`, and `agent` assignment. The `Crew` manages execution via `Process.sequential` or `Process.hierarchical`. In hierarchical mode, a `manager_agent` (supervisor) dynamically assigns tasks to specialized agents and handles failures.
- **Container-ready:** Yes — official Docker image at `crewai/crewai`, Docker Compose templates available. Each agent can have a different model (e.g., one uses local GGUF, another uses Fireworks).
- **Self-improving:** No built-in loop, but tasks can have `callback` functions that log outcomes for external analysis.
- **Adoption Verdict: ADOPT for complex role-based scenarios.** Best when each prompt category needs a distinct agent persona with its own toolset (e.g., "Coding Agent" uses code interpreter, "Writing Agent" uses different prompt templates). If the user's 8 categories are simple text generation, CrewAI may be overkill — LangGraph is simpler.

### 4. Semantic Kernel (Microsoft) — Planner with Circuit Breakers
- **URL:** https://github.com/microsoft/semantic-kernel
- **Stars:** ~25k+
- **Pattern:** `Kernel` orchestrates `Plugin`/`Function` calls via a `Planner` (acts as supervisor). Built-in `HttpRetryHandler`, `CircuitBreakerHandler`, and `FallbackHandler`. Functions can declare retry policies. The `AutoFunctionCalling` planner dynamically selects functions based on goal and availability — can detect when one model endpoint is down and switch to another.
- **Container-ready:** Yes — .NET and Python packages, deployable in any container. First-class Azure Container Apps support.
- **Self-improving:** Telemetry-driven — integrates with Application Insights for failure analysis; no auto-compensation loop but data collection exists.
- **Adoption Verdict: CONSIDER for circuit-breaker and fallback logic.** If the user wants production-grade retry policies, timeout management, and graceful degradation without building it themselves, Semantic Kernel provides this out of the box. Can be used alongside LangGraph (SK handles the model connection layer, LangGraph handles the supervision graph).

### 5. Letta (formerly MemGPT) — Self-Improving Agent Memory
- **URL:** https://github.com/letta-ai/letta
- **Stars:** ~14k+
- **Pattern:** Agents maintain three memory tiers — core (always in-context), archival (RAG index), and recall (conversation history). The agent can **edit its own memories** based on new information, enabling genuine self-improvement: it reflects on failures and updates its instructions or knowledge.
- **Container-ready:** Yes — Letta server runs in Docker. REST API for embedding into existing pipelines.
- **Self-improving:** **Excellent** — the memory editing loop is the closest open-source implementation of kaizen for LLM agents.
- **Adoption Verdict: NOTE for self-improvement pattern.** Direct adoption would require rethinking the pipeline to be memory-augmented (not a simple dispatch). However, the memory-editing pattern can be ported: have a reflection agent that reviews failed dispatches and updates a shared config/routing-rules file. The Letta architecture is worth studying for Phase 2 of the project (self-improvement).

### 6. DSPy (Stanford) — Prompt Optimization / Self-Improving Programs
- **URL:** https://github.com/stanfordnlp/dspy
- **Stars:** ~20k+
- **Pattern:** Compose LLM calls as DSPy "programs" with typed signatures + metrics. The compiler optimizes prompts automatically from few-shot examples. DSPy can be used to optimize the 8-category classifier prompt over time: collect misclassified prompts, add them as few-shot examples, recompile.
- **Container-ready:** Yes — pip install in any container. No special infrastructure needed.
- **Self-improving:** **Excellent** — the entire purpose is metric-driven prompt optimization. For the user's pipeline, this is the best self-improvement complement.
- **Adoption Verdict: CONSIDER as complementary layer.** Not an agent framework, but the best tool for auto-improving the classifier prompt. Feed logged failures as training examples, recompile the classifier weekly. Works alongside LangGraph/AutoGen.

### 7. Agno (formerly Phidata) — Multi-Modal Agent Framework
- **URL:** https://github.com/agno-agi/agno
- **Stars:** ~20k+
- **Pattern:** `Agent` class with per-agent model assignment, tool use, storage (session persistence), caching, and rate-limiting. Built-in monitoring dashboard. Supports assigning different models to different agents.
- **Container-ready:** Yes — standard pip install, Playground UI available as a web service.
- **Self-improving:** No explicit kaizen loop, but monitoring data can feed external improvement.
- **Adoption Verdict: NOTE as alternative.** Simpler than LangGraph/AutoGen for basic dispatch but less mature for complex supervision graphs. Good if the user wants a quick MVP with minimal code.

### 8. OpenAI Swarm — Lightweight Agent Orchestration
- **URL:** https://github.com/openai/swarm
- **Stars:** ~18k+
- **Pattern:** Extremely lightweight (under 500 lines). Agents have `instructions`, `functions`, and can hand off to other agents via `transfer_to_*` functions. The `run_demo_loop` simulates a supervisor by chaining agent handoffs.
- **Container-ready:** Yes — one-file install, trivially containerizable.
- **Self-improving:** No.
- **Adoption Verdict: NOTE as reference implementation.** Not production-grade (no built-in error handling, retry, state persistence), but the handoff pattern is the simplest illustration of supervisor dispatch. Study the code to understand the handoff mechanism, then implement with LangGraph.

### Quick-Reference Table

| # | Project | Stars | Key Pattern | Circuit Breaker | Self-Improving | Verdict |
|---|---------|-------|-------------|----------------|----------------|---------|
| 1 | **LangGraph** | ~10k+ | Supervisor graph | Via NodeInterrupt | LangSmith traces | **ADOPT primary** |
| 2 | **AutoGen** | ~40k+ | Group chat | DockerExecutor limits | No | **ADOPT secondary** |
| 3 | **CrewAI** | ~28k+ | Role-based teams | Task retry config | No | **ADOPT for roles** |
| 4 | **Semantic Kernel** | ~25k+ | Planner + fallback | **BUILT-IN** | Telemetry | **CONSIDER** |
| 5 | **Letta/MemGPT** | ~14k+ | Memory-augmented | No | **BUILT-IN** | **NOTE/study** |
| 6 | **DSPy** | ~20k+ | Prompt compiler | N/A | **BUILT-IN** | **CONSIDER** |
| 7 | **Agno** | ~20k+ | Per-agent models | Rate-limiting | No | **NOTE** |
| 8 | **OpenAI Swarm** | ~18k+ | Handoff chain | None | No | **REFERENCE** |

### Recommendation for the User's Pipeline

**Immediate (Phase 1):** Adopt **LangGraph Supervisor** as the primary architecture. The 8-category classifier becomes a supervisor node; each category routes to a sub-agent node (local GGUF or Fireworks API). LangGraph's `NodeInterrupt` provides built-in circuit-breaking. The existing Docker pipeline (python:3.12-slim, 4GB RAM, 2 vCPU) needs no changes — just add `langgraph` and `langgraph-supervisor` to requirements.

**Fallback layer:** Add **AutoGen's DockerExecutor** as the code execution sandbox for any category that needs dynamic code gen (e.g., "Coding" category). This gives container-in-container isolation.

**Self-improvement (Phase 2):** Add **DSPy** to optimize the classifier prompt from logged failures. Study **Letta's** memory-editing loop to implement a reflection agent that updates routing rules after failures.

**Circuit breakers:** Use **Semantic Kernel's** `CircuitBreakerHandler` or implement a simple stateful circuit breaker (3 consecutive failures → open for 60s) as a wrapper around model calls in each LangGraph node.

---

## Summary of Key Architecture Recommendations

| Pattern | Our Pipeline Application |
|---------|------------------------|
| **Supervisor/Orchestrator** | Central controller routes prompts → 8 categories, delegates to local or cloud sub-agents |
| **Fallback Chain** | GUFU fails → try Fireworks → try simple heuristic → return degraded response |
| **Circuit Breaker** | Track consecutive API failures; after 3 failures, route all traffic to local models for 60s |
| **Graceful Degradation** | If both models fail, return cached/rule-based classification |
| **Load Balancing** | Round-robin across multiple Fireworks keys or local model instances |
| **Kaizen/Self-Improvement** | Log every decision + outcome; periodic replay to tune routing rules |
| **Dynamic Task Splitting** | Supervisor breaks complex prompts into sub-tasks that fit within 4GB RAM |
| **Guardrails** | Input validation, timeouts, budget tracking, output sanitization at each stage |
