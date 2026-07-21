# Research: Best Agentic Architecture for Our Container Pipeline

> Date: 2026-07-13 12:00 CDT
> Run: research-001 (3 specialized researchers + 1 reviewer)
> Question: What is the best agentic architecture for a task-centered Docker pipeline that handles abnormal conditions, splits tasks dynamically, self-improves (Kaizen), and includes safety mechanisms?

---

## Summary

**Adopt LangGraph Supervisor as the primary architecture.** The 8-category classifier becomes a supervisor node routing to specialized sub-agents. Supplement with AutoGen's DockerExecutor for code-gen sandboxing and DSPy for self-improving prompt optimization. Borrow Semantic Kernel's circuit breaker pattern without adding the framework.

## Recommended Architecture

```
LangGraph Supervisor (classifier node)
  │
  ├── Category Node: code_gen → AutoGen DockerExecutor (sandboxed)
  ├── Category Node: code_debug → AutoGen DockerExecutor
  ├── Category Node: math → Local GGUF (Qwen2.5-1.5B)
  ├── Category Node: logic → Fireworks API
  ├── Category Node: factual → Local GGUF + FactDB
  ├── Category Node: sentiment → Local GGUF (VADER hybrid)
  ├── Category Node: ner → Fireworks API
  └── Category Node: summarization → Fireworks API
  │
  └── Circuit Breaker (per-node): 3 failures → open 60s → fallback
  └── Logging → DSPy compiler (weekly prompt optimization)
```

## Key Findings by Source

### Web Research (9 patterns)
| Pattern | Source | Verdict |
|---------|--------|---------|
| Supervisor/Orchestrator | Anthropic's Building Effective Agents | The classifier IS the supervisor |
| Fallback Chain | OpenAI Agentic Systems Guide | Essential for 4GB RAM constraint |
| Supervisor Graph | LangGraph Multi-Agent | Dynamic agent routing + retry on failure |
| Circuit Breaker | Microsoft Azure AI Patterns | Critical for Fireworks API route |
| Planning + Reflection | Andrew Ng's Agentic Design Patterns | Kaizen self-improvement loop |
| Container-Native Agents | Google Cloud Agentic AI | Health checks, graceful shutdown |
| Kaizen Self-Improvement | AutoGPT + Generative Agents | Log → analyze → auto-update routing |
| Hierarchical Agent | CrewAI | Ready-made supervisor-agent pattern |
| Guardrails | NeMo Guardrails | Input validation, timeouts, budget tracking |

### OSS Projects (8 reviewed)
| Project | Stars | Verdict |
|---------|-------|---------|
| **LangGraph** | 10K★ | **ADOPT primary** — supervisor graph maps to 8-category dispatch |
| **AutoGen** | 40K★ | **ADOPT secondary** — DockerExecutor for code generation sandboxing |
| **CrewAI** | 28K★ | **ADOPT for roles** — when per-category agents need distinct personas |
| **Semantic Kernel** | 25K★ | **CONSIDER** — borrow CircuitBreakerHandler pattern, not the framework |
| **DSPy** | 20K★ | **CONSIDER** — metric-driven prompt optimization from logged failures |
| **Letta/MemGPT** | 14K★ | **NOTE/study** — memory-editing loop for Phase 2 self-improvement |
| **Agno** | 20K★ | **NOTE** — simple multi-model dispatch, good for MVP |
| **OpenAI Swarm** | 18K★ | **REFERENCE** — study handoff pattern, don't use in production |

### Academic Papers (11 reviewed)
| Paper | Key Insight | Verdict |
|-------|-------------|---------|
| **Generative Agents** (2304.03442) | Self-reflection mechanism for Kaizen | Core cognitive architecture for self-improvement |
| **Reflexion** (2303.11366) | Verbal reinforcement without model retraining | Most practical self-improvement mechanism |
| **Hierarchical MARL** (1709.02311) | Supervisor learns optimal task decomposition | Formalizes our routing optimization |
| **Concrete AI Safety** (1606.06565) | 5 safety problems for production agents | Safe exploration → handle novel prompts |
| **AutoGen** (2308.08155) | Agent termination, HITL, error propagation | Bounding LLM calls in resource-constrained container |

## Safety & Resilience Layer

Each node in the supervisor graph wraps model calls with:

1. **Circuit Breaker**: 3 consecutive failures → circuit opens (60s) → all traffic to fallback → half-open probe → close on success
2. **Fallback Chain**: Primary model → simpler model → heuristic rule → safe default response
3. **Graceful Degradation**: Each component has a defined degraded mode
4. **Guardrails**: Input validation, timeout enforcement, token budget tracking, output sanitization

## Self-Improvement (Kaizen) Loop

Phase 1 (now): Log every routing decision + outcome to a structured failure log
Phase 2 (next): DSPy re-optimizes classifier prompts from logged failures weekly
Phase 3 (future): Reflection agent (Reflexion pattern) reviews failures and updates routing rules autonomously

## Files
- Full scratchpad: `research/scratch/research-001/findings.md` (243 lines, 30KB)
