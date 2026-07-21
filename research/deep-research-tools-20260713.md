# Deep Research Tools — Open Source Landscape

> Generated: 2026-07-13
> Context: Building a deep researcher agent role for the Hermes dispatcher profile
> Source: Research sub-agent batch (deleg_8074461c)

## Top Candidates

### STORM (Stanford) — 17K+ stars
- **GitHub**: https://github.com/stanford-oval/storm
- Multi-perspective source interviewing, iterative retrieval, Wikipedia-style article generation
- Best for: generating comprehensive survey-style reports on any topic

### GPT-Researcher — 16K+ stars
- **GitHub**: https://github.com/assafelovic/gpt-researcher
- Two-phase: structured research plan → systematic per-topic research → full report with citations
- Best for: web-based deep research with structured output

### PaperQA2 — 8K+ stars
- **GitHub**: https://github.com/whitead/paper-qa
- Agentic RAG for scientific literature, multi-agent (search, read, answer, fact-check)
- Best for: paper-grounded research with citations

### Open Deep Research (dzhng) — 10K+ stars
- **GitHub**: https://github.com/dzhng/deep-research
- Iterative: question → search → extract → update → next query
- Best for: customizable-depth research loops

### CrewAI Research Pattern — 25K+ stars
- **GitHub**: https://github.com/joaomdmoura/crewai
- Multi-agent team: Researcher + Writer + Reviewer + Critic with feedback loops
- Best for: iterative refinement with human feedback

### AutoGen Research Agents (Microsoft) — 37K+ stars
- **GitHub**: https://github.com/microsoft/autogen
- Multi-agent conversations, human-in-the-loop, diverse tool access
- Best for: flexible orchestrations with human oversight

### RAGFlow — 31K+ stars
- **GitHub**: https://github.com/infiniflow/ragflow
- Deep document understanding, multi-step reasoning, structured report generation
- Best for: document-grounded research with RAG

## Recommended Integration Strategy

Build a HERMES RESEARCHER role that combines:
1. **GPT-Researcher approach** — for initial web-based deep dive (plan → execute → report)
2. **PaperQA2 approach** — for paper-grounded follow-up when sources are academic
3. **CrewAI feedback pattern** — Researcher sub-agent → Review sub-agent → iterate
4. Save to `reports/research/` with structured template

## Caveats
- All star counts approximate (from training knowledge)
- Need to verify current activity/release status before committing to integration
- Some tools are standalone (would need adaptation for Hermes sub-agent pattern)
