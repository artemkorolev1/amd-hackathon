# AI Engineering Organization — Architecture

> Vision: Self-improving AI teams with shared knowledge, distinct skills/memory per team,
> and a Kaizen process that continuously optimizes the system.

## Team Structure

```
You (Director)
  │
  ├─ 🔬 Research Team ─── FIRST. Consulted before any build.
  │   ├─ Manager: research lead (search strategy)
  │   ├─ Members: 2-3 researchers (web, papers, repos, HFace, Kaggle)
  │   ├─ Memory: what's been researched, what was rejected, interesting findings
  │   ├─ Skills: GPT-Researcher pattern, STORM methodology, PaperQA2 for papers
  │   └─ Output: structured reports with repo links, trade-offs, sources
  │
  ├─ 🏗️ Architecture/Developer Team (same team, different members)
  │   ├─ Manager: software architect (high-quality context, knowledge base)
  │   ├─ Members: code reviewers, back-end engineers, front-end engineers
  │   ├─ Memory: design decisions, trade-offs, references, patterns
  │   └─ Input from: Research Team (best practices, library recommendations)
  │
  ├─ 🐛 Bug Fixing Team (separate — different skillset, different mentality)
  │   ├─ Manager: debugging lead
  │   ├─ Members: root cause analysts, regression hunters
  │   └─ Memory: common bug patterns, past fixes, root cause catalog
  │
  ├─ ⚙️ Optimization Team (ML/Training + Optimization — merged)
  │   ├─ Manager: optimization lead
  │   ├─ Members: prompt engineers, classifier tuners, GEPA runners
  │   ├─ Memory: config trials, what worked, Pareto fronts, model benchmarks
  │   └─ Techniques: GEPA, Bayesian optimization, A/B testing, ensemble tuning
  │
  ├─ 📊 Evaluation Team (separate, critical — owns the truth)
  │   ├─ Manager: eval lead
  │   ├─ Members: benchmark runners, result analysts, metric designers
  │   ├─ Memory: baselines, regressions, per-category trends, parameter sweep logs
  │   └─ Standard: Extensive excel report with all parameters + per-item detail + summary
  │
  ├─ 🗄️ Data Quality Team (heaviest in scripts)
  │   ├─ Manager: data lead
  │   ├─ Members: data preparers, synthetic data generators, dataset curators
  │   ├─ Memory: dataset quality metrics, provenance, preprocessing pipelines
  │   └─ Tools: synthetic data generators, cleaning scripts, validation pipelines
  │
  ├─ 🧹 Maintenance Team
  │   ├─ Manager: infrastructure administrator
  │   ├─ Members:
  │   │   ├─ Administrator: safety checks, API keys, secrets, scripts, CI/CD
  │   │   └─ Librarian: wiki, logs, project memory, knowledge organization
  │   ├─ Memory: system health, naming conventions, style guides
  │   └─ Automation: style enforcement, code minimization, ruff lint, safety scans
  │
  ├─ 🚀 Deployment Team (future)
  │   └─ (not yet defined)
  │
  └─ 🔄 Kaizen Team (continuous improvement — CRON-based)
      ├─ Trigger: after tasks complete or on idle
      ├─ Input: logs of all calls, results, success/failure, Hermes memory
      ├─ Actions:
      │   ├─ Adjust skills based on actual usage patterns
      │   ├─ Tweak communication templates (caveman prompts)
      │   ├─ Prune ineffective patterns, amplify effective ones
      │   └─ Recommend model downgrades (replace 4B → 1.5B where possible)
      ├─ Long-term goal: understand tasks so well that LLMs can be replaced
      │   by simple models (1.5B) with sophisticated skills
      └─ Output: skill patches, config updates, protocol adjustments

## Cross-Cutting Principles

### Research-First
- Before ANY build: research team checks if open-source solution exists
- If yes: prefer adoption over building
- Research findings stored in team's memory + project library

### Reviewers in Every Team
- Every team has a reviewer member
- Reviewers check: quality, consistency, adherence to standards
- Cross-team reviews for integration points

### Team Memory Isolation
- Research memory doesn't bleed into architecture
- Eval memory doesn't bleed into ML training
- Each team has its own canonical category in Mnemosyne
- Integration only via structured handoffs between managers

### Knowledge Base (per team)
- Books (downloaded, organized by librarian)
- GitHub repos (curated by research team)
- HuggingFace/Kaggle datasets and models
- Architecture patterns with references
- All stored and tagged for cross-team discovery

### Reporting Standard
- Results: top-line summary first
- Detail: full parameter spreadsheet (CSV/Excel)
- Per-item: available on demand
- Educational: "why this approach, what are the trade-offs, references"
- Concise: director-level first, engineer-level on request

### Ultimate Goal
- Understand task nature so deeply that:
  - Expensive LLM calls → 1.5B models with sophisticated skills
  - Token costs approach zero
  - Speed approaches deterministic
  - Quality maintained or improved
