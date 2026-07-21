# Architecture Plans B & C — Saved for Future Reference
## Plan-and-Solve Multi-Run Workflow Integration

Saved: 2026-07-13
Chosen plan: A (Extended Cell with Inline Steps)

### Plan B — WorkflowCell Subclass (Clean OO Separation)
Create a `WorkflowCell` subclass of `Cell` with its own evaluation path, pipeline method, and mutation operators.

**Key design**:
- `WorkflowCell(Cell)` adds `steps: list[StepConfig]` — base `Cell` stays pristine
- `WorkflowEngine` class executes step sequences with artifact interpolation
- Tool registry via in-process MCP: `ToolRegistry.register("sympy", fn)` — dispatches tool steps
- `WorkflowMutationAgent` with 7 operators (mutate_step_prompt, insert_step, remove_step, etc.)
- `EvaluationAgent` gets `evaluate_workflow()` that records per-step accuracy/latency/tokens

**Files**: `agent/workflow_cell.py`, `agent/workflow_engine.py`, `agent/tools/sympy_tool.py`, `agent/tools/python_tool.py`, `agent/tools/spacy_tool.py`
**Effort**: Large (4-6 days)
**Trade-off**: Cleanest separation but most upfront code

### Plan C — Separate Workflow Registry (Decoupled)
Leave `Cell` completely unchanged. Independent `Workflow` dataclass + `WorkflowRegistry`. Steps reference Cells by name. Meta-agent loops over steps.

**Key design**:
- `Workflow` defines a sequence of `WorkflowStep` objects, each referencing a `Cell` by name
- Cells are shared across workflows — a single Cell can be reused in multiple workflows
- `WorkflowOrchestrator` is a meta-agent that calls `Pipeline.infer_with_cell()` for each step
- Workflow structure is **fixed** (template per task type) — only prompts inside evolve
- Tools live in the orchestration layer (post-processing on artifacts between steps)
- One new mutation operator: `swap_workflow_step_cell`

**Files**: `agent/workflow.py`, `agent/workflow_orchestrator.py`, `agent/templates/` (YAML templates per task)
**Effort**: Medium (3-4 days)
**Trade-off**: Maximum flexibility/decoupling, but workflow structure doesn't evolve — only prompts inside
