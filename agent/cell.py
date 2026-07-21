#!/usr/bin/env python3
"""Cell — compositional unit in the GEPA framework.

A Cell bundles a task type, an LLM, a prompt template, decoding parameters,
and an aggregation/judge strategy into a single evolvable unit.

Usage:
    cell = Cell(
        task_id="T01",
        model_key="qwen2.5-1.5b",
        system_prompt="Fact:",
        temperature=0.0,
        max_tokens=64,
    )
    cell.to_dict()              # JSON-serialisable snapshot
    Cell.from_dict(d)           # reconstruct from snapshot
    cell.eq_content(other)      # compare prompts+params (ignoring metadata)
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Supported decoding params ────────────────────────────────────────────────
DECODING_PARAM_KEYS = [
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "seed",
]

# ── Known aggregation strategies ────────────────────────────────────────────
AGGREGATION_STRATEGIES = (
    "single",           # single inference
    "majority_vote",    # k samples, majority answer wins
    "self_consistency", # k samples, weighted by logprob
    "judge_select",     # judge model picks best among k candidates
    "ensemble_vote",    # different models, majority vote
    "workflow",         # multi-step Plan-and-Solve workflow
)

# ── Task IDs for the 8 core tasks ────────────────────────────────────────────
TASK_IDS = ("T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08")
TASK_LABELS = {
    "T01": "factual",
    "T02": "math",
    "T03": "sentiment",
    "T04": "summarization",
    "T05": "ner",
    "T06": "code_debug",
    "T07": "code_gen",
    "T08": "logic",
}
# Internal short names (pipeline uses "code_debug", "code_gen", "logic" too)
TASK_TO_PIPELINE_CAT = {
    "T01": "factual",
    "T02": "math",
    "T03": "sentiment",
    "T04": "summarization",
    "T05": "ner",
    "T06": "code_debug",
    "T07": "code_gen",
    "T08": "logic",
}
PIPELINE_CAT_TO_TASK = {v: k for k, v in TASK_TO_PIPELINE_CAT.items()}
# Allow both T01-T08 and pipeline category strings as task_id
VALID_TASK_IDS = TASK_IDS + tuple(TASK_TO_PIPELINE_CAT.values())
VALID_TASK_IDS = tuple(sorted(set(VALID_TASK_IDS)))


@dataclass
class DecodingConfig:
    """Decoding hyperparameters for a single inference call."""

    temperature: float = 0.0
    max_tokens: int = 64
    top_p: float = 1.0
    top_k: int = 40
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DecodingConfig:
        return cls(**{k: d.get(k, v)
                       for k, v in cls().__dict__.items() if not k.startswith("_")})

    def merged(self, **overrides) -> DecodingConfig:
        """Return a new DecodingConfig with selected fields overridden."""
        d = self.to_dict()
        d.update(overrides)
        return DecodingConfig.from_dict(d)


@dataclass
class StepConfig:
    """Configuration for a single step in a multi-run workflow.

    Each step is either an LLM inference call or a deterministic tool call.
    Steps share context via an artifact dict keyed by step name.
    """

    name: str                               # unique step id: "plan", "solve", "critique", "compose"
    system_prompt: str = ""                 # system prompt for this step
    model_key: Optional[str] = None         # override Cell's model_key per step
    decoding: Optional[DecodingConfig] = None  # override Cell's decoding per step
    input_from: str = "_input"              # artifact key to feed as user message (default: original prompt)
    tool: Optional[str] = None              # "sympy" | "python" | "spacy" | None = LLM call
    max_retries: int = 1

    def get_decoding(self, fallback: DecodingConfig) -> DecodingConfig:
        return self.decoding or fallback

    def to_dict(self) -> dict:
        d = {"name": self.name, "system_prompt": self.system_prompt, "input_from": self.input_from}
        if self.model_key:
            d["model_key"] = self.model_key
        if self.decoding:
            d["decoding"] = self.decoding.to_dict()
        if self.tool:
            d["tool"] = self.tool
        if self.max_retries > 1:
            d["max_retries"] = self.max_retries
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StepConfig:
        dec = d.pop("decoding", None)
        if isinstance(dec, dict):
            d["decoding"] = DecodingConfig.from_dict(dec)
        return cls(**d)


@dataclass
class Cell:
    """A compositional GEPA cell.

    Fields
    ------
    task_id : str        — one of TASK_IDS ("T01"…"T05")
    model_key : str      — key into MODEL_PATHS / ModelCache
    system_prompt : str  — the system-level instruction
    decoding : DecodingConfig
    aggregation : str    — strategy from AGGREGATION_STRATEGIES
    name : str           — human-readable label for the cell
    parent : str         — (optional) parent cell name for provenance
    generation : int     — which generation this cell was created in
    metadata : dict      — free-form (accuracy, latency, Pareto rank, etc.)
    created_at : float   — unix timestamp
    """

    task_id: str = ""
    model_key: str = ""
    system_prompt: str = ""
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    aggregation: str = "single"
    name: str = ""
    parent: str = ""
    generation: int = 0
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    steps: Optional[list[StepConfig]] = None  # multi-step workflow; None = single-shot

    def __post_init__(self):
        if self.task_id and self.task_id not in VALID_TASK_IDS:
            raise ValueError(f"Unknown task_id '{self.task_id}'. Must be one of {VALID_TASK_IDS}")
        if self.steps:
            # Multi-step workflow — auto-set aggregation
            self.aggregation = "workflow"
            if not self.system_prompt:
                self.system_prompt = "Workflow cell — see steps"
        else:
            if self.aggregation not in AGGREGATION_STRATEGIES:
                raise ValueError(f"Unknown aggregation '{self.aggregation}'. "
                                 f"Must be one of {AGGREGATION_STRATEGIES}")
        if not self.name:
            self.name = f"cell_{self.task_id}_{self.model_key}_{int(self.created_at)}"

    # ── Convenience accessors ────────────────────────────────────────────

    @property
    def task_label(self) -> str:
        return TASK_LABELS.get(self.task_id, self.task_id)

    @property
    def pipeline_category(self) -> str:
        return TASK_TO_PIPELINE_CAT.get(self.task_id, self.task_id)

    @property
    def temperature(self) -> float:
        return self.decoding.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.decoding.temperature = value

    @property
    def max_tokens(self) -> int:
        return self.decoding.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self.decoding.max_tokens = value

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self, include_metadata: bool = True) -> dict:
        """Return a JSON-serialisable snapshot."""
        d = {
            "task_id": self.task_id,
            "model_key": self.model_key,
            "system_prompt": self.system_prompt,
            "decoding": self.decoding.to_dict(),
            "aggregation": self.aggregation,
            "name": self.name,
            "parent": self.parent,
            "generation": self.generation,
            "created_at": self.created_at,
        }
        if self.steps:
            d["steps"] = [s.to_dict() for s in self.steps]
        if include_metadata and self.metadata:
            # Strip large detail arrays from metadata for readability
            meta = dict(self.metadata)
            for key in ("details", "raw_scores", "raw_answers"):
                if key in meta and isinstance(meta[key], list) and len(meta[key]) > 5:
                    meta[key] = f"<{len(meta[key])} items, omitted>"
            d["metadata"] = meta
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> Cell:
        d = dict(d)
        dec = d.pop("decoding", None)
        if isinstance(dec, dict):
            d["decoding"] = DecodingConfig.from_dict(dec)
        steps_raw = d.pop("steps", None)
        meta = d.pop("metadata", None)
        cell = cls(**d)
        if meta:
            cell.metadata = meta
        if steps_raw:
            cell.steps = [StepConfig.from_dict(s) for s in steps_raw]
        return cell

    def clone(self, new_name: str = "", **overrides) -> Cell:
        """Create a deep copy with optional field overrides."""
        d = self.to_dict()
        d.update(overrides)
        d["name"] = new_name or f"{self.name}_clone"
        d["created_at"] = time.time()
        d["parent"] = self.name
        d["generation"] = self.generation + 1
        return Cell.from_dict(d)

    def eq_content(self, other: Cell) -> bool:
        """Equality ignoring name, parent, generation, metadata, timing."""
        base = (
            self.task_id == other.task_id
            and self.model_key == other.model_key
            and self.system_prompt == other.system_prompt
            and self.decoding == other.decoding
            and self.aggregation == other.aggregation
        )
        if not base:
            return False
        # Compare workflow steps if both have them
        if self.steps and other.steps:
            if len(self.steps) != len(other.steps):
                return False
            return all(
                s1.name == s2.name and s1.system_prompt == s2.system_prompt
                and s1.model_key == s2.model_key and s1.tool == s2.tool
                for s1, s2 in zip(self.steps, other.steps)
            )
        if self.steps or other.steps:
            return False  # one has steps, the other doesn't
        return True

    def __repr__(self) -> str:
        wf = " [workflow]" if self.steps else ""
        return (f"Cell(name={self.name!r}, task={self.task_id}, "
                f"model={self.model_key}, agg={self.aggregation}, "
                f"gen={self.generation}){wf}")


# ── Population helpers ───────────────────────────────────────────────────────

def population_to_json(population: list[Cell], path: str, **kwargs):
    """Save a population of cells to a JSON file."""
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in population], f, indent=2, **kwargs)


def population_from_json(path: str) -> list[Cell]:
    """Load a population of cells from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [Cell.from_dict(d) for d in data]


def deduplicate_population(population: list[Cell]) -> list[Cell]:
    """Remove cells with identical (task_id, model_key, system_prompt, decoding)."""
    seen: set[tuple] = set()
    result: list[Cell] = []
    for c in population:
        key = (c.task_id, c.model_key, c.system_prompt,
               c.decoding.temperature, c.decoding.max_tokens)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ── Artifact ──────────────────────────────────────────────────────────────────

@dataclass
class Artifact:
    """Lightweight provenance wrapper for inter-cell data passing.

    No typed ports — just source tracking and optional metadata.
    """
    source: str          # work cell name
    content: Any         # the actual payload (almost always a string)
    metadata: dict = field(default_factory=dict)


# ── Execution-backend roles ─────────────────────────────────────────────────
# Maps to how the system actually routes work, not software-engineering roles
WORK_CELL_ROLES = (
    "deterministic",   # regex/heuristic solvers, zero LLM cost
    "local_llm",       # local GGUF inference via llama-cpp-python
    "api_llm",         # external API inference (Fireworks, OpenAI)
    "workflow",        # multi-step Plan-and-Solve with artifact passing
    "aggregator",      # consensus voting, majority, judge-select, ensemble
)


# ── WorkCell — Cell with execution-backend role ──────────────────────────────

@dataclass
class WorkCell(Cell):
    """Extension of Cell with execution-backend role.

    Backward compatible: WorkCell IS-A Cell. Code expecting Cell works unchanged.
    role defaults to "local_llm" — safe default for existing cells.
    """
    role: str = "local_llm"

    def __post_init__(self):
        super().__post_init__()
        if self.role not in WORK_CELL_ROLES:
            raise ValueError(f"Unknown role '{self.role}'. Must be one of {WORK_CELL_ROLES}")

    def to_dict(self, include_metadata=True) -> dict:
        d = super().to_dict(include_metadata=include_metadata)
        d["role"] = self.role
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkCell":
        d = dict(d)  # copy to avoid mutating the original
        role = d.pop("role", "local_llm")
        cell = Cell.from_dict(d)
        wc = cls(
            task_id=cell.task_id,
            model_key=cell.model_key,
            system_prompt=cell.system_prompt,
            decoding=cell.decoding,
            aggregation=cell.aggregation,
            name=cell.name,
            parent=cell.parent,
            generation=cell.generation,
            metadata=cell.metadata,
            created_at=cell.created_at,
            steps=cell.steps,
            role=role,
        )
        return wc


def cell_to_workcell(cell: Cell, role: str = "local_llm") -> WorkCell:
    """Promote a Cell to a WorkCell with the given role."""
    if isinstance(cell, WorkCell):
        cell.role = role
        return cell
    return WorkCell(
        task_id=cell.task_id, model_key=cell.model_key,
        system_prompt=cell.system_prompt, decoding=cell.decoding,
        aggregation=cell.aggregation, name=cell.name,
        parent=cell.parent, generation=cell.generation,
        metadata=cell.metadata, created_at=cell.created_at,
        steps=cell.steps, role=role,
    )
