"""
contracts.py — Data contracts for the answer-centered pipeline.

Defines the request/response types used by CellRunner and all execution
backends. Designed for backward compatibility with the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class CellMetrics:
    """Metrics for a single cell execution."""

    elapsed_ms: float = 0.0
    model: Optional[str] = None
    tokens_out: int = 0
    tool: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnswerRequest:
    """Input to a cell execution."""

    prompt: str = ""
    category: str = "general"
    task_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AnswerResponse:
    """Structured output from a cell execution."""

    answer: str = ""
    confidence: float = 0.0
    metrics: CellMetrics = field(default_factory=CellMetrics)
    metadata: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass
class CellConfig:
    """Configuration for executing a cell via CellRunner.

    Lightweight wrapper around a WorkCell's routing properties.
    Created via CellConfig.from_workcell(workcell).
    """

    role: str = "local_llm"
    name: str = ""
    model_key: Optional[str] = None
    system_prompt: str = ""
    steps: Optional[list] = None
    decoding: dict = field(default_factory=dict)

    @classmethod
    def from_workcell(cls, wc) -> CellConfig:
        """Build a CellConfig from a WorkCell or Cell instance."""
        decoding = getattr(wc, "decoding", None)
        if hasattr(decoding, "to_dict"):
            decoding = decoding.to_dict()
        elif not isinstance(decoding, dict):
            decoding = {}
        return cls(
            role=getattr(wc, "role", "local_llm"),
            name=getattr(wc, "name", ""),
            model_key=getattr(wc, "model_key", None),
            system_prompt=getattr(wc, "system_prompt", ""),
            steps=getattr(wc, "steps", None),
            decoding=decoding,
        )


@dataclass
class QualityCheckResult:
    """Output of a code quality check cell."""
    passed: bool = True
    answer: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: str = ""
    elapsed_ms: float = 0.0


@dataclass
class WorkflowRequest:
    """Request for the workflow executor (Plan-and-Solve).

    Carries the cell definition + input + step-level overrides.
    """
    steps: list  # list of StepConfig dicts
    prompt: str = ""
    model_key: Optional[str] = None
    system_prompt: str = ""
    artifacts: dict = field(default_factory=dict)
